use super::parameters::{MaxFeatures, RandomForestClassifierParams, RandomForestRegressorParams};
use super::tree::{BuildNode, NO_CHILD, PackedTree};
use crate::api::ModelError;
use crate::data::MatrixView;
use crate::numeric::{OwnedRng, derive_tree_seed};
use std::thread;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ForestConfig {
    pub(super) n_estimators: usize,
    pub(super) max_depth: Option<usize>,
    pub(super) min_samples_split: usize,
    pub(super) min_samples_leaf: usize,
    pub(super) max_features: MaxFeatures,
    pub(super) bootstrap: bool,
    pub(super) random_state: u64,
    pub(super) n_jobs: usize,
}

impl From<&RandomForestClassifierParams> for ForestConfig {
    fn from(params: &RandomForestClassifierParams) -> Self {
        Self {
            n_estimators: params.n_estimators(),
            max_depth: params.max_depth(),
            min_samples_split: params.min_samples_split(),
            min_samples_leaf: params.min_samples_leaf(),
            max_features: params.max_features(),
            bootstrap: params.bootstrap(),
            random_state: params.random_state(),
            n_jobs: params.n_jobs().resolved(),
        }
    }
}

impl From<&RandomForestRegressorParams> for ForestConfig {
    fn from(params: &RandomForestRegressorParams) -> Self {
        Self {
            n_estimators: params.n_estimators(),
            max_depth: params.max_depth(),
            min_samples_split: params.min_samples_split(),
            min_samples_leaf: params.min_samples_leaf(),
            max_features: params.max_features(),
            bootstrap: params.bootstrap(),
            random_state: params.random_state(),
            n_jobs: params.n_jobs().resolved(),
        }
    }
}

pub(super) trait Objective<Y>: Copy + Send + Sync {
    fn value(self, y: &Y) -> f64;
    fn impurity(self, sum: f64, sum_sq: f64, weight: u64) -> f64;
    fn leaf_value(self, sum: f64, weight: u64) -> f32 {
        (sum / weight as f64) as f32
    }
    fn pure(self, sum: f64, sum_sq: f64, weight: u64) -> bool;
}

#[derive(Clone, Copy)]
pub(super) struct Classification;

impl Objective<u8> for Classification {
    fn value(self, y: &u8) -> f64 {
        f64::from(*y)
    }

    fn impurity(self, positives: f64, _sum_sq: f64, weight: u64) -> f64 {
        let p = positives / weight as f64;
        2.0 * p * (1.0 - p)
    }

    fn pure(self, positives: f64, _sum_sq: f64, weight: u64) -> bool {
        positives == 0.0 || positives == weight as f64
    }
}

#[derive(Clone, Copy)]
pub(super) struct Regression;

impl Objective<f32> for Regression {
    fn value(self, y: &f32) -> f64 {
        f64::from(*y)
    }

    fn impurity(self, sum: f64, sum_sq: f64, weight: u64) -> f64 {
        // Population variance.  Clamp cancellation noise at zero.
        let mean = sum / weight as f64;
        (sum_sq / weight as f64 - mean * mean).max(0.0)
    }

    fn pure(self, sum: f64, sum_sq: f64, weight: u64) -> bool {
        self.impurity(sum, sum_sq, weight) == 0.0
    }
}

pub(super) fn train_forest<Y, O>(
    data: &MatrixView<'_>,
    targets: &[Y],
    config: &ForestConfig,
    objective: O,
) -> Result<Vec<PackedTree>, ModelError>
where
    Y: Sync,
    O: Objective<Y>,
{
    let worker_count = config.n_jobs.min(config.n_estimators);
    if worker_count == 1 {
        return (0..config.n_estimators)
            .map(|index| train_tree(data, targets, config, objective, index))
            .collect();
    }

    let mut indexed = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            handles.push(scope.spawn(move || {
                let mut trees = Vec::new();
                for index in (worker..config.n_estimators).step_by(worker_count) {
                    trees.push((index, train_tree(data, targets, config, objective, index)));
                }
                trees
            }));
        }
        let mut results = Vec::with_capacity(config.n_estimators);
        for handle in handles {
            results.extend(handle.join().map_err(|_| ModelError::WorkerPanicked)?);
        }
        Ok::<_, ModelError>(results)
    })?;

    indexed.sort_unstable_by_key(|(index, _)| *index);
    indexed.into_iter().map(|(_, tree)| tree).collect()
}

fn train_tree<Y, O>(
    data: &MatrixView<'_>,
    targets: &[Y],
    config: &ForestConfig,
    objective: O,
    tree_index: usize,
) -> Result<PackedTree, ModelError>
where
    O: Objective<Y>,
{
    let mut rng = OwnedRng::new(derive_tree_seed(config.random_state, tree_index as u64));
    let mut counts = vec![0u32; data.rows()];
    if config.bootstrap {
        for _ in 0..data.rows() {
            let row = rng.index(data.rows());
            counts[row] += 1;
        }
    } else {
        counts.fill(1);
    }
    let rows: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(row, &count)| (count != 0).then_some(row))
        .collect();
    let builder = TreeBuilder {
        data,
        targets,
        counts: &counts,
        config,
        objective,
        rng: &mut rng,
        nodes: Vec::new(),
    };
    builder.build_tree(rows)
}

struct TreeBuilder<'a, 'm, Y, O> {
    data: &'a MatrixView<'m>,
    targets: &'a [Y],
    counts: &'a [u32],
    config: &'a ForestConfig,
    objective: O,
    rng: &'a mut OwnedRng,
    nodes: Vec<BuildNode>,
}

#[derive(Clone, Copy)]
struct Split {
    feature: usize,
    threshold: f32,
    score: f64,
}

#[derive(Clone, Copy)]
enum Attachment {
    Root,
    Left(u32),
    Right(u32),
}

struct PendingNode {
    rows: Vec<usize>,
    depth: usize,
    attachment: Attachment,
}

impl<Y, O: Objective<Y>> TreeBuilder<'_, '_, Y, O> {
    fn build_tree(mut self, rows: Vec<usize>) -> Result<PackedTree, ModelError> {
        let mut pending = vec![PendingNode {
            rows,
            depth: 0,
            attachment: Attachment::Root,
        }];
        while let Some(task) = pending.pop() {
            let (sum, sum_sq, weight) = self.totals(&task.rows);
            let prediction = self.objective.leaf_value(sum, weight);
            let node_index = self.push_node(BuildNode::leaf(prediction))?;
            match task.attachment {
                Attachment::Root => debug_assert_eq!(node_index, 0),
                Attachment::Left(parent) => self.nodes[parent as usize].left = node_index,
                Attachment::Right(parent) => self.nodes[parent as usize].right = node_index,
            }

            let depth_limited = self
                .config
                .max_depth
                .is_some_and(|max_depth| task.depth >= max_depth);
            if depth_limited
                || weight < self.config.min_samples_split as u64
                || self.objective.pure(sum, sum_sq, weight)
            {
                continue;
            }

            let parent_impurity = self.objective.impurity(sum, sum_sq, weight);
            let Some(split) = self.best_split(&task.rows, sum, sum_sq, weight) else {
                continue;
            };
            // A split must reduce impurity, not merely reshuffle equal values.
            if split.score >= parent_impurity {
                continue;
            }

            let mut left_rows = Vec::with_capacity(task.rows.len());
            let mut right_rows = Vec::with_capacity(task.rows.len());
            for row in task.rows {
                if self.data.row(row).expect("known row")[split.feature] <= split.threshold {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            }
            self.nodes[node_index as usize] = BuildNode {
                feature: split.feature as u32,
                left: NO_CHILD,
                right: NO_CHILD,
                payload: split.threshold,
            };
            // Right is pushed first so the left subtree remains contiguous and
            // the final node buffer is deterministic pre-order traversal.
            pending.push(PendingNode {
                rows: right_rows,
                depth: task.depth + 1,
                attachment: Attachment::Right(node_index),
            });
            pending.push(PendingNode {
                rows: left_rows,
                depth: task.depth + 1,
                attachment: Attachment::Left(node_index),
            });
        }
        PackedTree::from_build_nodes(self.nodes, self.data.columns())
    }

    fn push_node(&mut self, node: BuildNode) -> Result<u32, ModelError> {
        let index = u32::try_from(self.nodes.len()).map_err(|_| ModelError::TreeTooLarge)?;
        if index == NO_CHILD {
            return Err(ModelError::TreeTooLarge);
        }
        self.nodes.push(node);
        Ok(index)
    }

    fn totals(&self, rows: &[usize]) -> (f64, f64, u64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut weight = 0u64;
        for &row in rows {
            let count = u64::from(self.counts[row]);
            let value = self.objective.value(&self.targets[row]);
            let count_f = count as f64;
            sum += value * count_f;
            sum_sq += value * value * count_f;
            weight += count;
        }
        (sum, sum_sq, weight)
    }

    fn best_split(
        &mut self,
        rows: &[usize],
        total_sum: f64,
        total_sum_sq: f64,
        total_weight: u64,
    ) -> Option<Split> {
        let feature_count = match self.config.max_features {
            MaxFeatures::All => self.data.columns(),
            MaxFeatures::Sqrt => integer_sqrt(self.data.columns()).max(1),
            MaxFeatures::Count(count) => count,
        };
        let mut features: Vec<usize> = (0..self.data.columns()).collect();
        for index in 0..feature_count {
            let other = index + self.rng.index(features.len() - index);
            features.swap(index, other);
        }

        let mut ordered = Vec::with_capacity(rows.len());
        let mut best: Option<Split> = None;
        for &feature in &features[..feature_count] {
            ordered.clear();
            ordered.extend_from_slice(rows);
            ordered.sort_unstable_by(|&left, &right| {
                self.data.row(left).expect("known row")[feature]
                    .total_cmp(&self.data.row(right).expect("known row")[feature])
                    .then_with(|| left.cmp(&right))
            });

            let mut left_sum = 0.0;
            let mut left_sum_sq = 0.0;
            let mut left_weight = 0u64;
            for boundary in 0..ordered.len().saturating_sub(1) {
                let row = ordered[boundary];
                let count = u64::from(self.counts[row]);
                let value = self.objective.value(&self.targets[row]);
                let count_f = count as f64;
                left_sum += value * count_f;
                left_sum_sq += value * value * count_f;
                left_weight += count;

                let a = self.data.row(row).expect("known row")[feature];
                let b = self.data.row(ordered[boundary + 1]).expect("known row")[feature];
                if a == b {
                    continue;
                }
                let right_weight = total_weight - left_weight;
                if left_weight < self.config.min_samples_leaf as u64
                    || right_weight < self.config.min_samples_leaf as u64
                {
                    continue;
                }
                let right_sum = total_sum - left_sum;
                let right_sum_sq = total_sum_sq - left_sum_sq;
                let left_impurity = self.objective.impurity(left_sum, left_sum_sq, left_weight);
                let right_impurity = self
                    .objective
                    .impurity(right_sum, right_sum_sq, right_weight);
                let score = (left_weight as f64 * left_impurity
                    + right_weight as f64 * right_impurity)
                    / total_weight as f64;
                if best.as_ref().is_none_or(|current| score < current.score) {
                    best = Some(Split {
                        feature,
                        threshold: split_threshold(a, b),
                        score,
                    });
                }
            }
        }
        best
    }
}

fn split_threshold(a: f32, b: f32) -> f32 {
    let midpoint = a + (b - a) * 0.5;
    if midpoint.is_finite() && midpoint >= a && midpoint < b {
        midpoint
    } else {
        // `a` itself exactly represents the desired `<= a` partition, including
        // adjacent floats and opposite-sign values whose subtraction overflows.
        a
    }
}

fn integer_sqrt(value: usize) -> usize {
    // Floating point is only an initial estimate; the corrections make this
    // exact even for usize values beyond f64's integer precision.
    let mut root = (value as f64).sqrt() as usize;
    while root.checked_add(1).is_some_and(|next| next <= value / next) {
        root += 1;
    }
    while root != 0 && root > value / root {
        root -= 1;
    }
    root
}
