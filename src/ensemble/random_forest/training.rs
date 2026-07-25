use super::parameters::{MaxFeatures, RandomForestClassifierParams, RandomForestRegressorParams};
use super::tree::{BuildNode, ClassTree, NO_CHILD, PackedTree};
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

/// One node's statistics are weighted totals rather than row counts.
///
/// `weight` is the total row weight reaching a node: the bootstrap replication
/// count of each row multiplied by its sample weight. Without sample weights
/// every term is an integer, and an integer sum accumulated in `f64` is exact
/// below `2^53`, so the unweighted arithmetic is bit-for-bit what the earlier
/// `u64` counter produced.
pub(super) trait Objective<Y>: Copy + Send + Sync {
    fn value(self, y: &Y) -> f64;
    fn impurity(self, sum: f64, sum_sq: f64, weight: f64) -> f64;
    fn leaf_value(self, sum: f64, weight: f64) -> f32 {
        (sum / weight) as f32
    }
    fn pure(self, sum: f64, sum_sq: f64, weight: f64) -> bool;
}

#[derive(Clone, Copy)]
pub(super) struct Classification;

impl Objective<u8> for Classification {
    fn value(self, y: &u8) -> f64 {
        f64::from(*y)
    }

    fn impurity(self, positives: f64, _sum_sq: f64, weight: f64) -> f64 {
        let p = positives / weight;
        2.0 * p * (1.0 - p)
    }

    fn pure(self, positives: f64, _sum_sq: f64, weight: f64) -> bool {
        positives == 0.0 || positives == weight
    }
}

#[derive(Clone, Copy)]
pub(super) struct Regression;

impl Objective<f32> for Regression {
    fn value(self, y: &f32) -> f64 {
        f64::from(*y)
    }

    fn impurity(self, sum: f64, sum_sq: f64, weight: f64) -> f64 {
        // Population variance.  Clamp cancellation noise at zero.
        let mean = sum / weight;
        (sum_sq / weight - mean * mean).max(0.0)
    }

    fn pure(self, sum: f64, sum_sq: f64, weight: f64) -> bool {
        self.impurity(sum, sum_sq, weight) == 0.0
    }
}

/// Runs `build` once per tree, in a fixed index order whatever the thread count.
///
/// Every tree's randomness comes from a seed derived from its index alone, and
/// finished trees are sorted back into index order, so a serial fit and a
/// parallel fit produce the same forest.
fn train_trees<T, F>(config: &ForestConfig, build: F) -> Result<Vec<T>, ModelError>
where
    T: Send,
    F: Fn(usize) -> Result<T, ModelError> + Sync,
{
    let worker_count = config.n_jobs.min(config.n_estimators);
    if worker_count == 1 {
        return (0..config.n_estimators).map(build).collect();
    }

    let build = &build;
    let mut indexed = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            handles.push(scope.spawn(move || {
                let mut trees = Vec::new();
                for index in (worker..config.n_estimators).step_by(worker_count) {
                    trees.push((index, build(index)));
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

/// One tree's per-row training weights and the rows it retains.
///
/// A row's weight is how many times the resample drew it multiplied by its
/// sample weight; without bootstrapping the replication count is one. A row of
/// weight zero contributes to nothing and is left out of the row list entirely.
///
/// The unweighted arm is deliberately separate rather than a special case of
/// the weighted one. It draws exactly the indices it always drew and stores
/// exact integers, so every unweighted fit is bit-for-bit unchanged.
///
/// The weighted arm resamples the **positively weighted rows only**, and draws
/// as many times as there are of them. A zero weight means the row is not in
/// the training sample at all, so it no more consumes a bootstrap draw than a
/// deleted row would — and the drawn sample can therefore never come back
/// empty, which a division by a zero total weight would otherwise produce.
fn tree_sample(
    rows: usize,
    bootstrap: bool,
    sample_weights: Option<&[f32]>,
    rng: &mut OwnedRng,
) -> (Vec<f64>, Vec<usize>) {
    let Some(sample_weights) = sample_weights else {
        let mut weights = vec![0.0_f64; rows];
        if bootstrap {
            for _ in 0..rows {
                weights[rng.index(rows)] += 1.0;
            }
        } else {
            weights.fill(1.0);
        }
        let retained = weights
            .iter()
            .enumerate()
            .filter_map(|(row, &weight)| (weight != 0.0).then_some(row))
            .collect();
        return (weights, retained);
    };

    let eligible: Vec<usize> = (0..rows).filter(|&row| sample_weights[row] > 0.0).collect();
    let mut weights = vec![0.0_f64; rows];
    if bootstrap {
        for _ in 0..eligible.len() {
            weights[eligible[rng.index(eligible.len())]] += 1.0;
        }
    } else {
        for &row in &eligible {
            weights[row] = 1.0;
        }
    }
    let mut retained = Vec::with_capacity(eligible.len());
    for &row in &eligible {
        if weights[row] != 0.0 {
            weights[row] *= f64::from(sample_weights[row]);
            retained.push(row);
        }
    }
    (weights, retained)
}

pub(super) fn train_forest<Y, O>(
    data: &MatrixView<'_>,
    targets: &[Y],
    sample_weights: Option<&[f32]>,
    config: &ForestConfig,
    objective: O,
) -> Result<Vec<PackedTree>, ModelError>
where
    Y: Sync,
    O: Objective<Y>,
{
    train_trees(config, |index| {
        let mut rng = OwnedRng::new(derive_tree_seed(config.random_state, index as u64));
        let (weights, rows) = tree_sample(data.rows(), config.bootstrap, sample_weights, &mut rng);
        let builder = TreeBuilder {
            data,
            targets,
            row_weights: &weights,
            config,
            objective,
            rng: &mut rng,
            nodes: Vec::new(),
        };
        builder.build_tree(rows)
    })
}

/// Fits one forest of natively multiclass trees.
///
/// `class_of_row` holds each row's column in the sorted class list, so the
/// builder never touches a label value.
pub(super) fn train_class_forest(
    data: &MatrixView<'_>,
    class_of_row: &[usize],
    classes: usize,
    sample_weights: Option<&[f32]>,
    config: &ForestConfig,
) -> Result<Vec<ClassTree>, ModelError> {
    train_trees(config, |index| {
        let mut rng = OwnedRng::new(derive_tree_seed(config.random_state, index as u64));
        let (weights, rows) = tree_sample(data.rows(), config.bootstrap, sample_weights, &mut rng);
        let builder = ClassTreeBuilder {
            data,
            class_of_row,
            classes,
            row_weights: &weights,
            config,
            rng: &mut rng,
            nodes: Vec::new(),
            class_weights: Vec::new(),
            weights: Vec::new(),
        };
        builder.build_tree(rows)
    })
}

/// Partially shuffles `0..columns` so its first `count` entries are the drawn
/// features, in the generator's order.
///
/// Shared by both builders because the sampling — not the sweep that follows
/// it — is what a fitted forest's reproducibility depends on. It runs once per
/// *node*, which is the crate's hottest fitting path outside the sweep itself,
/// so it is `#[inline]`: written inline this was part of its caller's body, and
/// sharing it must not quietly add a call boundary there. It returns the full
/// permutation rather than truncating, so the caller keeps the drawn count in a
/// register the way it did before.
#[inline]
fn sample_features(rng: &mut OwnedRng, columns: usize, count: usize) -> Vec<usize> {
    let mut features: Vec<usize> = (0..columns).collect();
    for index in 0..count {
        let other = index + rng.index(features.len() - index);
        features.swap(index, other);
    }
    features
}

/// How many features one node draws. Also per node, also `#[inline]`.
#[inline]
fn resolved_feature_count(max_features: MaxFeatures, columns: usize) -> usize {
    match max_features {
        MaxFeatures::All => columns,
        MaxFeatures::Sqrt => integer_sqrt(columns).max(1),
        MaxFeatures::Count(count) => count,
    }
}

struct TreeBuilder<'a, 'm, Y, O> {
    data: &'a MatrixView<'m>,
    targets: &'a [Y],
    /// Total training weight of each row in this tree's sample.
    row_weights: &'a [f64],
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
                || weight < self.config.min_samples_split as f64
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

    fn totals(&self, rows: &[usize]) -> (f64, f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut weight = 0.0;
        for &row in rows {
            let row_weight = self.row_weights[row];
            let value = self.objective.value(&self.targets[row]);
            sum += value * row_weight;
            sum_sq += value * value * row_weight;
            weight += row_weight;
        }
        (sum, sum_sq, weight)
    }

    fn best_split(
        &mut self,
        rows: &[usize],
        total_sum: f64,
        total_sum_sq: f64,
        total_weight: f64,
    ) -> Option<Split> {
        let feature_count = resolved_feature_count(self.config.max_features, self.data.columns());
        let features = sample_features(self.rng, self.data.columns(), feature_count);

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
            let mut left_weight = 0.0;
            for boundary in 0..ordered.len().saturating_sub(1) {
                let row = ordered[boundary];
                let row_weight = self.row_weights[row];
                let value = self.objective.value(&self.targets[row]);
                left_sum += value * row_weight;
                left_sum_sq += value * value * row_weight;
                left_weight += row_weight;

                let a = self.data.row(row).expect("known row")[feature];
                let b = self.data.row(ordered[boundary + 1]).expect("known row")[feature];
                if a == b {
                    continue;
                }
                let right_weight = total_weight - left_weight;
                if left_weight < self.config.min_samples_leaf as f64
                    || right_weight < self.config.min_samples_leaf as f64
                {
                    continue;
                }
                let right_sum = total_sum - left_sum;
                let right_sum_sq = total_sum_sq - left_sum_sq;
                let left_impurity = self.objective.impurity(left_sum, left_sum_sq, left_weight);
                let right_impurity = self
                    .objective
                    .impurity(right_sum, right_sum_sq, right_weight);
                let score =
                    (left_weight * left_impurity + right_weight * right_impurity) / total_weight;
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

/// Builds one natively multiclass tree.
///
/// This is deliberately a second builder rather than a generalization of
/// [`TreeBuilder`]. Making the scalar builder generic over its node statistics
/// would move `left_sum`/`left_sum_sq` out of locals and into a slice inside
/// the crate's hottest fitting loop, and this sprint holds no benchmark lock to
/// prove that costs nothing. Everything outside that loop — feature sampling,
/// threshold selection, row partitioning, bootstrap sampling, tree ordering —
/// is shared. Unifying the sweep itself is a measured change, not a tidy-up.
struct ClassTreeBuilder<'a, 'm> {
    data: &'a MatrixView<'m>,
    /// Each row's column in the sorted class list.
    class_of_row: &'a [usize],
    classes: usize,
    /// Total training weight of each row in this tree's sample.
    row_weights: &'a [f64],
    config: &'a ForestConfig,
    rng: &'a mut OwnedRng,
    nodes: Vec<BuildNode>,
    /// `classes` weighted counts per build node, parallel to `nodes`.
    class_weights: Vec<f64>,
    /// Total weight per build node, parallel to `nodes`.
    weights: Vec<f64>,
}

impl ClassTreeBuilder<'_, '_> {
    fn build_tree(mut self, rows: Vec<usize>) -> Result<ClassTree, ModelError> {
        let classes = self.classes;
        let mut totals = vec![0.0_f64; classes];
        let mut left = vec![0.0_f64; classes];
        let mut right = vec![0.0_f64; classes];
        let mut pending = vec![PendingNode {
            rows,
            depth: 0,
            attachment: Attachment::Root,
        }];
        while let Some(task) = pending.pop() {
            let weight = self.totals(&task.rows, &mut totals);
            // A leaf's stored value is derived at packing time from these
            // retained statistics, so the payload slot stays free for the
            // threshold this node may still gain.
            let node_index = self.push_node(BuildNode::leaf(0.0), &totals, weight)?;
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
                || weight < self.config.min_samples_split as f64
                || totals.contains(&weight)
            {
                continue;
            }

            let parent_impurity = gini(&totals, weight);
            let Some(split) = self.best_split(&task.rows, &totals, weight, &mut left, &mut right)
            else {
                continue;
            };
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
        ClassTree::from_build_nodes(
            self.nodes,
            &self.class_weights,
            &self.weights,
            classes,
            self.data.columns(),
        )
    }

    fn push_node(
        &mut self,
        node: BuildNode,
        class_weights: &[f64],
        weight: f64,
    ) -> Result<u32, ModelError> {
        let index = u32::try_from(self.nodes.len()).map_err(|_| ModelError::TreeTooLarge)?;
        if index == NO_CHILD {
            return Err(ModelError::TreeTooLarge);
        }
        self.nodes.push(node);
        self.class_weights.extend_from_slice(class_weights);
        self.weights.push(weight);
        Ok(index)
    }

    fn totals(&self, rows: &[usize], class_weights: &mut [f64]) -> f64 {
        class_weights.fill(0.0);
        let mut weight = 0.0;
        for &row in rows {
            let row_weight = self.row_weights[row];
            class_weights[self.class_of_row[row]] += row_weight;
            weight += row_weight;
        }
        weight
    }

    fn best_split(
        &mut self,
        rows: &[usize],
        totals: &[f64],
        total_weight: f64,
        left: &mut [f64],
        right: &mut [f64],
    ) -> Option<Split> {
        let feature_count = resolved_feature_count(self.config.max_features, self.data.columns());
        let features = sample_features(self.rng, self.data.columns(), feature_count);

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

            left.fill(0.0);
            let mut left_weight = 0.0;
            for boundary in 0..ordered.len().saturating_sub(1) {
                let row = ordered[boundary];
                let row_weight = self.row_weights[row];
                left[self.class_of_row[row]] += row_weight;
                left_weight += row_weight;

                let a = self.data.row(row).expect("known row")[feature];
                let b = self.data.row(ordered[boundary + 1]).expect("known row")[feature];
                if a == b {
                    continue;
                }
                let right_weight = total_weight - left_weight;
                if left_weight < self.config.min_samples_leaf as f64
                    || right_weight < self.config.min_samples_leaf as f64
                {
                    continue;
                }
                for (slot, (&total, &left)) in right.iter_mut().zip(totals.iter().zip(left.iter()))
                {
                    *slot = total - left;
                }
                let score = (left_weight * gini(left, left_weight)
                    + right_weight * gini(right, right_weight))
                    / total_weight;
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

/// Gini impurity of one node's weighted class counts.
///
/// `1 - Σ pₖ²` over any number of classes. At two classes it is exactly the
/// `2p(1-p)` the binary objective above uses, so the two criteria agree where
/// they overlap instead of being two different notions of impurity.
fn gini(class_weights: &[f64], weight: f64) -> f64 {
    let mut squares = 0.0_f64;
    for &count in class_weights {
        let proportion = count / weight;
        squares += proportion * proportion;
    }
    1.0 - squares
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
