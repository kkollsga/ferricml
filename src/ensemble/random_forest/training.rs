use super::parameters::{RandomForestClassifierParams, RandomForestRegressorParams};
use crate::api::ModelError;
use crate::data::MatrixView;
use crate::numeric::{OwnedRng, derive_tree_seed};
use crate::tree::{
    ClassTree, GrowerConfig, Objective, PackedTree, grow_class_tree, grow_tree,
    unbootstrapped_sample,
};
use std::thread;

/// What an ensemble adds around the shared grower.
///
/// The per-tree limits live in [`GrowerConfig`] rather than being repeated
/// here, so a forest member and a standalone tree are grown under one type
/// carrying one set of values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ForestConfig {
    pub(super) n_estimators: usize,
    pub(super) bootstrap: bool,
    pub(super) random_state: u64,
    pub(super) n_jobs: usize,
    pub(super) grower: GrowerConfig,
}

impl From<&RandomForestClassifierParams> for ForestConfig {
    fn from(params: &RandomForestClassifierParams) -> Self {
        Self {
            n_estimators: params.n_estimators(),
            bootstrap: params.bootstrap(),
            random_state: params.random_state(),
            n_jobs: params.n_jobs().resolved(),
            grower: GrowerConfig {
                max_depth: params.max_depth(),
                min_samples_split: params.min_samples_split(),
                min_samples_leaf: params.min_samples_leaf(),
                max_features: params.max_features(),
            },
        }
    }
}

impl From<&RandomForestRegressorParams> for ForestConfig {
    fn from(params: &RandomForestRegressorParams) -> Self {
        Self {
            n_estimators: params.n_estimators(),
            bootstrap: params.bootstrap(),
            random_state: params.random_state(),
            n_jobs: params.n_jobs().resolved(),
            grower: GrowerConfig {
                max_depth: params.max_depth(),
                min_samples_split: params.min_samples_split(),
                min_samples_leaf: params.min_samples_leaf(),
                max_features: params.max_features(),
            },
        }
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
///
/// Note what this does *not* consume: with `bootstrap` disabled and no sample
/// weights it draws nothing at all, which is what lets a one-tree no-bootstrap
/// forest member and a standalone tree at the same derived seed be
/// bit-identical rather than merely similar.
fn tree_sample(
    rows: usize,
    bootstrap: bool,
    sample_weights: Option<&[f32]>,
    rng: &mut OwnedRng,
) -> (Vec<f64>, Vec<usize>) {
    if !bootstrap {
        // Resampling is the only thing a forest adds here, so without it this
        // is exactly the standalone tree's sample and is shared rather than
        // restated: `1.0` per eligible row, scaled by that row's weight.
        return unbootstrapped_sample(rows, sample_weights);
    }

    let Some(sample_weights) = sample_weights else {
        let mut weights = vec![0.0_f64; rows];
        for _ in 0..rows {
            weights[rng.index(rows)] += 1.0;
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
    for _ in 0..eligible.len() {
        weights[eligible[rng.index(eligible.len())]] += 1.0;
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
        grow_tree(
            data,
            targets,
            &weights,
            rows,
            &config.grower,
            objective,
            &mut rng,
        )
    })
}

/// Fits one forest of natively multiclass trees.
///
/// `class_of_row` holds each row's column in the sorted class list, so the
/// grower never touches a label value.
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
        grow_class_tree(
            data,
            class_of_row,
            classes,
            &weights,
            rows,
            &config.grower,
            &mut rng,
        )
    })
}
