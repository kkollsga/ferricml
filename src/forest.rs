//! Deterministic random forests for dense, finite `f32` data.
//!
//! Training uses exact CART splits.  The implementation deliberately owns its
//! small RNG so a model does not change when a dependency, platform, or worker
//! count changes.

use crate::api::{Classifier, Estimator, HasParams, ModelError, Regressor};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};
use crate::ensemble::{MaxFeatures, RandomForestClassifierParams, RandomForestRegressorParams};
use std::thread;

const LEAF_FEATURE: u32 = u32::MAX;
const NO_CHILD: u32 = u32::MAX;
const LEFT_IS_LEAF: u32 = 1 << 31;
const RIGHT_IS_LEAF: u32 = 1 << 30;
const FEATURE_MASK: u32 = RIGHT_IS_LEAF - 1;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ForestConfig {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    bootstrap: bool,
    random_state: u64,
    n_jobs: usize,
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

/// Temporary uniform node used while building a tree.
///
/// A leaf has `feature == u32::MAX`, no children, and stores its prediction in
/// `payload`. A branch stores its threshold in the same field and sends values
/// `<= threshold` left and other values right.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct BuildNode {
    feature: u32,
    left: u32,
    right: u32,
    /// Split threshold for branches, prediction for leaves.
    payload: f32,
}

impl BuildNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.feature == LEAF_FEATURE
    }

    #[inline]
    fn threshold(&self) -> f32 {
        debug_assert!(!self.is_leaf());
        self.payload
    }

    #[inline]
    fn value(&self) -> f32 {
        debug_assert!(self.is_leaf());
        self.payload
    }

    fn leaf(value: f32) -> Self {
        Self {
            feature: LEAF_FEATURE,
            left: NO_CHILD,
            right: NO_CHILD,
            payload: value,
        }
    }
}

/// One fixed-width inference branch. Two flag bits distinguish child branch
/// indices from inline leaf `f32` bits. This avoids fetching a second node or
/// array entry merely to discover a leaf.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct PackedNode {
    left: u32,
    right: u32,
    threshold: f32,
    feature_and_flags: u32,
}

/// A compact decision tree optimized for inference.
#[derive(Clone, Debug, PartialEq)]
struct PackedTree {
    nodes: Vec<PackedNode>,
    root_leaf: Option<f32>,
}

impl PackedTree {
    fn from_build_nodes(build: Vec<BuildNode>) -> Result<Self, ModelError> {
        if build[0].is_leaf() {
            return Ok(Self {
                nodes: Vec::new(),
                root_leaf: Some(build[0].value()),
            });
        }
        let mut branch_indices = vec![NO_CHILD; build.len()];
        let mut nodes = Vec::new();
        // Assign tokens in the builder's pre-order so a parent and its left
        // descendants remain adjacent during inference.
        for (index, node) in build.iter().enumerate() {
            if !node.is_leaf() {
                let packed_index =
                    u32::try_from(nodes.len()).map_err(|_| ModelError::TreeTooLarge)?;
                nodes.push(PackedNode {
                    left: 0,
                    right: 0,
                    threshold: node.threshold(),
                    feature_and_flags: node.feature,
                });
                branch_indices[index] = packed_index;
            }
        }
        for (index, node) in build.iter().enumerate().filter(|(_, node)| !node.is_leaf()) {
            let left = build[node.left as usize];
            let right = build[node.right as usize];
            let packed = &mut nodes[branch_indices[index] as usize];
            if left.is_leaf() {
                packed.feature_and_flags |= LEFT_IS_LEAF;
                packed.left = left.value().to_bits();
            } else {
                packed.left = branch_indices[node.left as usize];
            }
            if right.is_leaf() {
                packed.feature_and_flags |= RIGHT_IS_LEAF;
                packed.right = right.value().to_bits();
            } else {
                packed.right = branch_indices[node.right as usize];
            }
        }
        Ok(Self {
            nodes,
            root_leaf: None,
        })
    }

    #[inline(always)]
    fn predict(&self, row: &[f32]) -> f32 {
        if let Some(value) = self.root_leaf {
            return value;
        }
        let mut index = 0usize;
        loop {
            // Tree construction validates every token before the fitted model
            // becomes observable, and prediction validates the row width once
            // per batch. Avoid repeating those bounds checks at every level.
            let node = unsafe { self.nodes.get_unchecked(index) };
            let value =
                unsafe { *row.get_unchecked((node.feature_and_flags & FEATURE_MASK) as usize) };
            if value <= node.threshold {
                if node.feature_and_flags & LEFT_IS_LEAF != 0 {
                    return f32::from_bits(node.left);
                }
                index = node.left as usize;
            } else {
                if node.feature_and_flags & RIGHT_IS_LEAF != 0 {
                    return f32::from_bits(node.right);
                }
                index = node.right as usize;
            }
        }
    }
}

/// A random-forest binary classifier.
///
/// Class labels are sorted, and probability columns follow that order. Models
/// fitted on a single class expose one probability column containing `1.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestClassifier {
    n_features_in: usize,
    params: RandomForestClassifierParams,
    classes: Vec<u8>,
    trees: Vec<PackedTree>,
}

/// A random-forest regressor.  Predictions are averages of tree leaf means.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestRegressor {
    n_features_in: usize,
    params: RandomForestRegressorParams,
    trees: Vec<PackedTree>,
}

impl RandomForestClassifier {
    /// Returns the feature width required by this model.
    #[inline]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact parameters used to fit this model.
    #[inline]
    pub fn get_params(&self) -> &RandomForestClassifierParams {
        &self.params
    }

    /// Returns sorted class labels observed during fitting.
    #[inline]
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.as_slice().len(), &config)?;
        for (index, &value) in targets.as_slice().iter().enumerate() {
            if value > 1 {
                return Err(ModelError::InvalidBinaryTarget { index, value });
            }
        }
        let saw_zero = targets.as_slice().contains(&0);
        let saw_one = targets.as_slice().contains(&1);
        let classes = match (saw_zero, saw_one) {
            (true, true) => vec![0, 1],
            (true, false) => vec![0],
            (false, true) => vec![1],
            (false, false) => unreachable!("non-empty validated binary targets"),
        };
        let trees = train_forest(data, targets.as_slice(), &config, Classification)?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            trees,
        })
    }

    /// Predicts the class label for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(self.classes[0]);
        }
        let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
        // `classes` is sorted as [0, 1]. An exact tie selects its first class.
        Ok(u8::from(positive > 0.5))
    }

    /// Predicts one label per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        if self.classes.len() == 1 {
            check_prediction_data(data, data.rows(), data.rows(), self.n_features_in)?;
            return Ok(vec![self.classes[0]; data.rows()]);
        }
        // The allocating API may use a temporary score buffer. Processing one
        // tree across the batch keeps its nodes hot and is materially faster
        // for the locked 32+ row workloads. The `_into` label API remains the
        // strictly allocation-free option.
        let mut scores = vec![0.0; data.rows()];
        self.accumulate_positive_into(data, &mut scores)?;
        Ok(scores
            .into_iter()
            .map(|positive| u8::from(positive > 0.5))
            .collect())
    }

    /// Predicts one label per row without allocating.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(self.classes[0]);
            return Ok(());
        }
        for (row, slot) in data.iter_rows().zip(output) {
            let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
            *slot = u8::from(positive > 0.5);
        }
        Ok(())
    }

    /// Predicts probabilities for one sample in [`Self::classes`] order.
    pub fn predict_proba_one(&self, row: &[f32]) -> Result<Vec<f32>, ModelError> {
        let mut output = vec![0.0; self.classes.len()];
        self.predict_proba_one_into(row, &mut output)?;
        Ok(output)
    }

    /// Predicts probabilities for one sample into caller-owned storage.
    pub fn predict_proba_one_into(
        &self,
        row: &[f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_row(row, self.n_features_in)?;
        check_output_len(output.len(), self.classes.len())?;
        if self.classes.len() == 1 {
            output[0] = 1.0;
        } else {
            let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
            output[0] = 1.0 - positive;
            output[1] = positive;
        }
        Ok(())
    }

    /// Predicts row-major probabilities, allocating `rows * classes().len()`
    /// values.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities without allocating.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let expected = probability_output_len(data.rows(), self.classes.len())?;
        check_prediction_data(data, output.len(), expected, self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(1.0);
            return Ok(());
        }
        output.fill(0.0);
        for tree in &self.trees {
            for (row, probabilities) in data.iter_rows().zip(output.chunks_exact_mut(2)) {
                probabilities[1] += tree.predict(row);
            }
        }
        for probabilities in output.chunks_exact_mut(2) {
            let positive = (probabilities[1] / self.trees.len() as f32).clamp(0.0, 1.0);
            probabilities[0] = 1.0 - positive;
            probabilities[1] = positive;
        }
        Ok(())
    }

    /// Returns the requested fitted-class probability for one sample.
    pub fn predict_class_proba_one(&self, row: &[f32], class: u8) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        let class_index = self.class_index(class)?;
        if self.classes.len() == 1 {
            return Ok(1.0);
        }
        let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
        Ok(if class_index == 0 {
            1.0 - positive
        } else {
            positive
        })
    }

    /// Predicts one fitted-class probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_class_proba(self, data, class)
    }

    /// Predicts one fitted-class probability column without allocating.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let class_index = self.class_index(class)?;
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(1.0);
            return Ok(());
        }
        self.accumulate_positive_into(data, output)?;
        if class_index == 0 {
            for slot in output {
                *slot = 1.0 - *slot;
            }
        }
        Ok(())
    }

    fn accumulate_positive_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        output.fill(0.0);
        for tree in &self.trees {
            for (row, slot) in data.iter_rows().zip(output.iter_mut()) {
                *slot += tree.predict(row);
            }
        }
        for slot in output {
            *slot = (*slot / self.trees.len() as f32).clamp(0.0, 1.0);
        }
        Ok(())
    }

    #[inline]
    fn class_index(&self, class: u8) -> Result<usize, ModelError> {
        self.classes
            .binary_search(&class)
            .map_err(|_| ModelError::UnknownClass { class })
    }

    /// Returns the positive-class probability for one sample.
    ///
    /// This explicit method preserves the Phase A probability behavior. The
    /// label and two-column probability methods land in Phase B.
    pub fn predict_positive_proba(&self, row: &[f32]) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(f32::from(self.classes[0]));
        }
        Ok(mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0))
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub fn predict_positive_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(f32::from(self.classes[0]));
            return Ok(());
        }
        for (index, slot) in output.iter_mut().enumerate() {
            *slot =
                mean_tree_prediction(&self.trees, data.row(index).expect("validated row index"))
                    .clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Internal bytes used only for deterministic implementation tests.
    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        packed_model_bytes(self.n_features_in, &self.trees, b"FRFC")
    }
}

impl Estimator for RandomForestClassifier {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Classifier for RandomForestClassifier {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        RandomForestClassifier::predict_into(self, data, output)
    }

    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        RandomForestClassifier::predict_proba_into(self, data, output)
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        RandomForestClassifier::predict_class_proba_into(self, data, class, output)
    }
}

impl HasParams for RandomForestClassifier {
    type Params = RandomForestClassifierParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl RandomForestRegressor {
    /// Returns the feature width required by this model.
    #[inline]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact parameters used to fit this model.
    #[inline]
    pub fn get_params(&self) -> &RandomForestRegressorParams {
        &self.params
    }

    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: RandomForestRegressorParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.as_slice().len(), &config)?;
        for (index, value) in targets.as_slice().iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::NonFiniteTarget { index });
            }
        }
        let trees = train_forest(data, targets.as_slice(), &config, Regression)?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            trees,
        })
    }

    /// Predicts one regression value for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        Ok(mean_tree_prediction(&self.trees, row))
    }

    /// Predicts one value per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        for (index, slot) in output.iter_mut().enumerate() {
            *slot =
                mean_tree_prediction(&self.trees, data.row(index).expect("validated row index"));
        }
        Ok(())
    }

    /// Internal bytes used only for deterministic implementation tests.
    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        packed_model_bytes(self.n_features_in, &self.trees, b"FRFR")
    }
}

impl Estimator for RandomForestRegressor {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Regressor for RandomForestRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        RandomForestRegressor::predict_into(self, data, output)
    }
}

impl HasParams for RandomForestRegressor {
    type Params = RandomForestRegressorParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

fn mean_tree_prediction(trees: &[PackedTree], row: &[f32]) -> f32 {
    let sum: f32 = trees.iter().map(|tree| tree.predict(row)).sum();
    sum / trees.len() as f32
}

fn check_row(row: &[f32], expected: usize) -> Result<(), ModelError> {
    if row.len() != expected {
        return Err(ModelError::FeatureDimension {
            expected,
            actual: row.len(),
        });
    }
    if let Some(column) = row.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteFeature { row: 0, column });
    }
    Ok(())
}

fn check_prediction_data(
    data: &MatrixView<'_>,
    output_len: usize,
    expected_output_len: usize,
    expected_features: usize,
) -> Result<(), ModelError> {
    if data.columns() != expected_features {
        return Err(ModelError::FeatureDimension {
            expected: expected_features,
            actual: data.columns(),
        });
    }
    check_output_len(output_len, expected_output_len)
}

fn check_output_len(actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::OutputLength { expected, actual });
    }
    Ok(())
}

fn probability_output_len(rows: usize, classes: usize) -> Result<usize, ModelError> {
    rows.checked_mul(classes)
        .ok_or(ModelError::OutputShapeOverflow {
            rows,
            columns: classes,
        })
}

fn validate_common(
    data: &MatrixView<'_>,
    target_len: usize,
    config: &ForestConfig,
) -> Result<(), ModelError> {
    if data.rows() == 0 || data.columns() == 0 {
        return Err(ModelError::EmptyData);
    }
    if target_len == 0 {
        return Err(ModelError::EmptyTargets);
    }
    if target_len != data.rows() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: target_len,
        });
    }
    if config.n_estimators == 0 {
        return Err(ModelError::InvalidEstimatorCount);
    }
    if config.max_depth == Some(0) {
        return Err(ModelError::InvalidMaxDepth);
    }
    if config.min_samples_split < 2 {
        return Err(ModelError::InvalidMinSamplesSplit);
    }
    if config.min_samples_leaf == 0 {
        return Err(ModelError::InvalidMinSamplesLeaf);
    }
    if config.n_jobs == 0 {
        return Err(ModelError::InvalidJobCount);
    }
    if data.rows() > u32::MAX as usize {
        return Err(ModelError::TooManyRows);
    }
    if data.columns() > FEATURE_MASK as usize {
        return Err(ModelError::TooManyFeatures);
    }
    if let MaxFeatures::Count(requested) = config.max_features
        && (requested == 0 || requested > data.columns())
    {
        return Err(ModelError::InvalidMaxFeatures {
            requested,
            available: data.columns(),
        });
    }
    for row in 0..data.rows() {
        for (column, value) in data
            .row(row)
            .expect("validated row index")
            .iter()
            .enumerate()
        {
            if !value.is_finite() {
                return Err(ModelError::NonFiniteFeature { row, column });
            }
        }
    }
    Ok(())
}

trait Objective<Y>: Copy + Send + Sync {
    fn value(self, y: &Y) -> f64;
    fn impurity(self, sum: f64, sum_sq: f64, weight: u64) -> f64;
    fn leaf_value(self, sum: f64, weight: u64) -> f32 {
        (sum / weight as f64) as f32
    }
    fn pure(self, sum: f64, sum_sq: f64, weight: u64) -> bool;
}

#[derive(Clone, Copy)]
struct Classification;

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
struct Regression;

impl Objective<f32> for Regression {
    fn value(self, y: &f32) -> f64 {
        f64::from(*y)
    }

    fn impurity(self, sum: f64, sum_sq: f64, weight: u64) -> f64 {
        // Population variance.  Clamp cancellation noise at zero.
        (sum_sq / weight as f64 - (sum / weight as f64).powi(2)).max(0.0)
    }

    fn pure(self, sum: f64, sum_sq: f64, weight: u64) -> bool {
        self.impurity(sum, sum_sq, weight) == 0.0
    }
}

fn train_forest<Y, O>(
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
        PackedTree::from_build_nodes(self.nodes)
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

/// SplitMix64 with rejection-sampled bounded integers.
struct OwnedRng {
    state: u64,
}

impl OwnedRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bound = upper as u64;
        let reject_below = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                return (value % bound) as usize;
            }
        }
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_tree_seed(global_seed: u64, tree_index: u64) -> u64 {
    mix64(global_seed ^ tree_index.wrapping_mul(0xd1b5_4a32_d192_ed03))
}

#[cfg(test)]
fn packed_model_bytes(n_features: usize, trees: &[PackedTree], magic: &[u8; 4]) -> Vec<u8> {
    let node_count: usize = trees.iter().map(|tree| tree.nodes.len()).sum();
    let mut bytes = Vec::with_capacity(24 + trees.len() * 13 + node_count * 16);
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(n_features as u64).to_le_bytes());
    bytes.extend_from_slice(&(trees.len() as u64).to_le_bytes());
    for tree in trees {
        bytes.push(u8::from(tree.root_leaf.is_some()));
        bytes.extend_from_slice(&tree.root_leaf.unwrap_or_default().to_bits().to_le_bytes());
        bytes.extend_from_slice(&(tree.nodes.len() as u64).to_le_bytes());
        for node in &tree.nodes {
            bytes.extend_from_slice(&node.feature_and_flags.to_le_bytes());
            bytes.extend_from_slice(&node.left.to_le_bytes());
            bytes.extend_from_slice(&node.right.to_le_bytes());
            bytes.extend_from_slice(&node.threshold.to_bits().to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinaryTargets, DenseMatrix, RegressionTargets};
    use sha2::{Digest, Sha256};

    fn matrix(rows: &[&[f32]]) -> DenseMatrix {
        let cols = rows.first().map_or(0, |row| row.len());
        assert!(rows.iter().all(|row| row.len() == cols));
        let values = rows.iter().flat_map(|row| row.iter().copied()).collect();
        DenseMatrix::new(values, rows.len(), cols).unwrap()
    }

    fn classifier_params(random_state: u64) -> RandomForestClassifierParams {
        RandomForestClassifierParams::default()
            .with_n_estimators(31)
            .with_max_depth(Some(8))
            .with_max_features(MaxFeatures::All)
            .with_random_state(random_state)
    }

    fn regressor_params(random_state: u64) -> RandomForestRegressorParams {
        RandomForestRegressorParams::default()
            .with_n_estimators(31)
            .with_max_depth(Some(8))
            .with_max_features(MaxFeatures::All)
            .with_random_state(random_state)
    }

    #[test]
    fn classifies_separable_data_and_probabilities_are_bounded() {
        let x = matrix(&[
            &[-3.0],
            &[-2.0],
            &[-1.0],
            &[-0.5],
            &[0.5],
            &[1.0],
            &[2.0],
            &[3.0],
        ]);
        let y = BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).unwrap();
        let forest = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(4)).unwrap();
        let mut predictions = vec![0.0; x.rows()];
        forest
            .predict_positive_proba_into(&x.as_view(), &mut predictions)
            .unwrap();
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!(predictions[..4].iter().all(|&p| p < 0.5));
        assert!(predictions[4..].iter().all(|&p| p > 0.5));
    }

    #[test]
    fn nonlinear_forest_learns_repeated_xor() {
        let x = matrix(&[
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
        ]);
        let y = BinaryTargets::new(vec![0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]).unwrap();
        let cfg = classifier_params(99).with_n_estimators(101);
        let forest = RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap();
        for (row, &expected) in y.as_slice().iter().take(4).enumerate() {
            let p = forest.predict_positive_proba(x.row(row).unwrap()).unwrap();
            assert_eq!(p >= 0.5, expected == 1, "row {row}: {p}");
        }
    }

    #[test]
    fn regresses_piecewise_values() {
        let x = matrix(&[
            &[-3.0],
            &[-2.0],
            &[-1.0],
            &[0.0],
            &[1.0],
            &[2.0],
            &[3.0],
            &[4.0],
        ]);
        let y = RegressionTargets::new(vec![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]).unwrap();
        let cfg = regressor_params(7).with_n_estimators(61);
        let forest = RandomForestRegressor::fit(&x.as_view(), &y, cfg).unwrap();
        let mut output = vec![0.0; x.rows()];
        forest.predict_into(&x.as_view(), &mut output).unwrap();
        let mae = output
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / output.len() as f32;
        assert!(mae < 1.5, "mae={mae}, predictions={output:?}");
    }

    #[test]
    fn reference_defaults_distinguish_classification_and_regression() {
        assert_eq!(
            RandomForestClassifierParams::default().max_features(),
            MaxFeatures::Sqrt
        );
        assert_eq!(
            RandomForestRegressorParams::default().max_features(),
            MaxFeatures::All
        );
    }

    #[test]
    fn exact_classifier_split_and_leaf_probabilities_match_the_oracle() {
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let y = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let cfg = RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All);
        let forest = RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap();
        let tree = &forest.trees[0];
        assert_eq!(tree.nodes.len(), 1);
        let root = &tree.nodes[0];
        assert_eq!(root.feature_and_flags & FEATURE_MASK, 0);
        assert_eq!(root.threshold, 1.5);
        assert_ne!(root.feature_and_flags & LEFT_IS_LEAF, 0);
        assert_ne!(root.feature_and_flags & RIGHT_IS_LEAF, 0);
        assert_eq!(f32::from_bits(root.left), 0.0);
        assert_eq!(f32::from_bits(root.right), 1.0);

        let leaf_cfg = RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_min_samples_split(5);
        let leaf = RandomForestClassifier::fit(&x.as_view(), &y, leaf_cfg).unwrap();
        assert_eq!(leaf.predict_positive_proba(&[100.0]).unwrap(), 0.5);
    }

    #[test]
    fn exact_regression_leaf_is_the_target_mean() {
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let y = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cfg = RandomForestRegressorParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_min_samples_split(5);
        let forest = RandomForestRegressor::fit(&x.as_view(), &y, cfg).unwrap();
        assert_eq!(forest.predict_one(&[-50.0]).unwrap(), 2.5);

        let stump_cfg = RandomForestRegressorParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_max_depth(Some(1));
        let stump = RandomForestRegressor::fit(&x.as_view(), &y, stump_cfg).unwrap();
        assert_eq!(stump.trees[0].nodes[0].threshold, 1.5);
        assert_eq!(stump.predict_one(&[-50.0]).unwrap(), 1.5);
        assert_eq!(stump.predict_one(&[100.0]).unwrap(), 3.5);
    }

    #[test]
    fn model_is_identical_across_repeats_and_thread_counts() {
        let x = matrix(&[
            &[0.0, 3.0],
            &[1.0, 2.0],
            &[2.0, 1.0],
            &[3.0, 0.0],
            &[4.0, 7.0],
            &[5.0, 6.0],
            &[6.0, 5.0],
            &[7.0, 4.0],
        ]);
        let y = BinaryTargets::new(vec![0, 1, 1, 0, 1, 0, 0, 1]).unwrap();
        let one = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(123)).unwrap();
        let repeat = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(123)).unwrap();
        let parallel_config = classifier_params(123).with_n_jobs(crate::ensemble::NJobs::Count(4));
        let parallel = RandomForestClassifier::fit(&x.as_view(), &y, parallel_config).unwrap();
        assert_eq!(one.to_bytes(), repeat.to_bytes());
        assert_eq!(one.to_bytes(), parallel.to_bytes());
    }

    #[test]
    fn packed_classifier_and_regressor_fingerprints_are_frozen() {
        let x = matrix(&[
            &[0.0, 3.0],
            &[1.0, 2.0],
            &[2.0, 1.0],
            &[3.0, 0.0],
            &[4.0, 7.0],
            &[5.0, 6.0],
            &[6.0, 5.0],
            &[7.0, 4.0],
        ]);
        let classifier = RandomForestClassifier::fit(
            &x.as_view(),
            &BinaryTargets::new(vec![0, 1, 1, 0, 1, 0, 0, 1]).unwrap(),
            classifier_params(123),
        )
        .unwrap();
        let regressor = RandomForestRegressor::fit(
            &x.as_view(),
            &RegressionTargets::new(vec![0.0, 1.0, 1.5, 2.5, 8.0, 7.0, 6.0, 5.0]).unwrap(),
            regressor_params(123),
        )
        .unwrap();
        let regressor_repeat = RandomForestRegressor::fit(
            &x.as_view(),
            &RegressionTargets::new(vec![0.0, 1.0, 1.5, 2.5, 8.0, 7.0, 6.0, 5.0]).unwrap(),
            regressor_params(123),
        )
        .unwrap();
        assert_eq!(regressor.to_bytes(), regressor_repeat.to_bytes());

        for (name, bytes, expected_len, expected_digest) in [
            (
                "classifier",
                classifier.to_bytes(),
                1595,
                [
                    180, 124, 71, 225, 4, 107, 44, 127, 181, 142, 154, 67, 201, 35, 134, 98, 57,
                    65, 187, 73, 172, 213, 231, 42, 36, 177, 233, 251, 92, 178, 60, 101,
                ],
            ),
            (
                "regressor",
                regressor.to_bytes(),
                2587,
                [
                    100, 242, 214, 182, 27, 5, 82, 121, 64, 157, 253, 240, 23, 181, 188, 179, 232,
                    105, 178, 228, 17, 225, 213, 116, 97, 196, 21, 239, 13, 206, 129, 77,
                ],
            ),
        ] {
            assert_eq!(bytes.len(), expected_len, "{name} packed bytes changed");
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            assert_eq!(digest, expected_digest, "{name} packed bytes changed");
        }
    }

    #[test]
    fn bootstrap_and_seed_are_deterministic_but_seed_affects_model() {
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
        let y = BinaryTargets::new(vec![0, 1, 0, 1, 0, 1]).unwrap();
        let a = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(8)).unwrap();
        let b = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(8)).unwrap();
        let c = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(9)).unwrap();
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert_ne!(a.to_bytes(), c.to_bytes());
    }

    #[test]
    fn rejects_invalid_configuration_and_data() {
        let x = matrix(&[&[0.0], &[1.0]]);
        let y = BinaryTargets::new(vec![0, 1]).unwrap();
        let cfg = classifier_params(1).with_n_estimators(0);
        assert_eq!(
            RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap_err(),
            ModelError::InvalidEstimatorCount
        );
        let cfg = classifier_params(1).with_min_samples_split(1);
        assert_eq!(
            RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap_err(),
            ModelError::InvalidMinSamplesSplit
        );
        let cfg = classifier_params(1).with_min_samples_leaf(0);
        assert_eq!(
            RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap_err(),
            ModelError::InvalidMinSamplesLeaf
        );
        let cfg = classifier_params(1).with_max_features(MaxFeatures::Count(2));
        assert!(matches!(
            RandomForestClassifier::fit(&x.as_view(), &y, cfg),
            Err(ModelError::InvalidMaxFeatures { .. })
        ));
    }

    #[test]
    fn checks_prediction_dimensions_and_output_size() {
        let x = matrix(&[&[0.0], &[1.0]]);
        let y = BinaryTargets::new(vec![0, 1]).unwrap();
        let forest = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(2)).unwrap();
        assert!(matches!(
            forest.predict_positive_proba(&[0.0, 1.0]),
            Err(ModelError::FeatureDimension { .. })
        ));
        let mut too_short = [0.0];
        assert!(matches!(
            forest.predict_positive_proba_into(&x.as_view(), &mut too_short),
            Err(ModelError::OutputLength { .. })
        ));
    }

    #[test]
    fn every_packed_tree_has_valid_topology() {
        let x = matrix(&[
            &[0.0, 2.0],
            &[1.0, 3.0],
            &[2.0, 0.0],
            &[3.0, 1.0],
            &[4.0, 6.0],
            &[5.0, 7.0],
            &[6.0, 4.0],
            &[7.0, 5.0],
        ]);
        let y = BinaryTargets::new(vec![0, 0, 1, 1, 0, 1, 0, 1]).unwrap();
        let forest = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(55)).unwrap();
        assert_eq!(std::mem::size_of::<PackedNode>(), 16);
        for tree in &forest.trees {
            assert!(!tree.nodes.is_empty() || tree.root_leaf.is_some());
            assert!(tree.root_leaf.is_none_or(f32::is_finite));
            for node in &tree.nodes {
                assert!(((node.feature_and_flags & FEATURE_MASK) as usize) < forest.n_features_in);
                assert!(node.threshold.is_finite());
                if node.feature_and_flags & LEFT_IS_LEAF != 0 {
                    assert!(f32::from_bits(node.left).is_finite());
                } else {
                    assert!((node.left as usize) < tree.nodes.len());
                }
                if node.feature_and_flags & RIGHT_IS_LEAF != 0 {
                    assert!(f32::from_bits(node.right).is_finite());
                } else {
                    assert!((node.right as usize) < tree.nodes.len());
                }
            }
        }
    }

    #[test]
    fn pathological_deep_tree_uses_an_explicit_builder_stack() {
        const ROWS: usize = 4096;
        let values = (0..ROWS).map(|row| row as f32).collect();
        let labels = (0..ROWS).map(|row| (row & 1) as u8).collect();
        let x = DenseMatrix::new(values, ROWS, 1).unwrap();
        let y = BinaryTargets::new(labels).unwrap();
        let cfg = RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All);
        let forest = RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap();
        assert_eq!(forest.trees[0].nodes.len(), ROWS - 1);
    }
}
