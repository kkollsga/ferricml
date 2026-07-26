//! The fitted state and prediction arithmetic every bagged tree ensemble
//! shares.
//!
//! A random forest and a randomized ensemble differ in how each member tree is
//! grown, and in nothing that happens afterwards: both average scalar leaves
//! the same way, average probability vectors the same way, break the same ties,
//! and validate the same shapes. Stating that once here is what keeps the
//! difference between the two families where it actually is.

use super::training::ForestConfig;
use crate::api::{ModelError, validate_prediction};
use crate::data::{MatrixView, SampleWeights};
use crate::tree::{ClassTree, FEATURE_MASK, MaxFeatures, PackedTree};

/// Every class label is a `u8`, so no fit can observe more classes than this.
/// Scalar and single-column prediction paths keep one averaged probability row
/// on the stack at this width rather than allocating inside an `_into` method.
pub(crate) const MAX_CLASSES: usize = 256;

/// The fitted trees behind an ensemble classifier.
///
/// The two flavours are kept apart rather than unified because their leaf
/// arithmetic is genuinely different: a binary leaf stores one probability and
/// the ensemble averages that scalar, while a multiclass leaf stores one
/// probability per class and the ensemble averages the vector. Forcing the
/// binary fit through the vector path would change values it has already
/// frozen, for no gain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Forest {
    /// Trees whose leaf is the probability of class `1`.
    Binary(Vec<PackedTree>),
    /// Trees whose leaf is one probability per observed class.
    Multiclass(Vec<ClassTree>),
}

impl Forest {
    /// The scalar trees of a binary fit, for in-crate structural tests.
    #[cfg(test)]
    pub(crate) fn binary_trees(&self) -> &[PackedTree] {
        match self {
            Self::Binary(trees) => trees,
            Self::Multiclass(_) => unreachable!("binary fixture"),
        }
    }
}

/// The fitted state and prediction arithmetic of an ensemble classifier.
///
/// Every method below is what a caller sees through either public classifier
/// facade. Holding them here rather than on each estimator is the whole point
/// of the split: two ensembles differ in how their members are grown, and a
/// second copy of the batch-ordered averaging loops would be a second place for
/// a tie rule or a clamp to drift.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClassifierCore {
    pub(crate) n_features_in: usize,
    pub(crate) classes: Vec<u8>,
    pub(crate) forest: Forest,
}

impl ClassifierCore {
    /// Predicts the class label for one sample.
    pub(crate) fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        check_row(row, self.n_features_in)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    return Ok(self.classes[0]);
                }
                let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                // `classes` is sorted as [0, 1]. An exact tie selects its first class.
                Ok(u8::from(positive > 0.5))
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                average_class_probabilities(trees, row, probabilities);
                Ok(self.classes[argmax(probabilities)])
            }
        }
    }

    /// Predicts one label per row, allocating the output vector.
    pub(crate) fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        // Hoisted above the match so all three branches agree: the batch is
        // refused before any of them sizes an output buffer. Each branch's own
        // `_into` primitive repeats the check for callers that reach it
        // directly, and reports the same error from the same two numbers.
        check_prediction_data(data, data.rows(), data.rows(), self.n_features_in)?;
        match &self.forest {
            Forest::Binary(_) => {
                if self.classes.len() == 1 {
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
            Forest::Multiclass(_) => {
                let mut output = vec![0; data.rows()];
                self.predict_into(data, &mut output)?;
                Ok(output)
            }
        }
    }

    /// Predicts one label per row without allocating.
    pub(crate) fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [u8],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    output.fill(self.classes[0]);
                    return Ok(());
                }
                for (row, slot) in data.iter_rows().zip(output) {
                    let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                    *slot = u8::from(positive > 0.5);
                }
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                for (row, slot) in data.iter_rows().zip(output) {
                    // Argmax of the averaged probabilities, so a label can never
                    // disagree with the probability row a caller can read.
                    average_class_probabilities(trees, row, probabilities);
                    *slot = self.classes[argmax(probabilities)];
                }
            }
        }
        Ok(())
    }

    /// Predicts probabilities for one sample in `classes` order.
    pub(crate) fn predict_proba_one(&self, row: &[f32]) -> Result<Vec<f32>, ModelError> {
        let mut output = vec![0.0; self.classes.len()];
        self.predict_proba_one_into(row, &mut output)?;
        Ok(output)
    }

    /// Predicts probabilities for one sample into caller-owned storage.
    pub(crate) fn predict_proba_one_into(
        &self,
        row: &[f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_row(row, self.n_features_in)?;
        check_output_len(output.len(), self.classes.len())?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    output[0] = 1.0;
                } else {
                    let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                    output[0] = 1.0 - positive;
                    output[1] = positive;
                }
            }
            Forest::Multiclass(trees) => average_class_probabilities(trees, row, output),
        }
        Ok(())
    }

    /// Predicts row-major probabilities without allocating.
    pub(crate) fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let columns = self.classes.len();
        let expected = probability_output_len(data.rows(), columns)?;
        check_prediction_data(data, output.len(), expected, self.n_features_in)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if columns == 1 {
                    output.fill(1.0);
                    return Ok(());
                }
                output.fill(0.0);
                for tree in trees {
                    for (row, probabilities) in data.iter_rows().zip(output.chunks_exact_mut(2)) {
                        probabilities[1] += tree.predict(row);
                    }
                }
                for probabilities in output.chunks_exact_mut(2) {
                    let positive = (probabilities[1] / trees.len() as f32).clamp(0.0, 1.0);
                    probabilities[0] = 1.0 - positive;
                    probabilities[1] = positive;
                }
            }
            Forest::Multiclass(trees) => {
                // One tree across the whole batch, then the next, so each
                // tree's nodes and leaf block stay hot.
                output.fill(0.0);
                for tree in trees {
                    for (row, probabilities) in
                        data.iter_rows().zip(output.chunks_exact_mut(columns))
                    {
                        for (slot, &value) in probabilities.iter_mut().zip(tree.probabilities(row))
                        {
                            *slot += value;
                        }
                    }
                }
                let divisor = trees.len() as f32;
                for slot in output {
                    *slot = (*slot / divisor).clamp(0.0, 1.0);
                }
            }
        }
        Ok(())
    }

    /// Returns the requested fitted-class probability for one sample.
    pub(crate) fn predict_class_proba_one(
        &self,
        row: &[f32],
        class: u8,
    ) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        let class_index = self.class_index(class)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    return Ok(1.0);
                }
                let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                Ok(if class_index == 0 {
                    1.0 - positive
                } else {
                    positive
                })
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                average_class_probabilities(trees, row, probabilities);
                Ok(probabilities[class_index])
            }
        }
    }

    /// Predicts one fitted-class probability column without allocating.
    pub(crate) fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let class_index = self.class_index(class)?;
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        match &self.forest {
            Forest::Binary(_) => {
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
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                for (row, slot) in data.iter_rows().zip(output) {
                    average_class_probabilities(trees, row, probabilities);
                    *slot = probabilities[class_index];
                }
            }
        }
        Ok(())
    }

    fn accumulate_positive_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let Forest::Binary(trees) = &self.forest else {
            unreachable!("the scalar accumulation serves the binary fit only");
        };
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        output.fill(0.0);
        for tree in trees {
            for (row, slot) in data.iter_rows().zip(output.iter_mut()) {
                *slot += tree.predict(row);
            }
        }
        for slot in output {
            *slot = (*slot / trees.len() as f32).clamp(0.0, 1.0);
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
    /// Defined only for a binary fit. A multiclass fit has no positive class
    /// and reports [`ModelError::MulticlassOutput`] instead of returning one
    /// column of a vector that has no distinguished member.
    pub(crate) fn predict_positive_proba_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        let trees = self.require_binary_forest()?;
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(f32::from(self.classes[0]));
        }
        Ok(mean_tree_prediction(trees, row).clamp(0.0, 1.0))
    }

    /// Predicts every row's positive-class probability, allocating the output.
    pub(crate) fn predict_positive_proba(
        &self,
        data: &MatrixView<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        // Before the buffer, not inside `_into` after it, so an unusable
        // request costs no allocation.
        self.require_binary_forest()?;
        check_prediction_data(data, data.rows(), data.rows(), self.n_features_in)?;
        let mut output = vec![0.0; data.rows()];
        self.predict_positive_proba_into(data, &mut output)?;
        Ok(output)
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub(crate) fn predict_positive_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let trees = self.require_binary_forest()?;
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(f32::from(self.classes[0]));
            return Ok(());
        }
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = mean_tree_prediction(trees, data.row(index).expect("validated row index"))
                .clamp(0.0, 1.0);
        }
        Ok(())
    }

    fn require_binary_forest(&self) -> Result<&[PackedTree], ModelError> {
        match &self.forest {
            Forest::Binary(trees) => Ok(trees),
            Forest::Multiclass(_) => Err(ModelError::MulticlassOutput {
                columns: self.classes.len(),
            }),
        }
    }
}

/// The fitted state and prediction arithmetic of an ensemble regressor.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RegressorCore {
    pub(crate) n_features_in: usize,
    pub(crate) trees: Vec<PackedTree>,
}

impl RegressorCore {
    /// Predicts one regression value for one sample.
    pub(crate) fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        validate_prediction(mean_tree_prediction(&self.trees, row), 0)
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub(crate) fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = validate_prediction(
                mean_tree_prediction(&self.trees, data.row(index).expect("validated row index")),
                index,
            )?;
        }
        Ok(())
    }
}

pub(crate) fn mean_tree_prediction(trees: &[PackedTree], row: &[f32]) -> f32 {
    let sum: f32 = trees.iter().map(|tree| tree.predict(row)).sum();
    sum / trees.len() as f32
}

/// Soft averaging: the mean of the per-tree probability vectors.
///
/// This is not a vote. Every tree contributes its whole leaf distribution, so
/// the ensemble can produce values no count of `trees.len()` labels could —
/// which is exactly what distinguishes the two rules observably.
///
/// The accumulation is in the storage width under rule 3 of the accumulation
/// policy: the term count is the fitted tree count and every value is bounded
/// by one.
pub(crate) fn average_class_probabilities(trees: &[ClassTree], row: &[f32], output: &mut [f32]) {
    debug_assert!(
        trees
            .first()
            .is_none_or(|tree| tree.classes() == output.len())
    );
    output.fill(0.0);
    for tree in trees {
        for (slot, &value) in output.iter_mut().zip(tree.probabilities(row)) {
            *slot += value;
        }
    }
    let divisor = trees.len() as f32;
    for slot in output {
        *slot = (*slot / divisor).clamp(0.0, 1.0);
    }
}

/// Index of the largest value, with an exact tie going to the lowest index.
///
/// Class labels are sorted, so this is the smallest tied *label* — which is not
/// the same as the first class: with classes `[5, 9, 20]`, a tie between the
/// last two selects `9`.
pub(crate) fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

/// Whether averaging every tree can stay inside `f32`.
///
/// Prediction sums leaf values before dividing by the tree count, so the
/// bound is the sum of per-tree leaf magnitudes rather than their mean.
pub(crate) fn prediction_bound_is_finite(trees: &[PackedTree]) -> bool {
    let mut bound = 0.0_f64;
    for tree in trees {
        bound += f64::from(tree.max_abs_leaf());
        if !bound.is_finite() || bound > f64::from(f32::MAX) {
            return false;
        }
    }
    true
}

pub(crate) fn check_row(row: &[f32], expected: usize) -> Result<(), ModelError> {
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

pub(crate) fn check_prediction_data(
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

pub(crate) fn check_output_len(actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::OutputLength { expected, actual });
    }
    Ok(())
}

pub(crate) fn probability_output_len(rows: usize, classes: usize) -> Result<usize, ModelError> {
    rows.checked_mul(classes)
        .ok_or(ModelError::OutputShapeOverflow {
            rows,
            columns: classes,
        })
}

/// Validates shapes and parameters before any allocation or training work.
pub(crate) fn validate_common(
    data: &MatrixView<'_>,
    target_len: usize,
    sample_weights: Option<&SampleWeights>,
    config: &ForestConfig,
) -> Result<(), ModelError> {
    if data.rows() == 0 || data.columns() == 0 {
        return Err(ModelError::EmptyData);
    }
    if target_len != data.rows() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: target_len,
        });
    }
    if let Some(sample_weights) = sample_weights
        && data.rows() != sample_weights.len()
    {
        return Err(ModelError::SampleWeightLength {
            rows: data.rows(),
            weights: sample_weights.len(),
        });
    }
    if config.n_estimators == 0 {
        return Err(ModelError::InvalidEstimatorCount);
    }
    if config.grower.max_depth == Some(0) {
        return Err(ModelError::InvalidMaxDepth);
    }
    if config.grower.min_samples_split < 2 {
        return Err(ModelError::InvalidMinSamplesSplit);
    }
    if config.grower.min_samples_leaf == 0 {
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
    if let MaxFeatures::Count(requested) = config.grower.max_features
        && (requested == 0 || requested > data.columns())
    {
        return Err(ModelError::InvalidMaxFeatures {
            requested,
            available: data.columns(),
        });
    }
    // No finiteness scan of `data`, for the reason given in
    // `tree::validation::validate_fit`: a `MatrixView` is finite by
    // construction, so this could only re-derive the container's own invariant
    // at O(rows × columns) on every member's training matrix.
    Ok(())
}
