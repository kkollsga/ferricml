//! Deterministic dense histogram gradient-boosted regression.

use crate::api::{Estimator, HasParams, ModelError, Regressor};
use crate::boosting::binning::Binner;
use crate::boosting::grower::{GrowConfig, grow_tree};
use crate::boosting::predictor::CompactTree;
use crate::boosting::{BoostingError, MAX_BINS, MAX_TREE_DEPTH, MAX_TREE_LEAVES, MAX_TREES};
use crate::data::{MatrixView, RegressionTargets};

/// Parameters for [`HistGradientBoostingRegressor`].
#[derive(Clone, Debug, PartialEq)]
pub struct HistGradientBoostingRegressorParams {
    learning_rate: f32,
    max_iter: usize,
    max_leaf_nodes: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2_regularization: f32,
    max_bins: usize,
}

impl Default for HistGradientBoostingRegressorParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iter: 100,
            max_leaf_nodes: 31,
            max_depth: None,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            max_bins: 255,
        }
    }
}

impl HistGradientBoostingRegressorParams {
    /// Sets the shrinkage applied to each fitted tree.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the number of boosting iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the maximum number of leaves in each tree.
    #[must_use]
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: usize) -> Self {
        self.max_leaf_nodes = max_leaf_nodes;
        self
    }

    /// Sets the maximum depth of each tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum number of training rows in each leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Sets the non-negative L2 term in each leaf denominator.
    #[must_use]
    pub fn with_l2_regularization(mut self, l2_regularization: f32) -> Self {
        self.l2_regularization = l2_regularization;
        self
    }

    /// Sets the maximum number of histogram bins per feature.
    #[must_use]
    pub fn with_max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = max_bins;
        self
    }

    /// Returns the per-tree shrinkage.
    pub const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the requested number of boosting iterations.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the per-tree leaf-count limit.
    pub const fn max_leaf_nodes(&self) -> usize {
        self.max_leaf_nodes
    }

    /// Returns the per-tree depth limit.
    pub const fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Returns the minimum number of training rows in each leaf.
    pub const fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Returns the L2 leaf-denominator term.
    pub const fn l2_regularization(&self) -> f32 {
        self.l2_regularization
    }

    /// Returns the maximum histogram bin count.
    pub const fn max_bins(&self) -> usize {
        self.max_bins
    }
}

/// Serial squared-error histogram gradient-boosted regressor.
#[derive(Clone, Debug, PartialEq)]
pub struct HistGradientBoostingRegressor {
    n_features_in: usize,
    params: HistGradientBoostingRegressorParams,
    baseline: f32,
    trees: Vec<CompactTree>,
}

impl HistGradientBoostingRegressor {
    /// Fits a deterministic dense squared-error boosted ensemble.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: HistGradientBoostingRegressorParams,
    ) -> Result<Self, ModelError> {
        validate_fit(data, targets, &params)?;
        let binner = Binner::fit(data, params.max_bins).map_err(map_boosting_error)?;
        let binned = binner.transform(data).map_err(map_boosting_error)?;
        let baseline = (targets
            .as_slice()
            .iter()
            .map(|&target| f64::from(target))
            .sum::<f64>()
            / targets.len() as f64) as f32;
        if !baseline.is_finite() {
            return Err(ModelError::NumericalOverflow);
        }
        let mut predictions = vec![baseline; data.rows()];
        let mut residuals = compute_residuals(targets.as_slice(), &predictions)?;
        let mut trees = Vec::with_capacity(params.max_iter);
        let config = GrowConfig {
            max_leaf_nodes: params.max_leaf_nodes,
            max_depth: params.max_depth,
            min_samples_leaf: params.min_samples_leaf,
            l2_regularization: params.l2_regularization,
        };
        for _ in 0..params.max_iter {
            let tree =
                grow_tree(&binned, &binner, &residuals, config).map_err(map_boosting_error)?;
            tree.add_predictions(data, params.learning_rate, &mut predictions);
            if predictions.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NumericalOverflow);
            }
            residuals = compute_residuals(targets.as_slice(), &predictions)?;
            trees.push(tree);
        }
        Ok(Self {
            n_features_in: data.columns(),
            params,
            baseline,
            trees,
        })
    }

    /// Returns the fitted mean baseline.
    pub const fn baseline(&self) -> f32 {
        self.baseline
    }

    /// Returns the number of fitted boosting iterations.
    pub fn n_iter(&self) -> usize {
        self.trees.len()
    }

    /// Returns the fitted input width.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fitted parameters.
    pub const fn get_params(&self) -> &HistGradientBoostingRegressorParams {
        &self.params
    }

    /// Predicts one regression value.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        if row.len() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: row.len(),
            });
        }
        let prediction = self.trees.iter().fold(self.baseline, |prediction, tree| {
            prediction + self.params.learning_rate * tree.predict_one(row)
        });
        if !prediction.is_finite() {
            return Err(ModelError::NonFinitePrediction { row: 0 });
        }
        Ok(prediction)
    }

    /// Predicts one value per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predicts one value per row into caller-owned storage.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Regressor>::predict_into(self, data, output)
    }

    #[allow(dead_code)] // Used by the stable logical-tree codec in the next phase.
    pub(crate) fn trees(&self) -> &[CompactTree] {
        &self.trees
    }
}

impl Estimator for HistGradientBoostingRegressor {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for HistGradientBoostingRegressor {
    type Params = HistGradientBoostingRegressorParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Regressor for HistGradientBoostingRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        if data.columns() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: data.columns(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        for (row_index, row) in data.iter_rows().enumerate() {
            let prediction = self.trees.iter().fold(self.baseline, |prediction, tree| {
                prediction + self.params.learning_rate * tree.predict_one(row)
            });
            if !prediction.is_finite() {
                return Err(ModelError::NonFinitePrediction { row: row_index });
            }
        }
        for (row, slot) in data.iter_rows().zip(output) {
            *slot = self.trees.iter().fold(self.baseline, |prediction, tree| {
                prediction + self.params.learning_rate * tree.predict_one(row)
            });
        }
        Ok(())
    }
}

fn compute_residuals(targets: &[f32], predictions: &[f32]) -> Result<Vec<f32>, ModelError> {
    let mut residuals = Vec::with_capacity(targets.len());
    for (&target, &prediction) in targets.iter().zip(predictions) {
        let residual = (f64::from(target) - f64::from(prediction)) as f32;
        if !residual.is_finite() {
            return Err(ModelError::NumericalOverflow);
        }
        residuals.push(residual);
    }
    Ok(residuals)
}

fn validate_fit(
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    params: &HistGradientBoostingRegressorParams,
) -> Result<(), ModelError> {
    if data.rows() != targets.len() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: targets.len(),
        });
    }
    if !params.learning_rate.is_finite() || params.learning_rate <= 0.0 {
        return Err(ModelError::InvalidLearningRate);
    }
    if !(1..=MAX_TREES).contains(&params.max_iter) {
        return Err(ModelError::InvalidBoostingIterationCount);
    }
    if !(2..=MAX_TREE_LEAVES).contains(&params.max_leaf_nodes) {
        return Err(ModelError::InvalidMaxLeafNodes);
    }
    if params
        .max_depth
        .is_some_and(|max_depth| !(1..=MAX_TREE_DEPTH).contains(&max_depth))
    {
        return Err(ModelError::InvalidBoostingMaxDepth);
    }
    if params.min_samples_leaf == 0 {
        return Err(ModelError::InvalidMinSamplesLeaf);
    }
    if !params.l2_regularization.is_finite() || params.l2_regularization < 0.0 {
        return Err(ModelError::InvalidL2Regularization);
    }
    if !(2..=MAX_BINS).contains(&params.max_bins) {
        return Err(ModelError::InvalidMaxBins);
    }
    Ok(())
}

fn map_boosting_error(error: BoostingError) -> ModelError {
    match error {
        BoostingError::InvalidMaxBins => ModelError::InvalidMaxBins,
        BoostingError::InvalidMaxLeafNodes => ModelError::InvalidMaxLeafNodes,
        BoostingError::InvalidMaxDepth => ModelError::InvalidBoostingMaxDepth,
        BoostingError::InvalidMinSamplesLeaf => ModelError::InvalidMinSamplesLeaf,
        BoostingError::InvalidL2Regularization => ModelError::InvalidL2Regularization,
        BoostingError::FeatureDimension { expected, actual } => {
            ModelError::FeatureDimension { expected, actual }
        }
        BoostingError::TooManyFeatures => ModelError::TooManyFeatures,
        BoostingError::TreeTooLarge | BoostingError::InvalidTree => ModelError::TreeTooLarge,
        BoostingError::ResidualLength { .. } | BoostingError::NonFiniteResidual { .. } => {
            ModelError::NumericalOverflow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn piecewise() -> (DenseMatrix, RegressionTargets) {
        (
            DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap(),
            RegressionTargets::new(vec![0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]).unwrap(),
        )
    }

    fn one_tree_params() -> HistGradientBoostingRegressorParams {
        HistGradientBoostingRegressorParams::default()
            .with_learning_rate(1.0)
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1)
    }

    #[test]
    fn one_boosting_update_matches_baseline_residual_and_leaf_values() {
        let (data, targets) = piecewise();
        let model =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, one_tree_params())
                .unwrap();
        assert_eq!(model.baseline(), 2.0);
        assert_eq!(model.n_iter(), 1);
        assert_eq!(model.predict(&data.as_view()).unwrap(), targets.as_slice());
    }

    #[test]
    fn constant_targets_and_features_are_deterministic() {
        let data = DenseMatrix::new(vec![1.0; 12], 6, 2).unwrap();
        let targets = RegressionTargets::new(vec![3.5; 6]).unwrap();
        let params = HistGradientBoostingRegressorParams::default()
            .with_max_iter(4)
            .with_min_samples_leaf(1);
        let first =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, params.clone()).unwrap();
        let second = HistGradientBoostingRegressor::fit(&data.as_view(), &targets, params).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.predict(&data.as_view()).unwrap(), vec![3.5; 6]);
    }

    #[test]
    fn scalar_batch_and_into_predictions_agree() {
        let (data, targets) = piecewise();
        let model = HistGradientBoostingRegressor::fit(
            &data.as_view(),
            &targets,
            HistGradientBoostingRegressorParams::default()
                .with_max_iter(4)
                .with_max_leaf_nodes(3)
                .with_min_samples_leaf(1),
        )
        .unwrap();
        let allocating = model.predict(&data.as_view()).unwrap();
        let mut output = vec![0.0; data.rows()];
        model.predict_into(&data.as_view(), &mut output).unwrap();
        assert_eq!(allocating, output);
        for (row, &expected) in data.iter_rows().zip(&output) {
            assert_eq!(model.predict_one(row).unwrap(), expected);
        }
    }

    #[test]
    fn validates_every_parameter_and_output_before_writes() {
        let (data, targets) = piecewise();
        let cases = [
            (
                HistGradientBoostingRegressorParams::default().with_learning_rate(0.0),
                ModelError::InvalidLearningRate,
            ),
            (
                HistGradientBoostingRegressorParams::default().with_max_iter(0),
                ModelError::InvalidBoostingIterationCount,
            ),
            (
                HistGradientBoostingRegressorParams::default().with_max_leaf_nodes(1),
                ModelError::InvalidMaxLeafNodes,
            ),
            (
                HistGradientBoostingRegressorParams::default().with_max_depth(Some(0)),
                ModelError::InvalidBoostingMaxDepth,
            ),
            (
                HistGradientBoostingRegressorParams::default().with_min_samples_leaf(0),
                ModelError::InvalidMinSamplesLeaf,
            ),
            (
                HistGradientBoostingRegressorParams::default().with_l2_regularization(-1.0),
                ModelError::InvalidL2Regularization,
            ),
            (
                HistGradientBoostingRegressorParams::default().with_max_bins(1),
                ModelError::InvalidMaxBins,
            ),
        ];
        for (params, expected) in cases {
            assert_eq!(
                HistGradientBoostingRegressor::fit(&data.as_view(), &targets, params),
                Err(expected)
            );
        }

        let model =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, one_tree_params())
                .unwrap();
        let mut output = [91.0; 7];
        assert_eq!(
            model.predict_into(&data.as_view(), &mut output),
            Err(ModelError::OutputLength {
                expected: 8,
                actual: 7
            })
        );
        assert_eq!(output, [91.0; 7]);
    }

    #[test]
    fn finite_extremes_report_residual_overflow() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0], 3, 1).unwrap();
        let targets = RegressionTargets::new(vec![-f32::MAX, -f32::MAX, f32::MAX]).unwrap();
        assert_eq!(
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, one_tree_params()),
            Err(ModelError::NumericalOverflow)
        );
    }
}
