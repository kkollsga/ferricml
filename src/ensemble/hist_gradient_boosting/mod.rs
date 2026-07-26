//! Deterministic dense histogram gradient-boosted regression and binary
//! classification.
//!
//! Both estimators share one binner, one grower, and one set of growth
//! controls, and differ only in the objective they descend: squared error for
//! the regressor, binary log loss for the classifier. The grower is generic over
//! that objective, so neither estimator contains a copy of the other's search.

use self::binning::Binner;
use self::controls::{
    BoostingControls, map_boosting_error, validate_control_bounds, validate_controls,
};
use self::error::{MAX_TOTAL_NODES, MAX_TREES};
use self::grower::{SampleStatistics, grow_tree};
use self::predictor::{CompactTree, prediction_bound_is_finite};
use crate::api::{
    Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor,
    validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND,
    MIN_ENCODED_TREE_BYTES, SchemaRole, decode_component, decode_logical_tree, decode_v2_envelope,
    encode_component, encode_logical_tree, encode_v2_envelope,
};
use crate::data::{MatrixView, RegressionTargets, SampleWeights};
use crate::loss::{BoostingObjective, Objective, SquaredError};
use crate::numeric::sum_in_order;

const ARTIFACT_PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;

/// The objective this estimator's artifacts are fitted against.
///
/// Read from the objective type rather than restated, so the field can never
/// name a loss the model was not fitted with. The estimator kind is what stops
/// a regressor's artifact from decoding as some other model; this is what stops
/// it from decoding as a *different objective* under the same kind, which is
/// the only distinction a kind cannot make.
const OBJECTIVE_VERSION: u32 = <SquaredError as BoostingObjective>::ARTIFACT_OBJECTIVE_TAG;

mod binning;
mod classifier;
mod controls;
mod error;
mod grower;
mod predictor;

pub use classifier::{HistGradientBoostingClassifier, HistGradientBoostingClassifierParams};

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

    /// These parameters as the growth controls shared with the classifier.
    const fn controls(&self) -> BoostingControls {
        BoostingControls {
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            max_leaf_nodes: self.max_leaf_nodes,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
            max_bins: self.max_bins,
        }
    }
}

/// Serial squared-error histogram gradient-boosted regressor.
///
/// Where a forest averages many independent trees, boosting grows them in
/// sequence: each iteration fits a tree to what the previous ones got wrong and
/// adds a shrunk share of it. More iterations means a closer fit, which is why
/// `learning_rate` and `max_iter` trade against each other.
///
/// Note the row count below. `min_samples_leaf` defaults to `20`, so a node
/// cannot split unless both sides keep at least that many rows — on a dataset
/// smaller than twice that, no split is ever admissible and the fit succeeds
/// with the baseline alone. That is the intended meaning of the parameter, not
/// a failure, but it is why a toy-sized boosting example predicts a constant.
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::ensemble::{HistGradientBoostingRegressor, HistGradientBoostingRegressorParams};
/// use ferricml::metrics::mean_squared_error;
///
/// // 64 rows, so the default `min_samples_leaf` of 20 admits a split.
/// let values: Vec<f32> = (0..64).map(|index| index as f32).collect();
/// let squares: Vec<f32> = values.iter().map(|value| value * value).collect();
/// let data = DenseMatrix::new(values, 64, 1)?;
/// let targets = RegressionTargets::new(squares)?;
///
/// let brief = HistGradientBoostingRegressor::fit(
///     &data.as_view(),
///     &targets,
///     HistGradientBoostingRegressorParams::default().with_max_iter(2),
/// )?;
/// let longer = HistGradientBoostingRegressor::fit(
///     &data.as_view(),
///     &targets,
///     HistGradientBoostingRegressorParams::default().with_max_iter(60),
/// )?;
///
/// // More boosting iterations fit the training data more closely.
/// let brief_error = mean_squared_error(
///     targets.as_slice(),
///     &brief.predict(&data.as_view())?,
/// )?;
/// let longer_error = mean_squared_error(
///     targets.as_slice(),
///     &longer.predict(&data.as_view())?,
/// )?;
/// assert!(longer_error < brief_error);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits with per-row sample weights.
    ///
    /// A weight scales the row's gradient and its share of every node's weight
    /// total, so the baseline is a weighted mean and the minimum leaf size
    /// counts weight rather than rows. Weights of exactly one reproduce
    /// [`Self::fit`] bit for bit, and an integer weight is the same fit as
    /// repeating that row that many times.
    ///
    /// The bin grid is deliberately **not** weighted. It is fitted from the
    /// distinct observed feature values, which neither a weight nor a repeated
    /// row changes — which is precisely why repeating a row and weighting it
    /// agree.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: HistGradientBoostingRegressorParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: HistGradientBoostingRegressorParams,
    ) -> Result<Self, ModelError> {
        validate_fit(data, targets, sample_weights, &params)?;
        let weights = sample_weights.map(SampleWeights::as_slice);
        let binner = Binner::fit(data, params.max_bins).map_err(map_boosting_error)?;
        let binned = binner.transform(data).map_err(map_boosting_error)?;
        let baseline = match sample_weights {
            None => {
                (sum_in_order(targets.as_slice().iter().map(|&target| f64::from(target)))
                    / targets.len() as f64) as f32
            }
            Some(sample_weights) => {
                (sum_in_order(
                    targets
                        .as_slice()
                        .iter()
                        .zip(sample_weights.as_slice())
                        .map(|(&target, &weight)| f64::from(weight) * f64::from(target)),
                ) / sample_weights.total()) as f32
            }
        };
        if !baseline.is_finite() {
            return Err(ModelError::NumericalOverflow);
        }
        let mut predictions = vec![baseline; data.rows()];
        let mut residuals = compute_residuals(targets.as_slice(), &predictions)?;
        let mut trees = Vec::with_capacity(params.max_iter);
        let config = params.controls().grow_config();
        for _ in 0..params.max_iter {
            let tree = grow_tree::<SquaredError>(
                &binned,
                &binner,
                SampleStatistics {
                    negative_gradients: &residuals,
                    hessians: &[],
                    sample_weights: weights,
                },
                config,
            )
            .map_err(map_boosting_error)?;
            tree.add_predictions(data, params.learning_rate, &mut predictions);
            if predictions.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NumericalOverflow);
            }
            residuals = compute_residuals(targets.as_slice(), &predictions)?;
            trees.push(tree);
        }
        if !prediction_bound_is_finite(baseline, params.learning_rate, &trees) {
            return Err(ModelError::NumericalOverflow);
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
        validate_scalar_row(row, self.n_features_in)?;
        validate_prediction(self.predict_value(row), 0)
    }

    fn predict_value(&self, row: &[f32]) -> f32 {
        self.trees.iter().fold(self.baseline, |prediction, tree| {
            prediction + self.params.learning_rate * tree.predict_one(row)
        })
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

    /// Encodes the fitted baseline, parameters, and canonical logical trees.
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_leaf_nodes =
            u32::try_from(self.params.max_leaf_nodes).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_depth = self
            .params
            .max_depth
            .map(u32::try_from)
            .transpose()
            .map_err(|_| ArtifactError::InvalidPayload)?
            .unwrap_or(0);
        let min_samples_leaf = u32::try_from(self.params.min_samples_leaf)
            .map_err(|_| ArtifactError::InvalidPayload)?;
        let max_bins =
            u32::try_from(self.params.max_bins).map_err(|_| ArtifactError::InvalidPayload)?;
        let tree_count =
            u32::try_from(self.trees.len()).map_err(|_| ArtifactError::InvalidPayload)?;
        let total_nodes = self.trees.iter().try_fold(0_usize, |total, tree| {
            total
                .checked_add(tree.nodes().len())
                .ok_or(ArtifactError::InvalidPayload)
        })?;
        let total_nodes = u32::try_from(total_nodes).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut metadata = ArtifactPayloadWriter::with_capacity(12 * 4);
        metadata.u32(OBJECTIVE_VERSION);
        metadata.u32(n_features);
        metadata.f32(self.params.learning_rate);
        metadata.u32(max_iter);
        metadata.u32(max_leaf_nodes);
        metadata.u32(max_depth);
        metadata.u32(min_samples_leaf);
        metadata.f32(self.params.l2_regularization);
        metadata.u32(max_bins);
        metadata.f32(self.baseline);
        metadata.u32(tree_count);
        metadata.u32(total_nodes);
        let metadata = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        let mut payload = metadata;
        for tree in &self.trees {
            payload.extend_from_slice(&encode_component(
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
                &encode_logical_tree(&tree.to_logical_nodes())?,
            )?);
        }
        encode_v2_envelope(
            HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND,
            ARTIFACT_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes and validates logical trees before building runtime state.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND,
            ARTIFACT_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let objective_version = metadata.u32()?;
        let n_features_in = metadata.u32()? as usize;
        let learning_rate = metadata.f32()?;
        let max_iter = metadata.u32()? as usize;
        let max_leaf_nodes = metadata.u32()? as usize;
        let encoded_depth = metadata.u32()? as usize;
        let min_samples_leaf = metadata.u32()? as usize;
        let l2_regularization = metadata.f32()?;
        let max_bins = metadata.u32()? as usize;
        let baseline = metadata.f32()?;
        let tree_count = metadata.u32()? as usize;
        let declared_total_nodes = metadata.u32()? as usize;
        let params = HistGradientBoostingRegressorParams {
            learning_rate,
            max_iter,
            max_leaf_nodes,
            max_depth: (encoded_depth != 0).then_some(encoded_depth),
            min_samples_leaf,
            l2_regularization,
            max_bins,
        };
        if !metadata.is_empty()
            || objective_version != OBJECTIVE_VERSION
            || n_features_in == 0
            || n_features_in > 1_000_000
            || !baseline.is_finite()
            || tree_count == 0
            || tree_count > MAX_TREES
            || tree_count != max_iter
            || declared_total_nodes == 0
            || declared_total_nodes > MAX_TOTAL_NODES
            || validate_controls(params.controls()).is_err()
            || validate_control_bounds(params.controls()).is_err()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut trees =
            Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
        let mut actual_total_nodes = 0_usize;
        for _ in 0..tree_count {
            let logical_nodes = decode_logical_tree(decode_component(
                &mut envelope,
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
            )?)?;
            let tree = CompactTree::from_logical_nodes(logical_nodes, n_features_in)?;
            actual_total_nodes = actual_total_nodes
                .checked_add(tree.nodes().len())
                .ok_or(ArtifactError::InvalidPayload)?;
            if actual_total_nodes > declared_total_nodes || actual_total_nodes > MAX_TOTAL_NODES {
                return Err(ArtifactError::InvalidPayload);
            }
            trees.push(tree);
        }
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        if actual_total_nodes != declared_total_nodes {
            return Err(ArtifactError::InvalidPayload);
        }
        if !prediction_bound_is_finite(baseline, params.learning_rate, &trees) {
            return Err(ArtifactError::InvalidPayload);
        }
        Ok(Self {
            n_features_in,
            params,
            baseline,
            trees,
        })
    }
}

impl Estimator for HistGradientBoostingRegressor {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for HistGradientBoostingRegressor {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
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
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            *slot = validate_prediction(self.predict_value(row), row_index)?;
        }
        Ok(())
    }
}

/// Negative gradients of the fitted objective at the current predictions.
///
/// For squared error these are the residuals `target - prediction`, taken from
/// the objective rather than restated here so the next tree always descends the
/// loss the model claims to minimize.
fn compute_residuals(targets: &[f32], predictions: &[f32]) -> Result<Vec<f32>, ModelError> {
    let mut residuals = Vec::with_capacity(targets.len());
    for (&target, &prediction) in targets.iter().zip(predictions) {
        let residual =
            SquaredError::negative_gradient(f64::from(prediction), f64::from(target)) as f32;
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
    sample_weights: Option<&SampleWeights>,
    params: &HistGradientBoostingRegressorParams,
) -> Result<(), ModelError> {
    if data.rows() != targets.len() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: targets.len(),
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
    validate_controls(params.controls())?;
    validate_control_bounds(params.controls())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::error::MAX_TREE_LEAVES;
    use super::*;
    use crate::data::DenseMatrix;
    use crate::linear_model::Ridge;
    use sha2::{Digest, Sha256};

    fn resign_artifact(bytes: &mut [u8]) {
        let checksum_offset = bytes.len() - 32;
        let checksum = Sha256::digest(&bytes[..checksum_offset]);
        bytes[checksum_offset..].copy_from_slice(&checksum);
    }

    fn piecewise() -> (DenseMatrix, RegressionTargets) {
        (
            DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap(),
            RegressionTargets::new(vec![0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]).unwrap(),
        )
    }

    /// Eight rows of distinct integer features and integer targets, so every
    /// weighted accumulation over the fixture is exact and the equivalences
    /// below can be asserted on artifact bytes rather than a tolerance.
    fn weight_fixture() -> (DenseMatrix, Vec<f32>) {
        (
            DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap(),
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 7.0, 6.0, 5.0],
        )
    }

    fn weight_params() -> HistGradientBoostingRegressorParams {
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(4)
            .with_max_leaf_nodes(4)
            .with_min_samples_leaf(1)
            .with_max_bins(8)
    }

    #[test]
    fn unit_weights_reproduce_the_unweighted_fit_bit_for_bit() {
        let (data, values) = weight_fixture();
        let targets = RegressionTargets::new(values).unwrap();
        let ones = SampleWeights::new(vec![1.0; data.rows()]).unwrap();
        let plain =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, weight_params()).unwrap();
        let weighted = HistGradientBoostingRegressor::fit_weighted(
            &data.as_view(),
            &targets,
            &ones,
            weight_params(),
        )
        .unwrap();
        assert_eq!(plain, weighted);
        assert_eq!(
            plain.to_artifact([5; 32]).unwrap(),
            weighted.to_artifact([5; 32]).unwrap()
        );
    }

    /// An integer weight is the same fit as repeating the row that many times.
    ///
    /// The bin grid survives the comparison because it is fitted from the
    /// *distinct* observed values, which a repeated row does not change.
    #[test]
    fn an_integer_weight_is_the_same_fit_as_repeating_the_row() {
        let (data, values) = weight_fixture();
        let repeat = 5;
        let times = 3;
        let mut weights = vec![1.0_f32; values.len()];
        weights[repeat] = times as f32;
        let weighted = HistGradientBoostingRegressor::fit_weighted(
            &data.as_view(),
            &RegressionTargets::new(values.clone()).unwrap(),
            &SampleWeights::new(weights).unwrap(),
            weight_params(),
        )
        .unwrap();

        let mut rows = Vec::new();
        let mut repeated_values = Vec::new();
        for (row, &value) in values.iter().enumerate() {
            let copies = if row == repeat { times } else { 1 };
            for _ in 0..copies {
                rows.extend_from_slice(data.as_view().row(row).unwrap());
                repeated_values.push(value);
            }
        }
        let repeated_data = DenseMatrix::new(rows, repeated_values.len(), 1).unwrap();
        let repeated = HistGradientBoostingRegressor::fit(
            &repeated_data.as_view(),
            &RegressionTargets::new(repeated_values).unwrap(),
            weight_params(),
        )
        .unwrap();

        assert_eq!(weighted.baseline(), repeated.baseline());
        assert_eq!(
            weighted.to_artifact([5; 32]).unwrap(),
            repeated.to_artifact([5; 32]).unwrap()
        );
    }

    #[test]
    fn weighted_fitting_rejects_a_length_mismatch_and_is_not_inert() {
        let (data, values) = weight_fixture();
        let targets = RegressionTargets::new(values).unwrap();
        assert_eq!(
            HistGradientBoostingRegressor::fit_weighted(
                &data.as_view(),
                &targets,
                &SampleWeights::new(vec![1.0; data.rows() - 1]).unwrap(),
                weight_params(),
            )
            .unwrap_err(),
            ModelError::SampleWeightLength {
                rows: data.rows(),
                weights: data.rows() - 1,
            }
        );

        let mut skewed = vec![1.0_f32; data.rows()];
        skewed[0] = 9.0;
        let weighted = HistGradientBoostingRegressor::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(skewed).unwrap(),
            weight_params(),
        )
        .unwrap();
        let plain =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, weight_params()).unwrap();
        assert_ne!(plain.baseline(), weighted.baseline());
    }

    fn one_tree_params() -> HistGradientBoostingRegressorParams {
        HistGradientBoostingRegressorParams::default()
            .with_learning_rate(1.0)
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1)
    }

    #[test]
    fn an_all_negative_zero_target_keeps_its_negatively_signed_baseline() {
        // The baseline is a reduction over the targets, and IEEE addition's
        // identity is `-0.0`, not `+0.0`. A reduction seeded with the wrong
        // one returns `0.0` here, which is a different artifact byte pattern
        // for a model that is otherwise unchanged.
        let data = DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap();
        for (targets, expected) in [
            (vec![-0.0_f32; 8], (-0.0_f32).to_bits()),
            (vec![0.0_f32; 8], 0.0_f32.to_bits()),
            (
                vec![0.0_f32, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
                0.0_f32.to_bits(),
            ),
        ] {
            let targets = RegressionTargets::new(targets).unwrap();
            let model =
                HistGradientBoostingRegressor::fit(&data.as_view(), &targets, one_tree_params())
                    .unwrap();
            assert_eq!(model.baseline().to_bits(), expected);
        }
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
    fn a_dataset_too_small_for_the_leaf_bound_fits_the_baseline_alone() {
        // `min_samples_leaf` defaults to 20, so a node can only split when both
        // sides keep at least that many rows. Below twice that, no split is
        // admissible: fitting succeeds and returns a model that is its baseline
        // and nothing else, however many iterations were requested.
        //
        // This is the parameter working, not failing — but it is worth pinning,
        // because the surprising part is that it is silent. There is no error
        // and no warning distinguishing "boosting had nothing to add" from
        // "boosting was never able to start", and someone trying the estimator
        // on a small dataset sees a constant prediction with no explanation.
        let data = DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap();
        let targets =
            RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]).unwrap();

        let brief = HistGradientBoostingRegressor::fit(
            &data.as_view(),
            &targets,
            HistGradientBoostingRegressorParams::default().with_max_iter(2),
        )
        .unwrap();
        let longer = HistGradientBoostingRegressor::fit(
            &data.as_view(),
            &targets,
            HistGradientBoostingRegressorParams::default().with_max_iter(200),
        )
        .unwrap();

        let baseline = brief.baseline();
        assert_eq!(brief.predict(&data.as_view()).unwrap(), vec![baseline; 8]);
        // A hundred times the iteration budget buys exactly nothing, because
        // every iteration produced a root-only tree.
        assert_eq!(
            longer.predict(&data.as_view()).unwrap(),
            brief.predict(&data.as_view()).unwrap()
        );

        // Lowering the bound is what unblocks it, and is the fix a caller wants.
        let split = HistGradientBoostingRegressor::fit(
            &data.as_view(),
            &targets,
            HistGradientBoostingRegressorParams::default()
                .with_max_iter(2)
                .with_min_samples_leaf(1),
        )
        .unwrap();
        assert_ne!(split.predict(&data.as_view()).unwrap(), vec![baseline; 8]);
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

    #[test]
    fn artifact_round_trip_is_deterministic_schema_bound_and_kind_isolated() {
        let (data, targets) = piecewise();
        let model =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, one_tree_params())
                .unwrap();
        let schema = [23; 32];
        let left = model.to_artifact(schema).unwrap();
        let right = model.to_artifact(schema).unwrap();
        assert_eq!(left, right);

        let decoded = HistGradientBoostingRegressor::from_artifact(&left, schema).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(
            decoded.predict(&data.as_view()).unwrap(),
            model.predict(&data.as_view()).unwrap()
        );
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&left, [24; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        assert_eq!(
            Ridge::from_artifact(&left, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind {
                found: HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND,
            }
        );

        let mut corrupted = left.clone();
        corrupted[140] ^= 1;
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }

    #[test]
    fn artifact_rejects_invalid_metadata_tree_records_and_framing() {
        let (data, targets) = piecewise();
        let model =
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, one_tree_params())
                .unwrap();
        let schema = [31; 32];
        let bytes = model.to_artifact(schema).unwrap();

        let mut objective = bytes.clone();
        objective[68..72].copy_from_slice(&2_u32.to_le_bytes());
        resign_artifact(&mut objective);
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&objective, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut total_nodes = bytes.clone();
        total_nodes[112..116].copy_from_slice(&4_u32.to_le_bytes());
        resign_artifact(&mut total_nodes);
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&total_nodes, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut component_kind = bytes.clone();
        component_kind[116..118].copy_from_slice(&3_u16.to_le_bytes());
        resign_artifact(&mut component_kind);
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&component_kind, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut feature = bytes.clone();
        feature[140..144].copy_from_slice(&1_u32.to_le_bytes());
        resign_artifact(&mut feature);
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&feature, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&bytes[..120], schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }

    #[test]
    fn aggregate_model_bound_is_checked_before_training() {
        let (data, targets) = piecewise();
        let params = HistGradientBoostingRegressorParams::default()
            .with_max_iter(MAX_TREES)
            .with_max_leaf_nodes(MAX_TREE_LEAVES);
        assert_eq!(
            HistGradientBoostingRegressor::fit(&data.as_view(), &targets, params),
            Err(ModelError::BoostingModelTooLarge)
        );
    }
}
