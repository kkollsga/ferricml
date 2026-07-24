//! Dense ordinary least-squares regression.

use super::least_squares;
use crate::api::{
    Estimator, HasParams, ModelError, Regressor, validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, LINEAR_REGRESSION_ARTIFACT_KIND,
    MODEL_ARTIFACT_VERSION, SchemaRole, artifact_version, decode_component, decode_v2_envelope,
    encode_component, encode_v2_envelope,
};
use crate::data::{MatrixView, RegressionTargets, SampleWeights};

const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const LINEAR_PAYLOAD_VERSION: u16 = 1;
const LINEAR_STATE_COMPONENT_KIND: u16 = 1;
const LINEAR_STATE_COMPONENT_VERSION: u16 = 1;
const LINEAR_FIXED_PAYLOAD_BYTES: usize = 6 * 4;

/// Parameters for [`LinearRegression`].
#[derive(Clone, Debug, PartialEq)]
pub struct LinearRegressionParams {
    fit_intercept: bool,
    tol: f32,
}

impl Default for LinearRegressionParams {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            tol: 1.0e-6,
        }
    }
}

impl LinearRegressionParams {
    /// Enables or disables the fitted intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Sets the relative singular-value cutoff used to determine rank.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns whether an intercept is fitted.
    pub const fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Returns the relative singular-value cutoff.
    pub const fn tol(&self) -> f32 {
        self.tol
    }
}

/// Dense single-target ordinary least-squares regression.
///
/// Fitting uses a deterministic SVD and returns the minimum-norm solution for
/// rank-deficient or underdetermined inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearRegression {
    n_features_in: usize,
    params: LinearRegressionParams,
    coefficients: Vec<f32>,
    intercept: f32,
    rank: usize,
}

impl LinearRegression {
    /// Fits an unweighted dense least-squares model.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: LinearRegressionParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits a dense least-squares model with per-row sample weights.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: LinearRegressionParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: LinearRegressionParams,
    ) -> Result<Self, ModelError> {
        validate_fit(data, targets, sample_weights, &params)?;
        let fit = least_squares::fit_dense(
            data,
            targets.as_slice(),
            sample_weights,
            params.fit_intercept,
            params.tol,
        )?;
        let coefficients = fit
            .coefficients
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let intercept = fit.intercept as f32;
        if coefficients.iter().any(|value| !value.is_finite()) || !intercept.is_finite() {
            return Err(ModelError::LinearSolveFailed);
        }
        Ok(Self {
            n_features_in: data.columns(),
            params,
            coefficients,
            intercept,
            rank: fit.rank,
        })
    }

    /// Returns fitted coefficients in input-feature order.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Returns the fitted intercept.
    pub const fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns the effective rank of the weighted, centered design matrix.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fit parameters.
    pub const fn get_params(&self) -> &LinearRegressionParams {
        &self.params
    }

    /// Predicts one regression value.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        validate_scalar_row(row, self.n_features_in)?;
        validate_prediction(self.predict_value(row), 0)
    }

    fn predict_value(&self, row: &[f32]) -> f32 {
        row.iter()
            .zip(&self.coefficients)
            .fold(self.intercept, |sum, (&value, &coefficient)| {
                sum + value * coefficient
            })
    }

    /// Allocating batch prediction.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Allocation-free batch prediction.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Regressor>::predict_into(self, data, output)
    }

    /// Encodes this model in FerricML's bounded artifact format.
    pub fn to_artifact(&self, feature_schema_sha256: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        if self.n_features_in > MAX_ARTIFACT_FEATURES {
            return Err(ArtifactError::InvalidPayload);
        }
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let rank = u32::try_from(self.rank).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut state = ArtifactPayloadWriter::with_capacity(
            LINEAR_FIXED_PAYLOAD_BYTES + self.coefficients.len() * 4,
        );
        state.u32(n_features);
        state.u32(u32::from(self.params.fit_intercept));
        state.f32(self.params.tol);
        state.u32(rank);
        state.f32(self.intercept);
        state.u32(n_features);
        for &coefficient in &self.coefficients {
            state.f32(coefficient);
        }
        let component = encode_component(
            LINEAR_STATE_COMPONENT_KIND,
            LINEAR_STATE_COMPONENT_VERSION,
            &state.finish(),
        )?;
        encode_v2_envelope(
            LINEAR_REGRESSION_ARTIFACT_KIND,
            LINEAR_PAYLOAD_VERSION,
            &[(SchemaRole::Input, feature_schema_sha256)],
            &component,
        )
    }

    /// Decodes a linear model after checking integrity and feature identity.
    pub fn from_artifact(
        bytes: &[u8],
        expected_feature_schema_sha256: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let version = artifact_version(bytes)?;
        if version != MODEL_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedVersion { found: version });
        }
        let mut envelope = decode_v2_envelope(
            bytes,
            LINEAR_REGRESSION_ARTIFACT_KIND,
            LINEAR_PAYLOAD_VERSION,
            &[(SchemaRole::Input, expected_feature_schema_sha256)],
        )?;
        let component = decode_component(
            &mut envelope,
            LINEAR_STATE_COMPONENT_KIND,
            LINEAR_STATE_COMPONENT_VERSION,
        )?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Self::decode_payload(component)
    }

    fn decode_payload(mut cursor: ArtifactCursor<'_>) -> Result<Self, ArtifactError> {
        let n_features_in = cursor.u32()? as usize;
        let fit_intercept = match cursor.u32()? {
            0 => false,
            1 => true,
            _ => return Err(ArtifactError::InvalidPayload),
        };
        let tol = cursor.f32()?;
        let rank = cursor.u32()? as usize;
        let intercept = cursor.f32()?;
        let coefficient_count = cursor.u32()? as usize;
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || coefficient_count != n_features_in
            || !tol.is_finite()
            || tol < 0.0
            || rank > n_features_in
            || !intercept.is_finite()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut coefficients = Vec::with_capacity(coefficient_count);
        for _ in 0..coefficient_count {
            let value = cursor.f32()?;
            if !value.is_finite() {
                return Err(ArtifactError::InvalidPayload);
            }
            coefficients.push(value);
        }
        if !cursor.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in,
            params: LinearRegressionParams { fit_intercept, tol },
            coefficients,
            intercept,
            rank,
        })
    }
}

impl Estimator for LinearRegression {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for LinearRegression {
    type Params = LinearRegressionParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Regressor for LinearRegression {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        validate_predict(data, output.len(), self.n_features_in)?;
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            *slot = validate_prediction(self.predict_value(row), row_index)?;
        }
        Ok(())
    }
}

fn validate_fit(
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    sample_weights: Option<&SampleWeights>,
    params: &LinearRegressionParams,
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
    if !params.tol.is_finite() || params.tol < 0.0 {
        return Err(ModelError::InvalidLeastSquaresTolerance);
    }
    Ok(())
}

fn validate_predict(
    data: &MatrixView<'_>,
    output_len: usize,
    features: usize,
) -> Result<(), ModelError> {
    if data.columns() != features {
        return Err(ModelError::FeatureDimension {
            expected: features,
            actual: data.columns(),
        });
    }
    if output_len != data.rows() {
        return Err(ModelError::OutputLength {
            expected: data.rows(),
            actual: output_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn fits_full_rank_intercept_model() {
        let data = DenseMatrix::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0], 4, 2).unwrap();
        let targets = RegressionTargets::new(vec![3.0, 4.0, 5.0, 11.0]).unwrap();
        let model =
            LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
                .unwrap();
        assert_close(model.coefficients()[0], 1.0);
        assert_close(model.coefficients()[1], 2.0);
        assert_close(model.intercept(), 3.0);
        assert_eq!(model.rank(), 2);
        assert_eq!(model.predict(&data.as_view()).unwrap(), targets.as_slice());
    }

    #[test]
    fn returns_minimum_norm_rank_deficient_and_underdetermined_solutions() {
        let rank_deficient = DenseMatrix::new(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2).unwrap();
        let model = LinearRegression::fit(
            &rank_deficient.as_view(),
            &RegressionTargets::new(vec![1.0, 2.0, 3.0]).unwrap(),
            LinearRegressionParams::default()
                .with_fit_intercept(false)
                .with_tol(0.0),
        )
        .unwrap();
        assert_close(model.coefficients()[0], 0.2);
        assert_close(model.coefficients()[1], 0.4);
        assert_eq!(model.rank(), 1);
        assert_eq!(model.intercept().to_bits(), 0.0_f32.to_bits());

        let underdetermined = DenseMatrix::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 2, 3).unwrap();
        let model = LinearRegression::fit(
            &underdetermined.as_view(),
            &RegressionTargets::new(vec![1.0, 1.0]).unwrap(),
            LinearRegressionParams::default()
                .with_fit_intercept(false)
                .with_tol(0.0),
        )
        .unwrap();
        assert_close(model.coefficients()[0], 1.0 / 3.0);
        assert_close(model.coefficients()[1], 1.0 / 3.0);
        assert_close(model.coefficients()[2], 2.0 / 3.0);
        assert_eq!(model.rank(), 2);
    }

    #[test]
    fn weighted_fit_matches_integer_replication() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 4.0], 4, 1).unwrap();
        let targets = RegressionTargets::new(vec![1.0, 2.0, 2.0, 5.0]).unwrap();
        let weights = SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0]).unwrap();
        let weighted = LinearRegression::fit_weighted(
            &data.as_view(),
            &targets,
            &weights,
            LinearRegressionParams::default(),
        )
        .unwrap();
        let replicated_data = DenseMatrix::new(vec![0.0, 1.0, 1.0, 2.0, 4.0, 4.0], 6, 1).unwrap();
        let replicated_targets =
            RegressionTargets::new(vec![1.0, 2.0, 2.0, 2.0, 5.0, 5.0]).unwrap();
        let replicated = LinearRegression::fit(
            &replicated_data.as_view(),
            &replicated_targets,
            LinearRegressionParams::default(),
        )
        .unwrap();
        assert_close(weighted.coefficients()[0], replicated.coefficients()[0]);
        assert_close(weighted.intercept(), replicated.intercept());
    }

    #[test]
    fn constant_columns_and_tolerance_cutoff_are_deterministic() {
        let data = DenseMatrix::new(vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0], 3, 2).unwrap();
        let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0]).unwrap();
        let left =
            LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
                .unwrap();
        let right =
            LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
                .unwrap();
        assert_eq!(left, right);
        assert_close(left.coefficients()[0], 0.0);
        assert_close(left.coefficients()[1], 2.0);
        assert_close(left.intercept(), 1.0);
        assert_eq!(left.rank(), 1);

        let cutoff_data = DenseMatrix::new(vec![1.0, 0.0, 0.0, 1.0e-7], 2, 2).unwrap();
        let cutoff_targets = RegressionTargets::new(vec![1.0, 1.0e-7]).unwrap();
        let truncated = LinearRegression::fit(
            &cutoff_data.as_view(),
            &cutoff_targets,
            LinearRegressionParams::default()
                .with_fit_intercept(false)
                .with_tol(1.0e-6),
        )
        .unwrap();
        let full = LinearRegression::fit(
            &cutoff_data.as_view(),
            &cutoff_targets,
            LinearRegressionParams::default()
                .with_fit_intercept(false)
                .with_tol(0.0),
        )
        .unwrap();
        assert_eq!(truncated.rank(), 1);
        assert_close(truncated.coefficients()[1], 0.0);
        assert_eq!(full.rank(), 2);
        assert_close(full.coefficients()[1], 1.0);
    }

    #[test]
    fn validates_tolerance_shapes_and_output_before_writing() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0], 3, 1).unwrap();
        let targets = RegressionTargets::new(vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            LinearRegression::fit(
                &data.as_view(),
                &RegressionTargets::new(vec![0.0, 1.0]).unwrap(),
                LinearRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::TargetLength {
                rows: 3,
                targets: 2,
            }
        );
        assert_eq!(
            LinearRegression::fit_weighted(
                &data.as_view(),
                &targets,
                &SampleWeights::new(vec![1.0, 1.0]).unwrap(),
                LinearRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::SampleWeightLength {
                rows: 3,
                weights: 2,
            }
        );
        assert_eq!(
            LinearRegression::fit(
                &data.as_view(),
                &targets,
                LinearRegressionParams::default().with_tol(-1.0),
            )
            .unwrap_err(),
            ModelError::InvalidLeastSquaresTolerance
        );
        let model =
            LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
                .unwrap();
        let mut output = [9.0; 2];
        assert_eq!(
            model
                .predict_into(&data.as_view(), &mut output)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 3,
                actual: 2,
            }
        );
        assert_eq!(output, [9.0; 2]);
    }

    #[test]
    fn artifact_is_distinct_deterministic_and_schema_bound() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0], 3, 1).unwrap();
        let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0]).unwrap();
        let model =
            LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
                .unwrap();
        let schema = [3; 32];
        let left = model.to_artifact(schema).unwrap();
        let right = model.to_artifact(schema).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            LinearRegression::from_artifact(&left, schema).unwrap(),
            model
        );
        assert_eq!(
            LinearRegression::from_artifact(&left, [4; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
    }
}
