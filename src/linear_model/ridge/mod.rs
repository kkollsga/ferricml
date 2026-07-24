//! Dense L2-regularized linear regression.

use super::least_squares;
use crate::api::{
    Estimator, HasParams, ModelError, Regressor, validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, MODEL_ARTIFACT_VERSION,
    RIDGE_ARTIFACT_KIND, SchemaRole, artifact_version, decode_component, decode_v2_envelope,
    encode_component, encode_v2_envelope,
};
use crate::data::{MatrixView, RegressionTargets, SampleWeights};

const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const PAYLOAD_VERSION: u16 = 1;
const STATE_COMPONENT_KIND: u16 = 1;
const STATE_COMPONENT_VERSION: u16 = 1;
const FIXED_PAYLOAD_BYTES: usize = 5 * 4;

/// Parameters for [`Ridge`].
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeParams {
    alpha: f32,
    fit_intercept: bool,
}

impl Default for RidgeParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }
}

impl RidgeParams {
    /// Sets the non-negative L2 penalty applied to coefficients.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enables or disables the fitted intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Returns the L2 coefficient penalty.
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns whether an intercept is fitted.
    pub const fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }
}

/// Dense single-target ridge regression.
#[derive(Clone, Debug, PartialEq)]
pub struct Ridge {
    n_features_in: usize,
    params: RidgeParams,
    coefficients: Vec<f32>,
    intercept: f32,
}

impl Ridge {
    /// Fits an unweighted dense ridge model.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: RidgeParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits a dense ridge model with per-row sample weights.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: RidgeParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: RidgeParams,
    ) -> Result<Self, ModelError> {
        validate_fit(data, targets, sample_weights, &params)?;
        let fit = least_squares::fit_ridge_dense(
            data,
            targets.as_slice(),
            sample_weights,
            params.fit_intercept,
            params.alpha,
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

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fit parameters.
    pub const fn get_params(&self) -> &RidgeParams {
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
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        if self.n_features_in > MAX_ARTIFACT_FEATURES {
            return Err(ArtifactError::InvalidPayload);
        }
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut state =
            ArtifactPayloadWriter::with_capacity(FIXED_PAYLOAD_BYTES + self.coefficients.len() * 4);
        state.u32(n_features);
        state.f32(self.params.alpha);
        state.u32(u32::from(self.params.fit_intercept));
        state.f32(self.intercept);
        state.u32(n_features);
        for &coefficient in &self.coefficients {
            state.f32(coefficient);
        }
        let component = encode_component(
            STATE_COMPONENT_KIND,
            STATE_COMPONENT_VERSION,
            &state.finish(),
        )?;
        encode_v2_envelope(
            RIDGE_ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &component,
        )
    }

    /// Decodes a ridge model after checking integrity and feature identity.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let version = artifact_version(bytes)?;
        if version != MODEL_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedVersion { found: version });
        }
        let mut envelope = decode_v2_envelope(
            bytes,
            RIDGE_ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let component =
            decode_component(&mut envelope, STATE_COMPONENT_KIND, STATE_COMPONENT_VERSION)?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Self::decode_payload(component)
    }

    fn decode_payload(mut cursor: ArtifactCursor<'_>) -> Result<Self, ArtifactError> {
        let n_features_in = cursor.u32()? as usize;
        let alpha = cursor.f32()?;
        let fit_intercept = match cursor.u32()? {
            0 => false,
            1 => true,
            _ => return Err(ArtifactError::InvalidPayload),
        };
        let intercept = cursor.f32()?;
        let coefficient_count = cursor.u32()? as usize;
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || coefficient_count != n_features_in
            || !alpha.is_finite()
            || alpha < 0.0
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
            params: RidgeParams {
                alpha,
                fit_intercept,
            },
            coefficients,
            intercept,
        })
    }
}

impl Estimator for Ridge {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for Ridge {
    type Params = RidgeParams;
    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Regressor for Ridge {
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

fn validate_fit(
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    weights: Option<&SampleWeights>,
    params: &RidgeParams,
) -> Result<(), ModelError> {
    if data.rows() != targets.len() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: targets.len(),
        });
    }
    if let Some(weights) = weights
        && data.rows() != weights.len()
    {
        return Err(ModelError::SampleWeightLength {
            rows: data.rows(),
            weights: weights.len(),
        });
    }
    if !params.alpha.is_finite() || params.alpha < 0.0 {
        return Err(ModelError::InvalidRidgeAlpha);
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
    fn fits_default_ridge_and_excludes_intercept_from_penalty() {
        let data = DenseMatrix::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0], 4, 2).unwrap();
        let targets = RegressionTargets::new(vec![3.0, 4.0, 5.0, 11.0]).unwrap();
        let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
        assert_close(model.coefficients()[0], 0.942_028_9);
        assert_close(model.coefficients()[1], 1.739_130_3);
        assert_close(model.intercept(), 3.304_348);
    }

    #[test]
    fn alpha_zero_matches_minimum_norm_linear_regression() {
        let data = DenseMatrix::new(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2).unwrap();
        let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0]).unwrap();
        let ridge = Ridge::fit(
            &data.as_view(),
            &targets,
            RidgeParams::default()
                .with_alpha(0.0)
                .with_fit_intercept(false),
        )
        .unwrap();
        assert_close(ridge.coefficients()[0], 0.2);
        assert_close(ridge.coefficients()[1], 0.4);
        assert_eq!(ridge.intercept().to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn weighted_artifact_round_trip_and_validation() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 4.0], 4, 1).unwrap();
        let targets = RegressionTargets::new(vec![1.0, 2.0, 2.0, 5.0]).unwrap();
        let model = Ridge::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0]).unwrap(),
            RidgeParams::default(),
        )
        .unwrap();
        let bytes = model.to_artifact([9; 32]).unwrap();
        assert_eq!(Ridge::from_artifact(&bytes, [9; 32]).unwrap(), model);
        assert_eq!(
            Ridge::fit(
                &data.as_view(),
                &targets,
                RidgeParams::default().with_alpha(-1.0)
            )
            .unwrap_err(),
            ModelError::InvalidRidgeAlpha
        );
    }
}
