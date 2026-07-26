//! Dense L2-regularized linear regression.

use super::least_squares;
use crate::api::{
    Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor,
    validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, MODEL_ARTIFACT_VERSION, ModelArtifact,
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
///
/// Ridge adds an L2 penalty to the least-squares objective, which shrinks
/// coefficients toward zero without ever setting one exactly to zero. Use it
/// where features are correlated or the fit is unstable; use [`Lasso`] instead
/// when the goal is to *remove* features.
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::linear_model::{Ridge, RidgeParams};
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
/// let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;
///
/// let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())?;
///
/// assert_eq!(model.n_features_in(), 1);
/// let predictions = model.predict(&data.as_view())?;
/// assert_eq!(predictions.len(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # `alpha` shrinks, and the retained value is exact
///
/// A larger `alpha` gives a strictly smaller coefficient. The value a fit was
/// given is retained exactly and readable through
/// [`get_params`](crate::api::HasParams::get_params), so a fitted model can
/// always say what it was fitted with.
///
/// ```
/// use ferricml::api::HasParams;
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::linear_model::{Ridge, RidgeParams};
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
/// let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;
///
/// let weak = Ridge::fit(
///     &data.as_view(),
///     &targets,
///     RidgeParams::default().with_alpha(0.01),
/// )?;
/// let strong = Ridge::fit(
///     &data.as_view(),
///     &targets,
///     RidgeParams::default().with_alpha(100.0),
/// )?;
///
/// assert!(strong.coefficients()[0].abs() < weak.coefficients()[0].abs());
/// assert_eq!(strong.get_params().alpha(), 100.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Note that this `alpha` accompanies an *undivided* squared-error term, so it
/// is a different quantity from [`Lasso`]'s and [`ElasticNet`]'s, which are
/// measured against a mean. The two agree at
/// `ridge_alpha = alpha * total_weight`; see the frozen reference semantics for
/// why the scales are stated rather than reconciled.
///
/// [`Lasso`]: crate::linear_model::Lasso
/// [`ElasticNet`]: crate::linear_model::ElasticNet
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
        let mut coefficients = Vec::with_capacity(cursor.bounded_capacity(coefficient_count, 4));
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

impl ModelArtifact for Ridge {
    const ARTIFACT_KIND: u16 = RIDGE_ARTIFACT_KIND;

    /// Encodes this model in FerricML's bounded artifact format.
    fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
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
            Self::ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &component,
        )
    }

    /// Decodes a ridge model after checking integrity and feature identity.
    fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let version = artifact_version(bytes)?;
        if version != MODEL_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedVersion { found: version });
        }
        let mut envelope = decode_v2_envelope(
            bytes,
            Self::ARTIFACT_KIND,
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
}

impl Estimator for Ridge {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for Ridge {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
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
    fn weighted_and_intercept_fit_bits_are_frozen() {
        let data = DenseMatrix::new(
            vec![0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0, 1.0, 5.0, 3.0],
            6,
            2,
        )
        .unwrap();
        let targets = RegressionTargets::new(vec![0.5, 1.5, 1.0, 3.5, 4.0, 5.5]).unwrap();
        let weights = SampleWeights::new(vec![1.0, 2.0, 0.5, 1.5, 3.0, 2.0]).unwrap();

        let cases = [
            (false, true, [1_062_563_850, 1_052_534_438], 1_035_716_202),
            (false, false, [1_062_832_872, 1_053_127_666], 0),
            (true, true, [1_063_562_964, 1_052_156_503], 3_181_533_034),
            (true, false, [1_063_313_837, 1_051_753_503], 0),
        ];
        for (weighted, fit_intercept, expected_coefficients, expected_intercept) in cases {
            let params = RidgeParams::default().with_fit_intercept(fit_intercept);
            let model = if weighted {
                Ridge::fit_weighted(&data.as_view(), &targets, &weights, params).unwrap()
            } else {
                Ridge::fit(&data.as_view(), &targets, params).unwrap()
            };
            assert_eq!(
                model
                    .coefficients()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_coefficients
            );
            assert_eq!(model.intercept().to_bits(), expected_intercept);
        }
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

    /// `fit_intercept` is the one payload field with two legal encodings, and
    /// only one of them was ever decoded anywhere in the crate: every other
    /// round-trip, fingerprint and hardening fixture persists the fitting
    /// default. A reader that lost the `0` arm would reject every stored
    /// no-intercept model and no test would notice, so both configurations are
    /// decoded here — and each is decoded back to the *same bytes*, which is
    /// what makes the flag round-trip rather than merely survive.
    #[test]
    fn both_persisted_intercept_configurations_decode_to_the_model_that_wrote_them() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0], 4, 2).unwrap();
        let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 5.0]).unwrap();
        let mut encodings = Vec::new();
        for fit_intercept in [false, true] {
            let model = Ridge::fit(
                &data.as_view(),
                &targets,
                RidgeParams::default().with_fit_intercept(fit_intercept),
            )
            .unwrap();
            assert_eq!(model.get_params().fit_intercept(), fit_intercept);
            let bytes = model.to_artifact([5; 32]).unwrap();
            let decoded = Ridge::from_artifact(&bytes, [5; 32]).unwrap();
            assert_eq!(decoded, model);
            assert_eq!(decoded.get_params().fit_intercept(), fit_intercept);
            assert_eq!(decoded.to_artifact([5; 32]).unwrap(), bytes);
            encodings.push(bytes);
        }
        assert_ne!(
            encodings[0], encodings[1],
            "the two configurations must not encode to the same bytes"
        );
    }
}
