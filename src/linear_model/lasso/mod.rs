//! Dense L1-regularized linear regression.

use super::coordinate_descent::fit_elastic_net_dense;
use crate::api::{
    Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor,
    validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, LASSO_ARTIFACT_KIND,
    MODEL_ARTIFACT_VERSION, ModelArtifact, SchemaRole, artifact_version, decode_component,
    decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::{MatrixView, RegressionTargets, SampleWeights};
use crate::loss::ElasticNetPenalty;

const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const PAYLOAD_VERSION: u16 = 1;
const STATE_COMPONENT_KIND: u16 = 1;
const STATE_COMPONENT_VERSION: u16 = 1;
const FIXED_PAYLOAD_BYTES: usize = 8 * 4;

/// Parameters for [`Lasso`].
#[derive(Clone, Debug, PartialEq)]
pub struct LassoParams {
    alpha: f32,
    fit_intercept: bool,
    max_iter: usize,
    tol: f32,
}

impl Default for LassoParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            max_iter: 1_000,
            tol: 1.0e-4,
        }
    }
}

impl LassoParams {
    /// Sets the non-negative L1 penalty applied to coefficients.
    ///
    /// `alpha` multiplies `||b||_1` against a squared-error term divided by
    /// twice the total sample weight, so it is a *mean*-scaled penalty: the
    /// same `alpha` means the same thing at any row count. `alpha = 0` is an
    /// ordinary least-squares fit, for which [`LinearRegression`](
    /// super::LinearRegression) is the better-conditioned choice.
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

    /// Sets the maximum number of coordinate sweeps.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    ///
    /// A sweep has converged when the largest absolute coefficient change
    /// across it is at most this value.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns the L1 coefficient penalty.
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns whether an intercept is fitted.
    pub const fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Returns the maximum number of coordinate sweeps.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    pub const fn tol(&self) -> f32 {
        self.tol
    }
}

/// Dense single-target L1-regularized linear regression.
///
/// Lasso is the linear model that *removes* features rather than shrinking them
/// toward zero: coefficients it excludes are exactly `0.0`, so
/// [`coefficients`](Self::coefficients) can be read as a selection and not only
/// as a set of weights.
///
/// # What `alpha` is measured against
///
/// The minimized objective is
///
/// ```text
/// (1 / (2 * W)) * sum_i w_i * (y_i - b0 - x_i . b)^2 + alpha * ||b||_1
/// ```
///
/// with `W` the total sample weight — the row count when none are supplied. The
/// data term is divided by `2W`, which is what makes `alpha` independent of the
/// sample size, and it is also why this `alpha` is **not** the same quantity as
/// [`Ridge`](super::Ridge)'s: a ridge penalty accompanies an undivided
/// squared-error term.
///
/// # The penalty applies to raw-scale coefficients
///
/// Fitting centers the design and the target when an intercept is requested and
/// does not rescale the columns, so a feature measured in larger units is
/// penalized less. This is a frozen, documented choice rather than an
/// oversight. A caller who wants scale-free selection composes a
/// [`StandardScaler`](crate::preprocessing::StandardScaler) in front, where the
/// transformation is explicit and persists alongside the model.
///
/// # Persistence
///
/// This estimator declares no artifact capability. Its on-disk schema is a
/// separate contract from its semantics and is not part of this addition.
#[derive(Clone, Debug, PartialEq)]
pub struct Lasso {
    n_features_in: usize,
    params: LassoParams,
    coefficients: Vec<f32>,
    intercept: f32,
    sweeps: usize,
}

impl Lasso {
    /// Fits an unweighted dense lasso model.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: LassoParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits a dense lasso model with per-row sample weights.
    ///
    /// A weight is a fractional row count: an integer weight of `k` is the same
    /// fit as the row appearing `k` times, and scaling every weight by the same
    /// factor changes nothing, because the data term is divided by the total.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: LassoParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: LassoParams,
    ) -> Result<Self, ModelError> {
        super::validate_penalized_fit(
            data,
            targets.len(),
            sample_weights,
            params.alpha,
            None,
            params.max_iter,
            params.tol,
        )?;
        let fit = fit_elastic_net_dense(
            data,
            targets.as_slice(),
            sample_weights,
            params.fit_intercept,
            ElasticNetPenalty::new(f64::from(params.alpha), 1.0),
            params.max_iter,
            params.tol,
        )?;
        let sweeps = fit.sweeps;
        let (coefficients, intercept) = super::narrow_dense_fit(fit.coefficients, fit.intercept)?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            coefficients,
            intercept,
            sweeps,
        })
    }

    /// Returns fitted coefficients in input-feature order.
    ///
    /// A removed coefficient is exactly `0.0`, and positively signed.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Returns the fitted intercept, which is never penalized.
    pub const fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns how many coefficients the fit left at exactly zero.
    pub fn n_zero_coefficients(&self) -> usize {
        self.coefficients.iter().filter(|v| **v == 0.0).count()
    }

    /// Returns the number of coordinate sweeps performed.
    pub const fn n_iter(&self) -> usize {
        self.sweeps
    }

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fit parameters.
    pub const fn get_params(&self) -> &LassoParams {
        &self.params
    }

    /// Predicts one regression value.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        validate_scalar_row(row, self.n_features_in)?;
        validate_prediction(
            super::dense_prediction(row, &self.coefficients, self.intercept),
            0,
        )
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
}

impl Estimator for Lasso {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for Lasso {
    /// Weighted fitting and persistence.
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
}

impl ModelArtifact for Lasso {
    const ARTIFACT_KIND: u16 = LASSO_ARTIFACT_KIND;

    /// Encodes the fitted sparse coefficient vector and the penalty it came
    /// from.
    ///
    /// The sweep count is stored rather than recomputed. It is fitted state a
    /// caller can read through [`Lasso::n_iter`], so a decoded model that
    /// re-derived it would not equal the model that was written.
    fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        if self.n_features_in > MAX_ARTIFACT_FEATURES {
            return Err(ArtifactError::InvalidPayload);
        }
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let sweeps = u32::try_from(self.sweeps).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut state =
            ArtifactPayloadWriter::with_capacity(FIXED_PAYLOAD_BYTES + self.coefficients.len() * 4);
        state.u32(n_features);
        state.f32(self.params.alpha);
        state.u32(u32::from(self.params.fit_intercept));
        state.u32(max_iter);
        state.f32(self.params.tol);
        state.u32(sweeps);
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

    /// Decodes a lasso model after checking integrity and feature identity.
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

impl Lasso {
    fn decode_payload(mut cursor: ArtifactCursor<'_>) -> Result<Self, ArtifactError> {
        let n_features_in = cursor.u32()? as usize;
        let alpha = cursor.f32()?;
        let fit_intercept = match cursor.u32()? {
            0 => false,
            1 => true,
            _ => return Err(ArtifactError::InvalidPayload),
        };
        let max_iter = cursor.u32()? as usize;
        let tol = cursor.f32()?;
        let sweeps = cursor.u32()? as usize;
        let intercept = cursor.f32()?;
        let coefficient_count = cursor.u32()? as usize;
        // Every bound a fit enforces, re-enforced here: bytes are never
        // trusted to describe a model a fit could have produced. A sweep count
        // above the iteration budget is the one that only a decoder can see.
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || coefficient_count != n_features_in
            || !alpha.is_finite()
            || alpha < 0.0
            || max_iter == 0
            || !tol.is_finite()
            || tol <= 0.0
            || sweeps > max_iter
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
            params: LassoParams {
                alpha,
                fit_intercept,
                max_iter,
                tol,
            },
            coefficients,
            intercept,
            sweeps,
        })
    }
}

impl HasParams for Lasso {
    type Params = LassoParams;
    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Regressor for Lasso {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        super::predict_dense_into(
            data,
            output,
            self.n_features_in,
            &self.coefficients,
            self.intercept,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::ElasticNet;
    use super::*;
    use crate::data::DenseMatrix;

    fn problem() -> (DenseMatrix, RegressionTargets) {
        // Two informative columns, one pure noise column, one duplicate.
        let rows = 24;
        let columns = 4;
        let mut state = 0x1a5_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1_u32 << 31) as f32) * 2.0 - 1.0
        };
        let values = (0..rows * columns).map(|_| next()).collect::<Vec<f32>>();
        let targets = (0..rows)
            .map(|row| 2.0 * values[row * columns] - 1.0 * values[row * columns + 2] + 0.5)
            .collect::<Vec<f32>>();
        (
            DenseMatrix::new(values, rows, columns).expect("matrix"),
            RegressionTargets::new(targets).expect("targets"),
        )
    }

    #[test]
    fn a_moderate_penalty_selects_the_informative_columns() {
        let (data, targets) = problem();
        let model = Lasso::fit(
            &data.as_view(),
            &targets,
            LassoParams::default().with_alpha(0.05).with_tol(1.0e-8),
        )
        .expect("fit");
        assert!(model.coefficients()[0] > 1.5);
        assert!(model.coefficients()[2] < -0.5);
        assert!(
            model.n_zero_coefficients() >= 1,
            "{:?}",
            model.coefficients()
        );
        for &coefficient in model.coefficients() {
            if coefficient == 0.0 {
                assert!(coefficient.is_sign_positive());
            }
        }
    }

    #[test]
    fn a_large_penalty_removes_everything_and_leaves_the_target_mean() {
        let (data, targets) = problem();
        let model = Lasso::fit(
            &data.as_view(),
            &targets,
            LassoParams::default().with_alpha(50.0),
        )
        .expect("fit");
        assert_eq!(model.n_zero_coefficients(), model.coefficients().len());
        let mean = targets.as_slice().iter().sum::<f32>() / targets.len() as f32;
        assert!((model.intercept() - mean).abs() <= 1.0e-5);
        // Every prediction is that same constant.
        let predictions = model.predict(&data.as_view()).expect("predict");
        assert!(predictions.iter().all(|value| *value == model.intercept()));
    }

    #[test]
    fn parameters_are_validated_before_any_fitting_work() {
        let (data, targets) = problem();
        for (params, expected) in [
            (
                LassoParams::default().with_alpha(-1.0),
                ModelError::InvalidPenaltyAlpha,
            ),
            (
                LassoParams::default().with_alpha(f32::NAN),
                ModelError::InvalidPenaltyAlpha,
            ),
            (
                LassoParams::default().with_max_iter(0),
                ModelError::InvalidIterationCount,
            ),
            (
                LassoParams::default().with_tol(0.0),
                ModelError::InvalidTolerance,
            ),
        ] {
            assert_eq!(
                Lasso::fit(&data.as_view(), &targets, params).unwrap_err(),
                expected
            );
        }
        assert_eq!(
            Lasso::fit_weighted(
                &data.as_view(),
                &targets,
                &SampleWeights::new(vec![1.0; 3]).expect("weights"),
                LassoParams::default(),
            )
            .unwrap_err(),
            ModelError::SampleWeightLength {
                rows: data.rows(),
                weights: 3,
            }
        );
    }

    #[test]
    fn an_exhausted_sweep_budget_is_reported_rather_than_returned() {
        let (data, targets) = problem();
        assert!(matches!(
            Lasso::fit(
                &data.as_view(),
                &targets,
                LassoParams::default()
                    .with_alpha(1.0e-4)
                    .with_max_iter(1)
                    .with_tol(1.0e-12),
            ),
            Err(ModelError::SolverDidNotConverge { iterations: 1 })
        ));
    }

    #[test]
    fn refitting_reproduces_the_same_bits_and_retains_its_parameters() {
        let (data, targets) = problem();
        let params = LassoParams::default().with_alpha(0.02);
        let first = Lasso::fit(&data.as_view(), &targets, params.clone()).expect("fit");
        let second = Lasso::fit(&data.as_view(), &targets, params.clone()).expect("refit");
        assert_eq!(first, second);
        assert_eq!(first.get_params(), &params);
        assert_eq!(first.n_features_in(), data.columns());
    }

    #[test]
    fn the_declared_capabilities_match_the_entry_points_that_exist() {
        assert!(Lasso::CAPABILITIES.sample_weights());
        assert!(Lasso::CAPABILITIES.artifact());
        assert!(!Lasso::CAPABILITIES.multiclass());
    }

    #[test]
    fn a_fitted_model_round_trips_through_its_artifact_and_predicts_identically() {
        const SCHEMA: [u8; 32] = [7; 32];
        let (data, targets) = problem();
        let model = Lasso::fit(
            &data.as_view(),
            &targets,
            LassoParams::default().with_alpha(0.05).with_tol(1.0e-8),
        )
        .expect("fit");
        // A penalty that actually zeroes a coefficient, so the sparse vector
        // the artifact exists to carry is the one being round-tripped.
        assert!(model.n_zero_coefficients() >= 1);

        let bytes = model.to_artifact(SCHEMA).expect("encode");
        assert_eq!(bytes, model.to_artifact(SCHEMA).expect("re-encode"));

        let restored = Lasso::from_artifact(&bytes, SCHEMA).expect("decode");
        assert_eq!(restored, model);
        assert_eq!(restored.n_iter(), model.n_iter());
        assert_eq!(
            restored.predict(&data.as_view()).expect("predict"),
            model.predict(&data.as_view()).expect("predict")
        );
    }

    #[test]
    fn a_decoder_refuses_another_schema_and_another_estimators_bytes() {
        const SCHEMA: [u8; 32] = [7; 32];
        const OTHER: [u8; 32] = [9; 32];
        let (data, targets) = problem();
        let model = Lasso::fit(
            &data.as_view(),
            &targets,
            LassoParams::default().with_alpha(0.05).with_tol(1.0e-8),
        )
        .expect("fit");
        let bytes = model.to_artifact(SCHEMA).expect("encode");

        assert_eq!(
            Lasso::from_artifact(&bytes, OTHER),
            Err(ArtifactError::FeatureSchemaMismatch)
        );

        let mut corrupted = bytes.clone();
        let last = corrupted.len() - 40;
        corrupted[last] ^= 1;
        assert_eq!(
            Lasso::from_artifact(&corrupted, SCHEMA),
            Err(ArtifactError::ChecksumMismatch)
        );

        // The kind is what keeps the two penalized readers off each other's
        // bytes: their payload layouts differ by one word, so a reader that
        // trusted the layout alone would misread every field after it.
        assert_eq!(
            ElasticNet::from_artifact(&bytes, SCHEMA).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 69 }
        );
    }
}
