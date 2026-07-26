//! Dense linear regression under a combined L1 and L2 penalty.

use super::coordinate_descent::fit_elastic_net_dense;
use crate::api::{
    Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor,
    validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, ELASTIC_NET_ARTIFACT_KIND,
    MODEL_ARTIFACT_VERSION, ModelArtifact, SchemaRole, artifact_version, decode_component,
    decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::{MatrixView, RegressionTargets, SampleWeights};
use crate::loss::ElasticNetPenalty;

const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const PAYLOAD_VERSION: u16 = 1;
const STATE_COMPONENT_KIND: u16 = 1;
const STATE_COMPONENT_VERSION: u16 = 1;
const FIXED_PAYLOAD_BYTES: usize = 9 * 4;

/// Parameters for [`ElasticNet`].
#[derive(Clone, Debug, PartialEq)]
pub struct ElasticNetParams {
    alpha: f32,
    l1_ratio: f32,
    fit_intercept: bool,
    max_iter: usize,
    tol: f32,
}

impl Default for ElasticNetParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            fit_intercept: true,
            max_iter: 1_000,
            tol: 1.0e-4,
        }
    }
}

impl ElasticNetParams {
    /// Sets the non-negative penalty strength shared by both terms.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the mixing parameter, which must lie in `0.0..=1.0`.
    ///
    /// `1.0` is a pure L1 penalty and reproduces [`Lasso`](super::Lasso) at the
    /// same `alpha`; `0.0` is a pure L2 penalty. Anything between mixes them in
    /// the proportion the ratio names — see [`ElasticNet`] for the exact
    /// objective the two terms sit in.
    #[must_use]
    pub fn with_l1_ratio(mut self, l1_ratio: f32) -> Self {
        self.l1_ratio = l1_ratio;
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

    /// Returns the shared penalty strength.
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the mixing parameter.
    pub const fn l1_ratio(&self) -> f32 {
        self.l1_ratio
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

/// Dense single-target linear regression under an elastic-net penalty.
///
/// # The objective
///
/// ```text
/// (1 / (2 * W)) * sum_i w_i * (y_i - b0 - x_i . b)^2
///   + alpha * l1_ratio * ||b||_1
///   + 0.5 * alpha * (1 - l1_ratio) * ||b||_2^2
/// ```
///
/// with `W` the total sample weight, which is the row count when none are
/// supplied. This is the reference contract's documented parametrization, so a
/// caller who knows one knows the other.
///
/// # Why mix the two penalties
///
/// A pure L1 penalty is unstable where features are correlated: it picks one of
/// a correlated group essentially arbitrarily and zeroes the rest, and which
/// one it picks can change under a small perturbation of the data. Adding an L2
/// term makes the objective strictly convex, which restores a unique solution
/// and spreads weight across a correlated group instead of choosing between its
/// members. `l1_ratio` is how much sparsity is traded for that stability.
///
/// # Relationships worth knowing
///
/// - `l1_ratio = 1` is exactly [`Lasso`](super::Lasso) at the same `alpha`.
/// - `l1_ratio = 0` is the ridge objective, but **not** at
///   [`Ridge`](super::Ridge)'s `alpha`: the two agree at
///   `ridge_alpha = alpha * total_weight`, because a ridge penalty accompanies
///   an undivided squared-error term. [`Ridge`](super::Ridge) also solves in
///   closed form rather than by sweeps, which is faster and better conditioned,
///   so it remains the right choice for a pure L2 fit.
/// - `alpha = 0` is ordinary least squares, for which
///   [`LinearRegression`](super::LinearRegression) is better conditioned.
///
/// # Standardization and persistence
///
/// As with [`Lasso`](super::Lasso): the penalty applies to raw-scale
/// coefficients, fitting centers but does not rescale, the intercept is never
/// penalized, and the estimator persists through [`ModelArtifact`] under its
/// own artifact kind.
#[derive(Clone, Debug, PartialEq)]
pub struct ElasticNet {
    n_features_in: usize,
    params: ElasticNetParams,
    coefficients: Vec<f32>,
    intercept: f32,
    sweeps: usize,
}

impl ElasticNet {
    /// Fits an unweighted dense elastic-net model.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: ElasticNetParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits a dense elastic-net model with per-row sample weights.
    ///
    /// A weight is a fractional row count: an integer weight of `k` is the same
    /// fit as the row appearing `k` times, and scaling every weight by the same
    /// factor changes nothing, because the data term is divided by the total.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: ElasticNetParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: ElasticNetParams,
    ) -> Result<Self, ModelError> {
        super::validate_penalized_fit(
            data,
            targets.len(),
            sample_weights,
            params.alpha,
            Some(params.l1_ratio),
            params.max_iter,
            params.tol,
        )?;
        let fit = fit_elastic_net_dense(
            data,
            targets.as_slice(),
            sample_weights,
            params.fit_intercept,
            ElasticNetPenalty::new(f64::from(params.alpha), f64::from(params.l1_ratio)),
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
    /// A removed coefficient is exactly `0.0`, and positively signed. A pure L2
    /// mixture removes nothing, so this is a selection only when `l1_ratio` is
    /// above zero.
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
    pub const fn get_params(&self) -> &ElasticNetParams {
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

impl Estimator for ElasticNet {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for ElasticNet {
    /// Weighted fitting and persistence.
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
}

impl ModelArtifact for ElasticNet {
    const ARTIFACT_KIND: u16 = ELASTIC_NET_ARTIFACT_KIND;

    /// Encodes the fitted coefficient vector and the mixed penalty it came
    /// from.
    ///
    /// `l1_ratio` is stored beside `alpha` rather than folded into it. The two
    /// numbers are separately readable through
    /// [`get_params`](crate::api::HasParams::get_params), and a pair that
    /// multiplied to the same product would be a different fitted model with
    /// the same bytes.
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
        state.f32(self.params.l1_ratio);
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

    /// Decodes an elastic-net model after checking integrity and feature
    /// identity.
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

impl ElasticNet {
    fn decode_payload(mut cursor: ArtifactCursor<'_>) -> Result<Self, ArtifactError> {
        let n_features_in = cursor.u32()? as usize;
        let alpha = cursor.f32()?;
        let l1_ratio = cursor.f32()?;
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
        // The same bounds a fit enforces, including the `0..=1` that makes
        // `l1_ratio` a mixing weight rather than an arbitrary scale.
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || coefficient_count != n_features_in
            || !alpha.is_finite()
            || alpha < 0.0
            || !l1_ratio.is_finite()
            || !(0.0..=1.0).contains(&l1_ratio)
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
            params: ElasticNetParams {
                alpha,
                l1_ratio,
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

impl HasParams for ElasticNet {
    type Params = ElasticNetParams;
    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Regressor for ElasticNet {
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
    use super::super::{Lasso, LassoParams, Ridge, RidgeParams};
    use super::*;
    use crate::data::DenseMatrix;

    /// Two correlated informative columns plus two irrelevant ones, which is
    /// the situation elastic net exists for.
    fn correlated_problem() -> (DenseMatrix, RegressionTargets) {
        let rows = 30;
        let columns = 4;
        let mut state = 0x2b1_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1_u32 << 31) as f32) * 2.0 - 1.0
        };
        let mut values = Vec::with_capacity(rows * columns);
        for _ in 0..rows {
            let first = next();
            values.push(first);
            // Strongly but not pathologically correlated with the first.
            values.push(first + 0.4 * next());
            values.push(next());
            values.push(next());
        }
        let targets = (0..rows)
            .map(|row| 1.5 * values[row * columns] + 1.5 * values[row * columns + 1] + 0.75)
            .collect::<Vec<f32>>();
        (
            DenseMatrix::new(values, rows, columns).expect("matrix"),
            RegressionTargets::new(targets).expect("targets"),
        )
    }

    fn tight(alpha: f32, l1_ratio: f32) -> ElasticNetParams {
        ElasticNetParams::default()
            .with_alpha(alpha)
            .with_l1_ratio(l1_ratio)
            .with_max_iter(100_000)
            .with_tol(1.0e-7)
    }

    #[test]
    fn a_unit_ratio_is_exactly_the_lasso_at_the_same_alpha() {
        // Not "close to": the same penalty reaches the same solver by the same
        // path, so the fitted bits have to match.
        let (data, targets) = correlated_problem();
        let elastic = ElasticNet::fit(&data.as_view(), &targets, tight(0.05, 1.0)).expect("fit");
        let lasso = Lasso::fit(
            &data.as_view(),
            &targets,
            LassoParams::default()
                .with_alpha(0.05)
                .with_max_iter(100_000)
                .with_tol(1.0e-7),
        )
        .expect("fit");
        assert_eq!(
            elastic
                .coefficients()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            lasso
                .coefficients()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(elastic.intercept().to_bits(), lasso.intercept().to_bits());
        assert_eq!(elastic.n_iter(), lasso.n_iter());
    }

    #[test]
    fn a_zero_ratio_agrees_with_ridge_at_the_documented_scaling() {
        // The two parametrizations differ by the total sample weight, which is
        // a documented consequence rather than something to hide.
        let (data, targets) = correlated_problem();
        let alpha = 0.2_f32;
        let elastic = ElasticNet::fit(&data.as_view(), &targets, tight(alpha, 0.0)).expect("fit");
        let ridge = Ridge::fit(
            &data.as_view(),
            &targets,
            RidgeParams::default().with_alpha(alpha * data.rows() as f32),
        )
        .expect("fit");
        for (index, (left, right)) in elastic
            .coefficients()
            .iter()
            .zip(ridge.coefficients())
            .enumerate()
        {
            assert!(
                (left - right).abs() <= 2.0e-5,
                "coefficient {index}: elastic {left}, ridge {right}"
            );
        }
        assert!((elastic.intercept() - ridge.intercept()).abs() <= 2.0e-5);
        // A pure L2 penalty shrinks but never removes.
        assert_eq!(elastic.n_zero_coefficients(), 0);
    }

    #[test]
    fn mixing_in_l2_spreads_a_correlated_pair_that_l1_alone_collapses() {
        // The behaviour the estimator exists for, stated as an inequality
        // between the two fits rather than as a description.
        let (data, targets) = correlated_problem();
        let lasso = ElasticNet::fit(&data.as_view(), &targets, tight(0.05, 1.0)).expect("fit");
        let mixed = ElasticNet::fit(&data.as_view(), &targets, tight(0.05, 0.2)).expect("fit");
        let gap = |model: &ElasticNet| (model.coefficients()[0] - model.coefficients()[1]).abs();
        assert!(
            gap(&mixed) < gap(&lasso),
            "mixed gap {} did not narrow lasso's {}",
            gap(&mixed),
            gap(&lasso)
        );
        assert!(lasso.n_zero_coefficients() >= mixed.n_zero_coefficients());
    }

    #[test]
    fn increasing_alpha_shrinks_monotonically_at_any_ratio() {
        let (data, targets) = correlated_problem();
        for &l1_ratio in &[0.0_f32, 0.25, 0.5, 1.0] {
            let mut previous = f32::INFINITY;
            for step in 0..=10 {
                let alpha = step as f32 / 10.0;
                let model = ElasticNet::fit(&data.as_view(), &targets, tight(alpha, l1_ratio))
                    .expect("fit");
                let magnitude = model.coefficients().iter().map(|v| v.abs()).sum::<f32>();
                assert!(
                    magnitude <= previous + 1.0e-5,
                    "l1_ratio {l1_ratio} alpha {alpha}: {magnitude} exceeds {previous}"
                );
                previous = magnitude;
            }
        }
    }

    #[test]
    fn parameters_are_validated_before_any_fitting_work() {
        let (data, targets) = correlated_problem();
        for (params, expected) in [
            (
                ElasticNetParams::default().with_alpha(-1.0),
                ModelError::InvalidPenaltyAlpha,
            ),
            (
                ElasticNetParams::default().with_l1_ratio(1.5),
                ModelError::InvalidL1Ratio,
            ),
            (
                ElasticNetParams::default().with_l1_ratio(-0.001),
                ModelError::InvalidL1Ratio,
            ),
            (
                ElasticNetParams::default().with_l1_ratio(f32::NAN),
                ModelError::InvalidL1Ratio,
            ),
            (
                ElasticNetParams::default().with_max_iter(0),
                ModelError::InvalidIterationCount,
            ),
            (
                ElasticNetParams::default().with_tol(-1.0),
                ModelError::InvalidTolerance,
            ),
        ] {
            assert_eq!(
                ElasticNet::fit(&data.as_view(), &targets, params).unwrap_err(),
                expected
            );
        }
        // The penalty is checked before the mixing parameter, which is the
        // order the family reports its errors in.
        assert_eq!(
            ElasticNet::fit(
                &data.as_view(),
                &targets,
                ElasticNetParams::default()
                    .with_alpha(-1.0)
                    .with_l1_ratio(2.0),
            )
            .unwrap_err(),
            ModelError::InvalidPenaltyAlpha
        );
        // Both endpoints of the ratio are accepted.
        for &l1_ratio in &[0.0_f32, 1.0] {
            assert!(
                ElasticNet::fit(
                    &data.as_view(),
                    &targets,
                    ElasticNetParams::default().with_l1_ratio(l1_ratio),
                )
                .is_ok()
            );
        }
    }

    /// The `correlated_problem` design with its second column pushed to near
    /// collinearity, which is where coordinate descent's convergence rate
    /// collapses.
    fn near_collinear_problem() -> (DenseMatrix, RegressionTargets) {
        let rows = 30;
        let columns = 4;
        let mut state = 0x2b1_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1_u32 << 31) as f32) * 2.0 - 1.0
        };
        let mut values = Vec::with_capacity(rows * columns);
        for _ in 0..rows {
            let first = next();
            values.push(first);
            values.push(first + 0.01 * next());
            values.push(next());
            values.push(next());
        }
        let targets = (0..rows)
            .map(|row| 1.5 * values[row * columns] + 1.5 * values[row * columns + 1] + 0.75)
            .collect::<Vec<f32>>();
        (
            DenseMatrix::new(values, rows, columns).expect("matrix"),
            RegressionTargets::new(targets).expect("targets"),
        )
    }

    #[test]
    fn a_near_collinear_design_is_refused_rather_than_half_fitted() {
        // Coordinate descent moves one axis at a time, so its rate collapses as
        // two columns approach collinearity and the optimum runs diagonally
        // between them. That is a real limit of the method. The estimator's
        // answer is a typed refusal — not a fit that stopped on the way and
        // looks exactly like one that arrived.
        let (data, targets) = near_collinear_problem();
        let strict = ElasticNetParams::default()
            .with_alpha(0.05)
            .with_l1_ratio(1.0)
            .with_max_iter(100_000)
            .with_tol(1.0e-8);
        assert!(
            matches!(
                ElasticNet::fit(&data.as_view(), &targets, strict.clone()),
                Err(ModelError::SolverDidNotConverge {
                    iterations: 100_000
                })
            ),
            "a pure L1 penalty on this design must not report convergence"
        );
        // Adding an L2 term restores strict convexity, and with it a
        // well-conditioned sweep. That is the other half of why the mixed
        // penalty exists, beside the sparsity trade.
        let mixed = ElasticNet::fit(&data.as_view(), &targets, strict.with_l1_ratio(0.5))
            .expect("an L2 term makes the same design tractable");
        assert!(mixed.n_iter() < 1_000, "{} sweeps", mixed.n_iter());
    }

    #[test]
    fn refitting_reproduces_the_same_bits_and_retains_its_parameters() {
        let (data, targets) = correlated_problem();
        let params = tight(0.02, 0.6);
        let first = ElasticNet::fit(&data.as_view(), &targets, params.clone()).expect("fit");
        let second = ElasticNet::fit(&data.as_view(), &targets, params.clone()).expect("refit");
        assert_eq!(first, second);
        assert_eq!(first.get_params(), &params);
    }

    #[test]
    fn integer_sample_weights_match_replicating_the_rows() {
        let values = [0.0_f32, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0];
        let targets = [1.0_f32, 2.0, 4.0, 7.0];
        let weights = [1.0_f32, 2.0, 1.0, 3.0];
        let data = DenseMatrix::new(values.to_vec(), 4, 2).expect("matrix");
        let weighted = ElasticNet::fit_weighted(
            &data.as_view(),
            &RegressionTargets::new(targets.to_vec()).expect("targets"),
            &SampleWeights::new(weights.to_vec()).expect("weights"),
            tight(0.05, 0.6),
        )
        .expect("weighted fit");

        let mut replicated_values = Vec::new();
        let mut replicated_targets = Vec::new();
        for row in 0..4 {
            for _ in 0..weights[row] as usize {
                replicated_values.extend_from_slice(&values[row * 2..row * 2 + 2]);
                replicated_targets.push(targets[row]);
            }
        }
        let rows = replicated_targets.len();
        let replicated = DenseMatrix::new(replicated_values, rows, 2).expect("matrix");
        let unweighted = ElasticNet::fit(
            &replicated.as_view(),
            &RegressionTargets::new(replicated_targets).expect("targets"),
            tight(0.05, 0.6),
        )
        .expect("replicated fit");
        for (index, (left, right)) in weighted
            .coefficients()
            .iter()
            .zip(unweighted.coefficients())
            .enumerate()
        {
            assert!(
                (left - right).abs() <= 1.0e-6,
                "coefficient {index}: weighted {left}, replicated {right}"
            );
        }
        assert!((weighted.intercept() - unweighted.intercept()).abs() <= 1.0e-6);
    }

    #[test]
    fn the_declared_capabilities_match_the_entry_points_that_exist() {
        assert!(ElasticNet::CAPABILITIES.sample_weights());
        assert!(ElasticNet::CAPABILITIES.artifact());
        assert!(!ElasticNet::CAPABILITIES.multiclass());
    }

    #[test]
    fn a_fitted_model_round_trips_through_its_artifact_and_predicts_identically() {
        const SCHEMA: [u8; 32] = [7; 32];
        let (data, targets) = correlated_problem();
        let model = ElasticNet::fit(
            &data.as_view(),
            &targets,
            ElasticNetParams::default()
                .with_alpha(0.05)
                .with_l1_ratio(0.75)
                .with_tol(1.0e-8),
        )
        .expect("fit");
        // A penalty that actually zeroes a coefficient, so the sparse vector
        // the artifact exists to carry is the one being round-tripped.
        assert!(model.n_zero_coefficients() >= 1);

        let bytes = model.to_artifact(SCHEMA).expect("encode");
        assert_eq!(bytes, model.to_artifact(SCHEMA).expect("re-encode"));

        let restored = ElasticNet::from_artifact(&bytes, SCHEMA).expect("decode");
        assert_eq!(restored, model);
        assert_eq!(restored.n_iter(), model.n_iter());
        assert_eq!(restored.get_params().l1_ratio(), 0.75);
        assert_eq!(
            restored.predict(&data.as_view()).expect("predict"),
            model.predict(&data.as_view()).expect("predict")
        );
    }

    #[test]
    fn a_decoder_refuses_another_schema_and_another_estimators_bytes() {
        const SCHEMA: [u8; 32] = [7; 32];
        const OTHER: [u8; 32] = [9; 32];
        let (data, targets) = correlated_problem();
        let model = ElasticNet::fit(
            &data.as_view(),
            &targets,
            ElasticNetParams::default()
                .with_alpha(0.05)
                .with_l1_ratio(0.75)
                .with_tol(1.0e-8),
        )
        .expect("fit");
        let bytes = model.to_artifact(SCHEMA).expect("encode");

        assert_eq!(
            ElasticNet::from_artifact(&bytes, OTHER),
            Err(ArtifactError::FeatureSchemaMismatch)
        );

        let mut corrupted = bytes.clone();
        let last = corrupted.len() - 40;
        corrupted[last] ^= 1;
        assert_eq!(
            ElasticNet::from_artifact(&corrupted, SCHEMA),
            Err(ArtifactError::ChecksumMismatch)
        );

        // The kind is what keeps the two penalized readers off each other's
        // bytes: their payload layouts differ by one word, so a reader that
        // trusted the layout alone would misread every field after it.
        assert_eq!(
            Lasso::from_artifact(&bytes, SCHEMA).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 70 }
        );
    }
}
