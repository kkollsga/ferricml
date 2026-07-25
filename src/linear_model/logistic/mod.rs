//! Logistic regression: an asymmetric binary fit and a joint multinomial fit.

mod lbfgs;
mod multinomial;

use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, HasParams, ModelError,
    ProbabilisticClassifier, validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, LOGISTIC_ARTIFACT_KIND, MODEL_ARTIFACT_VERSION,
    SchemaRole, artifact_payload_version, artifact_version, decode_component,
    decode_legacy_envelope, decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::{BinaryTargets, ClassTargets, MatrixView, SampleWeights};
use crate::loss::{BinaryLogLoss, accumulate_newton_row, raw_score};
use crate::numeric::{sigmoid_f32, softmax_in_place_f32, sum_in_order};

const BINARY_CLASSES: [u8; 2] = [0, 1];
/// Every class label is a `u8`, so no fit can observe more classes than this.
/// Scalar and single-column prediction paths keep one row of scores on the
/// stack at this width rather than allocating inside an `_into` method.
const MAX_CLASSES: usize = 256;
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const LOGISTIC_FIXED_PAYLOAD_BYTES: usize = 8 * 4;
const LOGISTIC_PAYLOAD_VERSION: u16 = 1;
/// Payload schema for a joint multinomial fit: one class list, `K` intercepts,
/// and `K` coefficient rows. It is a second schema under the same estimator
/// kind rather than a widening of version 1, so every binary artifact ever
/// written keeps its exact bytes and its exact reader.
const LOGISTIC_MULTICLASS_PAYLOAD_VERSION: u16 = 2;
const LOGISTIC_STATE_COMPONENT_KIND: u16 = 1;
const LOGISTIC_STATE_COMPONENT_VERSION: u16 = 1;
const LOGISTIC_MULTICLASS_STATE_COMPONENT_VERSION: u16 = 2;
/// `n_features`, `fit_intercept`, `C`, `max_iter`, `tol`, `iterations`,
/// `class_count`, and `coefficient_count`.
const LOGISTIC_MULTICLASS_FIXED_PAYLOAD_BYTES: usize = 8 * 4;

/// The update rule a logistic fit uses to reach its coefficients.
///
/// Both solvers minimize the same L2-penalized objective and agree on its
/// minimizer; they differ in cost, in storage, and in what
/// [`with_tol`](LogisticRegressionParams::with_tol) measures.
///
/// # Choosing
///
/// [`Newton`](Self::Newton) is the default and stays the default. It takes the
/// exact second-order step, which converges in single-digit iterations, and it
/// is what every fitted artifact FerricML has ever written was produced by.
/// Its cost is one `parameters x parameters` factorization per iteration, so a
/// joint multinomial fit — whose system is `classes * parameters` square —
/// refuses rather than allocate one it cannot hold.
///
/// [`Lbfgs`](Self::Lbfgs) never forms that system. Its storage is linear in the
/// parameter count, so it accepts sixty-four times as many multiclass
/// parameters within the same storage budget, and it is the path to select when
/// the exact one reports [`ModelError::MulticlassSystemTooLarge`]. The bound is
/// a property of the selected solver rather than of the model: a shape the
/// exact path accepts still takes the exact path and produces the identical
/// fit. It needs more iterations to reach
/// the same answer, and it reports
/// [`ModelError::SolverDidNotConverge`] rather than returning an
/// unconverged model — both when `max_iter` runs out and when the requested
/// `tol` is below what the objective's own numerical resolution can certify,
/// which is around `1e-9` for a log-loss of order one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LogisticSolver {
    /// Exact Newton steps over the full second-order system. The default.
    #[default]
    Newton,
    /// Matrix-free limited-memory BFGS with a strong-Wolfe line search.
    Lbfgs,
}

/// Parameters for [`LogisticRegression`].
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegressionParams {
    c: f32,
    fit_intercept: bool,
    max_iter: usize,
    tol: f32,
    solver: LogisticSolver,
}

impl Default for LogisticRegressionParams {
    fn default() -> Self {
        Self {
            c: 1.0,
            fit_intercept: true,
            max_iter: 100,
            tol: 1.0e-4,
            solver: LogisticSolver::Newton,
        }
    }
}

impl LogisticRegressionParams {
    /// Sets inverse L2 regularization strength. Smaller values regularize more.
    #[must_use]
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Enables or disables the fitted intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Sets the maximum Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    ///
    /// What it measures depends on the selected [`LogisticSolver`], because the
    /// two solvers have genuinely different convergence tests and pretending
    /// otherwise would make one of them wrong. Under
    /// [`LogisticSolver::Newton`] it is the largest absolute coefficient
    /// update; under [`LogisticSolver::Lbfgs`] it is the infinity norm of the
    /// gradient of the mean penalized objective.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Selects the update rule used to reach the coefficients.
    ///
    /// The default is [`LogisticSolver::Newton`] and changing it changes the
    /// fitted coefficients in their last digits, so a model fitted under a
    /// non-default solver has no artifact representation — see
    /// [`LogisticRegression::to_artifact`].
    #[must_use]
    pub const fn with_solver(mut self, solver: LogisticSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Returns inverse L2 regularization strength.
    pub const fn c(&self) -> f32 {
        self.c
    }

    /// Returns whether an intercept is fitted.
    pub const fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Returns the maximum optimization iteration count.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    pub const fn tol(&self) -> f32 {
        self.tol
    }

    /// Returns the selected update rule.
    pub const fn solver(&self) -> LogisticSolver {
        self.solver
    }
}

/// L2-regularized logistic regression.
///
/// Fitting scales features internally for numerical conditioning and centers
/// them only when an intercept is requested. The transformation is folded into
/// the stored coefficients, so prediction is one allocation-free dot product
/// per score row.
///
/// # Two shapes, deliberately
///
/// A binary fit is **asymmetric**: it keeps one coefficient row, and
/// [`decision_function`](Self::decision_function) produces one score per row,
/// whose sigmoid is the probability of class `1`. A multiclass fit
/// ([`fit_multiclass`](Self::fit_multiclass)) is **joint** — one multinomial
/// optimization, not one binary model per class — and keeps one coefficient row
/// per class, so `decision_function` produces
/// [`n_decision_columns`](Self::n_decision_columns) scores per row whose softmax
/// is the probability vector. Those scores are *centred*: no class is pinned as
/// a reference, and each row of scores sums to approximately zero.
///
/// Probabilities are always `classes().len()` columns wide, so binary and
/// multiclass agree there and only the raw scores differ.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegression {
    n_features_in: usize,
    params: LogisticRegressionParams,
    /// Sorted observed labels; `[0, 1]` for every binary fit.
    classes: Vec<u8>,
    /// Row-major, `intercepts.len()` rows of `n_features_in` coefficients.
    coefficients: Vec<f32>,
    /// One intercept per score row; exactly one for a binary fit.
    intercepts: Vec<f32>,
    iterations: usize,
}

impl LogisticRegression {
    /// Fits a binary logistic classifier with deterministic Newton updates.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: LogisticRegressionParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits a binary logistic classifier with per-row sample weights.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: &SampleWeights,
        params: LogisticRegressionParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    /// Fits one joint multinomial classifier over the observed class set.
    ///
    /// This is a single optimization over all classes at once, not a wrapper
    /// around several binary fits: the probabilities are the softmax of one
    /// centred score vector, which is a different — and not recoverable —
    /// definition from normalizing independent per-class binary probabilities.
    ///
    /// Two observed classes are required. A single observed class is
    /// [`ModelError::RequiresTwoClasses`], exactly as the binary entry point
    /// already reports it: a one-class fit has no identifiable coefficients and
    /// would claim certainty it never measured.
    ///
    /// Two observed classes are accepted here and produce a *two-row* centred
    /// model, which is a different parametrization from [`Self::fit`]'s single
    /// asymmetric row. Both are correct; they are not interchangeable, and only
    /// [`Self::fit`] persists to an artifact today.
    pub fn fit_multiclass(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        params: LogisticRegressionParams,
    ) -> Result<Self, ModelError> {
        multinomial::fit(data, targets, None, params)
    }

    /// Fits a joint multinomial classifier with per-row sample weights.
    pub fn fit_multiclass_weighted(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        sample_weights: &SampleWeights,
        params: LogisticRegressionParams,
    ) -> Result<Self, ModelError> {
        multinomial::fit(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: Option<&SampleWeights>,
        params: LogisticRegressionParams,
    ) -> Result<Self, ModelError> {
        validate_fit(data, targets, sample_weights, &params)?;
        let rows = data.rows();
        let columns = data.columns();
        let total_weight = sample_weights.map_or(rows as f64, SampleWeights::total);
        let Standardization { means, scales } =
            standardize(data, sample_weights, total_weight, params.fit_intercept);

        let parameter_count = columns + usize::from(params.fit_intercept);
        let intercept_index = params.fit_intercept.then_some(columns);
        let design = build_design(data, &means, &scales, parameter_count, intercept_index);
        if params.solver == LogisticSolver::Lbfgs {
            let penalties = lbfgs::scaled_penalties(&scales, params.c);
            let (theta, iterations) = lbfgs::fit_binary(
                lbfgs::DesignView {
                    design: &design,
                    sample_weights,
                    penalties: &penalties,
                    columns,
                    parameter_count,
                    intercept_index,
                    inverse_total_weight: 1.0 / total_weight,
                },
                targets.as_slice(),
                &params,
            )?;
            return Self::from_standardized(
                &theta,
                &means,
                &scales,
                columns,
                intercept_index,
                BINARY_CLASSES.to_vec(),
                params,
                iterations,
            );
        }
        let mut theta = vec![0.0_f64; parameter_count];
        let mut gradient = vec![0.0_f64; parameter_count];
        let mut hessian = vec![0.0_f64; parameter_count * parameter_count];
        let mut update = vec![0.0_f64; parameter_count];
        let lambda = 1.0 / f64::from(params.c);
        let mut iterations = 0;
        for iteration in 0..params.max_iter {
            gradient.fill(0.0);
            hessian.fill(0.0);
            let mut accumulate_row = |design_row: &[f64], target: u8, sample_weight: f64| {
                let score = raw_score(&theta, design_row, columns, intercept_index);
                accumulate_newton_row::<BinaryLogLoss>(
                    design_row,
                    score,
                    f64::from(target),
                    sample_weight,
                    &mut gradient,
                    &mut hessian,
                );
            };
            if let Some(weights) = sample_weights {
                for ((design_row, &target), &sample_weight) in design
                    .chunks_exact(parameter_count)
                    .zip(targets.as_slice())
                    .zip(weights.as_slice())
                {
                    accumulate_row(design_row, target, f64::from(sample_weight));
                }
            } else {
                for (design_row, &target) in
                    design.chunks_exact(parameter_count).zip(targets.as_slice())
                {
                    accumulate_row(design_row, target, 1.0);
                }
            }
            for column in 0..columns {
                let scaled_penalty = lambda / (scales[column] * scales[column]);
                gradient[column] += scaled_penalty * theta[column];
                hessian[column * parameter_count + column] += scaled_penalty;
            }
            solve_positive_definite(&mut hessian, &gradient, &mut update, parameter_count)?;
            let max_update = update
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            for (value, &update) in theta.iter_mut().zip(&update) {
                *value -= update;
            }
            iterations = iteration + 1;
            if max_update <= f64::from(params.tol) {
                break;
            }
        }

        Self::from_standardized(
            &theta,
            &means,
            &scales,
            columns,
            intercept_index,
            BINARY_CLASSES.to_vec(),
            params,
            iterations,
        )
    }

    /// Builds a fitted model from coefficients in standardized design space.
    ///
    /// Fitting standardizes the design for conditioning; the model does not,
    /// because prediction has to be one dot product against the caller's own
    /// features. Undoing the transformation is therefore part of every fit, and
    /// stating it once means the binary and multinomial paths — and the two
    /// solvers — cannot drift into two different definitions of the same model.
    ///
    /// `theta` is one standardized coefficient row per score row, stacked in
    /// class order. The reduction that folds the centering into the intercept
    /// runs in ascending column order per the accumulation policy, and is what
    /// a fitted intercept's exact bits depend on.
    #[allow(clippy::too_many_arguments)]
    fn from_standardized(
        theta: &[f64],
        means: &[f64],
        scales: &[f64],
        columns: usize,
        intercept_index: Option<usize>,
        classes: Vec<u8>,
        params: LogisticRegressionParams,
        iterations: usize,
    ) -> Result<Self, ModelError> {
        let parameter_count = columns + usize::from(intercept_index.is_some());
        debug_assert!(parameter_count > 0 && theta.len().is_multiple_of(parameter_count));
        let score_rows = theta.len() / parameter_count;
        let mut coefficients = Vec::with_capacity(score_rows * columns);
        let mut intercepts = Vec::with_capacity(score_rows);
        for row in theta.chunks_exact(parameter_count) {
            for column in 0..columns {
                coefficients.push((row[column] / scales[column]) as f32);
            }
            let intercept = intercept_index.map_or(0.0, |index| row[index])
                - sum_in_order(
                    (0..columns).map(|column| row[column] * means[column] / scales[column]),
                );
            intercepts.push(intercept as f32);
        }
        // Checked after narrowing, so a coefficient that is finite at `f64` but
        // overflows the storage type is a failed fit rather than a stored
        // infinity that only shows up as a non-finite prediction later.
        if coefficients
            .iter()
            .chain(&intercepts)
            .any(|value| !value.is_finite())
        {
            return Err(ModelError::LinearSolveFailed);
        }
        Ok(Self {
            n_features_in: columns,
            params,
            classes,
            coefficients,
            intercepts,
            iterations,
        })
    }

    /// Returns fitted coefficients in input-feature order.
    ///
    /// A binary fit has exactly one row, so this is one coefficient per input
    /// feature. A multiclass fit stores
    /// [`n_decision_columns`](Self::n_decision_columns) rows of that width,
    /// row-major and ordered like [`classes`](Self::classes).
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Returns the fitted intercept of the first score row.
    ///
    /// This is *the* intercept of a binary fit. A multiclass fit has one per
    /// class; read them through [`intercepts`](Self::intercepts).
    pub fn intercept(&self) -> f32 {
        self.intercepts[0]
    }

    /// Returns one fitted intercept per score row, ordered like the
    /// coefficient rows.
    pub fn intercepts(&self) -> &[f32] {
        &self.intercepts
    }

    /// Returns the number of raw decision scores produced for each input row.
    ///
    /// One for a binary fit, one per class for a multiclass fit. This is the
    /// multiplier a caller sizes a [`decision_function_into`](
    /// Self::decision_function_into) buffer by; probability buffers use
    /// [`classes`](Self::classes)`.len()` instead, which differs from this for
    /// a binary fit.
    pub fn n_decision_columns(&self) -> usize {
        self.intercepts.len()
    }

    /// Returns sorted class labels observed during fitting.
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    /// Whether this model keeps the asymmetric single-row binary parametrization.
    #[inline]
    fn is_binary(&self) -> bool {
        self.intercepts.len() == 1
    }

    /// Rejects a scalar-valued request against a model whose value is a vector.
    fn require_scalar_output(&self) -> Result<(), ModelError> {
        if self.is_binary() {
            return Ok(());
        }
        Err(ModelError::MulticlassOutput {
            columns: self.intercepts.len(),
        })
    }

    /// Returns the number of optimization iterations performed.
    pub const fn n_iter(&self) -> usize {
        self.iterations
    }

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fit parameters.
    pub const fn get_params(&self) -> &LogisticRegressionParams {
        &self.params
    }

    /// Returns the raw linear decision score for one row.
    ///
    /// Defined only for a binary fit, which produces one score per row. A
    /// multiclass fit produces a vector, and reports
    /// [`ModelError::MulticlassOutput`] rather than silently returning one of
    /// its components.
    pub fn decision_function_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        self.require_scalar_output()?;
        validate_scalar_row(row, self.n_features_in)?;
        validate_prediction(self.decision_value(row), 0)
    }

    /// The single raw score of a binary fit.
    ///
    /// The intercept seeds the accumulator and the feature terms follow in
    /// ascending column order, matching the fitting-time reduction.
    fn decision_value(&self, row: &[f32]) -> f32 {
        row.iter()
            .zip(&self.coefficients)
            .fold(self.intercepts[0], |sum, (&value, &coefficient)| {
                sum + value * coefficient
            })
    }

    /// Writes one raw score per class into `scores`, in class order.
    fn decision_values_into(&self, row: &[f32], scores: &mut [f32]) {
        for (class, slot) in scores.iter_mut().enumerate() {
            let coefficients =
                &self.coefficients[class * self.n_features_in..(class + 1) * self.n_features_in];
            *slot = row
                .iter()
                .zip(coefficients)
                .fold(self.intercepts[class], |sum, (&value, &coefficient)| {
                    sum + value * coefficient
                });
        }
    }

    /// Returns raw linear decision scores, one row of
    /// [`n_decision_columns`](Self::n_decision_columns) values per input row.
    ///
    /// A binary fit therefore returns one value per row, and a multiclass fit
    /// returns a row-major matrix of centred scores.
    pub fn decision_function(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        let output_len = decision_output_len(data.rows(), self.n_decision_columns())?;
        let mut output = vec![0.0; output_len];
        self.decision_function_into(data, &mut output)?;
        Ok(output)
    }

    /// Writes raw linear decision scores into caller-owned storage.
    ///
    /// `output.len()` must equal
    /// `data.rows() * self.n_decision_columns()`.
    pub fn decision_function_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        if self.is_binary() {
            validate_predict(data, output.len(), self.n_features_in)?;
            for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
                *slot = validate_prediction(self.decision_value(row), row_index)?;
            }
            return Ok(());
        }
        let columns = self.n_decision_columns();
        validate_matrix_output(data, output.len(), columns, self.n_features_in)?;
        for (row_index, (row, scores)) in data
            .iter_rows()
            .zip(output.chunks_exact_mut(columns))
            .enumerate()
        {
            self.decision_values_into(row, scores);
            for score in scores {
                validate_prediction(*score, row_index)?;
            }
        }
        Ok(())
    }

    /// Predicts one positive-class probability.
    ///
    /// Defined only for a binary fit; see [`Self::decision_function_one`].
    pub fn predict_positive_proba(&self, row: &[f32]) -> Result<f32, ModelError> {
        Ok(sigmoid_f32(self.decision_function_one(row)?))
    }

    /// Predicts one label.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        if self.is_binary() {
            return Ok(u8::from(self.predict_positive_proba(row)? > 0.5));
        }
        validate_scalar_row(row, self.n_features_in)?;
        let mut storage = [0.0_f32; MAX_CLASSES];
        let scores = &mut storage[..self.classes.len()];
        self.decision_values_into(row, scores);
        for score in scores.iter() {
            validate_prediction(*score, 0)?;
        }
        softmax_in_place_f32(scores);
        Ok(self.classes[argmax(scores)])
    }

    /// Allocating label prediction convenience method.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Allocation-free label prediction.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        <Self as Classifier>::predict_into(self, data, output)
    }

    /// Allocating probability prediction convenience method.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba(self, data)
    }

    /// Allocation-free probability prediction.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba_into(self, data, output)
    }

    /// Predicts one requested class probability column.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_class_proba(self, data, class)
    }

    /// Encodes this model in FerricML's stable checksummed artifact format.
    ///
    /// Both fits persist. The two parametrizations are different models, so
    /// they are different payload schemas under one estimator kind: a binary
    /// fit writes payload version 1, which stores one coefficient row and one
    /// intercept, and a multiclass fit writes payload version 2, which stores
    /// the observed class list, one intercept per class, and one coefficient
    /// row per class. Decoding selects the reader from the recorded version, so
    /// neither can decode as the other.
    ///
    /// # Solver provenance
    ///
    /// Both payload schemas predate [`LogisticSolver`] and store no solver
    /// field, so widening either to carry one would change the bytes of every
    /// artifact ever written. A model fitted under a non-default solver
    /// therefore reports [`ArtifactError::UnsupportedModelState`] instead:
    /// writing it would produce bytes that decode as a model claiming Newton
    /// provenance it does not have, and a decoded model that compares unequal
    /// to the one encoded is worse than a refusal.
    pub fn to_artifact(&self, feature_schema_sha256: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        if self.params.solver != LogisticSolver::Newton {
            return Err(ArtifactError::UnsupportedModelState);
        }
        if self.is_binary() {
            self.to_binary_artifact(feature_schema_sha256)
        } else {
            self.to_multiclass_artifact(feature_schema_sha256)
        }
    }

    fn to_binary_artifact(
        &self,
        feature_schema_sha256: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        if self.n_features_in > MAX_ARTIFACT_FEATURES {
            return Err(ArtifactError::InvalidPayload);
        }
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let iterations =
            u32::try_from(self.iterations).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut state = ArtifactPayloadWriter::with_capacity(
            LOGISTIC_FIXED_PAYLOAD_BYTES + self.coefficients.len() * 4,
        );
        state.u32(n_features);
        state.u32(u32::from(self.params.fit_intercept));
        state.f32(self.params.c);
        state.u32(max_iter);
        state.f32(self.params.tol);
        state.u32(iterations);
        state.f32(self.intercepts[0]);
        state.u32(n_features);
        for &coefficient in &self.coefficients {
            state.f32(coefficient);
        }
        let component = encode_component(
            LOGISTIC_STATE_COMPONENT_KIND,
            LOGISTIC_STATE_COMPONENT_VERSION,
            &state.finish(),
        )?;
        encode_v2_envelope(
            LOGISTIC_ARTIFACT_KIND,
            LOGISTIC_PAYLOAD_VERSION,
            &[(SchemaRole::Input, feature_schema_sha256)],
            &component,
        )
    }

    /// Encodes a joint multinomial fit under payload version 2.
    ///
    /// The class list is stored explicitly: probability columns follow it, and
    /// a decoded model that guessed its labels would silently relabel every
    /// prediction. Labels are written as fixed-width words in the sorted order
    /// the fit observed, which is also the only order decode accepts, so one
    /// model has exactly one encoding.
    fn to_multiclass_artifact(
        &self,
        feature_schema_sha256: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        if self.n_features_in > MAX_ARTIFACT_FEATURES || self.classes.len() > MAX_CLASSES {
            return Err(ArtifactError::InvalidPayload);
        }
        debug_assert_eq!(self.classes.len(), self.intercepts.len());
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let iterations =
            u32::try_from(self.iterations).map_err(|_| ArtifactError::InvalidPayload)?;
        let class_count =
            u32::try_from(self.classes.len()).map_err(|_| ArtifactError::InvalidPayload)?;
        let coefficient_count =
            u32::try_from(self.coefficients.len()).map_err(|_| ArtifactError::InvalidPayload)?;
        if self.intercepts.len() != self.classes.len()
            || self.coefficients.len() != self.classes.len() * self.n_features_in
        {
            return Err(ArtifactError::UnsupportedModelState);
        }
        let mut state = ArtifactPayloadWriter::with_capacity(
            LOGISTIC_MULTICLASS_FIXED_PAYLOAD_BYTES
                + self.classes.len() * 8
                + self.coefficients.len() * 4,
        );
        state.u32(n_features);
        state.u32(u32::from(self.params.fit_intercept));
        state.f32(self.params.c);
        state.u32(max_iter);
        state.f32(self.params.tol);
        state.u32(iterations);
        state.u32(class_count);
        state.u32(coefficient_count);
        for &class in &self.classes {
            state.u32(u32::from(class));
        }
        for &intercept in &self.intercepts {
            state.f32(intercept);
        }
        for &coefficient in &self.coefficients {
            state.f32(coefficient);
        }
        let component = encode_component(
            LOGISTIC_STATE_COMPONENT_KIND,
            LOGISTIC_MULTICLASS_STATE_COMPONENT_VERSION,
            &state.finish(),
        )?;
        encode_v2_envelope(
            LOGISTIC_ARTIFACT_KIND,
            LOGISTIC_MULTICLASS_PAYLOAD_VERSION,
            &[(SchemaRole::Input, feature_schema_sha256)],
            &component,
        )
    }

    /// Decodes a logistic model after checking integrity and feature identity.
    ///
    /// The recorded payload version selects the reader before anything is
    /// hashed or parsed, so a binary artifact and a multiclass artifact never
    /// reach each other's parser. Every field that selection depends on is
    /// re-read and re-validated by the reader it selected.
    pub fn from_artifact(
        bytes: &[u8],
        expected_feature_schema_sha256: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        if artifact_version(bytes)? == MODEL_ARTIFACT_VERSION
            && artifact_payload_version(bytes)? == LOGISTIC_MULTICLASS_PAYLOAD_VERSION
        {
            return Self::from_multiclass_artifact(bytes, expected_feature_schema_sha256);
        }
        let cursor = if artifact_version(bytes)? == MODEL_ARTIFACT_VERSION {
            let mut envelope = decode_v2_envelope(
                bytes,
                LOGISTIC_ARTIFACT_KIND,
                LOGISTIC_PAYLOAD_VERSION,
                &[(SchemaRole::Input, expected_feature_schema_sha256)],
            )?;
            let component = decode_component(
                &mut envelope,
                LOGISTIC_STATE_COMPONENT_KIND,
                LOGISTIC_STATE_COMPONENT_VERSION,
            )?;
            if !envelope.is_empty() {
                return Err(ArtifactError::TrailingBytes);
            }
            component
        } else {
            decode_legacy_envelope(
                bytes,
                LOGISTIC_ARTIFACT_KIND,
                expected_feature_schema_sha256,
                LOGISTIC_FIXED_PAYLOAD_BYTES,
            )?
        };
        Self::decode_payload(cursor)
    }

    fn from_multiclass_artifact(
        bytes: &[u8],
        expected_feature_schema_sha256: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            LOGISTIC_ARTIFACT_KIND,
            LOGISTIC_MULTICLASS_PAYLOAD_VERSION,
            &[(SchemaRole::Input, expected_feature_schema_sha256)],
        )?;
        let mut cursor = decode_component(
            &mut envelope,
            LOGISTIC_STATE_COMPONENT_KIND,
            LOGISTIC_MULTICLASS_STATE_COMPONENT_VERSION,
        )?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }

        let n_features_in = cursor.u32()? as usize;
        let fit_intercept = match cursor.u32()? {
            0 => false,
            1 => true,
            _ => return Err(ArtifactError::InvalidPayload),
        };
        let c = cursor.f32()?;
        let max_iter = cursor.u32()? as usize;
        let tol = cursor.f32()?;
        let iterations = cursor.u32()? as usize;
        let class_count = cursor.u32()? as usize;
        let coefficient_count = cursor.u32()? as usize;
        let expected_coefficients = class_count.checked_mul(n_features_in);
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            // A multinomial fit needs two observed classes, so a one-row model
            // is a binary artifact wearing the wrong schema.
            || !(2..=MAX_CLASSES).contains(&class_count)
            || expected_coefficients != Some(coefficient_count)
            || !c.is_finite()
            || c <= 0.0
            || max_iter == 0
            || !tol.is_finite()
            || tol <= 0.0
            || iterations == 0
            || iterations > max_iter
        {
            return Err(ArtifactError::InvalidPayload);
        }

        let mut classes: Vec<u8> = Vec::with_capacity(cursor.bounded_capacity(class_count, 4));
        for _ in 0..class_count {
            let label = u8::try_from(cursor.u32()?).map_err(|_| ArtifactError::InvalidPayload)?;
            // Sorted and deduplicated is the only accepted order, so a model
            // has exactly one encoding and its probability columns cannot be
            // permuted by a rewrite.
            if classes.last().is_some_and(|&previous| previous >= label) {
                return Err(ArtifactError::InvalidPayload);
            }
            classes.push(label);
        }
        let mut intercepts = Vec::with_capacity(cursor.bounded_capacity(class_count, 4));
        for _ in 0..class_count {
            let value = cursor.f32()?;
            if !value.is_finite() {
                return Err(ArtifactError::InvalidPayload);
            }
            intercepts.push(value);
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
            params: LogisticRegressionParams {
                c,
                fit_intercept,
                max_iter,
                tol,
                // No payload schema carries a solver, and `to_artifact`
                // refuses to write a model fitted under a non-default one, so
                // every decodable artifact has Newton provenance by
                // construction rather than by assumption.
                solver: LogisticSolver::Newton,
            },
            classes,
            coefficients,
            intercepts,
            iterations,
        })
    }

    fn decode_payload(
        mut cursor: crate::artifact::ArtifactCursor<'_>,
    ) -> Result<Self, ArtifactError> {
        let n_features_in = cursor.u32()? as usize;
        let fit_intercept = match cursor.u32()? {
            0 => false,
            1 => true,
            _ => return Err(ArtifactError::InvalidPayload),
        };
        let c = cursor.f32()?;
        let max_iter = cursor.u32()? as usize;
        let tol = cursor.f32()?;
        let iterations = cursor.u32()? as usize;
        let intercept = cursor.f32()?;
        let coefficient_count = cursor.u32()? as usize;
        if n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || coefficient_count != n_features_in
            || !c.is_finite()
            || c <= 0.0
            || max_iter == 0
            || !tol.is_finite()
            || tol <= 0.0
            || iterations == 0
            || iterations > max_iter
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
            params: LogisticRegressionParams {
                c,
                fit_intercept,
                max_iter,
                tol,
                // No payload schema carries a solver, and `to_artifact`
                // refuses to write a model fitted under a non-default one, so
                // every decodable artifact has Newton provenance by
                // construction rather than by assumption.
                solver: LogisticSolver::Newton,
            },
            classes: BINARY_CLASSES.to_vec(),
            coefficients,
            intercepts: vec![intercept],
            iterations,
        })
    }
}

impl Estimator for LogisticRegression {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for LogisticRegression {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true)
        .with_multiclass(true)
        .with_decision_function(true)
        .with_probability(true);
}

impl HasParams for LogisticRegression {
    type Params = LogisticRegressionParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Classifier for LogisticRegression {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        validate_predict(data, output.len(), self.n_features_in)?;
        if self.is_binary() {
            for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
                let decision = validate_prediction(self.decision_value(row), row_index)?;
                *slot = u8::from(sigmoid_f32(decision) > 0.5);
            }
            return Ok(());
        }
        let mut storage = [0.0_f32; MAX_CLASSES];
        let scores = &mut storage[..self.classes.len()];
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            self.decision_values_into(row, scores);
            for score in scores.iter() {
                validate_prediction(*score, row_index)?;
            }
            // Argmax of the probabilities, not of the scores: taking it from
            // the same values the caller can read is what makes a label and its
            // probability row impossible to disagree.
            softmax_in_place_f32(scores);
            *slot = self.classes[argmax(scores)];
        }
        Ok(())
    }
}

impl ProbabilisticClassifier for LogisticRegression {
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let columns = self.classes.len();
        validate_matrix_output(data, output.len(), columns, self.n_features_in)?;
        if self.is_binary() {
            for (row_index, (row, probabilities)) in data
                .iter_rows()
                .zip(output.chunks_exact_mut(columns))
                .enumerate()
            {
                let decision = validate_prediction(self.decision_value(row), row_index)?;
                let positive = sigmoid_f32(decision);
                probabilities[0] = 1.0 - positive;
                probabilities[1] = positive;
            }
            return Ok(());
        }
        for (row_index, (row, probabilities)) in data
            .iter_rows()
            .zip(output.chunks_exact_mut(columns))
            .enumerate()
        {
            self.decision_values_into(row, probabilities);
            for score in probabilities.iter() {
                validate_prediction(*score, row_index)?;
            }
            softmax_in_place_f32(probabilities);
        }
        Ok(())
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let column = self
            .classes
            .binary_search(&class)
            .map_err(|_| ModelError::UnknownClass { class })?;
        validate_predict(data, output.len(), self.n_features_in)?;
        if self.is_binary() {
            for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
                let decision = validate_prediction(self.decision_value(row), row_index)?;
                let positive = sigmoid_f32(decision);
                *slot = if column == 1 {
                    positive
                } else {
                    1.0 - positive
                };
            }
            return Ok(());
        }
        let mut storage = [0.0_f32; MAX_CLASSES];
        let scores = &mut storage[..self.classes.len()];
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            self.decision_values_into(row, scores);
            for score in scores.iter() {
                validate_prediction(*score, row_index)?;
            }
            softmax_in_place_f32(scores);
            *slot = scores[column];
        }
        Ok(())
    }
}

/// Index of the largest value, with an exact tie going to the lowest index.
///
/// Class labels are sorted, so "lowest index" is "smallest tied label" — which
/// is not the same as "the first class": with classes `[5, 9, 20]` a tie
/// between the last two selects `9`.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

/// Shared length, weight, and parameter validation for every fitting entry
/// point, in the order the frozen contract reports them.
fn validate_common_fit(
    data: &MatrixView<'_>,
    target_len: usize,
    sample_weights: Option<&SampleWeights>,
    params: &LogisticRegressionParams,
) -> Result<(), ModelError> {
    if data.rows() != target_len {
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
    if !params.c.is_finite() || params.c <= 0.0 {
        return Err(ModelError::InvalidRegularization);
    }
    if params.max_iter == 0 {
        return Err(ModelError::InvalidIterationCount);
    }
    if !params.tol.is_finite() || params.tol <= 0.0 {
        return Err(ModelError::InvalidTolerance);
    }
    Ok(())
}

fn validate_fit(
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    sample_weights: Option<&SampleWeights>,
    params: &LogisticRegressionParams,
) -> Result<(), ModelError> {
    validate_common_fit(data, targets.len(), sample_weights, params)?;
    if !targets.as_slice().contains(&0) || !targets.as_slice().contains(&1) {
        return Err(ModelError::RequiresTwoClasses);
    }
    Ok(())
}

fn sample_weight(sample_weights: Option<&SampleWeights>, row: usize) -> f64 {
    sample_weights.map_or(1.0, |weights| f64::from(weights.as_slice()[row]))
}

/// The per-column centering and scaling applied to the design matrix.
///
/// Centering happens only when an intercept is fitted, and a column with no
/// spread scales by one rather than by zero.
struct Standardization {
    means: Vec<f64>,
    scales: Vec<f64>,
}

fn standardize(
    data: &MatrixView<'_>,
    sample_weights: Option<&SampleWeights>,
    total_weight: f64,
    fit_intercept: bool,
) -> Standardization {
    let columns = data.columns();
    let mut means = vec![0.0_f64; columns];
    if fit_intercept {
        for (row_index, row) in data.iter_rows().enumerate() {
            let sample_weight = sample_weight(sample_weights, row_index);
            for (column, &value) in row.iter().enumerate() {
                means[column] += sample_weight * f64::from(value);
            }
        }
        for mean in &mut means {
            *mean /= total_weight;
        }
    }
    let mut scales = vec![0.0_f64; columns];
    for (row_index, row) in data.iter_rows().enumerate() {
        let sample_weight = sample_weight(sample_weights, row_index);
        for (column, &value) in row.iter().enumerate() {
            let centered = f64::from(value) - means[column];
            scales[column] += sample_weight * centered * centered;
        }
    }
    for scale in &mut scales {
        *scale = (*scale / total_weight).sqrt();
        if *scale <= f64::EPSILON {
            *scale = 1.0;
        }
    }
    Standardization { means, scales }
}

fn build_design(
    data: &MatrixView<'_>,
    means: &[f64],
    scales: &[f64],
    parameter_count: usize,
    intercept_index: Option<usize>,
) -> Vec<f64> {
    let columns = data.columns();
    let mut design = vec![0.0_f64; data.rows() * parameter_count];
    for (row, design_row) in data
        .iter_rows()
        .zip(design.chunks_exact_mut(parameter_count))
    {
        for column in 0..columns {
            design_row[column] = (f64::from(row[column]) - means[column]) / scales[column];
        }
        if let Some(index) = intercept_index {
            design_row[index] = 1.0;
        }
    }
    design
}

fn decision_output_len(rows: usize, columns: usize) -> Result<usize, ModelError> {
    rows.checked_mul(columns)
        .ok_or(ModelError::OutputShapeOverflow { rows, columns })
}

/// Validates a caller-owned `rows x columns` output buffer before any write.
///
/// The shape overflow is reported before the feature width, and the feature
/// width before the length, which is the order the frozen contract states.
fn validate_matrix_output(
    data: &MatrixView<'_>,
    output_len: usize,
    columns: usize,
    features: usize,
) -> Result<(), ModelError> {
    let expected = decision_output_len(data.rows(), columns)?;
    validate_feature_width(data, features)?;
    if output_len != expected {
        return Err(ModelError::OutputLength {
            expected,
            actual: output_len,
        });
    }
    Ok(())
}

fn validate_predict(
    data: &MatrixView<'_>,
    output_len: usize,
    features: usize,
) -> Result<(), ModelError> {
    validate_feature_width(data, features)?;
    if output_len != data.rows() {
        return Err(ModelError::OutputLength {
            expected: data.rows(),
            actual: output_len,
        });
    }
    Ok(())
}

fn validate_feature_width(data: &MatrixView<'_>, features: usize) -> Result<(), ModelError> {
    if data.columns() != features {
        return Err(ModelError::FeatureDimension {
            expected: features,
            actual: data.columns(),
        });
    }
    Ok(())
}

fn solve_positive_definite(
    matrix: &mut [f64],
    right: &[f64],
    solution: &mut [f64],
    size: usize,
) -> Result<(), ModelError> {
    for row in 0..size {
        for column in 0..=row {
            let mut value = matrix[row * size + column];
            for index in 0..column {
                value -= matrix[row * size + index] * matrix[column * size + index];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(ModelError::LinearSolveFailed);
                }
                matrix[row * size + column] = value.sqrt();
            } else {
                matrix[row * size + column] = value / matrix[column * size + column];
            }
        }
    }
    solution.copy_from_slice(right);
    for row in 0..size {
        for column in 0..row {
            solution[row] -= matrix[row * size + column] * solution[column];
        }
        solution[row] /= matrix[row * size + row];
    }
    for row in (0..size).rev() {
        for column in row + 1..size {
            solution[row] -= matrix[column * size + row] * solution[column];
        }
        solution[row] /= matrix[row * size + row];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DenseMatrix, SampleWeights};
    use sha2::{Digest, Sha256};

    fn simple_data() -> (DenseMatrix, BinaryTargets) {
        (
            DenseMatrix::new(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], 6, 1).unwrap(),
            BinaryTargets::new(vec![0, 0, 0, 1, 1, 1]).unwrap(),
        )
    }

    #[test]
    fn weighted_and_intercept_fit_bits_are_frozen() {
        let data = DenseMatrix::new(
            vec![0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0, 1.0, 5.0, 3.0],
            6,
            2,
        )
        .unwrap();
        let targets = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1]).unwrap();
        let weights = SampleWeights::new(vec![1.0, 2.0, 0.5, 1.5, 3.0, 2.0]).unwrap();
        let cases = [
            (
                false,
                true,
                [1_065_531_399, 1_055_929_814],
                3_225_976_169,
                5,
            ),
            (false, false, [1_052_252_415, 3_160_061_965], 0, 4),
            (true, true, [1_067_926_424, 1_055_849_859], 3_228_471_089, 5),
            (true, false, [1_057_229_716, 3_189_823_142], 0, 5),
        ];
        for (weighted, fit_intercept, expected_coefficients, expected_intercept, expected_iter) in
            cases
        {
            let params = LogisticRegressionParams::default()
                .with_fit_intercept(fit_intercept)
                .with_max_iter(25);
            let model = if weighted {
                LogisticRegression::fit_weighted(&data.as_view(), &targets, &weights, params)
                    .unwrap()
            } else {
                LogisticRegression::fit(&data.as_view(), &targets, params).unwrap()
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
            assert_eq!(model.n_iter(), expected_iter);
        }
    }

    fn phase_zero_artifact() -> (LogisticRegression, [u8; 32], Vec<u8>) {
        let model = LogisticRegression {
            n_features_in: 2,
            params: LogisticRegressionParams {
                c: 2.0,
                fit_intercept: true,
                max_iter: 100,
                tol: f32::from_bits(0x3586_37bd),
                solver: LogisticSolver::Newton,
            },
            classes: BINARY_CLASSES.to_vec(),
            coefficients: vec![f32::from_bits(0x3fb0_56aa), f32::from_bits(0x3eab_d102)],
            intercepts: vec![f32::from_bits(0xc00e_fe0f)],
            iterations: 5,
        };
        let bytes = decode_hex(
            "4645525249434d4c01000100070707070707070707070707070707070707070707070707070707070707070702000000010000000000004064000000bd378635050000000ffe0ec002000000aa56b03f02d1ab3e6d72aa073218a54d30e6d6e5fc5d19b0a8a4e0726ac51369976bfa79a3ae9ec3",
        );
        (model, [7; 32], bytes)
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert_eq!(value.len() % 2, 0);
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let text = std::str::from_utf8(pair).expect("ASCII hex");
                u8::from_str_radix(text, 16).expect("valid hex")
            })
            .collect()
    }

    fn resign_legacy_artifact(bytes: &mut [u8]) {
        let payload_len = bytes.len() - 32;
        let checksum = Sha256::digest(&bytes[..payload_len]);
        bytes[payload_len..].copy_from_slice(&checksum);
    }

    #[test]
    fn separates_a_simple_binary_problem() {
        let (data, targets) = simple_data();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default().with_c(10.0),
        )
        .unwrap();
        assert_eq!(model.predict(&data.as_view()).unwrap(), targets.as_slice());
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        assert!(
            probabilities
                .chunks_exact(2)
                .all(|row| (row[0] + row[1] - 1.0).abs() < 1.0e-6)
        );
        assert!(model.n_iter() > 0);
    }

    #[test]
    fn validates_parameters_classes_and_output() {
        let (data, targets) = simple_data();
        assert_eq!(
            LogisticRegression::fit(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default().with_c(0.0),
            )
            .unwrap_err(),
            ModelError::InvalidRegularization
        );
        let one_class = BinaryTargets::new(vec![0; 6]).unwrap();
        assert_eq!(
            LogisticRegression::fit(
                &data.as_view(),
                &one_class,
                LogisticRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::RequiresTwoClasses
        );
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        assert_eq!(
            model.predict_into(&data.as_view(), &mut [0; 2]),
            Err(ModelError::OutputLength {
                expected: 6,
                actual: 2
            })
        );
    }

    #[test]
    fn fit_is_deterministic_and_handles_constant_columns() {
        let data = DenseMatrix::new(vec![1.0, -2.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2.0], 4, 2).unwrap();
        let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let left = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let right = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        assert_eq!(left, right);
        assert_eq!(left.coefficients()[0], 0.0);
    }

    #[test]
    fn matches_frozen_logistic_reference_fixture() {
        let train = DenseMatrix::new(
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 3.0, 0.0, 3.0, 1.0,
            ],
            8,
            2,
        )
        .unwrap();
        let targets = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1, 1, 0]).unwrap();
        let test = DenseMatrix::new(
            vec![-1.0, 0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 0.5, 4.0, 0.5],
            5,
            2,
        )
        .unwrap();
        let model = LogisticRegression::fit(
            &train.as_view(),
            &targets,
            LogisticRegressionParams::default().with_tol(1.0e-8),
        )
        .unwrap();
        let expected = [
            0.815_812_1,
            0.184_187_87,
            0.644_578_2,
            0.355_421_75,
            0.5,
            0.5,
            0.355_421_78,
            0.644_578_2,
            0.184_187_83,
            0.815_812_2,
        ];
        let actual = model.predict_proba(&test.as_view()).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "{actual} != {expected}"
            );
        }
        assert!((model.coefficients()[0] - 0.595_291_14).abs() <= 2.0e-5);
        assert!(model.coefficients()[1].abs() <= 2.0e-5);
        assert!((model.intercept() + 0.892_936_77).abs() <= 2.0e-5);
    }

    #[test]
    fn mirrored_rows_have_opposite_no_intercept_scores() {
        let data =
            DenseMatrix::new(vec![2.0, 1.0, -2.0, -1.0, 1.0, -1.0, -1.0, 1.0], 4, 2).unwrap();
        let targets = BinaryTargets::new(vec![1, 0, 1, 0]).unwrap();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default()
                .with_fit_intercept(false)
                .with_tol(1.0e-8),
        )
        .unwrap();

        assert_eq!(model.intercept().to_bits(), 0.0_f32.to_bits());
        for rows in data.as_slice().chunks_exact(4) {
            let positive = model.predict_positive_proba(&rows[..2]).unwrap();
            let mirrored = model.predict_positive_proba(&rows[2..]).unwrap();
            assert!((positive + mirrored - 1.0).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn uniform_weights_are_bit_equivalent_and_scaled_weights_change_regularization() {
        let (data, targets) = simple_data();
        let params = LogisticRegressionParams::default().with_tol(1.0e-8);
        let unweighted =
            LogisticRegression::fit(&data.as_view(), &targets, params.clone()).unwrap();
        let uniform = LogisticRegression::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(vec![1.0; data.rows()]).unwrap(),
            params.clone(),
        )
        .unwrap();
        assert_eq!(uniform, unweighted);

        let scaled = LogisticRegression::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(vec![10.0; data.rows()]).unwrap(),
            params,
        )
        .unwrap();
        assert!(scaled.coefficients()[0].abs() > unweighted.coefficients()[0].abs());
    }

    #[test]
    fn integer_weights_match_replicated_rows() {
        let (data, targets) = simple_data();
        let weights = [1_u8, 2, 1, 3, 1, 2];
        let weighted = LogisticRegression::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(weights.iter().map(|&value| f32::from(value)).collect()).unwrap(),
            LogisticRegressionParams::default().with_tol(1.0e-8),
        )
        .unwrap();

        let mut replicated_data = Vec::new();
        let mut replicated_targets = Vec::new();
        for ((row, &target), &count) in data.iter_rows().zip(targets.as_slice()).zip(&weights) {
            for _ in 0..count {
                replicated_data.extend_from_slice(row);
                replicated_targets.push(target);
            }
        }
        let replicated_data =
            DenseMatrix::new(replicated_data, replicated_targets.len(), data.columns()).unwrap();
        let replicated_targets = BinaryTargets::new(replicated_targets).unwrap();
        let replicated = LogisticRegression::fit(
            &replicated_data.as_view(),
            &replicated_targets,
            LogisticRegressionParams::default().with_tol(1.0e-8),
        )
        .unwrap();

        assert!((weighted.intercept() - replicated.intercept()).abs() <= 1.0e-6);
        for (&weighted, &replicated) in weighted
            .coefficients()
            .iter()
            .zip(replicated.coefficients())
        {
            assert!((weighted - replicated).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn decision_scores_validate_before_writing_and_drive_probabilities() {
        let (data, targets) = simple_data();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let scores = model.decision_function(&data.as_view()).unwrap();
        let mut output = vec![0.0; data.rows()];
        model
            .decision_function_into(&data.as_view(), &mut output)
            .unwrap();
        assert_eq!(output, scores);
        for ((row, &score), probability) in data
            .iter_rows()
            .zip(&scores)
            .zip(model.predict_class_proba(&data.as_view(), 1).unwrap())
        {
            assert_eq!(model.decision_function_one(row).unwrap(), score);
            assert_eq!(sigmoid_f32(score), probability);
        }

        let mut untouched = [9.0; 2];
        assert_eq!(
            model
                .decision_function_into(&data.as_view(), &mut untouched)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: data.rows(),
                actual: 2,
            }
        );
        assert_eq!(untouched, [9.0; 2]);
    }

    #[test]
    fn weighted_fit_rejects_row_count_mismatch() {
        let (data, targets) = simple_data();
        let weights = SampleWeights::new(vec![1.0; data.rows() - 1]).unwrap();
        assert_eq!(
            LogisticRegression::fit_weighted(
                &data.as_view(),
                &targets,
                &weights,
                LogisticRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::SampleWeightLength {
                rows: data.rows(),
                weights: data.rows() - 1,
            }
        );
    }

    #[test]
    fn artifact_round_trip_checks_schema_and_corruption() {
        let (data, targets) = simple_data();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let schema = [7_u8; 32];
        let bytes = model.to_artifact(schema).unwrap();
        let decoded = LogisticRegression::from_artifact(&bytes, schema).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(
            decoded.predict_proba(&data.as_view()).unwrap(),
            model.predict_proba(&data.as_view()).unwrap()
        );
        assert_eq!(
            LogisticRegression::from_artifact(&bytes, [8; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );

        let mut corrupted = bytes.clone();
        corrupted[20] ^= 1;
        assert_eq!(
            LogisticRegression::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
        assert_eq!(
            LogisticRegression::from_artifact(&bytes[..40], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }

    #[test]
    fn legacy_artifact_fixture_decodes_exactly() {
        let (model, schema, expected) = phase_zero_artifact();
        assert_eq!(expected.len(), 116);
        assert_eq!(
            LogisticRegression::from_artifact(&expected, schema).unwrap(),
            model
        );
    }

    #[test]
    fn v2_artifact_bytes_are_deterministic_and_round_trip() {
        let (model, schema, _) = phase_zero_artifact();
        let left = model.to_artifact(schema).unwrap();
        let right = model.to_artifact(schema).unwrap();
        let expected = decode_hex(
            "4645525249434d4c02000100010000003000000001000000010000000707070707070707070707070707070707070707070707070707070707070707010001002800000002000000010000000000004064000000bd378635050000000ffe0ec002000000aa56b03f02d1ab3ea6361781d561733ab80f4fd31372ffecb2f42c7dfc24050b0d11f7b524d7a90f",
        );
        assert_eq!(left, right);
        assert_eq!(left, expected);
        assert_eq!(left.len(), 140);
        assert_eq!(&left[..8], b"FERRICML");
        assert_eq!(u16::from_le_bytes(left[8..10].try_into().unwrap()), 2);
        assert_eq!(
            LogisticRegression::from_artifact(&left, schema).unwrap(),
            model
        );
    }

    #[test]
    fn legacy_artifact_error_precedence_and_payload_validation_are_frozen() {
        let (_, schema, bytes) = phase_zero_artifact();

        let mut invalid_magic_without_checksum = bytes.clone();
        invalid_magic_without_checksum[0] ^= 1;
        assert_eq!(
            LogisticRegression::from_artifact(&invalid_magic_without_checksum, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );

        let mut invalid_magic = bytes.clone();
        invalid_magic[0] ^= 1;
        resign_legacy_artifact(&mut invalid_magic);
        assert_eq!(
            LogisticRegression::from_artifact(&invalid_magic, schema).unwrap_err(),
            ArtifactError::InvalidMagic
        );

        let mut unsupported_version = bytes.clone();
        unsupported_version[8..10].copy_from_slice(&3_u16.to_le_bytes());
        resign_legacy_artifact(&mut unsupported_version);
        assert_eq!(
            LogisticRegression::from_artifact(&unsupported_version, schema).unwrap_err(),
            ArtifactError::UnsupportedVersion { found: 3 }
        );

        let mut unsupported_kind = bytes.clone();
        unsupported_kind[10..12].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut unsupported_kind);
        assert_eq!(
            LogisticRegression::from_artifact(&unsupported_kind, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 2 }
        );

        let mut invalid_flag = bytes.clone();
        invalid_flag[48..52].copy_from_slice(&2_u32.to_le_bytes());
        resign_legacy_artifact(&mut invalid_flag);
        assert_eq!(
            LogisticRegression::from_artifact(&invalid_flag, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut count_mismatch = bytes.clone();
        count_mismatch[72..76].copy_from_slice(&1_u32.to_le_bytes());
        resign_legacy_artifact(&mut count_mismatch);
        assert_eq!(
            LogisticRegression::from_artifact(&count_mismatch, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut non_finite = bytes.clone();
        non_finite[68..72].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        resign_legacy_artifact(&mut non_finite);
        assert_eq!(
            LogisticRegression::from_artifact(&non_finite, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut trailing = bytes[..bytes.len() - 32].to_vec();
        trailing.push(0);
        trailing.extend_from_slice(&[0; 32]);
        resign_legacy_artifact(&mut trailing);
        assert_eq!(
            LogisticRegression::from_artifact(&trailing, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        assert_eq!(
            LogisticRegression::from_artifact(&bytes[..107], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }

    #[test]
    fn v2_artifact_rejects_unknown_metadata_and_bad_framing() {
        let (model, schema, _) = phase_zero_artifact();
        let bytes = model.to_artifact(schema).unwrap();

        let mut flags_without_checksum = bytes.clone();
        flags_without_checksum[14..16].copy_from_slice(&1_u16.to_le_bytes());
        assert_eq!(
            LogisticRegression::from_artifact(&flags_without_checksum, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );

        let mut flags = flags_without_checksum;
        resign_legacy_artifact(&mut flags);
        assert_eq!(
            LogisticRegression::from_artifact(&flags, schema).unwrap_err(),
            ArtifactError::UnsupportedRequiredFlags { found: 1 }
        );

        let mut payload_version = bytes.clone();
        payload_version[12..14].copy_from_slice(&9_u16.to_le_bytes());
        resign_legacy_artifact(&mut payload_version);
        assert_eq!(
            LogisticRegression::from_artifact(&payload_version, schema).unwrap_err(),
            ArtifactError::UnsupportedPayloadVersion { found: 9 }
        );

        // Payload version 2 is the multiclass schema. Relabelling a binary
        // artifact into it selects that reader, which rejects the version-1
        // state component it finds — the two schemas cannot be crossed by
        // rewriting one header field.
        let mut multiclass_version = bytes.clone();
        multiclass_version[12..14].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut multiclass_version);
        assert_eq!(
            LogisticRegression::from_artifact(&multiclass_version, schema).unwrap_err(),
            ArtifactError::UnsupportedPayloadVersion { found: 1 }
        );

        let mut schema_role = bytes.clone();
        schema_role[24..26].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut schema_role);
        assert_eq!(
            LogisticRegression::from_artifact(&schema_role, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut component_kind = bytes.clone();
        component_kind[60..62].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut component_kind);
        assert_eq!(
            LogisticRegression::from_artifact(&component_kind, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut component_version = bytes.clone();
        component_version[62..64].copy_from_slice(&2_u16.to_le_bytes());
        resign_legacy_artifact(&mut component_version);
        assert_eq!(
            LogisticRegression::from_artifact(&component_version, schema).unwrap_err(),
            ArtifactError::UnsupportedPayloadVersion { found: 2 }
        );

        let mut short_payload = bytes.clone();
        short_payload[16..20].copy_from_slice(&47_u32.to_le_bytes());
        resign_legacy_artifact(&mut short_payload);
        assert_eq!(
            LogisticRegression::from_artifact(&short_payload, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        let mut long_payload = bytes.clone();
        long_payload[16..20].copy_from_slice(&49_u32.to_le_bytes());
        resign_legacy_artifact(&mut long_payload);
        assert_eq!(
            LogisticRegression::from_artifact(&long_payload, schema).unwrap_err(),
            ArtifactError::Truncated
        );

        let mut trailing = bytes[..bytes.len() - 32].to_vec();
        trailing.push(0);
        trailing.extend_from_slice(&[0; 32]);
        resign_legacy_artifact(&mut trailing);
        assert_eq!(
            LogisticRegression::from_artifact(&trailing, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        let mut oversized = vec![0_u8; crate::artifact::MAX_MODEL_ARTIFACT_BYTES + 1];
        oversized[8..10].copy_from_slice(&MODEL_ARTIFACT_VERSION.to_le_bytes());
        assert_eq!(
            LogisticRegression::from_artifact(&oversized, schema).unwrap_err(),
            ArtifactError::SizeLimitExceeded {
                limit: crate::artifact::MAX_MODEL_ARTIFACT_BYTES,
                actual: oversized.len(),
            }
        );
    }
}
