//! The matrix-free logistic fit behind [`super::LogisticSolver::Lbfgs`].
//!
//! # Why a second solver exists
//!
//! Newton's method is the right update rule for a binary logistic fit over a
//! handful of features: it converges in single-digit iterations and its cost is
//! dominated by one `parameters x parameters` factorization. It stops being
//! right as the system grows. A joint multinomial fit stacks that system to
//! `(classes * parameters)` square, so its storage grows with the *fourth*
//! power of the feature count times the square of the class count — which is
//! why the exact path refuses above
//! [`MAX_NEWTON_PARAMETERS`](super::multinomial::MAX_NEWTON_PARAMETERS)
//! rather than allocating something that cannot fit.
//!
//! Limited-memory BFGS never forms that system. It needs the objective's value
//! and gradient and a fixed number of stored vector pairs, so its storage is
//! linear in the parameter count.
//!
//! # What this module owns and what it does not
//!
//! It owns the two objectives — the mapping from coefficients to a value and a
//! gradient — and nothing else. The update rule, the line search, the budgets
//! and the determinism guarantees all live in [`crate::optimize`], which knows
//! nothing about logistic regression. That separation is the point: a third
//! objective reaches the same solver without touching it.
//!
//! # Scaling, and what `tol` therefore means
//!
//! Both objectives are the **mean** penalized negative log-likelihood: the
//! weighted loss sum plus the penalty, divided by the total sample weight. The
//! minimizer is unaffected, and the scaling is what makes a gradient tolerance
//! meaningful — an unscaled gradient grows with the row count, so a fixed
//! `tol` would mean a different thing at every sample size. Under this solver
//! `tol` is the infinity norm of that mean gradient; under Newton it stays the
//! largest absolute coefficient update. The two are different convergence
//! tests and are documented as such rather than pretended to be one.

use super::{LogisticRegressionParams, sample_weight};
use crate::api::ModelError;
use crate::data::SampleWeights;
use crate::loss::{BinaryLogLoss, Objective, raw_score};
use crate::numeric::{log_sum_exp, softmax_in_place};
use crate::optimize::{
    DEFAULT_MEMORY, LbfgsOptions, LbfgsWorkspace, OptimizeError, Problem, minimize,
};

/// Everything both objectives need about the standardized design matrix.
pub(super) struct DesignView<'a> {
    /// Row-major, `parameter_count` entries per row.
    pub(super) design: &'a [f64],
    pub(super) sample_weights: Option<&'a SampleWeights>,
    /// One scaled L2 penalty per feature column; the intercept has none.
    pub(super) penalties: &'a [f64],
    pub(super) columns: usize,
    pub(super) parameter_count: usize,
    pub(super) intercept_index: Option<usize>,
    /// Reciprocal of the total sample weight, applied once per evaluation.
    pub(super) inverse_total_weight: f64,
}

/// The L2 penalty each feature column carries in standardized space.
///
/// This is the same quantity the Newton path adds to its gradient and its
/// hessian diagonal — `1/C` divided by the squared column scale — computed here
/// as a vector because a matrix-free objective touches it once per evaluation
/// rather than once per iteration.
pub(super) fn scaled_penalties(scales: &[f64], c: f32) -> Vec<f64> {
    let lambda = 1.0 / f64::from(c);
    scales
        .iter()
        .map(|scale| lambda / (scale * scale))
        .collect()
}

/// Binary logistic negative log-likelihood over the standardized design.
struct BinaryProblem<'a> {
    view: DesignView<'a>,
    targets: &'a [u8],
}

impl Problem for BinaryProblem<'_> {
    fn dimension(&self) -> usize {
        self.view.parameter_count
    }

    fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
        let view = &self.view;
        gradient.fill(0.0);
        // Seeded from IEEE addition's identity rather than `+0.0`, so a total
        // that cancels to zero keeps its sign; see the accumulation policy in
        // `crate::numeric`. Rows are visited in ascending index order and
        // columns in ascending order within a row, which is what makes the
        // fitted coefficients reproducible.
        let mut total = -0.0_f64;
        for (row_index, design_row) in view.design.chunks_exact(view.parameter_count).enumerate() {
            let weight = sample_weight(view.sample_weights, row_index);
            let raw = raw_score(point, design_row, view.columns, view.intercept_index);
            let target = f64::from(self.targets[row_index]);
            total += weight * BinaryLogLoss::value(raw, target);
            let residual = weight * BinaryLogLoss::gradient(raw, target);
            for (slot, &value) in gradient.iter_mut().zip(design_row) {
                *slot += residual * value;
            }
        }
        penalize_and_scale(total, point, gradient, view)
    }
}

/// Joint multinomial negative log-likelihood over the standardized design.
///
/// The coefficient vector is the class rows stacked in class order, so class
/// `k`'s parameters occupy `k * parameter_count .. (k + 1) * parameter_count` —
/// the same layout the exact Newton path uses, which is what lets the two
/// solvers produce comparable models.
struct MultinomialProblem<'a> {
    view: DesignView<'a>,
    /// Column index in the sorted class list, one per row.
    class_of_row: &'a [usize],
    classes: usize,
    /// One row of scores, reused across rows and iterations.
    scores: Vec<f64>,
}

impl Problem for MultinomialProblem<'_> {
    fn dimension(&self) -> usize {
        self.classes * self.view.parameter_count
    }

    fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
        let view = &self.view;
        let parameter_count = view.parameter_count;
        gradient.fill(0.0);
        let mut total = -0.0_f64;
        for (row_index, design_row) in view.design.chunks_exact(parameter_count).enumerate() {
            let weight = sample_weight(view.sample_weights, row_index);
            let observed = self.class_of_row[row_index];
            for (class, slot) in self.scores.iter_mut().enumerate() {
                *slot = raw_score(
                    &point[class * parameter_count..(class + 1) * parameter_count],
                    design_row,
                    view.columns,
                    view.intercept_index,
                );
            }
            // `log_sum_exp` rather than `-ln(softmax)`: the probability of the
            // observed class underflows to exactly zero for a confidently wrong
            // row, and its logarithm would be an infinity for a loss that is
            // large but perfectly finite.
            total += weight * (log_sum_exp(&self.scores) - self.scores[observed]);
            softmax_in_place(&mut self.scores);
            for (class, &probability) in self.scores.iter().enumerate() {
                let residual = weight * (probability - f64::from(class == observed));
                let block = &mut gradient[class * parameter_count..(class + 1) * parameter_count];
                for (slot, &value) in block.iter_mut().zip(design_row) {
                    *slot += residual * value;
                }
            }
        }
        penalize_and_scale(total, point, gradient, view)
    }
}

/// Adds the L2 penalty to the accumulated loss and its gradient, then divides
/// both by the total sample weight.
///
/// The intercept is deliberately unpenalized, exactly as on the Newton path:
/// penalizing it would make the fit depend on where the targets happen to sit
/// on the score axis. The stacked multinomial vector is handled by walking one
/// class block at a time, so one function serves both objectives.
fn penalize_and_scale(
    loss_total: f64,
    point: &[f64],
    gradient: &mut [f64],
    view: &DesignView<'_>,
) -> f64 {
    let mut penalty = -0.0_f64;
    for block in 0..point.len() / view.parameter_count {
        let offset = block * view.parameter_count;
        for (column, &scaled) in view.penalties.iter().enumerate() {
            let coefficient = point[offset + column];
            penalty += 0.5 * scaled * coefficient * coefficient;
            gradient[offset + column] += scaled * coefficient;
        }
    }
    for slot in gradient.iter_mut() {
        *slot *= view.inverse_total_weight;
    }
    (loss_total + penalty) * view.inverse_total_weight
}

/// Translates an optimizer refusal into the estimator's error vocabulary.
///
/// Every variant stays a refusal. Two of them mean the same thing to a caller
/// and are reported the same way: whether the iteration budget ran out or the
/// line search could no longer bracket a better point, the solver did not reach
/// `tol` and the current iterate is not a converged fit.
///
/// A collapsed bracket is the observable form of a **tolerance below the
/// objective's numerical resolution**. The line search compares objective
/// values, so once consecutive trial values are within `f64` rounding of each
/// other it cannot certify a further decrease — which happens at a gradient
/// norm of roughly `sqrt(f64::EPSILON)` times the objective scale, near `1e-9`
/// for a log-loss of order one. Asking for less is reported rather than
/// answered with an iterate that never met the request.
fn into_model_error(error: OptimizeError) -> ModelError {
    match error {
        OptimizeError::NotConverged { iterations }
        | OptimizeError::LineSearchFailed { iterations } => {
            ModelError::SolverDidNotConverge { iterations }
        }
        OptimizeError::NonFiniteObjective { .. } => ModelError::LinearSolveFailed,
    }
}

/// Runs the solver and reports the fitted coefficients and iteration count.
fn solve<P: Problem>(
    problem: &mut P,
    dimension: usize,
    params: &LogisticRegressionParams,
) -> Result<(Vec<f64>, usize), ModelError> {
    let mut theta = vec![0.0_f64; dimension];
    let mut workspace = LbfgsWorkspace::new(dimension, DEFAULT_MEMORY);
    let options = LbfgsOptions::new(params.max_iter(), f64::from(params.tol()));
    let report =
        minimize(problem, &mut theta, &mut workspace, &options).map_err(into_model_error)?;
    // `iterations` is the number of accepted steps. The Newton path reports the
    // iteration on which its update fell below `tol`, so both count completed
    // work rather than convergence checks.
    Ok((theta, report.iterations))
}

/// Fits the binary objective, returning standardized coefficients.
pub(super) fn fit_binary(
    view: DesignView<'_>,
    targets: &[u8],
    params: &LogisticRegressionParams,
) -> Result<(Vec<f64>, usize), ModelError> {
    let dimension = view.parameter_count;
    let mut problem = BinaryProblem { view, targets };
    solve(&mut problem, dimension, params)
}

/// Fits the joint multinomial objective, returning stacked standardized
/// coefficient rows in class order.
pub(super) fn fit_multinomial(
    view: DesignView<'_>,
    class_of_row: &[usize],
    classes: usize,
    params: &LogisticRegressionParams,
) -> Result<(Vec<f64>, usize), ModelError> {
    let dimension = classes * view.parameter_count;
    let mut problem = MultinomialProblem {
        view,
        class_of_row,
        classes,
        scores: vec![0.0; classes],
    };
    solve(&mut problem, dimension, params)
}

#[cfg(test)]
mod tests {
    use super::super::{LogisticRegression, LogisticRegressionParams, LogisticSolver};
    use super::{BinaryProblem, DesignView, MultinomialProblem, scaled_penalties};
    use crate::api::ModelError;
    use crate::artifact::{ArtifactError, ModelArtifact};
    use crate::data::{BinaryTargets, ClassTargets, DenseMatrix, SampleWeights};
    use crate::numeric::OwnedRng;
    use crate::optimize::Problem;

    /// A separable-but-not-perfectly-separable binary problem.
    fn binary_problem() -> (DenseMatrix, BinaryTargets, SampleWeights) {
        let values = vec![
            0.0, 0.0, 0.5, 0.2, 0.2, 0.6, 1.0, 0.3, 2.0, 0.1, 1.8, 0.5, 2.2, 0.9, 0.3, 2.0, 0.8,
            2.4, 1.2, 2.2, 1.0, 3.0, 0.1, 1.0,
        ];
        (
            DenseMatrix::new(values, 12, 2).expect("fixture matrix"),
            BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]).expect("fixture targets"),
            SampleWeights::new(vec![
                1.0, 2.0, 0.5, 1.5, 3.0, 2.0, 1.0, 0.5, 2.5, 1.0, 1.0, 2.0,
            ])
            .expect("fixture weights"),
        )
    }

    fn three_class_problem() -> (DenseMatrix, ClassTargets) {
        let values = vec![
            0.0, 0.0, 0.5, 0.2, 0.2, 0.6, 1.0, 0.3, 2.0, 0.1, 1.8, 0.5, 2.2, 0.9, 0.3, 2.0, 0.8,
            2.4, 1.2, 2.2, 1.0, 3.0, 0.1, 1.0,
        ];
        (
            DenseMatrix::new(values, 12, 2).expect("fixture matrix"),
            ClassTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0]).expect("fixture targets"),
        )
    }

    /// Both solvers converged to the same minimizer, to a stated tolerance.
    ///
    /// The two convergence tests are different quantities, so this is an
    /// agreement bound rather than an equality: what is being proven is that a
    /// second update rule reaches the *same optimum*, not that it reaches it
    /// the same way.
    fn assert_agrees(newton: &LogisticRegression, lbfgs: &LogisticRegression, tolerance: f32) {
        assert_eq!(newton.classes(), lbfgs.classes());
        for (index, (left, right)) in newton
            .coefficients()
            .iter()
            .zip(lbfgs.coefficients())
            .enumerate()
        {
            assert!(
                (left - right).abs() <= tolerance,
                "coefficient {index}: newton {left}, lbfgs {right}"
            );
        }
        for (index, (left, right)) in newton
            .intercepts()
            .iter()
            .zip(lbfgs.intercepts())
            .enumerate()
        {
            assert!(
                (left - right).abs() <= tolerance,
                "intercept {index}: newton {left}, lbfgs {right}"
            );
        }
    }

    fn tight(solver: LogisticSolver) -> LogisticRegressionParams {
        LogisticRegressionParams::default()
            .with_solver(solver)
            .with_max_iter(500)
            .with_tol(1.0e-8)
    }

    // -----------------------------------------------------------------------
    // The objectives' own gradients, against a central difference of their own
    // values.
    //
    // Everything else about this solver is checked by comparing it with the
    // Newton path, which minimizes the *same* objective — so a mistake inside
    // `value_and_gradient` that both paths share would agree with itself. A
    // finite difference does not: it uses only the objective's value, so it
    // reconstructs the gradient from something the gradient code never touches.
    // -----------------------------------------------------------------------

    /// One randomly generated standardized design and its labels.
    struct ObjectiveCase {
        design: Vec<f64>,
        penalties: Vec<f64>,
        columns: usize,
        parameter_count: usize,
        intercept_index: Option<usize>,
        inverse_total_weight: f64,
        weights: Option<SampleWeights>,
        targets: Vec<u8>,
        class_of_row: Vec<usize>,
        classes: usize,
    }

    impl ObjectiveCase {
        fn random(rng: &mut OwnedRng) -> Self {
            let rows = 4 + rng.index(17);
            let columns = 1 + rng.index(4);
            let fit_intercept = rng.index(2) == 1;
            let parameter_count = columns + usize::from(fit_intercept);
            let intercept_index = fit_intercept.then_some(columns);

            let mut design = Vec::with_capacity(rows * parameter_count);
            for _ in 0..rows {
                for _ in 0..columns {
                    design.push(rng.unit_f64() * 4.0 - 2.0);
                }
                if fit_intercept {
                    // The intercept's design entry is the constant one, exactly
                    // as the fitting path lays it out.
                    design.push(1.0);
                }
            }

            let scales = (0..columns)
                .map(|_| 0.5 + rng.unit_f64() * 1.5)
                .collect::<Vec<_>>();
            let c = 10.0_f32.powf(rng.unit_f64() as f32 * 2.5 - 1.25);
            let penalties = scaled_penalties(&scales, c);

            let weights = if rng.index(2) == 1 {
                Some(
                    SampleWeights::new(
                        (0..rows)
                            .map(|_| (0.1 + rng.unit_f64() * 2.9) as f32)
                            .collect(),
                    )
                    .expect("positive finite weights"),
                )
            } else {
                None
            };
            let total_weight = weights
                .as_ref()
                .map_or(rows as f64, |weights| weights.total());

            let classes = 2 + rng.index(3);
            Self {
                design,
                penalties,
                columns,
                parameter_count,
                intercept_index,
                inverse_total_weight: 1.0 / total_weight,
                weights,
                targets: (0..rows).map(|_| rng.index(2) as u8).collect(),
                class_of_row: (0..rows).map(|_| rng.index(classes)).collect(),
                classes,
            }
        }

        fn view(&self) -> DesignView<'_> {
            DesignView {
                design: &self.design,
                sample_weights: self.weights.as_ref(),
                penalties: &self.penalties,
                columns: self.columns,
                parameter_count: self.parameter_count,
                intercept_index: self.intercept_index,
                inverse_total_weight: self.inverse_total_weight,
            }
        }
    }

    /// Worst relative discrepancy between an objective's own gradient and a
    /// central difference of its value, over every coordinate.
    fn worst_central_difference_error<P: Problem>(problem: &mut P, point: &[f64]) -> f64 {
        let dimension = point.len();
        let mut analytic = vec![0.0_f64; dimension];
        let base = problem.value_and_gradient(point, &mut analytic);
        assert!(base.is_finite(), "objective is not finite at {point:?}");
        let mut scratch = vec![0.0_f64; dimension];
        let mut probe = point.to_vec();
        let mut worst = 0.0_f64;
        for index in 0..dimension {
            let step = 1.0e-5 * (1.0 + point[index].abs());
            probe[index] = point[index] + step;
            let forward = problem.value_and_gradient(&probe, &mut scratch);
            probe[index] = point[index] - step;
            let backward = problem.value_and_gradient(&probe, &mut scratch);
            probe[index] = point[index];
            let approximate = (forward - backward) / (2.0 * step);
            worst =
                worst.max((approximate - analytic[index]).abs() / (1.0 + analytic[index].abs()));
        }
        worst
    }

    /// An objective whose value is right and whose gradient is not, used to
    /// prove the check above can fail.
    struct BiasedGradient<P> {
        inner: P,
        index: usize,
    }

    impl<P: Problem> Problem for BiasedGradient<P> {
        fn dimension(&self) -> usize {
            self.inner.dimension()
        }

        fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
            let value = self.inner.value_and_gradient(point, gradient);
            gradient[self.index] += 0.01;
            value
        }
    }

    /// The largest relative error a correct gradient may show at this step
    /// size. A central difference is accurate to about `1e-10` here; anything
    /// near a sign error, a missing term, or a wrong scale factor is orders of
    /// magnitude above this.
    const FINITE_DIFFERENCE_TOLERANCE: f64 = 1.0e-6;

    #[test]
    fn both_objectives_agree_with_a_central_difference_of_their_own_values() {
        let mut rng = OwnedRng::new(0x9b1d_0e11_5eed_0003);
        let (mut binary_points, mut multinomial_points) = (0_usize, 0_usize);
        let (mut worst_binary, mut worst_multinomial) = (0.0_f64, 0.0_f64);
        let (mut worst_control, mut controls) = (f64::INFINITY, 0_usize);
        let mut coordinates = 0_usize;

        for _ in 0..96 {
            let case = ObjectiveCase::random(&mut rng);
            for scale in [0.5_f64, 2.0, 6.0] {
                let point = (0..case.parameter_count)
                    .map(|_| (rng.unit_f64() * 2.0 - 1.0) * scale)
                    .collect::<Vec<_>>();
                let mut problem = BinaryProblem {
                    view: case.view(),
                    targets: &case.targets,
                };
                worst_binary =
                    worst_binary.max(worst_central_difference_error(&mut problem, &point));
                binary_points += 1;
                coordinates += point.len();

                let stacked = (0..case.classes * case.parameter_count)
                    .map(|_| (rng.unit_f64() * 2.0 - 1.0) * scale)
                    .collect::<Vec<_>>();
                let mut problem = MultinomialProblem {
                    view: case.view(),
                    class_of_row: &case.class_of_row,
                    classes: case.classes,
                    scores: vec![0.0; case.classes],
                };
                worst_multinomial =
                    worst_multinomial.max(worst_central_difference_error(&mut problem, &stacked));
                multinomial_points += 1;
                coordinates += stacked.len();
            }

            // Control: the same check against a gradient that is wrong in one
            // coordinate by a hundredth.
            let point = (0..case.parameter_count)
                .map(|_| rng.unit_f64() * 2.0 - 1.0)
                .collect::<Vec<_>>();
            let mut biased = BiasedGradient {
                inner: BinaryProblem {
                    view: case.view(),
                    targets: &case.targets,
                },
                index: rng.index(case.parameter_count),
            };
            worst_control = worst_control.min(worst_central_difference_error(&mut biased, &point));
            controls += 1;
        }

        println!(
            "logistic objectives: {binary_points} binary and {multinomial_points} multinomial \
             points over {coordinates} coordinates; worst relative gradient error \
             {worst_binary:e} (binary), {worst_multinomial:e} (multinomial)"
        );
        println!(
            "logistic objective control: {controls} deliberately biased gradients, smallest \
             reported error {worst_control:e} against a tolerance of \
             {FINITE_DIFFERENCE_TOLERANCE:e}"
        );

        assert!(
            worst_binary <= FINITE_DIFFERENCE_TOLERANCE,
            "binary objective gradient error {worst_binary:e}"
        );
        assert!(
            worst_multinomial <= FINITE_DIFFERENCE_TOLERANCE,
            "multinomial objective gradient error {worst_multinomial:e}"
        );
        // Non-vacuity: every biased gradient has to be caught, or passing above
        // is not evidence of anything.
        assert!(
            worst_control > FINITE_DIFFERENCE_TOLERANCE,
            "a gradient wrong by 0.01 in one coordinate reported only {worst_control:e}"
        );
    }

    #[test]
    fn the_default_solver_is_newton() {
        // The one property this whole sprint is written against.
        assert_eq!(
            LogisticRegressionParams::default().solver(),
            LogisticSolver::Newton
        );
        assert_eq!(LogisticSolver::default(), LogisticSolver::Newton);
    }

    #[test]
    fn the_matrix_free_solver_reaches_the_same_binary_optimum() {
        let (data, targets, _) = binary_problem();
        for fit_intercept in [true, false] {
            for c in [0.1_f32, 1.0, 10.0] {
                let newton = LogisticRegression::fit(
                    &data.as_view(),
                    &targets,
                    tight(LogisticSolver::Newton)
                        .with_fit_intercept(fit_intercept)
                        .with_c(c),
                )
                .expect("newton fit");
                let lbfgs = LogisticRegression::fit(
                    &data.as_view(),
                    &targets,
                    tight(LogisticSolver::Lbfgs)
                        .with_fit_intercept(fit_intercept)
                        .with_c(c),
                )
                .expect("lbfgs fit");
                assert_agrees(&newton, &lbfgs, 2.0e-6);
            }
        }
    }

    #[test]
    fn the_matrix_free_solver_reaches_the_same_weighted_binary_optimum() {
        let (data, targets, weights) = binary_problem();
        let newton = LogisticRegression::fit_weighted(
            &data.as_view(),
            &targets,
            &weights,
            tight(LogisticSolver::Newton),
        )
        .expect("newton fit");
        let lbfgs = LogisticRegression::fit_weighted(
            &data.as_view(),
            &targets,
            &weights,
            tight(LogisticSolver::Lbfgs),
        )
        .expect("lbfgs fit");
        assert_agrees(&newton, &lbfgs, 2.0e-6);
    }

    #[test]
    fn the_matrix_free_solver_reaches_the_same_multinomial_optimum() {
        let (data, targets) = three_class_problem();
        for fit_intercept in [true, false] {
            let newton = LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                tight(LogisticSolver::Newton).with_fit_intercept(fit_intercept),
            )
            .expect("newton fit");
            let lbfgs = LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                tight(LogisticSolver::Lbfgs).with_fit_intercept(fit_intercept),
            )
            .expect("lbfgs fit");
            assert_agrees(&newton, &lbfgs, 5.0e-5);
        }
    }

    #[test]
    fn the_matrix_free_multinomial_fit_stays_centred() {
        // The parametrization pins itself by the gradient summing to zero
        // across classes at every iterate. A solver that broke that would
        // still fit the data and would silently produce a different — and
        // unfrozen — coefficient representative of the same probabilities.
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            tight(LogisticSolver::Lbfgs),
        )
        .expect("lbfgs fit");
        let scores = model.decision_function(&data.as_view()).expect("scores");
        for row in scores.chunks_exact(model.n_decision_columns()) {
            let total: f32 = row.iter().sum();
            assert!(total.abs() <= 1.0e-4, "uncentred score row {row:?}");
        }
    }

    #[test]
    fn refitting_under_the_matrix_free_solver_is_bit_identical() {
        let (data, targets, weights) = binary_problem();
        let params = tight(LogisticSolver::Lbfgs);
        let first =
            LogisticRegression::fit_weighted(&data.as_view(), &targets, &weights, params.clone())
                .expect("fit");
        let second = LogisticRegression::fit_weighted(&data.as_view(), &targets, &weights, params)
            .expect("refit");
        assert_eq!(
            first
                .coefficients()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .coefficients()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(first.intercept().to_bits(), second.intercept().to_bits());
        assert_eq!(first.n_iter(), second.n_iter());
    }

    #[test]
    fn an_exhausted_iteration_budget_is_reported_rather_than_returned() {
        // The failure the exact path has always had and never reported: a fit
        // that stopped because it ran out of iterations looks identical to one
        // that stopped because it converged.
        let (data, targets, _) = binary_problem();
        let error = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default()
                .with_solver(LogisticSolver::Lbfgs)
                .with_c(1.0e6)
                .with_max_iter(2)
                .with_tol(1.0e-12),
        )
        .expect_err("must not return an unconverged model");
        assert_eq!(error, ModelError::SolverDidNotConverge { iterations: 2 });
    }

    #[test]
    fn a_tolerance_below_the_objectives_resolution_is_refused_not_approximated() {
        // The line search certifies progress by comparing objective values, so
        // a gradient tolerance below `sqrt(f64::EPSILON)` times the objective
        // scale is not something any value-comparing search can reach. The
        // boundary is documented and enforced rather than papered over: 1e-8 is
        // attainable on this fixture and 1e-12 is not.
        let (data, targets, _) = binary_problem();
        let attainable = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            tight(LogisticSolver::Lbfgs).with_c(0.1),
        );
        assert!(attainable.is_ok(), "{attainable:?}");
        let unattainable = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            tight(LogisticSolver::Lbfgs).with_c(0.1).with_tol(1.0e-12),
        );
        assert!(
            matches!(unattainable, Err(ModelError::SolverDidNotConverge { .. })),
            "{unattainable:?}"
        );
    }

    #[test]
    fn a_model_fitted_under_a_non_default_solver_has_no_artifact() {
        // Neither payload schema records a solver, so writing one would produce
        // bytes that decode as a model claiming Newton provenance.
        let (data, targets, _) = binary_problem();
        let model = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default().with_solver(LogisticSolver::Lbfgs),
        )
        .expect("fit");
        assert_eq!(
            model.to_artifact([7; 32]),
            Err(ArtifactError::UnsupportedModelState)
        );
        let newton = LogisticRegression::fit(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .expect("fit");
        let bytes = newton.to_artifact([7; 32]).expect("newton persists");
        assert_eq!(
            LogisticRegression::from_artifact(&bytes, [7; 32]).expect("decode"),
            newton
        );
    }

    #[test]
    fn the_solver_parameter_is_retained_and_validated_like_every_other() {
        let (data, targets, _) = binary_problem();
        let params = LogisticRegressionParams::default().with_solver(LogisticSolver::Lbfgs);
        let model =
            LogisticRegression::fit(&data.as_view(), &targets, params.clone()).expect("fit");
        assert_eq!(model.get_params(), &params);
        // Parameter validation happens before any solver runs, so an invalid
        // budget is the same typed error whichever solver was selected.
        assert_eq!(
            LogisticRegression::fit(&data.as_view(), &targets, params.clone().with_max_iter(0))
                .unwrap_err(),
            ModelError::InvalidIterationCount
        );
        assert_eq!(
            LogisticRegression::fit(&data.as_view(), &targets, params.with_c(0.0)).unwrap_err(),
            ModelError::InvalidRegularization
        );
    }
}
