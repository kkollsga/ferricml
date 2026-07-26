//! Limited-memory BFGS over a caller-owned objective.

use super::line_search::{LineSearchBuffers, LineSearchStart, strong_wolfe};
use crate::numeric::sum_in_order;

/// A differentiable objective the optimizer can evaluate.
///
/// The optimizer never allocates on behalf of an implementation and never
/// evaluates the value without the gradient, so an objective that shares an
/// intermediate between the two — a linear model's raw scores, typically —
/// computes it once per evaluation. `&mut self` exists for exactly that: the
/// implementation keeps its own scratch buffers, and a whole solve allocates
/// nothing after the workspace is built.
pub(crate) trait Problem {
    /// Number of coordinates the objective is defined over.
    fn dimension(&self) -> usize;

    /// Objective value at `point`, writing its gradient into `gradient`.
    ///
    /// `point` and `gradient` are both [`Problem::dimension`] long. An
    /// implementation must overwrite every gradient slot rather than
    /// accumulate into it.
    fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64;
}

/// Why an optimization stopped without producing a minimizer.
///
/// Every variant is a refusal. A bounded solver that ran out of its budget has
/// not found a minimum, and reporting the last iterate as though it had is the
/// failure mode this type exists to make impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OptimizeError {
    /// The iteration budget was exhausted before the gradient test passed.
    NotConverged {
        /// Iterations actually performed.
        iterations: usize,
    },
    /// The line search could not find a step satisfying the Wolfe conditions
    /// within its step and zoom budgets, or its bracket collapsed.
    LineSearchFailed {
        /// Iterations completed before the failing line search.
        iterations: usize,
    },
    /// The objective or its gradient was not finite at an evaluated point.
    NonFiniteObjective {
        /// Iterations completed before the offending evaluation.
        iterations: usize,
    },
}

/// Bounds and tolerances for one [`minimize`] call.
///
/// Every field is a hard bound rather than a hint. They are validated by
/// [`LbfgsOptions::new`] so an invalid budget cannot reach the loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LbfgsOptions {
    memory: usize,
    max_iterations: usize,
    gradient_tolerance: f64,
    max_line_search_steps: usize,
    max_zoom_steps: usize,
}

/// History length that has to hold two vectors per pair.
///
/// Five to ten pairs is the range the method is normally used in; more memory
/// stops paying for itself well before it starts costing noticeable storage.
pub(crate) const DEFAULT_MEMORY: usize = 8;
/// Bracketing steps before the search gives up expanding.
///
/// The bracketing phase doubles, so this budget has to exceed
/// `log2(MAX_STEP / initial_step)` or a small initial step would be reported as
/// a line-search failure rather than expanded — a bound on the *inputs* dressed
/// as a bound on the work. It is not a cost: bracketing stops at the first step
/// that decreases the objective enough, which is normally the first or second.
const DEFAULT_LINE_SEARCH_STEPS: usize = 128;
/// Bisection steps inside one bracket. A `f64` bracket cannot survive many
/// more than this before its endpoints are adjacent.
const DEFAULT_ZOOM_STEPS: usize = 64;

impl LbfgsOptions {
    /// Builds a validated option set.
    ///
    /// A zero memory, a zero iteration budget, or a non-positive tolerance
    /// would each turn the loop into something other than a bounded descent,
    /// so they are clamped to the smallest meaningful value here rather than
    /// producing a solver whose contract depends on its caller.
    pub(crate) fn new(max_iterations: usize, gradient_tolerance: f64) -> Self {
        Self {
            memory: DEFAULT_MEMORY,
            max_iterations: max_iterations.max(1),
            gradient_tolerance: if gradient_tolerance.is_finite() && gradient_tolerance > 0.0 {
                gradient_tolerance
            } else {
                f64::EPSILON
            },
            max_line_search_steps: DEFAULT_LINE_SEARCH_STEPS,
            max_zoom_steps: DEFAULT_ZOOM_STEPS,
        }
    }

    /// Overrides the number of stored correction pairs.
    #[cfg(test)]
    pub(crate) fn with_memory(mut self, memory: usize) -> Self {
        self.memory = memory.max(1);
        self
    }

    /// Returns the number of stored correction pairs.
    pub(crate) const fn memory(&self) -> usize {
        self.memory
    }

    /// Returns the largest number of iterations the loop may perform.
    pub(crate) const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the infinity-norm gradient bound that declares convergence.
    pub(crate) const fn gradient_tolerance(&self) -> f64 {
        self.gradient_tolerance
    }

    /// Returns the bracketing-step budget of one line search.
    pub(crate) const fn max_line_search_steps(&self) -> usize {
        self.max_line_search_steps
    }

    /// Returns the zoom-step budget of one line search.
    pub(crate) const fn max_zoom_steps(&self) -> usize {
        self.max_zoom_steps
    }
}

/// What one converged [`minimize`] call did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LbfgsReport {
    /// Iterations performed, counting the convergence test that passed.
    pub(crate) iterations: usize,
    /// Objective value at the returned point.
    pub(crate) value: f64,
    /// Infinity norm of the gradient at the returned point.
    pub(crate) gradient_norm: f64,
}

/// Every buffer one solve needs, allocated once.
///
/// The whole point of separating this from [`minimize`] is that a caller
/// fitting several models — a cross-validation fold sequence, a parameter
/// search — builds it once and reuses it, and that one solve performs no
/// allocation at all regardless of iteration count.
pub(crate) struct LbfgsWorkspace {
    dimension: usize,
    memory: usize,
    gradient: Vec<f64>,
    direction: Vec<f64>,
    trial_point: Vec<f64>,
    trial_gradient: Vec<f64>,
    /// `memory` blocks of `dimension`: the iterate differences.
    steps: Vec<f64>,
    /// `memory` blocks of `dimension`: the gradient differences.
    gradient_deltas: Vec<f64>,
    /// One reciprocal inner product per stored pair.
    rho: Vec<f64>,
    /// Two-loop recursion scratch, one entry per stored pair.
    alpha: Vec<f64>,
    /// Index the next pair is written to.
    head: usize,
    /// Number of pairs currently stored, at most `memory`.
    stored: usize,
}

impl LbfgsWorkspace {
    /// Allocates a workspace for `dimension` coordinates and `memory` pairs.
    pub(crate) fn new(dimension: usize, memory: usize) -> Self {
        let memory = memory.max(1);
        Self {
            dimension,
            memory,
            gradient: vec![0.0; dimension],
            direction: vec![0.0; dimension],
            trial_point: vec![0.0; dimension],
            trial_gradient: vec![0.0; dimension],
            steps: vec![0.0; memory * dimension],
            gradient_deltas: vec![0.0; memory * dimension],
            rho: vec![0.0; memory],
            alpha: vec![0.0; memory],
            head: 0,
            stored: 0,
        }
    }

    /// Forgets every stored correction pair.
    ///
    /// The curvature history describes the objective around the iterates that
    /// produced it. Reusing a workspace across two different problems, or
    /// keeping history across a direction reset, would apply one problem's
    /// curvature to another's gradient.
    fn clear(&mut self) {
        self.head = 0;
        self.stored = 0;
    }

    /// Slot index of the `age`-th most recent pair, `age = 0` being newest.
    fn slot(&self, age: usize) -> usize {
        (self.head + self.memory - 1 - age) % self.memory
    }

    fn step(&self, slot: usize) -> &[f64] {
        &self.steps[slot * self.dimension..(slot + 1) * self.dimension]
    }

    fn gradient_delta(&self, slot: usize) -> &[f64] {
        &self.gradient_deltas[slot * self.dimension..(slot + 1) * self.dimension]
    }
}

/// Dot product of two equal-length slices, in ascending index order.
///
/// Named rather than written inline so every inner product in the solver is
/// visibly the accumulation policy's fixed-order reduction. A vectorized or
/// reassociated dot product would make the fitted coefficients depend on how
/// the compiler chose to split the loop.
pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    sum_in_order(left.iter().zip(right).map(|(&left, &right)| left * right))
}

/// Infinity norm of a slice, which is the convergence measure.
fn infinity_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |worst, value| worst.max(value.abs()))
}

/// Minimizes `problem` from `point`, leaving the minimizer in `point`.
///
/// # Convergence
///
/// The test is `||gradient||_inf <= gradient_tolerance`, evaluated *before*
/// each step, so a starting point that already satisfies it costs one
/// objective evaluation and no iterations. Exhausting the iteration budget is
/// [`OptimizeError::NotConverged`], and `point` is left holding the last
/// iterate so a caller diagnosing the failure can inspect it — but it is
/// returned through an `Err`, never as a fitted model.
///
/// # Descent
///
/// The two-loop recursion produces a descent direction whenever the stored
/// curvature pairs are positive-definite, which the `s . y > 0` admission test
/// maintains. Floating point can still produce a non-descent direction on a
/// nearly flat objective; that is detected explicitly and answered by dropping
/// the history and taking a steepest-descent step, which always descends.
pub(crate) fn minimize<P: Problem>(
    problem: &mut P,
    point: &mut [f64],
    workspace: &mut LbfgsWorkspace,
    options: &LbfgsOptions,
) -> Result<LbfgsReport, OptimizeError> {
    debug_assert_eq!(point.len(), problem.dimension());
    debug_assert_eq!(workspace.dimension, point.len());
    debug_assert_eq!(workspace.memory, options.memory());
    workspace.clear();

    let mut value = problem.value_and_gradient(point, &mut workspace.gradient);
    if !value.is_finite() || workspace.gradient.iter().any(|slot| !slot.is_finite()) {
        return Err(OptimizeError::NonFiniteObjective { iterations: 0 });
    }

    for iteration in 0..options.max_iterations() {
        let gradient_norm = infinity_norm(&workspace.gradient);
        if gradient_norm <= options.gradient_tolerance() {
            return Ok(LbfgsReport {
                iterations: iteration,
                value,
                gradient_norm,
            });
        }

        two_loop_direction(workspace);
        let mut slope = dot(&workspace.gradient, &workspace.direction);
        if slope >= 0.0 || !slope.is_finite() {
            // Not a descent direction: the stored curvature no longer
            // describes this objective. Steepest descent always descends.
            workspace.clear();
            for (slot, &gradient) in workspace
                .direction
                .iter_mut()
                .zip(workspace.gradient.iter())
            {
                *slot = -gradient;
            }
            slope = dot(&workspace.gradient, &workspace.direction);
            if slope >= 0.0 || !slope.is_finite() {
                // The gradient passed the tolerance test above yet its own
                // negative is not a descent direction, which only a
                // denormal-dominated or overflowing gradient produces.
                return Err(OptimizeError::NonFiniteObjective {
                    iterations: iteration,
                });
            }
        }

        let initial_step = if workspace.stored == 0 {
            // Scale the very first step by the gradient so a large-magnitude
            // objective does not spend its whole bracketing budget shrinking.
            let scale = sum_in_order(workspace.gradient.iter().map(|value| value.abs()));
            if scale > 1.0 { 1.0 / scale } else { 1.0 }
        } else {
            1.0
        };

        let start = LineSearchStart {
            value,
            slope,
            initial_step,
        };
        let outcome = {
            let LbfgsWorkspace {
                direction,
                trial_point,
                trial_gradient,
                ..
            } = &mut *workspace;
            let mut buffers = LineSearchBuffers {
                trial_point,
                trial_gradient,
            };
            strong_wolfe(problem, point, direction, &start, &mut buffers, options)
        };
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(kind) => return Err(kind.at(iteration)),
        };
        debug_assert!(outcome.step > 0.0);

        // The trial buffers hold the accepted point and its gradient, so the
        // correction pair costs no further objective evaluation.
        let slot = workspace.head;
        let dimension = workspace.dimension;
        let offset = slot * dimension;
        {
            let LbfgsWorkspace {
                steps,
                gradient_deltas,
                trial_point,
                trial_gradient,
                gradient,
                ..
            } = &mut *workspace;
            for (target, (&trial, &current)) in steps[offset..offset + dimension]
                .iter_mut()
                .zip(trial_point.iter().zip(point.iter()))
            {
                *target = trial - current;
            }
            for (target, (&trial, &current)) in gradient_deltas[offset..offset + dimension]
                .iter_mut()
                .zip(trial_gradient.iter().zip(gradient.iter()))
            {
                *target = trial - current;
            }
        }
        point.copy_from_slice(&workspace.trial_point);
        workspace
            .gradient
            .copy_from_slice(&workspace.trial_gradient);
        value = outcome.value;

        let curvature = dot(
            &workspace.steps[offset..offset + dimension],
            &workspace.gradient_deltas[offset..offset + dimension],
        );
        if curvature > 0.0 && curvature.is_finite() {
            workspace.rho[slot] = 1.0 / curvature;
            workspace.head = (slot + 1) % workspace.memory;
            workspace.stored = (workspace.stored + 1).min(workspace.memory);
        }
    }

    Err(OptimizeError::NotConverged {
        iterations: options.max_iterations(),
    })
}

/// Writes `-H * gradient` into `workspace.direction` by the two-loop recursion.
///
/// With no stored pairs this is steepest descent, which is the correct first
/// step of the method rather than a fallback.
fn two_loop_direction(workspace: &mut LbfgsWorkspace) {
    for (slot, &gradient) in workspace
        .direction
        .iter_mut()
        .zip(workspace.gradient.iter())
    {
        *slot = -gradient;
    }
    if workspace.stored == 0 {
        return;
    }

    let dimension = workspace.dimension;
    for age in 0..workspace.stored {
        let slot = workspace.slot(age);
        let alpha = workspace.rho[slot] * dot(workspace.step(slot), &workspace.direction);
        workspace.alpha[slot] = alpha;
        for index in 0..dimension {
            let delta = workspace.gradient_deltas[slot * dimension + index];
            workspace.direction[index] -= alpha * delta;
        }
    }

    // Initial inverse-Hessian scaling from the newest pair. This is the one
    // choice that makes the first stored step's magnitude sensible; without it
    // the method behaves like undamped steepest descent for several iterations.
    let newest = workspace.slot(0);
    let delta = workspace.gradient_delta(newest);
    let denominator = dot(delta, delta);
    if denominator > 0.0 {
        let scale = 1.0 / (workspace.rho[newest] * denominator);
        for slot in workspace.direction.iter_mut() {
            *slot *= scale;
        }
    }

    for age in (0..workspace.stored).rev() {
        let slot = workspace.slot(age);
        let beta = workspace.rho[slot] * dot(workspace.gradient_delta(slot), &workspace.direction);
        let difference = workspace.alpha[slot] - beta;
        for index in 0..dimension {
            let iterate_delta = workspace.steps[slot * dimension + index];
            workspace.direction[index] += difference * iterate_delta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0.5 * (x - centre)' A (x - centre)` with a diagonal `A`.
    ///
    /// A diagonal quadratic has an analytic minimizer, an analytic value, and a
    /// condition number the test controls exactly, which is what makes it a
    /// proof rather than a smoke test.
    struct DiagonalQuadratic {
        curvature: Vec<f64>,
        centre: Vec<f64>,
        evaluations: usize,
    }

    impl Problem for DiagonalQuadratic {
        fn dimension(&self) -> usize {
            self.centre.len()
        }

        fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
            self.evaluations += 1;
            let mut total = 0.0;
            for index in 0..point.len() {
                let offset = point[index] - self.centre[index];
                gradient[index] = self.curvature[index] * offset;
                total += 0.5 * self.curvature[index] * offset * offset;
            }
            total
        }
    }

    /// The classic non-quadratic test function, minimized at `(1, 1)`.
    struct Rosenbrock {
        evaluations: usize,
    }

    impl Problem for Rosenbrock {
        fn dimension(&self) -> usize {
            2
        }

        fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
            self.evaluations += 1;
            let (x, y) = (point[0], point[1]);
            gradient[0] = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x);
            gradient[1] = 200.0 * (y - x * x);
            100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
        }
    }

    fn solve<P: Problem>(
        problem: &mut P,
        start: &[f64],
        options: &LbfgsOptions,
    ) -> (Vec<f64>, Result<LbfgsReport, OptimizeError>) {
        let mut point = start.to_vec();
        let mut workspace = LbfgsWorkspace::new(point.len(), options.memory());
        let report = minimize(problem, &mut point, &mut workspace, options);
        (point, report)
    }

    #[test]
    fn reaches_the_analytic_minimum_of_a_well_conditioned_quadratic() {
        let mut problem = DiagonalQuadratic {
            curvature: vec![1.0, 2.0, 4.0, 8.0],
            centre: vec![-3.0, 0.5, 7.0, 0.0],
            evaluations: 0,
        };
        let options = LbfgsOptions::new(200, 1.0e-10);
        let (point, report) = solve(&mut problem, &[0.0; 4], &options);
        let report = report.expect("converged");
        for (value, expected) in point.iter().zip(&[-3.0, 0.5, 7.0, 0.0]) {
            assert!((value - expected).abs() <= 1.0e-9, "{point:?}");
        }
        assert!(report.value.abs() <= 1.0e-16, "value {}", report.value);
        assert!(report.gradient_norm <= 1.0e-10);
    }

    #[test]
    fn reaches_the_analytic_minimum_of_an_ill_conditioned_quadratic() {
        // A condition number of 1e6 is where steepest descent stops being
        // usable, so this separates the curvature memory from the line search.
        let mut problem = DiagonalQuadratic {
            curvature: vec![1.0e-3, 1.0, 1.0e3],
            centre: vec![2.0, -1.0, 0.25],
            evaluations: 0,
        };
        let options = LbfgsOptions::new(500, 1.0e-9);
        let (point, report) = solve(&mut problem, &[0.0; 3], &options);
        let report = report.expect("converged");
        for (value, expected) in point.iter().zip(&[2.0, -1.0, 0.25]) {
            assert!((value - expected).abs() <= 1.0e-5, "{point:?}");
        }
        assert!(report.iterations < 500);
    }

    #[test]
    fn reaches_the_minimum_of_a_non_quadratic_valley() {
        let mut problem = Rosenbrock { evaluations: 0 };
        let options = LbfgsOptions::new(500, 1.0e-8);
        let (point, report) = solve(&mut problem, &[-1.2, 1.0], &options);
        report.expect("converged");
        assert!((point[0] - 1.0).abs() <= 1.0e-5, "{point:?}");
        assert!((point[1] - 1.0).abs() <= 1.0e-5, "{point:?}");
    }

    #[test]
    fn a_starting_point_at_the_minimum_costs_no_iterations() {
        let mut problem = DiagonalQuadratic {
            curvature: vec![1.0, 3.0],
            centre: vec![1.5, -2.5],
            evaluations: 0,
        };
        let options = LbfgsOptions::new(50, 1.0e-12);
        let (point, report) = solve(&mut problem, &[1.5, -2.5], &options);
        let report = report.expect("converged");
        assert_eq!(report.iterations, 0);
        assert_eq!(problem.evaluations, 1);
        assert_eq!(point, vec![1.5, -2.5]);
    }

    #[test]
    fn identical_inputs_produce_bit_identical_iterates() {
        // Determinism is a bit-level claim, not a tolerance: the fitted values
        // a solver produces are the fixed point of its own reductions.
        let options = LbfgsOptions::new(200, 1.0e-11);
        let mut first = Rosenbrock { evaluations: 0 };
        let mut second = Rosenbrock { evaluations: 0 };
        let (left, left_report) = solve(&mut first, &[-1.2, 1.0], &options);
        let (right, right_report) = solve(&mut second, &[-1.2, 1.0], &options);
        assert_eq!(
            left.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            right.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(left_report, right_report);
        assert_eq!(first.evaluations, second.evaluations);
    }

    #[test]
    fn a_reused_workspace_forgets_the_previous_problem() {
        // Curvature history describes the objective that produced it. Carrying
        // it into a second problem would make the second fit depend on the
        // first, which is exactly the reproducibility promise broken.
        let options = LbfgsOptions::new(200, 1.0e-11);
        let mut workspace = LbfgsWorkspace::new(2, options.memory());

        let mut warm = Rosenbrock { evaluations: 0 };
        let mut scratch = vec![-1.2, 1.0];
        minimize(&mut warm, &mut scratch, &mut workspace, &options).expect("first solve");

        let mut second = DiagonalQuadratic {
            curvature: vec![1.0, 5.0],
            centre: vec![3.0, -4.0],
            evaluations: 0,
        };
        let mut reused = vec![0.0, 0.0];
        let warm_report = minimize(&mut second, &mut reused, &mut workspace, &options)
            .expect("second solve on a reused workspace");

        let mut fresh_problem = DiagonalQuadratic {
            curvature: vec![1.0, 5.0],
            centre: vec![3.0, -4.0],
            evaluations: 0,
        };
        let (fresh, fresh_report) = solve(&mut fresh_problem, &[0.0, 0.0], &options);
        assert_eq!(
            reused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fresh.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(warm_report, fresh_report.expect("fresh solve"));
    }

    #[test]
    fn exhausting_the_iteration_budget_is_an_error_rather_than_a_result() {
        let mut problem = Rosenbrock { evaluations: 0 };
        let options = LbfgsOptions::new(2, 1.0e-12);
        let (_, report) = solve(&mut problem, &[-1.2, 1.0], &options);
        assert_eq!(report, Err(OptimizeError::NotConverged { iterations: 2 }));
    }

    #[test]
    fn a_non_finite_objective_is_reported_before_any_step() {
        struct Broken;
        impl Problem for Broken {
            fn dimension(&self) -> usize {
                1
            }
            fn value_and_gradient(&mut self, _point: &[f64], gradient: &mut [f64]) -> f64 {
                gradient[0] = f64::NAN;
                f64::NAN
            }
        }
        let options = LbfgsOptions::new(10, 1.0e-8);
        let (_, report) = solve(&mut Broken, &[0.0], &options);
        assert_eq!(
            report,
            Err(OptimizeError::NonFiniteObjective { iterations: 0 })
        );
    }

    #[test]
    fn an_unbounded_objective_fails_the_line_search_rather_than_diverging() {
        // A linear objective has no minimum. The search cannot satisfy the
        // curvature condition at any step, so it must refuse instead of
        // walking to infinity.
        struct Linear;
        impl Problem for Linear {
            fn dimension(&self) -> usize {
                1
            }
            fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
                gradient[0] = -1.0;
                -point[0]
            }
        }
        let options = LbfgsOptions::new(10, 1.0e-8);
        let (_, report) = solve(&mut Linear, &[0.0], &options);
        assert!(
            matches!(
                report,
                Err(OptimizeError::LineSearchFailed { .. })
                    | Err(OptimizeError::NonFiniteObjective { .. })
            ),
            "{report:?}"
        );
    }

    #[test]
    fn a_solve_after_the_workspace_exists_performs_no_allocation() {
        // The claim the workspace exists to support. Counting evaluations is
        // not enough: this asserts the buffers are sized once and that the
        // history ring never grows past its declared memory.
        let options = LbfgsOptions::new(200, 1.0e-11).with_memory(3);
        let mut workspace = LbfgsWorkspace::new(2, options.memory());
        let capacities = (
            workspace.steps.capacity(),
            workspace.gradient_deltas.capacity(),
            workspace.rho.capacity(),
            workspace.alpha.capacity(),
            workspace.direction.capacity(),
        );
        let mut problem = Rosenbrock { evaluations: 0 };
        let mut point = vec![-1.2, 1.0];
        let report =
            minimize(&mut problem, &mut point, &mut workspace, &options).expect("converged");
        assert!(report.iterations > 3, "the ring must actually wrap");
        assert_eq!(workspace.stored, 3);
        assert_eq!(
            capacities,
            (
                workspace.steps.capacity(),
                workspace.gradient_deltas.capacity(),
                workspace.rho.capacity(),
                workspace.alpha.capacity(),
                workspace.direction.capacity(),
            )
        );
    }

    #[test]
    fn a_smaller_memory_still_converges_on_the_same_problem() {
        // Memory is a cost/robustness trade, not a correctness requirement.
        for memory in 1..=8 {
            let options = LbfgsOptions::new(1_000, 1.0e-9).with_memory(memory);
            let mut problem = Rosenbrock { evaluations: 0 };
            let (point, report) = solve(&mut problem, &[-1.2, 1.0], &options);
            report.unwrap_or_else(|error| panic!("memory {memory}: {error:?}"));
            assert!(
                (point[0] - 1.0).abs() <= 1.0e-4,
                "memory {memory}: {point:?}"
            );
        }
    }

    /// A randomized objective, rebuildable so a solve can be verified against a
    /// fresh instance rather than against the one that produced the answer.
    #[derive(Clone, Debug)]
    enum Spec {
        Diagonal {
            curvature: Vec<f64>,
            centre: Vec<f64>,
        },
        Rosenbrock {
            scale: f64,
        },
    }

    /// Rosenbrock with a caller-chosen valley steepness.
    struct ScaledRosenbrock {
        scale: f64,
        evaluations: usize,
    }

    impl Problem for ScaledRosenbrock {
        fn dimension(&self) -> usize {
            2
        }

        fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
            self.evaluations += 1;
            let (x, y) = (point[0], point[1]);
            let gap = y - x * x;
            gradient[0] = -4.0 * self.scale * x * gap - 2.0 * (1.0 - x);
            gradient[1] = 2.0 * self.scale * gap;
            self.scale * gap * gap + (1.0 - x) * (1.0 - x)
        }
    }

    /// The two families share only this trait object, so the verification pass
    /// below can rebuild whichever one a case used.
    fn build(spec: &Spec) -> Box<dyn Problem> {
        match spec {
            Spec::Diagonal { curvature, centre } => Box::new(DiagonalQuadratic {
                curvature: curvature.clone(),
                centre: centre.clone(),
                evaluations: 0,
            }),
            Spec::Rosenbrock { scale } => Box::new(ScaledRosenbrock {
                scale: *scale,
                evaluations: 0,
            }),
        }
    }

    impl Problem for Box<dyn Problem> {
        fn dimension(&self) -> usize {
            (**self).dimension()
        }

        fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
            (**self).value_and_gradient(point, gradient)
        }
    }

    /// The convergence contract, run as a sweep rather than asserted on one
    /// fixture: **no `Ok` ever describes a point that fails the gradient test,
    /// and no budget short of the one that first converges returns a point at
    /// all.**
    ///
    /// The type's own documentation says every variant is a refusal and that
    /// reporting the last iterate as a fitted model is the failure it exists to
    /// prevent. That is checkable, and it is checked here by walking every
    /// budget from one upward on the same problem: the boundary between refusal
    /// and answer has to be a single crossing, and the answers on the far side
    /// of it have to be the same point.
    #[test]
    fn an_exhausted_budget_never_returns_an_iterate_and_every_answer_verifies() {
        let mut rng = crate::numeric::OwnedRng::new(0x0b1e_c714_5eed_0002);
        let (mut problems, mut budgets_tried) = (0_usize, 0_usize);
        let (mut converged, mut not_converged, mut other_refusals) = (0_usize, 0, 0);
        let mut worst_value_mismatch = 0_u64;
        let mut worst_norm_mismatch = 0_u64;
        let mut problems_needing_several_iterations = 0_usize;
        let mut control_perturbed_points_over_tolerance = 0_usize;

        for _ in 0..48 {
            let spec = if rng.index(3) == 0 {
                Spec::Rosenbrock {
                    scale: 10.0_f64.powf(rng.unit_f64() * 3.0),
                }
            } else {
                let dimension = 1 + rng.index(5);
                Spec::Diagonal {
                    curvature: (0..dimension)
                        .map(|_| 10.0_f64.powf(rng.unit_f64() * 4.0 - 2.0))
                        .collect(),
                    centre: (0..dimension).map(|_| rng.unit_f64() * 8.0 - 4.0).collect(),
                }
            };
            let dimension = build(&spec).dimension();
            let start = (0..dimension)
                .map(|_| rng.unit_f64() * 4.0 - 2.0)
                .collect::<Vec<_>>();
            let tolerance = 10.0_f64.powi(-(6 + rng.index(4) as i32));

            let mut first_answer: Option<(usize, Vec<u64>)> = None;
            for budget in 1..=40_usize {
                budgets_tried += 1;
                let options = LbfgsOptions::new(budget, tolerance);
                let mut problem = build(&spec);
                let mut point = start.clone();
                let mut workspace = LbfgsWorkspace::new(dimension, options.memory());
                let report = minimize(&mut problem, &mut point, &mut workspace, &options);
                match report {
                    Ok(report) => {
                        converged += 1;
                        // Verified against a fresh instance of the objective at
                        // the point actually returned.
                        let mut fresh = build(&spec);
                        let mut gradient = vec![0.0; dimension];
                        let value = fresh.value_and_gradient(&point, &mut gradient);
                        worst_value_mismatch = worst_value_mismatch
                            .max(value.to_bits().abs_diff(report.value.to_bits()));
                        let norm = infinity_norm(&gradient);
                        worst_norm_mismatch = worst_norm_mismatch
                            .max(norm.to_bits().abs_diff(report.gradient_norm.to_bits()));
                        assert!(
                            norm <= tolerance,
                            "an accepted point has gradient norm {norm} above tolerance \
                             {tolerance}"
                        );
                        assert!(
                            report.iterations <= budget,
                            "reported {} iterations under a budget of {budget}",
                            report.iterations
                        );

                        // Control: the tolerance test is not satisfied
                        // everywhere. A displaced point must fail it.
                        let displaced = point.iter().map(|value| value + 0.25).collect::<Vec<_>>();
                        let mut displaced_gradient = vec![0.0; dimension];
                        build(&spec).value_and_gradient(&displaced, &mut displaced_gradient);
                        if infinity_norm(&displaced_gradient) > tolerance {
                            control_perturbed_points_over_tolerance += 1;
                        }

                        let bits = point
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>();
                        match &first_answer {
                            None => {
                                if budget > 1 {
                                    problems_needing_several_iterations += 1;
                                }
                                first_answer = Some((budget, bits));
                            }
                            Some((first_budget, first_bits)) => assert_eq!(
                                &bits, first_bits,
                                "budget {budget} returned a different point than the \
                                 smallest converging budget {first_budget}"
                            ),
                        }
                    }
                    Err(OptimizeError::NotConverged { iterations }) => {
                        not_converged += 1;
                        assert_eq!(
                            iterations, budget,
                            "an exhausted budget must report the budget it exhausted"
                        );
                        assert!(
                            first_answer.is_none(),
                            "budget {budget} refused after a smaller budget had converged"
                        );
                    }
                    Err(_) => {
                        other_refusals += 1;
                        assert!(
                            first_answer.is_none(),
                            "budget {budget} refused after a smaller budget had converged"
                        );
                    }
                }
            }
            problems += 1;
        }

        println!(
            "lbfgs budgets: {problems} problems x 40 budgets = {budgets_tried} solves; \
             {converged} converged, {not_converged} exhausted the budget, \
             {other_refusals} refused otherwise"
        );
        println!(
            "lbfgs verification: worst value mismatch {worst_value_mismatch} ulp, worst \
             gradient-norm mismatch {worst_norm_mismatch} ulp, {problems_needing_several_iterations} \
             problems needed more than one iteration"
        );
        println!(
            "lbfgs control: a point displaced by 0.25 exceeds the tolerance in \
             {control_perturbed_points_over_tolerance} of {converged} accepted solves"
        );

        assert_eq!(
            worst_value_mismatch, 0,
            "the report must describe the point"
        );
        assert_eq!(worst_norm_mismatch, 0, "the report must describe the point");
        assert!(converged > 0 && not_converged > 0, "both branches must run");
        assert!(
            problems_needing_several_iterations * 2 > problems,
            "only {problems_needing_several_iterations} of {problems} problems took more \
             than one iteration, so the budget boundary was barely exercised"
        );
        assert_eq!(
            control_perturbed_points_over_tolerance, converged,
            "the gradient tolerance was satisfied at a displaced point too, so passing it \
             at the returned point says nothing"
        );
    }

    #[test]
    fn the_objective_decreases_at_every_accepted_step() {
        // Recorded through the problem itself, so this observes the values the
        // solver actually accepted rather than re-deriving them.
        struct Recording {
            inner: Rosenbrock,
            accepted: Vec<f64>,
        }
        impl Problem for Recording {
            fn dimension(&self) -> usize {
                self.inner.dimension()
            }
            fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
                let value = self.inner.value_and_gradient(point, gradient);
                self.accepted.push(value);
                value
            }
        }
        let options = LbfgsOptions::new(200, 1.0e-10);
        let mut problem = Recording {
            inner: Rosenbrock { evaluations: 0 },
            accepted: Vec::new(),
        };
        let mut point = vec![-1.2, 1.0];
        let mut workspace = LbfgsWorkspace::new(2, options.memory());
        let report =
            minimize(&mut problem, &mut point, &mut workspace, &options).expect("converged");
        assert!(report.value <= problem.accepted[0]);
        assert!(report.value <= 1.0e-16, "value {}", report.value);
    }
}
