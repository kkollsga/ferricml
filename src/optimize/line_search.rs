//! A bounded strong-Wolfe line search.
//!
//! A quasi-Newton direction is only a direction. How far to move along it is
//! the other half of the method, and it is the half that decides whether the
//! curvature pairs the outer loop stores are meaningful: the Wolfe curvature
//! condition is exactly what guarantees `s . y > 0`, which is what keeps the
//! approximate inverse Hessian positive definite.
//!
//! The search is the textbook bracket-then-zoom pair (Nocedal & Wright,
//! *Numerical Optimization*, algorithms 3.5 and 3.6). Two departures from the
//! usual presentation are deliberate:
//!
//! - **The zoom phase bisects rather than interpolating.** Cubic interpolation
//!   converges in fewer evaluations on smooth problems, but its next trial
//!   depends on a fitted polynomial whose coefficients are themselves rounded,
//!   and a safeguard that fires on one target and not another would make the
//!   iterate sequence platform-dependent. Bisection's next trial is one
//!   addition and one halving of the current bracket, so it is a function of
//!   the bracket alone.
//! - **Every budget is finite and exhausting one is an error.** A line search
//!   that returns its last trial because it ran out of steps hands the outer
//!   loop a step that satisfies neither Wolfe condition, and the fitted model
//!   that comes out the far end looks exactly like a converged one.

use super::lbfgs::{LbfgsOptions, OptimizeError, Problem, dot};

/// Armijo sufficient-decrease constant. The conventional value; it has to be
/// small enough that a Newton-scale unit step is normally accepted.
const SUFFICIENT_DECREASE: f64 = 1.0e-4;
/// Wolfe curvature constant. The conventional quasi-Newton value: loose enough
/// that the unit step usually passes, tight enough to force real curvature
/// information into the stored pair.
const CURVATURE: f64 = 0.9;
/// Largest step the bracketing phase will expand to. A descent direction that
/// has not bracketed a minimizer by here is on an objective with no minimum
/// along it, which is a refusal rather than a longer search.
const MAX_STEP: f64 = 1.0e20;

/// The starting point's already-known scalars, so the search need not
/// re-evaluate the objective it was called from.
pub(super) struct LineSearchStart {
    /// Objective value at step zero.
    pub(super) value: f64,
    /// Directional derivative at step zero; must be strictly negative.
    pub(super) slope: f64,
    /// First step the bracketing phase tries.
    pub(super) initial_step: f64,
}

/// Caller-owned storage the search evaluates through.
///
/// On a successful return these hold the accepted point and its gradient,
/// which is what lets the outer loop form its correction pair without a
/// further objective evaluation.
pub(super) struct LineSearchBuffers<'a> {
    pub(super) trial_point: &'a mut [f64],
    pub(super) trial_gradient: &'a mut [f64],
}

/// The accepted step and the objective value there.
pub(super) struct LineSearchOutcome {
    pub(super) step: f64,
    pub(super) value: f64,
}

/// Why a line search refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LineSearchFailure {
    /// A step or zoom budget ran out, or the bracket collapsed to adjacent
    /// floating-point values without satisfying the conditions.
    Exhausted,
    /// The objective or its gradient was not finite at a trial point.
    NonFinite,
}

impl LineSearchFailure {
    /// Attaches the outer loop's iteration count to this failure.
    pub(super) const fn at(self, iterations: usize) -> OptimizeError {
        match self {
            Self::Exhausted => OptimizeError::LineSearchFailed { iterations },
            Self::NonFinite => OptimizeError::NonFiniteObjective { iterations },
        }
    }
}

/// Evaluates the objective at `point + step * direction`.
///
/// Returns the value and the directional derivative there, or `None` when
/// either is not finite — which a large trial step reaches on any objective
/// with an exponential in it.
fn probe<P: Problem>(
    problem: &mut P,
    point: &[f64],
    direction: &[f64],
    step: f64,
    buffers: &mut LineSearchBuffers<'_>,
) -> Option<(f64, f64)> {
    for (slot, (&start, &delta)) in buffers
        .trial_point
        .iter_mut()
        .zip(point.iter().zip(direction))
    {
        *slot = start + step * delta;
    }
    let value = problem.value_and_gradient(buffers.trial_point, buffers.trial_gradient);
    if !value.is_finite() || buffers.trial_gradient.iter().any(|slot| !slot.is_finite()) {
        return None;
    }
    Some((value, dot(buffers.trial_gradient, direction)))
}

/// Finds a step satisfying the strong Wolfe conditions along `direction`.
pub(super) fn strong_wolfe<P: Problem>(
    problem: &mut P,
    point: &[f64],
    direction: &[f64],
    start: &LineSearchStart,
    buffers: &mut LineSearchBuffers<'_>,
    options: &LbfgsOptions,
) -> Result<LineSearchOutcome, LineSearchFailure> {
    debug_assert!(start.slope < 0.0);
    let mut previous_step = 0.0_f64;
    let mut previous_value = start.value;
    let mut step = start.initial_step;

    for index in 0..options.max_line_search_steps() {
        let (value, slope) =
            probe(problem, point, direction, step, buffers).ok_or(LineSearchFailure::NonFinite)?;
        if value > start.value + SUFFICIENT_DECREASE * step * start.slope
            || (index > 0 && value >= previous_value)
        {
            return zoom(
                problem,
                point,
                direction,
                start,
                Bracket {
                    low: previous_step,
                    high: step,
                    low_value: previous_value,
                },
                buffers,
                options,
            );
        }
        if slope.abs() <= -CURVATURE * start.slope {
            return Ok(LineSearchOutcome { step, value });
        }
        if slope >= 0.0 {
            return zoom(
                problem,
                point,
                direction,
                start,
                Bracket {
                    low: step,
                    high: previous_step,
                    low_value: value,
                },
                buffers,
                options,
            );
        }
        if step >= MAX_STEP {
            return Err(LineSearchFailure::Exhausted);
        }
        previous_step = step;
        previous_value = value;
        step = (step * 2.0).min(MAX_STEP);
    }
    Err(LineSearchFailure::Exhausted)
}

/// A bracket known to contain a step satisfying both Wolfe conditions.
///
/// `low` is the endpoint with the smaller objective value and need not be the
/// numerically smaller step; that asymmetry is what the algorithm's
/// endpoint-swap test maintains.
struct Bracket {
    low: f64,
    high: f64,
    low_value: f64,
}

fn zoom<P: Problem>(
    problem: &mut P,
    point: &[f64],
    direction: &[f64],
    start: &LineSearchStart,
    mut bracket: Bracket,
    buffers: &mut LineSearchBuffers<'_>,
    options: &LbfgsOptions,
) -> Result<LineSearchOutcome, LineSearchFailure> {
    for _ in 0..options.max_zoom_steps() {
        let step = 0.5 * (bracket.low + bracket.high);
        if step == bracket.low || step == bracket.high {
            // The endpoints are adjacent representable values, so no further
            // trial exists. Refusing is the only honest answer.
            return Err(LineSearchFailure::Exhausted);
        }
        let (value, slope) =
            probe(problem, point, direction, step, buffers).ok_or(LineSearchFailure::NonFinite)?;
        if value > start.value + SUFFICIENT_DECREASE * step * start.slope
            || value >= bracket.low_value
        {
            bracket.high = step;
        } else {
            if slope.abs() <= -CURVATURE * start.slope {
                return Ok(LineSearchOutcome { step, value });
            }
            if slope * (bracket.high - bracket.low) >= 0.0 {
                bracket.high = bracket.low;
            }
            bracket.low = step;
            bracket.low_value = value;
        }
    }
    Err(LineSearchFailure::Exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0.5 * a * x^2 + b * x` restricted to one coordinate, so the search can
    /// be exercised without the outer loop.
    struct Scalar {
        curvature: f64,
        slope: f64,
        evaluations: usize,
    }

    impl Problem for Scalar {
        fn dimension(&self) -> usize {
            1
        }
        fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
            self.evaluations += 1;
            gradient[0] = self.curvature * point[0] + self.slope;
            0.5 * self.curvature * point[0] * point[0] + self.slope * point[0]
        }
    }

    fn search(
        problem: &mut Scalar,
        initial_step: f64,
        options: &LbfgsOptions,
    ) -> Result<(f64, f64, f64), LineSearchFailure> {
        let point = [0.0_f64];
        let mut gradient = [0.0_f64];
        let value = problem.value_and_gradient(&point, &mut gradient);
        let direction = [-gradient[0]];
        let slope = gradient[0] * direction[0];
        let mut trial_point = [0.0_f64];
        let mut trial_gradient = [0.0_f64];
        let mut buffers = LineSearchBuffers {
            trial_point: &mut trial_point,
            trial_gradient: &mut trial_gradient,
        };
        let outcome = strong_wolfe(
            problem,
            &point,
            &direction,
            &LineSearchStart {
                value,
                slope,
                initial_step,
            },
            &mut buffers,
            options,
        )?;
        Ok((outcome.step, outcome.value, trial_point[0]))
    }

    fn options() -> LbfgsOptions {
        LbfgsOptions::new(100, 1.0e-10)
    }

    #[test]
    fn both_wolfe_conditions_hold_at_the_accepted_step() {
        for curvature in [0.25_f64, 1.0, 4.0, 100.0] {
            let mut problem = Scalar {
                curvature,
                slope: -3.0,
                evaluations: 0,
            };
            let (step, value, position) = search(&mut problem, 1.0, &options()).expect("accepted");
            let start_value = 0.0;
            let start_slope = -9.0; // direction is -gradient(0) = 3, slope = -3 * 3
            assert!(
                value <= start_value + SUFFICIENT_DECREASE * step * start_slope,
                "sufficient decrease at curvature {curvature}"
            );
            let mut gradient = [0.0_f64];
            problem.value_and_gradient(&[position], &mut gradient);
            let slope = gradient[0] * 3.0;
            assert!(
                slope.abs() <= -CURVATURE * start_slope,
                "curvature condition at {curvature}: slope {slope}"
            );
        }
    }

    #[test]
    fn the_accepted_buffers_describe_the_accepted_step() {
        // The outer loop forms its correction pair from these buffers without
        // re-evaluating, so a search that returned a step not matching them
        // would corrupt the curvature history rather than fail loudly.
        let mut problem = Scalar {
            curvature: 2.0,
            slope: -5.0,
            evaluations: 0,
        };
        let (step, _, position) = search(&mut problem, 1.0, &options()).expect("accepted");
        // direction is -gradient(0) = 5.
        assert_eq!(position.to_bits(), (0.0 + step * 5.0).to_bits());
    }

    #[test]
    fn a_tiny_initial_step_is_expanded_until_it_brackets() {
        let mut problem = Scalar {
            curvature: 1.0,
            slope: -1.0,
            evaluations: 0,
        };
        let (step, _, _) = search(&mut problem, 1.0e-12, &options()).expect("accepted");
        assert!(step > 1.0e-6, "step {step} was never expanded");
    }

    #[test]
    fn the_bracketing_budget_spans_the_whole_expansion_range() {
        // The budget bounds work; it must not silently bound the inputs. One
        // doubling per step means it has to cover log2(MAX_STEP / tiny).
        let span = (MAX_STEP / 1.0e-12_f64).log2().ceil();
        assert!(
            options().max_line_search_steps() as f64 >= span,
            "budget {} cannot expand across {span} doublings",
            options().max_line_search_steps()
        );
    }

    #[test]
    fn an_oversized_initial_step_is_zoomed_back_down() {
        let mut problem = Scalar {
            curvature: 1.0,
            slope: -1.0,
            evaluations: 0,
        };
        let (step, _, _) = search(&mut problem, 1.0e8, &options()).expect("accepted");
        assert!(step < 1.0e6, "step {step} was never reduced");
    }

    #[test]
    fn an_objective_with_no_minimum_along_the_direction_is_refused() {
        let mut problem = Scalar {
            curvature: 0.0,
            slope: -1.0,
            evaluations: 0,
        };
        assert_eq!(
            search(&mut problem, 1.0, &options()),
            Err(LineSearchFailure::Exhausted)
        );
    }

    #[test]
    fn a_non_finite_trial_is_refused_rather_than_accepted() {
        struct Exploding;
        impl Problem for Exploding {
            fn dimension(&self) -> usize {
                1
            }
            fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
                gradient[0] = if point[0] > 0.5 { f64::NAN } else { -1.0 };
                if point[0] > 0.5 { f64::NAN } else { -point[0] }
            }
        }
        let point = [0.0_f64];
        let direction = [1.0_f64];
        let mut trial_point = [0.0_f64];
        let mut trial_gradient = [0.0_f64];
        let mut buffers = LineSearchBuffers {
            trial_point: &mut trial_point,
            trial_gradient: &mut trial_gradient,
        };
        let outcome = strong_wolfe(
            &mut Exploding,
            &point,
            &direction,
            &LineSearchStart {
                value: 0.0,
                slope: -1.0,
                initial_step: 1.0,
            },
            &mut buffers,
            &options(),
        );
        assert_eq!(outcome, Err(LineSearchFailure::NonFinite));
    }

    #[test]
    fn a_collapsed_bracket_is_reported_rather_than_looped_on() {
        // A zoom budget large enough to exhaust an f64 bracket must terminate
        // on the adjacent-endpoint test, not on the counter.
        struct Cliff;
        impl Problem for Cliff {
            fn dimension(&self) -> usize {
                1
            }
            fn value_and_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> f64 {
                // Decreases with an ever-steepening slope, so no step ever
                // satisfies the curvature condition inside the bracket.
                gradient[0] = -1.0 / (point[0] + f64::MIN_POSITIVE);
                -(point[0] + f64::MIN_POSITIVE).ln()
            }
        }
        let point = [1.0_f64];
        let direction = [-1.0_f64];
        let mut trial_point = [0.0_f64];
        let mut trial_gradient = [0.0_f64];
        let mut buffers = LineSearchBuffers {
            trial_point: &mut trial_point,
            trial_gradient: &mut trial_gradient,
        };
        let outcome = strong_wolfe(
            &mut Cliff,
            &point,
            &direction,
            &LineSearchStart {
                value: 0.0,
                slope: -1.0,
                initial_step: 1.0,
            },
            &mut buffers,
            &options(),
        );
        assert!(outcome.is_err());
    }

    impl PartialEq for LineSearchOutcome {
        fn eq(&self, other: &Self) -> bool {
            self.step == other.step && self.value == other.value
        }
    }

    impl std::fmt::Debug for LineSearchOutcome {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("LineSearchOutcome")
                .field("step", &self.step)
                .field("value", &self.value)
                .finish()
        }
    }
}
