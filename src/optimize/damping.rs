//! Armijo backtracking: the step length that makes an exact Newton step safe.
//!
//! An exact Newton step is the right *direction* and the wrong *length*. On a
//! convex objective the step solves the local quadratic model exactly, and near
//! the minimum that model is the objective, so the full step is what gives
//! Newton's method its quadratic tail. Far from the minimum the model is a
//! guess, and on a badly scaled near-separable logistic design it is a bad
//! one: the full step overshoots, the next model is built at a worse point, and
//! the iteration walks away from the minimum instead of towards it. The
//! undamped exact step is not globally convergent, and that is a property of the
//! method rather than of any particular input.
//!
//! This module owns the standard remedy — accept the full step when it reduces
//! the objective enough, and otherwise halve until it does — as a seam rather
//! than a per-estimator loop, because two Newton paths already need it and a
//! third would otherwise copy it.
//!
//! # Why this and not the strong-Wolfe search next door
//!
//! [`super::line_search`] implements a bounded strong-Wolfe search, and it
//! would work here. It is deliberately *not* what the Newton paths consume, for
//! two reasons that are about the method rather than about convenience.
//!
//! The Wolfe **curvature** condition exists to guarantee `s . y > 0`, which is
//! what keeps L-BFGS's stored inverse-Hessian approximation positive definite.
//! A Newton path stores no curvature pairs: it rebuilds the exact Hessian from
//! the data at every iteration and factorizes it, and the factorization
//! succeeding *is* the positive-definiteness certificate. So the condition
//! guards an invariant this consumer does not have, and enforcing it would
//! reject full steps that are already correct. Measured over 1,600 ordinary
//! binary fits the curvature condition rejects the exact step on none of them
//! and sufficient decrease rejects it on 108, so on well-conditioned data the
//! two rules agree; over the ill-conditioned region the curvature condition
//! rejects 273 of 972 where sufficient decrease rejects 432, and the extra
//! rejections buy no fit that sufficient decrease alone does not already
//! rescue.
//!
//! Second, the strong-Wolfe search reports a refusal of its own when a budget
//! runs out. Its consumer's whole answer is that step, so that is the right
//! design there. A Newton path already has an acceptance test for an iterate it
//! could not improve — the Newton decrement — and routing a stalled step length
//! into a second, differently-motivated error would give one condition two
//! names.
//!
//! # Determinism
//!
//! The trial sequence is `1, 1/2, 1/4, ...`: exact powers of two, each an
//! exactly representable `f64`, and the trial point is one multiplication and
//! one subtraction per coordinate. Nothing here interpolates, so no trial
//! depends on a fitted polynomial's rounding, and the sequence does not even
//! depend on a bracket — only on the halving index. The acceptance test is a
//! comparison of two `f64` values, which IEEE-754 evaluates exactly. The number
//! of halvings is therefore a function of the data, the parameters, the seed and
//! the thread count alone, which is what rule 2 of the accumulation policy in
//! [`crate::numeric`] requires of an iterative solver.

/// Armijo sufficient-decrease constant.
///
/// The conventional value, and the same one [`super::line_search`] uses. It has
/// to be small enough that a Newton-scale full step is normally accepted, which
/// is the property that keeps damping from moving fits that never needed it.
const SUFFICIENT_DECREASE: f64 = 1.0e-4;

/// Halvings before a step length is abandoned.
///
/// `2^-60` is far below the point at which a trial point stops differing from
/// its origin in `f64`, so exhausting this budget means no representable step
/// along the direction decreases the objective, not that the search was too
/// short.
const MAX_HALVINGS: usize = 60;

/// The step length an Armijo backtracking search accepted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DampedStep {
    /// Fraction of the full step taken; `1.0` when the full step was accepted.
    pub(crate) factor: f64,
    /// Objective value at the accepted point, so the caller's next iteration
    /// does not have to re-evaluate it.
    pub(crate) value: f64,
}

/// Moves `point` along `-step` by the largest halving that decreases `value_at`.
///
/// `decrement` is the directional derivative's magnitude at `point`, which for
/// a Newton step is `gradient . step` and is strictly positive whenever the
/// factorized curvature was positive definite — so `-step` is a descent
/// direction and a short enough move along it must decrease the objective.
///
/// Returns `None` when no representable halving decreases the objective, which
/// leaves `point` untouched. That is not a failure to report on its own: it
/// means the iterate cannot be improved along this direction, and the caller's
/// own acceptance test is what decides whether being unable to improve is
/// convergence or non-convergence.
///
/// `trial` is caller-owned scratch of the same length as `point`, so a whole
/// solve allocates nothing after its workspace.
pub(crate) fn armijo_backtracking(
    point: &mut [f64],
    step: &[f64],
    trial: &mut [f64],
    value: f64,
    decrement: f64,
    mut value_at: impl FnMut(&[f64]) -> f64,
) -> Option<DampedStep> {
    debug_assert_eq!(point.len(), step.len());
    debug_assert_eq!(point.len(), trial.len());
    let mut factor = 1.0_f64;
    for _ in 0..MAX_HALVINGS {
        for (slot, (&start, &delta)) in trial.iter_mut().zip(point.iter().zip(step)) {
            *slot = start - factor * delta;
        }
        let candidate = value_at(trial);
        // Strict decrease *and* the Armijo bound. The bound alone is not enough
        // at the bottom of the halving sequence: once `factor` is small enough
        // that `SUFFICIENT_DECREASE * factor * decrement` underflows relative to
        // `value`, the right-hand side rounds back to `value` exactly and the
        // comparison starts accepting steps that achieve nothing — including, at
        // the very bottom, a trial point bit-identical to the origin. Requiring
        // the objective to actually go down costs a genuine descent nothing,
        // because a genuine descent is strictly lower.
        //
        // A non-finite candidate fails both comparisons rather than being
        // branched on separately, which is also what makes a non-finite `value`
        // or `decrement` refuse instead of accepting anything.
        if candidate < value && candidate <= value - SUFFICIENT_DECREASE * factor * decrement {
            point.copy_from_slice(trial);
            return Some(DampedStep {
                factor,
                value: candidate,
            });
        }
        factor *= 0.5;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0.5 * curvature * x^2` in one coordinate, whose exact Newton step from
    /// any point lands on the minimum, so the full step must be accepted.
    #[test]
    fn an_exact_step_on_a_quadratic_is_accepted_whole() {
        let mut point = [3.0_f64];
        let mut trial = [0.0_f64];
        let value_at = |probe: &[f64]| 0.5 * probe[0] * probe[0];
        let step = [3.0_f64];
        let decrement = 3.0 * 3.0;
        let outcome = armijo_backtracking(
            &mut point,
            &step,
            &mut trial,
            value_at(&[3.0]),
            decrement,
            value_at,
        )
        .expect("a quadratic's exact step decreases the objective");
        assert_eq!(outcome.factor, 1.0);
        assert_eq!(point, [0.0]);
        assert_eq!(outcome.value, 0.0);
    }

    /// A step far longer than the objective supports is halved until it fits,
    /// and the accepted factor is an exact power of two.
    #[test]
    fn an_overlong_step_is_halved_to_a_power_of_two() {
        // `x^4` in one coordinate: gradient `4 x^3`, so a step of 16 from
        // `x = 1` overshoots the minimum at zero by a factor of fifteen and
        // lands at a far worse point.
        let value_at = |probe: &[f64]| probe[0] * probe[0] * probe[0] * probe[0];
        let mut point = [1.0_f64];
        let mut trial = [0.0_f64];
        let step = [16.0_f64];
        let decrement = 4.0 * 16.0;
        let outcome = armijo_backtracking(
            &mut point,
            &step,
            &mut trial,
            value_at(&[1.0]),
            decrement,
            value_at,
        )
        .expect("halving reaches a decreasing step");
        assert!(
            outcome.factor < 1.0,
            "factor {} was not damped",
            outcome.factor
        );
        assert_eq!(
            outcome.factor,
            outcome.factor.recip().recip(),
            "an exact power of two survives a round trip through its reciprocal"
        );
        assert_eq!(outcome.factor.log2(), outcome.factor.log2().trunc());
        assert!(
            outcome.value < 1.0,
            "value {} did not decrease",
            outcome.value
        );
    }

    /// A direction that does not descend is refused rather than stepped along,
    /// and the point is left exactly where it was.
    ///
    /// This is also the test that pins the strict-decrease half of the
    /// acceptance rule. Stepping along `+x` from `x = 1` on `0.5 x^2` raises the
    /// objective at every length, so the Armijo bound should reject every
    /// halving — but once `factor` reaches `2^-53` the trial point rounds back
    /// to `1.0` exactly, the candidate value is exactly the base value, and the
    /// required decrease `1e-4 * factor * decrement` underflows to nothing when
    /// subtracted from it. Dropping `candidate < value` from the rule makes this
    /// accept a factor of `1.11e-16` and a step that moves nowhere.
    #[test]
    fn a_non_descent_direction_is_refused_and_leaves_the_point_alone() {
        let value_at = |probe: &[f64]| 0.5 * probe[0] * probe[0];
        let mut point = [1.0_f64];
        let mut trial = [0.0_f64];
        let step = [-1.0_f64];
        assert_eq!(
            armijo_backtracking(
                &mut point,
                &step,
                &mut trial,
                value_at(&[1.0]),
                1.0,
                value_at
            ),
            None
        );
        assert_eq!(point, [1.0]);
    }

    /// A zero step is refused rather than accepted as a decrease.
    ///
    /// An iterate already at the minimum produces an exactly zero Newton step,
    /// and every trial point along it is the origin. Accepting one would report
    /// progress that did not happen and let the caller's loop spend its whole
    /// budget standing still.
    #[test]
    fn a_zero_step_is_refused() {
        let value_at = |probe: &[f64]| 0.5 * probe[0] * probe[0];
        let mut point = [0.0_f64];
        let mut trial = [0.0_f64];
        assert_eq!(
            armijo_backtracking(&mut point, &[0.0], &mut trial, 0.0, 0.0, value_at),
            None
        );
        assert_eq!(point, [0.0]);
    }

    /// A non-finite objective at every trial is refused by the comparison, not
    /// accepted as an improvement.
    #[test]
    fn a_non_finite_objective_is_refused() {
        let mut point = [1.0_f64];
        let mut trial = [0.0_f64];
        assert_eq!(
            armijo_backtracking(&mut point, &[1.0], &mut trial, 1.0, 1.0, |_| f64::NAN),
            None
        );
        assert_eq!(point, [1.0]);
        assert_eq!(
            armijo_backtracking(&mut point, &[1.0], &mut trial, 1.0, 1.0, |_| f64::INFINITY),
            None
        );
        assert_eq!(point, [1.0]);
    }

    /// The same inputs produce the same accepted factor every time, including
    /// the number of halvings taken to reach it.
    #[test]
    fn the_accepted_factor_is_reproducible() {
        let value_at = |probe: &[f64]| probe[0] * probe[0] * probe[0] * probe[0];
        let run = || {
            let mut point = [1.0_f64];
            let mut trial = [0.0_f64];
            armijo_backtracking(&mut point, &[16.0], &mut trial, 1.0, 64.0, value_at).map(
                |outcome| {
                    (
                        outcome.factor.to_bits(),
                        outcome.value.to_bits(),
                        point[0].to_bits(),
                    )
                },
            )
        };
        assert_eq!(run(), run());
    }
}
