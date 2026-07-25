//! The differentiable objective contract shared by every FerricML solver.

use super::link::Link;

/// A per-sample objective expressed in raw-score space.
///
/// An objective owns the loss and its derivatives; a solver owns the update
/// rule. Keeping them apart is what lets a new loss reach both the linear and
/// the ensemble solvers without either of them changing.
///
/// # Compile-time dispatch
///
/// Every member is an associated function or constant, and consumers take the
/// objective as a generic parameter. There is deliberately no trait object and
/// no per-row branch on a concrete loss type: a caller that needs to vary its
/// behavior reads a declared property instead of matching on the type.
///
/// # Scaling convention
///
/// FerricML documents its own scaling rather than matching any third party's.
///
/// - [`Objective::value`] is the loss of **one** sample at **one** raw score.
///   Sample weights, regularization, and any reduction over rows belong to the
///   caller, because only the caller knows the accumulation order its fitted
///   artifact depends on.
/// - [`Objective::gradient`] and [`Objective::hessian`] are the first and
///   second derivatives of `value` with respect to `raw`, on exactly that
///   per-sample scale. This is what makes finite differences over `value` a
///   complete proof of both.
/// - [`Objective::negative_gradient`] is `-gradient`, written in whatever form
///   avoids a signed-zero flip. It is a separate member rather than a negation
///   at the call site because `-(a - b)` and `b - a` differ in the sign of zero
///   when `a == b`, and a fitted leaf value of `-0.0` is a different artifact
///   byte pattern from `0.0`.
/// - [`Objective::gradient_and_curvature`] is the solver-facing pair. Its
///   second element is the curvature a solver may consume, which is the
///   hessian floored at [`Objective::CURVATURE_FLOOR`]; the floor keeps a
///   Newton system positive definite where the true curvature collapses.
pub(crate) trait Objective {
    /// The mean function this objective compares its target against.
    type Link: Link;

    /// Whether the hessian is the same for every sample and every raw score.
    ///
    /// A consumer that accumulates node statistics uses this to replace a
    /// per-sample hessian sum with a row count times the constant, which is
    /// both cheaper and exactly equal.
    const CONSTANT_HESSIAN: bool;

    /// Whether the curvature a solver consumes may differ from the exact
    /// second derivative.
    ///
    /// True exactly when [`Objective::CURVATURE_FLOOR`] is positive, since the
    /// floor is the only approximation this contract permits today.
    const APPROX_HESSIAN: bool;

    /// Whether one sample carries more than one raw score.
    ///
    /// Every objective in the crate is single-output today. A solver that
    /// cannot handle several raw scores per sample asserts this at compile
    /// time, so a future multiclass objective fails to build against it rather
    /// than fitting nonsense.
    const IS_MULTICLASS: bool;

    /// Lower bound applied to the hessian before a solver consumes it.
    const CURVATURE_FLOOR: f64;

    /// The loss of one sample at one raw score.
    ///
    /// This is the defining quantity of the objective: `gradient` and `hessian`
    /// are its derivatives and are verified against it by finite differences.
    /// It carried no solver consumer while the only update rules were Newton's
    /// and the histogram grower's, neither of which tests a loss value; the
    /// line search inside [`crate::optimize`] is that consumer, because
    /// deciding how far to step means comparing the objective at two points.
    fn value(raw: f64, target: f64) -> f64;

    /// First derivative of [`Objective::value`] with respect to `raw`.
    fn gradient(raw: f64, target: f64) -> f64;

    /// Second derivative of [`Objective::value`] with respect to `raw`.
    fn hessian(raw: f64, target: f64) -> f64;

    /// `-gradient`, in the form that keeps the sign of an exact zero positive.
    fn negative_gradient(raw: f64, target: f64) -> f64;

    /// The gradient and the floored curvature, evaluated together.
    ///
    /// An objective whose two derivatives share an expensive intermediate —
    /// the inverse link, typically — overrides this so a solver's inner loop
    /// evaluates that intermediate once per row instead of twice.
    fn gradient_and_curvature(raw: f64, target: f64) -> (f64, f64) {
        (
            Self::gradient(raw, target),
            Self::hessian(raw, target).max(Self::CURVATURE_FLOOR),
        )
    }
}

#[cfg(test)]
pub(crate) mod proof {
    //! Shared finite-difference battery for objective implementations.

    use super::Objective;

    /// Asserts that `gradient` and `hessian` really are the derivatives of
    /// `value`.
    ///
    /// Central differences are used in both orders, so the truncation error is
    /// second order in the step and the comparison is meaningful at the stated
    /// tolerances. The caller supplies the raw scores to probe, because a
    /// saturating objective has regions where the loss is flat to within double
    /// precision and a finite difference there proves nothing.
    pub(crate) fn finite_differences_agree<O: Objective>(
        raws: &[f64],
        targets: &[f64],
        gradient_tolerance: f64,
        hessian_tolerance: f64,
    ) {
        for &target in targets {
            for &raw in raws {
                let first_step = 1.0e-6_f64.mul_add(raw.abs(), 1.0e-6);
                let numeric_gradient = (O::value(raw + first_step, target)
                    - O::value(raw - first_step, target))
                    / (2.0 * first_step);
                let gradient = O::gradient(raw, target);
                assert!(
                    (numeric_gradient - gradient).abs() <= gradient_tolerance,
                    "gradient at raw={raw} target={target}: {gradient} vs finite difference \
                     {numeric_gradient}"
                );

                let second_step = 1.0e-3_f64.mul_add(raw.abs(), 1.0e-3);
                let numeric_hessian = (O::value(raw + second_step, target)
                    - 2.0 * O::value(raw, target)
                    + O::value(raw - second_step, target))
                    / (second_step * second_step);
                let hessian = O::hessian(raw, target);
                assert!(
                    (numeric_hessian - hessian).abs() <= hessian_tolerance,
                    "hessian at raw={raw} target={target}: {hessian} vs finite difference \
                     {numeric_hessian}"
                );
            }
        }
    }

    /// Asserts the members defined in terms of each other stay consistent.
    pub(crate) fn declared_properties_are_coherent<O: Objective>(raws: &[f64], targets: &[f64]) {
        assert_eq!(
            O::APPROX_HESSIAN,
            O::CURVATURE_FLOOR > 0.0,
            "an approximate hessian is declared exactly when the curvature is floored"
        );
        assert!(!O::IS_MULTICLASS, "no multiclass objective exists yet");
        let mut constant_hessian = None;
        for &target in targets {
            for &raw in raws {
                let gradient = O::gradient(raw, target);
                let hessian = O::hessian(raw, target);
                let negative = O::negative_gradient(raw, target);
                assert_eq!(
                    negative, -gradient,
                    "negative gradient at raw={raw} target={target}"
                );
                if negative == 0.0 {
                    assert!(
                        negative.is_sign_positive(),
                        "an exactly zero negative gradient stays positively signed"
                    );
                }
                if O::CONSTANT_HESSIAN {
                    let first = *constant_hessian.get_or_insert(hessian);
                    assert_eq!(
                        hessian.to_bits(),
                        first.to_bits(),
                        "a constant hessian must not vary at raw={raw} target={target}"
                    );
                }
                let (paired_gradient, curvature) = O::gradient_and_curvature(raw, target);
                assert_eq!(
                    paired_gradient.to_bits(),
                    gradient.to_bits(),
                    "paired gradient at raw={raw} target={target}"
                );
                assert_eq!(
                    curvature.to_bits(),
                    hessian.max(O::CURVATURE_FLOOR).to_bits(),
                    "paired curvature at raw={raw} target={target}"
                );
            }
        }
    }
}
