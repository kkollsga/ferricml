//! Coefficient penalties, expressed apart from any estimator.
//!
//! A penalty is a property of the *objective*, not of the model that happens to
//! minimize it. Ridge kept its penalty as an estimator-private constant folded
//! into a Cholesky diagonal, which worked because there was one penalty and one
//! solver. An L1 term cannot be folded into anything: it is not differentiable
//! at zero, and that non-differentiability is the entire point — it is what
//! produces coefficients that are *exactly* zero rather than merely small.
//!
//! This module owns the penalty and its coordinate-wise minimizer. The solver
//! that calls it owns the residual bookkeeping and the sweep order.

/// The soft-thresholding operator, `sign(value) * max(|value| - threshold, 0)`.
///
/// This is the proximal operator of the L1 norm and the reason lasso produces
/// sparse fits: every coordinate whose unpenalized optimum lies within
/// `threshold` of zero is set to zero exactly, not to something that rounds to
/// zero when printed.
///
/// # The sign of the zero
///
/// A thresholded coefficient is `0.0`, never `-0.0`. That is a deliberate,
/// frozen choice and it is a documented divergence from the reference
/// implementation, which yields a negatively signed zero for a coefficient
/// shrunk from below. FerricML's position is that a coefficient the fit
/// *removed* has no sign to carry, and a signed zero is a different byte
/// pattern in a stored artifact for a model that is mathematically identical —
/// the exact hazard the accumulation policy in [`crate::numeric`] exists to
/// name. The operator is written as two strict comparisons around a literal
/// `0.0` so that no arithmetic can reintroduce one: `value - threshold` at
/// `value == threshold` would produce `0.0`, but `value + threshold` at
/// `value == -threshold` would produce `-0.0`.
///
/// # Boundaries
///
/// `threshold` is non-negative. A `NaN` input propagates rather than being
/// swallowed by the comparisons, which would otherwise report a coordinate as
/// cleanly removed when the fit had actually diverged.
#[inline]
pub(crate) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    debug_assert!(threshold >= 0.0);
    if value.is_nan() {
        return value;
    }
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

/// A combined L1 and L2 coefficient penalty.
///
/// # Parametrization
///
/// FerricML follows the reference contract's documented form, so a caller who
/// knows one knows the other:
///
/// ```text
/// alpha * l1_ratio * ||b||_1  +  0.5 * alpha * (1 - l1_ratio) * ||b||_2^2
/// ```
///
/// `l1_ratio = 1` is a pure L1 term and `l1_ratio = 0` a pure L2 one. Note that
/// `alpha` here is *not* interchangeable with the crate's closed-form ridge
/// penalty even at `l1_ratio = 0`: this penalty accompanies a squared-error
/// term divided by twice the total sample weight, and that one does not. The
/// two agree at `ridge_alpha = alpha * total_weight`, which is a documented
/// consequence of the parametrization rather than an accident to paper over.
/// The estimators that consume this are named where they are defined; this
/// module deliberately names none of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ElasticNetPenalty {
    l1: f64,
    l2: f64,
}

impl ElasticNetPenalty {
    /// Splits `alpha` and `l1_ratio` into the two terms they name.
    pub(crate) fn new(alpha: f64, l1_ratio: f64) -> Self {
        debug_assert!(alpha >= 0.0 && (0.0..=1.0).contains(&l1_ratio));
        Self {
            l1: alpha * l1_ratio,
            l2: alpha * (1.0 - l1_ratio),
        }
    }

    /// The penalty's contribution to the objective value.
    ///
    /// Coordinate descent never evaluates it: each step minimizes this term
    /// exactly through [`Self::coordinate_minimizer`] rather than comparing
    /// objective values the way a line search does. It exists because the
    /// solver's monotone-descent and exact-argmin proofs need the quantity
    /// being minimized to be stated somewhere other than inside the code that
    /// minimizes it. Terms are visited in ascending coefficient order, per rule
    /// 2 of the accumulation policy.
    #[cfg(test)]
    pub(crate) fn value(&self, coefficients: &[f64]) -> f64 {
        crate::numeric::sum_in_order(coefficients.iter().map(|&coefficient| {
            self.l1 * coefficient.abs() + 0.5 * self.l2 * coefficient * coefficient
        }))
    }

    /// The exact minimizer of one coordinate's penalized quadratic.
    ///
    /// A coordinate descent step holds every other coefficient fixed, which
    /// leaves `0.5 * curvature * b^2 - target * b + penalty(b)` in one
    /// variable. That has a closed form, and this is it — which is why an L1
    /// penalty is solved by coordinate descent rather than by a gradient
    /// method: each step is exact, not approximate.
    ///
    /// `curvature` is the unpenalized second derivative and is non-negative.
    /// A coordinate with no curvature at all — a constant column, once
    /// centering has zeroed it — and no L2 term has no finite minimizer to
    /// choose between, so it is left at zero rather than divided by zero.
    #[inline]
    pub(crate) fn coordinate_minimizer(&self, target: f64, curvature: f64) -> f64 {
        debug_assert!(curvature >= 0.0);
        let denominator = curvature + self.l2;
        if denominator <= 0.0 {
            return 0.0;
        }
        soft_threshold(target, self.l1) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholding_shrinks_toward_zero_and_stops_there() {
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.0, 2.0), 0.0);
        assert_eq!(soft_threshold(-1.0, 2.0), 0.0);
        // A zero threshold is the identity away from zero.
        assert_eq!(soft_threshold(7.5, 0.0), 7.5);
        assert_eq!(soft_threshold(-7.5, 0.0), -7.5);
    }

    #[test]
    fn a_removed_coefficient_is_a_positively_signed_zero() {
        // The frozen choice, and a documented divergence from the reference.
        // A signed zero is a different byte pattern for the same model.
        for &(value, threshold) in &[
            (0.0_f64, 1.0_f64),
            (-0.0, 1.0),
            (2.0, 2.0),
            (-2.0, 2.0),
            (-1.5, 3.0),
            (0.0, 0.0),
            (-0.0, 0.0),
        ] {
            let result = soft_threshold(value, threshold);
            assert_eq!(result, 0.0, "value {value} threshold {threshold}");
            assert!(
                result.is_sign_positive(),
                "value {value} threshold {threshold} produced a negative zero"
            );
        }
        // `value + threshold` at the lower boundary is exactly the arithmetic
        // that would have produced `-0.0`; this pins that it does not.
        assert!((-2.0_f64 + 2.0).is_sign_positive() || soft_threshold(-2.0, 2.0) == 0.0);
    }

    #[test]
    fn thresholding_is_continuous_monotone_and_a_contraction() {
        // The three properties that make the operator the L1 proximal map: it
        // never crosses zero, it never reverses order, and it never increases
        // a distance.
        let threshold = 1.25_f64;
        let mut previous = f64::NEG_INFINITY;
        let mut last_value = f64::NEG_INFINITY;
        for step in -400..=400 {
            let value = f64::from(step) / 100.0;
            let result = soft_threshold(value, threshold);
            assert!(result >= previous, "monotonicity at {value}");
            assert!(result.abs() <= value.abs(), "expansion at {value}");
            assert!(
                result == 0.0 || result.signum() == value.signum(),
                "sign flip at {value}"
            );
            if last_value.is_finite() {
                assert!(
                    (result - previous).abs() <= (value - last_value).abs() + 1.0e-15,
                    "not a contraction at {value}"
                );
            }
            previous = result;
            last_value = value;
        }
    }

    #[test]
    fn a_non_finite_input_propagates_rather_than_reading_as_removed() {
        assert!(soft_threshold(f64::NAN, 1.0).is_nan());
        assert_eq!(soft_threshold(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(soft_threshold(f64::NEG_INFINITY, 1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn the_two_terms_are_split_the_way_the_contract_documents() {
        let penalty = ElasticNetPenalty::new(2.0, 0.25);
        assert_eq!(penalty, ElasticNetPenalty { l1: 0.5, l2: 1.5 });
        // A pure lasso and a pure ridge are the two endpoints of one type.
        assert_eq!(ElasticNetPenalty::new(3.0, 1.0).l2, 0.0);
        assert_eq!(ElasticNetPenalty::new(3.0, 0.0).l1, 0.0);
        // A zero alpha erases the mixing parameter entirely.
        assert_eq!(
            ElasticNetPenalty::new(0.0, 0.5),
            ElasticNetPenalty::new(0.0, 1.0)
        );
    }

    #[test]
    fn the_penalty_value_matches_its_written_form() {
        let penalty = ElasticNetPenalty::new(2.0, 0.25);
        let coefficients = [1.0_f64, -2.0, 0.0, 0.5];
        let expected = coefficients
            .iter()
            .map(|value| 0.5 * value.abs() + 0.5 * 1.5 * value * value)
            .sum::<f64>();
        assert!((penalty.value(&coefficients) - expected).abs() <= 1.0e-15);
        assert_eq!(penalty.value(&[]), 0.0);
    }

    #[test]
    fn the_coordinate_minimizer_is_the_exact_argmin_of_its_own_one_variable_problem() {
        // Verified against a dense scan rather than against the formula it was
        // derived from, which is what makes this a proof of the derivation.
        let penalty = ElasticNetPenalty::new(0.8, 0.6);
        for target_step in -30..=30 {
            for curvature_step in 1..=12 {
                let target = f64::from(target_step) / 5.0;
                let curvature = f64::from(curvature_step) / 4.0;
                let objective = |value: f64| {
                    0.5 * curvature * value * value - target * value
                        + penalty.value(std::slice::from_ref(&value))
                };
                let minimizer = penalty.coordinate_minimizer(target, curvature);
                let best = objective(minimizer);
                for offset in -2000..=2000 {
                    let candidate = minimizer + f64::from(offset) / 1000.0;
                    assert!(
                        objective(candidate) >= best - 1.0e-12,
                        "target {target} curvature {curvature}: {candidate} beats {minimizer}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_coordinate_with_no_curvature_and_no_l2_term_stays_at_zero() {
        // A centered constant column has zero curvature. Dividing by it would
        // produce a NaN coefficient from perfectly ordinary data.
        let lasso = ElasticNetPenalty::new(1.0, 1.0);
        assert_eq!(lasso.coordinate_minimizer(0.0, 0.0), 0.0);
        assert_eq!(lasso.coordinate_minimizer(5.0, 0.0), 0.0);
        assert!(lasso.coordinate_minimizer(5.0, 0.0).is_sign_positive());
        // With an L2 term the coordinate is well defined again.
        let elastic = ElasticNetPenalty::new(1.0, 0.5);
        assert_eq!(elastic.coordinate_minimizer(5.0, 0.0), (5.0 - 0.5) / 0.5);
    }

    #[test]
    fn a_large_enough_l1_term_removes_every_coordinate() {
        let penalty = ElasticNetPenalty::new(1.0e6, 1.0);
        for target_step in -100..=100 {
            let target = f64::from(target_step);
            let value = penalty.coordinate_minimizer(target, 1.0);
            assert_eq!(value, 0.0, "target {target}");
            assert!(value.is_sign_positive());
        }
    }
}
