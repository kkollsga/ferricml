//! Binary log-loss over the logit link.

use super::link::{Link, Logit};
use super::objective::Objective;
use crate::numeric::log_sum_exp;

/// Binary cross-entropy of a `0`/`1` target against a logit-linked score.
///
/// The loss of one sample is `log(1 + exp(raw)) - target * raw`, which is the
/// negative log-likelihood `-(y log p + (1 - y) log(1 - p))` rewritten so that
/// no probability is ever taken through a logarithm. That matters at the
/// saturation boundary: `p` reaches exactly `0` or `1` for a raw score of large
/// magnitude, and the probability form would return an infinity there while the
/// raw form stays finite and exact.
///
/// FerricML scales the loss per sample and does not average, halve, or add a
/// regularization term inside the objective; those belong to the solver.
pub(crate) enum BinaryLogLoss {}

impl Objective for BinaryLogLoss {
    type Link = Logit;

    const CONSTANT_HESSIAN: bool = false;
    const APPROX_HESSIAN: bool = true;
    const IS_MULTICLASS: bool = false;
    /// The curvature `p * (1 - p)` collapses to zero once the fitted score
    /// separates a row confidently, which would make a Newton system singular.
    /// Flooring it keeps the Cholesky factorization defined and turns the step
    /// into a damped one instead of a failure.
    const CURVATURE_FLOOR: f64 = 1.0e-12;

    fn value(raw: f64, target: f64) -> f64 {
        log_sum_exp(&[0.0, raw]) - target * raw
    }

    fn gradient(raw: f64, target: f64) -> f64 {
        Self::Link::inverse(raw) - target
    }

    fn hessian(raw: f64, _target: f64) -> f64 {
        let probability = Self::Link::inverse(raw);
        probability * (1.0 - probability)
    }

    fn negative_gradient(raw: f64, target: f64) -> f64 {
        target - Self::Link::inverse(raw)
    }

    fn gradient_and_curvature(raw: f64, target: f64) -> (f64, f64) {
        let probability = Self::Link::inverse(raw);
        (
            probability - target,
            (probability * (1.0 - probability)).max(Self::CURVATURE_FLOOR),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::objective::proof::{
        declared_properties_are_coherent, finite_differences_agree,
    };

    const TARGETS: [f64; 2] = [0.0, 1.0];

    fn probed_raw_scores() -> Vec<f64> {
        // Kept inside the range where the loss still varies at double
        // precision; past roughly +-35 the sigmoid saturates and a finite
        // difference of the value would prove nothing.
        (-300..=300).map(|step| f64::from(step) / 10.0).collect()
    }

    #[test]
    fn derivatives_agree_with_finite_differences() {
        finite_differences_agree::<BinaryLogLoss>(&probed_raw_scores(), &TARGETS, 1.0e-6, 1.0e-6);
    }

    #[test]
    fn declared_properties_stay_coherent() {
        declared_properties_are_coherent::<BinaryLogLoss>(&probed_raw_scores(), &TARGETS);
    }

    #[test]
    fn value_matches_the_probability_form_where_that_form_is_defined() {
        // Only inside +-15: past that the probability itself is within an ulp
        // of `1`, so `1 - p` cancels and the probability form loses relative
        // accuracy long before it diverges outright.
        for &target in &TARGETS {
            for step in -150..=150 {
                let raw = f64::from(step) / 10.0;
                let probability = Logit::inverse(raw);
                let naive =
                    -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln());
                let value = BinaryLogLoss::value(raw, target);
                assert!(
                    (value - naive).abs() <= 1.0e-8 * naive.abs().max(1.0),
                    "value at raw={raw} target={target}: {value} vs {naive}"
                );
            }
        }
    }

    #[test]
    fn value_stays_finite_where_the_probability_form_diverges() {
        // A confidently wrong score saturates the sigmoid, so `ln(p)` is
        // `-inf`. The raw-score form grows linearly instead.
        for &raw in &[40.0_f64, 100.0, 1.0e6, 1.0e300] {
            let wrong = BinaryLogLoss::value(raw, 0.0);
            assert!(wrong.is_finite(), "value at raw={raw} target=0");
            assert!((wrong - raw).abs() <= 1.0e-9 * raw, "softplus at raw={raw}");
            assert_eq!(Logit::inverse(raw).ln(), 0.0);
            assert_eq!((1.0 - Logit::inverse(raw)).ln(), f64::NEG_INFINITY);

            let right = BinaryLogLoss::value(-raw, 0.0);
            assert!(right.is_finite() && right >= 0.0, "value at raw={}", -raw);
        }
    }

    #[test]
    fn saturated_curvature_is_floored_but_the_hessian_is_not() {
        let raw = 100.0;
        assert_eq!(BinaryLogLoss::hessian(raw, 1.0), 0.0);
        let (gradient, curvature) = BinaryLogLoss::gradient_and_curvature(raw, 1.0);
        assert_eq!(gradient, 0.0);
        assert_eq!(curvature, BinaryLogLoss::CURVATURE_FLOOR);
    }
}
