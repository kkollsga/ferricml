//! Squared error over the identity link.

use super::link::{Identity, Link};
use super::objective::Objective;

/// Half the squared difference between a raw score and its target.
///
/// The halving is FerricML's convention, and it is load-bearing rather than
/// cosmetic: it makes the hessian exactly `1`, so a Newton leaf update reduces
/// to the mean of the negative gradients and the L2 term added to a leaf
/// denominator is measured in whole samples. Doubling the loss would halve
/// every leaf value and silently rescale the meaning of `l2_regularization`.
///
/// The negative gradient of this objective is the familiar residual
/// `target - raw`, which is why a squared-error boosted tree fits residuals.
pub(crate) enum SquaredError {}

impl Objective for SquaredError {
    type Link = Identity;

    const CONSTANT_HESSIAN: bool = true;
    const APPROX_HESSIAN: bool = false;
    const IS_MULTICLASS: bool = false;
    const CURVATURE_FLOOR: f64 = 0.0;

    fn value(raw: f64, target: f64) -> f64 {
        let residual = Self::Link::inverse(raw) - target;
        0.5 * residual * residual
    }

    fn gradient(raw: f64, target: f64) -> f64 {
        Self::Link::inverse(raw) - target
    }

    fn hessian(_raw: f64, _target: f64) -> f64 {
        1.0
    }

    fn negative_gradient(raw: f64, target: f64) -> f64 {
        target - Self::Link::inverse(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::objective::proof::{
        declared_properties_are_coherent, finite_differences_agree,
    };

    fn probed_raw_scores() -> Vec<f64> {
        (-200..=200).map(|step| f64::from(step) / 8.0).collect()
    }

    const TARGETS: [f64; 5] = [-4.0, -0.5, 0.0, 1.25, 9.0];

    #[test]
    fn derivatives_agree_with_finite_differences() {
        finite_differences_agree::<SquaredError>(&probed_raw_scores(), &TARGETS, 1.0e-6, 1.0e-6);
    }

    #[test]
    fn declared_properties_stay_coherent() {
        declared_properties_are_coherent::<SquaredError>(&probed_raw_scores(), &TARGETS);
    }

    #[test]
    fn a_perfect_prediction_has_a_positively_signed_zero_negative_gradient() {
        // `-(raw - target)` would produce `-0.0` here, and a leaf value of
        // `-0.0` is a different artifact byte pattern from `0.0`.
        for &target in &TARGETS {
            assert_eq!(SquaredError::value(target, target), 0.0);
            let negative = SquaredError::negative_gradient(target, target);
            assert_eq!(negative, 0.0);
            assert!(negative.is_sign_positive(), "target {target}");
            assert!((-SquaredError::gradient(target, target)).is_sign_negative());
        }
    }

    #[test]
    fn the_negative_gradient_is_the_residual_at_every_probe() {
        for &target in &TARGETS {
            for &raw in &probed_raw_scores() {
                assert_eq!(
                    SquaredError::negative_gradient(raw, target).to_bits(),
                    (target - raw).to_bits(),
                    "residual at raw={raw} target={target}"
                );
            }
        }
    }
}
