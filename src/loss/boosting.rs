//! Mapping an objective into a boosted tree's leaf statistics.
//!
//! A gradient-boosted tree never sees a loss directly. It sees, per node, a sum
//! of negative gradients and a total hessian, and turns that pair into a leaf
//! value and a split score. This module owns that translation so the grower
//! stays a search over histograms and the objective stays the only place a
//! derivative is written down.

use super::binary_log_loss::BinaryLogLoss;
use super::objective::Objective;
use super::squared_error::SquaredError;
use crate::numeric::sum_in_order;

/// What a boosted tree needs from an objective beyond the shared contract.
///
/// The [`Objective`] contract is deliberately per-sample and solver-agnostic.
/// A histogram grower needs one thing more that is specific to *this* consumer
/// — how a node's second-order denominator is formed, which is the only place a
/// constant hessian and a varying one differ — and stating it here keeps it out
/// of the crate-wide contract every other solver would then have to satisfy.
///
/// The member is an associated function, so a grower resolves it at compile
/// time and no per-row branch on the concrete loss can appear by accident.
pub(crate) trait BoostingObjective: Objective {
    /// The name this objective goes by inside a persisted boosting artifact.
    ///
    /// A boosted model's leaf values are only meaningful against the loss they
    /// were fitted to descend, so the artifact records which loss that was. The
    /// numbering is **crate-wide** rather than per estimator: an estimator kind
    /// already separates one model type from another, and this separates one
    /// *objective* from another inside a kind that carries more than one — the
    /// forward-compatibility hook the kind cannot provide. A decoder requires
    /// exactly its own value, so the two discriminators are independent and a
    /// crossed artifact fails both.
    ///
    /// Values are permanent. Squared error is `1` because that is what
    /// `HistGradientBoostingRegressor` has always written, and no value is ever
    /// reused for a different loss.
    const ARTIFACT_OBJECTIVE_TAG: u32;

    /// Second-order denominator of one node, before L2 regularization.
    ///
    /// `weight` is the node's total sample weight and `hessian_sum` is the
    /// weighted sum of its per-sample hessians. An objective whose hessian is
    /// constant reads the first and ignores the second, which is exactly what
    /// lets its grower skip accumulating a hessian histogram at all; an
    /// objective with a varying hessian does the reverse.
    ///
    /// Expressing the choice as an associated function rather than a branch on
    /// [`Objective::CONSTANT_HESSIAN`] is deliberate: the constant arm's
    /// `weight * hessian` is only *meaningful* where the hessian really is
    /// constant, and a trait implementation is where that precondition can be
    /// stated once instead of asserted at every call.
    fn node_hessian_total(weight: f64, hessian_sum: f64) -> f64;
}

impl BoostingObjective for SquaredError {
    const ARTIFACT_OBJECTIVE_TAG: u32 = 1;

    fn node_hessian_total(weight: f64, _hessian_sum: f64) -> f64 {
        constant_hessian_total::<Self>(weight)
    }
}

impl BoostingObjective for BinaryLogLoss {
    const ARTIFACT_OBJECTIVE_TAG: u32 = 2;

    fn node_hessian_total(_weight: f64, hessian_sum: f64) -> f64 {
        hessian_sum
    }
}

/// Sum of the negative gradients carried by one node's samples.
///
/// The reduction visits `samples` in the order the caller holds them, which is
/// ascending row order for a freshly split node, and widens each `f32` term
/// under rule 1 of the accumulation policy in [`crate::numeric`].
///
/// A sample weight scales that sample's gradient, which is what makes a weight
/// of `k` the same contribution as `k` copies of the row. The unweighted arm is
/// separate rather than a weight of one, so an unweighted fit performs exactly
/// the multiplications it always did.
#[inline]
pub(crate) fn negative_gradient_sum(
    samples: &[usize],
    negative_gradients: &[f32],
    sample_weights: Option<&[f32]>,
) -> f64 {
    weighted_sample_sum(samples, negative_gradients, sample_weights)
}

/// Weighted sum of the hessians carried by one node's samples.
///
/// This is the other half of the statistic pair a Newton step needs, and it is
/// accumulated only for an objective whose hessian actually varies — a constant
/// one is recovered from the node's weight total instead, which is both cheaper
/// and exactly equal. The reduction is the same one
/// [`negative_gradient_sum`] performs, so the two statistics of one node are
/// summed in the same order and a weight of `k` remains the contribution of `k`
/// copies of the row for both.
#[inline]
pub(crate) fn hessian_sum(
    samples: &[usize],
    hessians: &[f32],
    sample_weights: Option<&[f32]>,
) -> f64 {
    weighted_sample_sum(samples, hessians, sample_weights)
}

/// The one reduction both node statistics use.
///
/// Written once so the gradient and hessian halves of a node cannot drift into
/// different accumulation orders, which would make a fitted leaf depend on
/// which statistic a future edit touched.
#[inline]
fn weighted_sample_sum(samples: &[usize], values: &[f32], sample_weights: Option<&[f32]>) -> f64 {
    match sample_weights {
        None => sum_in_order(samples.iter().map(|&sample| f64::from(values[sample]))),
        Some(weights) => sum_in_order(
            samples
                .iter()
                .map(|&sample| f64::from(weights[sample]) * f64::from(values[sample])),
        ),
    }
}

/// Total hessian of samples carrying `weight` in total.
///
/// A constant hessian is exactly the property that lets a grower carry one
/// weight total where a general objective would have to carry a second
/// per-sample histogram. The constant is read from the objective rather than
/// assumed, so an objective that scales its loss differently scales the leaf
/// denominator with it. Unweighted, `weight` is the node's row count.
pub(crate) fn constant_hessian_total<O: Objective>(weight: f64) -> f64 {
    const {
        assert!(
            O::CONSTANT_HESSIAN,
            "a per-sample hessian cannot be recovered from a weight total"
        );
    }
    // Any point evaluates the same, the hessian being constant by declaration.
    weight * O::hessian(0.0, 0.0)
}

/// Newton-optimal constant prediction for one leaf.
///
/// Minimizing the objective's second-order expansion over a leaf gives
/// `-G / (H + lambda)`, written here with the negated gradient sum so an
/// exactly zero leaf keeps a positively signed zero.
#[inline]
pub(crate) fn newton_leaf_value(
    negative_gradient_sum: f64,
    hessian_total: f64,
    l2_regularization: f32,
) -> f32 {
    (negative_gradient_sum / (hessian_total + f64::from(l2_regularization))) as f32
}

/// Score of one candidate node in the split search.
///
/// This is the loss reduction that node's Newton-optimal value achieves, up to
/// the factor of one half that is common to every term of a gain and therefore
/// never affects which split wins. A split's gain is the two child scores minus
/// the parent's.
#[inline]
pub(crate) fn newton_split_score(
    negative_gradient_sum: f64,
    hessian_total: f64,
    l2_regularization: f32,
) -> f64 {
    negative_gradient_sum * negative_gradient_sum / (hessian_total + f64::from(l2_regularization))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::{BinaryLogLoss, SquaredError};
    use crate::numeric::sigmoid_f64;

    #[test]
    fn a_constant_hessian_total_is_the_weight_total_for_squared_error() {
        for weight in [0.0_f64, 1.0, 7.0, 1_000_000.0, 2.5] {
            assert_eq!(
                constant_hessian_total::<SquaredError>(weight).to_bits(),
                weight.to_bits()
            );
        }
    }

    #[test]
    fn the_leaf_value_is_the_regularized_mean_of_the_negative_gradients() {
        let negative_gradients = [2.0_f32, 4.0, -3.0, 9.0];
        let samples = [0_usize, 1, 2];
        let sum = negative_gradient_sum(&samples, &negative_gradients, None);
        assert_eq!(sum, 3.0);
        let total = constant_hessian_total::<SquaredError>(samples.len() as f64);
        assert_eq!(newton_leaf_value(sum, total, 0.0), 1.0);
        assert_eq!(newton_leaf_value(sum, total, 3.0), 0.5);
    }

    #[test]
    fn an_exactly_balanced_leaf_keeps_a_positively_signed_zero() {
        let negative_gradients = [1.5_f32, -1.5];
        let samples = [0_usize, 1];
        let sum = negative_gradient_sum(&samples, &negative_gradients, None);
        let value = newton_leaf_value(sum, constant_hessian_total::<SquaredError>(2.0), 0.0);
        assert_eq!(value, 0.0);
        assert!(value.is_sign_positive());
    }

    #[test]
    fn a_split_that_separates_the_gradients_scores_above_its_parent() {
        let negative_gradients = [-2.0_f32, -2.0, 3.0, 3.0];
        let all = [0_usize, 1, 2, 3];
        let left = [0_usize, 1];
        let right = [2_usize, 3];
        let score = |samples: &[usize]| {
            newton_split_score(
                negative_gradient_sum(samples, &negative_gradients, None),
                constant_hessian_total::<SquaredError>(samples.len() as f64),
                0.0,
            )
        };
        assert!(score(&left) + score(&right) > score(&all));

        // A split that leaves both sides with the same mean gains nothing.
        let uniform = [1.0_f32, 1.0, 1.0, 1.0];
        let flat = |samples: &[usize]| {
            newton_split_score(
                negative_gradient_sum(samples, &uniform, None),
                constant_hessian_total::<SquaredError>(samples.len() as f64),
                0.0,
            )
        };
        assert_eq!(flat(&left) + flat(&right), flat(&all));
    }

    /// A weight of `k` contributes exactly what `k` copies of the row do, and
    /// unit weights leave every bit of the unweighted reduction alone.
    #[test]
    fn a_sample_weight_scales_a_gradient_like_repeating_its_row() {
        let negative_gradients = [2.0_f32, 4.0, -3.0];
        let samples = [0_usize, 1, 2];
        let unweighted = negative_gradient_sum(&samples, &negative_gradients, None);
        assert_eq!(
            negative_gradient_sum(&samples, &negative_gradients, Some(&[1.0, 1.0, 1.0])).to_bits(),
            unweighted.to_bits()
        );

        let repeated = [0_usize, 1, 1, 1, 2];
        assert_eq!(
            negative_gradient_sum(&samples, &negative_gradients, Some(&[1.0, 3.0, 1.0])),
            negative_gradient_sum(&repeated, &negative_gradients, None)
        );
        assert_eq!(
            negative_gradient_sum(&samples, &negative_gradients, Some(&[1.0, 0.0, 1.0])),
            negative_gradient_sum(&[0_usize, 2], &negative_gradients, None)
        );
    }

    /// Two objectives may never share an on-disk name, and squared error's is
    /// pinned to the value already written into every boosting artifact in the
    /// wild.
    #[test]
    fn every_objective_has_its_own_permanent_artifact_name() {
        assert_eq!(SquaredError::ARTIFACT_OBJECTIVE_TAG, 1);
        assert_ne!(
            SquaredError::ARTIFACT_OBJECTIVE_TAG,
            BinaryLogLoss::ARTIFACT_OBJECTIVE_TAG
        );
        // Zero stays unassigned so a zeroed metadata field never names a loss.
        assert_ne!(SquaredError::ARTIFACT_OBJECTIVE_TAG, 0);
        assert_ne!(BinaryLogLoss::ARTIFACT_OBJECTIVE_TAG, 0);
    }

    /// Squared error reads its denominator from the weight total and log loss
    /// from the hessian sum, and neither reads the other's argument.
    #[test]
    fn each_objective_forms_its_denominator_from_the_statistic_it_declares() {
        for weight in [0.0_f64, 1.0, 7.5, 4096.0] {
            for hessian in [0.0_f64, 0.25, 3.5] {
                assert_eq!(
                    SquaredError::node_hessian_total(weight, hessian).to_bits(),
                    weight.to_bits(),
                    "squared error must ignore a hessian sum"
                );
                assert_eq!(
                    BinaryLogLoss::node_hessian_total(weight, hessian).to_bits(),
                    hessian.to_bits(),
                    "log loss must ignore a weight total"
                );
            }
        }
    }

    /// The accumulated hessian sum is the quantity a Newton leaf divides by, so
    /// a leaf of log-loss residuals is the reference's `sum(y - p) / sum(p(1-p))`
    /// rather than a plain mean.
    #[test]
    fn a_log_loss_leaf_divides_by_the_accumulated_curvature() {
        let raws = [-2.0_f64, 0.0, 1.5, 3.0];
        let targets = [1.0_f64, 0.0, 1.0, 1.0];
        let negative_gradients = raws
            .iter()
            .zip(targets)
            .map(|(&raw, target)| BinaryLogLoss::negative_gradient(raw, target) as f32)
            .collect::<Vec<_>>();
        let hessians = raws
            .iter()
            .map(|&raw| BinaryLogLoss::hessian(raw, 0.0) as f32)
            .collect::<Vec<_>>();
        let samples = [0_usize, 1, 2, 3];
        let gradient = negative_gradient_sum(&samples, &negative_gradients, None);
        let curvature = hessian_sum(&samples, &hessians, None);
        assert_eq!(
            BinaryLogLoss::node_hessian_total(samples.len() as f64, curvature),
            curvature
        );

        let expected_gradient = raws
            .iter()
            .zip(targets)
            .map(|(&raw, target)| target - sigmoid_f64(raw))
            .fold(-0.0_f64, |sum, term| sum + f64::from(term as f32));
        let expected_curvature = raws
            .iter()
            .map(|&raw| sigmoid_f64(raw) * (1.0 - sigmoid_f64(raw)))
            .fold(-0.0_f64, |sum, term| sum + f64::from(term as f32));
        assert_eq!(gradient.to_bits(), expected_gradient.to_bits());
        assert_eq!(curvature.to_bits(), expected_curvature.to_bits());
        assert_eq!(
            newton_leaf_value(gradient, curvature, 0.0),
            (expected_gradient / expected_curvature) as f32
        );
    }

    /// A weight of `k` scales a hessian exactly as it scales a gradient, which
    /// is what keeps "an integer weight equals repeating the row" true for a
    /// varying-hessian objective too.
    #[test]
    fn a_sample_weight_scales_a_hessian_like_repeating_its_row() {
        let hessians = [0.25_f32, 0.1875, 0.09];
        let samples = [0_usize, 1, 2];
        assert_eq!(
            hessian_sum(&samples, &hessians, Some(&[1.0, 1.0, 1.0])).to_bits(),
            hessian_sum(&samples, &hessians, None).to_bits()
        );
        assert_eq!(
            hessian_sum(&samples, &hessians, Some(&[1.0, 3.0, 1.0])),
            hessian_sum(&[0_usize, 1, 1, 1, 2], &hessians, None)
        );
        assert_eq!(
            hessian_sum(&samples, &hessians, Some(&[1.0, 0.0, 1.0])),
            hessian_sum(&[0_usize, 2], &hessians, None)
        );
    }

    #[test]
    fn regularization_shrinks_a_leaf_toward_zero_without_changing_its_sign() {
        let negative_gradients = [-6.0_f32, -6.0];
        let samples = [0_usize, 1];
        let sum = negative_gradient_sum(&samples, &negative_gradients, None);
        let total = constant_hessian_total::<SquaredError>(samples.len() as f64);
        let plain = newton_leaf_value(sum, total, 0.0);
        let shrunk = newton_leaf_value(sum, total, 2.0);
        assert!(shrunk > plain && shrunk < 0.0);
    }
}
