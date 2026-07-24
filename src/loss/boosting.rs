//! Mapping an objective into a boosted tree's leaf statistics.
//!
//! A gradient-boosted tree never sees a loss directly. It sees, per node, a sum
//! of negative gradients and a total hessian, and turns that pair into a leaf
//! value and a split score. This module owns that translation so the grower
//! stays a search over histograms and the objective stays the only place a
//! derivative is written down.

use super::objective::Objective;
use crate::numeric::sum_in_order;

/// Sum of the negative gradients carried by one node's samples.
///
/// The reduction visits `samples` in the order the caller holds them, which is
/// ascending row order for a freshly split node, and widens each `f32` term
/// under rule 1 of the accumulation policy in [`crate::numeric`].
#[inline]
pub(crate) fn negative_gradient_sum(samples: &[usize], negative_gradients: &[f32]) -> f64 {
    sum_in_order(
        samples
            .iter()
            .map(|&sample| f64::from(negative_gradients[sample])),
    )
}

/// Total hessian of `rows` samples of a constant-hessian objective.
///
/// A constant hessian is exactly the property that lets a grower carry a row
/// count where a general objective would have to carry a second per-sample
/// histogram. The constant is read from the objective rather than assumed, so
/// an objective that scales its loss differently scales the leaf denominator
/// with it.
pub(crate) fn constant_hessian_total<O: Objective>(rows: usize) -> f64 {
    const {
        assert!(
            O::CONSTANT_HESSIAN,
            "a per-sample hessian cannot be recovered from a row count"
        );
    }
    // Any point evaluates the same, the hessian being constant by declaration.
    rows as f64 * O::hessian(0.0, 0.0)
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
    use crate::loss::SquaredError;

    #[test]
    fn a_constant_hessian_total_is_the_row_count_for_squared_error() {
        for rows in [0_usize, 1, 7, 1_000_000] {
            assert_eq!(
                constant_hessian_total::<SquaredError>(rows).to_bits(),
                (rows as f64).to_bits()
            );
        }
    }

    #[test]
    fn the_leaf_value_is_the_regularized_mean_of_the_negative_gradients() {
        let negative_gradients = [2.0_f32, 4.0, -3.0, 9.0];
        let samples = [0_usize, 1, 2];
        let sum = negative_gradient_sum(&samples, &negative_gradients);
        assert_eq!(sum, 3.0);
        let total = constant_hessian_total::<SquaredError>(samples.len());
        assert_eq!(newton_leaf_value(sum, total, 0.0), 1.0);
        assert_eq!(newton_leaf_value(sum, total, 3.0), 0.5);
    }

    #[test]
    fn an_exactly_balanced_leaf_keeps_a_positively_signed_zero() {
        let negative_gradients = [1.5_f32, -1.5];
        let samples = [0_usize, 1];
        let sum = negative_gradient_sum(&samples, &negative_gradients);
        let value = newton_leaf_value(sum, constant_hessian_total::<SquaredError>(2), 0.0);
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
                negative_gradient_sum(samples, &negative_gradients),
                constant_hessian_total::<SquaredError>(samples.len()),
                0.0,
            )
        };
        assert!(score(&left) + score(&right) > score(&all));

        // A split that leaves both sides with the same mean gains nothing.
        let uniform = [1.0_f32, 1.0, 1.0, 1.0];
        let flat = |samples: &[usize]| {
            newton_split_score(
                negative_gradient_sum(samples, &uniform),
                constant_hessian_total::<SquaredError>(samples.len()),
                0.0,
            )
        };
        assert_eq!(flat(&left) + flat(&right), flat(&all));
    }

    #[test]
    fn regularization_shrinks_a_leaf_toward_zero_without_changing_its_sign() {
        let negative_gradients = [-6.0_f32, -6.0];
        let samples = [0_usize, 1];
        let sum = negative_gradient_sum(&samples, &negative_gradients);
        let total = constant_hessian_total::<SquaredError>(samples.len());
        let plain = newton_leaf_value(sum, total, 0.0);
        let shrunk = newton_leaf_value(sum, total, 2.0);
        assert!(shrunk > plain && shrunk < 0.0);
    }
}
