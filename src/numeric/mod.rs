//! Crate-private numeric kernels and FerricML's accumulation policy.
//!
//! This module owns the small scalar primitives that more than one estimator
//! family needs. It is deliberately private: keeping the kernels internal lets
//! them be specialized later without any public API churn, and stops estimator
//! families from re-deriving their own copy of a stability-sensitive routine.
//!
//! # Accumulation policy
//!
//! FerricML promises an identical fitted artifact for identical data,
//! parameters, seed, and thread count. Floating-point addition is neither
//! associative nor commutative, so that promise is a property of the fixed
//! *evaluation order* as much as of the arithmetic itself. The rules below are
//! binding on every module in the crate; a new estimator cites this section
//! instead of inventing its own convention.
//!
//! 1. **Fitting accumulates in `f64`.** Features, targets, and sample weights
//!    are stored as `f32`. Every sum, sum of squares, dot product, mean, or
//!    variance computed while *fitting* widens each term with `f64::from` and
//!    accumulates in `f64`, narrowing to `f32` exactly once when the fitted
//!    value is stored. Cancellation-prone quantities such as a population
//!    variance are clamped at their mathematical bound rather than allowed to
//!    go negative.
//! 2. **Evaluation order is fixed and sequential.** A reduction visits its
//!    terms in ascending row order, and ascending column order within a row.
//!    No path may reorder a reduction for speed, and no path may use an
//!    order that depends on how work happened to be scheduled. Parallel
//!    training partitions work so each partition's result depends only on its
//!    own index — the forest derives tree `i`'s seed from `i` alone and sorts
//!    the finished trees back into index order — and combines partition
//!    results in that fixed index order. [`sum_in_order`] is this rule written
//!    as code: a path that reduces a sequence of `f64` terms names it instead
//!    of reaching for whichever fold reads well locally, so the guarantee is
//!    visible at the call site rather than inferred from it.
//! 3. **Inference may accumulate in the storage width** when the number of
//!    terms is bounded by the fitted model (one term per tree, one per
//!    boosting iteration, one per feature) and the result is validated finite
//!    before it is returned. Such an accumulation is still strictly
//!    sequential in model order, because the fitted model defines that order.
//! 4. **Compensated summation is a documented exception, not a default.**
//!    Kahan/Neumaier compensation is required only where a reduction runs over
//!    an unbounded number of terms *and* widening to `f64` is unavailable —
//!    for instance a future `f64` streaming statistic. No path in the crate
//!    meets both conditions today, so the compensated helper lands with its
//!    first real consumer rather than speculatively. [`sum_in_order`] does not
//!    compensate, and deliberately so: every reduction it serves widens `f32`
//!    terms under rule 1, where compensation would buy accuracy the fitted
//!    `f32` result cannot represent while changing every frozen artifact.
//! 5. **Saturation is explicit and happens at the boundary that produces the
//!    value.** Probabilities are clamped into `[0, 1]` by the routine that
//!    computes them, not by their consumer, so every downstream metric and
//!    artifact observes the same value.
//! 6. **One seeded generator serves the crate.** Reproducible randomness comes
//!    from [`OwnedRng`]; a module must not define its own generator, because a
//!    seed has to mean the same thing in every estimator and in inspection.

// The quantile primitive lands one commit ahead of its first consumer, so the
// shared-surface change stays reviewable on its own. Both attributes are
// `expect` rather than `allow`, and both are scoped to the configuration that
// actually lacks a consumer: the tests below exercise the primitive, so only a
// non-test build sees it as dead. Once robust scaling arrives the expectations
// go unfulfilled and the compiler demands these lines back, which is what keeps
// a temporary suppression from becoming a permanent one.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "robust scaling is the first consumer and lands in the next commit"
    )
)]
mod quantile;
mod rng;

#[expect(
    unused_imports,
    reason = "robust scaling is the first consumer and lands in the next commit"
)]
pub(crate) use quantile::{QuantileRule, quantile_sorted, sort_for_quantiles};
pub(crate) use rng::{OwnedRng, derive_tree_seed};

/// Logistic sigmoid over `f64`.
///
/// The branch keeps the exponential argument non-positive, so a large
/// magnitude of either sign saturates to exactly `1.0` or `0.0` instead of
/// overflowing to infinity and producing a NaN quotient. Saturation is
/// asymmetric: `1.0` is reached once `exp(-value)` falls below half an
/// epsilon, well before `0.0` is reached at the underflow boundary of
/// `exp(value)`.
pub(crate) fn sigmoid_f64(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

/// Logistic sigmoid over `f32`, using the same branch as [`sigmoid_f64`].
pub(crate) fn sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

/// Sum of a sequence of `f64` terms in the order the sequence produces them.
///
/// This is the crate's one reduction primitive over an unbounded number of
/// terms, and it exists to make rule 2 of the accumulation policy above
/// checkable rather than conventional: each term is added exactly once, in
/// sequence order. Nothing here reassociates, vectorizes, or compensates, so
/// two runs over the same terms in the same order produce the same bits on the
/// same target.
///
/// The accumulator starts at `-0.0`, which is IEEE addition's true identity:
/// `x + -0.0` is `x` for every `x` including `-0.0`, whereas seeding with
/// `+0.0` would turn a sum of negative zeros into a positive zero and change
/// the bits of any fitted value derived from it. `f64`'s own [`Sum`] uses the
/// same seed, so this helper is a bit-for-bit substitute for it.
///
/// A caller that needs a different order sorts or indexes its terms before
/// calling; the ordering decision belongs to the caller, because it is part of
/// what that caller's fitted artifact is frozen against.
///
/// [`Sum`]: std::iter::Sum
#[inline]
pub(crate) fn sum_in_order(terms: impl IntoIterator<Item = f64>) -> f64 {
    terms
        .into_iter()
        .fold(-0.0, |total, term: f64| total + term)
}

/// Natural logarithm of a sum of exponentials, without forming the sum.
///
/// The naive `values.iter().map(f64::exp).sum().ln()` overflows to `inf` for
/// arguments above roughly `710` and underflows to `-inf` below roughly
/// `-746`, in both cases destroying a result that is perfectly representable.
/// Shifting by the maximum keeps every exponential in `(0, 1]` and makes the
/// largest term exactly `1`, so the sum is at least `1` and its logarithm is
/// always defined.
///
/// The reduction visits its terms in ascending index order, per rule 2 of the
/// accumulation policy above; the caller's ordering of `values` is therefore
/// part of the result.
///
/// Boundary cases are exact rather than approximate: an empty slice and an
/// all-`-inf` slice both return `-inf` (the identity of the underlying sum), a
/// slice containing `+inf` returns `+inf`, and a `NaN` anywhere propagates.
pub(crate) fn log_sum_exp(values: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        if value.is_nan() {
            return f64::NAN;
        }
        if value > max {
            max = value;
        }
    }
    if !max.is_finite() {
        return max;
    }
    let mut total = 0.0_f64;
    for &value in values {
        total += (value - max).exp();
    }
    max + total.ln()
}

/// Softmax of one row of raw scores, computed in place.
///
/// This is the multiclass counterpart of [`sigmoid_f64`], and it is stabilized
/// the same way [`log_sum_exp`] is: shifting by the row maximum keeps every
/// exponential in `(0, 1]` and makes the largest term exactly `1`, so the sum
/// is at least `1`. The naive `exp(v) / Σ exp(v)` instead overflows to `inf`
/// above roughly `710` and underflows the whole sum to `0` below roughly
/// `-746`, producing `NaN` for a result that is perfectly representable — and
/// a raw score of either magnitude is reachable from finite data.
///
/// The reduction visits its terms in ascending index order, per rule 2 of the
/// accumulation policy above, so the caller's column order is part of the
/// result.
///
/// # Row sums
///
/// Each value is a quotient by the exact sum of the same values, so every
/// result lies in `[0, 1]` by construction and rule 5's clamp would be a no-op.
/// The *sum* of the results is `1` only to rounding. FerricML deliberately does
/// not renormalize: a second pass would move mass without measuring anything,
/// and the residual is a frozen part of the probability contract rather than a
/// defect to correct.
///
/// # Boundary cases
///
/// A `NaN` anywhere makes the whole row `NaN`. An infinite maximum has no
/// finite shift, so the limit is taken instead: `+inf` entries share the mass
/// equally and every other entry is exactly `0`, and an all-`-inf` row is
/// uniform. An empty row writes nothing.
pub(crate) fn softmax_in_place(row: &mut [f64]) {
    if row.is_empty() {
        return;
    }
    let mut max = f64::NEG_INFINITY;
    for &value in row.iter() {
        if value.is_nan() {
            row.fill(f64::NAN);
            return;
        }
        if value > max {
            max = value;
        }
    }
    if !max.is_finite() {
        let winners = if max == f64::INFINITY {
            row.iter().filter(|value| **value == f64::INFINITY).count()
        } else {
            row.len()
        };
        let share = 1.0 / winners as f64;
        for value in row.iter_mut() {
            *value = if max == f64::INFINITY && *value != f64::INFINITY {
                0.0
            } else {
                share
            };
        }
        return;
    }
    let mut total = 0.0_f64;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    for value in row {
        *value /= total;
    }
}

/// Softmax over `f32`, using the same shift and the same order as
/// [`softmax_in_place`].
///
/// Inference accumulates in the storage width under rule 3 of the accumulation
/// policy: the term count is bounded by the fitted class count and the caller
/// validates the result. Fitting uses the `f64` form.
pub(crate) fn softmax_in_place_f32(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &value in row.iter() {
        if value.is_nan() {
            row.fill(f32::NAN);
            return;
        }
        if value > max {
            max = value;
        }
    }
    if !max.is_finite() {
        let winners = if max == f32::INFINITY {
            row.iter().filter(|value| **value == f32::INFINITY).count()
        } else {
            row.len()
        };
        let share = 1.0 / winners as f32;
        for value in row.iter_mut() {
            *value = if max == f32::INFINITY && *value != f32::INFINITY {
                0.0
            } else {
                share
            };
        }
        return;
    }
    let mut total = 0.0_f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    for value in row {
        *value /= total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_saturates_exactly_at_extreme_magnitudes_in_both_signs() {
        for &value in &[1.0e3_f64, 1.0e30, 1.0e300, f64::MAX] {
            assert_eq!(sigmoid_f64(value), 1.0);
            assert_eq!(sigmoid_f64(-value), 0.0);
        }
        for &value in &[1.0e3_f32, 1.0e30, f32::MAX] {
            assert_eq!(sigmoid_f32(value), 1.0);
            assert_eq!(sigmoid_f32(-value), 0.0);
        }
        assert_eq!(sigmoid_f64(0.0), 0.5);
        assert_eq!(sigmoid_f32(0.0), 0.5);
    }

    #[test]
    fn sigmoid_is_finite_in_range_and_monotone_non_decreasing() {
        let mut previous_f64 = 0.0_f64;
        let mut previous_f32 = 0.0_f32;
        for step in -2_000..=2_000 {
            let value = f64::from(step) / 20.0;
            let probability = sigmoid_f64(value);
            assert!((0.0..=1.0).contains(&probability), "f64 range at {value}");
            assert!(probability >= previous_f64, "f64 monotonicity at {value}");
            previous_f64 = probability;

            let value = value as f32;
            let probability = sigmoid_f32(value);
            assert!((0.0..=1.0).contains(&probability), "f32 range at {value}");
            assert!(probability >= previous_f32, "f32 monotonicity at {value}");
            previous_f32 = probability;
        }
        assert_eq!(previous_f64, 1.0);
        assert_eq!(previous_f32, 1.0);
    }

    #[test]
    fn sigmoid_reflects_around_one_half_and_agrees_across_widths() {
        // Single-precision evaluation of a double-precision-representable
        // argument stays within one f32 ulp of the f64 result.
        for step in -400..=400 {
            let value = f64::from(step) / 8.0;
            let wide = sigmoid_f64(value);
            assert!(
                (wide + sigmoid_f64(-value) - 1.0).abs() <= 1.0e-15,
                "f64 reflection at {value}"
            );
            let narrow = sigmoid_f32(value as f32);
            assert!(
                (f64::from(narrow) - wide).abs() <= 1.0e-6,
                "width agreement at {value}: {narrow} vs {wide}"
            );
        }
    }

    #[test]
    fn sigmoid_reaches_exact_zero_and_one_only_past_the_representable_boundary() {
        // Saturation is asymmetric, and both boundaries are contractual.
        // `1.0` appears as soon as `exp(-value)` falls below half an epsilon;
        // `0.0` only once `exp(value)` underflows, which is far further out.
        assert!(sigmoid_f64(36.0) < 1.0);
        assert_eq!(sigmoid_f64(37.0), 1.0);
        assert!(sigmoid_f64(-700.0) > 0.0);
        assert_eq!(sigmoid_f64(-800.0), 0.0);

        assert!(sigmoid_f32(16.0) < 1.0);
        assert_eq!(sigmoid_f32(17.0), 1.0);
        assert!(sigmoid_f32(-80.0) > 0.0);
        assert_eq!(sigmoid_f32(-200.0), 0.0);

        // The complement of a saturated value is exact rather than negative.
        assert_eq!(1.0 - sigmoid_f64(37.0), 0.0);
        assert_eq!(1.0 - sigmoid_f32(17.0), 0.0);
    }

    #[test]
    fn sum_in_order_matches_a_sequential_fold_and_is_repeatable() {
        let terms = (0..1_000)
            .map(|step| f64::from(step) * 0.1 - 37.5)
            .collect::<Vec<_>>();
        let expected = terms
            .iter()
            .copied()
            .fold(-0.0_f64, |total, term| total + term);
        assert_eq!(
            sum_in_order(terms.iter().copied()).to_bits(),
            expected.to_bits()
        );
        assert_eq!(
            sum_in_order(terms.iter().copied()).to_bits(),
            sum_in_order(terms.iter().copied()).to_bits()
        );
    }

    #[test]
    fn sum_in_order_is_a_bit_for_bit_substitute_for_the_standard_sum() {
        // Seeding the accumulator with `+0.0` instead of IEEE addition's true
        // identity would silently flip the sign of a zero result, and a fitted
        // `-0.0` is a different artifact byte pattern from `0.0`.
        let cases: [Vec<f64>; 7] = [
            vec![],
            vec![-0.0],
            vec![-0.0; 8],
            vec![0.0; 8],
            vec![0.0, -0.0],
            vec![-0.0, 0.0],
            vec![1.0, -1.0, 2.5, -2.5],
        ];
        for terms in &cases {
            assert_eq!(
                sum_in_order(terms.iter().copied()).to_bits(),
                terms.iter().copied().sum::<f64>().to_bits(),
                "sum of {terms:?}"
            );
        }
        assert!(sum_in_order(std::iter::empty()).is_sign_negative());
        assert_eq!(sum_in_order(std::iter::empty()), 0.0);
    }

    #[test]
    fn sum_in_order_is_order_sensitive_rather_than_reassociating() {
        // Cancellation makes the order observable, which is exactly why the
        // policy fixes it. A helper that reassociated would hide this.
        let ascending = [1.0_f64, 1.0e16, -1.0e16];
        let descending = [1.0e16_f64, -1.0e16, 1.0];
        assert_eq!(sum_in_order(ascending), 0.0);
        assert_eq!(sum_in_order(descending), 1.0);
    }

    #[test]
    fn sum_in_order_widens_narrow_terms_without_an_intermediate_rounding() {
        let terms = [1.0_f32, f32::EPSILON / 4.0, f32::EPSILON / 4.0];
        let widened = sum_in_order(terms.iter().map(|&term| f64::from(term)));
        assert!(widened > 1.0, "f64 accumulation keeps the small terms");
        let narrow = terms.iter().fold(0.0_f32, |total, &term| total + term);
        assert_eq!(narrow, 1.0, "f32 accumulation would have dropped them");
    }

    fn naive_log_sum_exp(values: &[f64]) -> f64 {
        values.iter().map(|value| value.exp()).sum::<f64>().ln()
    }

    #[test]
    fn log_sum_exp_agrees_with_the_naive_formulation_in_its_safe_range() {
        for left in -30..=30 {
            for right in -30..=30 {
                let values = [f64::from(left) / 2.0, f64::from(right) / 2.0];
                let stable = log_sum_exp(&values);
                let naive = naive_log_sum_exp(&values);
                assert!(
                    (stable - naive).abs() <= 1.0e-12 * naive.abs().max(1.0),
                    "log_sum_exp{values:?}: {stable} vs {naive}"
                );
            }
        }
    }

    #[test]
    fn log_sum_exp_survives_extreme_magnitudes_in_both_signs() {
        // `exp` overflows above roughly 709.8 and underflows below roughly
        // -745.2, so past those bounds the naive formulation returns an
        // infinity for a result that is perfectly representable. The shifted
        // reduction stays within an ulp of the exact answer in both signs.
        for &magnitude in &[300.0_f64, 750.0, 1.0e5, 1.0e300] {
            for &value in &[magnitude, -magnitude] {
                let doubled = log_sum_exp(&[value, value]);
                assert!(
                    (doubled - (value + std::f64::consts::LN_2)).abs()
                        <= 1.0e-12 * value.abs().max(1.0),
                    "log_sum_exp of a repeated {value}: {doubled}"
                );
            }
            if magnitude > 750.0 {
                assert!(
                    !naive_log_sum_exp(&[magnitude, magnitude]).is_finite(),
                    "naive formulation unexpectedly survived +{magnitude}"
                );
                assert!(
                    !naive_log_sum_exp(&[-magnitude, -magnitude]).is_finite(),
                    "naive formulation unexpectedly survived -{magnitude}"
                );
            }
        }
        // The first magnitude past each boundary, stated exactly.
        assert!(!naive_log_sum_exp(&[710.0, 710.0]).is_finite());
        assert!(!naive_log_sum_exp(&[-746.0, -746.0]).is_finite());
        assert!((log_sum_exp(&[710.0, 710.0]) - (710.0 + std::f64::consts::LN_2)).abs() <= 1.0e-9);
        assert!(
            (log_sum_exp(&[-746.0, -746.0]) - (-746.0 + std::f64::consts::LN_2)).abs() <= 1.0e-9
        );
    }

    #[test]
    fn log_sum_exp_dominated_by_one_term_returns_that_term() {
        for &value in &[0.0_f64, 50.0, -50.0, 1.0e6, -1.0e6] {
            assert_eq!(log_sum_exp(&[value]), value);
            assert_eq!(log_sum_exp(&[value, value - 1.0e6]), value);
        }
        assert_eq!(log_sum_exp(&[0.0, 0.0]), std::f64::consts::LN_2);
    }

    #[test]
    fn log_sum_exp_boundary_inputs_are_exact() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY, 2.0]), 2.0);
        assert_eq!(log_sum_exp(&[f64::INFINITY, 0.0]), f64::INFINITY);
        assert!(log_sum_exp(&[f64::NAN, 0.0]).is_nan());
        assert!(log_sum_exp(&[0.0, f64::NAN]).is_nan());
    }

    fn naive_softmax(values: &[f64]) -> Vec<f64> {
        let total = values.iter().map(|value| value.exp()).sum::<f64>();
        values.iter().map(|value| value.exp() / total).collect()
    }

    fn softmax(values: &[f64]) -> Vec<f64> {
        let mut row = values.to_vec();
        softmax_in_place(&mut row);
        row
    }

    #[test]
    fn softmax_agrees_with_the_naive_formulation_where_the_naive_one_is_safe() {
        // The naive quotient is trustworthy only while every exponential is
        // representable, which is exactly the range swept here.
        for left in -30..=30 {
            for right in -30..=30 {
                for third in [-4, 0, 7] {
                    let values = [
                        f64::from(left) / 2.0,
                        f64::from(right) / 2.0,
                        f64::from(third),
                    ];
                    let stable = softmax(&values);
                    let naive = naive_softmax(&values);
                    for (index, (stable, naive)) in stable.iter().zip(&naive).enumerate() {
                        assert!(
                            (stable - naive).abs() <= 1.0e-14,
                            "softmax{values:?}[{index}]: {stable} vs {naive}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn softmax_survives_extreme_magnitudes_in_both_signs() {
        // Past `exp`'s overflow and underflow boundaries the naive formulation
        // returns NaN for a result that is perfectly ordinary. The shifted row
        // stays exact, in both directions.
        for &magnitude in &[300.0_f64, 750.0, 1.0e5, 1.0e300] {
            for &sign in &[1.0_f64, -1.0] {
                let base = sign * magnitude;
                let uniform = softmax(&[base, base, base]);
                for value in &uniform {
                    assert!(
                        (value - 1.0 / 3.0).abs() <= 1.0e-15,
                        "equal scores at {base} must be uniform, got {uniform:?}"
                    );
                }
                // A dominant entry saturates rather than producing NaN. The
                // gap is scaled so it stays representable at every magnitude.
                let gap = 1.0e4_f64.max(base.abs() * 1.0e-6);
                let dominated = softmax(&[base, base - gap]);
                assert_eq!(dominated, vec![1.0, 0.0], "dominant entry at {base}");
            }
            if magnitude > 750.0 {
                let naive = naive_softmax(&[magnitude, magnitude, magnitude]);
                assert!(
                    naive.iter().any(|value| !value.is_finite()),
                    "naive formulation unexpectedly survived +{magnitude}"
                );
                let naive = naive_softmax(&[-magnitude, -magnitude, -magnitude]);
                assert!(
                    naive.iter().any(|value| !value.is_finite()),
                    "naive formulation unexpectedly survived -{magnitude}"
                );
            }
        }
        // The first magnitude past each boundary, stated exactly.
        assert!(naive_softmax(&[710.0, 710.0])[0].is_nan());
        assert!(naive_softmax(&[-746.0, -746.0])[0].is_nan());
        assert_eq!(softmax(&[710.0, 710.0]), vec![0.5, 0.5]);
        assert_eq!(softmax(&[-746.0, -746.0]), vec![0.5, 0.5]);
    }

    #[test]
    fn softmax_stays_in_range_and_is_shift_invariant() {
        let base = [-2.5_f64, 0.0, 1.25, 7.0, -13.0];
        let reference = softmax(&base);
        assert!(reference.iter().all(|value| (0.0..=1.0).contains(value)));
        for shift in [-500.0_f64, -1.0, 0.0, 1.0, 500.0] {
            let shifted = base.iter().map(|value| value + shift).collect::<Vec<_>>();
            for (index, (shifted, reference)) in
                softmax(&shifted).iter().zip(&reference).enumerate()
            {
                assert!(
                    (shifted - reference).abs() <= 1.0e-15,
                    "shift {shift} changed column {index}: {shifted} vs {reference}"
                );
            }
        }
        // Equal scores are exactly uniform, which is what makes an exact tie
        // resolve on the class order rather than on rounding.
        for width in 1..=8 {
            let uniform = softmax(&vec![3.5_f64; width]);
            assert!(uniform.iter().all(|&value| value == 1.0 / width as f64));
        }
    }

    #[test]
    fn softmax_rows_are_not_renormalized_and_may_miss_one_by_rounding() {
        // The documented contract: a row sums to 1 only to floating-point
        // rounding. This asserts the residual is bounded, not that it is zero.
        let mut worst = 0.0_f64;
        let mut inexact = 0_usize;
        for step in -400..=400 {
            let value = f64::from(step) / 7.0;
            let row = softmax(&[value, -value, value / 3.0, 0.0, 1.0 - value]);
            let deviation = (row.iter().sum::<f64>() - 1.0).abs();
            worst = worst.max(deviation);
            inexact += usize::from(row.iter().sum::<f64>() != 1.0);
        }
        assert!(worst <= 8.0 * f64::EPSILON, "row-sum deviation {worst}");
        assert!(inexact > 0, "the residual is real, not hypothetical");
    }

    #[test]
    fn softmax_boundary_inputs_are_exact() {
        let mut empty: [f64; 0] = [];
        softmax_in_place(&mut empty);

        assert_eq!(softmax(&[7.5]), vec![1.0]);
        assert!(softmax(&[f64::NAN, 0.0]).iter().all(|value| value.is_nan()));
        assert!(softmax(&[0.0, f64::NAN]).iter().all(|value| value.is_nan()));
        assert_eq!(
            softmax(&[f64::INFINITY, 0.0, f64::INFINITY]),
            vec![0.5, 0.0, 0.5]
        );
        assert_eq!(softmax(&[f64::INFINITY, 3.0]), vec![1.0, 0.0]);
        assert_eq!(
            softmax(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            vec![0.5, 0.5]
        );
        assert_eq!(softmax(&[f64::NEG_INFINITY, 2.0]), vec![0.0, 1.0]);
    }

    #[test]
    fn softmax_agrees_with_the_sigmoid_on_two_centred_scores() {
        // The binary path stays asymmetric on purpose, so this is the identity
        // that keeps the two definitions from drifting apart.
        for step in -600..=600 {
            let raw = f64::from(step) / 20.0;
            let row = softmax(&[-raw / 2.0, raw / 2.0]);
            assert!(
                (row[1] - sigmoid_f64(raw)).abs() <= 1.0e-15,
                "centred pair at {raw}: {} vs {}",
                row[1],
                sigmoid_f64(raw)
            );
        }
    }

    fn softmax_f32(values: &[f32]) -> Vec<f32> {
        let mut row = values.to_vec();
        softmax_in_place_f32(&mut row);
        row
    }

    #[test]
    fn softmax_agrees_across_widths_and_keeps_its_boundaries_at_f32() {
        for step in -600..=600 {
            let raw = f64::from(step) / 20.0;
            let wide = softmax(&[raw, -raw / 3.0, 0.5]);
            let narrow = softmax_f32(&[raw as f32, (-raw / 3.0) as f32, 0.5]);
            for (index, (narrow, wide)) in narrow.iter().zip(&wide).enumerate() {
                assert!(
                    (f64::from(*narrow) - wide).abs() <= 1.0e-6,
                    "width agreement at {raw} column {index}: {narrow} vs {wide}"
                );
            }
        }
        // `f32::exp` overflows above roughly 88.7 and underflows below roughly
        // -103.9, both far earlier than at f64, so the shift matters more here.
        for &magnitude in &[50.0_f32, 100.0, 1.0e10, f32::MAX] {
            for &sign in &[1.0_f32, -1.0] {
                let base = sign * magnitude;
                let uniform = softmax_f32(&[base, base, base]);
                assert!(
                    uniform
                        .iter()
                        .all(|value| (value - 1.0 / 3.0).abs() <= 1.0e-7),
                    "equal scores at {base} must be uniform, got {uniform:?}"
                );
            }
        }
        assert!(
            (f32::MAX.exp() / (f32::MAX.exp() + f32::MAX.exp())).is_nan(),
            "the naive f32 quotient really does fail at this magnitude"
        );
        assert_eq!(softmax_f32(&[90.0, 90.0]), vec![0.5, 0.5]);
        assert_eq!(softmax_f32(&[-110.0, -110.0]), vec![0.5, 0.5]);
        assert!(softmax_f32(&[f32::NAN, 0.0]).iter().all(|v| v.is_nan()));
        assert_eq!(softmax_f32(&[f32::INFINITY, 0.0]), vec![1.0, 0.0]);
        assert!(
            softmax_f32(&[1.0, 2.0, 3.0])
                .iter()
                .all(|&value| (0.0..=1.0).contains(&value))
        );
    }

    #[test]
    fn log_sum_exp_is_monotone_in_each_argument() {
        let mut previous = f64::NEG_INFINITY;
        for step in -1_000..=1_000 {
            let value = log_sum_exp(&[0.0, f64::from(step) / 10.0]);
            assert!(value >= previous, "monotonicity at step {step}");
            assert!(value >= 0.0, "softplus stays non-negative at step {step}");
            previous = value;
        }
    }
}
