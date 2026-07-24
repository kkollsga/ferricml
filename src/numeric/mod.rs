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
//!    results in that fixed index order.
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
//!    first real consumer rather than speculatively.
//! 5. **Saturation is explicit and happens at the boundary that produces the
//!    value.** Probabilities are clamped into `[0, 1]` by the routine that
//!    computes them, not by their consumer, so every downstream metric and
//!    artifact observes the same value.

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
}
