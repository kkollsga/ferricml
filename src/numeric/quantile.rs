//! Sample quantiles, and the rule each evaluation is taken under.
//!
//! A quantile is not one function. Small samples do not contain the value a
//! percentile asks for, so every library interpolates, and the defensible
//! interpolations disagree — at nine values out of ten on a four-element
//! sample. The rule is therefore a *documented semantic choice* rather than an
//! implementation detail, and it is carried as a typed parameter at every call
//! site instead of being a default some caller has to know about.
//!
//! That the rule is explicit is what makes a second rule addable without
//! silently repointing the first consumer at it: FerricML's transformers do not
//! all want the same definition, and a hidden default would have to be wrong
//! for one of them.

/// Which quantile definition an evaluation is taken under.
///
/// One variant today. It is still an enum rather than an implied default
/// because the choice belongs at the call site: a reader of
/// `quantile_sorted(column, 50.0, QuantileRule::Linear)` can see which
/// definition a fitted value was frozen against, and a second rule becomes an
/// added variant rather than a changed meaning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QuantileRule {
    /// Linear interpolation between the two bracketing order statistics.
    ///
    /// Sort the sample ascending as `x[0] <= ... <= x[n-1]`. For a percentile
    /// `p` in `[0, 100]`, with `h = (p / 100) * (n - 1)`, `j = floor(h)` and
    /// `g = h - j`:
    ///
    /// ```text
    /// Q(p) = x[j]                          if j == n - 1
    /// Q(p) = x[j] + g * (x[j+1] - x[j])    otherwise
    /// ```
    ///
    /// For `n == 1`, `Q(p) = x[0]` at every `p`. This is the continuous rule
    /// classified as Hyndman–Fan type 7, with plotting position `(k-1)/(n-1)`.
    /// It is the definition FerricML's robust scaling is frozen against.
    Linear,
}

/// Orders a scratch buffer so [`quantile_sorted`] can read it.
///
/// Sorting is unstable on purpose. Two `f64` values that compare equal under
/// [`f64::total_cmp`] have identical bit patterns, so no permutation of them is
/// observable in a value read back out by index — which makes the cheaper sort
/// exactly as deterministic as a stable one here, and determinism is the
/// property that matters. `total_cmp` rather than a partial comparison because
/// the sort must be a total order to be well defined at all.
pub(crate) fn sort_for_quantiles(scratch: &mut [f64]) {
    scratch.sort_unstable_by(f64::total_cmp);
}

/// Evaluates one percentile of an ascending-sorted sample.
///
/// `sorted` must be non-empty and ascending — [`sort_for_quantiles`] is how a
/// caller gets there — and `percentile` must lie in `[0, 100]`. Both are
/// caller obligations rather than checks, because every caller in the crate
/// validates its percentile range once at its own public boundary and then
/// evaluates per column.
///
/// The general expression is applied uniformly, including at `p = 50`. Some
/// implementations special-case the median at even `n` to the average of the
/// two middle order statistics, which differs from the general evaluation by
/// one ulp; FerricML does not, so one expression describes every quantile it
/// reports and the difference is carried by comparison tolerances rather than
/// by a branch.
pub(crate) fn quantile_sorted(sorted: &[f64], percentile: f64, rule: QuantileRule) -> f64 {
    debug_assert!(!sorted.is_empty(), "a quantile needs at least one value");
    debug_assert!(
        (0.0..=100.0).contains(&percentile),
        "percentile {percentile} is outside 0..=100"
    );
    match rule {
        QuantileRule::Linear => linear(sorted, percentile),
    }
}

/// Hyndman–Fan type 7, evaluated exactly as [`QuantileRule::Linear`] states it.
fn linear(sorted: &[f64], percentile: f64) -> f64 {
    let last = sorted.len() - 1;
    if last == 0 {
        return sorted[0];
    }
    let position = (percentile / 100.0) * last as f64;
    let lower = position.floor();
    let fraction = position - lower;
    // `position` is in `0..=last` for a percentile in `0..=100`, so the cast is
    // exact; the bound is re-checked rather than assumed, because a rounded
    // `position` of exactly `last` must read the final order statistic instead
    // of indexing past it.
    let index = lower as usize;
    if index >= last {
        return sorted[last];
    }
    sorted[index] + fraction * (sorted[index + 1] - sorted[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The percentiles the specification room found the conventions disagree on.
    const DIVERGENT: [f64; 7] = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0];

    fn linear_at(sample: &[f64], percentile: f64) -> f64 {
        quantile_sorted(sample, percentile, QuantileRule::Linear)
    }

    #[test]
    fn a_two_element_sample_interpolates_rather_than_picking_an_order_statistic() {
        // The single sharpest row: every discontinuous rule returns 0.0, 1.0,
        // or 0.5 here, and only linear interpolation returns 0.25.
        assert_eq!(linear_at(&[0.0, 1.0], 25.0), 0.25);
        assert_eq!(linear_at(&[0.0, 1.0], 50.0), 0.5);
        assert_eq!(linear_at(&[0.0, 1.0], 75.0), 0.75);
    }

    #[test]
    fn the_four_element_reference_row_is_reproduced_bit_for_bit() {
        let sample = [0.0, 1.0, 2.0, 10.0];
        let expected: [f64; 7] = [
            0.0,
            0.300_000_000_000_000_04,
            0.75,
            1.5,
            4.0,
            7.600_000_000_000_001,
            10.0,
        ];
        for (percentile, expected) in DIVERGENT.iter().zip(expected) {
            let actual = linear_at(&sample, *percentile);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "Q({percentile}) was {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn the_robust_scaling_statistics_match_the_worked_examples() {
        // Each row is (sample, Q25, Q50, Q75). The centre and the spread a
        // robust scaler removes are read straight off these three values — the
        // spread is `Q75 - Q25` and carries no assertion of its own here,
        // because with both quartiles already pinned by exact equality their
        // difference cannot come out any other way. Asserting it as well would
        // read like a fourth check and be arithmetic restating the two above.
        let rows: [(&[f64], f64, f64, f64); 3] = [
            (&[0.0, 1.0], 0.25, 0.5, 0.75),
            (&[0.0, 1.0, 10.0], 0.5, 1.0, 5.5),
            (&[-3.0, 0.5, 2.0, 11.0], -0.375, 1.25, 4.25),
        ];
        for (sample, lower, median, upper) in rows {
            assert_eq!(linear_at(sample, 25.0), lower, "Q25 of {sample:?}");
            assert_eq!(linear_at(sample, 50.0), median, "Q50 of {sample:?}");
            assert_eq!(linear_at(sample, 75.0), upper, "Q75 of {sample:?}");
        }
    }

    #[test]
    fn a_single_value_is_every_percentile_of_itself() {
        for percentile in DIVERGENT {
            assert_eq!(linear_at(&[5.0], percentile), 5.0);
            assert_eq!(
                linear_at(&[-0.0], percentile).to_bits(),
                (-0.0_f64).to_bits()
            );
        }
    }

    #[test]
    fn the_endpoints_are_the_extrema_exactly() {
        let samples: [&[f64]; 4] = [
            &[5.0],
            &[0.0, 1.0],
            &[-3.0, 0.5, 2.0, 11.0],
            &[1.0, 1.0, 1.0, 2.0, 900.0],
        ];
        for sample in samples {
            assert_eq!(linear_at(sample, 0.0), sample[0], "Q0 of {sample:?}");
            assert_eq!(
                linear_at(sample, 100.0),
                sample[sample.len() - 1],
                "Q100 of {sample:?}"
            );
        }
    }

    #[test]
    fn repeated_values_collapse_the_interpolation_onto_themselves() {
        // A tied bracket has nothing to interpolate between, so the fraction
        // cannot move the result off the tie.
        // All three quartiles land inside the tie, so the interquartile range a
        // robust scaler would divide by is zero while the column is anything
        // but constant. That consequence is the point of the row; it is left as
        // prose because `Q75 - Q25` over two values already pinned to `5.0` is
        // their difference and cannot report anything they did not.
        let sample = [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0];
        assert_eq!(linear_at(&sample, 25.0), 5.0);
        assert_eq!(linear_at(&sample, 50.0), 5.0);
        assert_eq!(linear_at(&sample, 75.0), 5.0);
    }

    /// The sweep rises, stalls exactly where the sample is tied, and lands on
    /// the extrema.
    ///
    /// A non-decreasing sweep on its own is a weak thing to assert: a `linear`
    /// that returned `sorted[0]` at every percentile satisfies it, and so does
    /// one that returned any constant. So the sweep also counts its ascents,
    /// which such an implementation cannot produce, and bounds them from above,
    /// which an implementation that interpolated across the tied bracket rather
    /// than collapsing onto it cannot satisfy. The two anchors then pin the
    /// sweep to *this* sample instead of to any rising sequence.
    #[test]
    fn quantiles_are_monotone_in_the_percentile() {
        // Nine values with one tied pair, so 1/8 of the percentile range —
        // 125 of the 1000 steps — is a bracket with nothing to interpolate.
        let sample = [-7.5, -0.25, 0.0, 0.0, 1.0, 3.5, 12.0, 900.0, 1e6];
        let mut previous = f64::NEG_INFINITY;
        let mut ascents = 0_usize;
        for step in 0..=1_000 {
            let percentile = f64::from(step) / 10.0;
            let value = linear_at(&sample, percentile);
            assert!(value >= previous, "Q({percentile}) = {value} fell back");
            ascents += usize::from(previous.is_finite() && value > previous);
            previous = value;
        }
        assert_eq!(
            ascents, 875,
            "the sweep rose at {ascents} of its 1000 steps; every step but the \
             125 inside the tied bracket has to rise, so a constant or \
             order-statistic result is short and an interpolation across the \
             tie is long"
        );
        assert_eq!(linear_at(&sample, 0.0), sample[0], "the sweep starts low");
        assert_eq!(previous, sample[sample.len() - 1], "and ends at the top");
    }

    #[test]
    fn sorting_is_what_makes_input_order_irrelevant() {
        let unsorted = [11.0, -3.0, 2.0, 0.5];
        let mut scratch = unsorted;
        sort_for_quantiles(&mut scratch);
        assert_eq!(scratch, [-3.0, 0.5, 2.0, 11.0]);
        assert_eq!(linear_at(&scratch, 25.0), -0.375);

        // A different permutation of the same values sorts to the same buffer,
        // which is the whole determinism argument for the unstable sort.
        let mut other = [2.0, 11.0, 0.5, -3.0];
        sort_for_quantiles(&mut other);
        assert_eq!(other, scratch);
    }

    #[test]
    fn sorting_orders_signed_zeros_and_leaves_no_pair_incomparable() {
        let mut scratch = [0.0, -1.0, -0.0, 1.0];
        sort_for_quantiles(&mut scratch);
        assert_eq!(scratch, [-1.0, -0.0, 0.0, 1.0]);
        assert!(scratch[1].is_sign_negative() && scratch[2].is_sign_positive());
    }

    #[test]
    fn the_median_uses_the_general_expression_rather_than_a_midpoint_branch() {
        // The two forms differ by one ulp on some even-length samples. The
        // divergence is real rather than theoretical, so this asserts both
        // halves: that a disagreeing pair exists at all, and that every
        // evaluation follows the general expression.
        let mut disagreements = 0;
        for step in 1..2_000 {
            let lower = f64::from(step) / 7.0;
            let upper = lower * 3.0 + 1.0 / 3.0;
            let general = lower + 0.5 * (upper - lower);
            let midpoint = (lower + upper) / 2.0;
            let actual = linear_at(&[lower, upper], 50.0);
            assert_eq!(
                actual.to_bits(),
                general.to_bits(),
                "median of [{lower}, {upper}] left the general expression"
            );
            disagreements += usize::from(general.to_bits() != midpoint.to_bits());
        }
        assert!(
            disagreements > 0,
            "the midpoint form never disagreed, so this test proves nothing"
        );
    }
}
