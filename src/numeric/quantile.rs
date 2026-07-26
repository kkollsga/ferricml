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
//! for one of them. That is not a hypothetical. Of the four transformers
//! reading quantiles, three take the linear rule and exactly one — binning —
//! takes the averaged inverted CDF, so the parameter carries one exception
//! rather than being a general dial.
//!
//! # Two rule types, because weighting is not a modifier
//!
//! [`QuantileRule`] and [`WeightedQuantileRule`] are deliberately disjoint
//! types rather than one enum plus an optional weight slice. Weighting is
//! **only defined for the inverted-CDF family**: a weighted linear quantile is
//! not a rule anyone has agreed on, it is an invention, and the reference
//! refuses the combination outright rather than guessing. Splitting the types
//! makes that call unrepresentable instead of merely discouraged — there is no
//! `weighted_quantile_sorted(.., QuantileRule::Linear)` to write and no runtime
//! guard to forget. The cost is one extra variant name; the benefit is that the
//! constraint cannot be violated by a caller who never read this paragraph.

/// Which quantile definition an unweighted evaluation is taken under.
///
/// An enum rather than an implied default because the choice belongs at the
/// call site: a reader of
/// `quantile_sorted(column, 50.0, QuantileRule::Linear)` can see which
/// definition a fitted value was frozen against, and a second rule is an added
/// variant rather than a changed meaning.
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
    /// It is the definition FerricML's robust scaling, quantile mapping and
    /// spline knot placement are frozen against.
    Linear,
    /// The discontinuous inverted CDF, averaging across an exact landing.
    ///
    /// Sort ascending as `x[0] <= ... <= x[n-1]`. For a percentile `p`, let
    /// `t = (p / 100) * n` and let `k` be the smallest index whose running
    /// count `k + 1` reaches `t`, clamped to `n - 1`:
    ///
    /// ```text
    /// Q(p) = (x[k] + x[k+1]) / 2    if k + 1 == t exactly and k < n - 1
    /// Q(p) = x[k]                   otherwise
    /// ```
    ///
    /// This is Hyndman–Fan type 2. It is a *step* rule: it returns an order
    /// statistic, or the midpoint of two of them, and never interpolates a
    /// fraction of the way between neighbours. It is the definition FerricML's
    /// binning is frozen against, and the only place in the crate that leaves
    /// [`QuantileRule::Linear`].
    ///
    /// The two disagree materially rather than marginally on small samples: on
    /// `[0, 1, 2, 10]`, `Q(25)` is `0.5` here and `0.75` under the linear rule.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "binning is the first consumer; it removes this attribute"
        )
    )]
    AveragedInvertedCdf,
}

/// Which quantile definition a *weighted* evaluation is taken under.
///
/// Both variants are step rules, and that is the whole reason this type is
/// separate from [`QuantileRule`] — see the module documentation. A weight
/// generalises the running count of the inverted-CDF walk into a running
/// weight; it has no meaning for a rule that interpolates between order
/// statistics, so no such variant exists here to be selected by mistake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "weighted binning is the first consumer; it removes this attribute"
    )
)]
pub(crate) enum WeightedQuantileRule {
    /// The plain inverted CDF: the first order statistic whose running weight
    /// reaches the target.
    ///
    /// With `c[k]` the running weight `w[0] + ... + w[k]`, `W = c[n-1]` and
    /// `t = (p / 100) * W`, this is `x[k]` for the smallest `k` with
    /// `c[k] >= t`, clamped to `n - 1`.
    ///
    /// Hyndman–Fan type 1. It preserves the integer-weight identity **exactly**:
    /// a weight of `k` is bit for bit a `k`-fold repetition of that row, which
    /// is the same property FerricML's tree weights rely on.
    InvertedCdf,
    /// The same walk, averaging across an exact landing.
    ///
    /// `Q(p) = (x[k] + x[k+1]) / 2` when `c[k] == t` exactly and `k < n - 1`,
    /// and `x[k]` otherwise. Hyndman–Fan type 2, and the unit-weight case of
    /// this is exactly [`QuantileRule::AveragedInvertedCdf`].
    ///
    /// The averaging step rounds, so unlike [`Self::InvertedCdf`] this rule
    /// preserves the integer-weight/row-repetition identity *semantically* but
    /// not bit for bit — the measured worst case is `8.9e-16`. That is the same
    /// class of trade-off already recorded for tree weights, and it is stated
    /// here so a caller choosing between the two knows what it is choosing.
    AveragedInvertedCdf,
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

/// Orders a `(value, weight)` scratch buffer so [`weighted_quantile_sorted`]
/// can read it.
///
/// The pair is sorted as one element rather than as two parallel buffers,
/// because a weight that lost track of its value would produce a plausible
/// wrong quantile rather than a failure. The weight is carried, never compared.
///
/// Sorting is **stable** here, which is the one place this differs from
/// [`sort_for_quantiles`]. There the argument for instability is that equal
/// `f64` values are bit-identical, so no permutation of them is observable.
/// That argument does not survive the second field: two pairs with equal values
/// and *different* weights compare equal and are not interchangeable, and the
/// running weight the inverted-CDF walk accumulates depends on the order they
/// end up in. A stable sort makes that order the input order, which is what
/// keeps a weighted fit deterministic.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "weighted binning is the first consumer; it removes this attribute"
    )
)]
pub(crate) fn sort_weighted_for_quantiles(scratch: &mut [(f64, f64)]) {
    scratch.sort_by(|left, right| left.0.total_cmp(&right.0));
}

/// Evaluates one percentile of an ascending-sorted sample.
///
/// Three obligations, all the caller's: `sorted` is non-empty, `sorted` is
/// ascending under [`f64::total_cmp`] — [`sort_for_quantiles`] is how a caller
/// gets there — and `percentile` is a non-NaN value in `[0, 100]`. They are
/// obligations rather than typed errors because every caller in the crate
/// validates its percentile range once at its own public boundary and then
/// evaluates per column, against a buffer it has itself just sorted.
///
/// *Which build each one is checked in* follows the rule
/// `MatrixView::from_validated_parts` already sets for the same situation: an
/// O(1) invariant is asserted in every build, and an invariant whose check
/// would repeat the O(n) work that establishes it is a `debug_assert!`. This
/// function is three consumers' shared primitive, so leaving the rule implicit
/// would mean deciding it separately three times.
///
/// * **Non-empty — asserted in every build.** Below, `len() - 1` on an empty
///   sample wraps wherever overflow checks are off, and the arbitrary index
///   that follows is the difference between a named failure and reading
///   whatever happens to be nearby.
/// * **`percentile` in `[0, 100]` — asserted in every build.** Out of range,
///   the saturating cast below clamps the index while the fraction keeps the
///   out-of-range remainder, so the two stop describing the same position:
///   `Q(-10)` of `[0, 1, 2, 10]` evaluated to `0.7`, which is not an error
///   value but a plausible quantile of that very sample. A NaN percentile
///   returned NaN. Both are measured, and both are why this one does not wait
///   for a debug build.
/// * **Ascending — `debug_assert!`ed.** This is the one obligation whose check
///   is the scan that establishes it, so it cannot ride along beside the work
///   the way the other two do. A debug check is what fits; before this, the
///   obligation the doc named loudest was the only one with nothing behind it.
///
/// The general expression is applied uniformly, including at `p = 50`. Some
/// implementations special-case the median at even `n` to the average of the
/// two middle order statistics, which differs from the general evaluation by
/// one ulp; FerricML does not, so one expression describes every quantile it
/// reports and the difference is carried by comparison tolerances rather than
/// by a branch.
pub(crate) fn quantile_sorted(sorted: &[f64], percentile: f64, rule: QuantileRule) -> f64 {
    assert!(!sorted.is_empty(), "a quantile needs at least one value");
    assert!(
        (0.0..=100.0).contains(&percentile),
        "percentile {percentile} is outside 0..=100"
    );
    // `total_cmp` rather than `<=` so "ascending" means ascending under the
    // order `sort_for_quantiles` imposes, and not a second, weaker order that
    // a buffer straight out of that sort could fail.
    debug_assert!(
        sorted
            .windows(2)
            .all(|pair| pair[0].total_cmp(&pair[1]).is_le()),
        "the sample is not ascending, so its order statistics are not its own"
    );
    match rule {
        QuantileRule::Linear => linear(sorted, percentile),
        QuantileRule::AveragedInvertedCdf => {
            // At unit weights the running weight of the walk below *is* the
            // running count, so the unweighted rule is the weighted one with
            // `c[k] = k + 1` and `W = n`. Expressing it that way rather than as
            // a second formula is what keeps one definition of the inverted-CDF
            // family in the crate: the two could not drift apart, because there
            // is only one walk.
            let count = sorted.len();
            let (index, exact) = inverted_cdf_position(
                (1..=count).map(|running| running as f64),
                count,
                (percentile / 100.0) * count as f64,
            );
            averaged(sorted, index, exact)
        }
    }
}

/// Evaluates one percentile of an ascending-sorted, weighted sample.
///
/// `sorted` holds `(value, weight)` pairs ordered by value —
/// [`sort_weighted_for_quantiles`] is how a caller gets there. The caller's
/// obligations are [`quantile_sorted`]'s, plus weights that are finite,
/// non-negative, and not all zero. Those three are exactly what
/// [`SampleWeights`](crate::data::SampleWeights) already guarantees at the
/// public boundary, so this function inherits them rather than re-checking
/// them per column; the total is still asserted positive, because a walk
/// against a zero total would read an arbitrary index rather than fail.
///
/// # Why two passes rather than a prefix-sum buffer
///
/// The total is summed first and the running weight is accumulated again
/// during the walk, so each call is two sequential passes over `sorted`. A
/// caller-owned prefix-sum buffer would make it one, and is deliberately not
/// taken: it is a second structure parallel to the first, it can disagree with
/// it, and the disagreement would be a wrong quantile rather than a failure.
/// Both passes are `O(n)` against a sort that is `O(n log n)` and a bin count
/// that is small, so the trade costs an order nothing and buys the absence of a
/// synchronisation obligation. If a profile ever disagrees, that is evidence
/// and this comment is where the decision it overturns is written down.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "weighted binning is the first consumer; it removes this attribute"
    )
)]
pub(crate) fn weighted_quantile_sorted(
    sorted: &[(f64, f64)],
    percentile: f64,
    rule: WeightedQuantileRule,
) -> f64 {
    assert!(!sorted.is_empty(), "a quantile needs at least one value");
    assert!(
        (0.0..=100.0).contains(&percentile),
        "percentile {percentile} is outside 0..=100"
    );
    debug_assert!(
        sorted
            .windows(2)
            .all(|pair| pair[0].0.total_cmp(&pair[1].0).is_le()),
        "the sample is not ascending, so its order statistics are not its own"
    );
    debug_assert!(
        sorted.iter().all(|&(_, weight)| weight >= 0.0),
        "a negative weight makes the running weight non-monotone, so \
         'the first index reaching the target' stops being well defined"
    );

    // Rule 2 of the accumulation policy: one sequential pass in ascending
    // index order, named rather than folded, so the walk below accumulates the
    // same terms in the same order and reaches exactly this value at the end.
    let total = super::sum_in_order(sorted.iter().map(|&(_, weight)| weight));
    assert!(
        total > 0.0,
        "the sample weights total {total}, so no percentile of it is defined"
    );

    let mut running = 0.0;
    let (index, exact) = inverted_cdf_position(
        sorted.iter().map(move |&(_, weight)| {
            running += weight;
            running
        }),
        sorted.len(),
        (percentile / 100.0) * total,
    );
    let values = |position: usize| sorted[position].0;
    match rule {
        WeightedQuantileRule::InvertedCdf => values(index),
        WeightedQuantileRule::AveragedInvertedCdf => {
            if exact && index + 1 < sorted.len() {
                (values(index) + values(index + 1)) / 2.0
            } else {
                values(index)
            }
        }
    }
}

/// Where an inverted-CDF rule reads, and whether it lands on a boundary.
///
/// `cumulative` yields the running weight `c[0], c[1], ..., c[n-1]` in
/// ascending index order; `target` is `(p / 100) * c[n-1]`. Returns the
/// smallest index whose running weight reaches the target — clamped to the
/// final order statistic, which is what makes `p = 100` read the maximum
/// instead of running off the end — together with whether it reaches it
/// **exactly**, the equality the averaged rules' midpoint branch turns on.
///
/// This is the single definition of the walk. Both step rules and both the
/// weighted and unweighted forms reach it, which is why the difference between
/// them is a branch on the returned pair rather than a second traversal that
/// could come to disagree with this one.
fn inverted_cdf_position(
    cumulative: impl Iterator<Item = f64>,
    count: usize,
    target: f64,
) -> (usize, bool) {
    let last = count - 1;
    for (index, running) in cumulative.enumerate() {
        if running >= target {
            return (index, running == target);
        }
        if index == last {
            // Only reachable when rounding leaves the final running weight a
            // hair below the target it should equal; the maximum is still the
            // answer, and reporting an inexact landing keeps the averaged rule
            // from reading past the end.
            return (last, false);
        }
    }
    unreachable!("a non-empty sample yields at least one running weight")
}

/// Applies the averaged-inverted-CDF midpoint branch at an already-located
/// index.
fn averaged(sorted: &[f64], index: usize, exact: bool) -> f64 {
    if exact && index + 1 < sorted.len() {
        (sorted[index] + sorted[index + 1]) / 2.0
    } else {
        sorted[index]
    }
}

/// Hyndman–Fan type 7, evaluated exactly as [`QuantileRule::Linear`] states it.
fn linear(sorted: &[f64], percentile: f64) -> f64 {
    let last = sorted.len() - 1;
    let position = (percentile / 100.0) * last as f64;
    let lower = position.floor();
    let fraction = position - lower;
    // `position` is in `0..=last` for a percentile in `0..=100`, so the cast is
    // exact; the bound is re-checked rather than assumed, because a rounded
    // `position` of exactly `last` must read the final order statistic instead
    // of indexing past it.
    //
    // This guard is also the whole of the one-element case: `last` is `0`, so
    // every percentile takes it and returns `x[0]`, bit for bit, `-0.0`
    // included. There was an `n == 1` early return above saying the same thing
    // first. It is gone — nothing could observe which of the two answered, so
    // it was not a guard but a second description of this one, and a second
    // description is a thing that can come to disagree.
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

    /// Each precondition fails loudly, in the build its documentation names.
    ///
    /// A precondition with no falsifier is prose. These are the falsifiers, and
    /// the second is the one that matters most: without its assertion the call
    /// returned `0.7`, which is `x[0] + 0.7 * (x[1] - x[0])` — a number
    /// indistinguishable from a real quantile of the same sample, produced by
    /// the saturating cast clamping the index to `0` while the fraction kept
    /// the out-of-range remainder. A wrong number that looks right is worse
    /// than a panic, which is why it does not wait for a debug build.
    mod preconditions {
        use super::*;

        #[test]
        #[should_panic = "a quantile needs at least one value"]
        fn an_empty_sample_fails_instead_of_wrapping_its_length() {
            let _ = linear_at(&[], 50.0);
        }

        #[test]
        #[should_panic = "percentile -10 is outside"]
        fn a_percentile_below_the_range_fails_instead_of_returning_a_plausible_one() {
            let _ = linear_at(&[0.0, 1.0, 2.0, 10.0], -10.0);
        }

        #[test]
        #[should_panic = "percentile 110 is outside"]
        fn a_percentile_above_the_range_fails_instead_of_saturating_onto_the_maximum() {
            let _ = linear_at(&[0.0, 1.0, 2.0, 10.0], 110.0);
        }

        #[test]
        #[should_panic = "percentile NaN is outside"]
        fn a_nan_percentile_fails_instead_of_returning_nan() {
            let _ = linear_at(&[0.0, 1.0, 2.0, 10.0], f64::NAN);
        }

        /// Sortedness is the debug-only obligation, so its falsifier is too.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic = "not ascending"]
        fn a_descending_sample_fails_in_a_debug_build() {
            let _ = linear_at(&[1.0, 0.0], 50.0);
        }

        /// And the check reads the sort's own order, not a weaker one.
        ///
        /// `[0.0, -0.0]` is non-decreasing under `<=` and *descending* under
        /// `total_cmp`, so a check written with `<=` would accept a buffer
        /// `sort_for_quantiles` would never produce.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic = "not ascending"]
        fn signed_zeros_out_of_total_order_fail_too() {
            let _ = linear_at(&[0.0, -0.0], 50.0);
        }
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

    /// The step rules, weighted and not.
    ///
    /// Every expected value here is a row the specification room derived and
    /// confirmed bit-exactly against independently written implementations of
    /// the stated formulas. They are asserted with exact equality for that
    /// reason: an approximate assertion would accept a rule that is merely
    /// nearby, and "merely nearby" is precisely what every rejected candidate
    /// definition was.
    mod step_rules {
        use super::*;

        fn averaged_at(sample: &[f64], percentile: f64) -> f64 {
            quantile_sorted(sample, percentile, QuantileRule::AveragedInvertedCdf)
        }

        fn weighted_at(sample: &[(f64, f64)], percentile: f64, rule: WeightedQuantileRule) -> f64 {
            weighted_quantile_sorted(sample, percentile, rule)
        }

        fn pairs(values: &[f64], weights: &[f64]) -> Vec<(f64, f64)> {
            values
                .iter()
                .copied()
                .zip(weights.iter().copied())
                .collect()
        }

        /// The worked example, all four cells of it.
        ///
        /// One sample, two rules, weighted and unweighted, is the smallest
        /// thing that pins *both* axes at once: a bug that ignored the weights
        /// would still pass the unweighted column, and one that ignored the
        /// rule would still pass a single row.
        #[test]
        fn the_worked_example_reproduces_all_four_cells() {
            let values = [1.0, 2.0, 3.0, 10.0];
            let unit = pairs(&values, &[1.0; 4]);
            let heavy = pairs(&values, &[1.0, 1.0, 1.0, 3.0]);
            let edges = [0.0, 50.0, 100.0];

            let inverted_unweighted: Vec<f64> = edges
                .iter()
                .map(|&p| weighted_at(&unit, p, WeightedQuantileRule::InvertedCdf))
                .collect();
            assert_eq!(inverted_unweighted, vec![1.0, 2.0, 10.0]);

            let inverted_weighted: Vec<f64> = edges
                .iter()
                .map(|&p| weighted_at(&heavy, p, WeightedQuantileRule::InvertedCdf))
                .collect();
            assert_eq!(inverted_weighted, vec![1.0, 3.0, 10.0]);

            let averaged_unweighted: Vec<f64> = edges
                .iter()
                .map(|&p| weighted_at(&unit, p, WeightedQuantileRule::AveragedInvertedCdf))
                .collect();
            assert_eq!(averaged_unweighted, vec![1.0, 2.5, 10.0]);

            let averaged_weighted: Vec<f64> = edges
                .iter()
                .map(|&p| weighted_at(&heavy, p, WeightedQuantileRule::AveragedInvertedCdf))
                .collect();
            assert_eq!(averaged_weighted, vec![1.0, 6.5, 10.0]);
        }

        /// The unweighted rule is the weighted one at unit weights, and the
        /// binning rule differs materially from the scaling rule.
        #[test]
        fn the_unweighted_step_rule_is_the_unit_weight_case_and_leaves_the_linear_one() {
            let sample = [0.0, 1.0, 2.0, 10.0];
            let unit = pairs(&sample, &[1.0; 4]);
            for percentile in DIVERGENT {
                let unweighted = averaged_at(&sample, percentile);
                let weighted =
                    weighted_at(&unit, percentile, WeightedQuantileRule::AveragedInvertedCdf);
                assert_eq!(
                    unweighted.to_bits(),
                    weighted.to_bits(),
                    "Q({percentile}) disagreed between the unweighted rule and \
                     its own unit-weight case"
                );
            }

            // The row the whole two-rule design exists for: at the quartiles
            // the two definitions are not near each other.
            assert_eq!(averaged_at(&sample, 25.0), 0.5);
            assert_eq!(linear_at(&sample, 25.0), 0.75);
            assert_eq!(averaged_at(&sample, 75.0), 6.0);
            assert_eq!(linear_at(&sample, 75.0), 4.0);

            // And where they happen to agree, they agree exactly, so a test
            // that only probed the median would prove nothing.
            assert_eq!(averaged_at(&sample, 50.0), linear_at(&sample, 50.0));
        }

        /// An integer weight is a repeated row — exactly for one rule, and to a
        /// bounded rounding error for the other.
        ///
        /// The asymmetry is the recorded trade-off between the two step rules,
        /// so it is asserted in both directions: the exact rule must be exact,
        /// and the averaging rule must be *within* the bound rather than
        /// merely close to it.
        #[test]
        fn an_integer_weight_is_a_repeated_row() {
            let values = [1.0, 2.0, 3.0, 10.0];
            let weighted = pairs(&values, &[1.0, 2.0, 1.0, 3.0]);
            let repeated = pairs(&[1.0, 2.0, 2.0, 3.0, 10.0, 10.0, 10.0], &[1.0; 7]);

            let mut averaged_disagreements = 0;
            for step in 0..=100 {
                let percentile = f64::from(step);
                assert_eq!(
                    weighted_at(&weighted, percentile, WeightedQuantileRule::InvertedCdf),
                    weighted_at(&repeated, percentile, WeightedQuantileRule::InvertedCdf),
                    "the inverted CDF must reproduce row repetition bit for bit \
                     at p = {percentile}"
                );

                let rule = WeightedQuantileRule::AveragedInvertedCdf;
                let from_weights = weighted_at(&weighted, percentile, rule);
                let from_rows = weighted_at(&repeated, percentile, rule);
                assert!(
                    (from_weights - from_rows).abs() <= 8.9e-16 * from_rows.abs().max(1.0),
                    "the averaged rule left its recorded rounding envelope at \
                     p = {percentile}: {from_weights} against {from_rows}"
                );
                averaged_disagreements +=
                    usize::from(from_weights.to_bits() != from_rows.to_bits());
            }
            assert_eq!(
                averaged_disagreements, 0,
                "on this sample the averaging rule happens to agree bit for \
                 bit; the envelope above is what is claimed, and this counter \
                 records what was observed rather than asserting exactness"
            );
        }

        /// Fractional weights move the answer, and a zero weight removes a row
        /// without removing its value from the sample.
        #[test]
        fn fractional_and_zero_weights_behave_as_a_running_weight_should() {
            let values = [1.0, 2.0, 3.0, 10.0];
            let fractional = pairs(&values, &[0.25, 0.5, 0.25, 7.5]);
            // Nearly all the weight sits on the last value, so the median has
            // to be there too.
            assert_eq!(
                weighted_at(&fractional, 50.0, WeightedQuantileRule::InvertedCdf),
                10.0
            );

            // A zero-weight row contributes nothing to the running weight, so
            // the walk steps straight past it.
            let zeroed = pairs(&values, &[1.0, 0.0, 0.0, 1.0]);
            assert_eq!(
                weighted_at(&zeroed, 25.0, WeightedQuantileRule::InvertedCdf),
                1.0
            );
            assert_eq!(
                weighted_at(&zeroed, 75.0, WeightedQuantileRule::InvertedCdf),
                10.0
            );
        }

        #[test]
        fn a_single_value_is_every_percentile_of_itself_under_both_step_rules() {
            let one = [(5.0, 3.0)];
            for percentile in DIVERGENT {
                assert_eq!(averaged_at(&[5.0], percentile), 5.0);
                assert_eq!(
                    weighted_at(&one, percentile, WeightedQuantileRule::InvertedCdf),
                    5.0
                );
                assert_eq!(
                    weighted_at(&one, percentile, WeightedQuantileRule::AveragedInvertedCdf),
                    5.0
                );
            }
        }

        #[test]
        fn the_endpoints_are_the_extrema_under_both_step_rules() {
            let sample = [-3.0, 0.5, 2.0, 11.0];
            let weighted = pairs(&sample, &[2.0, 1.0, 4.0, 0.5]);
            for rule in [
                WeightedQuantileRule::InvertedCdf,
                WeightedQuantileRule::AveragedInvertedCdf,
            ] {
                assert_eq!(weighted_at(&weighted, 0.0, rule), -3.0, "{rule:?} at p=0");
                assert_eq!(
                    weighted_at(&weighted, 100.0, rule),
                    11.0,
                    "{rule:?} at p=100"
                );
            }
            assert_eq!(averaged_at(&sample, 0.0), -3.0);
            assert_eq!(averaged_at(&sample, 100.0), 11.0);
        }

        /// A tied bracket has nothing to average, so the midpoint branch cannot
        /// move the result off the tie.
        #[test]
        fn ties_collapse_the_midpoint_branch_onto_themselves() {
            let sample = [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0];
            assert_eq!(averaged_at(&sample, 25.0), 5.0);
            assert_eq!(averaged_at(&sample, 50.0), 5.0);
            assert_eq!(averaged_at(&sample, 75.0), 5.0);
        }

        /// Both step rules are non-decreasing in the percentile.
        ///
        /// The sweep also counts its ascents, because a rule that returned a
        /// constant would satisfy monotonicity alone. A step rule over nine
        /// distinct values can only rise a bounded number of times, which is
        /// what separates it from the continuous rule sweeping the same sample.
        #[test]
        fn the_step_rules_are_monotone_and_actually_step() {
            let sample = [-7.5, -0.25, 0.0, 1.0, 3.5, 12.0, 900.0, 1e6];
            let mut previous = f64::NEG_INFINITY;
            let mut distinct = 0_usize;
            for step in 0..=1_000 {
                let value = averaged_at(&sample, f64::from(step) / 10.0);
                assert!(value >= previous, "the sweep fell back at step {step}");
                distinct += usize::from(previous.is_finite() && value > previous);
                previous = value;
            }
            assert_eq!(
                previous,
                sample[sample.len() - 1],
                "the sweep ends at the top"
            );
            assert!(
                (1..=2 * sample.len()).contains(&distinct),
                "the sweep rose {distinct} times; a step rule over {} values \
                 rises a bounded number of times, and a constant or a \
                 continuous rule would fall outside that",
                sample.len()
            );
        }

        /// Each precondition of the weighted evaluator fails loudly.
        #[test]
        #[should_panic = "a quantile needs at least one value"]
        fn an_empty_weighted_sample_fails() {
            let _ = weighted_at(&[], 50.0, WeightedQuantileRule::InvertedCdf);
        }

        #[test]
        #[should_panic = "percentile 110 is outside"]
        fn an_out_of_range_weighted_percentile_fails() {
            let sample = pairs(&[0.0, 1.0], &[1.0, 1.0]);
            let _ = weighted_at(&sample, 110.0, WeightedQuantileRule::InvertedCdf);
        }

        /// A zero total is the one weight failure that is not already excluded
        /// by `SampleWeights`, because it is a property of the whole vector
        /// rather than of any one entry — and a walk against it would read an
        /// arbitrary index rather than fail.
        #[test]
        #[should_panic = "total 0"]
        fn an_all_zero_weight_vector_fails_instead_of_reading_an_arbitrary_index() {
            let sample = pairs(&[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]);
            let _ = weighted_at(&sample, 50.0, WeightedQuantileRule::InvertedCdf);
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic = "not ascending"]
        fn a_descending_weighted_sample_fails_in_a_debug_build() {
            let sample = pairs(&[1.0, 0.0], &[1.0, 1.0]);
            let _ = weighted_at(&sample, 50.0, WeightedQuantileRule::InvertedCdf);
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic = "non-monotone"]
        fn a_negative_weight_fails_in_a_debug_build() {
            let sample = pairs(&[0.0, 1.0], &[1.0, -1.0]);
            let _ = weighted_at(&sample, 50.0, WeightedQuantileRule::InvertedCdf);
        }

        /// Sorting carries each weight with its own value.
        ///
        /// The failure this guards against is silent: a sort that ordered the
        /// values and left the weights behind produces a perfectly plausible
        /// quantile of the wrong distribution.
        #[test]
        fn sorting_keeps_each_weight_with_its_value() {
            let mut scratch = [(10.0, 3.0), (1.0, 1.0), (3.0, 1.0), (2.0, 1.0)];
            sort_weighted_for_quantiles(&mut scratch);
            assert_eq!(scratch, [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (10.0, 3.0)]);
            assert_eq!(
                weighted_at(&scratch, 50.0, WeightedQuantileRule::AveragedInvertedCdf),
                6.5,
                "the worked example, reached through the sort rather than \
                 through a hand-ordered literal"
            );
        }

        /// Equal values with different weights keep their input order.
        ///
        /// This is what the stable sort buys, and it is not observable through
        /// the values alone — which is exactly why it needs its own assertion.
        #[test]
        fn equal_values_keep_their_input_order_so_the_running_weight_is_fixed() {
            let mut scratch = [(5.0, 1.0), (5.0, 2.0), (5.0, 3.0), (1.0, 1.0)];
            sort_weighted_for_quantiles(&mut scratch);
            assert_eq!(
                scratch,
                [(1.0, 1.0), (5.0, 1.0), (5.0, 2.0), (5.0, 3.0)],
                "the three equal values kept the order they were given in"
            );
        }
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
