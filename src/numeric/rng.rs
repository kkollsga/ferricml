//! The crate's shared deterministic pseudo-random source.
//!
//! One SplitMix64 generator serves every module that needs reproducible
//! randomness — bootstrap sampling and feature subsetting in the forests,
//! permutation-based inspection, and every shuffled dataset split — so a seed
//! means the same thing everywhere and no consumer has to reach into another
//! module's internals.
//!
//! "One" is literal, and `rng-single-source` in `scripts/check_source_layout.py`
//! keeps it that way: a second definition outside this module fails the gate.
//! `model_selection::split` carried a character-identical private copy until
//! 2026-07-26, documented as an independent stream while emitting the same
//! values for the same seed — the shape rule 6 exists to forbid.
//!
//! The integration-test crates cannot reach a `pub(crate)` item, so they have one
//! generator of their own in `tests/support/rng.rs` and a sibling rule,
//! `test-rng-single-source`, holding it to the same "exactly one" standard. Both
//! test binaries that needed the crate's own stream carried a third and fourth
//! copy of this core until the same day.
//!
//! The stream is part of FerricML's determinism contract: for a given seed the
//! sequence of `next_u64` values, and therefore every fitted artifact derived
//! from it, is frozen. Changing the mixing constants or the rejection bound
//! would change fitted models, so those bytes are covered by a frozen-stream
//! test below as well as by the forests' packed fingerprints.
//!
//! Frozen streams do not cover the *rejection* itself, and it took measuring to
//! say so: at every bound a caller passes, and at every bound the tests used
//! until 2026-07-26, a draw is rejected with probability between `5e-20` and
//! `3e-17`. Deleting the rejection loop outright left the whole suite green,
//! frozen fixtures included. The branch is therefore covered by its own test, at
//! a bound where a third of the stream is rejected.

/// SplitMix64 with rejection-sampled bounded integers.
pub(crate) struct OwnedRng {
    state: u64,
}

impl OwnedRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    /// A uniform draw in `[0, 1)`, one `next_u64` per call.
    ///
    /// The top 53 bits are used because that is exactly an `f64`'s significand:
    /// every representable multiple of `2^-53` in `[0, 1)` is drawn with the
    /// same probability, and none is drawn twice as often as its neighbour —
    /// which the low bits of a shorter draw scaled up would not give. `1.0` is
    /// unreachable, so a caller mapping onto `[min, max)` cannot land on its
    /// upper endpoint by arithmetic alone.
    ///
    /// Consuming exactly one `next_u64` is part of the contract, not an
    /// implementation detail: a fitted tree's reproducibility depends on how
    /// many values each decision takes out of the stream.
    pub(crate) fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// A uniform draw in `[0, upper)`, by rejection sampling.
    ///
    /// `2^64` is not a multiple of `upper` in general, so the `2^64 mod upper`
    /// lowest values of the stream have one residue more than the rest. Drawing
    /// them again instead of reducing them is what makes the result uniform;
    /// without it the low residues are over-represented by up to a factor of
    /// two. The rejected region is exactly `reject_below` values wide, so the
    /// accepted region is a whole number of periods of `bound`.
    ///
    /// # Panics
    ///
    /// If `upper` is zero. The check is `assert!` rather than `debug_assert!`
    /// because it costs one comparison next to a division and holds in every
    /// build: without it a release build reaches `% 0` and panics from inside
    /// the generator with a message naming arithmetic rather than the caller's
    /// empty candidate set. A typed refusal would be the wrong shape here —
    /// every caller has already established that its node, feature set or
    /// sample is non-empty, so an empty `upper` is a defect in this crate and
    /// not an input to reject, and returning a `Result` would thread one
    /// through the innermost loop of forest training to say so.
    pub(crate) fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "a bounded draw needs at least one candidate");
        let bound = upper as u64;
        let reject_below = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                return (value % bound) as usize;
            }
        }
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn derive_tree_seed(global_seed: u64, tree_index: u64) -> u64 {
    mix64(global_seed ^ tree_index.wrapping_mul(0xd1b5_4a32_d192_ed03))
}

/// Derives one repetition's seed from a configured seed.
///
/// Repeated K-fold and grouped shuffle splitting both draw a sequence of
/// independent partitions from one number, and both need consecutive seeds and
/// consecutive repetitions not to produce overlapping streams — so the index is
/// mixed rather than added.
///
/// It lives here beside [`derive_tree_seed`] rather than in the splitters for
/// the reason rule 6 gives: a seed has to mean the same thing everywhere, and a
/// module deriving its own seeds from its own copy of the mixer is the same
/// defect as a module defining its own generator. Splitters call this and
/// nothing else; the mixing function itself stays private to this file.
pub(crate) fn derive_repetition_seed(global_seed: u64, repetition: u64) -> u64 {
    mix64(global_seed ^ mix64(repetition ^ 0x9e37_79b9_7f4a_7c15))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streams captured from the pre-promotion implementation in
    /// `ensemble::random_forest::rng`. They are the proof that moving the
    /// generator into `numeric` did not perturb a single fitted model.
    #[test]
    fn stream_matches_the_pre_promotion_forest_generator() {
        let expected: [(u64, [u64; 8]); 4] = [
            (
                0,
                [
                    16294208416658607535,
                    7960286522194355700,
                    487617019471545679,
                    17909611376780542444,
                    1961750202426094747,
                    6038094601263162090,
                    3207296026000306913,
                    14232521865600346940,
                ],
            ),
            (
                1,
                [
                    10451216379200822465,
                    13757245211066428519,
                    17911839290282890590,
                    8196980753821780235,
                    8195237237126968761,
                    14072917602864530048,
                    16184226688143867045,
                    9648886400068060533,
                ],
            ),
            (
                42,
                [
                    13679457532755275413,
                    2949826092126892291,
                    5139283748462763858,
                    6349198060258255764,
                    701532786141963250,
                    16015981125662989062,
                    4028864712777624925,
                    14769051326987775908,
                ],
            ),
            (
                u64::MAX,
                [
                    16490336266968443936,
                    16834447057089888969,
                    4048727598324417001,
                    7862637804313477842,
                    13015481187462834606,
                    15212506146343009075,
                    17388166129998380965,
                    4638043754431676516,
                ],
            ),
        ];
        for (seed, stream) in expected {
            let mut rng = OwnedRng::new(seed);
            let actual: Vec<u64> = (0..stream.len()).map(|_| rng.next_u64()).collect();
            assert_eq!(actual, stream, "stream changed for seed {seed}");
        }
    }

    #[test]
    fn bounded_indices_match_the_pre_promotion_forest_generator() {
        let expected: [(u64, [usize; 12]); 3] = [
            (0, [5, 0, 9, 4, 7, 0, 3, 0, 9, 0, 1, 6]),
            (7, [7, 4, 6, 3, 4, 5, 8, 2, 5, 5, 3, 6]),
            (u64::MAX, [6, 9, 1, 2, 6, 5, 5, 6, 0, 2, 9, 7]),
        ];
        for (seed, indices) in expected {
            let mut rng = OwnedRng::new(seed);
            let actual: Vec<usize> = (0..indices.len()).map(|_| rng.index(10)).collect();
            assert_eq!(actual, indices, "indices changed for seed {seed}");
        }

        // A single-element bound is the degenerate case every caller hits when
        // a node or feature set has exactly one candidate.
        let mut rng = OwnedRng::new(3);
        assert_eq!((0..8).map(|_| rng.index(1)).collect::<Vec<_>>(), vec![0; 8]);
    }

    #[test]
    fn derived_tree_seeds_match_the_pre_promotion_forest_generator() {
        assert_eq!(
            (0..6).map(|i| derive_tree_seed(0, i)).collect::<Vec<_>>(),
            vec![
                0,
                9370218965779684112,
                7792259576135971849,
                6957767622843056530,
                8786639878720926469,
                8577097995239418960,
            ]
        );
        assert_eq!(
            (0..6)
                .map(|i| derive_tree_seed(0xdead_beef, i))
                .collect::<Vec<_>>(),
            vec![
                5622224078331092714,
                12620482824835280752,
                8826565329999008893,
                9691975008012567232,
                17013454048661918233,
                16161299606447644327,
            ]
        );
    }

    /// Seeds captured from `model_selection::split::repeat_seed` *before* the
    /// split module's private duplicate of this generator was deleted. They are
    /// the proof that folding the two streams into one moved no partition:
    /// every repeated K-fold and grouped shuffle split starts from one of
    /// these numbers.
    #[test]
    fn derived_repetition_seeds_match_the_pre_unification_split_generator() {
        assert_eq!(
            (0..6)
                .map(|i| derive_repetition_seed(0, i))
                .collect::<Vec<_>>(),
            vec![
                5197578548964807871,
                4922461756044938104,
                16576549522093199164,
                15916886550466581944,
                16438634200498821406,
                14037225222889099931,
            ]
        );
        assert_eq!(
            (0..6)
                .map(|i| derive_repetition_seed(0xdead_beef, i))
                .collect::<Vec<_>>(),
            vec![
                948475220252533093,
                14531754899820632363,
                8025580981012917048,
                6004716973091366453,
                14508645644872444640,
                618719326021940468,
            ]
        );

        // Consecutive configured seeds must not shift the same sequence by one
        // repetition — the property the double mix exists for, asserted rather
        // than described.
        assert_eq!(derive_repetition_seed(4, 0), 1257538232492452125);
        assert_eq!(derive_repetition_seed(8, 0), 2834716988604184534);
        assert_ne!(derive_repetition_seed(4, 1), derive_repetition_seed(5, 0));
    }

    /// The unit draw is part of the same frozen determinism contract the
    /// integer stream is: a fitted randomized tree's thresholds come straight
    /// out of these values, so changing the construction would change models.
    #[test]
    fn unit_draws_are_frozen_and_stay_in_the_half_open_unit_interval() {
        let expected: [(u64, [f64; 4]); 2] = [
            (
                0,
                [
                    0.8833108082136426,
                    0.43152799704850997,
                    0.026433771592597743,
                    0.9708819781538285,
                ],
            ),
            (
                7,
                [
                    0.3898297483912715,
                    0.01678829452815611,
                    0.9007606806068834,
                    0.5829302930280781,
                ],
            ),
        ];
        for (seed, stream) in expected {
            let mut rng = OwnedRng::new(seed);
            let actual: Vec<f64> = (0..stream.len()).map(|_| rng.unit_f64()).collect();
            assert_eq!(actual, stream, "unit stream changed for seed {seed}");
        }

        // One `next_u64` per call, stated as a test rather than as a comment:
        // a tree's reproducibility depends on how many values each decision
        // takes out of the stream, so a second draw here would move models.
        let mut counted = OwnedRng::new(99);
        let _ = counted.unit_f64();
        let mut stepped = OwnedRng::new(99);
        let _ = stepped.next_u64();
        assert_eq!(counted.next_u64(), stepped.next_u64());

        // `1.0` is unreachable and `0.0` is not, which is what makes a draw
        // mapped onto `[min, max)` unable to reach its upper endpoint.
        let mut rng = OwnedRng::new(u64::MAX);
        for _ in 0..20_000 {
            let value = rng.unit_f64();
            assert!((0.0..1.0).contains(&value), "unit draw escaped: {value}");
        }
    }

    #[test]
    fn bounded_indices_stay_in_range_and_reject_uniformly_at_awkward_bounds() {
        // A bound that does not divide 2^64 exercises the rejection branch.
        for upper in [3_usize, 7, 10, 1000] {
            let mut rng = OwnedRng::new(u64::from(upper as u32));
            let mut seen = vec![false; upper.min(16)];
            for _ in 0..4_000 {
                let index = rng.index(upper);
                assert!(index < upper, "index {index} escaped bound {upper}");
                if index < seen.len() {
                    seen[index] = true;
                }
            }
            assert!(seen.iter().all(|&hit| hit), "unreachable index for {upper}");
        }
    }

    /// The bound above at which rejection is *likely*, so the branch that makes
    /// the draw uniform is actually covered.
    ///
    /// `2^64 mod bound` is what decides how often a draw is rejected, and for
    /// every bound a caller in this crate passes it is minute: at the bounds the
    /// test above uses — `3`, `7`, `10`, `1000` — the rejection probability runs
    /// from `5e-20` to `3e-17`, so no run of that test has ever taken the
    /// branch. Measured by deleting the rejection loop outright: the whole test
    /// suite, including every frozen-stream fixture above and
    /// `tests/reference_semantics.rs`, stayed green.
    ///
    /// A bound near `(2/3) · 2^64` is where the bias is largest. Rejection then
    /// happens on a third of the draws, and the residues below `2^64 mod bound`
    /// — the lower *half* of the range at this bound — would otherwise be drawn
    /// twice as often as the upper half. That is the difference between one half
    /// and two thirds of the draws, which no amount of stream is needed to see.
    ///
    /// A 64-bit `usize` is what makes such a bound expressible; on a narrower
    /// target `index` cannot be handed one, and the branch is unreachable rather
    /// than untested.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn a_bound_that_rejects_a_third_of_the_stream_is_still_drawn_uniformly() {
        const BOUND: usize = 0xAAAA_AAAA_AAAA_AAAA;
        // The mathematical definition, so this is not the implementation's
        // `wrapping_neg` idiom restated — it also checks that idiom.
        let reject_below = ((1_u128 << 64) % BOUND as u128) as u64;

        // The premise, asserted rather than described: a third of the stream is
        // rejected, and the doubled residues are half of the bound.
        assert_eq!(reject_below, 6_148_914_691_236_517_206);
        assert_eq!(reject_below as u128 * 3, (1_u128 << 64) + 2);
        assert_eq!(reject_below as u128 * 2, BOUND as u128 + 2);

        // One call, modelled against the specification: a rejection-sampled
        // draw skips every value below `reject_below` and reduces the first one
        // that is not, leaving the stream one value past it.
        let mut rng = OwnedRng::new(11);
        let drawn = rng.index(BOUND);

        let mut oracle = OwnedRng::new(11);
        let mut skipped = 0_usize;
        let accepted = loop {
            let value = oracle.next_u64();
            if value >= reject_below {
                break value;
            }
            skipped += 1;
        };
        assert_eq!(
            skipped, 2,
            "seed 11 was chosen because its first two draws are rejected"
        );
        assert_eq!(drawn as u64, accepted % BOUND as u64);
        assert_eq!(
            rng.next_u64(),
            oracle.next_u64(),
            "the draw consumed a different number of values than rejection sampling does"
        );

        // And the distribution the branch exists for. Deleting the rejection
        // loop leaves `value % bound`, under which the lower half of the range
        // has two preimages and the upper half one, so this fraction becomes
        // two thirds. Every seed is fixed, so the numbers below are exact and
        // the band is a statement about bias rather than a flake budget.
        for seed in [0_u64, 1, 7, 42, u64::MAX] {
            let mut rng = OwnedRng::new(seed);
            let draws = 4_000;
            let low = (0..draws)
                .filter(|_| (rng.index(BOUND) as u64) < reject_below)
                .count();
            let fraction = low as f64 / draws as f64;
            assert!(
                (0.45..=0.55).contains(&fraction),
                "seed {seed}: {fraction} of the draws fell in the lower half, \
                 which a uniform draw puts at one half and a modulo reduction \
                 of the raw stream at two thirds"
            );
        }
    }

    /// An empty candidate set is a caller defect, and it says so in every build.
    ///
    /// Without the release assertion this reaches `% 0` and panics from inside
    /// the generator with a message about arithmetic, which names neither the
    /// invariant nor the caller that broke it.
    #[test]
    #[should_panic(expected = "a bounded draw needs at least one candidate")]
    fn a_zero_bound_is_refused_by_name() {
        OwnedRng::new(0).index(0);
    }
}
