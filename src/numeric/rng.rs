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

/// Derives a synthetic dataset's stream state from a caller's seed.
///
/// `src/datasets/` generates design matrices that are then handed to estimators
/// which draw from this same generator. If a recipe seeded with `s` walked the
/// same sequence as a forest seeded with `s`, the data would be correlated with
/// the model's own randomness — the exposure `tests/support/rng.rs` names as the
/// reason a separate test generator exists at all. Mixing here makes the two
/// disjoint by construction instead of by convention.
///
/// It lives beside [`derive_tree_seed`] and [`derive_repetition_seed`] for the
/// reason rule 6 gives and those two already follow: [`mix64`] stays private to
/// this file, so a module that needs a derived seed calls a named entry point
/// here rather than carrying its own copy of the mixer. `rng-single-source`
/// enforces exactly that — the SplitMix64 markers appearing under
/// `src/datasets/` would fail the gate.
///
/// The constant is the first eight bytes of SHA-512's initial state, chosen for
/// having no relationship to anything else in this file rather than for any
/// property of its own.
///
/// **This is for new recipes only.** The absorbed reference and bench presets
/// pin *raw* stream states, because their frozen outputs were recorded against
/// `OwnedRng::new(seed)` directly; routing those through a derivation would move
/// every fixture they protect. `datasets::Source::Sampled` therefore carries a
/// raw state, and only the convenience constructor for new recipes passes a
/// caller's seed through here.
///
/// # Disjointness is exact, not probabilistic
///
/// [`mix64`] is a bijection, so `derive_tree_seed(s, i)` and this function
/// collide exactly when `i * 0xd1b5_4a32_d192_ed03 == 0x6a09_e667_f3bc_c908`
/// modulo `2^64`. That multiplier is odd and therefore invertible, so there is
/// **exactly one** such `i`, it is the same for every seed, and it is
/// `10_380_603_426_675_257_432`. The same argument over
/// [`derive_repetition_seed`] gives exactly one repetition index,
/// `10_759_190_110_990_431_431`. Both are past `10^19`; a forest fitting that
/// many trees, or a splitter running that many repetitions, would not finish.
/// The test below asserts both collisions rather than describing them, which is
/// also what proves the count is one and not zero.
#[cfg(any(feature = "datasets", test))]
pub(crate) fn derive_dataset_stream(seed: u64) -> u64 {
    mix64(seed ^ 0x6a09_e667_f3bc_c908)
}

/// A 32-bit xorshift generator, the crate's second and last generator core.
///
/// This exists for one reason: three benchmark fixtures were written against
/// xorshift32 streams before `src/datasets/` did, and their outputs are the
/// inputs to `bench-history`, whose per-release results are immutable. Absorbing
/// those fixtures into the dataset generator has to reproduce their bytes
/// exactly, and no SplitMix64 construction can. So the stream moves here rather
/// than the fixtures moving.
///
/// Rule 6 in `src/numeric/mod.rs` names it as the only permitted core besides
/// [`OwnedRng`], and it is deliberately not a general-purpose alternative: a new
/// consumer wanting reproducible randomness uses [`OwnedRng`]. This one is a
/// compatibility surface for frozen bench inputs, and its stream is pinned by a
/// test below for the same reason SplitMix64's is.
///
/// What lives here is the *core* and nothing else. The map from a draw onto a
/// design-matrix value is `src/datasets/`'s, beside the fixtures it is frozen
/// against; [`OwnedRng::unit_f64`] is on the generator instead because its draw
/// count is part of every fitted estimator's determinism contract, which no
/// dataset value map is.
#[cfg(any(feature = "datasets", test))]
pub(crate) struct Xorshift32 {
    state: u32,
}

#[cfg(any(feature = "datasets", test))]
impl Xorshift32 {
    /// Starts a stream from `state` exactly.
    ///
    /// # Panics
    ///
    /// If `state` is zero. Zero is xorshift's fixed point — it maps to itself
    /// forever — so a zero-seeded stream is a constant, not a sequence. Every
    /// caller inside this crate passes a compile-time constant, so this is a
    /// defect rather than an input to reject, which is the same reasoning
    /// [`OwnedRng::index`] applies to an empty bound. A *caller-supplied* state
    /// arrives through `datasets::Recipe`, which refuses zero at its
    /// constructor with a typed error before reaching this.
    pub(crate) const fn from_state(state: u32) -> Self {
        assert!(state != 0, "a xorshift32 stream cannot start at zero");
        Self { state }
    }

    /// The next raw draw, post-mutation.
    ///
    /// The shift triple and the fact that the *mutated* state is returned are
    /// both part of the frozen contract: the bench fixtures compute their value
    /// from the state after all three shifts, so returning the pre-mutation
    /// state would shift every fixture by one draw.
    pub(crate) fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
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

    /// The dataset derivation is frozen for the same reason the tree and
    /// repetition derivations are.
    ///
    /// A recipe built through `datasets::Recipe::seeded` starts its stream at
    /// one of these numbers, so changing the constant or the mixing here would
    /// move every design matrix a caller generates from a seed — silently, and
    /// with no aggregate quality lane able to see it.
    #[test]
    fn derived_dataset_streams_are_frozen() {
        let expected: [(u64, u64); 5] = [
            (0, 5272463233947570727),
            (1, 3847398142028685078),
            (42, 1767972203709790677),
            (0xdead_beef, 11053417610396211674),
            (u64::MAX, 18280701212104791470),
        ];
        for (seed, state) in expected {
            assert_eq!(
                derive_dataset_stream(seed),
                state,
                "dataset stream changed for seed {seed}"
            );
        }
    }

    /// A dataset stream is disjoint from every estimator stream the same seed
    /// reaches, asserted rather than described.
    ///
    /// This is the whole reason [`derive_dataset_stream`] exists: a design
    /// matrix drawn from seed `s` must not walk the sequence a forest seeded
    /// with `s` walks, or the data is correlated with the model's own
    /// randomness. `tests/support/rng.rs` names that exposure as the reason a
    /// separate test generator exists at all, and a dataset generator inside the
    /// crate has it in the same shape.
    ///
    /// The collision indices at the end are not decoration. [`mix64`] is a
    /// bijection, so *some* tree index and *some* repetition index reach the
    /// dataset state for any seed; the question is only which. Pinning them
    /// proves the count is exactly one in each family — a "no collision
    /// anywhere" claim would be false — and shows both are past `10^19`, where
    /// no forest or splitter can reach.
    #[test]
    fn a_dataset_stream_is_disjoint_from_the_estimator_streams_of_the_same_seed() {
        for seed in [0_u64, 1, 7, 42, 0xdead_beef, u64::MAX] {
            let dataset = derive_dataset_stream(seed);
            assert_ne!(
                dataset, seed,
                "seed {seed} derives to itself, so `Sampled` would reuse the raw state"
            );
            assert_ne!(
                dataset,
                derive_tree_seed(seed, 0),
                "seed {seed} collides with its own forest's first tree"
            );
            for index in 0..4_096_u64 {
                assert_ne!(
                    dataset,
                    derive_tree_seed(seed, index),
                    "seed {seed} collides with tree {index}"
                );
                assert_ne!(
                    dataset,
                    derive_repetition_seed(seed, index),
                    "seed {seed} collides with repetition {index}"
                );
            }

            // The one index in each family that does collide, and it is the
            // same index for every seed because the mixer is applied last.
            assert_eq!(derive_tree_seed(seed, 10_380_603_426_675_257_432), dataset);
            assert_eq!(
                derive_repetition_seed(seed, 10_759_190_110_990_431_431),
                dataset
            );
        }
    }

    /// The xorshift32 stream is frozen at the two states the bench fixtures
    /// use, captured before those fixtures were ported.
    ///
    /// `benches/models.rs` starts at `0x9e37_79b9` and `benches/boosting.rs` at
    /// `0x243f_6a88`. `bench-history` compares against immutable per-release
    /// results, so a changed draw here invalidates every historical baseline
    /// rather than reporting a regression — which makes these literals a
    /// stronger contract than an ordinary frozen stream, not a weaker one.
    #[test]
    fn the_xorshift32_stream_is_frozen_at_the_states_the_bench_fixtures_use() {
        let expected: [(u32, [u32; 8]); 3] = [
            (
                1,
                [
                    270369, 67634689, 2647435461, 307599695, 2398689233, 745495504, 632435482,
                    435756210,
                ],
            ),
            (
                0x9e37_79b9,
                [
                    1359758873, 3761132862, 2075758394, 25405621, 3862129951, 4186559031,
                    3122997712, 4244368831,
                ],
            ),
            (
                0x243f_6a88,
                [
                    3836725727, 2937111989, 1130492582, 3683945404, 377943917, 3305986136,
                    2932573243, 400419005,
                ],
            ),
        ];
        for (state, stream) in expected {
            let mut rng = Xorshift32::from_state(state);
            let actual: Vec<u32> = (0..stream.len()).map(|_| rng.next_u32()).collect();
            assert_eq!(actual, stream, "stream changed for state {state:#x}");
        }

        // The *mutated* state is what comes back. Written out here as the three
        // shifts rather than as another literal, because returning the
        // pre-mutation state is the one plausible transcription error and it
        // would shift every bench fixture by exactly one draw.
        let mut state = 0x9e37_79b9_u32;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        assert_eq!(Xorshift32::from_state(0x9e37_79b9).next_u32(), state);
        assert_ne!(state, 0x9e37_79b9);
    }

    /// Zero is unreachable from a non-zero state, which is what makes
    /// [`Xorshift32::from_state`]'s assertion sufficient rather than a first
    /// line of defence.
    ///
    /// Each of the three steps is an invertible map on `u32` — `x ^ (x << k)`
    /// and `x ^ (x >> k)` both are, for any `k` — so the whole step is a
    /// permutation and only zero maps to zero. The sweep is the falsifiable
    /// half: a transposed shift width would still permute, but the run below
    /// would find the orbit shorter or the zero reachable.
    #[test]
    fn a_non_zero_xorshift32_state_never_reaches_zero_or_returns_early() {
        for state in [1_u32, 0x9e37_79b9, 0x243f_6a88, u32::MAX] {
            let mut rng = Xorshift32::from_state(state);
            for step in 0..200_000 {
                let draw = rng.next_u32();
                assert_ne!(draw, 0, "state {state:#x} reached zero at step {step}");
                assert_ne!(
                    draw, state,
                    "state {state:#x} returned to its start at step {step}, so the \
                     orbit is far shorter than the 2^32 - 1 the shift triple gives"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "a xorshift32 stream cannot start at zero")]
    fn a_zero_xorshift32_state_is_refused_by_name() {
        let _ = Xorshift32::from_state(0);
    }
}
