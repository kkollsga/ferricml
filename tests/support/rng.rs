//! A deterministic test-only generator for randomized property sweeps.
//!
//! The crate's own `OwnedRng` is private, and an experiment that drew from it
//! would be sampling with the same stream the code under test uses. This is a
//! separate SplitMix64 written here so a sweep's inputs are independent of
//! anything the library does with a seed, and so a recorded result can be
//! reproduced from the seed alone.
//!
//! Included both through `support/mod.rs` and directly by `#[path]` from the
//! randomized-sweep binaries, which use different parts of it.
//!
//! # One generator for the test crates, as `src/numeric/rng.rs` is for the crate
//!
//! `rng-single-source` in `scripts/check_source_layout.py` used to stop at
//! `src/`, and its stated reason was that an integration crate cannot see a
//! `pub(crate)` generator. That was true and it was not a licence for
//! duplication: `tests/reference_semantics.rs` and `tests/artifact_hardening.rs`
//! each carried a private SplitMix64 whose core was character-identical to this
//! one and to the crate's — three copies of one stream, which is the exact shape
//! rule 6 in `src/numeric/mod.rs` exists to forbid, one directory over. The rule
//! now covers `tests/` too, with this file as the single permitted source.
//!
//! Two constructors exist because the two callers need different things, and
//! collapsing them would have moved fixtures:
//!
//! * [`TestRng::new`] perturbs the seed, so a sweep's stream is *unrelated* to
//!   whatever the library does with the same number. That is what a randomized
//!   experiment wants.
//! * [`TestRng::from_state`] takes the state raw, which is the crate generator's
//!   own spelling and therefore the same stream for the same seed. Callers whose
//!   recorded outputs are frozen against an existing stream need this one, and
//!   they pin the stream they depend on with literals beside their fixtures.
#![allow(dead_code)]

/// SplitMix64. Small, seekable, and adequate for generating test inputs.
pub struct TestRng {
    state: u64,
}

impl TestRng {
    /// Starts a stream from `seed`, perturbed away from the library's own.
    ///
    /// The seed is mixed with a constant so that a sweep drawing from `seed`
    /// and an estimator seeded with `seed` do not walk the same sequence. Use
    /// [`TestRng::from_state`] where the stream itself is what is recorded.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x243f_6a88_85a3_08d3,
        }
    }

    /// Starts a stream from `state` exactly, with no perturbation.
    ///
    /// This is `src/numeric/rng.rs`'s `OwnedRng::new`, spelled for the test
    /// crates: the same seed yields the same `next_u64` sequence. It exists for
    /// the callers whose *outputs* are frozen — the reference-parity fixtures are
    /// recorded against data generated from this stream, and the artifact fuzz
    /// campaign's reach floors are calibrated against it — so the stream is a
    /// contract to those files rather than an implementation detail, and each of
    /// them pins it with literals of its own.
    pub const fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Next raw draw.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    /// A uniform draw in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// A uniform draw in `[low, high)`.
    pub fn range(&mut self, low: f64, high: f64) -> f64 {
        low + self.unit() * (high - low)
    }

    /// A uniform `f32` draw in `[low, high)`.
    pub fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        self.range(f64::from(low), f64::from(high)) as f32
    }

    /// A uniform `f32` draw in `[-1, 1)` from the top 24 bits of one draw.
    ///
    /// Deliberately *not* `range_f32(-1.0, 1.0)`: this takes 24 bits and
    /// computes in `f32` throughout, which is the construction the frozen
    /// reference-parity design matrices were generated with. Every operation is
    /// exact — the numerator is below `2^24`, the divisor is a power of two, and
    /// the affine map cannot round — so the value is reproducible bit for bit,
    /// which is the only reason a fixture may depend on it.
    pub fn signed_unit(&mut self) -> f32 {
        let fraction = (self.next_u64() >> 40) as f32 / (1_u32 << 24) as f32;
        fraction * 2.0 - 1.0
    }

    /// A uniform index below `upper`, which must be positive.
    pub fn below(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "an index bound must be positive");
        let bound = upper as u64;
        let reject_below = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                return (value % bound) as usize;
            }
        }
    }

    /// An inclusive integer draw in `[low, high]`.
    pub fn between(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low + 1)
    }

    /// A fair coin.
    pub fn flag(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Fisher-Yates, so a caller can permute an input list.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = self.below(index + 1);
            values.swap(index, other);
        }
    }
}

/// SplitMix64's mixing function, split out for the same reason
/// `src/numeric/rng.rs` splits it out: the constants are the stream, and one
/// place to read them is one place to change them.
///
/// `mix64` and the golden-ratio increment above are the two markers
/// `test-rng-single-source` in `scripts/check_source_layout.py` looks for, so a
/// second copy of this generator anywhere under `tests/` fails the gate.
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
