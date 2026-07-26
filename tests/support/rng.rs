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
#![allow(dead_code)]

/// SplitMix64. Small, seekable, and adequate for generating test inputs.
pub struct TestRng {
    state: u64,
}

impl TestRng {
    /// Starts a stream from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x243f_6a88_85a3_08d3,
        }
    }

    /// Next raw draw.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
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
