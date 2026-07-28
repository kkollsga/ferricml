//! The deterministic streams a design matrix is drawn from, and the maps that
//! turn one draw into one design value.
//!
//! # Why the value maps live here and the generators do not
//!
//! `src/numeric/rng.rs` owns the generator cores and the seed derivations,
//! because `rng-single-source` in `scripts/check_source_layout.py` requires
//! exactly one definition of each and because the mixing function they share is
//! private to that file. This module owns the other half: the arithmetic that
//! turns a raw draw into a number in a design matrix.
//!
//! The split is not arbitrary. A generator's draw *count* is part of every
//! fitted estimator's determinism contract — that is why
//! `OwnedRng::unit_f64` is on the generator — whereas a
//! design value's construction is part of a *dataset's* contract, and is frozen
//! against dataset fixtures rather than against fitted models. Keeping the maps
//! beside those fixtures means a change to one is visible next to the literals
//! it would move.
//!
//! # Every map here is exact
//!
//! Nothing in this module rounds in a way that depends on the platform's libm,
//! and that is a requirement rather than an observation: the absorbed reference
//! and benchmark fixtures are compared with `assert_eq!` on `f32`, so a design
//! value has to be reproducible bit for bit on every target. Each map is an
//! integer-to-float conversion followed by a division and an affine step, all of
//! which IEEE-754 defines exactly or correctly-rounds. A future family drawing
//! Gaussian values needs `ln`, `sqrt` and `cos`, whose last bits are not
//! portable across libm implementations; such a family belongs beside its own
//! portability statement and not in this file.

use super::recipe::Source;
use crate::numeric::{OwnedRng, Xorshift32};

/// The largest lattice modulus whose residues are all exact in `f32`.
///
/// `2^24` is the last integer after which `f32` can no longer represent every
/// integer below it, so a larger modulus would map two distinct residues onto
/// one design value.
pub(super) const LATTICE_MODULUS_LIMIT: u64 = 1 << 24;

/// One SplitMix64 draw mapped onto `[-1, 1)`, from its top 24 bits.
///
/// Transcribed from the construction the frozen reference design matrices were
/// generated with (`tests/support/rng.rs`), and deliberately not simplified:
/// the numerator is below `2^24` and the divisor is a power of two, so the
/// quotient is exact, and the affine step cannot round either. The top bits are
/// taken rather than the bottom ones because SplitMix64's low bits are the
/// weakest part of its output.
///
/// Taking 24 bits rather than 53 and computing in `f32` throughout is what makes
/// this different from `OwnedRng::unit_f64`; the two are not
/// interchangeable and swapping one for the other would move every fixture that
/// depends on this one.
fn sampled_signed_unit(draw: u64) -> f32 {
    let fraction = (draw >> 40) as f32 / (1_u32 << 24) as f32;
    fraction * 2.0 - 1.0
}

/// One xorshift32 draw mapped onto `[-1, 1]`, in `f32` throughout.
///
/// Transcribed verbatim from the benchmark fixtures rather than improved. The
/// division is by `u32::MAX` rather than by `2^32`, so the map reaches `1.0`
/// exactly and is very slightly non-uniform — a property of the frozen stream,
/// not a defect to fix here, because `bench-history` compares against immutable
/// per-release results and a changed draw invalidates them all. Computing in
/// `f32` throughout is equally load-bearing: widening the intermediate to `f64`
/// changes rounding and therefore changes benchmark inputs.
fn xorshift_signed_unit(draw: u32) -> f32 {
    (draw as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// One lattice cell mapped onto `[-1, 1)`.
///
/// The divisor is half the modulus, so residue `0` maps to `-1` and the largest
/// residue lands just below `1`. Both the residue and the modulus are below
/// [`LATTICE_MODULUS_LIMIT`], so both conversions are exact; halving is exact
/// for every `f32`; and the quotient and the subtraction each round once.
fn lattice_signed_unit(cell: u64, divisor: f32) -> f32 {
    (cell as f32 / divisor) - 1.0
}

/// Writes `rows * columns` design values into `values`, in row-major order.
///
/// The buffer is cleared and refilled rather than appended to, so a caller
/// reusing one across recipes gets the recipe's output and not a concatenation.
/// Row-major fill order is part of the stream contract: a source drawing from a
/// generator advances it once per element, so transposing the loop would
/// permute every value in the matrix while leaving the multiset of draws
/// unchanged — a difference no distributional check can see.
pub(super) fn fill_design(source: &Source, rows: usize, columns: usize, values: &mut Vec<f32>) {
    let cells = rows * columns;
    values.clear();
    values.reserve(cells);
    match *source {
        Source::Sampled { state } => {
            let mut rng = OwnedRng::new(state);
            for _ in 0..cells {
                values.push(sampled_signed_unit(rng.next_u64()));
            }
        }
        Source::Lattice {
            row_stride,
            column_stride,
            modulus,
        } => {
            // Half the modulus, formed once. The modulus is below `2^24` and so
            // exact in `f32`, and halving an `f32` is exact, so this is the same
            // number a per-element `modulus as f32 / 2.0` would give.
            let divisor = modulus as f32 / 2.0;
            for row in 0..rows as u64 {
                let row_term = row.wrapping_mul(row_stride);
                for column in 0..columns as u64 {
                    // Reduced once, at the end. Reducing the two terms
                    // separately gives the same residue but is a different
                    // expression, and this one is what the absorbed benchmark
                    // fixture computes. The multiplications wrap rather than
                    // panicking on a release build's overflow, which keeps the
                    // lattice defined at every shape instead of at some of them.
                    let cell = row_term.wrapping_add(column.wrapping_mul(column_stride)) % modulus;
                    values.push(lattice_signed_unit(cell, divisor));
                }
            }
        }
        Source::Xorshift32 { state } => {
            let mut rng = Xorshift32::from_state(state);
            for _ in 0..cells {
                values.push(xorshift_signed_unit(rng.next_u32()));
            }
        }
    }
}
