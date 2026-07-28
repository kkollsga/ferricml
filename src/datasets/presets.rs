//! The frozen quality lanes FerricML's conformance suite is recorded against.
//!
//! # Why a preset is a different thing from a recipe
//!
//! [`Recipe`] describes data anybody may want. The presets here describe data
//! that already exists: the design matrices and targets behind the frozen
//! reference outputs in `tests/fixtures/reference_semantics_v1.rs`. Their
//! parameters are not tuning knobs, and the expressions below are not open to
//! improvement — every one of them was transcribed character for character from
//! the private lane functions that used to live in `tests/reference_semantics.rs`,
//! because the recorded accuracy, Brier and error numbers are only meaningful
//! against the exact bytes they were measured on.
//!
//! That is a stronger constraint than it looks. The quality lanes compare
//! aggregate accuracy and Brier within `0.02`, so a generator emitting a
//! *different but similarly distributed* stream passes every one of them while
//! silently changing every design matrix. Distributional agreement cannot see
//! this port; pinned values can, and
//! `the_absorbed_lanes_reproduce_their_recorded_values` is where they are.
//!
//! # What is load-bearing in the transcription
//!
//! Three things, each of which a plausible cleanup would break:
//!
//! * **The noise terms widen to `f64` and narrow back to `f32` at the very
//!   end.** Computing them in `f32` throughout is a different number.
//! * **Association order.** Rust does not contract `a * b + c` to a fused
//!   multiply-add, so a transcription that preserves the parenthesisation is
//!   bit-exact and one that "simplifies" `0.7 * x * x` to `0.7 * (x * x)` is a
//!   fixture change wearing a refactor's clothes.
//! * **The row index restarts at zero for the test split.** The noise terms are
//!   functions of the row index, and the lanes they came from built the two
//!   splits as two separate matrices. A preset that indexed the test rows
//!   continuously from `768` would produce a different label vector from the
//!   same design.
//!
//! # One stream, two splits
//!
//! The lanes drew the training matrix and then the test matrix from *one*
//! generator, so the test half continues the stream the training half left off
//! at rather than restarting it. That is reproduced here by generating a single
//! `TRAIN_ROWS + TEST_ROWS` design and splitting it by row: the source advances
//! once per element in row-major order, so the concatenation and the two
//! successive draws are the same values in the same places.

use super::dataset::{Dataset, Target, Truth};
use super::recipe::{Recipe, Source};
use crate::data::{BinaryTargets, DenseMatrix, RegressionTargets};

/// Which frozen conformance lane a [`ReferenceQuality`] preset reproduces.
///
/// The four binary lanes threshold a real-valued score at zero and differ in
/// what makes them hard: a nonlinear boundary, a nearly separable one, a
/// deliberately rare positive class, and a boundary buried under noise. The
/// fifth draws a continuous target instead.
///
/// It is `#[non_exhaustive]` because absorbing another frozen lane must not be a
/// breaking change for a caller that matches only the ones it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceLane {
    /// A product term, a square, a linear term and an interaction, offset so the
    /// classes are close to balanced. No linear model reproduces this boundary.
    NonlinearBinary,
    /// A plain linear score over three features, and therefore the lane a linear
    /// classifier should do well on.
    SeparableBinary,
    /// A linear score with a large negative offset, so positives are rare —
    /// roughly one row in fourteen. This is the lane that catches a classifier
    /// scoring well by predicting the majority class.
    ImbalancedBinary,
    /// A weak linear signal plus a noise term larger than it, so the achievable
    /// accuracy is bounded well below one and a model reporting more than that
    /// is reporting leakage.
    NoisyBinary,
    /// A continuous target: linear, squared, interaction and noise terms.
    Regression,
}

/// One frozen conformance lane at one recorded seed.
///
/// A preset is fully described by its lane and its seed, and produces the same
/// bytes as the private lane functions FerricML's reference suite used before
/// this module absorbed them. The recorded seeds are `11`, `22`, `33`, `44` and
/// `55`; the type accepts any [`u64`] because nothing about the construction
/// depends on the seed being one of those, and refusing the others would make
/// the preset harder to explore without making the frozen ones safer.
///
/// # The seed is a raw stream state, not a derived one
///
/// [`Recipe::seeded`] mixes a caller's number into a stream disjoint from the
/// one an estimator seeded with the same number draws from, and that is the
/// right default for new work. It is the wrong thing here: these lanes were
/// recorded against the *raw* state, so the preset names it through
/// [`Source::Sampled`]. Routing it through the derivation would move every
/// frozen fixture at once, which is exactly the failure this module's pinned
/// values exist to catch.
///
/// ```
/// use ferricml::datasets::{ReferenceLane, ReferenceQuality, Source};
///
/// let preset = ReferenceQuality::new(ReferenceLane::SeparableBinary, 11);
/// assert_eq!(preset.recipe().source(), Source::Sampled { state: 11 });
///
/// let train = preset.train();
/// let test = preset.test();
/// assert_eq!(train.features().rows(), ReferenceQuality::TRAIN_ROWS);
/// assert_eq!(test.features().rows(), ReferenceQuality::TEST_ROWS);
///
/// // The test half continues the training half's stream rather than
/// // restarting it, so no row is shared between them.
/// assert_ne!(train.features().row(0), test.features().row(0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceQuality {
    lane: ReferenceLane,
    seed: u64,
}

impl ReferenceQuality {
    /// Rows in the training split of every frozen lane.
    pub const TRAIN_ROWS: usize = 768;

    /// Rows in the test split of every frozen lane.
    pub const TEST_ROWS: usize = 384;

    /// Columns in every frozen lane.
    ///
    /// Twelve rather than the six the widest lane reads, because the recorded
    /// numbers were measured on twelve — the unused columns are noise features
    /// a model has to decline to split on, and dropping them would both change
    /// the stream and make the lanes easier.
    pub const COLUMNS: usize = 12;

    /// Names a lane and a seed.
    #[inline]
    pub const fn new(lane: ReferenceLane, seed: u64) -> Self {
        Self { lane, seed }
    }

    /// Returns the lane this preset reproduces.
    #[inline]
    pub const fn lane(&self) -> ReferenceLane {
        self.lane
    }

    /// Returns the raw stream state the lane was recorded against.
    #[inline]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the single recipe both splits are drawn from.
    ///
    /// It is `TRAIN_ROWS + TEST_ROWS` rows tall, because the lanes drew both
    /// matrices from one generator and splitting one design by row is what
    /// reproduces that. Both [`ReferenceQuality::train`] and
    /// [`ReferenceQuality::test`] therefore report this recipe's digest: they
    /// are two halves of its output, and the digest identifies the recipe.
    ///
    /// # Panics
    ///
    /// Never. The shape is a compile-time constant and every `u64` is a valid
    /// [`Source::Sampled`] state, so the constructor's validation cannot fail
    /// for any preset that exists.
    pub fn recipe(&self) -> Recipe {
        Recipe::new(
            Self::TRAIN_ROWS + Self::TEST_ROWS,
            Self::COLUMNS,
            Source::Sampled { state: self.seed },
        )
        .expect("a preset's shape and source are fixed and valid")
    }

    /// Generates the training split.
    pub fn train(&self) -> Dataset {
        self.split(true)
    }

    /// Generates the test split.
    ///
    /// Generating one split regenerates the whole design and discards the other
    /// half. That is deliberate: the alternative is a second recipe naming the
    /// stream state the training half stops at, which no caller could write
    /// down without reaching into the generator's internals. The discarded work
    /// is one pass over `TEST_ROWS * COLUMNS` values against fitting a model on
    /// the half that is kept.
    pub fn test(&self) -> Dataset {
        self.split(false)
    }

    /// Generates one split, with its own row indices starting at zero.
    fn split(&self, training: bool) -> Dataset {
        let recipe = self.recipe();
        let mut values = Vec::new();
        recipe.design_into(&mut values);
        let boundary = Self::TRAIN_ROWS * Self::COLUMNS;
        let (values, rows) = if training {
            values.truncate(boundary);
            (values, Self::TRAIN_ROWS)
        } else {
            (values.split_off(boundary), Self::TEST_ROWS)
        };
        let design = DenseMatrix::new(values, rows, Self::COLUMNS)
            .expect("a preset's split is a whole number of generated rows");
        let target = self.target(&design);
        Dataset::from_parts(
            design,
            Some(target),
            None,
            Truth::Unrecorded,
            None,
            recipe.spec_digest(),
        )
    }

    /// Draws the lane's targets over one split's design.
    ///
    /// Each arm is the lane's own expression, unchanged. `index` is the row's
    /// position *within this split*, which is what the lanes passed and what the
    /// noise terms are functions of.
    ///
    /// The one spelling that differs is the integer arithmetic inside the noise
    /// terms: the lanes wrote `*` and `+`, which panic on overflow in a debug
    /// build. Neither can overflow at a row index below `1152`, and no lane
    /// reached one — but a `seed` near [`u64::MAX`] would, and a public
    /// constructor that panics for some of its inputs is not a contract this
    /// crate writes. `wrapping_mul` and `wrapping_add` are what a release build
    /// already did, so they give the same value everywhere the original was
    /// defined and a defined one everywhere it was not.
    fn target(&self, design: &DenseMatrix) -> Target {
        let seed = self.seed;
        match self.lane {
            ReferenceLane::NonlinearBinary => Target::Binary(thresholded(design, |_, row| {
                row[0] * row[1] + 0.7 * row[2] * row[2] - 0.45 * row[3] + 0.2 * row[4] * row[5]
                    - 0.15
            })),
            ReferenceLane::SeparableBinary => Target::Binary(thresholded(design, |_, row| {
                1.2 * row[0] - 0.9 * row[1] + 0.5 * row[2]
            })),
            ReferenceLane::ImbalancedBinary => Target::Binary(thresholded(design, |_, row| {
                1.3 * row[0] + 0.8 * row[1] - 0.35 * row[2] * row[2] - 1.25
            })),
            ReferenceLane::NoisyBinary => Target::Binary(thresholded(design, |index, row| {
                let noise = (((index.wrapping_mul(1_103_515_245).wrapping_add(seed)) & 0xffff)
                    as f64
                    / 32_768.0
                    - 1.0) as f32;
                0.25 * row[0] + noise
            })),
            ReferenceLane::Regression => {
                let targets = design
                    .iter_rows()
                    .enumerate()
                    .map(|(index, row)| {
                        let index = index as u64;
                        let noise = (((index
                            .wrapping_mul(214_013)
                            .wrapping_add(seed.wrapping_mul(2_531_011)))
                            & 0xffff) as f64
                            / 32_768.0
                            - 1.0) as f32;
                        1.7 * row[0] - 0.8 * row[1] * row[1]
                            + 0.6 * row[2] * row[3]
                            + 0.3 * row[4]
                            + 0.1 * noise
                    })
                    .collect();
                Target::Regression(
                    RegressionTargets::new(targets)
                        .expect("a bounded polynomial of finite values is finite"),
                )
            }
        }
    }
}

/// Labels every row by `score > 0.0`, which is the comparison the lanes wrote.
///
/// Strictly greater rather than `>=`, so a row scoring exactly zero is a
/// negative. That is a transcription rather than a preference: the two spellings
/// differ only on exact zeros, which is precisely the kind of difference a
/// prevalence check averages away.
fn thresholded(design: &DenseMatrix, score: impl Fn(u64, &[f32]) -> f32) -> BinaryTargets {
    let labels = design
        .iter_rows()
        .enumerate()
        .map(|(index, row)| u8::from(score(index as u64, row) > 0.0))
        .collect();
    BinaryTargets::new(labels).expect("a thresholded score is 0 or 1 by construction")
}
