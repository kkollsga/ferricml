//! The frozen fixtures FerricML's own benchmark suite is measured on.
//!
//! # Why these are transcriptions rather than recipes
//!
//! `bench-history` gates each release against immutable per-release results at a
//! `1.10` ratio limit, and a benchmark's result only means anything against the
//! data it was measured on. Change one design value and every historical
//! baseline becomes non-comparable — not wrong, *unusable*, because there is no
//! way to tell a regression from a different dataset after the fact. So the
//! expressions below are not open to improvement, and the digests in
//! `src/datasets/tests.rs` were captured from the private `fixture` functions in
//! `benches/forest.rs`, `benches/models.rs` and `benches/boosting.rs` before this
//! module replaced them, in the commit that deleted them.
//!
//! # What is load-bearing in the transcription
//!
//! * **`f32` throughout.** Every accumulation, every product, every conversion.
//!   Widening an intermediate to `f64` changes rounding and therefore changes
//!   what the estimators are timed on.
//! * **Association order.** Rust does not contract `a * b + c` into a fused
//!   multiply-add, so a parenthesisation-preserving copy is bit-exact and a
//!   "simplified" one is a fixture change wearing a refactor's clothes. The
//!   boosted target in particular is a left-associated chain of six `+` terms and
//!   is written as one.
//! * **The score columns are a prefix, and a short design keeps its own
//!   prefix.** The originals guarded their accumulator with `if column < k`
//!   inside the generation loop, so a design narrower than `k` summed only the
//!   columns it had. `take(k)` is that behaviour, and a `[0.0; k]` fill that
//!   assumed the columns existed would silently differ on a narrow shape.
//! * **The row index is the design's own.** Every noise term is a function of
//!   `row % 11` or `row % 17` over the generated matrix's own row numbering.
//!
//! # Why the lanes are named after benchmark suites
//!
//! [`BenchmarkLane::ModelsRegression`] is not a description of a task anybody
//! would choose; it is the target expression `benches/models.rs` was recorded
//! against. Naming it after the file it came from is what keeps the provenance
//! visible, because provenance is the *only* reason these particular constants
//! exist. A caller wanting a nonlinear regression problem chosen on its merits
//! wants a task family, not one of these.

use super::dataset::{Dataset, Target, Truth};
use super::error::DatasetError;
use super::recipe::{Recipe, Source};
use crate::data::{BinaryTargets, DenseMatrix, RegressionTargets};

/// The lattice `benches/forest.rs` draws its design from.
///
/// `(row * 131 + column * 17) % 1009`, mapped onto `[-1, 1)` by halving the
/// modulus. Two coprime strides against a prime modulus, which is what keeps the
/// columns from repeating at the widths the suite measures.
pub(super) const FOREST_LATTICE: Source = Source::Lattice {
    row_stride: 131,
    column_stride: 17,
    modulus: 1009,
};

/// Columns the forest fixture's separating score reads.
const FOREST_SCORE_COLUMNS: usize = 4;

/// Columns the `benches/models.rs` target expression reads.
const MODELS_SELECTED_COLUMNS: usize = 6;

/// Columns the `benches/boosting.rs` target expression is written over.
///
/// Twelve, though only the first nine are read: the original declared a
/// `[0.0_f32; 12]` buffer and filled every column below its length. The count is
/// kept because it is what the fixture did, and because a later term reaching
/// index nine or ten would then be reading a filled slot rather than a
/// coincidentally zero one.
const BOOSTING_SELECTED_COLUMNS: usize = 12;

/// Which benchmark fixture a [`BenchmarkFixture`] reproduces.
///
/// Each lane is one benchmark suite's design source together with the target
/// expression that suite drew over it. The two forest lanes share one design and
/// differ only in the target, exactly as `benches/forest.rs` did: its regressor
/// arms were derived from the classifier arms' labels so that both measure the
/// same matrix.
///
/// It is `#[non_exhaustive]` because absorbing another benchmark fixture must not
/// be a breaking change for a caller matching only the lanes it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BenchmarkLane {
    /// `benches/forest.rs`: a lattice design, labelled by whether a weighted sum
    /// of its first four columns is positive.
    ForestBinary,
    /// `benches/forest.rs`: the same design and the same labels, mapped to a
    /// continuous target so the regressor arms measure the classifier arms' data.
    ForestRegression,
    /// `benches/models.rs`: a xorshift32 design with a linear, interaction,
    /// threshold and row-index target.
    ModelsRegression,
    /// `benches/boosting.rs`: a wider xorshift32 design with a six-term target
    /// carrying two thresholds and two interactions.
    BoostingRegression,
}

impl BenchmarkLane {
    /// The design source this lane was recorded against.
    const fn source(self) -> Source {
        match self {
            Self::ForestBinary | Self::ForestRegression => FOREST_LATTICE,
            Self::ModelsRegression => Source::Xorshift32 { state: 0x9e37_79b9 },
            Self::BoostingRegression => Source::Xorshift32 { state: 0x243f_6a88 },
        }
    }
}

/// One benchmark fixture at one shape.
///
/// The shape is a parameter because the suites measure the same fixture at
/// several: `benches/forest.rs` fits at `2048x64` and round-trips artifacts at
/// `512x16`, and `benches/models.rs` reaches five shapes between `256x8` and
/// `1024x512`. Every source here fills row by row from a fixed start, so a
/// shorter design is a prefix of a longer one at the same column count — which is
/// why an inference lane can take the first `1024` rows of a training matrix and
/// still be measuring the same data.
///
/// ```
/// use ferricml::datasets::{BenchmarkFixture, BenchmarkLane, Target};
///
/// let fixture = BenchmarkFixture::new(BenchmarkLane::ForestBinary, 512, 16)?;
/// let dataset = fixture.generate();
///
/// assert_eq!(dataset.features().rows(), 512);
/// // The first row is the lattice origin, and residue zero maps to `-1`.
/// assert_eq!(dataset.features().get(0, 0), Some(-1.0));
/// assert!(matches!(dataset.target(), Some(Target::Binary(_))));
/// # Ok::<(), ferricml::datasets::DatasetError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkFixture {
    lane: BenchmarkLane,
    recipe: Recipe,
}

impl BenchmarkFixture {
    /// Validates a shape against a lane's source.
    ///
    /// The shape is checked by [`Recipe::new`], before anything is allocated, so
    /// a `BenchmarkFixture` that exists describes data that can be produced and
    /// [`BenchmarkFixture::generate`] has no failure left to report.
    ///
    /// ```
    /// use ferricml::datasets::{BenchmarkFixture, BenchmarkLane, DatasetError};
    ///
    /// assert_eq!(
    ///     BenchmarkFixture::new(BenchmarkLane::ModelsRegression, 0, 8),
    ///     Err(DatasetError::ZeroRows),
    /// );
    /// # Ok::<(), DatasetError>(())
    /// ```
    pub fn new(lane: BenchmarkLane, rows: usize, columns: usize) -> Result<Self, DatasetError> {
        Ok(Self {
            lane,
            recipe: Recipe::new(rows, columns, lane.source())?,
        })
    }

    /// Returns the lane this fixture reproduces.
    #[inline]
    pub const fn lane(&self) -> BenchmarkLane {
        self.lane
    }

    /// Returns the recipe the design matrix is generated from.
    #[inline]
    pub const fn recipe(&self) -> Recipe {
        self.recipe
    }

    /// Generates the fixture.
    ///
    /// The result carries [`Truth::Unrecorded`]: a task *was* drawn, but these
    /// expressions were written to give the estimators something to fit rather
    /// than to be right about anything, and no coefficient vector or noise-free
    /// target was ever kept. Claiming one now would be a claim this module cannot
    /// support.
    pub fn generate(&self) -> Dataset {
        let design = self.recipe.design();
        let target = self.target(&design);
        Dataset::from_parts(
            design,
            Some(target),
            None,
            Truth::Unrecorded,
            None,
            None,
            self.recipe.spec_digest(),
        )
    }

    /// Draws the lane's target over a generated design.
    fn target(&self, design: &DenseMatrix) -> Target {
        match self.lane {
            BenchmarkLane::ForestBinary => Target::Binary(forest_labels(design)),
            BenchmarkLane::ForestRegression => {
                Target::Regression(forest_regression(&forest_labels(design)))
            }
            BenchmarkLane::ModelsRegression => Target::Regression(models_regression(design)),
            BenchmarkLane::BoostingRegression => Target::Regression(boosting_regression(design)),
        }
    }
}

/// `benches/forest.rs`'s separable labels: is a weighted sum of the first four
/// columns positive?
///
/// The weights ascend with the column index and the accumulation runs left to
/// right, both transcribed. Strictly greater than zero, so a row scoring exactly
/// zero is a negative — a transcription rather than a preference, and exactly the
/// kind of difference a prevalence check averages away, which is why the pinned
/// digests rather than the pinned positive counts are what settle it.
fn forest_labels(design: &DenseMatrix) -> BinaryTargets {
    let labels = design
        .iter_rows()
        .map(|row| {
            let mut score = 0.0_f32;
            for (column, &value) in row.iter().take(FOREST_SCORE_COLUMNS).enumerate() {
                score += value * (column + 1) as f32;
            }
            u8::from(score > 0.0)
        })
        .collect();
    BinaryTargets::new(labels).expect("a thresholded score is 0 or 1 by construction")
}

/// `benches/forest.rs`'s regression targets, derived from its own labels.
///
/// Four units of class separation plus a sawtooth in the row index. The sawtooth
/// is what stops the target being a two-valued step function, which a regressor
/// would fit with one split.
fn forest_regression(labels: &BinaryTargets) -> RegressionTargets {
    let targets = labels
        .as_slice()
        .iter()
        .enumerate()
        .map(|(row, &label)| f32::from(label) * 4.0 + (row % 11) as f32)
        .collect();
    RegressionTargets::new(targets).expect("a bounded sum of finite values is finite")
}

/// `benches/models.rs`'s target: two linear terms, one interaction, one
/// threshold and a row-index sawtooth.
fn models_regression(design: &DenseMatrix) -> RegressionTargets {
    let targets = design
        .iter_rows()
        .enumerate()
        .map(|(row, values)| {
            let selected = selected_prefix::<MODELS_SELECTED_COLUMNS>(values);
            let nonlinear = selected[2] * selected[3]
                + if selected[4] > 0.0 { 0.8 } else { -0.8 }
                + 0.25 * ((row % 11) as f32 - 5.0);
            1.7 * selected[0] - 0.9 * selected[1] + nonlinear
        })
        .collect();
    RegressionTargets::new(targets).expect("a bounded polynomial of finite values is finite")
}

/// `benches/boosting.rs`'s target: six terms, two of them thresholds.
///
/// The two thresholds are what make this a boosting fixture rather than a linear
/// one — a piecewise-constant boundary is what a tree ensemble can represent and
/// a linear model cannot.
fn boosting_regression(design: &DenseMatrix) -> RegressionTargets {
    let targets = design
        .iter_rows()
        .enumerate()
        .map(|(row, values)| {
            let selected = selected_prefix::<BOOSTING_SELECTED_COLUMNS>(values);
            2.0 * selected[0] - selected[1]
                + 1.5 * selected[2] * selected[3]
                + if selected[4] > 0.0 { 1.2 } else { -1.2 }
                + if selected[5] + selected[6] > 0.25 {
                    0.9
                } else {
                    -0.4
                }
                + 0.3 * selected[7] * selected[8]
                + 0.15 * ((row % 17) as f32 - 8.0)
        })
        .collect();
    RegressionTargets::new(targets).expect("a bounded polynomial of finite values is finite")
}

/// A row's first `N` values, zero-filled where the design is narrower.
///
/// The originals declared a fixed-size buffer and filled it under
/// `if column < buffer.len()`, so a design with fewer than `N` columns left the
/// remaining slots at zero and the target expression read those zeros. That is
/// reproduced rather than refused: the fixture shapes in use are all wide enough,
/// but a narrower one must still generate the values the original would have,
/// because a panic here would turn a frozen transcription into a partial
/// function.
fn selected_prefix<const N: usize>(row: &[f32]) -> [f32; N] {
    let mut selected = [0.0_f32; N];
    for (slot, &value) in selected.iter_mut().zip(row) {
        *slot = value;
    }
    selected
}
