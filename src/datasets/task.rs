//! The [`Task`] vocabulary, the regression and binary-classification families,
//! and the ground truth each of them records.
//!
//! The structural families — many classes, clusters, time order, ranked pairs —
//! are named by [`Task`] here and drawn in `structural.rs`, because each of them
//! produces a second array that has to agree with the design row for row and
//! that would otherwise be buried in a match arm beside a scalar target.
//!
//! # Why these live apart from `source.rs`
//!
//! `source.rs` and the absorbed presets are transcendental-free by requirement:
//! their output is compared against frozen fixtures with `assert_eq!` on `f32`,
//! so every value has to be reproducible bit for bit on every target. Most of
//! the families here cannot be. A Bayes probability is a logistic function, a
//! log-link mean is an exponential, and a requested condition number is a real
//! power — `exp`, `ln` and `powf` are correctly rounded on no libm anyone ships,
//! so their last bits move between platforms and between libm versions.
//!
//! That is decision D3 of the generator plan, and the resolution is that the
//! portability envelope is a property of a **family**, never of the kernel. Each
//! family declares its own through [`Task::portability`], the declaration is
//! part of the public API, and the two envelopes are held to different evidence:
//!
//! * A [`Portability::BitExact`] family is pinned by literal values in
//!   `src/datasets/family_tests.rs`, the same way the absorbed lanes are.
//! * A [`Portability::PerRunner`] family is held to properties and tolerances —
//!   a recovered coefficient, a realized prevalence, a realized condition
//!   number — because a pinned literal would be a promise this crate cannot
//!   keep across a libm change it does not control.
//!
//! Nothing here is reachable from `source.rs`, `presets.rs` or `benchmarks.rs`,
//! so no frozen fixture can acquire a transcendental dependency by accident.
//!
//! # Where a family's auxiliary randomness comes from
//!
//! A design matrix consumes exactly `rows * columns` draws from the recipe's
//! source, and a family needs more: coefficients, noise, label draws, selectors
//! for the contamination knobs. Continuing the design's stream would make the
//! coefficients a function of the design's shape, so that widening a matrix by
//! one column would move every coefficient.
//!
//! Instead each auxiliary stream is seeded from one 64-bit word of a digest over
//! the recipe's shape, source and task — see [`stream`]. Four disjoint words
//! give four streams that cannot interleave, and two recipes differing anywhere
//! in that encoding draw different values.
//!
//! **The contamination and the task's own dials are deliberately outside that
//! digest.** They are overlays, not reseeds: switching a knob has to change
//! exactly what the knob describes and leave every other draw where it was.
//! Every selector is drawn unconditionally, whether or not the knob that reads
//! it is set, so the streams stay aligned across contamination levels — and a
//! difficulty ladder over `separation`, `noise_scale` or `prevalence` walks one
//! problem rather than a sequence of unrelated ones. `Recipe::stream_digest`
//! records what the two versions of this that got it wrong actually did, and
//! [`Task`]'s own documentation lists which fields are which.

use super::contamination::Contamination;
use super::dataset::{Target, Truth};
use super::error::{DatasetError, Parameter};
use super::source::sampled_unit;
use super::structural::{
    ClassBalance, ClassGeometry, draw_clustered, draw_multiclass, draw_ranking, drifting_predictor,
    shape_clustered, validate_clustered, validate_multiclass, validate_ranking,
};
use crate::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use crate::numeric::{OwnedRng, sigmoid_f64, sum_in_order};
use crate::ranking::PairwiseObservation;

/// Stream word carrying a family's true coefficients.
pub(super) const STREAM_COEFFICIENTS: usize = 0;
/// Stream word carrying a regression family's additive noise.
pub(super) const STREAM_NOISE: usize = 1;
/// Stream word carrying label draws and label-noise flips.
pub(super) const STREAM_LABELS: usize = 2;
/// Stream word carrying the contamination selectors.
pub(super) const STREAM_CONTAMINATION: usize = 3;

/// The linear predictor is clamped here before a link is applied.
///
/// `exp(4)` is about `54.6`, which keeps a Poisson rate small enough that
/// Knuth's draw terminates in tens of iterations rather than hundreds, and keeps
/// a positive response inside a range where an `f32` still resolves its own
/// noise. Without a clamp a wide design with a large `coefficient_scale` reaches
/// `exp(700)` and the response is an infinity that `RegressionTargets` would
/// then refuse — turning a caller's parameter choice into a panic deep inside
/// generation instead of a bounded, documented saturation.
const LINK_ARGUMENT_LIMIT: f64 = 4.0;

/// How far an outlier displaces a regression target, in units of the target's
/// own mean absolute value.
///
/// Eight, so an outlier is unambiguously one — far outside anything the noise
/// reaches at a sane `noise_scale`, and still finite and scale-free, so the
/// knob means the same thing on a target whose values are near `0.01` and on
/// one whose values are near `10_000`.
const OUTLIER_DISPLACEMENT: f32 = 8.0;

/// Whether a family's output is reproducible bit for bit everywhere, or only on
/// one runner.
///
/// This is the crate's D3 determinism contract made into a value a caller can
/// read, rather than a paragraph a caller has to find. A harness comparing
/// FerricML against another library across two machines needs to know which of
/// the two statements it is entitled to: that the bytes are identical, or that
/// they are identical *here*.
///
/// It is `#[non_exhaustive]` because a later family may carry a third envelope —
/// a family reading a system entropy source, say — and a caller matching the two
/// that exist must not break when it arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Portability {
    /// Identical bytes on every target, every libm, and every rerun.
    ///
    /// Every operation on the path is an integer operation, an exact conversion,
    /// or an IEEE-754 arithmetic operation, all of which are correctly rounded
    /// and therefore fixed by the standard rather than by an implementation.
    BitExact,
    /// Identical bytes on one runner, and to within a tolerance elsewhere.
    ///
    /// The path evaluates at least one transcendental function — `exp`, `ln`,
    /// `sin`, `cos` or `powf` — whose last bits are an implementation choice.
    /// The generator plan's exchange format is what makes such a dataset
    /// portable: materialize it once, digest it, and ship the bytes.
    PerRunner,
}

impl Portability {
    /// The weaker of two envelopes.
    pub(super) const fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::BitExact, Self::BitExact) => Self::BitExact,
            _ => Self::PerRunner,
        }
    }
}

/// Which kind of problem a [`Task`] describes, with its parameters removed.
///
/// A `Task` is a family *and* a parameterisation, so a linear regression at two
/// noise levels is two values. A catalogue that has to span the generator asks
/// the coarser question — *which kinds of problem exist at all* — and this is
/// the answer to it. [`Task::family`] is the projection, and
/// [`AccuracySuite`](super::AccuracySuite) and
/// [`PerformanceGrid`](super::PerformanceGrid) are the two catalogues held to
/// covering every value of this type.
///
/// # The roster grows with the enum, and the compiler is what says so
///
/// A taxonomy nobody can enumerate is a taxonomy the suites can silently stop
/// spanning, so [`Family::ALL`] exists — and the interesting question is what
/// stops it going stale. Rust cannot enumerate an enum's variants, so the roster
/// is data; what keeps it honest is that four separate things fail before a
/// missing suite entry can reach a reader:
///
/// 1. A new [`Task`] variant does not compile until [`Task::family`]'s
///    exhaustive match names its family.
/// 2. A new `Family` variant does not compile until the crate-internal
///    declaration-order walk behind [`Family::COUNT`] places it.
/// 3. Placing it changes `COUNT`, which is the declared length of
///    [`Family::ALL`], so the roster literal stops matching its own type.
/// 4. Only then does anything reach a test, and `suite_tests.rs`'s
///    `every_family_has_an_accuracy_case` is what fails: the suites are
///    hand-written tables rather than a map over the roster, precisely so that
///    forgetting one is a red test rather than an unreachable branch.
///
/// The one gap left is a family declared to follow nothing while another family
/// already does — an actively wrong total order rather than an omission. It is
/// written down here rather than papered over.
///
/// It is `#[non_exhaustive]` because a new family must not be a breaking change
/// for a caller matching only the ones it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Family {
    /// [`Task::LinearRegression`].
    LinearRegression,
    /// [`Task::NonlinearRegression`].
    NonlinearRegression,
    /// [`Task::GlmRegression`].
    GlmRegression,
    /// [`Task::IllConditioned`].
    IllConditioned,
    /// [`Task::LinearBinary`].
    LinearBinary,
    /// [`Task::NonlinearBinary`].
    NonlinearBinary,
    /// [`Task::Multiclass`].
    Multiclass,
    /// [`Task::Clustered`].
    Clustered,
    /// [`Task::TimeOrdered`].
    TimeOrdered,
    /// [`Task::Ranking`].
    Ranking,
}

impl Family {
    /// How many families there are.
    ///
    /// Counted at compile time by walking the crate-internal `Family::next` from
    /// the first family, rather than written down: a new family placed in that
    /// order moves this number, and this number is the declared length of
    /// [`Family::ALL`]. A roster that did not grow with the enum would therefore
    /// fail to compile rather than fail to be noticed.
    pub const COUNT: usize = {
        let mut count = 1;
        let mut family = Self::LinearRegression;
        while let Some(next) = family.next() {
            family = next;
            count += 1;
        }
        count
    };

    /// Every family, in declaration order.
    ///
    /// ```
    /// use ferricml::datasets::Family;
    ///
    /// assert_eq!(Family::ALL.len(), Family::COUNT);
    /// assert_eq!(Family::ALL[0], Family::LinearRegression);
    /// ```
    pub const ALL: [Self; Self::COUNT] = [
        Self::LinearRegression,
        Self::NonlinearRegression,
        Self::GlmRegression,
        Self::IllConditioned,
        Self::LinearBinary,
        Self::NonlinearBinary,
        Self::Multiclass,
        Self::Clustered,
        Self::TimeOrdered,
        Self::Ranking,
    ];

    /// The family after this one in declaration order, or `None` at the end.
    ///
    /// Crate-internal, because it is machinery rather than vocabulary: its only
    /// callers are [`Family::COUNT`] and the test that asserts [`Family::ALL`]
    /// *is* this order. Exhaustive, so a new family cannot compile until it is
    /// placed.
    pub(super) const fn next(self) -> Option<Self> {
        match self {
            Self::LinearRegression => Some(Self::NonlinearRegression),
            Self::NonlinearRegression => Some(Self::GlmRegression),
            Self::GlmRegression => Some(Self::IllConditioned),
            Self::IllConditioned => Some(Self::LinearBinary),
            Self::LinearBinary => Some(Self::NonlinearBinary),
            Self::NonlinearBinary => Some(Self::Multiclass),
            Self::Multiclass => Some(Self::Clustered),
            Self::Clustered => Some(Self::TimeOrdered),
            Self::TimeOrdered => Some(Self::Ranking),
            Self::Ranking => None,
        }
    }

    /// A stable, lower-case, hyphenated name for this family.
    ///
    /// Stable in the sense that matters for measurement: it is the identity a
    /// recorded benchmark row or an accuracy report is filed under, so renaming
    /// one silently orphans every historical record that named it. It is not
    /// derived from the variant's spelling, so a Rust-level rename and a
    /// record-level rename are two decisions rather than one.
    ///
    /// ```
    /// use ferricml::datasets::Family;
    ///
    /// assert_eq!(Family::IllConditioned.label(), "ill-conditioned");
    /// ```
    pub const fn label(self) -> &'static str {
        match self {
            Self::LinearRegression => "linear-regression",
            Self::NonlinearRegression => "nonlinear-regression",
            Self::GlmRegression => "glm-regression",
            Self::IllConditioned => "ill-conditioned",
            Self::LinearBinary => "linear-binary",
            Self::NonlinearBinary => "nonlinear-binary",
            Self::Multiclass => "multiclass",
            Self::Clustered => "clustered",
            Self::TimeOrdered => "time-ordered",
            Self::Ranking => "ranking",
        }
    }
}

/// Which nonlinear shape a regression target takes.
///
/// Two of these are transcendental-free and two are not, which is exactly why
/// [`Task::portability`] reads the kind rather than the variant: a caller who
/// needs bit-exact data can have a nonlinear problem, just not a sinusoidal one.
///
/// It is `#[non_exhaustive]` because a new shape must not be a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonlinearKind {
    /// A product, a square and a linear term over the first four columns. No
    /// linear model reproduces it; a depth-two tree nearly does.
    Interaction,
    /// Two thresholds and a linear term over the first four columns — the shape
    /// a tree ensemble represents exactly and a linear model cannot represent at
    /// all.
    Piecewise,
    /// A sum of a sine and a cosine over the first three columns, so the target
    /// oscillates rather than merely bending.
    Sinusoid,
    /// Friedman's first benchmark function over the first five columns, with the
    /// design mapped from `[-1, 1)` onto the `[0, 1)` the function is defined
    /// on.
    Friedman,
}

impl NonlinearKind {
    /// Columns the shape reads.
    const fn columns_read(self) -> usize {
        match self {
            Self::Interaction | Self::Piecewise => 4,
            Self::Sinusoid => 3,
            Self::Friedman => 5,
        }
    }

    /// This shape's determinism envelope.
    const fn portability(self) -> Portability {
        match self {
            Self::Interaction | Self::Piecewise => Portability::BitExact,
            Self::Sinusoid | Self::Friedman => Portability::PerRunner,
        }
    }

    /// The noise-free target at one row.
    fn conditional_mean(self, row: &[f32]) -> f32 {
        match self {
            Self::Interaction => 2.0 * row[0] * row[1] + 1.5 * row[2] * row[2] - row[3],
            Self::Piecewise => {
                (if row[0] > 0.0 { 1.5 } else { -1.5 })
                    + (if row[1] + row[2] > 0.25 { 0.8 } else { -0.4 })
                    + 0.5 * row[3]
            }
            Self::Sinusoid => {
                (3.0 * f64::from(row[0])).sin() as f32
                    + 0.5 * (2.0 * f64::from(row[1])).cos() as f32
                    + 0.3 * row[2]
            }
            Self::Friedman => {
                // The design lives on `[-1, 1)` and Friedman's function is
                // defined on `[0, 1)`, so each input is mapped rather than the
                // function rewritten: rewriting it would give a differently
                // shaped surface under the same name.
                let unit = |value: f32| f64::from(value) * 0.5 + 0.5;
                let (u0, u1, u2, u3, u4) = (
                    unit(row[0]),
                    unit(row[1]),
                    unit(row[2]),
                    unit(row[3]),
                    unit(row[4]),
                );
                (10.0 * (std::f64::consts::PI * u0 * u1).sin()
                    + 20.0 * (u2 - 0.5) * (u2 - 0.5)
                    + 10.0 * u3
                    + 5.0 * u4) as f32
            }
        }
    }
}

/// Which nonlinear boundary a binary family draws its labels across.
///
/// It is `#[non_exhaustive]` because a new boundary must not be a breaking
/// change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryKind {
    /// The exclusive-or boundary: the sign of the first two columns' product.
    /// A linear classifier's best achievable accuracy is the majority rate.
    Xor,
    /// A sine boundary: the second column against a full-period sine of the
    /// first.
    ///
    /// The boundary is `x₂ = sin(2π x₁)`, one whole period across the design's
    /// `[-1, 1)` support, at the full amplitude of that support. Both properties
    /// are load-bearing. A sine of *fractional* period is nearly its own tangent
    /// line, and a boundary of amplitude much below one leaves a rule on `x₂`
    /// alone almost nothing to get wrong; the family's own shortfall instrument
    /// measured an earlier `x₂ = 0.6 sin(2 x₁)` boundary as linearly solvable to
    /// within half a point of its Bayes ceiling.
    Sinusoid,
    /// A circular boundary: inside or outside a disc centred at the origin.
    Circles,
    /// A four-cell checkerboard over the first two columns, so the boundary
    /// repeats rather than merely curving.
    Checkerboard,
}

impl BinaryKind {
    /// This boundary's determinism envelope, before the logistic link.
    ///
    /// `cfg(test)` because nothing in the shipped path needs it: the family's
    /// envelope is [`Task::portability`], which is `PerRunner` for every
    /// boundary because the link is. What this exists for is the assertion that
    /// the two differ — three of the four boundaries are exact arithmetic and
    /// still belong to a per-runner family, which is what "the envelope is a
    /// property of the family, not of its parts" means when it is checked
    /// rather than said.
    #[cfg(test)]
    pub(super) const fn boundary_portability(self) -> Portability {
        match self {
            Self::Xor | Self::Circles | Self::Checkerboard => Portability::BitExact,
            Self::Sinusoid => Portability::PerRunner,
        }
    }

    /// The raw score at one row, before separation and the prevalence offset.
    fn score(self, row: &[f32]) -> f64 {
        let (first, second) = (f64::from(row[0]), f64::from(row[1]));
        match self {
            Self::Xor => 4.0 * first * second,
            // The leading `2.0` is the same device as `Xor`'s `4.0` below: the
            // boundary's own expression carries the scale that makes its scores
            // comparable with the other three, so `separation` stays a dial the
            // caller turns rather than a correction the caller has to apply.
            // Without it the best linear rule falls only `0.15` short of Bayes
            // at the suite's `separation`, straddling the threshold the family
            // test holds curved boundaries to.
            Self::Sinusoid => 2.0 * (second - (2.0 * std::f64::consts::PI * first).sin()),
            Self::Circles => 0.5 - (first * first + second * second),
            Self::Checkerboard => {
                // `floor` is exact on every target — it is an IEEE-754
                // operation, not a libm approximation — so the parity below
                // carries no portability cost of its own.
                let cell = (2.0 * first).floor() + (2.0 * second).floor();
                if (cell as i64).rem_euclid(2) == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }
}

/// How a generalized linear family maps its linear predictor to a response.
///
/// It is `#[non_exhaustive]` because a new link must not be a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GlmLink {
    /// `μ = exp(η)`, and the response is a Poisson count.
    ///
    /// `dispersion` scales the drawn rate and rescales the draw, so the mean
    /// stays `μ` and the variance becomes `μ * dispersion`: at `1.0` this is an
    /// ordinary Poisson count, and above it an over-dispersed one whose values
    /// are multiples of the dispersion.
    ///
    /// **At least `1.0`, because a Poisson cannot be under-dispersed.** Below it
    /// the request names no distribution this family offers, and it would also
    /// be unbounded work: the draw is Knuth's, which costs one uniform per unit
    /// of rate, and the rate is `μ / dispersion`.
    LogCount,
    /// `μ = exp(η)`, and the response is a strictly positive continuous value.
    ///
    /// `dispersion` is the multiplicative noise half-width: the response is
    /// `μ * (1 + dispersion * u)` for `u` uniform on `[-1, 1)`, which stays
    /// positive exactly while `dispersion` is below `1`.
    LogPositive,
}

/// A task family drawn over a recipe's design.
///
/// A task is a request, not a promise: its parameters are checked by
/// [`Recipe::with_task`](super::Recipe::with_task), against the design's shape,
/// before anything is allocated. A `Recipe` carrying a task therefore generates
/// without failing.
///
/// Every variant records what it actually knows through [`Truth`], and the
/// variants differ in what that is. A linear family knows its coefficients; a
/// nonlinear one knows only its conditional mean, because no coefficient vector
/// produces it; a binary family knows the Bayes probability behind each label.
/// None of them reports [`Truth::Unrecorded`] — that is reserved for the
/// absorbed lanes, whose correct answer was never kept.
///
/// # A dial moves the difficulty; a structural field moves the problem
///
/// The fields here are not all the same kind of thing, and the difference is
/// what a sweep over one of them is entitled to conclude.
///
/// `separation`, `prevalence`, `noise_scale`, `drift`, `spread`,
/// `coefficient_scale`, `intercept`, `condition_number`, `dispersion`,
/// `balance`, `informative` and `rank` are **dials**. Two recipes differing only
/// in a dial draw from the same streams: the same design, the same coefficients
/// or centres, the same noise and label draws. A ladder over one of them is a
/// ladder over one problem, so the difference between two rungs is the knob.
///
/// `classes`, `blobs`, `queries`, `docs_per_query`, `grades`, `geometry`, `kind`
/// and `link` are **structural**. They change what the problem is — how many
/// classes exist, which expression is evaluated — and two recipes differing in
/// one of them are two different draws, deliberately.
///
/// `informative` and `rank` are the two counts on the dial side, and they earn
/// it in different ways. Widening `informative` **nests**: the coefficient
/// draw consumes one value per column whether that column is informative or
/// not, so at `informative = 4` the first two coefficients are bit-identical to
/// the ones at `informative = 2` and two more become non-zero. A ladder over it
/// really is one problem gaining informative columns, rather than two unrelated
/// coefficient vectors. `rank` never reaches a draw at all — the columns past it
/// are exact copies of the leading ones, a closed-form transform of a design the
/// source already produced, which is the same argument that makes
/// `condition_number` a dial.
///
/// Both move [`Recipe::spec_digest`](super::Recipe::spec_digest), because the
/// data moves either way. `Recipe::stream_digest`'s documentation records why
/// the partition exists and what a version without it measured.
///
/// It is `#[non_exhaustive]` because a new family must not be a breaking change
/// for a caller that only ever matches the ones it asked for.
///
/// # The variant fields are deliberately literal-constructible
///
/// The variants below carry no `#[non_exhaustive]` of their own, unlike
/// [`Truth`]'s, and that asymmetry is a recorded decision rather than an
/// oversight. `Truth` is an output: nothing constructs one, so protecting it
/// costs a `..` in a pattern that was optional anyway. A `Task` is a *request*,
/// and the property this API leans on is that the request is complete: the
/// compiler refuses a recipe that fails to state a knob, so `separation` and
/// `prevalence` cannot be transposed, and a family's four positional `usize`
/// fields cannot be permuted, in a way that still compiles. Constructors would
/// keep completeness and lose the names; builders would keep the names and lose
/// completeness, because no dial here has a neutral default — a `separation` of
/// zero is a coin, not an absence.
///
/// The cost of that choice is stated rather than hidden: **adding a knob to an
/// existing family is a breaking change**, taken deliberately as a minor
/// version, and `make semver-check` fails it offered as anything less. The
/// architecture keeps that rare by channelling growth elsewhere. A new shape is
/// a new variant, kind or link; a cross-cutting knob is a
/// [`Contamination`] setting, on an already-opaque
/// builder; and any future knob has to default to reproducing today's bytes
/// regardless, because [`Recipe::spec_digest`](super::Recipe::spec_digest) is an
/// identity. The two family-design questions open when this was written both
/// resolved without a field: the sine boundary was redesigned as an expression,
/// and `informative` and `rank` were reclassified as dials — a digest-routing
/// change, not a new parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Task {
    /// A regression target that is a known linear function of the design plus
    /// uniform noise.
    ///
    /// This is the family that makes "which library is closer to right"
    /// measurable: [`Truth::LinearPredictor`] carries the exact `β` the target
    /// was drawn from, so a fitted coefficient vector can be compared against
    /// the answer rather than against another implementation's answer.
    LinearRegression {
        /// Leading columns carrying a non-zero coefficient. The rest are noise
        /// features a model has to decline to use.
        informative: usize,
        /// Magnitude of the drawn coefficients.
        coefficient_scale: f32,
        /// The intercept added to the linear predictor.
        intercept: f32,
        /// Half-width of the uniform noise added to the target. Zero gives a
        /// noise-free target, which is the case a solver has no excuse on.
        noise_scale: f32,
    },
    /// A regression target with a known conditional mean that no linear model
    /// reproduces.
    NonlinearRegression {
        /// Which shape the conditional mean takes.
        kind: NonlinearKind,
        /// Half-width of the uniform noise added to the target.
        noise_scale: f32,
    },
    /// A count or positive-continuous response with a known rate.
    ///
    /// The linear predictor is known, so this is the family a generalized linear
    /// model is measured against: the recorded conditional mean is `E[y | x]`
    /// exactly, and the recorded coefficients are the ones a correctly specified
    /// fit should recover.
    GlmRegression {
        /// How the linear predictor becomes a response.
        link: GlmLink,
        /// Leading columns carrying a non-zero coefficient.
        informative: usize,
        /// Magnitude of the drawn coefficients.
        coefficient_scale: f32,
        /// The intercept added to the linear predictor.
        intercept: f32,
        /// The response's dispersion; see [`GlmLink`] for what it means at each
        /// link.
        dispersion: f32,
    },
    /// A linear regression problem over a design built to a requested condition
    /// number and rank.
    ///
    /// This is the family the crate's own least-squares path is exercised
    /// against. The design's columns are scaled geometrically so the ratio
    /// between the largest and smallest column magnitude is the requested
    /// condition number, and the columns past `rank` are exact copies of the
    /// leading ones, so the design's algebraic rank is the requested one exactly
    /// rather than approximately.
    ///
    /// **The recorded coefficients are the ones the target was drawn from, and
    /// on a rank-deficient design they are not the answer a solver should
    /// return.** A rank-deficient least-squares problem has an affine set of
    /// minimizers, and FerricML returns the minimum-norm point of that set;
    /// recovering the drawn `β` is only meaningful at full rank. The recorded
    /// [`Truth::ConditionedDesign::rank`] is what says which case a consumer is
    /// in.
    IllConditioned {
        /// Requested ratio between the largest and smallest column scale. `1.0`
        /// leaves the design as the source drew it.
        condition_number: f32,
        /// Requested algebraic rank, realized exactly by duplicating columns.
        rank: usize,
        /// Magnitude of the drawn coefficients.
        coefficient_scale: f32,
        /// Half-width of the uniform noise added to the target.
        noise_scale: f32,
    },
    /// A binary label drawn from a logistic function of a known linear score,
    /// at a requested marginal prevalence.
    ///
    /// The intercept is not a caller's parameter here: it is solved for, so the
    /// mean of the Bayes probabilities equals the requested `prevalence`. That
    /// is what makes prevalence a knob rather than an outcome — a caller asking
    /// for one positive row in twenty gets a problem whose *correct* answer has
    /// that prevalence, not a problem whose intercept was guessed at.
    LinearBinary {
        /// Leading columns carrying a non-zero coefficient.
        informative: usize,
        /// How steeply the Bayes probability moves with the score. A large
        /// separation approaches a hard boundary and a small one approaches an
        /// uninformative coin, so this is also what shapes the calibration curve
        /// a probabilistic classifier is measured against.
        separation: f32,
        /// Requested marginal positive rate, strictly between zero and one.
        prevalence: f32,
    },
    /// A binary label drawn from a logistic function of a nonlinear boundary, at
    /// a requested marginal prevalence.
    ///
    /// The Bayes probability is known and recorded; no coefficient vector is,
    /// because none produces this boundary. That distinction is the difference
    /// between [`Truth::Bayes`] and [`Truth::LinearBayes`].
    NonlinearBinary {
        /// Which boundary the labels are drawn across.
        kind: BinaryKind,
        /// How steeply the Bayes probability moves with the boundary score.
        separation: f32,
        /// Requested marginal positive rate, strictly between zero and one.
        prevalence: f32,
    },
    /// Labels over many classes, drawn from a known softmax of a known geometry
    /// at a requested class balance.
    ///
    /// Two knobs that are usually conflated are separated here. [`ClassBalance`]
    /// is *how often* each class occurs, solved for rather than observed, in
    /// exactly the sense [`Task::LinearBinary`]'s prevalence is. [`ClassGeometry`]
    /// is *which classes are confusable with which*, and the two available
    /// geometries differ in precisely that: blob centres confuse whichever pairs
    /// happen to land near each other, and a hierarchy confuses siblings far more
    /// than cousins. A metric that claims to be hierarchy-aware has to tell those
    /// two datasets apart at the same balance.
    ///
    /// The recorded [`Truth::MulticlassBayes`] is the whole probability row of
    /// every observation, because a scalar cannot say what a multiclass log loss
    /// or a one-versus-rest calibration curve should have been.
    Multiclass {
        /// Number of classes, at least two and at most `256`.
        classes: usize,
        /// How often each class occurs.
        balance: ClassBalance,
        /// Which classes are confusable with which.
        geometry: ClassGeometry,
        /// How steeply the class probabilities move with the geometry's scores.
        /// Normalized by the design's width, so it means the same thing at any
        /// column count.
        separation: f32,
    },
    /// Rows drawn around a known set of centres, with a known assignment and no
    /// target at all.
    ///
    /// This is the family that makes [`Dataset::target`](super::Dataset::target)
    /// an `Option`. An unsupervised problem has no targets, and handing back an
    /// empty or all-zero target vector would be a claim that one exists. What it
    /// does have is [`Truth::ClusterAssignment`], which is what a clusterer's
    /// output is scored against.
    Clustered {
        /// Number of clusters. Rows are dealt to them in turn, so the clusters
        /// are as equal as the row count allows.
        blobs: usize,
        /// How much of its own scatter each row keeps around its centre. Zero
        /// collapses every cluster to a point, which is the degenerate case a
        /// clusterer has no excuse on.
        spread: f32,
    },
    /// A regression target whose coefficients move linearly with the row's time.
    ///
    /// Row order is time order, and every row's time is recorded, so this data
    /// is correct for [`TimeSeriesSplit`](crate::model_selection::TimeSeriesSplit)
    /// without an adapter — that splitter takes a sample count and reads nothing
    /// but the order.
    ///
    /// [`Truth::DriftingPredictor`] records *both ends* of the coefficient
    /// vector rather than an average, which is what makes the drift a
    /// measurement: fit two windows, and the difference between the recovered
    /// coefficients is a number the recorded ends predict in advance.
    TimeOrdered {
        /// Leading columns carrying a non-zero coefficient.
        informative: usize,
        /// Magnitude of the coefficients at the first row.
        coefficient_scale: f32,
        /// How far each informative coefficient moves between the first row and
        /// the last. Zero gives a stationary series, which is the control case a
        /// drift detector must not fire on.
        drift: f32,
        /// The intercept added to the linear predictor. It does not drift.
        intercept: f32,
        /// Half-width of the uniform noise added to the target.
        noise_scale: f32,
    },
    /// Graded relevance over query blocks, with the within-query preference
    /// pairs already drawn.
    ///
    /// The design is `queries` blocks of `docs_per_query` rows. Each document's
    /// relevance grade is its rank within its own query under a known linear
    /// utility, so the pairs are exactly separable by the recorded coefficients
    /// and a ranker that fails on them has failed on the data rather than on the
    /// problem.
    ///
    /// The dataset carries the pairs as [`PairwiseObservation`]s and the query
    /// identifiers as `u64` groups, so it feeds
    /// [`PairwiseLinearRanker::fit`](crate::ranking::PairwiseLinearRanker::fit)
    /// and [`GroupKFold::split`](crate::model_selection::GroupKFold::split)
    /// without either one being adapted to it.
    Ranking {
        /// Number of query blocks.
        queries: usize,
        /// Documents in every query block, at least two. `queries *
        /// docs_per_query` must equal the recipe's row count.
        docs_per_query: usize,
        /// Number of distinct relevance grades, at least two. Fewer grades than
        /// documents produces tied pairs, which is what a real judgement looks
        /// like.
        grades: usize,
        /// Leading columns carrying a non-zero utility coefficient.
        informative: usize,
        /// Magnitude of the drawn utility coefficients.
        coefficient_scale: f32,
    },
}

/// Every parameter a task carries is refused unless it is finite, so no `Task`
/// that exists contains a NaN and `PartialEq` is reflexive on every value this
/// type can hold. That is the whole content of `Eq`, so it is implemented rather
/// than derived — `f32` is not `Eq` in general, and it is the *validation* that
/// makes it true here.
impl Eq for Task {}

impl Task {
    /// Which family this task belongs to, with its parameters dropped.
    ///
    /// The match is exhaustive, which is the point: a new variant of this enum
    /// does not compile until it says which family it joins, and [`Family`]'s
    /// own machinery then carries that as far as the suites. See [`Family`] for
    /// the whole chain and for the one gap left in it.
    ///
    /// ```
    /// use ferricml::datasets::{Family, NonlinearKind, Task};
    ///
    /// let task = Task::NonlinearRegression {
    ///     kind: NonlinearKind::Friedman,
    ///     noise_scale: 0.1,
    /// };
    /// assert_eq!(task.family(), Family::NonlinearRegression);
    /// assert_eq!(task.family().label(), "nonlinear-regression");
    /// ```
    pub const fn family(&self) -> Family {
        match self {
            Self::LinearRegression { .. } => Family::LinearRegression,
            Self::NonlinearRegression { .. } => Family::NonlinearRegression,
            Self::GlmRegression { .. } => Family::GlmRegression,
            Self::IllConditioned { .. } => Family::IllConditioned,
            Self::LinearBinary { .. } => Family::LinearBinary,
            Self::NonlinearBinary { .. } => Family::NonlinearBinary,
            Self::Multiclass { .. } => Family::Multiclass,
            Self::Clustered { .. } => Family::Clustered,
            Self::TimeOrdered { .. } => Family::TimeOrdered,
            Self::Ranking { .. } => Family::Ranking,
        }
    }

    /// This family's determinism envelope.
    ///
    /// ```
    /// use ferricml::datasets::{NonlinearKind, Portability, Task};
    ///
    /// // A linear target is integer draws, multiplications and additions.
    /// let linear = Task::LinearRegression {
    ///     informative: 3,
    ///     coefficient_scale: 1.0,
    ///     intercept: 0.0,
    ///     noise_scale: 0.1,
    /// };
    /// assert_eq!(linear.portability(), Portability::BitExact);
    ///
    /// // A sinusoidal one is not, and says so rather than promising bytes it
    /// // cannot deliver across a libm change.
    /// let wavy = Task::NonlinearRegression {
    ///     kind: NonlinearKind::Sinusoid,
    ///     noise_scale: 0.1,
    /// };
    /// assert_eq!(wavy.portability(), Portability::PerRunner);
    /// ```
    pub const fn portability(&self) -> Portability {
        match self {
            // Coefficients, a dot product and a uniform noise term: every
            // operation is exact or correctly rounded.
            Self::LinearRegression { .. } => Portability::BitExact,
            Self::NonlinearRegression { kind, .. } => kind.portability(),
            // `exp`.
            Self::GlmRegression { .. } => Portability::PerRunner,
            // `powf` for the column scales.
            Self::IllConditioned { .. } => Portability::PerRunner,
            // The logistic link, whichever boundary feeds it.
            Self::LinearBinary { .. } | Self::NonlinearBinary { .. } => Portability::PerRunner,
            // A softmax is a sum of exponentials, whichever geometry feeds it —
            // so a blob geometry whose own arithmetic is exact still reports
            // per-runner, exactly as an exact nonlinear boundary does.
            Self::Multiclass { .. } => Portability::PerRunner,
            // A centre draw, a multiply and an add.
            Self::Clustered { .. } => Portability::BitExact,
            // A time is a division of two integers, and the drifting predictor is
            // a dot product of interpolated coefficients.
            Self::TimeOrdered { .. } => Portability::BitExact,
            // A dot product, a sort, and integer grade arithmetic.
            Self::Ranking { .. } => Portability::BitExact,
        }
    }

    /// Whether this family's labels are the thing label noise and class
    /// balancing act on.
    ///
    /// [`Task::Ranking`] is deliberately excluded even though its target is a
    /// label vector: its grades are ranks within a query and its pairs are
    /// derived from them, so flipping a grade would leave the pairs describing an
    /// order the target contradicts. Refusing the combination is the same
    /// discipline the rest of the contamination checks follow.
    pub(super) const fn draws_labels(&self) -> bool {
        matches!(
            self,
            Self::LinearBinary { .. } | Self::NonlinearBinary { .. } | Self::Multiclass { .. }
        )
    }

    /// Whether this family's target carries the additive noise the
    /// noise-shaping contamination knobs reshape.
    ///
    /// A generalized linear response is excluded deliberately: its scatter is
    /// its own `dispersion`, and adding a symmetric term to a count or to a
    /// positive response would take it outside its own support.
    pub(super) const fn has_additive_noise(&self) -> bool {
        matches!(
            self,
            Self::LinearRegression { .. }
                | Self::NonlinearRegression { .. }
                | Self::IllConditioned { .. }
                | Self::TimeOrdered { .. }
        )
    }

    /// Whether this family has a continuous target an outlier can displace.
    ///
    /// Named positively rather than as "does not draw labels", which is what it
    /// used to be. That negation was correct only while every family either drew
    /// labels or drew a continuous target; [`Task::Clustered`] draws neither, and
    /// under the old spelling would have accepted an outlier fraction with
    /// nothing at all to displace.
    pub(super) const fn carries_outliers(self) -> bool {
        matches!(
            self,
            Self::LinearRegression { .. }
                | Self::NonlinearRegression { .. }
                | Self::GlmRegression { .. }
                | Self::IllConditioned { .. }
                | Self::TimeOrdered { .. }
        )
    }

    /// Whether this family's recorded truth is a function of the row *index*,
    /// which a row duplication would falsify.
    ///
    /// Only [`Task::Clustered`] is: its assignment is `row % blobs`, so a
    /// duplicated row would carry another cluster's features under its own
    /// recorded label. Every other family derives its truth from the finished
    /// design, so duplication reaches the truth and the data together.
    pub(super) const fn truth_is_positional(self) -> bool {
        matches!(self, Self::Clustered { .. })
    }

    /// Whether this family assigns the dataset's group labels itself.
    pub(super) const fn assigns_groups(self) -> bool {
        matches!(self, Self::Ranking { .. })
    }

    /// Whether this family produces targets at all.
    ///
    /// [`Task::Clustered`] does not, which is why "carries a task" and "has
    /// targets" are two questions rather than one.
    pub(super) const fn draws_target(self) -> bool {
        !matches!(self, Self::Clustered { .. })
    }

    /// Checks every parameter against its range and against the design's shape.
    ///
    /// Called by [`Recipe::with_task`](super::Recipe::with_task), before
    /// anything is allocated.
    pub(super) fn validate(&self, rows: usize, columns: usize) -> Result<(), DatasetError> {
        match *self {
            Self::LinearRegression {
                informative,
                coefficient_scale,
                intercept,
                noise_scale,
            } => {
                check_informative(informative, columns)?;
                check_positive(coefficient_scale, Parameter::CoefficientScale)?;
                check_finite(intercept, Parameter::Intercept)?;
                check_at_least_zero(noise_scale, Parameter::NoiseScale)?;
            }
            Self::NonlinearRegression { kind, noise_scale } => {
                check_informative(kind.columns_read(), columns)?;
                check_at_least_zero(noise_scale, Parameter::NoiseScale)?;
            }
            Self::GlmRegression {
                link,
                informative,
                coefficient_scale,
                intercept,
                dispersion,
            } => {
                check_informative(informative, columns)?;
                check_positive(coefficient_scale, Parameter::CoefficientScale)?;
                check_finite(intercept, Parameter::Intercept)?;
                check_positive(dispersion, Parameter::Dispersion)?;
                // The admissible range is link-dependent, and both halves are
                // the response's own. A multiplicative noise of one reaches
                // zero and above one reaches a negative response, neither of
                // which is a positive continuous value; and a Poisson cannot be
                // under-dispersed, so a count response's dispersion is at least
                // one — which also bounds Knuth's draw at `exp(4)` uniforms per
                // row instead of leaving it proportional to `1 / dispersion`.
                let admissible = match link {
                    GlmLink::LogCount => dispersion >= 1.0,
                    GlmLink::LogPositive => dispersion < 1.0,
                };
                if !admissible {
                    return Err(DatasetError::ParameterOutOfRange {
                        parameter: Parameter::Dispersion,
                    });
                }
            }
            Self::IllConditioned {
                condition_number,
                rank,
                coefficient_scale,
                noise_scale,
            } => {
                check_finite(condition_number, Parameter::ConditionNumber)?;
                if condition_number < 1.0 {
                    return Err(DatasetError::ParameterOutOfRange {
                        parameter: Parameter::ConditionNumber,
                    });
                }
                if rank == 0 {
                    return Err(DatasetError::ZeroRank);
                }
                if rank > columns {
                    return Err(DatasetError::RankExceedsDesign { rank, columns });
                }
                check_positive(coefficient_scale, Parameter::CoefficientScale)?;
                check_at_least_zero(noise_scale, Parameter::NoiseScale)?;
            }
            Self::LinearBinary {
                informative,
                separation,
                prevalence,
            } => {
                check_informative(informative, columns)?;
                check_positive(separation, Parameter::Separation)?;
                check_prevalence(prevalence)?;
            }
            Self::NonlinearBinary {
                separation,
                prevalence,
                ..
            } => {
                // Every boundary reads the first two columns.
                check_informative(2, columns)?;
                check_positive(separation, Parameter::Separation)?;
                check_prevalence(prevalence)?;
            }
            Self::Multiclass {
                classes,
                balance,
                separation,
                ..
            } => validate_multiclass(classes, balance, separation, columns)?,
            Self::Clustered { blobs, spread } => validate_clustered(blobs, spread, rows)?,
            Self::TimeOrdered {
                informative,
                coefficient_scale,
                drift,
                intercept,
                noise_scale,
            } => {
                check_informative(informative, columns)?;
                check_positive(coefficient_scale, Parameter::CoefficientScale)?;
                check_at_least_zero(drift, Parameter::Drift)?;
                check_finite(intercept, Parameter::Intercept)?;
                check_at_least_zero(noise_scale, Parameter::NoiseScale)?;
            }
            Self::Ranking {
                queries,
                docs_per_query,
                grades,
                informative,
                coefficient_scale,
            } => validate_ranking(
                queries,
                docs_per_query,
                grades,
                informative,
                coefficient_scale,
                rows,
                columns,
            )?,
        }
        Ok(())
    }

    /// Reshapes the design this family needs before its target is drawn.
    ///
    /// Two families do anything here, and for the same reason: their structure
    /// *is* the design, so it has to exist before the truth is recorded and
    /// before any consumer sees the matrix. [`Task::IllConditioned`] scales and
    /// duplicates columns; [`Task::Clustered`] moves every row onto its centre.
    /// Every other family draws over the design as the source produced it.
    ///
    /// The stream digest is passed in rather than derived here because a family
    /// that reshapes the design and a family that draws over it must read the
    /// *same* auxiliary streams — a clustered design's centres are recorded as
    /// truth, and two derivations of one number is how a design and its truth
    /// drift apart.
    pub(super) fn shape_design(
        &self,
        rows: usize,
        columns: usize,
        values: &mut [f32],
        digest: &[u8; 32],
    ) {
        match *self {
            Self::IllConditioned {
                condition_number,
                rank,
                ..
            } => {
                condition_columns(rows, columns, condition_number, values);
                duplicate_columns(rows, columns, rank, values);
            }
            Self::Clustered { blobs, spread } => {
                shape_clustered(rows, columns, blobs, spread, values, digest);
            }
            _ => {}
        }
    }

    /// Draws the family's target over a generated design, and records its truth.
    ///
    /// The two-stage split below is not ceremony: the structural families are
    /// the ones that produce a group vector or a pair list beside their target,
    /// and folding them into the scalar match would put four `None`s on every
    /// arm that never had either.
    pub(super) fn draw(
        &self,
        design: &DenseMatrix,
        contamination: &Contamination,
        digest: &[u8; 32],
    ) -> Drawn {
        match *self {
            Self::Multiclass {
                classes,
                balance,
                geometry,
                separation,
            } => {
                let (target, truth) = draw_multiclass(
                    design,
                    classes,
                    balance,
                    geometry,
                    separation,
                    contamination.label_noise(),
                    digest,
                );
                Drawn::plain(target, truth)
            }
            Self::Clustered { blobs, .. } => {
                let (target, truth) = draw_clustered(design, blobs, digest);
                Drawn::plain(target, truth)
            }
            Self::Ranking {
                queries,
                docs_per_query,
                grades,
                informative,
                coefficient_scale,
            } => {
                let (target, truth, groups, pairs) = draw_ranking(
                    design,
                    queries,
                    docs_per_query,
                    grades,
                    informative,
                    coefficient_scale,
                    digest,
                );
                Drawn {
                    target,
                    truth,
                    groups: Some(groups),
                    pairs: Some(pairs),
                }
            }
            _ => {
                let (target, truth) = self.draw_scalar(design, contamination, digest);
                Drawn::plain(Some(target), truth)
            }
        }
    }

    /// Draws the families whose whole output is one target value per row.
    fn draw_scalar(
        &self,
        design: &DenseMatrix,
        contamination: &Contamination,
        digest: &[u8; 32],
    ) -> (Target, Truth) {
        match *self {
            Self::LinearRegression {
                informative,
                coefficient_scale,
                intercept,
                noise_scale,
            } => {
                let coefficients =
                    draw_coefficients(design.columns(), informative, coefficient_scale, digest);
                let mean = linear_predictor(design, &coefficients, intercept);
                let targets = add_noise(design, &mean, noise_scale, contamination, digest);
                (
                    regression_target(targets),
                    Truth::LinearPredictor {
                        coefficients,
                        intercept,
                        conditional_mean: mean,
                    },
                )
            }
            Self::NonlinearRegression { kind, noise_scale } => {
                let mean: Vec<f32> = design
                    .iter_rows()
                    .map(|row| kind.conditional_mean(row))
                    .collect();
                let targets = add_noise(design, &mean, noise_scale, contamination, digest);
                (
                    regression_target(targets),
                    Truth::ConditionalMean { values: mean },
                )
            }
            Self::GlmRegression {
                link,
                informative,
                coefficient_scale,
                intercept,
                dispersion,
            } => {
                let coefficients =
                    draw_coefficients(design.columns(), informative, coefficient_scale, digest);
                let predictor = linear_predictor(design, &coefficients, intercept);
                let mean: Vec<f32> = predictor
                    .iter()
                    .map(|&value| {
                        f64::from(value)
                            .clamp(-LINK_ARGUMENT_LIMIT, LINK_ARGUMENT_LIMIT)
                            .exp() as f32
                    })
                    .collect();
                let targets = draw_glm_response(link, &mean, dispersion, contamination, digest);
                (
                    regression_target(targets),
                    Truth::LinearPredictor {
                        coefficients,
                        intercept,
                        conditional_mean: mean,
                    },
                )
            }
            Self::IllConditioned {
                rank,
                coefficient_scale,
                noise_scale,
                ..
            } => {
                let columns = design.columns();
                let coefficients = draw_coefficients(columns, columns, coefficient_scale, digest);
                let mean = linear_predictor(design, &coefficients, 0.0);
                let targets = add_noise(design, &mean, noise_scale, contamination, digest);
                (
                    regression_target(targets),
                    Truth::ConditionedDesign {
                        coefficients,
                        intercept: 0.0,
                        conditional_mean: mean,
                        rank: rank.min(columns),
                    },
                )
            }
            Self::LinearBinary {
                informative,
                separation,
                prevalence,
            } => {
                let coefficients =
                    draw_coefficients(design.columns(), informative, separation, digest);
                let scores: Vec<f64> = design
                    .iter_rows()
                    .map(|row| dot(row, &coefficients))
                    .collect();
                let offset = offset_for_prevalence(&scores, f64::from(prevalence));
                let (labels, probabilities) = draw_labels(&scores, offset, contamination, digest);
                (
                    Target::Binary(labels),
                    Truth::LinearBayes {
                        coefficients,
                        intercept: offset as f32,
                        probabilities,
                    },
                )
            }
            Self::NonlinearBinary {
                kind,
                separation,
                prevalence,
            } => {
                let scores: Vec<f64> = design
                    .iter_rows()
                    .map(|row| f64::from(separation) * kind.score(row))
                    .collect();
                let offset = offset_for_prevalence(&scores, f64::from(prevalence));
                let (labels, probabilities) = draw_labels(&scores, offset, contamination, digest);
                (Target::Binary(labels), Truth::Bayes { probabilities })
            }
            Self::TimeOrdered {
                informative,
                coefficient_scale,
                drift,
                intercept,
                noise_scale,
            } => {
                let (start_coefficients, end_coefficients, mean, times) = drifting_predictor(
                    design,
                    informative,
                    coefficient_scale,
                    drift,
                    intercept,
                    digest,
                );
                let targets = add_noise(design, &mean, noise_scale, contamination, digest);
                (
                    regression_target(targets),
                    Truth::DriftingPredictor {
                        start_coefficients,
                        end_coefficients,
                        intercept,
                        conditional_mean: mean,
                        times,
                    },
                )
            }
            // Reached only through `draw`, which routes every structural family
            // to its own arm before this one runs.
            Self::Multiclass { .. } | Self::Clustered { .. } | Self::Ranking { .. } => {
                unreachable!("a structural family is drawn by `draw`, not by `draw_scalar`")
            }
        }
    }
}

/// Everything a task family produces beside the design matrix.
///
/// A struct rather than a four-tuple because three of the four fields are `None`
/// for most families, and a tuple would make every call site count positions to
/// find out which.
pub(super) struct Drawn {
    /// The targets, or `None` for an unsupervised family.
    pub(super) target: Option<Target>,
    /// What the family is right about.
    pub(super) truth: Truth,
    /// Group labels the family assigned itself.
    pub(super) groups: Option<Vec<u64>>,
    /// Preference pairs the family drew.
    pub(super) pairs: Option<Vec<PairwiseObservation>>,
}

impl Drawn {
    /// A family that produced neither groups nor pairs.
    fn plain(target: Option<Target>, truth: Truth) -> Self {
        Self {
            target,
            truth,
            groups: None,
            pairs: None,
        }
    }
}

/// One auxiliary stream, seeded from one 64-bit word of a recipe's stream digest.
///
/// The digest is a SHA-256 of an injective encoding of the recipe's shape,
/// source and task, so its four words are four values that differ whenever any
/// of those differ and agree whenever none do. Slicing them is not a mixing
/// function of this module's own — `rng-single-source` forbids that, and this
/// file defines no generator — it is a read of bytes `sha2` already produced.
pub(super) fn stream(digest: &[u8; 32], word: usize) -> OwnedRng {
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[word * 8..word * 8 + 8]);
    OwnedRng::new(u64::from_le_bytes(bytes))
}

/// One draw mapped onto `[-1, 1)`, through exactly the map the design uses.
pub(super) fn signed_draw(rng: &mut OwnedRng) -> f32 {
    sampled_unit(rng.next_u64()) * 2.0 - 1.0
}

/// One draw mapped onto `[0, 1)`, through exactly the map the design uses.
pub(super) fn unit_draw(rng: &mut OwnedRng) -> f32 {
    sampled_unit(rng.next_u64())
}

/// The true coefficients: a scaled uniform draw on the informative columns, and
/// exactly zero elsewhere.
///
/// Zero rather than a small value on the uninformative columns, deliberately: a
/// consumer measuring whether a model *declined* to use a noise feature needs
/// the correct answer there to be zero and not merely small.
pub(super) fn draw_coefficients(
    columns: usize,
    informative: usize,
    scale: f32,
    digest: &[u8; 32],
) -> Vec<f32> {
    let mut rng = stream(digest, STREAM_COEFFICIENTS);
    (0..columns)
        .map(|column| {
            // Every column consumes a draw, informative or not, so the
            // coefficient of column `j` is a function of `j` and the stream
            // alone — and since `informative` is a dial, two recipes differing
            // in it *do* share a stream, so that positional encoding is
            // observable: widening the prefix leaves the coefficients of the
            // columns that already mattered bit-identical and turns on new ones.
            // A caller sweeping `informative` is adding informative columns to
            // one problem rather than drawing an unrelated one at each rung.
            let draw = signed_draw(&mut rng);
            if column < informative {
                scale * draw
            } else {
                0.0
            }
        })
        .collect()
}

/// `x · β` at one row, accumulated in `f64` in ascending column order.
pub(super) fn dot(row: &[f32], coefficients: &[f32]) -> f64 {
    sum_in_order(
        row.iter()
            .zip(coefficients)
            .map(|(&value, &coefficient)| f64::from(value) * f64::from(coefficient)),
    )
}

/// `Xβ + b`, one entry per row, narrowed to `f32` once at the end.
///
/// Rule 1 of the accumulation policy: the sum widens each term to `f64` and
/// narrows exactly once, so the conditional mean a consumer compares a fit
/// against is not itself a victim of cancellation across a wide design.
fn linear_predictor(design: &DenseMatrix, coefficients: &[f32], intercept: f32) -> Vec<f32> {
    design
        .iter_rows()
        .map(|row| (dot(row, coefficients) + f64::from(intercept)) as f32)
        .collect()
}

/// Adds the contamination-shaped noise to a conditional mean.
///
/// The three noise knobs compose in a fixed order, and the order is part of the
/// contract: the base draw is uniform, the heavy-tailed component divides it by
/// `1 - u`, the heteroscedastic factor scales it by the first feature's
/// magnitude, and the outlier displaces the finished target. Reordering them
/// gives a different distribution under the same parameters.
fn add_noise(
    design: &DenseMatrix,
    mean: &[f32],
    noise_scale: f32,
    contamination: &Contamination,
    digest: &[u8; 32],
) -> Vec<f32> {
    let mut noise_rng = stream(digest, STREAM_NOISE);
    let mut selector = stream(digest, STREAM_CONTAMINATION);
    let heavy_tail = contamination.heavy_tail();
    let heteroscedastic = contamination.heteroscedastic();
    let outlier_fraction = contamination.outlier_fraction();

    // The displacement is scale-free, so it is measured against the mean's own
    // magnitude rather than against an absolute constant a caller would have to
    // know.
    let spread = if mean.is_empty() {
        0.0
    } else {
        (sum_in_order(mean.iter().map(|&value| f64::from(value).abs())) / mean.len() as f64) as f32
    };
    let displacement = OUTLIER_DISPLACEMENT * spread.max(f32::MIN_POSITIVE);

    design
        .iter_rows()
        .zip(mean)
        .map(|(row, &centre)| {
            let base = signed_draw(&mut noise_rng);
            let tail_selector = unit_draw(&mut selector);
            let tail_magnitude = unit_draw(&mut selector);
            let outlier_selector = unit_draw(&mut selector);
            // `1 - u` is at least `2^-24` because `u` comes from a 24-bit map
            // whose largest value is `1 - 2^-24`, so the reciprocal is at most
            // `2^24` and the target stays finite by construction rather than by
            // a clamp.
            let drawn = if tail_selector < heavy_tail {
                base / (1.0 - tail_magnitude)
            } else {
                base
            };
            let scale = 1.0 + heteroscedastic * row.first().copied().unwrap_or(0.0).abs();
            let mut value = centre + noise_scale * scale * drawn;
            if outlier_selector < outlier_fraction {
                // Signed by which half of the selection interval the draw fell
                // in, so outliers do not all push the target the same way and
                // shift the target's mean by the contamination rate.
                let sign = if outlier_selector < outlier_fraction * 0.5 {
                    1.0
                } else {
                    -1.0
                };
                value += sign * displacement;
            }
            value
        })
        .collect()
}

/// Draws a generalized linear response from a known conditional mean.
fn draw_glm_response(
    link: GlmLink,
    mean: &[f32],
    dispersion: f32,
    contamination: &Contamination,
    digest: &[u8; 32],
) -> Vec<f32> {
    let mut rng = stream(digest, STREAM_NOISE);
    let mut selector = stream(digest, STREAM_CONTAMINATION);
    let outlier_fraction = contamination.outlier_fraction();
    mean.iter()
        .map(|&centre| {
            let outlier_selector = unit_draw(&mut selector);
            let value = match link {
                GlmLink::LogCount => {
                    let rate = f64::from(centre) / f64::from(dispersion);
                    (poisson(rate, &mut rng) * f64::from(dispersion)) as f32
                }
                GlmLink::LogPositive => centre * (1.0 + dispersion * signed_draw(&mut rng)),
            };
            if outlier_selector < outlier_fraction {
                // Multiplicative rather than additive, so a count stays a
                // non-negative count and a positive response stays positive.
                value * OUTLIER_DISPLACEMENT
            } else {
                value
            }
        })
        .collect()
}

/// Knuth's Poisson draw: multiply uniforms until the product falls below
/// `exp(-rate)`.
///
/// Exact for the rates this module reaches — the linear predictor is clamped at
/// `4`, so the rate is at most about `54.6` and the loop runs tens of times.
/// The uniforms come from the same 24-bit map every other draw here uses, and
/// one of them is eventually zero at worst, so the loop terminates for every
/// finite rate.
fn poisson(rate: f64, rng: &mut OwnedRng) -> f64 {
    let limit = (-rate).exp();
    let mut count = 0.0_f64;
    let mut product = 1.0_f64;
    loop {
        product *= f64::from(unit_draw(rng));
        if product <= limit {
            return count;
        }
        count += 1.0;
    }
}

/// The intercept that makes the mean Bayes probability equal the requested
/// prevalence.
///
/// Bisection rather than a closed form, because there is no closed form: the
/// mean of a logistic over an arbitrary score vector is not invertible. It is
/// monotone in the offset, which is what makes bisection exact to the last bit
/// it can represent — 64 halvings of a 100-wide bracket leave an interval of
/// about `5e-18`, well below the resolution of the probabilities it decides.
///
/// The bracket is `±50`, past which every probability has saturated to `0` or
/// `1` in `f64` and no further offset changes the mean. A prevalence that even
/// that cannot reach — every score identical and the request on the wrong side
/// of it — leaves the offset at the bracket's edge rather than looping, which is
/// a saturation the realized prevalence then shows.
///
/// What the halvings cost is a logistic over every row, per halving, and that
/// is the generator's dominant per-row cost at a narrow width. Most of those
/// passes are spent proving something already known: once the crossing has been
/// located, every midpoint that lands far from it has a decided comparison.
/// [`bracket_the_crossing`] locates it with a guarded Newton iteration and
/// returns two offsets whose comparison is *proved* rather than assumed, and the
/// halvings below run unchanged except that a midpoint outside that pair takes
/// its branch without a pass. The sequence of branches — and therefore the
/// returned bits — is the one the unconditional loop produces; only the number
/// of evaluated passes changes. `offset_for_prevalence_is_the_unconditional_bisection`
/// asserts that against a reference bisection rather than describing it.
pub(super) fn offset_for_prevalence(scores: &[f64], prevalence: f64) -> f64 {
    let (proved_below, proved_above) = bracket_the_crossing(scores, prevalence);
    let mut low = -50.0_f64;
    let mut high = 50.0_f64;
    for _ in 0..64 {
        let middle = 0.5 * (low + high);
        // The interval has reached the resolution of `f64` at this magnitude:
        // the midpoint is one of its own endpoints, so this update leaves the
        // interval where every later one would leave it. Applying it and
        // stopping is the same answer as running the remaining halvings.
        let settled = middle == low || middle == high;
        let below = if middle <= proved_below {
            true
        } else if middle >= proved_above {
            false
        } else {
            mean_probability(scores, middle) < prevalence
        };
        if below {
            low = middle;
        } else {
            high = middle;
        }
        if settled {
            break;
        }
    }
    0.5 * (low + high)
}

/// The mean Bayes probability at one offset — the quantity the bisection
/// compares, evaluated exactly as the unconditional loop evaluates it.
fn mean_probability(scores: &[f64], offset: f64) -> f64 {
    sum_in_order(scores.iter().map(|&score| sigmoid_f64(score + offset))) / scores.len() as f64
}

/// The mean and its derivative in one pass.
///
/// The derivative only steers the search for the crossing, so its bits are not
/// part of any answer; the mean it returns beside it is [`mean_probability`]'s,
/// term for term and in the same order.
fn mean_and_slope(scores: &[f64], offset: f64) -> (f64, f64) {
    let mut total = -0.0_f64;
    let mut slope = 0.0_f64;
    for &score in scores {
        let probability = sigmoid_f64(score + offset);
        total += probability;
        slope += probability * (1.0 - probability);
    }
    let rows = scores.len() as f64;
    (total / rows, slope / rows)
}

/// Two offsets straddling the crossing, at which the bisection's comparison is
/// **proved** without evaluating it: at or below the first the mean is below the
/// requested prevalence, at or above the second it is not.
///
/// The proof is the same monotonicity the bisection already rests on, made
/// two-sided. Rounding `score + offset` is monotone in the offset and the
/// logistic is monotone in its argument, so the exact mean of the exactly
/// rounded arguments — call it `F` — is non-decreasing in the offset however
/// coarsely the sum is formed. What the loop actually compares is a computed
/// `M` that differs from `F` by at most `E`: sequential summation of `n` terms
/// in `[0, 1]` contributes at most about `n · u · mean`, the logistic itself a
/// few ulps per term, the final division one more, so `E ≤ (n + 4) · u` with
/// `u = 2⁻⁵³`. Hence for any `m ≤ p`, `M(m) ≤ F(m) + E ≤ F(p) + E ≤ M(p) + 2E`,
/// and `M(p) + 2E < prevalence` decides every one of them at once. The mirrored
/// statement decides the other side. The band below is four times `2E`, which
/// buys headroom for a platform logarithm worse than the one this was measured
/// against without costing more than one halving.
///
/// A degenerate geometry — every score identical, a prevalence past what the
/// bracket can reach, a slope that has saturated to zero — proves nothing and
/// returns an empty pair, at which point every halving evaluates and the solve
/// costs exactly what it cost before.
fn bracket_the_crossing(scores: &[f64], prevalence: f64) -> (f64, f64) {
    let rows = scores.len() as f64;
    let band = 2.0 * (rows + 64.0) * f64::EPSILON;
    let mut proved_below = f64::NEG_INFINITY;
    let mut proved_above = f64::INFINITY;

    // Guarded Newton: the step is taken only while it stays inside the interval
    // the evaluated points have already bracketed, so a flat or saturated region
    // falls back to a halving instead of leaving the bracket.
    let mut guard_low = -50.0_f64;
    let mut guard_high = 50.0_f64;
    let mut point = 0.0_f64;
    let mut slope_at_point = 0.0_f64;
    for _ in 0..12 {
        let (mean, slope) = mean_and_slope(scores, point);
        if mean + band < prevalence {
            proved_below = proved_below.max(point);
        } else if mean >= prevalence + band {
            proved_above = proved_above.min(point);
        }
        if mean < prevalence {
            guard_low = point;
        } else {
            guard_high = point;
        }
        slope_at_point = slope;
        if slope <= 0.0 {
            break;
        }
        let step = (mean - prevalence) / slope;
        // Settled is tested before the step is taken, not after. The guard
        // below is a half-open interval whose upper end is this very iterate
        // once the mean has reached the prevalence, so a step of zero would be
        // rejected by it and replaced by the middle of the guard — throwing the
        // located crossing away at the exact moment it was found, and leaving
        // the pair below anchored nowhere near it.
        if step.abs() <= 1e-14 * point.abs().max(1.0) {
            break;
        }
        let next = point - step;
        let next = if next > guard_low && next < guard_high {
            next
        } else {
            0.5 * (guard_low + guard_high)
        };
        if next == point {
            break;
        }
        point = next;
    }

    // Newton's own iterates prove something only where they happened to land, so
    // the pair is tightened deliberately: step out by the distance the slope
    // says clears the band, and widen until both sides clear it. Every halving
    // between the returned offsets still evaluates, so a tighter pair is
    // directly fewer passes — and two passes buy roughly a dozen of them.
    if slope_at_point > 0.0 {
        let mut offset = (4.0 * band / slope_at_point).max(8.0 * point.abs() * f64::EPSILON);
        for _ in 0..8 {
            let mut both = true;
            let candidate = point - offset;
            if proved_below < candidate {
                if mean_probability(scores, candidate) + band < prevalence {
                    proved_below = candidate;
                } else {
                    both = false;
                }
            }
            let candidate = point + offset;
            if proved_above > candidate {
                if mean_probability(scores, candidate) >= prevalence + band {
                    proved_above = candidate;
                } else {
                    both = false;
                }
            }
            if both {
                break;
            }
            offset *= 4.0;
        }
    }

    // Two sides that cross would mean the bound above was violated. Nothing
    // observed does that, and if a platform ever did, the empty pair is the
    // answer that cannot be wrong.
    if proved_below < proved_above {
        (proved_below, proved_above)
    } else {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Draws labels from the Bayes probabilities, then applies label noise.
///
/// The returned probabilities are `P(observed label = 1 | x)`, which is what a
/// consumer measuring calibration needs: with a flip rate `e`, an observed label
/// is positive with probability `p(1 - e) + (1 - p)e`, and reporting the
/// pre-noise `p` instead would make a perfectly calibrated model look
/// mis-calibrated by exactly the noise the caller asked for.
fn draw_labels(
    scores: &[f64],
    offset: f64,
    contamination: &Contamination,
    digest: &[u8; 32],
) -> (BinaryTargets, Vec<f32>) {
    let mut rng = stream(digest, STREAM_LABELS);
    let flip_rate = f64::from(contamination.label_noise());
    let mut labels = Vec::with_capacity(scores.len());
    let mut probabilities = Vec::with_capacity(scores.len());
    for &score in scores {
        let clean = sigmoid_f64(score + offset);
        let drawn = u8::from(f64::from(unit_draw(&mut rng)) < clean);
        let flipped = f64::from(unit_draw(&mut rng)) < flip_rate;
        labels.push(if flipped { 1 - drawn } else { drawn });
        probabilities.push((clean * (1.0 - flip_rate) + (1.0 - clean) * flip_rate) as f32);
    }
    (
        BinaryTargets::new(labels).expect("a drawn label is 0 or 1 by construction"),
        probabilities,
    )
}

/// Wraps drawn values in the validated regression container.
fn regression_target(values: Vec<f32>) -> Target {
    Target::Regression(
        RegressionTargets::new(values).expect("a bounded expression over finite values is finite"),
    )
}

/// Scales column `j` by `condition_number^(-j / (columns - 1))`.
///
/// Written as a base-ten power of a base-ten logarithm rather than as a direct
/// `powf` of the condition number, because that is the expression FerricML's
/// least-squares corpus conditions its designs with
/// (`src/linear_model/least_squares.rs`): at the corpus's `1e8` the logarithm is
/// exactly `8`, the exponent is exactly the corpus's `-8 * column / (columns -
/// 1)`, and the two paths agree bit for bit. `family_tests.rs` asserts that
/// rather than describing it.
///
/// A condition number of `1.0` gives a logarithm of zero, every scale is exactly
/// `1.0`, and the design is left as the source drew it.
pub(super) fn condition_columns(
    rows: usize,
    columns: usize,
    condition_number: f32,
    values: &mut [f32],
) {
    if condition_number == 1.0 {
        return;
    }
    scale_columns(
        rows,
        columns,
        f64::from(condition_number).log10() as f32,
        values,
    );
}

/// Scales column `j` by `10^(-decades * j / (columns - 1))`.
///
/// The one place a real power appears in this module, and the whole reason both
/// the ill-conditioned family and the per-column scale spread report
/// [`Portability::PerRunner`]. Zero decades returns without touching the values,
/// so an uncontaminated design is byte-identical to one that never entered here
/// — which is what keeps the P1-P3 frozen streams unmoved by this phase.
pub(super) fn scale_columns(rows: usize, columns: usize, decades: f32, values: &mut [f32]) {
    if decades == 0.0 {
        return;
    }
    let decades = f64::from(decades);
    let denominator = (columns - 1).max(1) as f64;
    for column in 0..columns {
        let scale = 10.0_f64.powf(-decades * column as f64 / denominator) as f32;
        for row in 0..rows {
            values[row * columns + column] *= scale;
        }
    }
}

/// Replaces the columns past `rank` with exact copies of the leading ones.
///
/// Exact copies, so the design's algebraic rank is `rank` exactly rather than
/// numerically close to it: a solver's reported rank can then be compared with
/// `assert_eq!` instead of with a tolerance, which is what makes the rank
/// contract testable at all. With `rank == columns - 1` the last column becomes
/// the first, which is precisely the corpus case the crate's rank-deficiency
/// tests are written against.
pub(super) fn duplicate_columns(rows: usize, columns: usize, rank: usize, values: &mut [f32]) {
    if rank >= columns {
        return;
    }
    for column in rank..columns {
        let source = column - rank;
        for row in 0..rows {
            values[row * columns + column] = values[row * columns + source];
        }
    }
}

fn check_finite(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DatasetError::NonFiniteParameter { parameter })
    }
}

pub(super) fn check_positive(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    check_finite(value, parameter)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(DatasetError::ParameterOutOfRange { parameter })
    }
}

pub(super) fn check_at_least_zero(value: f32, parameter: Parameter) -> Result<(), DatasetError> {
    check_finite(value, parameter)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(DatasetError::ParameterOutOfRange { parameter })
    }
}

fn check_prevalence(value: f32) -> Result<(), DatasetError> {
    check_finite(value, Parameter::Prevalence)?;
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Prevalence,
        })
    }
}

pub(super) fn check_informative(informative: usize, columns: usize) -> Result<(), DatasetError> {
    if informative == 0 {
        return Err(DatasetError::ZeroInformativeColumns);
    }
    if informative > columns {
        return Err(DatasetError::InformativeColumnsExceedDesign {
            informative,
            columns,
        });
    }
    Ok(())
}
