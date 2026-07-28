use std::error::Error;
use std::fmt;

/// A named scalar knob on a task family, a contamination, or a weight pattern.
///
/// Every one of these is an `f32` a caller supplies, and every one of them has
/// an admissible range that the recipe constructor checks. The parameter is
/// carried as a variant rather than as a string so a caller can match on
/// *which* knob it got wrong without parsing a message, and so a knob that is
/// renamed cannot silently keep matching an old string.
///
/// It is `#[non_exhaustive]` because a later family arrives with its own knobs,
/// and a caller matching only the ones it sets must not break when it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Parameter {
    /// Magnitude of the true coefficients a linear family draws.
    CoefficientScale,
    /// The intercept a linear family adds to its linear predictor.
    Intercept,
    /// Half-width of the uniform noise added to a regression target.
    NoiseScale,
    /// Requested ratio between the largest and smallest column scale of an
    /// ill-conditioned design.
    ConditionNumber,
    /// A generalized linear family's response dispersion.
    Dispersion,
    /// How steeply a binary family's Bayes probability moves with its score.
    Separation,
    /// The requested marginal positive rate of a binary family.
    Prevalence,
    /// Fraction of labels flipped after they are drawn.
    LabelNoise,
    /// Fraction of rows whose target is displaced by an outlier.
    OutlierFraction,
    /// Fraction of rows whose noise is drawn from the heavy-tailed component.
    HeavyTail,
    /// How strongly the noise scale tracks the first feature's magnitude.
    Heteroscedastic,
    /// Fraction of rows replaced by a copy of an earlier row.
    DuplicateRows,
    /// Decades between the largest and smallest per-column magnitude.
    FeatureScaleSpread,
    /// The weight a ramp pattern starts at.
    WeightLow,
    /// The weight a ramp pattern ends at.
    WeightHigh,
    /// A weight an alternating pattern assigns to even rows.
    WeightFirst,
    /// A weight an alternating pattern assigns to odd rows.
    WeightSecond,
}

impl Parameter {
    /// The name a caller wrote in their own source.
    const fn name(self) -> &'static str {
        match self {
            Self::CoefficientScale => "coefficient_scale",
            Self::Intercept => "intercept",
            Self::NoiseScale => "noise_scale",
            Self::ConditionNumber => "condition_number",
            Self::Dispersion => "dispersion",
            Self::Separation => "separation",
            Self::Prevalence => "prevalence",
            Self::LabelNoise => "label_noise",
            Self::OutlierFraction => "outlier_fraction",
            Self::HeavyTail => "heavy_tail",
            Self::Heteroscedastic => "heteroscedastic",
            Self::DuplicateRows => "duplicate_rows",
            Self::FeatureScaleSpread => "feature_scale_spread",
            Self::WeightLow => "low",
            Self::WeightHigh => "high",
            Self::WeightFirst => "first",
            Self::WeightSecond => "second",
        }
    }

    /// The admissible range, spelled the way a reader would check it.
    ///
    /// `Dispersion` reads as two ranges because it has two, and both are the
    /// response's own. A count response is quasi-Poisson: its variance is
    /// `dispersion` times its mean, and a Poisson cannot be *under*-dispersed,
    /// so below `1` the request describes no distribution this family offers.
    /// A positive continuous response multiplies its mean by
    /// `1 + dispersion * u` with `u` in `[-1, 1)`, and would reach zero or
    /// below at `1`.
    const fn admissible(self) -> &'static str {
        match self {
            Self::CoefficientScale | Self::Separation => "a finite value above 0",
            Self::Intercept => "a finite value",
            Self::NoiseScale | Self::Heteroscedastic | Self::FeatureScaleSpread => {
                "a finite value at or above 0"
            }
            Self::ConditionNumber => "a finite value at or above 1",
            Self::Dispersion => {
                "at or above 1 for a count response, and above 0 and below 1 for a positive one"
            }
            Self::Prevalence => "strictly between 0 and 1",
            Self::LabelNoise => "between 0 and 0.5, above which the labels invert",
            Self::OutlierFraction | Self::DuplicateRows => "at or above 0 and below 1",
            Self::HeavyTail => "between 0 and 1",
            Self::WeightLow | Self::WeightHigh | Self::WeightFirst | Self::WeightSecond => {
                "a finite value at or above 0, with a positive total"
            }
        }
    }
}

/// An error encountered while constructing a dataset recipe.
///
/// Every variant is produced by [`Recipe::new`](super::Recipe::new) or
/// [`Recipe::seeded`](super::Recipe::seeded), which is to say before any
/// generation buffer is allocated. A `Recipe` that exists describes a dataset
/// that can be generated, so [`Recipe::generate`](super::Recipe::generate) and
/// its `_into` form return no error at all.
///
/// This is a separate type from [`DataError`](crate::data::DataError) rather
/// than a set of variants added to it: `DataError` says what was wrong with
/// *data a caller supplied*, and these say what was wrong with a *request to
/// produce data*. The shapes overlap — both refuse an empty matrix — but the
/// consumers do not, and a caller matching on one should not have to consider
/// the other's variants.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatasetError {
    /// A recipe was requested with no rows.
    ZeroRows,
    /// A recipe was requested with no columns.
    ZeroColumns,
    /// `rows * columns` cannot be represented by [`usize`].
    DimensionOverflow {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        columns: usize,
    },
    /// A [`Source::Xorshift32`](super::Source::Xorshift32) state was zero.
    ///
    /// Zero is xorshift's fixed point: it maps to itself forever, so the
    /// "stream" is one repeated value and every column of the design would be
    /// identical.
    ZeroXorshiftState,
    /// A [`Source::Lattice`](super::Source::Lattice) modulus was below two.
    ///
    /// A modulus of `1` leaves one residue, and a modulus of `0` has no
    /// residues at all and would divide by zero.
    LatticeModulusTooSmall {
        /// Requested modulus.
        modulus: u64,
    },
    /// A [`Source::Lattice`](super::Source::Lattice) modulus is too large for
    /// its residues to survive the `f32` the design is stored in.
    ///
    /// Every integer up to `2^24` is exactly representable as an `f32` and
    /// nothing above it is, so a larger modulus silently collapses distinct
    /// residues onto one value: the recipe would describe more distinct design
    /// values than the generated matrix can hold.
    LatticeModulusNotRepresentable {
        /// Requested modulus.
        modulus: u64,
        /// The largest modulus whose residues are all exact in `f32`.
        limit: u64,
    },
    /// A task, contamination or weight parameter was not finite.
    ///
    /// Separate from [`DatasetError::ParameterOutOfRange`] because the two are
    /// different mistakes: a NaN is almost always a computation that went wrong
    /// upstream, and an out-of-range value is almost always a misread of what
    /// the knob means.
    NonFiniteParameter {
        /// The knob that was not finite.
        parameter: Parameter,
    },
    /// A task, contamination or weight parameter was outside its range.
    ParameterOutOfRange {
        /// The knob that was out of range.
        parameter: Parameter,
    },
    /// A task was asked for no informative columns.
    ///
    /// A design with every coefficient zero has a correct answer — the zero
    /// vector — but nothing to recover, and a family whose whole point is a
    /// known `β` should say so at the constructor rather than hand back a
    /// dataset that silently measures nothing.
    ZeroInformativeColumns,
    /// A task needs more columns than the recipe's design has.
    ///
    /// Both the caller-set `informative` count and a nonlinear kind's own
    /// column appetite come through here: Friedman's expression reads five
    /// columns, so it cannot be drawn over four.
    InformativeColumnsExceedDesign {
        /// Columns the task needs.
        informative: usize,
        /// Columns the design has.
        columns: usize,
    },
    /// An ill-conditioned design was asked for rank zero.
    ZeroRank,
    /// An ill-conditioned design was asked for more rank than it has columns.
    RankExceedsDesign {
        /// Requested rank.
        rank: usize,
        /// Columns the design has.
        columns: usize,
    },
    /// Every column of the design would be a constant.
    ///
    /// A design with no varying column carries no signal at all, so this is
    /// refused rather than generated: contamination is supposed to make a
    /// problem harder, not empty.
    ConstantColumnsLeaveNoSignal {
        /// Requested constant columns.
        constant_columns: usize,
        /// Columns the design has.
        columns: usize,
    },
    /// More collinear pairs were requested than the design's varying columns
    /// can supply.
    ///
    /// Each pair overwrites one column with a multiple of another, so `n` pairs
    /// need `2n` varying columns; overlapping them would make a pair's source
    /// column itself a copy and the requested count would not describe the
    /// design.
    CollinearPairsExceedDesign {
        /// Requested pairs.
        pairs: usize,
        /// Varying columns available after the constant ones.
        available: usize,
    },
    /// A label-shaping contamination was asked for over a task that draws no
    /// labels.
    ///
    /// Refused rather than ignored. A robustness sweep that set a label-noise
    /// rate on a regression task and got a clean dataset back would report the
    /// model as robust to a contamination it never received, which is a worse
    /// outcome than a build error.
    ContaminationNeedsLabels {
        /// The knob that had nothing to act on.
        parameter: Parameter,
    },
    /// A noise-shaping contamination was asked for over a task whose target
    /// carries no additive noise.
    ///
    /// A generalized linear response's scatter is its own dispersion, and a
    /// label has no noise term at all; adding a symmetric term to either would
    /// take the response outside its own support. Refused for the same reason as
    /// [`DatasetError::ContaminationNeedsLabels`].
    ContaminationNeedsAdditiveNoise {
        /// The knob that had nothing to act on.
        parameter: Parameter,
    },
    /// A class-balancing weight pattern was asked for over a task that draws no
    /// labels.
    ///
    /// The pattern's weights are a function of the class counts, and a
    /// regression target has none. Refused at the constructor rather than
    /// silently degraded to uniform weights, which would look like it worked.
    WeightPatternNeedsLabels,
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRows => f.write_str("dataset row count must be non-zero"),
            Self::ZeroColumns => f.write_str("dataset column count must be non-zero"),
            Self::DimensionOverflow { rows, columns } => {
                write!(f, "dataset dimensions overflow usize: {rows} x {columns}")
            }
            Self::ZeroXorshiftState => f.write_str(
                "a xorshift32 source state must be non-zero, because zero is a fixed point",
            ),
            Self::LatticeModulusTooSmall { modulus } => write!(
                f,
                "lattice modulus {modulus} must be at least 2 to leave more than one residue"
            ),
            Self::LatticeModulusNotRepresentable { modulus, limit } => write!(
                f,
                "lattice modulus {modulus} exceeds {limit}, above which distinct residues \
                 collapse onto the same f32"
            ),
            Self::NonFiniteParameter { parameter } => {
                write!(f, "{} must be finite", parameter.name())
            }
            Self::ParameterOutOfRange { parameter } => write!(
                f,
                "{} must be {}",
                parameter.name(),
                parameter.admissible()
            ),
            Self::ZeroInformativeColumns => f.write_str(
                "a task must read at least one informative column, or there is no signal to recover",
            ),
            Self::InformativeColumnsExceedDesign {
                informative,
                columns,
            } => write!(
                f,
                "the task reads {informative} columns but the design has {columns}"
            ),
            Self::ZeroRank => {
                f.write_str("an ill-conditioned design must have rank at least 1")
            }
            Self::RankExceedsDesign { rank, columns } => write!(
                f,
                "requested rank {rank} exceeds the {columns} columns of the design"
            ),
            Self::ConstantColumnsLeaveNoSignal {
                constant_columns,
                columns,
            } => write!(
                f,
                "a design of {columns} columns cannot carry {constant_columns} constant \
                 ones and still vary"
            ),
            Self::CollinearPairsExceedDesign { pairs, available } => write!(
                f,
                "collinear pairs {pairs} need {} varying columns but {available} are available",
                pairs * 2
            ),
            Self::ContaminationNeedsLabels { parameter } => write!(
                f,
                "{} needs a task that draws labels",
                parameter.name()
            ),
            Self::ContaminationNeedsAdditiveNoise { parameter } => write!(
                f,
                "{} needs a task whose target carries additive noise",
                parameter.name()
            ),
            Self::WeightPatternNeedsLabels => f.write_str(
                "a class-balancing weight pattern needs a task that draws labels",
            ),
        }
    }
}

impl Error for DatasetError {}
