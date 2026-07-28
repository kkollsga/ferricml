use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

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
    /// The ratio between a multiclass family's most and least common class.
    BalanceRatio,
    /// Within-cluster spread of a clustered design.
    Spread,
    /// How far a time-ordered family's coefficients move across the whole span.
    Drift,
    /// The ratio between an unbalanced grouping's largest and smallest group.
    GroupSizeRatio,
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
    /// The name a caller wrote in their own source, qualified where two knobs
    /// share a field name.
    const fn name(self) -> &'static str {
        match self {
            Self::CoefficientScale => "coefficient_scale",
            Self::Intercept => "intercept",
            Self::NoiseScale => "noise_scale",
            Self::ConditionNumber => "condition_number",
            Self::Dispersion => "dispersion",
            Self::Separation => "separation",
            Self::Prevalence => "prevalence",
            // Both of these are spelled `ratio` at the call site, so the name
            // carries the qualifier the variant already knows: two knobs that
            // rendered identically would make a message ambiguous exactly when a
            // recipe sets both.
            Self::BalanceRatio => "balance ratio",
            Self::Spread => "spread",
            Self::Drift => "drift",
            Self::GroupSizeRatio => "group size ratio",
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
            Self::NoiseScale
            | Self::Heteroscedastic
            | Self::FeatureScaleSpread
            | Self::Spread
            | Self::Drift => "a finite value at or above 0",
            Self::ConditionNumber | Self::BalanceRatio | Self::GroupSizeRatio => {
                "a finite value at or above 1"
            }
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
    /// A multiclass family was asked for fewer than two classes.
    ///
    /// One class is not a classification problem: every prediction is correct,
    /// every metric is degenerate, and the requested balance describes nothing.
    /// The two-class case is [`Task::LinearBinary`](super::Task::LinearBinary),
    /// which records a scalar Bayes probability rather than a row of them.
    TooFewClasses {
        /// Requested class count.
        classes: usize,
    },
    /// A multiclass family was asked for more classes than a label can hold.
    ///
    /// Labels are `u8`, which is the vocabulary
    /// [`ClassTargets`](crate::data::ClassTargets) validates and every
    /// classifier in this crate consumes, so the largest label is `255` and the
    /// largest class count is `256`.
    TooManyClasses {
        /// Requested class count.
        classes: usize,
        /// The largest class count a `u8` label can express.
        limit: usize,
    },
    /// A clustered design was asked for no clusters.
    ZeroBlobs,
    /// A clustered design was asked for more clusters than it has rows.
    ///
    /// Some cluster would then be empty, and a fixture whose recorded truth
    /// names a cluster no row belongs to measures nothing about a clusterer.
    BlobsExceedRows {
        /// Requested cluster count.
        blobs: usize,
        /// Rows the design has.
        rows: usize,
    },
    /// A ranking family's `queries * docs_per_query` is not the recipe's row
    /// count.
    ///
    /// A ranking design is a stack of query blocks, and a partial trailing block
    /// would leave a query with a different pair count from every other one.
    /// Refused rather than truncated, because the caller's two numbers and the
    /// recipe's row count cannot all three be what they meant.
    RankingShapeMismatch {
        /// Rows the recipe produces.
        rows: usize,
        /// Requested query count.
        queries: usize,
        /// Requested documents per query.
        docs_per_query: usize,
    },
    /// A ranking family was asked for fewer than two documents per query.
    ///
    /// A query holding one document yields no pair at all, and the pairs are
    /// the whole output of the family.
    TooFewDocumentsPerQuery {
        /// Requested documents per query.
        docs_per_query: usize,
    },
    /// A ranking family was asked for fewer than two relevance grades.
    ///
    /// One grade makes every pair a tie, which is a dataset with no preference
    /// information in it.
    TooFewGrades {
        /// Requested grade count.
        grades: usize,
    },
    /// A group pattern was asked for no groups.
    ZeroGroups,
    /// A group pattern was asked for more groups than the design has rows.
    ///
    /// The labels must partition the rows, so an empty group would be an
    /// identifier no row carries — and a splitter counting distinct groups
    /// would disagree with the number the caller asked for.
    GroupsExceedRows {
        /// Requested group count.
        groups: usize,
        /// Rows the design has.
        rows: usize,
    },
    /// A group pattern was asked for over a task that assigns groups itself.
    ///
    /// [`Task::Ranking`](super::Task::Ranking) labels each row with its query,
    /// and those labels are what make its pairs within-query. Overwriting them
    /// with a pattern would leave the pairs and the groups describing different
    /// partitions of the same rows, so the combination is refused rather than
    /// resolved by precedence.
    GroupPatternConflictsWithTask,
    /// A contamination knob would falsify the truth the current task records.
    ///
    /// Row duplication over a clustered design is the case this exists for: the
    /// recorded cluster assignment is a function of the row index, and copying
    /// a row's features without copying its recorded cluster would make the
    /// truth wrong for exactly the duplicated rows. A silently wrong ground
    /// truth is worse than any contamination, so the request is refused.
    ContaminationConflictsWithTask {
        /// The knob whose effect the task's truth cannot survive.
        parameter: Parameter,
    },
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
            Self::TooFewClasses { classes } => write!(
                f,
                "a multiclass task needs at least 2 classes, not {classes}"
            ),
            Self::TooManyClasses { classes, limit } => write!(
                f,
                "a multiclass task of {classes} classes exceeds the {limit} a u8 label holds"
            ),
            Self::ZeroBlobs => f.write_str("a clustered design must have at least 1 cluster"),
            Self::BlobsExceedRows { blobs, rows } => write!(
                f,
                "a clustered design of {blobs} clusters over {rows} rows would leave a \
                 cluster empty"
            ),
            Self::RankingShapeMismatch {
                rows,
                queries,
                docs_per_query,
            } => write!(
                f,
                "a ranking design of {queries} queries holding {docs_per_query} documents is \
                 {} rows, not {rows}",
                queries.saturating_mul(*docs_per_query)
            ),
            Self::TooFewDocumentsPerQuery { docs_per_query } => write!(
                f,
                "a query of {docs_per_query} documents yields no pair; at least 2 are needed"
            ),
            Self::TooFewGrades { grades } => write!(
                f,
                "a ranking task of {grades} relevance grade makes every pair a tie; at least \
                 2 are needed"
            ),
            Self::ZeroGroups => f.write_str("a group pattern must have at least 1 group"),
            Self::GroupsExceedRows { groups, rows } => write!(
                f,
                "a grouping of {groups} groups over {rows} rows would leave a group empty"
            ),
            Self::GroupPatternConflictsWithTask => f.write_str(
                "the task assigns groups itself, so a group pattern would describe a \
                 different partition from its pairs",
            ),
            Self::ContaminationConflictsWithTask { parameter } => write!(
                f,
                "{} would falsify the truth this task records",
                parameter.name()
            ),
        }
    }
}

impl Error for DatasetError {}

/// An error encountered while materializing or loading an exchange container.
///
/// A separate type from [`DatasetError`] rather than more variants on it,
/// because the two answer different questions: `DatasetError` says what was
/// wrong with a *request to produce data*, and these say what was wrong with a
/// *stored container* — a file that may have been written by another version,
/// truncated by a failed copy, or edited outright. A caller building recipes
/// should not have to consider a checksum mismatch, and a caller reading files
/// should not have to consider a prevalence out of range.
///
/// It carries [`std::io::Error`] and therefore derives neither `Clone` nor
/// `PartialEq`. The refusals are matched with `matches!` in this crate's own
/// tests, which is what a `#[non_exhaustive]` error is for.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExchangeError {
    /// The container name is not a single lower-case file stem.
    ///
    /// Names become file names, so anything that could reach outside the
    /// exchange directory — a separator, a parent reference, an empty
    /// string — is refused before a path is built rather than after.
    InvalidName,
    /// The filesystem refused an operation.
    Io {
        /// The path the operation was attempted on.
        path: PathBuf,
        /// What the filesystem reported.
        source: io::Error,
    },
    /// A container file is longer than this reader will read.
    ///
    /// Checked against the file's own length before it is read, so an
    /// oversized file is refused rather than loaded and then rejected.
    SizeLimitExceeded {
        /// Hard byte limit for this file.
        limit: usize,
        /// Length the file actually has.
        actual: u64,
    },
    /// The manifest is not the schema this crate writes.
    ///
    /// The reader accepts one field order, one set of keys, and no string
    /// escapes, so this covers a foreign manifest as much as a corrupt one.
    MalformedManifest {
        /// Byte offset the reader stopped at.
        offset: usize,
    },
    /// The manifest declares a container format this reader does not know.
    UnsupportedFormat {
        /// Format version read from the manifest.
        found: u64,
    },
    /// The recipe in the manifest does not hash to the digest recorded beside
    /// it.
    ///
    /// This is what makes the recipe in a manifest trustworthy: an edited
    /// recipe still hashes to something, and what makes the edit visible is
    /// that it no longer hashes to the value written with it. The recorded
    /// determinism envelope is checked the same way, because a container
    /// promising bit-exact bytes for a transcendental family would mislead a
    /// harness comparing two machines.
    SpecDigestMismatch,
    /// The array file does not hash to the digest the manifest recorded.
    DataChecksumMismatch,
    /// The array table does not describe the array file exactly.
    ///
    /// Entries must be contiguous, in order, consistent with their own shapes,
    /// uniquely named, and must end on the file's last byte. Anything else is
    /// a second encoding of the same data, which is what a canonical container
    /// format refuses.
    InvalidArrayTable,
    /// The manifest describes a recipe this crate refuses to construct.
    InvalidRecipe(
        /// Why the recipe was refused.
        DatasetError,
    ),
    /// The container holds a dataset its recipe does not produce, and was asked
    /// for as though it did.
    ///
    /// A derived container records a recipe as provenance and the *same* spec
    /// digest that recipe's own output would carry, so nothing about the digest
    /// distinguishes the two — which is exactly why this refusal exists rather
    /// than a silently different dataset. Both halves of a frozen conformance
    /// lane are the case: they are slices of one design carrying a lane's own
    /// targets, and regenerating their recipe would produce the whole design
    /// with no targets at all.
    NotRegenerable {
        /// What the container actually holds.
        derivation: super::exchange::Derivation,
    },
}

impl fmt::Display for ExchangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName => f.write_str(
                "a container name must be a non-empty stem of lower-case letters, digits, \
                 hyphens and underscores",
            ),
            Self::Io { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            Self::SizeLimitExceeded { limit, actual } => {
                write!(f, "container file size {actual} exceeds limit {limit}")
            }
            Self::MalformedManifest { offset } => {
                write!(f, "dataset manifest is malformed at byte {offset}")
            }
            Self::UnsupportedFormat { found } => {
                write!(f, "unsupported dataset container format {found}")
            }
            Self::SpecDigestMismatch => {
                f.write_str("the manifest's recipe does not match its recorded spec digest")
            }
            Self::DataChecksumMismatch => f.write_str("dataset array file checksum mismatch"),
            Self::InvalidArrayTable => {
                f.write_str("the array table does not describe the array file")
            }
            Self::InvalidRecipe(error) => {
                write!(f, "the manifest describes no valid recipe: {error}")
            }
            Self::NotRegenerable { derivation } => {
                let super::exchange::Derivation::ReferenceSplit { lane, seed, split } = derivation;
                write!(
                    f,
                    "this container holds the {} split of the {} reference lane at seed {seed}, \
                     which its recipe does not produce",
                    split.label(),
                    lane.label(),
                )
            }
        }
    }
}

impl Error for ExchangeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidRecipe(error) => Some(error),
            _ => None,
        }
    }
}
