use super::contamination::{Contamination, WeightPattern};
use super::dataset::{Dataset, Target, Truth};
use super::error::DatasetError;
use super::source::{LATTICE_MODULUS_LIMIT, fill_design};
use super::structural::{ClassBalance, ClassGeometry, GroupPattern};
use super::task::{BinaryKind, GlmLink, NonlinearKind, Portability, Task};
use crate::data::DenseMatrix;
use crate::numeric::derive_dataset_stream;
use sha2::{Digest, Sha256};

/// The domain tag every spec digest starts with.
///
/// It makes the digest a statement about a dataset recipe specifically, so the
/// same bytes reached through some other encoding in this crate cannot collide
/// with it, and it carries a version so a later phase can extend the encoding
/// without silently reusing a digest that meant something else.
///
/// **Version five, for the same reason version four was cut: a recipe's output
/// moved under an unchanged encoding.** The rule the tag was introduced for is
/// that one identifier must never mean two things, and the earliest bumps served
/// it by moving whenever the *encoding* moved. `v4` served it from the other
/// side, when the task dials left [`Recipe::stream_digest`]; `v5` serves it
/// again. [`BinaryKind::Sinusoid`](super::BinaryKind::Sinusoid) evaluates a
/// different expression than the boundary it replaced, so every recipe carrying
/// it draws different data under a byte layout that did not move. A `v4` digest
/// and a `v5` digest of one
/// recipe name two different datasets, so they must not be one number. Nothing
/// outside this crate has recorded any of them — the feature is unreleased — so
/// the bump costs a cache invalidation nobody can observe, and a stale
/// materialized container now refuses to load rather than being served as a hit.
const SPEC_DOMAIN: &[u8] = b"ferricml.dataset.spec.v5";

/// The domain tag the task families' auxiliary stream seeds are derived under.
///
/// Distinct from [`SPEC_DOMAIN`] so the two digests of one recipe can never
/// collide: they cover different fields and mean different things, and a stream
/// seed that happened to equal an identity would be a coincidence a reader would
/// have to rule out.
///
/// **Version two, because this encoding really did change**: the task's dials
/// are no longer hashed here. Nothing outside this file reads a stream digest,
/// so the bump invalidates nothing; it exists so the two version numbers cannot
/// be read as a claim that the stream encoding stood still while the spec
/// encoding moved. It is the other way around.
const STREAM_DOMAIN: &[u8] = b"ferricml.dataset.stream.v2";

/// Where a design matrix's numbers come from.
///
/// Three sources exist because three kinds of frozen output have to be
/// reproduced bit for bit, not because three kinds of randomness are
/// interesting. Each is transcendental-free and therefore exact on every
/// target; see the module documentation for what that buys.
///
/// ```
/// use ferricml::datasets::{Recipe, Source};
///
/// // The same lattice on two separate recipes is the same matrix.
/// let source = Source::Lattice { row_stride: 131, column_stride: 17, modulus: 1009 };
/// let left = Recipe::new(4, 3, source)?;
/// let right = Recipe::new(4, 3, source)?;
/// assert_eq!(left.design().as_slice(), right.design().as_slice());
/// # Ok::<(), ferricml::datasets::DatasetError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Source {
    /// A SplitMix64 stream started at a **raw** state.
    ///
    /// The state is the generator's own, not a mixed seed, and that is
    /// deliberate: FerricML's frozen reference design matrices were recorded
    /// against exactly this construction, so a preset that reproduces them has
    /// to name the raw state. Routing every recipe through a derivation would
    /// have moved every fixture the presets protect.
    ///
    /// A recipe built from a caller's *seed* rather than from a stream state
    /// uses [`Recipe::seeded`], which mixes the seed into a state that is
    /// disjoint from every estimator stream the same number reaches.
    Sampled {
        /// The raw SplitMix64 state the stream starts at.
        state: u64,
    },
    /// A periodic lattice of index arithmetic, with no generator at all.
    ///
    /// Cell `(row, column)` is `(row * row_stride + column * column_stride)`
    /// reduced modulo `modulus`, then mapped onto `[-1, 1)`. There is no state:
    /// any cell can be computed without producing the ones before it, which is
    /// what makes this source's cost independent of where in the matrix you
    /// look.
    ///
    /// Strides that share a factor with the modulus give a design with repeated
    /// columns or repeated rows, and that is a legitimate thing to ask for
    /// rather than an error — collinearity and duplication are properties a
    /// consumer may want to measure an estimator against.
    Lattice {
        /// Multiplier applied to the row index.
        row_stride: u64,
        /// Multiplier applied to the column index.
        column_stride: u64,
        /// Modulus the cell index is reduced by. At least `2`, and at most
        /// `2^24` so every residue is exact in `f32`.
        modulus: u64,
    },
    /// A 32-bit xorshift stream started at a raw state.
    ///
    /// This exists for the same reason [`Sampled`](Source::Sampled) takes a raw
    /// state: FerricML's benchmark fixtures were written against xorshift32
    /// streams, `bench-history` compares against immutable per-release results,
    /// and no SplitMix64 construction reproduces those draws.
    Xorshift32 {
        /// The raw xorshift32 state the stream starts at. Never zero, which is
        /// the generator's fixed point.
        state: u32,
    },
}

impl Source {
    /// This variant's discriminant in the spec digest.
    ///
    /// Written out rather than derived from declaration order, because a
    /// reordering of the variants must not silently restate what a recorded
    /// digest means.
    const fn digest_tag(self) -> u8 {
        match self {
            Self::Sampled { .. } => 1,
            Self::Lattice { .. } => 2,
            Self::Xorshift32 { .. } => 3,
        }
    }
}

/// A validated request for a synthetic dataset.
///
/// A recipe is the whole identity of the data it produces: two recipes that
/// compare equal generate identical bytes, on the same machine and on any
/// other, because every source is transcendental-free. Nothing about the
/// generated data depends on when it was generated, how many threads were
/// available, or whether it was generated before.
///
/// # Validation happens here, before anything is allocated
///
/// Shape and source parameters are checked by the constructor, so a `Recipe`
/// that exists describes a dataset that can be produced. [`Recipe::generate`]
/// and [`Recipe::design`] therefore return no error: there is no failure left
/// for them to report.
///
/// ```
/// use ferricml::datasets::{DatasetError, Recipe, Source};
///
/// // A refused shape never reaches an allocation.
/// assert_eq!(
///     Recipe::seeded(0, 4, 7),
///     Err(DatasetError::ZeroRows),
/// );
/// assert_eq!(
///     Recipe::new(4, 4, Source::Xorshift32 { state: 0 }),
///     Err(DatasetError::ZeroXorshiftState),
/// );
/// # Ok::<(), DatasetError>(())
/// ```
///
/// # A seed here is not a seed an estimator sees
///
/// ```
/// use ferricml::datasets::Recipe;
///
/// let recipe = Recipe::seeded(64, 8, 11)?;
/// let design = recipe.design();
/// assert_eq!(design.rows(), 64);
/// assert_eq!(design.columns(), 8);
///
/// // Reusing the buffer produces the same values without a second allocation.
/// let mut buffer = Vec::new();
/// recipe.design_into(&mut buffer);
/// assert_eq!(buffer, design.as_slice());
/// # Ok::<(), ferricml::datasets::DatasetError>(())
/// ```
///
/// [`Recipe::seeded`] mixes the caller's number into a stream state that is
/// disjoint from the stream a model fitted with that same number draws from. If
/// they were the same stream the design matrix would be correlated with the
/// model's own randomness, which is the exposure that makes a separate
/// generator worth having in the first place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe {
    rows: usize,
    columns: usize,
    source: Source,
    task: Option<Task>,
    contamination: Contamination,
    weight_pattern: Option<WeightPattern>,
    group_pattern: Option<GroupPattern>,
}

impl Recipe {
    /// Validates a shape and a source.
    ///
    /// Both dimensions must be non-zero and their product must fit in `usize`.
    /// A [`Source::Xorshift32`] state must be non-zero, and a
    /// [`Source::Lattice`] modulus must be at least `2` and at most `2^24`.
    pub fn new(rows: usize, columns: usize, source: Source) -> Result<Self, DatasetError> {
        if rows == 0 {
            return Err(DatasetError::ZeroRows);
        }
        if columns == 0 {
            return Err(DatasetError::ZeroColumns);
        }
        if rows.checked_mul(columns).is_none() {
            return Err(DatasetError::DimensionOverflow { rows, columns });
        }
        match source {
            Source::Sampled { .. } => {}
            Source::Lattice { modulus, .. } => {
                if modulus < 2 {
                    return Err(DatasetError::LatticeModulusTooSmall { modulus });
                }
                if modulus > LATTICE_MODULUS_LIMIT {
                    return Err(DatasetError::LatticeModulusNotRepresentable {
                        modulus,
                        limit: LATTICE_MODULUS_LIMIT,
                    });
                }
            }
            Source::Xorshift32 { state } => {
                if state == 0 {
                    return Err(DatasetError::ZeroXorshiftState);
                }
            }
        }
        Ok(Self {
            rows,
            columns,
            source,
            task: None,
            contamination: Contamination::none(),
            weight_pattern: None,
            group_pattern: None,
        })
    }

    /// Validates a shape and derives a sampled source from a caller's seed.
    ///
    /// This is the constructor for *new* recipes. The seed is mixed into a
    /// SplitMix64 state that is disjoint from every stream an estimator seeded
    /// with the same number reaches, so a design matrix generated from seed `s`
    /// is uncorrelated with a forest fitted with seed `s`.
    ///
    /// A recipe that must reproduce an already-recorded stream names its raw
    /// state through [`Source::Sampled`] and [`Recipe::new`] instead.
    pub fn seeded(rows: usize, columns: usize, seed: u64) -> Result<Self, DatasetError> {
        Self::new(
            rows,
            columns,
            Source::Sampled {
                state: derive_dataset_stream(seed),
            },
        )
    }

    /// Adds a task family, checking its parameters against the design's shape.
    ///
    /// Everything a task can get wrong is caught here, before a single value is
    /// generated: a coefficient scale that is not positive, an informative
    /// prefix wider than the design, a prevalence outside `(0, 1)`, a requested
    /// rank above the column count. The recipe that comes back generates without
    /// failing.
    ///
    /// The contamination and weight pattern already on the recipe are rechecked
    /// against the new task, because their admissibility depends on it: a
    /// class-balancing weight pattern needs labels, and a noise-shaping knob
    /// needs a target with additive noise. Order of the builder calls therefore
    /// does not change which recipes exist.
    ///
    /// ```
    /// use ferricml::datasets::{DatasetError, Recipe, Task};
    ///
    /// let task = Task::LinearRegression {
    ///     informative: 40,
    ///     coefficient_scale: 1.0,
    ///     intercept: 0.0,
    ///     noise_scale: 0.1,
    /// };
    /// assert_eq!(
    ///     Recipe::seeded(64, 8, 3)?.with_task(task),
    ///     Err(DatasetError::InformativeColumnsExceedDesign {
    ///         informative: 40,
    ///         columns: 8,
    ///     }),
    /// );
    /// # Ok::<(), DatasetError>(())
    /// ```
    pub fn with_task(mut self, task: Task) -> Result<Self, DatasetError> {
        task.validate(self.rows, self.columns)?;
        self.task = Some(task);
        self.check_task_dependent_requests()?;
        Ok(self)
    }

    /// Adds a contamination, checking its knobs against the design and the task.
    ///
    /// A knob whose effect the current task cannot carry is refused rather than
    /// ignored: a label-noise rate on a regression target, or a heavy tail on a
    /// count response, would otherwise leave a robustness sweep reporting a
    /// model as robust to a contamination it never received.
    pub fn with_contamination(
        mut self,
        contamination: Contamination,
    ) -> Result<Self, DatasetError> {
        contamination.validate(self.columns)?;
        self.contamination = contamination;
        self.check_task_dependent_requests()?;
        Ok(self)
    }

    /// Adds a per-row weight pattern.
    ///
    /// [`WeightPattern::ClassBalanced`] needs a task that draws labels, and is
    /// refused without one rather than degraded to uniform weights.
    pub fn with_weights(mut self, pattern: WeightPattern) -> Result<Self, DatasetError> {
        pattern.validate()?;
        self.weight_pattern = Some(pattern);
        self.check_task_dependent_requests()?;
        Ok(self)
    }

    /// Adds a group pattern, checking it against the design's row count.
    ///
    /// Group labels mark rows that are not independent, and every pattern
    /// **partitions** the rows: exactly `groups` identifiers, none of them
    /// unused, one per row. They are `u64` because that is what this crate's
    /// grouped splitters take, so
    /// [`Dataset::groups`](super::Dataset::groups) feeds
    /// [`GroupKFold::split`](crate::model_selection::GroupKFold::split) with no
    /// adapter between them.
    ///
    /// A pattern is refused over a task that assigns groups itself — today that
    /// is [`Task::Ranking`], whose group labels are its query identifiers and
    /// whose pairs are within-query by construction. Letting a pattern win would
    /// leave the pairs and the groups describing two different partitions of the
    /// same rows.
    ///
    /// ```
    /// use ferricml::datasets::{GroupPattern, Recipe};
    /// use ferricml::model_selection::GroupKFold;
    ///
    /// let dataset = Recipe::seeded(120, 4, 9)?
    ///     .with_groups(GroupPattern::Contiguous { groups: 12 })?
    ///     .generate();
    ///
    /// let groups = dataset.groups().expect("a grouped recipe");
    /// let folds = GroupKFold::new(4).split(groups)?.count();
    /// assert_eq!(folds, 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_groups(mut self, pattern: GroupPattern) -> Result<Self, DatasetError> {
        pattern.validate(self.rows)?;
        self.group_pattern = Some(pattern);
        self.check_task_dependent_requests()?;
        Ok(self)
    }

    /// Rejects any request the current task cannot carry.
    ///
    /// Run after every builder call rather than only after `with_task`, so the
    /// same set of recipes exists whichever order a caller builds them in.
    fn check_task_dependent_requests(&self) -> Result<(), DatasetError> {
        let draws_labels = self.task.is_some_and(|task| task.draws_labels());
        let additive_noise = self.task.is_some_and(|task| task.has_additive_noise());
        if self.contamination.label_noise() != 0.0 && !draws_labels {
            return Err(DatasetError::ContaminationNeedsLabels {
                parameter: super::error::Parameter::LabelNoise,
            });
        }
        if let Some(parameter) = self.contamination.shapes_noise() {
            // An outlier fraction survives a generalized linear response, which
            // displaces it multiplicatively so a count stays a count; the other
            // two reshape an additive term that response does not have.
            let carried = if matches!(parameter, super::error::Parameter::OutlierFraction) {
                self.task.is_some_and(Task::carries_outliers)
            } else {
                additive_noise
            };
            if !carried {
                return Err(DatasetError::ContaminationNeedsAdditiveNoise { parameter });
            }
        }
        if self.weight_pattern.is_some_and(WeightPattern::needs_labels) && !draws_labels {
            return Err(DatasetError::WeightPatternNeedsLabels);
        }
        if self.contamination.duplicate_rows() != 0.0
            && self.task.is_some_and(Task::truth_is_positional)
        {
            // The clustered family's recorded assignment is a function of the
            // row index, so a duplicated row would carry another cluster's
            // features under its own recorded label. A silently wrong ground
            // truth is worse than any contamination.
            return Err(DatasetError::ContaminationConflictsWithTask {
                parameter: super::error::Parameter::DuplicateRows,
            });
        }
        if self.group_pattern.is_some() && self.task.is_some_and(Task::assigns_groups) {
            return Err(DatasetError::GroupPatternConflictsWithTask);
        }
        Ok(())
    }

    /// Returns the number of rows the recipe produces.
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns the recipe produces.
    #[inline]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the source the design matrix is drawn from.
    #[inline]
    pub const fn source(&self) -> Source {
        self.source
    }

    /// Returns the task family drawn over the design, if any.
    #[inline]
    pub const fn task(&self) -> Option<Task> {
        self.task
    }

    /// Returns the contamination applied to the generated data.
    #[inline]
    pub const fn contamination(&self) -> Contamination {
        self.contamination
    }

    /// Returns the per-row weight pattern, if any.
    #[inline]
    pub const fn weight_pattern(&self) -> Option<WeightPattern> {
        self.weight_pattern
    }

    /// Returns the per-row group pattern, if any.
    ///
    /// `None` does not mean the generated dataset has no groups: a task that
    /// assigns its own — [`Task::Ranking`] — reports none here and still
    /// produces [`Dataset::groups`](super::Dataset::groups).
    #[inline]
    pub const fn group_pattern(&self) -> Option<GroupPattern> {
        self.group_pattern
    }

    /// Returns this recipe's determinism envelope.
    ///
    /// The weaker of the task's and the contamination's: one transcendental
    /// anywhere on the path makes the whole dataset per-runner, and a recipe
    /// with neither a task nor a scale spread is bit-exact because every source
    /// is.
    ///
    /// ```
    /// use ferricml::datasets::{Contamination, Portability, Recipe, Task};
    ///
    /// let bare = Recipe::seeded(32, 4, 1)?;
    /// assert_eq!(bare.portability(), Portability::BitExact);
    ///
    /// // A scale spread is a real power of ten, and says so.
    /// let spread = bare.with_contamination(
    ///     Contamination::none().with_feature_scale_spread(3.0),
    /// )?;
    /// assert_eq!(spread.portability(), Portability::PerRunner);
    ///
    /// // As does a logistic Bayes probability.
    /// let logistic = bare.with_task(Task::LinearBinary {
    ///     informative: 2,
    ///     separation: 2.0,
    ///     prevalence: 0.3,
    /// })?;
    /// assert_eq!(logistic.portability(), Portability::PerRunner);
    /// # Ok::<(), ferricml::datasets::DatasetError>(())
    /// ```
    pub fn portability(&self) -> Portability {
        let task = self
            .task
            .map_or(Portability::BitExact, |task| task.portability());
        task.combine(self.contamination.portability())
    }

    /// Returns a digest of everything that determines this recipe's output.
    ///
    /// Two recipes share a digest exactly when they compare equal, and equal
    /// recipes produce identical bytes — which is what makes the digest usable
    /// as a cache key for materialized data, and as the thing a stored dataset
    /// is checked against before it is trusted to be what its name says.
    ///
    /// The encoding is injective rather than merely conventional: a fixed domain
    /// tag, then the shape, then a one-byte source discriminant that determines
    /// exactly how many `u64` fields follow it. No field is variable-length, so
    /// no two distinct recipes can produce the same byte string to hash.
    ///
    /// ```
    /// use ferricml::datasets::{Recipe, Source};
    ///
    /// let seeded = Recipe::seeded(16, 4, 11)?;
    /// assert_eq!(seeded.spec_digest(), Recipe::seeded(16, 4, 11)?.spec_digest());
    /// assert_ne!(seeded.spec_digest(), Recipe::seeded(16, 4, 12)?.spec_digest());
    /// // The shape is part of the identity, not only the stream.
    /// assert_ne!(seeded.spec_digest(), Recipe::seeded(4, 16, 11)?.spec_digest());
    /// // And so is which source produced it.
    /// assert_ne!(
    ///     Recipe::new(16, 4, Source::Sampled { state: 7 })?.spec_digest(),
    ///     Recipe::new(16, 4, Source::Xorshift32 { state: 7 })?.spec_digest(),
    /// );
    /// # Ok::<(), ferricml::datasets::DatasetError>(())
    /// ```
    pub fn spec_digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(SPEC_DOMAIN);
        digest.update((self.rows as u64).to_le_bytes());
        digest.update((self.columns as u64).to_le_bytes());
        digest.update([self.source.digest_tag()]);
        match self.source {
            Source::Sampled { state } => digest.update(state.to_le_bytes()),
            Source::Lattice {
                row_stride,
                column_stride,
                modulus,
            } => {
                digest.update(row_stride.to_le_bytes());
                digest.update(column_stride.to_le_bytes());
                digest.update(modulus.to_le_bytes());
            }
            Source::Xorshift32 { state } => digest.update(u64::from(state).to_le_bytes()),
        }
        encode_task(self.task, &mut TaskFields::whole(&mut digest));
        self.update_contamination_digest(&mut digest);
        self.update_weight_digest(&mut digest);
        self.update_group_digest(&mut digest);
        digest.finalize().into()
    }

    /// The digest the task families seed their auxiliary streams from.
    ///
    /// Shape, source, and the task's **structural** fields — and deliberately
    /// not the task's dials, not the contamination, and not the weights. Those
    /// exclusions are what make a knob an overlay rather than a reseed: two
    /// recipes differing only in an excluded field draw the same coefficients,
    /// the same noise, the same clean labels and the same per-row selectors, so
    /// switching a knob changes exactly what the knob describes and nothing
    /// else.
    ///
    /// It is not a hypothetical, and it has now been measured twice.
    ///
    /// Seeding the streams from the full spec digest made a five-percent
    /// label-noise request flip fifty-six percent of the labels, because the
    /// "clean" and "contaminated" datasets were two independent draws and the
    /// measured difference was the draw rather than the noise. A robustness
    /// sweep built on that would have compared a model against a different
    /// problem at every contamination level and called the difference
    /// sensitivity. That is why the contamination sits outside this digest.
    ///
    /// The task was left on the *other* side of that exclusion and the identical
    /// failure followed. With every task field hashed here, a difficulty sweep
    /// redrew the coefficients at every step, so the measured difference was
    /// again the draw rather than the knob. Bayes accuracy over
    /// [`Task::LinearBinary`] at `20000 x 8`, seed `31`, four informative
    /// columns, separation `0.9 / 1.0 / 1.1`, read `0.6198 / 0.5543 / 0.6707` —
    /// non-monotone in the one parameter whose whole purpose is monotonicity,
    /// with a step-to-step reversal larger than the knob's own effect across the
    /// interval. [`Task::Multiclass`] at the same shape read
    /// `0.5916 / 0.6411 / 0.6024`. Both ladders are swept as assertions now, in
    /// `family_tests.rs` and `structural_tests.rs`.
    ///
    /// # Which fields are which, and why the compiler decides
    ///
    /// A field is **structural** when it changes *what is drawn*: the support of
    /// the true coefficient vector, the number of classes or clusters or
    /// queries, the shape of a boundary, the algebraic rank. A redraw is the
    /// honest answer there — the two recipes describe two different problems,
    /// and holding one stream across them would only make the difference harder
    /// to read.
    ///
    /// A field is a **dial** when it modulates a *fixed* draw: how steeply, how
    /// noisily, how imbalanced, how ill-conditioned. Two recipes differing in a
    /// dial are one problem at two settings, and a sweep over one is
    /// interpretable only while the problem underneath it holds still.
    ///
    /// Two dials do reach the design matrix — [`Task::IllConditioned`]'s
    /// `condition_number` scales its columns and [`Task::Clustered`]'s `spread`
    /// moves its rows toward their centres — and they are dials all the same,
    /// because each applies a closed-form transform to a draw that is itself
    /// unchanged. Their evidence is stated as that transform rather than as byte
    /// identity.
    ///
    /// A partition maintained by hand rots, so this one is not maintained by
    /// hand. [`encode_task`] destructures every variant with no `..` rest
    /// pattern, so a new task field does not compile until the pattern names it
    /// (`E0027`), and naming it without routing it through
    /// [`TaskFields`] leaves an `unused_variables` warning the crate's clippy
    /// gate denies. One encoding serves both digests, so a field is classified
    /// exactly once or not at all.
    ///
    /// The full [`Recipe::spec_digest`] remains the recipe's identity: it is
    /// what a cache keys on and what a materialized dataset is checked against,
    /// and it must move when a dial moves, because the data does.
    ///
    /// Visible to the sibling test modules rather than private to this file,
    /// because "these two recipes share a stream" is the invariant the whole
    /// partition exists to provide and asserting it directly is stronger
    /// evidence than any consequence of it: a per-dial sweep can compare the
    /// digests themselves instead of inferring stream identity from bytes that
    /// happened to match.
    pub(super) fn stream_digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(STREAM_DOMAIN);
        digest.update((self.rows as u64).to_le_bytes());
        digest.update((self.columns as u64).to_le_bytes());
        digest.update([self.source.digest_tag()]);
        match self.source {
            Source::Sampled { state } => digest.update(state.to_le_bytes()),
            Source::Lattice {
                row_stride,
                column_stride,
                modulus,
            } => {
                digest.update(row_stride.to_le_bytes());
                digest.update(column_stride.to_le_bytes());
                digest.update(modulus.to_le_bytes());
            }
            Source::Xorshift32 { state } => digest.update(u64::from(state).to_le_bytes()),
        }
        encode_task(self.task, &mut TaskFields::structure_only(&mut digest));
        digest.finalize().into()
    }

    /// Hashes the contamination. Fixed width, so no discriminant is needed.
    fn update_contamination_digest(&self, digest: &mut Sha256) {
        let contamination = self.contamination;
        for value in [
            contamination.label_noise(),
            contamination.outlier_fraction(),
            contamination.heavy_tail(),
            contamination.heteroscedastic(),
            contamination.duplicate_rows(),
            contamination.feature_scale_spread(),
        ] {
            digest.update(value.to_bits().to_le_bytes());
        }
        digest.update((contamination.constant_columns() as u64).to_le_bytes());
        digest.update((contamination.collinear_pairs() as u64).to_le_bytes());
    }

    /// Hashes the weight pattern: a discriminant and two floats, always.
    ///
    /// The unused floats are zeroed rather than omitted, so the field is fixed
    /// width and the tag alone separates the patterns.
    fn update_weight_digest(&self, digest: &mut Sha256) {
        let (tag, first, second) = match self.weight_pattern {
            None => (0_u8, 0.0_f32, 0.0_f32),
            Some(WeightPattern::Uniform) => (1, 0.0, 0.0),
            Some(WeightPattern::Ramp { low, high }) => (2, low, high),
            Some(WeightPattern::Alternating { first, second }) => (3, first, second),
            Some(WeightPattern::ClassBalanced) => (4, 0.0, 0.0),
        };
        digest.update([tag]);
        digest.update(first.to_bits().to_le_bytes());
        digest.update(second.to_bits().to_le_bytes());
    }

    /// Hashes the group pattern: a discriminant, a count and one float, always.
    ///
    /// Fixed width for the reason the weight pattern is: the unused ratio is
    /// zeroed rather than omitted, so the tag alone separates the patterns and
    /// the encoding stays injective.
    fn update_group_digest(&self, digest: &mut Sha256) {
        let (tag, groups, ratio) = match self.group_pattern {
            None => (0_u8, 0_usize, 0.0_f32),
            Some(GroupPattern::RoundRobin { groups }) => (1, groups, 0.0),
            Some(GroupPattern::Contiguous { groups }) => (2, groups, 0.0),
            Some(GroupPattern::Unbalanced { groups, ratio }) => (3, groups, ratio),
        };
        digest.update([tag]);
        digest.update((groups as u64).to_le_bytes());
        digest.update(ratio.to_bits().to_le_bytes());
    }

    /// Generates the design matrix.
    ///
    /// Allocates the result. [`Recipe::design_into`] is the same values written
    /// into a buffer the caller already owns.
    ///
    /// # Panics
    ///
    /// Never, for any `Recipe` that exists. The matrix is handed to
    /// [`DenseMatrix::new`] rather than built past its validation, so the
    /// generated values are checked finite exactly as a caller's would be; every
    /// source here maps an integer through a bounded affine expression and
    /// cannot produce anything else, so a failure would be a defect in this
    /// crate rather than an input to reject. The scan is one pass of
    /// `is_finite` against a generation loop that does strictly more work per
    /// element, and it is what keeps this module an ordinary consumer of the
    /// data containers rather than a privileged producer.
    pub fn design(&self) -> DenseMatrix {
        let mut values = Vec::new();
        self.design_into(&mut values);
        DenseMatrix::new(values, self.rows, self.columns)
            .expect("a validated recipe generates a finite matrix of its own shape")
    }

    /// Writes the design matrix into a caller-owned buffer.
    ///
    /// `values` is cleared and refilled with `rows * columns` entries in
    /// row-major order, reusing its allocation when it is large enough. This is
    /// the form a sweep over many recipes uses: the buffer is allocated once and
    /// the generator writes through it, so generating `n` datasets costs `n`
    /// fills rather than `n` allocations.
    ///
    /// The task's own reshaping and the contamination's are part of the design
    /// and happen here, in that order: an ill-conditioned design *is* the
    /// conditioned matrix, and a constant column is a property of the data a
    /// consumer receives. Both are no-ops for a recipe that asked for neither,
    /// which is what keeps the absorbed reference and benchmark streams
    /// byte-identical to the ones this module froze before task families
    /// existed.
    pub fn design_into(&self, values: &mut Vec<f32>) {
        fill_design(&self.source, self.rows, self.columns, values);
        if let Some(task) = self.task {
            // The digest is formed only for a recipe that carries a task, and
            // only one of them reads it. A SHA-256 over a few dozen fixed-width
            // bytes is a fixed cost against a fill that is already `rows *
            // columns` draws, so it does not change what this method is for.
            task.shape_design(self.rows, self.columns, values, &self.stream_digest());
        }
        self.contamination
            .shape_design(self.rows, self.columns, values);
    }

    /// Generates the dataset this recipe describes.
    ///
    /// A recipe carrying no task family produces a design matrix, its recorded
    /// spec digest, and [`Truth::DesignOnly`] — there is nothing to be right
    /// about until a family assigns targets, and saying so with a variant is
    /// more honest than shipping an empty coefficient vector.
    ///
    /// ```
    /// use ferricml::datasets::{Recipe, Task};
    ///
    /// let dataset = Recipe::seeded(512, 6, 17)?
    ///     .with_task(Task::LinearBinary {
    ///         informative: 4,
    ///         separation: 3.0,
    ///         prevalence: 0.2,
    ///     })?
    ///     .generate();
    ///
    /// // The Bayes probabilities average to the requested prevalence, because
    /// // the intercept was solved for rather than guessed.
    /// let probabilities = dataset.truth().probabilities().expect("a binary family");
    /// let mean: f64 = probabilities.iter().map(|&p| f64::from(p)).sum::<f64>()
    ///     / probabilities.len() as f64;
    /// assert!((mean - 0.2).abs() < 1e-6, "mean probability was {mean}");
    /// # Ok::<(), ferricml::datasets::DatasetError>(())
    /// ```
    pub fn generate(&self) -> Dataset {
        let design = self.design();
        let spec_digest = self.spec_digest();
        let drawn = self
            .task
            .map(|task| task.draw(&design, &self.contamination, &self.stream_digest()));
        let (target, truth, task_groups, pairs) = match drawn {
            None => (None, Truth::DesignOnly, None, None),
            Some(drawn) => (drawn.target, drawn.truth, drawn.groups, drawn.pairs),
        };
        let weights = self.weight_pattern.map(|pattern| {
            let labels = match target.as_ref() {
                Some(Target::Binary(targets)) => Some(targets.as_slice()),
                Some(Target::Class(targets)) => Some(targets.as_slice()),
                _ => None,
            };
            pattern.weights(self.rows, labels)
        });
        // A task's own grouping and a caller's pattern never both exist: the
        // combination is refused at the constructor, so this is a choice between
        // one source and none rather than a precedence rule.
        let groups = task_groups.or_else(|| self.group_pattern.map(|p| p.labels(self.rows)));
        Dataset::from_parts(design, target, weights, truth, groups, pairs, spec_digest)
    }

    /// The task's targets as one value per row, allocated.
    ///
    /// `None` when the recipe carries no task. A classification task reports
    /// `0.0` and `1.0`: this is the numeric view, and
    /// [`Dataset::target`](super::Dataset::target) is the validated-container
    /// view that keeps the vocabularies apart.
    ///
    /// ```
    /// use ferricml::datasets::{NonlinearKind, Recipe, Task};
    ///
    /// let recipe = Recipe::seeded(128, 6, 4)?.with_task(Task::NonlinearRegression {
    ///     kind: NonlinearKind::Interaction,
    ///     noise_scale: 0.1,
    /// })?;
    ///
    /// // The caller-owned form writes the same values without allocating.
    /// let mut buffer = Vec::new();
    /// recipe.target_values_into(&mut buffer);
    /// assert_eq!(buffer, recipe.target_values().unwrap());
    ///
    /// // A recipe with no task has no targets, and the buffer is left empty.
    /// let bare = Recipe::seeded(128, 6, 4)?;
    /// bare.target_values_into(&mut buffer);
    /// assert!(buffer.is_empty());
    /// assert_eq!(bare.target_values(), None);
    /// # Ok::<(), ferricml::datasets::DatasetError>(())
    /// ```
    pub fn target_values(&self) -> Option<Vec<f32>> {
        let task = self.task?;
        // An unsupervised family carries a task and still has no targets, so
        // "has a task" stopped being the same question as "has targets" when
        // `Task::Clustered` arrived.
        if !task.draws_target() {
            return None;
        }
        let mut values = Vec::new();
        self.target_values_into(&mut values);
        Some(values)
    }

    /// Writes the task's targets into a caller-owned buffer, one value per row.
    ///
    /// The buffer is cleared and refilled, reusing its allocation, and is left
    /// empty when the recipe carries no task. This is the form a sweep uses when
    /// it wants the numbers rather than the containers.
    pub fn target_values_into(&self, values: &mut Vec<f32>) {
        values.clear();
        if !self.task.is_some_and(Task::draws_target) {
            return;
        }
        values.reserve(self.rows);
        match self.generate().target() {
            Some(Target::Binary(targets)) => {
                values.extend(targets.as_slice().iter().map(|&label| f32::from(label)));
            }
            Some(Target::Class(targets)) => {
                values.extend(targets.as_slice().iter().map(|&label| f32::from(label)));
            }
            Some(Target::Regression(targets)) => values.extend_from_slice(targets.as_slice()),
            None => {}
        }
    }
}

/// A task's fields on their way into a digest, tagged by the role that decides
/// which digests see them.
///
/// One sink with two settings rather than two encoders, because the whole
/// content of the partition is that each field is classified *once*. Two
/// encoders would be two lists to keep in step, which is the arrangement
/// [`Recipe::stream_digest`] explains this crate has already been burned by.
///
/// [`TaskFields::whole`] hashes everything and is what [`Recipe::spec_digest`]
/// uses; [`TaskFields::structure_only`] drops the dials and is what
/// [`Recipe::stream_digest`] uses. Dropping rather than zeroing is deliberate:
/// the variant tag still determines exactly how many fixed-width fields follow
/// it, so the shortened encoding is injective on its own terms, and a zeroed
/// placeholder would only invite a reader to think the value was meaningful.
struct TaskFields<'a> {
    digest: &'a mut Sha256,
    /// Whether the dials reach the digest.
    dials: bool,
    /// How many fields have been classified, by role.
    ///
    /// Test-only, and it is the second half of the enforcement: the per-dial
    /// sweep in `tests.rs` reads these counts and requires its own table to
    /// cover exactly that many fields for each family. A field that is
    /// classified here and never swept is then a red test rather than a silent
    /// gap, which matters because the compiler can insist a field be classified
    /// and cannot insist the classification be *right*.
    #[cfg(test)]
    counts: FieldCounts,
}

/// How many task fields fall on each side of the partition.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FieldCounts {
    /// Fields in the stream digest.
    pub(super) structural: usize,
    /// Fields excluded from it.
    pub(super) dials: usize,
}

impl<'a> TaskFields<'a> {
    /// The encoding that carries every field: a recipe's identity.
    fn whole(digest: &'a mut Sha256) -> Self {
        Self {
            digest,
            dials: true,
            #[cfg(test)]
            counts: FieldCounts::default(),
        }
    }

    /// The encoding that carries only the structural fields: a recipe's stream.
    fn structure_only(digest: &'a mut Sha256) -> Self {
        Self {
            digest,
            dials: false,
            #[cfg(test)]
            counts: FieldCounts::default(),
        }
    }

    /// The variant's own discriminant, which is not a field of it.
    ///
    /// Separate from [`TaskFields::structural_tag`] so the counts below are a
    /// count of *fields*: the discriminant is what fixes the layout the fields
    /// are read under, and counting it would make every family look one field
    /// wider than it is.
    fn variant(&mut self, tag: u8) {
        self.digest.update([tag]);
    }

    /// A field that is a discriminant — a kind, a link, a geometry — naming
    /// which expression is evaluated at all.
    fn structural_tag(&mut self, tag: u8) {
        #[cfg(test)]
        {
            self.counts.structural += 1;
        }
        self.digest.update([tag]);
    }

    /// A count that changes how many things are drawn, or which of them matter.
    fn structural_count(&mut self, value: usize) {
        #[cfg(test)]
        {
            self.counts.structural += 1;
        }
        self.digest.update((value as u64).to_le_bytes());
    }

    /// A continuous knob applied to a draw that is already fixed.
    ///
    /// Hashed as its bits rather than as a formatting, because two distinct
    /// values must never reach the same digest and a decimal rendering does not
    /// promise that.
    fn dial(&mut self, value: f32) {
        #[cfg(test)]
        {
            self.counts.dials += 1;
        }
        if self.dials {
            self.digest.update(value.to_bits().to_le_bytes());
        }
    }

    /// The one dial that is an enum: a class balance.
    ///
    /// Both of its variants read the same drawn scores and differ only in the
    /// marginals solved onto them, so the balance is a dial whole — tag and
    /// ratio together, one field, hashed in the two pieces the pre-partition
    /// encoding wrote them as. Splitting the write across this method and
    /// [`TaskFields::dial_balance_ratio`] exists only to keep those two pieces
    /// at the byte offsets they already occupied, with the geometry's
    /// discriminant between them; only this half counts the field, because there
    /// is one field.
    fn dial_balance_tag(&mut self, balance: ClassBalance) {
        #[cfg(test)]
        {
            self.counts.dials += 1;
        }
        if self.dials {
            let tag = match balance {
                ClassBalance::Balanced => 1_u8,
                ClassBalance::Imbalanced { .. } => 2,
            };
            self.digest.update([tag]);
        }
    }

    /// The ratio half of the balance dial counted by
    /// [`TaskFields::dial_balance_tag`].
    fn dial_balance_ratio(&mut self, balance: ClassBalance) {
        if self.dials {
            let ratio = match balance {
                ClassBalance::Balanced => 1.0_f32,
                ClassBalance::Imbalanced { ratio } => ratio,
            };
            self.digest.update(ratio.to_bits().to_le_bytes());
        }
    }
}

/// How many fields of a task fall on each side of the partition.
///
/// Reads the classification out of [`encode_task`] itself rather than restating
/// it, so the sweep that consumes it cannot drift from the encoder it is
/// checking.
#[cfg(test)]
pub(super) fn task_field_counts(task: Task) -> FieldCounts {
    let mut digest = Sha256::new();
    let mut fields = TaskFields::whole(&mut digest);
    encode_task(Some(task), &mut fields);
    fields.counts
}

/// Writes a task's fields into a digest, each through the role it was
/// classified under.
///
/// Zero means no task, and every other tag determines exactly how many
/// fixed-width fields follow it, so both encodings stay injective as the variant
/// list grows.
///
/// **Every arm destructures its variant completely, with no `..`.** That is the
/// enforcement mechanism [`Recipe::stream_digest`] describes: a task field added
/// without a classification fails to compile here, rather than defaulting into
/// one of the two roles and being noticed by whichever measurement it later
/// spoils. The bytes this emits under [`TaskFields::whole`] are exactly the
/// bytes the pre-partition encoding emitted, field for field and in the same
/// order — the spec digest's *layout* did not move in this change, only what the
/// data behind it is.
fn encode_task(task: Option<Task>, fields: &mut TaskFields<'_>) {
    let Some(task) = task else {
        fields.variant(0);
        return;
    };
    match task {
        Task::LinearRegression {
            informative,
            coefficient_scale,
            intercept,
            noise_scale,
        } => {
            fields.variant(1);
            fields.structural_count(informative);
            fields.dial(coefficient_scale);
            fields.dial(intercept);
            fields.dial(noise_scale);
        }
        Task::NonlinearRegression { kind, noise_scale } => {
            fields.variant(2);
            fields.structural_tag(nonlinear_tag(kind));
            fields.dial(noise_scale);
        }
        Task::GlmRegression {
            link,
            informative,
            coefficient_scale,
            intercept,
            dispersion,
        } => {
            fields.variant(3);
            // The link decides which response is drawn — a count or a positive
            // continuous value — so it is structural even though `dispersion`,
            // which shapes that response's scatter, is not.
            fields.structural_tag(link_tag(link));
            fields.structural_count(informative);
            fields.dial(coefficient_scale);
            fields.dial(intercept);
            fields.dial(dispersion);
        }
        Task::IllConditioned {
            condition_number,
            rank,
            coefficient_scale,
            noise_scale,
        } => {
            fields.variant(4);
            // A dial that reaches the design: the columns are scaled after they
            // are drawn, so the draw underneath a conditioning sweep holds.
            fields.dial(condition_number);
            // The rank is not, because duplicating a column replaces drawn
            // values with copies rather than transforming them.
            fields.structural_count(rank);
            fields.dial(coefficient_scale);
            fields.dial(noise_scale);
        }
        Task::LinearBinary {
            informative,
            separation,
            prevalence,
        } => {
            fields.variant(5);
            fields.structural_count(informative);
            fields.dial(separation);
            fields.dial(prevalence);
        }
        Task::NonlinearBinary {
            kind,
            separation,
            prevalence,
        } => {
            fields.variant(6);
            fields.structural_tag(binary_tag(kind));
            fields.dial(separation);
            fields.dial(prevalence);
        }
        Task::Multiclass {
            classes,
            balance,
            geometry,
            separation,
        } => {
            fields.variant(7);
            // The balance's two pieces straddle the geometry's discriminant,
            // which is what the pre-partition encoding did and is why they are
            // written through two methods rather than one. It is a single dial.
            fields.dial_balance_tag(balance);
            fields.structural_tag(geometry_tag(geometry));
            fields.structural_count(classes);
            fields.dial_balance_ratio(balance);
            fields.dial(separation);
        }
        Task::Clustered { blobs, spread } => {
            fields.variant(8);
            fields.structural_count(blobs);
            // The second dial that reaches the design. The centres and the
            // per-row scatter are both already drawn; the spread only decides
            // how much of the scatter survives the move onto the centre.
            fields.dial(spread);
        }
        Task::TimeOrdered {
            informative,
            coefficient_scale,
            drift,
            intercept,
            noise_scale,
        } => {
            fields.variant(9);
            fields.structural_count(informative);
            fields.dial(coefficient_scale);
            fields.dial(drift);
            fields.dial(intercept);
            fields.dial(noise_scale);
        }
        Task::Ranking {
            queries,
            docs_per_query,
            grades,
            informative,
            coefficient_scale,
        } => {
            fields.variant(10);
            fields.structural_count(queries);
            fields.structural_count(docs_per_query);
            fields.structural_count(grades);
            fields.structural_count(informative);
            // The utilities scale linearly with it, so the within-query order,
            // the grades and the pairs are all unchanged and only the recorded
            // utilities move — the clearest dial in the enum.
            fields.dial(coefficient_scale);
        }
    }
}

/// The digest discriminant of a nonlinear shape.
///
/// Written out rather than derived from declaration order, for the reason
/// [`Source::digest_tag`] is: reordering the variants must not restate what a
/// recorded digest means.
const fn nonlinear_tag(kind: NonlinearKind) -> u8 {
    match kind {
        NonlinearKind::Interaction => 1,
        NonlinearKind::Piecewise => 2,
        NonlinearKind::Sinusoid => 3,
        NonlinearKind::Friedman => 4,
    }
}

/// The digest discriminant of a nonlinear binary boundary.
const fn binary_tag(kind: BinaryKind) -> u8 {
    match kind {
        BinaryKind::Xor => 1,
        BinaryKind::Sinusoid => 2,
        BinaryKind::Circles => 3,
        BinaryKind::Checkerboard => 4,
    }
}

/// The digest discriminant of a generalized linear link.
const fn link_tag(link: GlmLink) -> u8 {
    match link {
        GlmLink::LogCount => 1,
        GlmLink::LogPositive => 2,
    }
}

/// The digest discriminant of a multiclass geometry.
const fn geometry_tag(geometry: ClassGeometry) -> u8 {
    match geometry {
        ClassGeometry::Blob => 1,
        ClassGeometry::Hierarchical => 2,
    }
}
