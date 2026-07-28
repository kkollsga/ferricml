use super::contamination::{Contamination, WeightPattern};
use super::dataset::{Dataset, Target, Truth};
use super::error::DatasetError;
use super::source::{LATTICE_MODULUS_LIMIT, fill_design};
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
/// **Version two, because this phase extended the encoding.** A recipe now
/// carries a task, a contamination and a weight pattern, and every one of them
/// changes the data. The version moved rather than the fields being appended
/// silently, which is the discipline the tag was introduced for: a `v1` digest
/// described a recipe that could not express a task at all, so reusing it for a
/// recipe that can would make one identifier mean two things. Nothing outside
/// this crate has recorded a `v1` digest — the feature is unreleased — so the
/// bump costs a cache invalidation nobody can observe.
const SPEC_DOMAIN: &[u8] = b"ferricml.dataset.spec.v2";

/// The domain tag the task families' auxiliary stream seeds are derived under.
///
/// Distinct from [`SPEC_DOMAIN`] so the two digests of one recipe can never
/// collide: they cover different fields and mean different things, and a stream
/// seed that happened to equal an identity would be a coincidence a reader would
/// have to rule out.
const STREAM_DOMAIN: &[u8] = b"ferricml.dataset.stream.v1";

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
        task.validate(self.columns)?;
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
                self.task.is_some_and(|task| !task.draws_labels())
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
        self.update_task_digest(&mut digest);
        self.update_contamination_digest(&mut digest);
        self.update_weight_digest(&mut digest);
        digest.finalize().into()
    }

    /// The digest the task families seed their auxiliary streams from.
    ///
    /// Shape, source and task — and deliberately **not** contamination or
    /// weights. That exclusion is what makes contamination an overlay rather
    /// than a reseed: two recipes differing only in their contamination draw the
    /// same coefficients, the same noise, the same clean labels and the same
    /// per-row selectors, so switching a knob on changes exactly what the knob
    /// describes and nothing else.
    ///
    /// It is not a hypothetical. Seeding the streams from the full spec digest
    /// made a five-percent label-noise request flip fifty-six percent of the
    /// labels, because the "clean" and "contaminated" datasets were two
    /// independent draws and the measured difference was the draw rather than
    /// the noise. A robustness sweep built on that would have compared a model
    /// against a different problem at every contamination level and called the
    /// difference sensitivity.
    ///
    /// The full [`Recipe::spec_digest`] remains the recipe's identity: it is
    /// what a cache keys on and what a materialized dataset is checked against,
    /// and it must move when the contamination moves, because the data does.
    fn stream_digest(&self) -> [u8; 32] {
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
        self.update_task_digest(&mut digest);
        digest.finalize().into()
    }

    /// Hashes the task: a discriminant that fixes the field layout, then the
    /// fields.
    ///
    /// Zero means no task, and every other tag determines exactly how many
    /// fixed-width fields follow, so the encoding stays injective as the variant
    /// list grows. Floats are hashed as their bits rather than as a formatting,
    /// because two distinct values must never reach the same digest and a
    /// decimal rendering does not promise that.
    fn update_task_digest(&self, digest: &mut Sha256) {
        let Some(task) = self.task else {
            digest.update([0_u8]);
            return;
        };
        match task {
            Task::LinearRegression {
                informative,
                coefficient_scale,
                intercept,
                noise_scale,
            } => {
                digest.update([1_u8]);
                digest.update((informative as u64).to_le_bytes());
                digest.update(coefficient_scale.to_bits().to_le_bytes());
                digest.update(intercept.to_bits().to_le_bytes());
                digest.update(noise_scale.to_bits().to_le_bytes());
            }
            Task::NonlinearRegression { kind, noise_scale } => {
                digest.update([2_u8, nonlinear_tag(kind)]);
                digest.update(noise_scale.to_bits().to_le_bytes());
            }
            Task::GlmRegression {
                link,
                informative,
                coefficient_scale,
                intercept,
                dispersion,
            } => {
                digest.update([3_u8, link_tag(link)]);
                digest.update((informative as u64).to_le_bytes());
                digest.update(coefficient_scale.to_bits().to_le_bytes());
                digest.update(intercept.to_bits().to_le_bytes());
                digest.update(dispersion.to_bits().to_le_bytes());
            }
            Task::IllConditioned {
                condition_number,
                rank,
                coefficient_scale,
                noise_scale,
            } => {
                digest.update([4_u8]);
                digest.update(condition_number.to_bits().to_le_bytes());
                digest.update((rank as u64).to_le_bytes());
                digest.update(coefficient_scale.to_bits().to_le_bytes());
                digest.update(noise_scale.to_bits().to_le_bytes());
            }
            Task::LinearBinary {
                informative,
                separation,
                prevalence,
            } => {
                digest.update([5_u8]);
                digest.update((informative as u64).to_le_bytes());
                digest.update(separation.to_bits().to_le_bytes());
                digest.update(prevalence.to_bits().to_le_bytes());
            }
            Task::NonlinearBinary {
                kind,
                separation,
                prevalence,
            } => {
                digest.update([6_u8, binary_tag(kind)]);
                digest.update(separation.to_bits().to_le_bytes());
                digest.update(prevalence.to_bits().to_le_bytes());
            }
        }
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
            task.shape_design(self.rows, self.columns, values);
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
        let (target, truth) = match self.task {
            None => (None, Truth::DesignOnly),
            Some(task) => {
                let (target, truth) =
                    task.draw(&design, &self.contamination, &self.stream_digest());
                (Some(target), truth)
            }
        };
        let weights = self.weight_pattern.map(|pattern| {
            let labels = match target.as_ref() {
                Some(Target::Binary(targets)) => Some(targets.as_slice()),
                Some(Target::Class(targets)) => Some(targets.as_slice()),
                _ => None,
            };
            pattern.weights(self.rows, labels)
        });
        Dataset::from_parts(design, target, weights, truth, None, spec_digest)
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
        self.task?;
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
        if self.task.is_none() {
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
        BinaryKind::Moons => 2,
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
