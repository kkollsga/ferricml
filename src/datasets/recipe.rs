use super::dataset::{Dataset, Truth};
use super::error::DatasetError;
use super::source::{LATTICE_MODULUS_LIMIT, fill_design};
use crate::data::DenseMatrix;
use crate::numeric::derive_dataset_stream;
use sha2::{Digest, Sha256};

/// The domain tag every spec digest starts with.
///
/// It makes the digest a statement about a dataset recipe specifically, so the
/// same bytes reached through some other encoding in this crate cannot collide
/// with it, and it carries a version so a later phase can extend the encoding
/// without silently reusing a digest that meant something else.
const SPEC_DOMAIN: &[u8] = b"ferricml.dataset.spec.v1";

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
        digest.finalize().into()
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
    pub fn design_into(&self, values: &mut Vec<f32>) {
        fill_design(&self.source, self.rows, self.columns, values);
    }

    /// Generates the dataset this recipe describes.
    ///
    /// A recipe carrying no task family produces a design matrix, its recorded
    /// spec digest, and [`Truth::DesignOnly`] — there is nothing to be right
    /// about until a family assigns targets, and saying so with a variant is
    /// more honest than shipping an empty coefficient vector.
    pub fn generate(&self) -> Dataset {
        Dataset::from_parts(
            self.design(),
            None,
            None,
            Truth::DesignOnly,
            None,
            self.spec_digest(),
        )
    }
}
