//! The exchange container: a manifest and an array file, on disk, keyed by
//! digest.
//!
//! # Why a file is the boundary
//!
//! Most task families evaluate a transcendental — a Bayes probability is a
//! logistic, a log link is an exponential, a requested condition number is a
//! real power — and no two libm implementations agree on the last bits of any
//! of them. A recipe therefore promises identical bytes *per runner* for those
//! families, which is not enough for two languages on two machines to compare
//! answers on the same data.
//!
//! The materialized file is what closes that. It is generated once, digested,
//! and read afterwards, so every consumer of a container sees the same numbers
//! whatever its own libm would have produced. [`MaterializedDataset::portability`]
//! is the recipe's envelope carried alongside, so a reader knows which of the
//! two statements it holds: bytes that any runner would have produced, or bytes
//! that this one did.
//!
//! # What is in a container
//!
//! Two files that share a stem. `<name>.manifest.json` is text — the recipe in
//! full, its spec digest, the determinism envelope, and a table saying where
//! every array sits. `<name>.bin` is the arrays themselves, little-endian and
//! concatenated in table order, with `f32` features, targets and truth, `u8`
//! labels, and `u64` groups and indices.
//!
//! Splitting them that way is what makes the container readable from Python
//! with no FerricML code at all: the manifest opens with `json.load` and the
//! array file maps with `numpy.memmap`, sliced at the offsets the table names.
//! Nothing in the array file needs parsing, which is the whole reason it
//! carries no header of its own.
//!
//! # What a reader refuses
//!
//! A container is untrusted input, so this module reads it the way
//! `src/artifact/` reads a model:
//!
//! * a manifest whose recipe does not hash to the digest recorded beside it is
//!   refused, so editing the recipe cannot quietly redefine what the data is;
//! * an array file that does not hash to its recorded digest is refused;
//! * the array table must describe the file *exactly* — contiguous, in order,
//!   ending on the last byte — so a container has one encoding rather than
//!   many; and
//! * **no reservation is made from a declared length before the bytes behind
//!   it are read.** [`ExchangeCursor::bounded_capacity`] clamps every
//!   allocation to what the unread bytes could supply, which is the same rule
//!   `ArtifactCursor::bounded_capacity` enforces and for the same measured
//!   reason: a 148-byte artifact once reserved 32 MB, a 216,000-fold
//!   amplification, while still returning the correct error. Only an
//!   allocation oracle sees that, so `tests/dataset_exchange.rs` measures peak
//!   allocation rather than asserting the error.

use super::dataset::{Dataset, Target, Truth};
use super::error::ExchangeError;
use super::manifest::{self, ArrayRecord, Manifest};
use super::presets::{ReferenceLane, Split};
use super::recipe::Recipe;
use super::task::Portability;
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// How to interpret one exchanged array's bytes.
///
/// Three types, because three is what the data is: features, targets and truth
/// are `f32`, class labels are `u8` because that is the vocabulary
/// [`ClassTargets`](crate::data::ClassTargets) validates, and groups, cluster
/// assignments and pair indices are `u64` because that is what this crate's
/// grouped splitters take.
///
/// It is `#[non_exhaustive]` because a later representation — a sparse layout,
/// a wider float — must not be a breaking change for a caller matching only
/// the ones it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArrayDtype {
    /// A little-endian IEEE-754 single-precision float.
    F32,
    /// A single unsigned byte.
    U8,
    /// A little-endian unsigned 64-bit integer.
    U64,
}

impl ArrayDtype {
    /// The name this type is written under in a manifest.
    ///
    /// Chosen to match NumPy's own spelling, so a reader can hand it straight
    /// to `numpy.dtype` rather than translating it.
    ///
    /// ```
    /// use ferricml::datasets::ArrayDtype;
    ///
    /// assert_eq!(ArrayDtype::F32.label(), "f32");
    /// ```
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U8 => "u8",
            Self::U64 => "u64",
        }
    }

    /// Bytes one value of this type occupies in the array file.
    pub const fn stride(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::U8 => 1,
            Self::U64 => 8,
        }
    }
}

/// The values of one array, in whichever of the three types it holds.
///
/// Private because the public view is the three typed accessors on
/// [`DatasetArray`]: a caller asks for the type it expects and gets `None` if
/// the array is not that type, which is a narrower question than matching a
/// sum type and is the only one any consumer has needed.
#[derive(Clone, Debug, PartialEq)]
enum ArrayValues {
    F32(Vec<f32>),
    U8(Vec<u8>),
    U64(Vec<u64>),
}

/// One named array of a materialized dataset.
///
/// Two-dimensional even when it is a vector: a target is `rows × 1`, a
/// multiclass probability table is `rows × classes`, and a coefficient vector
/// is `1 × columns`. Carrying the shape rather than inferring it is what lets a
/// reader reshape without knowing which family produced the container.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetArray {
    name: String,
    rows: usize,
    columns: usize,
    values: ArrayValues,
}

impl DatasetArray {
    /// Returns the array's name, unique within its container.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns how the array's bytes are interpreted.
    #[inline]
    pub const fn dtype(&self) -> ArrayDtype {
        match self.values {
            ArrayValues::F32(_) => ArrayDtype::F32,
            ArrayValues::U8(_) => ArrayDtype::U8,
            ArrayValues::U64(_) => ArrayDtype::U64,
        }
    }

    /// Returns the number of rows the array is laid out in, row-major.
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of values per row.
    #[inline]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the number of values, which is `rows * columns`.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.values {
            ArrayValues::F32(values) => values.len(),
            ArrayValues::U8(values) => values.len(),
            ArrayValues::U64(values) => values.len(),
        }
    }

    /// Whether the array holds no values.
    ///
    /// Only a family that drew nothing produces one — a ranking design with no
    /// pairs cannot exist, so in practice this is `false` for every array a
    /// container carries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the values as `f32`, or `None` when the array holds another
    /// type.
    #[inline]
    pub fn f32_values(&self) -> Option<&[f32]> {
        match &self.values {
            ArrayValues::F32(values) => Some(values),
            _ => None,
        }
    }

    /// Returns the values as `u8`, or `None` when the array holds another type.
    #[inline]
    pub fn u8_values(&self) -> Option<&[u8]> {
        match &self.values {
            ArrayValues::U8(values) => Some(values),
            _ => None,
        }
    }

    /// Returns the values as `u64`, or `None` when the array holds another
    /// type.
    #[inline]
    pub fn u64_values(&self) -> Option<&[u64]> {
        match &self.values {
            ArrayValues::U64(values) => Some(values),
            _ => None,
        }
    }

    fn encoded_bytes(&self) -> usize {
        self.len() * self.dtype().stride()
    }

    fn write_into(&self, bytes: &mut Vec<u8>) {
        match &self.values {
            ArrayValues::F32(values) => {
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            ArrayValues::U8(values) => bytes.extend_from_slice(values),
            ArrayValues::U64(values) => {
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
}

/// Whether a repeated request regenerated the data or read it back.
///
/// The distinction is the point of [`DatasetExchange::ensure`]: a container is
/// a cache keyed by the recipe's own digest, so asking twice for one recipe
/// should cost one generation and one file read, and a harness that wants to
/// prove that has to be told which happened.
///
/// It is `#[non_exhaustive]` because a third outcome — a partial reuse, say —
/// must not be a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheOutcome {
    /// The recipe was generated and written.
    Generated,
    /// A container for this exact recipe was already on disk and was read.
    Reused,
}

/// Which dataset a container holds, when it is not its recipe's own output.
///
/// A derivation is the *identity* of the derived dataset, not a description of
/// how to compute it: it says which recorded thing the arrays are, so a reader
/// meeting a container knows what it is looking at and a writer cannot label
/// two different datasets the same way.
///
/// One variant exists because one derived dataset exists. A frozen conformance
/// lane's split is not `recipe.generate()` — `ReferenceQuality` generates the
/// whole `TRAIN_ROWS + TEST_ROWS` design, slices it, and draws the lane's own
/// targets over the slice — so a container holding one of those halves carries
/// the recipe as provenance and this as identity.
///
/// It is `#[non_exhaustive]` because a second kind of derived dataset must not
/// be a breaking change for a caller matching only the reference splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Derivation {
    /// One split of one frozen conformance lane at one recorded seed.
    ReferenceSplit {
        /// Which lane's targets were drawn over the split.
        lane: ReferenceLane,
        /// The raw stream state the lane was recorded against.
        seed: u64,
        /// Which half of the lane's single design the arrays are.
        split: Split,
    },
}

/// What a container's arrays are, relative to the recipe recorded beside them.
///
/// # Why a container has to say this
///
/// Every container carries a recipe and that recipe's spec digest, and for most
/// of them the arrays are exactly [`Recipe::generate`]'s output — so a consumer
/// can regenerate the data from the recipe alone and get the same bytes. A
/// derived container carries the *same kind* of recipe field, records the *same*
/// digest, and holds different arrays. Nothing in the recipe distinguishes the
/// two.
///
/// That is not hypothetical. Both halves of a [`ReferenceQuality`] lane report
/// the digest of the single recipe they were sliced out of, so a digest-keyed
/// cache asked for that recipe's output would hand back a training split and be
/// right about the digest and wrong about the data. This block is what makes
/// the difference readable, and
/// [`MaterializedDataset::regenerate`] is where refusing to ignore it is
/// executable rather than documented.
///
/// [`ReferenceQuality`]: super::ReferenceQuality
///
/// It is `#[non_exhaustive]` for the same reason [`Derivation`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Payload {
    /// The arrays are the recorded recipe's own output.
    Generated,
    /// The arrays are a dataset derived from the recorded recipe, and the
    /// recipe alone does not reproduce them.
    Derived(
        /// Which derived dataset this is.
        Derivation,
    ),
}

/// A dataset that has been reduced to the arrays an exchange container carries.
///
/// This is both halves of the round trip: [`MaterializedDataset::new`] builds
/// it from a recipe without touching the filesystem, and
/// [`DatasetExchange::load`] rebuilds it from the two files. The two compare
/// equal exactly when the container survived the trip, which is what
/// `materialize_then_load_returns_the_same_container` asserts — the whole
/// container, not a sampled array.
///
/// ```
/// use ferricml::datasets::{MaterializedDataset, Recipe, Task};
///
/// let recipe = Recipe::seeded(64, 4, 3)?.with_task(Task::LinearRegression {
///     informative: 2,
///     coefficient_scale: 1.0,
///     intercept: 0.5,
///     noise_scale: 0.0,
/// })?;
/// let container = MaterializedDataset::new(&recipe);
///
/// // The design is there under a stable name, in its own shape.
/// let features = container.array("features").expect("every container has a design");
/// assert_eq!((features.rows(), features.columns()), (64, 4));
/// assert_eq!(features.f32_values(), Some(recipe.generate().features().as_slice()));
///
/// // So is the answer the family recorded.
/// assert!(container.array("truth_coefficients").is_some());
/// assert_eq!(container.spec_digest(), recipe.spec_digest());
/// # Ok::<(), ferricml::datasets::DatasetError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedDataset {
    recipe: Recipe,
    spec_digest: [u8; 32],
    portability: Portability,
    payload: Payload,
    data_bytes: usize,
    data_digest: [u8; 32],
    arrays: Vec<DatasetArray>,
}

impl MaterializedDataset {
    /// Generates a recipe and reduces it to arrays, without touching the
    /// filesystem.
    ///
    /// This is what [`DatasetExchange::materialize`] writes, so it is also the
    /// value a load is compared against.
    pub fn new(recipe: &Recipe) -> Self {
        Self::containing(recipe, &recipe.generate(), Payload::Generated)
    }

    /// Reduces a dataset that is *not* the recipe's own output to arrays.
    ///
    /// The recipe is provenance — it says which stream the design came from,
    /// and its digest is what a report files the container under — while
    /// [`Derivation`] is identity, and the container records both. Nothing
    /// checks that the dataset really is that derivation, because nothing
    /// could: the point of this entry point is that the pair cannot be
    /// recomputed from the recipe. What it does guarantee is that the container
    /// *says so*, so no reader mistakes the arrays for
    /// [`Recipe::generate`]'s.
    ///
    /// ```
    /// use ferricml::datasets::{
    ///     Derivation, MaterializedDataset, Payload, ReferenceLane, ReferenceQuality, Split,
    /// };
    ///
    /// let preset = ReferenceQuality::new(ReferenceLane::SeparableBinary, 11);
    /// let derivation = Derivation::ReferenceSplit {
    ///     lane: ReferenceLane::SeparableBinary,
    ///     seed: 11,
    ///     split: Split::Test,
    /// };
    /// let container =
    ///     MaterializedDataset::derived(&preset.recipe(), &preset.test(), derivation);
    ///
    /// assert_eq!(container.payload(), Payload::Derived(derivation));
    /// // The recipe is recorded, and it is *not* what produced these arrays.
    /// assert_eq!(container.spec_digest(), preset.recipe().spec_digest());
    /// assert!(container.regenerate().is_err());
    /// ```
    pub fn derived(recipe: &Recipe, dataset: &Dataset, derivation: Derivation) -> Self {
        Self::containing(recipe, dataset, Payload::Derived(derivation))
    }

    fn containing(recipe: &Recipe, dataset: &Dataset, payload: Payload) -> Self {
        let arrays = arrays_of(dataset);
        let mut bytes = Vec::with_capacity(arrays.iter().map(DatasetArray::encoded_bytes).sum());
        for array in &arrays {
            array.write_into(&mut bytes);
        }
        Self {
            recipe: *recipe,
            spec_digest: recipe.spec_digest(),
            portability: recipe.portability(),
            payload,
            data_bytes: bytes.len(),
            data_digest: Sha256::digest(&bytes).into(),
            arrays,
        }
    }

    /// Regenerates this container's recipe, refusing when the arrays are not
    /// that recipe's output.
    ///
    /// This is the refusal [`Payload`] exists for, as an operation rather than
    /// a paragraph. For a [`Payload::Generated`] container the result equals
    /// the original — same arrays, same digests — because that is what
    /// "generated" claims. For a derived one there is nothing to regenerate:
    /// the recipe would produce the whole design where the container holds a
    /// slice of it with another family's targets, so returning that would be a
    /// silently different dataset under a matching digest.
    ///
    /// The error carries the derivation, because a caller that reached here
    /// asked the wrong question of a container and the useful answer is *what
    /// it actually holds*.
    pub fn regenerate(&self) -> Result<Self, ExchangeError> {
        match self.payload {
            Payload::Generated => Ok(Self::new(&self.recipe)),
            Payload::Derived(derivation) => Err(ExchangeError::NotRegenerable { derivation }),
        }
    }

    /// Returns the recipe the data was generated from, or derived from.
    ///
    /// Which of the two it is, is [`MaterializedDataset::payload`]. A caller
    /// that wants the recipe's own output rather than the container's should go
    /// through [`MaterializedDataset::regenerate`], which refuses instead of
    /// answering when the two are not the same thing.
    #[inline]
    pub const fn recipe(&self) -> Recipe {
        self.recipe
    }

    /// Returns whether the arrays are the recipe's output or a derived dataset.
    #[inline]
    pub const fn payload(&self) -> Payload {
        self.payload
    }

    /// Returns the digest of that recipe.
    ///
    /// A loaded container reports the digest recorded in its manifest, and a
    /// load only succeeds when the recorded recipe hashes to it, so the two
    /// statements cannot drift apart.
    #[inline]
    pub const fn spec_digest(&self) -> [u8; 32] {
        self.spec_digest
    }

    /// Returns the determinism envelope the data was produced under.
    ///
    /// [`Portability::PerRunner`] does not make a container less usable — the
    /// bytes are the bytes — but it says that regenerating the recipe
    /// elsewhere may not reproduce them, which is exactly when a materialized
    /// file is worth having rather than a recipe.
    #[inline]
    pub const fn portability(&self) -> Portability {
        self.portability
    }

    /// Returns the length of the array file.
    #[inline]
    pub const fn data_bytes(&self) -> usize {
        self.data_bytes
    }

    /// Returns the SHA-256 of the array file.
    #[inline]
    pub const fn data_digest(&self) -> [u8; 32] {
        self.data_digest
    }

    /// Returns every array, in the order the file lays them out.
    #[inline]
    pub fn arrays(&self) -> &[DatasetArray] {
        &self.arrays
    }

    /// Returns the array with this name, or `None` when the container has
    /// none.
    ///
    /// Which arrays exist is a function of the recipe's family: every
    /// container has `features`, a supervised one has `target`, and the
    /// `truth_*` arrays are whatever the family actually recorded. Asking by
    /// name rather than matching a struct is what lets a consumer read the
    /// parts it understands out of a container produced by a family it does
    /// not.
    pub fn array(&self, name: &str) -> Option<&DatasetArray> {
        self.arrays.iter().find(|array| array.name == name)
    }

    fn records(&self) -> Vec<ArrayRecord> {
        let mut records = Vec::with_capacity(self.arrays.len());
        let mut byte_offset = 0;
        for array in &self.arrays {
            records.push(ArrayRecord {
                name: array.name.clone(),
                dtype: array.dtype(),
                rows: array.rows,
                columns: array.columns,
                byte_offset,
                len: array.len(),
            });
            byte_offset += array.encoded_bytes();
        }
        records
    }

    fn encoded(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data_bytes);
        for array in &self.arrays {
            array.write_into(&mut bytes);
        }
        bytes
    }
}

/// A directory of materialized datasets, keyed by recipe digest.
///
/// # The cache is the digest, not the name
///
/// A container's file name is a caller's label — it is what a report or a
/// Python loader refers to — and its *identity* is the spec digest inside it.
/// [`DatasetExchange::ensure`] reads both: it opens the container the name
/// points at and reuses it only when the recorded digest is the one the recipe
/// has now. A recipe that changed by one knob therefore regenerates under the
/// same name rather than silently returning the previous problem, which is the
/// failure a name-keyed cache would have.
///
/// ```
/// use ferricml::datasets::{CacheOutcome, DatasetExchange, Recipe};
///
/// # let root = std::env::temp_dir().join("ferricml-doc-dataset-exchange");
/// let exchange = DatasetExchange::new(&root);
/// let recipe = Recipe::seeded(256, 8, 11)?;
///
/// let (first, _) = exchange.ensure("demo", &recipe)?;
/// assert_eq!(first.recipe(), recipe);
///
/// // The second request is a file read, and returns the same container.
/// let (second, outcome) = exchange.ensure("demo", &recipe)?;
/// assert_eq!(outcome, CacheOutcome::Reused);
/// assert_eq!(first, second);
///
/// // A recipe that moved by one knob regenerates under the same name.
/// let moved = Recipe::seeded(256, 8, 12)?;
/// let (replaced, outcome) = exchange.ensure("demo", &moved)?;
/// assert_eq!(outcome, CacheOutcome::Generated);
/// assert_ne!(replaced.spec_digest(), first.spec_digest());
/// # std::fs::remove_dir_all(&root).ok();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetExchange {
    root: PathBuf,
}

impl DatasetExchange {
    /// Container format version this crate writes and accepts.
    ///
    /// One number for the manifest and the array file together, because they
    /// are written as a pair and mean nothing apart. A reader meeting another
    /// version refuses rather than guessing which half moved.
    pub const FORMAT_VERSION: u32 = manifest::FORMAT_VERSION;

    /// Largest manifest this reader will read.
    ///
    /// A manifest is a recipe and a table of at most a few dozen rows, so this
    /// is three orders of magnitude above anything legitimate. It is checked
    /// against the file's length *before* the file is read, so an oversized
    /// manifest is refused without being loaded.
    pub const MAX_MANIFEST_BYTES: usize = 256 * 1024;

    /// Largest array file this reader will read.
    ///
    /// The performance grid's widest point is a few megabytes, so this leaves
    /// room for designs far larger than anything FerricML measures itself on
    /// while still being a limit rather than an invitation.
    pub const MAX_DATA_BYTES: usize = 256 * 1024 * 1024;

    /// Names a directory containers live in.
    ///
    /// The directory is created when something is written to it, not here, so
    /// constructing an exchange touches no filesystem and cannot fail.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the directory containers live in.
    #[inline]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the path of a container's manifest.
    ///
    /// The name must be a single file stem — lower-case ASCII, digits,
    /// hyphens and underscores — so a caller's label cannot reach outside the
    /// exchange directory. Validation is here rather than at the write, so the
    /// refusal happens before anything is generated.
    pub fn manifest_path(&self, name: &str) -> Result<PathBuf, ExchangeError> {
        self.path(name, "manifest.json")
    }

    /// Returns the path of a container's array file.
    pub fn data_path(&self, name: &str) -> Result<PathBuf, ExchangeError> {
        self.path(name, "bin")
    }

    fn path(&self, name: &str, extension: &str) -> Result<PathBuf, ExchangeError> {
        if !is_container_name(name) {
            return Err(ExchangeError::InvalidName);
        }
        Ok(self.root.join(format!("{name}.{extension}")))
    }

    /// Generates a recipe and writes its container, replacing any container
    /// already under that name.
    ///
    /// The array file is written first and the manifest second, so a manifest
    /// that exists always has its data beside it: the manifest is what a
    /// reader opens, and the reverse order would leave a window in which it
    /// points at a file that does not exist yet.
    pub fn materialize(
        &self,
        name: &str,
        recipe: &Recipe,
    ) -> Result<MaterializedDataset, ExchangeError> {
        self.write(name, MaterializedDataset::new(recipe))
    }

    /// Writes a container holding a dataset the recipe does not produce.
    ///
    /// The derived counterpart of [`DatasetExchange::materialize`], and the
    /// entry point a frozen conformance lane reaches the exchange through. The
    /// dataset is the caller's because it has to be — [`Derivation`] names
    /// which recorded thing it is, and the exchange has no way to compute it.
    pub fn materialize_derived(
        &self,
        name: &str,
        recipe: &Recipe,
        dataset: &Dataset,
        derivation: Derivation,
    ) -> Result<MaterializedDataset, ExchangeError> {
        self.write(
            name,
            MaterializedDataset::derived(recipe, dataset, derivation),
        )
    }

    fn write(
        &self,
        name: &str,
        container: MaterializedDataset,
    ) -> Result<MaterializedDataset, ExchangeError> {
        let manifest_path = self.manifest_path(name)?;
        let data_path = self.data_path(name)?;
        let bytes = container.encoded();
        if bytes.len() > Self::MAX_DATA_BYTES {
            return Err(ExchangeError::SizeLimitExceeded {
                limit: Self::MAX_DATA_BYTES,
                actual: bytes.len() as u64,
            });
        }
        let text = manifest::render(&Manifest {
            recipe: container.recipe,
            spec_digest: container.spec_digest,
            portability: container.portability,
            payload: container.payload,
            data_file: data_file_name(name),
            data_bytes: container.data_bytes,
            data_digest: container.data_digest,
            arrays: container.records(),
        });
        if text.len() > Self::MAX_MANIFEST_BYTES {
            return Err(ExchangeError::SizeLimitExceeded {
                limit: Self::MAX_MANIFEST_BYTES,
                actual: text.len() as u64,
            });
        }
        fs::create_dir_all(&self.root).map_err(|source| ExchangeError::Io {
            path: self.root.clone(),
            source,
        })?;
        write(&data_path, &bytes)?;
        write(&manifest_path, text.as_bytes())?;
        Ok(container)
    }

    /// Reads a container back.
    ///
    /// Everything a container claims is checked before any of it is believed:
    /// the manifest's recipe against its recorded digest, the array file
    /// against its recorded digest and length, and the array table against the
    /// file it describes.
    pub fn load(&self, name: &str) -> Result<MaterializedDataset, ExchangeError> {
        let manifest_path = self.manifest_path(name)?;
        let data_path = self.data_path(name)?;

        let text = read_bounded(&manifest_path, Self::MAX_MANIFEST_BYTES)?;
        let text = String::from_utf8(text).map_err(|error| ExchangeError::MalformedManifest {
            offset: error.utf8_error().valid_up_to(),
        })?;
        let manifest = manifest::parse(&text)?;
        if manifest.data_file != data_file_name(name) {
            return Err(ExchangeError::InvalidArrayTable);
        }

        let bytes = read_bounded(&data_path, Self::MAX_DATA_BYTES)?;
        if bytes.len() != manifest.data_bytes {
            return Err(ExchangeError::InvalidArrayTable);
        }
        if Sha256::digest(&bytes).as_slice() != manifest.data_digest {
            return Err(ExchangeError::DataChecksumMismatch);
        }

        let arrays = decode_arrays(&manifest.arrays, &bytes)?;
        Ok(MaterializedDataset {
            recipe: manifest.recipe,
            spec_digest: manifest.spec_digest,
            portability: manifest.portability,
            payload: manifest.payload,
            data_bytes: manifest.data_bytes,
            data_digest: manifest.data_digest,
            arrays,
        })
    }

    /// Returns the container for a recipe, generating it only if the one on
    /// disk is not already that recipe's.
    ///
    /// A container whose manifest cannot be read, or which records a different
    /// recipe, is regenerated rather than reported: the request is for a
    /// recipe's data, and a stale or damaged file is an implementation detail
    /// of the cache. A failure to *write* the replacement is still an error,
    /// because at that point the request cannot be satisfied at all.
    ///
    /// # A derived container is refused rather than reused or replaced
    ///
    /// A [`Payload::Derived`] container under this name is neither: it is a
    /// deliberate recording of a dataset the recipe does not produce, and it
    /// records the recipe's digest, so reusing it would return arrays the
    /// caller believes it could regenerate and overwriting it would destroy a
    /// recording on a name collision. Both are worse than
    /// [`ExchangeError::NotRegenerable`], which names what is actually there.
    pub fn ensure(
        &self,
        name: &str,
        recipe: &Recipe,
    ) -> Result<(MaterializedDataset, CacheOutcome), ExchangeError> {
        if let Ok(container) = self.load(name) {
            if let Payload::Derived(derivation) = container.payload {
                return Err(ExchangeError::NotRegenerable { derivation });
            }
            if container.spec_digest == recipe.spec_digest() {
                return Ok((container, CacheOutcome::Reused));
            }
        }
        Ok((self.materialize(name, recipe)?, CacheOutcome::Generated))
    }

    /// Returns the container for a derived dataset, writing it only if the one
    /// on disk is not already that recipe's and that derivation's.
    ///
    /// Both halves of the key are checked because neither alone identifies the
    /// container: a lane's two splits share one recipe and therefore one
    /// digest, and one derivation at two seeds is two recipes.
    ///
    /// Unlike [`DatasetExchange::ensure`] this saves the *write* rather than
    /// the generation — the caller already holds the dataset, because a derived
    /// dataset is by definition not something the exchange could produce from
    /// what it was given. A container found under this name that records a
    /// generated payload is replaced, which is the mirror image of
    /// [`DatasetExchange::ensure`]'s refusal rather than an inconsistency with
    /// it: a generated container is reproducible from the recipe written inside
    /// it, so overwriting one loses nothing, and a derived container is the
    /// only copy of something no recipe reproduces.
    pub fn ensure_derived(
        &self,
        name: &str,
        recipe: &Recipe,
        dataset: &Dataset,
        derivation: Derivation,
    ) -> Result<(MaterializedDataset, CacheOutcome), ExchangeError> {
        if let Ok(container) = self.load(name)
            && container.payload == Payload::Derived(derivation)
            && container.spec_digest == recipe.spec_digest()
        {
            return Ok((container, CacheOutcome::Reused));
        }
        Ok((
            self.materialize_derived(name, recipe, dataset, derivation)?,
            CacheOutcome::Generated,
        ))
    }
}

/// The array file's name for a container stem.
fn data_file_name(name: &str) -> String {
    format!("{name}.bin")
}

/// Whether a name is a single file stem this exchange will use.
fn is_container_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), ExchangeError> {
    fs::write(path, bytes).map_err(|source| ExchangeError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads a whole file, refusing one longer than `limit` before reading it.
///
/// The length check is against the file's metadata rather than against what
/// was read, because `fs::read` sizes its buffer from that metadata: checking
/// afterwards would mean the oversized allocation had already happened.
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, ExchangeError> {
    let metadata = fs::metadata(path).map_err(|source| ExchangeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ExchangeError::Io {
            path: path.to_path_buf(),
            source: io::Error::new(io::ErrorKind::InvalidInput, "not a regular file"),
        });
    }
    if metadata.len() > limit as u64 {
        return Err(ExchangeError::SizeLimitExceeded {
            limit,
            actual: metadata.len(),
        });
    }
    fs::read(path).map_err(|source| ExchangeError::Io {
        path: path.to_path_buf(),
        source,
    })
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// A position in an array file.
///
/// Deliberately the same shape as `ArtifactCursor`, and for the same reason:
/// the crate already learned once that a decoder reserving from a declared
/// count is exploitable by a file far too small to contain what it declares.
/// The two cursors are separate because their error vocabularies are — a
/// dataset container is not a model artifact — and because a shared cursor
/// would have to carry both.
struct ExchangeCursor<'a> {
    remaining: &'a [u8],
}

impl<'a> ExchangeCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ExchangeError> {
        if self.remaining.len() < count {
            return Err(ExchangeError::InvalidArrayTable);
        }
        let (value, remaining) = self.remaining.split_at(count);
        self.remaining = remaining;
        Ok(value)
    }

    /// Capacity for `requested` values of `stride` bytes each, clamped to what
    /// the unread bytes could actually supply.
    ///
    /// The declared length is the array table's, and the array table is part
    /// of an untrusted file. The number of bytes still unread is the one
    /// quantity a hostile writer cannot inflate, so it is what bounds the
    /// reservation — and it bounds it *independently* of the table validation
    /// below, so removing either one still leaves the allocation bounded.
    const fn bounded_capacity(&self, requested: usize, stride: usize) -> usize {
        debug_assert!(stride > 0);
        let affordable = self.remaining.len() / stride;
        if requested < affordable {
            requested
        } else {
            affordable
        }
    }

    const fn consumed_from(&self, whole: &[u8]) -> usize {
        whole.len() - self.remaining.len()
    }

    const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}

/// Rebuilds every array the table names out of the array file.
///
/// The table has to describe the file exactly: entries in order, each starting
/// where the previous one ended, the last ending on the file's last byte, and
/// no two sharing a name. A container therefore has one encoding, so two files
/// that decode to the same arrays are the same file.
fn decode_arrays(
    records: &[ArrayRecord],
    bytes: &[u8],
) -> Result<Vec<DatasetArray>, ExchangeError> {
    if records.len() > manifest::MAX_ARRAYS {
        return Err(ExchangeError::InvalidArrayTable);
    }
    let mut cursor = ExchangeCursor::new(bytes);
    let mut arrays: Vec<DatasetArray> = Vec::with_capacity(records.len());
    for record in records {
        if record.byte_offset != cursor.consumed_from(bytes) {
            return Err(ExchangeError::InvalidArrayTable);
        }
        if record.rows.checked_mul(record.columns) != Some(record.len) {
            return Err(ExchangeError::InvalidArrayTable);
        }
        if arrays.iter().any(|array| array.name == record.name) {
            return Err(ExchangeError::InvalidArrayTable);
        }
        let values = match record.dtype {
            ArrayDtype::F32 => {
                let mut values = Vec::with_capacity(cursor.bounded_capacity(record.len, 4));
                for chunk in cursor.take(byte_span(record.len, 4)?)?.chunks_exact(4) {
                    values.push(f32::from_le_bytes(chunk.try_into().expect("exact length")));
                }
                ArrayValues::F32(values)
            }
            ArrayDtype::U8 => {
                let mut values = Vec::with_capacity(cursor.bounded_capacity(record.len, 1));
                values.extend_from_slice(cursor.take(record.len)?);
                ArrayValues::U8(values)
            }
            ArrayDtype::U64 => {
                let mut values = Vec::with_capacity(cursor.bounded_capacity(record.len, 8));
                for chunk in cursor.take(byte_span(record.len, 8)?)?.chunks_exact(8) {
                    values.push(u64::from_le_bytes(chunk.try_into().expect("exact length")));
                }
                ArrayValues::U64(values)
            }
        };
        arrays.push(DatasetArray {
            name: record.name.clone(),
            rows: record.rows,
            columns: record.columns,
            values,
        });
    }
    if !cursor.is_empty() {
        return Err(ExchangeError::InvalidArrayTable);
    }
    Ok(arrays)
}

/// The byte span of `len` values of `stride` bytes, refusing an overflow.
///
/// A declared length near [`usize::MAX`] overflows the multiplication rather
/// than describing a file, and wrapping it would turn an absurd claim into a
/// small plausible one.
fn byte_span(len: usize, stride: usize) -> Result<usize, ExchangeError> {
    len.checked_mul(stride)
        .ok_or(ExchangeError::InvalidArrayTable)
}

// ---------------------------------------------------------------------------
// Encoding a generated dataset
// ---------------------------------------------------------------------------

/// Reduces a generated dataset to the arrays a container carries.
///
/// The names are the contract, not the order: a consumer asks for `features`
/// or `truth_probabilities` and gets what the family recorded, or nothing when
/// that family records no such thing. Scalars a family knows — the intercept,
/// the rank, the class count — are one-value arrays rather than manifest
/// fields, so the container has exactly one kind of payload and a NumPy reader
/// has exactly one code path.
fn arrays_of(dataset: &Dataset) -> Vec<DatasetArray> {
    let mut arrays = Vec::new();
    let rows = dataset.features().rows();
    let columns = dataset.features().columns();
    arrays.push(f32_array(
        "features",
        rows,
        columns,
        dataset.features().as_slice().to_vec(),
    ));

    match dataset.target() {
        None => {}
        Some(Target::Binary(targets)) => {
            arrays.push(u8_array("target", rows, 1, targets.as_slice().to_vec()));
        }
        Some(Target::Class(targets)) => {
            arrays.push(u8_array("target", rows, 1, targets.as_slice().to_vec()));
        }
        Some(Target::Regression(targets)) => {
            arrays.push(f32_array("target", rows, 1, targets.as_slice().to_vec()));
        }
    }

    if let Some(weights) = dataset.weights() {
        arrays.push(f32_array("weights", rows, 1, weights.as_slice().to_vec()));
    }
    if let Some(groups) = dataset.groups() {
        arrays.push(u64_array("groups", rows, 1, groups.to_vec()));
    }
    if let Some(pairs) = dataset.pairs() {
        let count = pairs.len();
        arrays.push(u64_array(
            "pair_left",
            count,
            1,
            pairs.iter().map(|pair| pair.pair().left() as u64).collect(),
        ));
        arrays.push(u64_array(
            "pair_right",
            count,
            1,
            pairs
                .iter()
                .map(|pair| pair.pair().right() as u64)
                .collect(),
        ));
        arrays.push(u8_array(
            "pair_outcome",
            count,
            1,
            pairs
                .iter()
                .map(|pair| outcome_code(pair.outcome()))
                .collect(),
        ));
        arrays.push(f32_array(
            "pair_weight",
            count,
            1,
            pairs.iter().map(|pair| pair.weight()).collect(),
        ));
    }

    push_truth(&mut arrays, dataset.truth(), rows, columns);
    arrays
}

/// The code an outcome is written under.
///
/// Written out rather than derived from declaration order, for the reason a
/// digest tag is: reordering the variants must not restate what a recorded
/// container means.
const fn outcome_code(outcome: crate::ranking::PairOutcome) -> u8 {
    match outcome {
        crate::ranking::PairOutcome::LeftPreferred => 1,
        crate::ranking::PairOutcome::RightPreferred => 2,
        crate::ranking::PairOutcome::Tie => 3,
    }
}

/// Appends whatever the family recorded about the right answer.
///
/// Driven by [`Truth`]'s own accessors rather than by a match over its
/// variants, so a family that records a kind of truth another already records
/// exports it under the same name automatically. The one exhaustive match is
/// on the *scalars*, which have no shared accessor.
fn push_truth(arrays: &mut Vec<DatasetArray>, truth: &Truth, rows: usize, columns: usize) {
    if let Some(values) = truth.coefficients() {
        arrays.push(f32_array("truth_coefficients", 1, columns, values.to_vec()));
    }
    if let Some(values) = truth.start_coefficients() {
        arrays.push(f32_array(
            "truth_start_coefficients",
            1,
            columns,
            values.to_vec(),
        ));
    }
    if let Some(values) = truth.end_coefficients() {
        arrays.push(f32_array(
            "truth_end_coefficients",
            1,
            columns,
            values.to_vec(),
        ));
    }
    if let Some(value) = truth.intercept() {
        arrays.push(f32_array("truth_intercept", 1, 1, vec![value]));
    }
    if let Some(values) = truth.conditional_mean() {
        arrays.push(f32_array(
            "truth_conditional_mean",
            rows,
            1,
            values.to_vec(),
        ));
    }
    if let Some(values) = truth.probabilities() {
        arrays.push(f32_array("truth_probabilities", rows, 1, values.to_vec()));
    }
    if let Some(classes) = truth.classes() {
        let values = truth
            .class_probabilities()
            .expect("a family recording a class count records the probabilities behind it");
        arrays.push(f32_array(
            "truth_class_probabilities",
            rows,
            classes,
            values.to_vec(),
        ));
        arrays.push(u64_array("truth_classes", 1, 1, vec![classes as u64]));
    }
    if let Some(blobs) = truth.blobs() {
        let assignments = truth
            .cluster_assignments()
            .expect("a family recording a cluster count records the assignment");
        let centres = truth
            .cluster_centres()
            .expect("a family recording a cluster count records the centres");
        arrays.push(u64_array(
            "truth_cluster_assignments",
            rows,
            1,
            assignments.iter().map(|&value| value as u64).collect(),
        ));
        arrays.push(f32_array(
            "truth_cluster_centres",
            blobs,
            columns,
            centres.to_vec(),
        ));
        arrays.push(u64_array("truth_blobs", 1, 1, vec![blobs as u64]));
    }
    if let Some(values) = truth.times() {
        arrays.push(f32_array("truth_times", rows, 1, values.to_vec()));
    }
    if let Some(values) = truth.utilities() {
        arrays.push(f32_array("truth_utilities", rows, 1, values.to_vec()));
    }
    if let Some(rank) = truth.rank() {
        arrays.push(u64_array("truth_rank", 1, 1, vec![rank as u64]));
    }
    if let Some(grades) = truth.grades() {
        arrays.push(u64_array("truth_grades", 1, 1, vec![grades as u64]));
    }
}

fn f32_array(name: &str, rows: usize, columns: usize, values: Vec<f32>) -> DatasetArray {
    debug_assert_eq!(rows * columns, values.len(), "{name} has a wrong shape");
    DatasetArray {
        name: name.to_owned(),
        rows,
        columns,
        values: ArrayValues::F32(values),
    }
}

fn u8_array(name: &str, rows: usize, columns: usize, values: Vec<u8>) -> DatasetArray {
    debug_assert_eq!(rows * columns, values.len(), "{name} has a wrong shape");
    DatasetArray {
        name: name.to_owned(),
        rows,
        columns,
        values: ArrayValues::U8(values),
    }
}

fn u64_array(name: &str, rows: usize, columns: usize, values: Vec<u64>) -> DatasetArray {
    debug_assert_eq!(rows * columns, values.len(), "{name} has a wrong shape");
    DatasetArray {
        name: name.to_owned(),
        rows,
        columns,
        values: ArrayValues::U64(values),
    }
}
