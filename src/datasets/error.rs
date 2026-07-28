use std::error::Error;
use std::fmt;

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
        }
    }
}

impl Error for DatasetError {}
