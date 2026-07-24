use std::error::Error;
use std::fmt;

/// An error encountered while constructing a validated data container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataError {
    /// A matrix was requested with no rows.
    ZeroRows,
    /// A matrix was requested with no columns.
    ZeroColumns,
    /// `rows * columns` cannot be represented by [`usize`].
    DimensionOverflow {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        columns: usize,
    },
    /// The supplied buffer does not exactly fill the requested shape.
    LengthMismatch {
        /// Length required by the shape.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// A floating-point buffer contained NaN or infinity.
    NonFiniteValue {
        /// Flat, row-major index of the invalid value.
        index: usize,
    },
    /// A target vector was empty.
    EmptyTargets,
    /// A sample-weight vector was empty.
    EmptySampleWeights,
    /// A sample weight was NaN or infinite.
    NonFiniteSampleWeight {
        /// Index of the invalid sample weight.
        index: usize,
    },
    /// A sample weight was negative.
    NegativeSampleWeight {
        /// Index of the invalid sample weight.
        index: usize,
    },
    /// Every sample weight was zero.
    ZeroTotalSampleWeight,
    /// A binary target was neither zero nor one.
    InvalidBinaryTarget {
        /// Index of the invalid target.
        index: usize,
        /// Invalid target value.
        value: u8,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRows => f.write_str("matrix row count must be non-zero"),
            Self::ZeroColumns => f.write_str("matrix column count must be non-zero"),
            Self::DimensionOverflow { rows, columns } => {
                write!(f, "matrix dimensions overflow usize: {rows} x {columns}")
            }
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "matrix buffer length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteValue { index } => {
                write!(f, "floating-point value at index {index} is not finite")
            }
            Self::EmptyTargets => f.write_str("target vector must be non-empty"),
            Self::EmptySampleWeights => f.write_str("sample weights must be non-empty"),
            Self::NonFiniteSampleWeight { index } => {
                write!(f, "sample weight at index {index} is not finite")
            }
            Self::NegativeSampleWeight { index } => {
                write!(f, "sample weight at index {index} is negative")
            }
            Self::ZeroTotalSampleWeight => f.write_str("sample weights must have a positive total"),
            Self::InvalidBinaryTarget { index, value } => write!(
                f,
                "binary target at index {index} must be 0 or 1, got {value}"
            ),
        }
    }
}

impl Error for DataError {}

/// Errors produced while selecting rows or targets by index.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SelectionError {
    /// A selection contained no indices.
    Empty,
    /// A selected index was outside the source.
    IndexOutOfBounds {
        /// Position within the requested selection.
        position: usize,
        /// Invalid source index.
        index: usize,
        /// Number of available source values.
        available: usize,
    },
    /// Selected matrix dimensions could not be represented.
    OutputShapeOverflow {
        /// Requested output rows.
        rows: usize,
        /// Requested output columns.
        columns: usize,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("selection must contain at least one index"),
            Self::IndexOutOfBounds {
                position,
                index,
                available,
            } => write!(
                f,
                "selection index {index} at position {position} is outside 0..{available}"
            ),
            Self::OutputShapeOverflow { rows, columns } => {
                write!(
                    f,
                    "selected matrix dimensions overflow usize: {rows} x {columns}"
                )
            }
        }
    }
}

impl Error for SelectionError {}
