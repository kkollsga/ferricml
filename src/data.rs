//! Validated, row-major data containers used by FerricML.
//!
//! Validation is deliberately performed when a value is constructed.  Code on
//! a prediction hot path can therefore borrow rows and scalar values without
//! allocating or repeatedly checking whether the underlying data is finite.

use std::error::Error;
use std::fmt;
use std::slice::ChunksExact;

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

fn validated_matrix_len(rows: usize, columns: usize, actual: usize) -> Result<usize, DataError> {
    if rows == 0 {
        return Err(DataError::ZeroRows);
    }
    if columns == 0 {
        return Err(DataError::ZeroColumns);
    }

    let expected = rows
        .checked_mul(columns)
        .ok_or(DataError::DimensionOverflow { rows, columns })?;
    if actual != expected {
        return Err(DataError::LengthMismatch { expected, actual });
    }
    Ok(expected)
}

fn validate_finite(values: &[f32]) -> Result<(), DataError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(DataError::NonFiniteValue { index });
    }
    Ok(())
}

/// A validated borrowed view of a contiguous row-major matrix.
///
/// A `MatrixView` is small and [`Copy`]. Constructing it checks the dimensions,
/// exact buffer length, and finiteness of every element. Row and element access
/// after construction is allocation-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixView<'a> {
    values: &'a [f32],
    rows: usize,
    columns: usize,
}

impl<'a> MatrixView<'a> {
    /// Creates a view over `values` with the given row-major shape.
    ///
    /// Both dimensions must be non-zero, their product must fit in `usize`,
    /// `values` must contain exactly that many elements, and all values must be
    /// finite.
    pub fn new(values: &'a [f32], rows: usize, columns: usize) -> Result<Self, DataError> {
        validated_matrix_len(rows, columns, values.len())?;
        validate_finite(values)?;
        Ok(Self {
            values,
            rows,
            columns,
        })
    }

    /// Returns the number of rows.
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the row-major values backing this view.
    #[inline]
    pub const fn as_slice(&self) -> &'a [f32] {
        self.values
    }

    /// Returns a row, or `None` if `index` is outside the matrix.
    #[inline]
    pub fn row(&self, index: usize) -> Option<&'a [f32]> {
        let start = index.checked_mul(self.columns)?;
        let end = start.checked_add(self.columns)?;
        self.values.get(start..end)
    }

    /// Returns an element, or `None` if either index is outside the matrix.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<f32> {
        if column >= self.columns {
            return None;
        }
        let index = row.checked_mul(self.columns)?.checked_add(column)?;
        self.values.get(index).copied()
    }

    /// Iterates over rows without allocating.
    #[inline]
    pub fn iter_rows(&self) -> ChunksExact<'a, f32> {
        self.values.chunks_exact(self.columns)
    }
}

/// An owned, validated contiguous row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    values: Vec<f32>,
    rows: usize,
    columns: usize,
}

impl DenseMatrix {
    /// Creates an owned row-major matrix.
    ///
    /// The same shape, length, and finiteness invariants as
    /// [`MatrixView::new`] are enforced.
    pub fn new(values: Vec<f32>, rows: usize, columns: usize) -> Result<Self, DataError> {
        validated_matrix_len(rows, columns, values.len())?;
        validate_finite(&values)?;
        Ok(Self {
            values,
            rows,
            columns,
        })
    }

    /// Builds an owned matrix from storage already validated through a
    /// [`MatrixView`].
    ///
    /// This is crate-private so safe external transformer implementations must
    /// still validate their output before returning it.
    pub(crate) fn from_validated_parts(values: Vec<f32>, rows: usize, columns: usize) -> Self {
        debug_assert_eq!(values.len(), rows.saturating_mul(columns));
        Self {
            values,
            rows,
            columns,
        }
    }

    /// Returns the number of rows.
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Borrows this matrix as a validated view without rescanning the values.
    #[inline]
    pub fn as_view(&self) -> MatrixView<'_> {
        MatrixView {
            values: &self.values,
            rows: self.rows,
            columns: self.columns,
        }
    }

    /// Returns the matrix values in row-major order.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Returns a row, or `None` if `index` is outside the matrix.
    #[inline]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        self.as_view().row(index)
    }

    /// Returns an element, or `None` if either index is outside the matrix.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<f32> {
        self.as_view().get(row, column)
    }

    /// Iterates over rows without allocating.
    #[inline]
    pub fn iter_rows(&self) -> ChunksExact<'_, f32> {
        self.values.chunks_exact(self.columns)
    }

    /// Consumes the matrix and returns its row-major allocation.
    #[inline]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// An owned, non-empty vector of binary classification targets.
///
/// Every target is guaranteed to be either `0` or `1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryTargets {
    values: Vec<u8>,
}

impl BinaryTargets {
    /// Validates and stores binary targets.
    pub fn new(values: Vec<u8>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptyTargets);
        }
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, value)| **value > 1) {
            return Err(DataError::InvalidBinaryTarget { index, value });
        }
        Ok(Self { values })
    }

    /// Returns the number of targets.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this vector is empty.
    ///
    /// Valid instances are never empty; this method is provided for standard
    /// collection-style interfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the validated target values.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.values
    }

    /// Returns a target, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u8> {
        self.values.get(index).copied()
    }

    /// Consumes the targets and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<u8> {
        self.values
    }
}

/// An owned, non-empty vector of finite regression targets.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionTargets {
    values: Vec<f32>,
}

impl RegressionTargets {
    /// Validates and stores regression targets.
    pub fn new(values: Vec<f32>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptyTargets);
        }
        validate_finite(&values)?;
        Ok(Self { values })
    }

    /// Returns the number of targets.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this vector is empty.
    ///
    /// Valid instances are never empty; this method is provided for standard
    /// collection-style interfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the finite target values.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Returns a target, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Consumes the targets and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// An owned, non-empty vector of finite, non-negative sample weights.
///
/// At least one weight is positive. Estimators check that the weight count
/// matches the training row count at their public fit boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleWeights {
    values: Vec<f32>,
    total: f64,
}

impl SampleWeights {
    /// Validates and stores sample weights.
    pub fn new(values: Vec<f32>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptySampleWeights);
        }
        let mut total = 0.0_f64;
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DataError::NonFiniteSampleWeight { index });
            }
            if value < 0.0 {
                return Err(DataError::NegativeSampleWeight { index });
            }
            total += f64::from(value);
        }
        if total <= 0.0 {
            return Err(DataError::ZeroTotalSampleWeight);
        }
        Ok(Self { values, total })
    }

    /// Returns the number of sample weights.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the vector is empty.
    ///
    /// Valid instances are never empty; this method is provided for standard
    /// collection-style interfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the validated sample weights.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Returns a sample weight, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Returns the finite positive sum accumulated in input order.
    #[inline]
    pub const fn total(&self) -> f64 {
        self.total
    }

    /// Consumes the weights and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_view_exposes_shape_rows_elements_and_storage() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = MatrixView::new(&values, 2, 3).unwrap();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.columns(), 3);
        assert_eq!(matrix.as_slice(), &values);
        assert_eq!(matrix.row(0), Some(&values[0..3]));
        assert_eq!(matrix.row(1), Some(&values[3..6]));
        assert_eq!(matrix.row(2), None);
        assert_eq!(matrix.get(1, 2), Some(6.0));
        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.get(0, 3), None);
        assert_eq!(
            matrix.iter_rows().collect::<Vec<_>>(),
            vec![&values[0..3], &values[3..6]]
        );
    }

    #[test]
    fn matrix_rejects_zero_dimensions_before_other_validation() {
        assert_eq!(MatrixView::new(&[], 0, 1), Err(DataError::ZeroRows));
        assert_eq!(MatrixView::new(&[], 1, 0), Err(DataError::ZeroColumns));
        assert_eq!(MatrixView::new(&[], 0, 0), Err(DataError::ZeroRows));
    }

    #[test]
    fn matrix_rejects_dimension_overflow() {
        assert_eq!(
            MatrixView::new(&[], usize::MAX, 2),
            Err(DataError::DimensionOverflow {
                rows: usize::MAX,
                columns: 2,
            })
        );
    }

    #[test]
    fn matrix_requires_exact_buffer_length() {
        assert_eq!(
            MatrixView::new(&[1.0, 2.0, 3.0], 2, 2),
            Err(DataError::LengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(
            MatrixView::new(&[1.0; 5], 2, 2),
            Err(DataError::LengthMismatch {
                expected: 4,
                actual: 5,
            })
        );
    }

    #[test]
    fn matrices_reject_every_kind_of_non_finite_value() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                MatrixView::new(&[0.0, invalid], 1, 2),
                Err(DataError::NonFiniteValue { index: 1 })
            );
            assert_eq!(
                DenseMatrix::new(vec![0.0, invalid], 1, 2),
                Err(DataError::NonFiniteValue { index: 1 })
            );
        }
    }

    #[test]
    fn dense_matrix_borrows_without_copying_and_can_return_storage() {
        let matrix = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let storage_address = matrix.as_slice().as_ptr();
        let view = matrix.as_view();

        assert_eq!(view.as_slice().as_ptr(), storage_address);
        assert_eq!(matrix.row(1), Some(&[3.0, 4.0][..]));
        assert_eq!(matrix.get(0, 1), Some(2.0));
        assert_eq!(matrix.iter_rows().len(), 2);
        assert_eq!(matrix.into_values(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dense_matrix_applies_shape_validation() {
        assert_eq!(DenseMatrix::new(vec![], 0, 1), Err(DataError::ZeroRows));
        assert_eq!(DenseMatrix::new(vec![], 1, 0), Err(DataError::ZeroColumns));
        assert_eq!(
            DenseMatrix::new(vec![1.0], 1, 2),
            Err(DataError::LengthMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn binary_targets_accept_only_nonempty_zeroes_and_ones() {
        let targets = BinaryTargets::new(vec![0, 1, 1, 0]).unwrap();
        assert_eq!(targets.len(), 4);
        assert!(!targets.is_empty());
        assert_eq!(targets.as_slice(), &[0, 1, 1, 0]);
        assert_eq!(targets.get(2), Some(1));
        assert_eq!(targets.get(4), None);

        assert_eq!(BinaryTargets::new(vec![]), Err(DataError::EmptyTargets));
        assert_eq!(
            BinaryTargets::new(vec![0, 1, 2, 3]),
            Err(DataError::InvalidBinaryTarget { index: 2, value: 2 })
        );
    }

    #[test]
    fn binary_targets_return_their_storage() {
        let values = vec![1, 0, 1];
        assert_eq!(
            BinaryTargets::new(values.clone()).unwrap().into_values(),
            values
        );
    }

    #[test]
    fn regression_targets_are_nonempty_and_finite() {
        let targets = RegressionTargets::new(vec![-1.5, 0.0, 2.5]).unwrap();
        assert_eq!(targets.len(), 3);
        assert!(!targets.is_empty());
        assert_eq!(targets.as_slice(), &[-1.5, 0.0, 2.5]);
        assert_eq!(targets.get(0), Some(-1.5));
        assert_eq!(targets.get(3), None);

        assert_eq!(RegressionTargets::new(vec![]), Err(DataError::EmptyTargets));
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                RegressionTargets::new(vec![1.0, invalid]),
                Err(DataError::NonFiniteValue { index: 1 })
            );
        }
    }

    #[test]
    fn regression_targets_return_their_storage() {
        let values = vec![1.25, -3.5];
        assert_eq!(
            RegressionTargets::new(values.clone())
                .unwrap()
                .into_values(),
            values
        );
    }

    #[test]
    fn sample_weights_are_nonempty_finite_nonnegative_and_positive() {
        let weights = SampleWeights::new(vec![0.0, 1.5, 2.5]).unwrap();
        assert_eq!(weights.len(), 3);
        assert!(!weights.is_empty());
        assert_eq!(weights.as_slice(), &[0.0, 1.5, 2.5]);
        assert_eq!(weights.get(1), Some(1.5));
        assert_eq!(weights.get(3), None);
        assert_eq!(weights.total(), 4.0);

        assert_eq!(
            SampleWeights::new(vec![]),
            Err(DataError::EmptySampleWeights)
        );
        assert_eq!(
            SampleWeights::new(vec![1.0, f32::NAN]),
            Err(DataError::NonFiniteSampleWeight { index: 1 })
        );
        assert_eq!(
            SampleWeights::new(vec![1.0, -0.5]),
            Err(DataError::NegativeSampleWeight { index: 1 })
        );
        assert_eq!(
            SampleWeights::new(vec![0.0, 0.0]),
            Err(DataError::ZeroTotalSampleWeight)
        );
    }

    #[test]
    fn sample_weights_return_their_storage() {
        let values = vec![0.5, 1.5];
        assert_eq!(
            SampleWeights::new(values.clone()).unwrap().into_values(),
            values
        );
    }

    #[test]
    fn errors_have_actionable_messages() {
        assert_eq!(
            DataError::ZeroRows.to_string(),
            "matrix row count must be non-zero"
        );
        assert_eq!(
            DataError::InvalidBinaryTarget { index: 4, value: 7 }.to_string(),
            "binary target at index 4 must be 0 or 1, got 7"
        );
    }
}
