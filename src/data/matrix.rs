use super::selection::validate_selection;
use super::{DataError, SelectionError};
use std::slice::ChunksExact;

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
///
/// This is what estimators accept. A view can be built directly over a slice a
/// caller already owns, which is the path that avoids copying data into a
/// [`DenseMatrix`] purely to hand it to a model:
///
/// ```
/// use ferricml::data::MatrixView;
///
/// let buffer = vec![0.0_f32, 1.0, 2.0, 3.0];
/// let view = MatrixView::new(&buffer, 2, 2)?;
///
/// assert_eq!(view.row(0), Some(&[0.0, 1.0][..]));
/// assert_eq!(view.get(1, 1), Some(3.0));
///
/// // It is `Copy`, so passing it to several calls costs nothing.
/// let same = view;
/// assert_eq!(same, view);
/// # Ok::<(), ferricml::data::DataError>(())
/// ```
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

    /// Builds a view over storage whose shape and values were already
    /// validated by a crate-owned producer.
    pub(crate) fn from_validated_parts(values: &'a [f32], rows: usize, columns: usize) -> Self {
        debug_assert_eq!(values.len(), rows.saturating_mul(columns));
        debug_assert!(values.iter().all(|value| value.is_finite()));
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
///
/// This is the input every FerricML estimator fits and predicts on. Values are
/// stored row-major: element `(row, column)` is at flat index
/// `row * columns + column`, so a row is contiguous and iterating rows never
/// allocates.
///
/// Construction is where validation happens. Shape, exact buffer length and
/// finiteness are all checked before the matrix exists, so an estimator that
/// holds one is not re-checking them per row, and a malformed input is rejected
/// before any training work or allocation begins.
///
/// ```
/// use ferricml::data::DenseMatrix;
///
/// // Three rows of two features, written out row by row.
/// let data = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)?;
///
/// assert_eq!(data.row(1), Some(&[3.0, 4.0][..]));
/// assert_eq!(data.get(2, 0), Some(5.0));
/// assert_eq!(data.iter_rows().count(), 3);
/// # Ok::<(), ferricml::data::DataError>(())
/// ```
///
/// An invalid shape or a non-finite value is a typed error naming what was
/// wrong and where, never a panic and never a silently repaired input:
///
/// ```
/// use ferricml::data::{DataError, DenseMatrix};
///
/// assert_eq!(
///     DenseMatrix::new(vec![1.0, 2.0, 3.0], 2, 2),
///     Err(DataError::LengthMismatch { expected: 4, actual: 3 }),
/// );
/// assert_eq!(
///     DenseMatrix::new(vec![1.0, f32::NAN], 1, 2),
///     Err(DataError::NonFiniteValue { index: 1 }),
/// );
/// ```
///
/// Estimators take a borrowed [`MatrixView`] rather than the owned matrix, so
/// one allocation serves fitting, prediction and scoring: pass
/// [`as_view`](DenseMatrix::as_view).
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

    /// Copies selected rows into a new contiguous validated matrix.
    ///
    /// Selection order and repeated indices are preserved. Every index is
    /// validated before the result allocation is created.
    pub fn select_rows(&self, indices: &[usize]) -> Result<Self, SelectionError> {
        validate_selection(indices, self.rows)?;
        let output_len =
            indices
                .len()
                .checked_mul(self.columns)
                .ok_or(SelectionError::OutputShapeOverflow {
                    rows: indices.len(),
                    columns: self.columns,
                })?;
        let mut values = Vec::with_capacity(output_len);
        for &index in indices {
            values.extend_from_slice(self.row(index).expect("selection index was validated"));
        }
        Ok(Self {
            values,
            rows: indices.len(),
            columns: self.columns,
        })
    }
}
