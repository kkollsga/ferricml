//! Validated data containers, and the shared vocabulary over them.
//!
//! Validation is deliberately performed when a value is constructed.  Code on
//! a prediction hot path can therefore borrow rows and scalar values without
//! allocating or repeatedly checking whether the underlying data is finite.
//!
//! [`DenseMatrix`] owns row-major features and [`MatrixView`] borrows them;
//! [`BinaryTargets`], [`ClassTargets`], and [`RegressionTargets`] own targets;
//! [`SampleWeights`] owns per-row weights. Subsetting a matrix or a target
//! vector produces the same type with the same construction-time guarantees and
//! no revalidation, which is what lets a cross-validation fold be handed
//! straight back to a fit.
//!
//! [`ClassificationTargets`] is the vocabulary over the two classification
//! target types. It lives with the containers rather than with any one of its
//! consumers because it names target types, and both model selection and
//! inspection are generic over it.

mod error;
mod matrix;
mod selection;
mod targets;
mod weights;

pub use error::{DataError, SelectionError};
pub use matrix::{DenseMatrix, MatrixView};
pub use targets::{BinaryTargets, ClassTargets, ClassificationTargets, RegressionTargets};
pub use weights::SampleWeights;

#[cfg(test)]
mod tests;
