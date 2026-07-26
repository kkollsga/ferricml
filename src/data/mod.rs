//! Validated, row-major data containers used by FerricML.
//!
//! Validation is deliberately performed when a value is constructed.  Code on
//! a prediction hot path can therefore borrow rows and scalar values without
//! allocating or repeatedly checking whether the underlying data is finite.

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
