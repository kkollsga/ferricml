//! Owned model-switching enums with one dispatch per batch operation.

mod classifier;
mod regressor;

pub use classifier::{AnyClassifier, AnyClassifierParams};
pub use regressor::{AnyRegressor, AnyRegressorParams};
