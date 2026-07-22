//! Shared estimator contracts and model errors.
//!
//! The category traits are deliberately object-safe. Concrete estimators keep
//! their allocation-free inherent methods, while callers can use these traits
//! for metadata and batch-level model selection without per-row dispatch.

mod any;
mod error;
mod traits;

pub use any::{AnyClassifier, AnyClassifierParams, AnyRegressor, AnyRegressorParams};
pub use error::ModelError;
pub(crate) use traits::validate_transformed_shape;
pub use traits::{Classifier, Estimator, HasParams, Regressor, Transformer};
