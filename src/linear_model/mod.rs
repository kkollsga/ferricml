//! Linear estimators with scikit-style fit and prediction semantics.

mod least_squares;
mod linear;
mod logistic;

pub use linear::{LinearRegression, LinearRegressionParams};
pub use logistic::{LogisticRegression, LogisticRegressionParams};
