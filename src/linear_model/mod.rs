//! Linear estimators with stable fit and prediction semantics.

mod least_squares;
mod linear_regression;
mod logistic;
mod ridge;

pub use linear_regression::{LinearRegression, LinearRegressionParams};
pub use logistic::{LogisticRegression, LogisticRegressionParams};
pub use ridge::{Ridge, RidgeParams};
