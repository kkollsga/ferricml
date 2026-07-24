//! Baseline estimators that ignore their features.
//!
//! A baseline answers "is this model worth anything?" by predicting from the
//! training targets alone: the most frequent class, or the mean value. A real
//! estimator that cannot beat one on a given dataset has learned nothing from
//! the features, whatever its absolute score looks like.
//!
//! They deliberately have no tunable behavior, so their parameter types are
//! empty today and exist only to keep every FerricML estimator the same shape.
//! Neither declares any capability: there is no weighted entry point and no
//! artifact kind, because a baseline is refitted in a millisecond rather than
//! persisted.

mod classifier;
mod regressor;

pub use classifier::{DummyClassifier, DummyClassifierParams};
pub use regressor::{DummyRegressor, DummyRegressorParams};
