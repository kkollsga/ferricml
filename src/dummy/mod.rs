//! Baseline estimators that ignore their features.
//!
//! A baseline answers "is this model worth anything?" by predicting from the
//! training targets alone: the most frequent class, or the mean value. A real
//! estimator that cannot beat one on a given dataset has learned nothing from
//! the features, whatever its absolute score looks like.
//!
//! They deliberately have no tunable behavior, so their parameter types are
//! empty today and exist only to keep every FerricML estimator the same shape.
//! Neither declares a weighted entry point or an artifact kind, because a
//! baseline is refitted in a millisecond rather than persisted.
//! [`DummyClassifier`] does declare [`Capabilities::probability`](crate::api::Capabilities::probability):
//! its class frequencies are a genuine probability vector, and predicting one
//! is the only thing it does. That exception is stated here because this
//! paragraph used to read "neither declares any capability", which was a
//! defect written down as a fact — the same sentence above `DummyClassifier`
//! was already corrected, and the module header was missed.

mod classifier;
mod regressor;

pub use classifier::{DummyClassifier, DummyClassifierParams};
pub use regressor::{DummyRegressor, DummyRegressorParams};
