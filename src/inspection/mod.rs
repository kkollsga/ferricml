//! Model-agnostic inspection built on the prediction and scoring contracts.
//!
//! FerricML keeps fitted model internals private, so there are no
//! impurity-based importances to expose. Permutation importance answers
//! "which features mattered" without any estimator cooperation: it scores a
//! fitted model, then rescores it with one feature column randomly permuted,
//! and reports how much quality that destroyed.
//!
//! A fitted model is reached only through its public batch-prediction traits,
//! and a metric only through the
//! [`ClassificationScore`](crate::model_selection::ClassificationScore) and
//! [`RegressionScore`](crate::model_selection::RegressionScore) seam — so the
//! result is oriented by the score's own declaration, and a caller's own metric
//! is inspected exactly as a built-in one is.
//!
//! There is one entry point per estimator kind, each with an `_into` form that
//! writes into caller-owned buffers. The classifier one serves any
//! [`ClassificationTargets`](crate::data::ClassificationTargets) vocabulary,
//! because how many classes the labels name is a property of the metric rather
//! than of the permutation loop.

mod permutation;

pub use permutation::{
    InspectionError, PermutationImportance, PermutationImportanceParams,
    permutation_importance_classifier, permutation_importance_classifier_into,
    permutation_importance_regressor, permutation_importance_regressor_into,
};
