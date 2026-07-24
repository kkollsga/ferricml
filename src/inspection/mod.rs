//! Model-agnostic inspection built on the public prediction and scoring
//! contracts.
//!
//! FerricML keeps fitted model internals private, so there are no
//! impurity-based importances to expose. Permutation importance answers
//! "which features mattered" without any estimator cooperation: it scores a
//! fitted model, then rescores it with one feature column randomly permuted,
//! and reports how much quality that destroyed.

mod permutation;

pub use permutation::{
    InspectionError, PermutationImportance, PermutationImportanceParams,
    permutation_importance_classifier, permutation_importance_classifier_into,
    permutation_importance_regressor, permutation_importance_regressor_into,
};
