//! Deterministic index splits for finite, contiguous datasets.

mod cross_validation;
mod scoring;
mod split;

pub use cross_validation::{
    CrossValidationError, CrossValidationResult, cross_validate_classifier,
    cross_validate_regressor,
};
pub use scoring::{
    ClassificationScorer, RegressionScorer, ScoringError, score_classifier, score_regressor,
};
pub use split::{
    HoldoutParams, KFold, KFoldIter, Split, SplitError, SplitPartition, StratifiedKFold,
    StratifiedKFoldIter, TestSize, stratified_train_test_split, train_test_split,
};
