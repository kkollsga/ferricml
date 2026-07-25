//! Deterministic index splits for finite, contiguous datasets.

mod cross_validation;
mod scoring;
mod split;

pub use cross_validation::{
    CrossValidationError, CrossValidationResult, cross_validate_classifier,
    cross_validate_regressor,
};
pub use scoring::{
    ClassificationScore, ClassificationScorer, ClassifierOutput, ClassifierOutputKind,
    RegressionScore, RegressionScorer, ScoringError, ScoringWorkspace, score_classifier,
    score_classifier_with, score_regressor, score_regressor_with,
};
pub use split::{
    GroupKFold, GroupKFoldIter, GroupShuffleSplit, GroupShuffleSplitIter, HoldoutParams, KFold,
    KFoldIter, LeaveOneOut, LeaveOneOutIter, RepeatedKFold, RepeatedKFoldIter, Split, SplitError,
    SplitPartition, StratifiedKFold, StratifiedKFoldIter, TestGroupSize, TestSize, TimeSeriesSplit,
    TimeSeriesSplitIter, stratified_train_test_split, train_test_split,
};
