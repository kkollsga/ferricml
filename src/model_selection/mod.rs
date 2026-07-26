//! Deterministic index splits for finite, contiguous datasets.

mod cross_validation;
mod scoring;
mod search;
mod split;

pub use cross_validation::{
    CrossValidationError, CrossValidationResult, cross_validate_classifier,
    cross_validate_regressor,
};
pub use scoring::{
    ClassificationScore, ClassificationScorer, ClassifierOutput, ClassifierOutputKind,
    RegressionScore, RegressionScorer, ScorableClassifier, ScoringError, ScoringWorkspace,
    score_classifier, score_classifier_with, score_multiclass_classifier,
    score_multiclass_classifier_with, score_regressor, score_regressor_with,
};
pub use search::{
    CandidateScores, ParameterGrid, SearchError, SearchResult, grid_search_classifier,
    grid_search_regressor,
};
pub use split::{
    GroupKFold, GroupKFoldIter, GroupShuffleSplit, GroupShuffleSplitIter, HoldoutParams, KFold,
    KFoldIter, LeaveOneOut, LeaveOneOutIter, RepeatedKFold, RepeatedKFoldIter, Split, SplitError,
    SplitPartition, StratifiedKFold, StratifiedKFoldIter, TestGroupSize, TestSize, TimeSeriesSplit,
    TimeSeriesSplitIter, stratified_train_test_split, train_test_split,
};

/// The one classifier scoring implementation, for crate consumers that are
/// generic over the target vocabulary.
///
/// `inspection` reaches it here for the same reason cross-validation does: it
/// holds a `ClassificationTargets` value rather than a named container, so
/// neither public wrapper is a vocabulary it could pick. This is not a wider
/// contract than the public ones — it is the same one, minus the choice of
/// which target type to name.
pub(crate) use scoring::score_labelled;
