//! Splitting, scoring, cross-validation, and hyperparameter search.
//!
//! The four concerns compose in one direction. A splitter turns a row count
//! into deterministic validated [`Split`] partitions. A score turns one batch
//! of model output into a number, and declares both which output it reads and
//! which direction is better. Cross-validation fits one model per split and
//! scores it. Search evaluates a [`ParameterGrid`] by cross-validating every
//! candidate.
//!
//! [`ClassificationScore`] and [`RegressionScore`] are the seam that makes the
//! last three open: [`ClassificationScorer`] and [`RegressionScorer`] are one
//! implementation of them rather than a privileged one, and cross-validation,
//! search, and permutation importance reach a metric only through it. So a
//! caller evaluates on a metric FerricML does not enumerate without
//! reimplementing the fold loop, the prediction call, or the class-layout
//! handling.
//!
//! Two choices are carried rather than duplicated into more entry points: the
//! target vocabulary is a type parameter, any
//! [`ClassificationTargets`](crate::data::ClassificationTargets), and whether a
//! fitted model offers probabilities is a value, the [`ScorableClassifier`]
//! view. There is therefore one classifier cross-validation function and one
//! classifier search function, and a further target shape or model capability
//! is an implementation rather than another entry point.
//!
//! Everything here is serial and deterministic: a fixed split order, fit
//! parameters, seed, and thread count reproduce the same fitted models and the
//! same scores, and errors carry the zero-based fold the failure came from.

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
