//! Deterministic index splits for finite, contiguous datasets.

mod split;

pub use split::{
    HoldoutParams, KFold, KFoldIter, Split, SplitError, SplitPartition, StratifiedKFold,
    StratifiedKFoldIter, TestSize, stratified_train_test_split, train_test_split,
};
