//! The private core every bagged tree ensemble is built on.
//!
//! A random forest and an extremely randomized ensemble differ in exactly two
//! places: the split search their member trees use, and the artifact kind they
//! own. Everything else — the parameter vocabulary, the per-member seed
//! derivation, the bootstrap sample, the averaging arithmetic, the tie rules,
//! the validation, and the on-disk field order — is one implementation here,
//! consumed by both public facades.
//!
//! Nothing in this module is public. `Forest`, `ClassifierCore` and
//! `RegressorCore` are crate-private, so "callers must not be able to depend on
//! forest tree layout" stays true; only [`NJobs`] crosses the facade, and it is
//! already public vocabulary. The shared `MaxFeatures` is *consumed* here and
//! published from `crate::tree`, which owns it.

pub(super) mod codec;
pub(super) mod facade;
pub(super) mod model;
pub(super) mod parameters;
pub(super) mod training;

pub(crate) use facade::{forest_classifier, forest_regressor};
pub use parameters::NJobs;
pub(crate) use parameters::forest_params;
