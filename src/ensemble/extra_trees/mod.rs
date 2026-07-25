//! Extremely randomized tree ensembles and their public parameter types.
//!
//! The whole of this family is the shared `super::forest` core with two things
//! changed: each member tree draws one uniform threshold per candidate column
//! instead of optimizing within it, and bootstrap resampling is off by default.
//! Nothing else differs from a random forest, which is why nothing else is
//! restated here.

mod model;
mod parameters;

pub use model::{ExtraTreesClassifier, ExtraTreesRegressor};
pub use parameters::{ExtraTreesClassifierParams, ExtraTreesRegressorParams};

#[cfg(test)]
mod tests;
