//! Random-forest estimators and their public parameter types.
//!
//! Everything below the facade — the member seeding, the bootstrap sample, the
//! averaging arithmetic, the artifact codec — lives in the private
//! `super::forest` core and is shared with every other bagged tree ensemble.
//! What is random-forest-specific is exactly what appears here: two artifact
//! kinds, two parameter defaults, and the exhaustive split search.

mod model;
mod parameters;

pub use model::{RandomForestClassifier, RandomForestRegressor};
pub use parameters::{RandomForestClassifierParams, RandomForestRegressorParams};

#[cfg(test)]
mod tests;
