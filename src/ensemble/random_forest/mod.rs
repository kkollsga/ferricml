//! Random-forest estimators and their private implementation family.

mod model;
mod parameters;
mod training;

pub use model::{RandomForestClassifier, RandomForestRegressor};
pub use parameters::{
    MaxFeatures, NJobs, RandomForestClassifierParams, RandomForestRegressorParams,
};

#[cfg(test)]
use crate::tree::{FEATURE_MASK, LEFT_IS_LEAF, PackedNode, RIGHT_IS_LEAF};

#[cfg(test)]
mod tests;
