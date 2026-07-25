//! Ensemble estimators.

mod extra_trees;
mod forest;
mod hist_gradient_boosting;
mod random_forest;

#[cfg(test)]
mod equivalence;

pub use forest::{MaxFeatures, NJobs};

pub use extra_trees::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, ExtraTreesRegressor,
    ExtraTreesRegressorParams,
};

pub use hist_gradient_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams,
};

pub use random_forest::{
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
