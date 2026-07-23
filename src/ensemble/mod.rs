//! Ensemble estimators.

mod hist_gradient_boosting;
mod random_forest;

pub use hist_gradient_boosting::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams,
};

pub use random_forest::{
    MaxFeatures, NJobs, RandomForestClassifier, RandomForestClassifierParams,
    RandomForestRegressor, RandomForestRegressorParams,
};
