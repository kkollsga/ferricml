//! Ensemble estimators.

mod forest;
mod hist_gradient_boosting;
mod random_forest;

pub use forest::{MaxFeatures, NJobs};

pub use hist_gradient_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams,
};

pub use random_forest::{
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
