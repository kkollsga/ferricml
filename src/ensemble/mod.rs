//! Ensemble estimators.

mod random_forest;

pub use random_forest::{
    MaxFeatures, NJobs, RandomForestClassifier, RandomForestClassifierParams,
    RandomForestRegressor, RandomForestRegressorParams,
};
