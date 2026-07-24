//! Declared capability descriptors for every estimator type FerricML ships.
//!
//! This file is a change detector, not a behavioral proof. It states each
//! declaration exactly once so that flipping one appears as a reviewable
//! contract diff instead of drifting silently, in the same role the frozen API
//! snapshot plays for the public surface.
//!
//! Agreement between a declaration and real behavior is proven generically by
//! the estimator conformance battery, which selects its optional obligations
//! from these same constants: an estimator that declares a capability it does
//! not have, or has one it did not declare, fails there.

use ferricml::api::{Capabilities, HasCapabilities};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
};
use ferricml::linear_model::{LinearRegression, LogisticRegression, Ridge};
use ferricml::preprocessing::StandardScaler;

const WEIGHTED_AND_PERSISTED: Capabilities = Capabilities::NONE
    .with_sample_weights(true)
    .with_artifact(true);
const PERSISTED_ONLY: Capabilities = Capabilities::NONE.with_artifact(true);

#[test]
fn linear_estimators_declare_weighted_fitting_and_persistence() {
    assert_eq!(LogisticRegression::CAPABILITIES, WEIGHTED_AND_PERSISTED);
    assert_eq!(LinearRegression::CAPABILITIES, WEIGHTED_AND_PERSISTED);
    assert_eq!(Ridge::CAPABILITIES, WEIGHTED_AND_PERSISTED);
}

#[test]
fn the_standard_scaler_declares_weighted_fitting_and_persistence() {
    assert_eq!(StandardScaler::CAPABILITIES, WEIGHTED_AND_PERSISTED);
}

#[test]
fn tree_ensembles_declare_persistence_but_not_weighted_fitting() {
    assert_eq!(RandomForestRegressor::CAPABILITIES, PERSISTED_ONLY);
    assert_eq!(HistGradientBoostingRegressor::CAPABILITIES, PERSISTED_ONLY);
}

#[test]
fn the_forest_classifier_declares_nothing_yet() {
    // No weighted entry point, and no artifact kind until leaf probability
    // semantics are frozen. The conservative default is the honest answer.
    assert_eq!(RandomForestClassifier::CAPABILITIES, Capabilities::NONE);
}
