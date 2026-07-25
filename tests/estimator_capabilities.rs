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

use ferricml::api::{AnyClassifier, AnyRegressor, Capabilities, HasCapabilities};
use ferricml::calibration::{CalibratedClassifier, IsotonicRegression, PlattCalibrator};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::dummy::{DummyClassifier, DummyRegressor};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestClassifierParams,
    RandomForestRegressor,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    MaxAbsScaler, MinMaxScaler, MinMaxScalerParams, StandardScaler, StandardScalerParams,
};

const WEIGHTED_AND_PERSISTED: Capabilities = Capabilities::NONE
    .with_sample_weights(true)
    .with_artifact(true);
const PERSISTED_ONLY: Capabilities = Capabilities::NONE.with_artifact(true);
const MULTICLASS_ONLY: Capabilities = Capabilities::NONE.with_multiclass(true);

#[test]
fn linear_estimators_declare_weighted_fitting_and_persistence() {
    assert_eq!(
        LogisticRegression::CAPABILITIES,
        WEIGHTED_AND_PERSISTED.with_multiclass(true),
        "logistic regression is the one linear model that also fits a class set"
    );
    assert_eq!(LinearRegression::CAPABILITIES, WEIGHTED_AND_PERSISTED);
    assert_eq!(Ridge::CAPABILITIES, WEIGHTED_AND_PERSISTED);
}

#[test]
fn the_standard_scaler_declares_weighted_fitting_and_persistence() {
    assert_eq!(StandardScaler::CAPABILITIES, WEIGHTED_AND_PERSISTED);
}

#[test]
fn range_scalers_declare_persistence_but_not_weighted_fitting() {
    // Minima and maxima are order statistics, so a per-sample weight cannot
    // move them and there is no weighted entry point to declare.
    assert_eq!(MinMaxScaler::CAPABILITIES, PERSISTED_ONLY);
    assert_eq!(MaxAbsScaler::CAPABILITIES, PERSISTED_ONLY);
}

#[test]
fn tree_ensembles_declare_persistence_but_not_weighted_fitting() {
    assert_eq!(RandomForestRegressor::CAPABILITIES, PERSISTED_ONLY);
    assert_eq!(HistGradientBoostingRegressor::CAPABILITIES, PERSISTED_ONLY);
}

#[test]
fn the_forest_classifier_declares_multiclass_fitting_only() {
    // It fits an arbitrary class set, but has no weighted entry point and no
    // artifact kind until its leaf representation is persisted.
    assert_eq!(RandomForestClassifier::CAPABILITIES, MULTICLASS_ONLY);
}

#[test]
fn multiclass_fitting_is_declared_by_the_types_that_offer_it() {
    // A capability that never varies is not a capability, so this is the pair
    // that makes the field worth having.
    assert!(LogisticRegression::CAPABILITIES.multiclass());
    assert!(RandomForestClassifier::CAPABILITIES.multiclass());
    assert!(!DummyClassifier::CAPABILITIES.multiclass());
    // A composition whose `fit` takes binary targets does not offer it, and
    // the intersection says so without anyone maintaining a second table.
    assert!(
        !<Pipeline<StandardScaler, LogisticRegression> as HasCapabilities>::CAPABILITIES
            .multiclass()
    );
}

#[test]
fn baseline_estimators_declare_nothing() {
    // A baseline is refitted rather than persisted and has no weighted entry
    // point, so the conservative default is the whole truth.
    assert_eq!(DummyClassifier::CAPABILITIES, Capabilities::NONE);
    assert_eq!(DummyRegressor::CAPABILITIES, Capabilities::NONE);
}

#[test]
fn a_calibrated_composition_declares_only_what_its_calibrator_gives_it() {
    // Both compositions own already-fitted parts, so neither has a weighted
    // entry point, an artifact kind, or a multiclass fit — declared away
    // structurally rather than intersected from the wrapped model, which would
    // have promised entry points the wrapper does not have at all.
    assert_eq!(
        <CalibratedClassifier<RandomForestClassifier, IsotonicRegression> as HasCapabilities>::CAPABILITIES,
        Capabilities::NONE
    );
    // The parametric calibrator does add one thing the wrapped model never had:
    // a raw decision score, `slope * score + intercept`, whose sigmoid is the
    // calibrated probability. That is what makes this field vary rather than
    // being a constant dressed up as a capability.
    assert_eq!(
        <CalibratedClassifier<RandomForestClassifier, PlattCalibrator> as HasCapabilities>::CAPABILITIES,
        Capabilities::NONE.with_decision_function(true)
    );
}

#[test]
fn a_decision_function_is_declared_by_nothing_that_only_produces_probabilities() {
    // Every fitted estimator FerricML ships today declares no decision
    // function, including the one type that has the method. `LogisticRegression`
    // is the gap and is recorded here rather than left to be discovered:
    // its declaration lives beside the estimator, so the consumer that needed
    // the field could not also land the declaration.
    assert!(!LogisticRegression::CAPABILITIES.decision_function());
    assert!(!RandomForestClassifier::CAPABILITIES.decision_function());
    assert!(!DummyClassifier::CAPABILITIES.decision_function());
    assert!(!AnyClassifier::CAPABILITIES.decision_function());
    assert!(!AnyRegressor::CAPABILITIES.decision_function());
    assert!(!IsotonicRegression::CAPABILITIES.decision_function());
}

#[test]
fn the_monotone_calibrator_declares_nothing() {
    // A monotone map of one score has no weighted entry point, no artifact
    // kind, and no class set to widen, so the conservative default is the whole
    // truth rather than an omission.
    assert_eq!(IsotonicRegression::CAPABILITIES, Capabilities::NONE);
}

#[test]
fn runtime_dispatch_declares_only_what_every_variant_offers() {
    // The forest classifier does not persist, so neither can the enum that may
    // be holding one. Multiclass *fitting* is declared away structurally rather
    // than intersected: both variants offer it, but the enum owns fitted models
    // and no fitting entry point, so an intersection would have promised an
    // entry point that does not exist.
    assert_eq!(AnyClassifier::CAPABILITIES, Capabilities::NONE);
    assert!(LogisticRegression::CAPABILITIES.multiclass());
    assert!(RandomForestClassifier::CAPABILITIES.multiclass());
    assert!(!AnyClassifier::CAPABILITIES.multiclass());
    // Every regressor variant persists, so the enum does too.
    assert_eq!(AnyRegressor::CAPABILITIES, PERSISTED_ONLY);
}

#[test]
fn runtime_dispatch_reports_the_selected_variant_without_type_matching() {
    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();

    let ridge: AnyRegressor = Ridge::fit(&data.as_view(), &regression, RidgeParams::default())
        .unwrap()
        .into();
    let forest: AnyRegressor =
        RandomForestRegressor::fit(&data.as_view(), &regression, Default::default())
            .unwrap()
            .into();

    // A caller holding this particular model learns more than dispatch can
    // promise in general: this one could be refitted with weights.
    assert_eq!(ridge.capabilities(), WEIGHTED_AND_PERSISTED);
    assert!(!AnyRegressor::CAPABILITIES.sample_weights());
    assert_eq!(forest.capabilities(), PERSISTED_ONLY);

    let logistic: AnyClassifier = LogisticRegression::fit(
        &data.as_view(),
        &binary,
        LogisticRegressionParams::default(),
    )
    .unwrap()
    .into();
    let forest_classifier: AnyClassifier = RandomForestClassifier::fit(
        &data.as_view(),
        &binary,
        RandomForestClassifierParams::default(),
    )
    .unwrap()
    .into();
    assert_eq!(
        logistic.capabilities(),
        WEIGHTED_AND_PERSISTED.with_multiclass(true)
    );
    assert_eq!(forest_classifier.capabilities(), MULTICLASS_ONLY);
}

#[test]
fn declared_persistence_matches_the_artifact_entry_points_that_exist() {
    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let regressor: AnyRegressor = LinearRegression::fit(
        &data.as_view(),
        &regression,
        LinearRegressionParams::default(),
    )
    .unwrap()
    .into();

    assert!(AnyRegressor::CAPABILITIES.artifact());
    let bytes = regressor.to_artifact([5; 32]).unwrap();
    assert_eq!(
        AnyRegressor::from_artifact(&bytes, [5; 32]).unwrap(),
        regressor
    );

    // `AnyClassifier` has no artifact entry point at all, which is what its
    // declaration says. Requesting one is a compile error, not a wrong answer.
    assert!(!AnyClassifier::CAPABILITIES.artifact());
}

#[test]
fn a_fitted_composition_persists_when_both_parts_do() {
    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
    let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default()).unwrap();
    let transformed = scaler.transform(&data.as_view()).unwrap();
    let estimator = LogisticRegression::fit(
        &transformed.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let pipeline = Pipeline::new(scaler, estimator).unwrap();

    assert_eq!(
        <Pipeline<StandardScaler, LogisticRegression> as HasCapabilities>::CAPABILITIES,
        PERSISTED_ONLY
    );
    assert_eq!(
        <Pipeline<StandardScaler, Ridge> as HasCapabilities>::CAPABILITIES,
        PERSISTED_ONLY
    );
    assert_eq!(
        <Pipeline<StandardScaler, LinearRegression> as HasCapabilities>::CAPABILITIES,
        PERSISTED_ONLY
    );
    // Composing already-fitted parts cannot accept weights even though both
    // parts can be fitted with them.
    assert!(StandardScaler::CAPABILITIES.sample_weights());
    assert!(LogisticRegression::CAPABILITIES.sample_weights());
    assert!(
        !<Pipeline<StandardScaler, LogisticRegression> as HasCapabilities>::CAPABILITIES
            .sample_weights()
    );
    assert!(pipeline.to_artifact([1; 32], [2; 32]).is_ok());
}

#[test]
fn a_multi_stage_composition_persists_when_every_part_does() {
    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let staged: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> = StagedPipeline::fit(
        &data.as_view(),
        |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
        |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
        |batch| Ridge::fit(batch, &regression, RidgeParams::default()),
    )
    .unwrap();

    // One declaration covers every composition whose parts all persist,
    // instead of one hand-written declaration per concrete composition.
    assert_eq!(
        <StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> as HasCapabilities>::CAPABILITIES,
        PERSISTED_ONLY
    );
    assert_eq!(
        <StagedPipeline<(StandardScaler, MaxAbsScaler, MinMaxScaler), LinearRegression> as
            HasCapabilities>::CAPABILITIES,
        PERSISTED_ONLY
    );
    // Composing already-fitted parts still cannot accept weights, even though
    // two of these three parts can be fitted with them.
    assert!(StandardScaler::CAPABILITIES.sample_weights());
    assert!(Ridge::CAPABILITIES.sample_weights());
    assert!(
        !<StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> as HasCapabilities>::CAPABILITIES
            .sample_weights()
    );
    assert!(staged.to_artifact([1; 32], [2; 32]).is_ok());
}
