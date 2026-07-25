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
//!
//! Three mechanisms, deliberately not merged. `tests/capability_snapshot.rs`
//! is the mechanical change detector — exact values, diffed beside the frozen
//! API profile, and closed against it so no declaration escapes both. This
//! file is the *reasoned* record: why each declaration is what it is, which a
//! generated table cannot carry. The battery is the behavioral proof. A
//! reviewer reads the snapshot diff; a maintainer reads the reasoning here.

use ferricml::api::{AnyClassifier, AnyRegressor, Capabilities, HasCapabilities};
use ferricml::calibration::{CalibratedClassifier, IsotonicRegression, PlattCalibrator};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::dummy::{DummyClassifier, DummyRegressor};
use ferricml::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor,
};
use ferricml::linear_model::{
    ElasticNet, Lasso, LinearRegression, LinearRegressionParams, LogisticRegression,
    LogisticRegressionParams, Ridge, RidgeParams,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    MaxAbsScaler, MinMaxScaler, MinMaxScalerParams, StandardScaler, StandardScalerParams,
};
use ferricml::ranking::PairwiseLinearRanker;

const WEIGHTED_AND_PERSISTED: Capabilities = Capabilities::NONE
    .with_sample_weights(true)
    .with_artifact(true);
const PERSISTED_ONLY: Capabilities = Capabilities::NONE.with_artifact(true);

#[test]
fn linear_estimators_declare_weighted_fitting_and_persistence() {
    assert_eq!(
        LogisticRegression::CAPABILITIES,
        WEIGHTED_AND_PERSISTED
            .with_multiclass(true)
            .with_decision_function(true),
        "logistic regression is the one linear model that also fits a class set, \
         and the only shipped estimator exposing a raw unsquashed score"
    );
    assert_eq!(LinearRegression::CAPABILITIES, WEIGHTED_AND_PERSISTED);
    assert_eq!(Ridge::CAPABILITIES, WEIGHTED_AND_PERSISTED);
}

#[test]
fn the_penalized_regressors_declare_weighted_fitting_but_no_artifact_yet() {
    // Persistence is a separate contract from semantics: these estimators have
    // no artifact kind, so declaring one they do not have would promise bytes
    // that cannot be written. Flipping this is a reviewable diff, which is the
    // point of stating it here.
    assert_eq!(
        Lasso::CAPABILITIES,
        Capabilities::NONE.with_sample_weights(true)
    );
    assert_eq!(
        ElasticNet::CAPABILITIES,
        Capabilities::NONE.with_sample_weights(true)
    );
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
fn tree_ensembles_declare_weighted_fitting_and_persistence() {
    assert_eq!(RandomForestRegressor::CAPABILITIES, WEIGHTED_AND_PERSISTED);
    assert_eq!(
        HistGradientBoostingRegressor::CAPABILITIES,
        WEIGHTED_AND_PERSISTED
    );
}

#[test]
fn the_boosted_classifier_declares_weighted_fitting_persistence_and_a_decision_score() {
    // Binary log loss is fitted in raw-score space, so the additive score the
    // trees sum into is the model's own quantity rather than something derived
    // for the declaration's sake. Multiclass is absent because a multiclass
    // boosted model needs a different objective and one tree per class per
    // iteration, not a wider fit of this one.
    assert_eq!(
        HistGradientBoostingClassifier::CAPABILITIES,
        WEIGHTED_AND_PERSISTED.with_decision_function(true)
    );
    assert!(!HistGradientBoostingClassifier::CAPABILITIES.multiclass());
}

#[test]
fn the_forest_classifier_declares_weighted_and_multiclass_fitting_and_persistence() {
    // Its artifact covers both leaf representations, so persistence holds for
    // every fit the type offers rather than for one of its two entry points.
    assert_eq!(
        RandomForestClassifier::CAPABILITIES,
        WEIGHTED_AND_PERSISTED.with_multiclass(true)
    );
}

#[test]
fn multiclass_fitting_is_declared_by_the_types_that_offer_it() {
    // A capability that never varies is not a capability, so this is the pair
    // that makes the field worth having.
    assert!(LogisticRegression::CAPABILITIES.multiclass());
    assert!(RandomForestClassifier::CAPABILITIES.multiclass());
    assert!(!DummyClassifier::CAPABILITIES.multiclass());
    assert!(!HistGradientBoostingClassifier::CAPABILITIES.multiclass());
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
    // `LogisticRegression` and the boosted classifier are the shipped types
    // whose model is defined in raw-score space, and both declare one. The
    // logistic declaration was applied by the coordinator at merge rather than
    // by the sprint that added the field: declarations live beside their
    // estimator, so the consumer that needed the capability could not also land
    // the declaration from another track. Every other estimator only produces
    // probabilities, which is not what this field records.
    assert!(LogisticRegression::CAPABILITIES.decision_function());
    assert!(HistGradientBoostingClassifier::CAPABILITIES.decision_function());
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
fn the_ranker_declares_persistence_and_nothing_else() {
    // The one fitted estimator that could never answer a capability query, and
    // therefore the one every meta-layer would have had to special-case.
    //
    // Weighted fitting is absent because a pairwise weight belongs to a *pair
    // observation*, not to a row of the item matrix, so there is no
    // `SampleWeights` entry point to declare. A decision function is absent for
    // the reason that keeps that field honest: it records that a classifier
    // exposes a raw score whose squashing is its probability, and a ranker has
    // no probability to squash to — raw scores and pair margins are not
    // probabilities.
    assert_eq!(PairwiseLinearRanker::CAPABILITIES, PERSISTED_ONLY);
    assert!(!PairwiseLinearRanker::CAPABILITIES.sample_weights());
    assert!(!PairwiseLinearRanker::CAPABILITIES.decision_function());
    assert!(!PairwiseLinearRanker::CAPABILITIES.multiclass());
}

#[test]
fn runtime_dispatch_declares_only_what_every_variant_offers() {
    // Both classifier variants persist every fit they offer, so the enum does
    // too. Multiclass *fitting* is declared away structurally rather than
    // intersected: both variants offer it, but the enum owns fitted models and
    // no fitting entry point, so an intersection would have promised an entry
    // point that does not exist.
    assert_eq!(AnyClassifier::CAPABILITIES, PERSISTED_ONLY);
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
    assert_eq!(forest.capabilities(), WEIGHTED_AND_PERSISTED);

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
        WEIGHTED_AND_PERSISTED
            .with_multiclass(true)
            .with_decision_function(true)
    );
    assert_eq!(
        forest_classifier.capabilities(),
        WEIGHTED_AND_PERSISTED.with_multiclass(true)
    );
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

    // The declaration is a property of the type, so it covers every fit the
    // type offers. Logistic regression declares both persistence and
    // multiclass fitting, and the joint fit persists under its own payload
    // schema rather than being refused.
    let classes = ferricml::data::ClassTargets::new(vec![3, 7, 10, 7]).unwrap();
    let multiclass = LogisticRegression::fit_multiclass(
        &data.as_view(),
        &classes,
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let bytes = multiclass.to_artifact([5; 32]).unwrap();
    let decoded = LogisticRegression::from_artifact(&bytes, [5; 32]).unwrap();
    assert_eq!(decoded, multiclass);
    assert_eq!(decoded.classes(), [3, 7, 10]);

    // The classifier dispatch enum persists too, and restores the runtime
    // variant along with the payload schema that variant chose for itself.
    assert!(AnyClassifier::CAPABILITIES.artifact());
    let erased: AnyClassifier = multiclass.into();
    let bytes = erased.to_artifact([5; 32]).unwrap();
    let decoded = AnyClassifier::from_artifact(&bytes, [5; 32]).unwrap();
    assert_eq!(decoded, erased);
    assert_eq!(decoded.classes(), [3, 7, 10]);
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
