//! Prediction semantics.
//!
//! The obligations every fitted estimator shares are stated once in the generic
//! conformance battery (`support::conformance`); this file registers each
//! estimator into it and keeps only the semantics that belong to one estimator
//! and cannot be expressed generically — how a tie is broken, how a single-class
//! model is shaped, how a particular estimator is provoked into overflowing, and
//! what runtime dispatch adds on top of the trait contract.
//!
//! Adding an estimator means adding one case implementation and one line to the
//! registration list below. Do not add per-estimator repeats of a shared
//! obligation here; add the obligation to the battery instead, together with the
//! probe that proves it can fail.

mod support;

use ferricml::api::{
    AnyClassifier, AnyClassifierParams, AnyRegressor, AnyRegressorParams, Classifier, ModelError,
    ProbabilisticClassifier, Regressor,
};
use ferricml::artifact::{ArtifactError, ModelArtifact, StageArtifact};
use ferricml::calibration::{
    CalibratedClassifier, IsotonicRegression, PlattCalibrator, PlattParams,
};
use ferricml::data::{BinaryTargets, DenseMatrix, MatrixView, RegressionTargets};
use ferricml::dummy::{
    DummyClassifier, DummyClassifierParams, DummyRegressor, DummyRegressorParams,
};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, ExtraTreesRegressor,
    ExtraTreesRegressorParams, HistGradientBoostingClassifier,
    HistGradientBoostingClassifierParams, HistGradientBoostingRegressor,
    HistGradientBoostingRegressorParams, MaxFeatures, NJobs, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor, RandomForestRegressorParams,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::tree::{
    DecisionTreeClassifier, DecisionTreeClassifierParams, DecisionTreeRegressor,
    DecisionTreeRegressorParams,
};

use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LinearRegression, LinearRegressionParams,
    LogisticRegression, LogisticRegressionParams, Ridge, RidgeParams,
};
use ferricml::preprocessing::{
    Binarizer, BinarizerParams, FunctionTransformer, FunctionTransformerParams, MaxAbsScaler,
    MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams, Normalizer, NormalizerParams,
    RobustScaler, RobustScalerParams, StandardScaler, StandardScalerParams,
};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseError, PairwiseLinearRanker, PairwiseLinearRankerParams,
    PairwiseObservation,
};

use support::conformance::{
    ClassifierCase, Fixture, FixtureShape, OptionalFit, RegressorCase, RoundTrip, SCHEMA, Sample,
    ScalarClassifierCase, ScalarRegressorCase, ScalarWorkspaceRegressorCase, TransformerCase,
    WorkspaceClassifierCase, WorkspaceRegressorCase, check_batch_only_classifier,
    check_batch_only_regressor, check_classifier, check_regressor,
    check_scalar_workspace_regressor, check_transformer, check_workspace_classifier,
    check_workspace_regressor, probabilistic_hooks,
};

/// Second schema identity, for artifacts that bind an input and an output
/// schema.
const TRANSFORMED_SCHEMA: [u8; 32] = [43; 32];

fn round_trip<M>(
    encode: impl FnOnce() -> Result<Vec<u8>, ArtifactError>,
    decode: impl FnOnce(&[u8]) -> Result<M, ArtifactError>,
) -> RoundTrip<M> {
    Some((|| {
        let bytes = encode()?;
        let decoded = decode(&bytes)?;
        Ok((bytes, decoded))
    })())
}

fn forest_classifier_params() -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(5)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

fn forest_regressor_params() -> RandomForestRegressorParams {
    RandomForestRegressorParams::default()
        .with_n_estimators(5)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

/// One tree grown under the forest members' own limits.
///
/// `MaxFeatures::All` and a fixed seed keep the fit reproducible without making
/// the battery depend on how many columns the shared fixture happens to have.
fn tree_classifier_params() -> DecisionTreeClassifierParams {
    DecisionTreeClassifierParams::default()
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

fn tree_regressor_params() -> DecisionTreeRegressorParams {
    DecisionTreeRegressorParams::default()
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

/// The randomized ensemble under the same shape the forest cases use, so a
/// difference between the two registrations is a difference in the estimator
/// rather than in its workload.
fn extra_trees_classifier_params() -> ExtraTreesClassifierParams {
    ExtraTreesClassifierParams::default()
        .with_n_estimators(5)
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

fn extra_trees_regressor_params() -> ExtraTreesRegressorParams {
    ExtraTreesRegressorParams::default()
        .with_n_estimators(5)
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

fn boosting_params() -> HistGradientBoostingRegressorParams {
    HistGradientBoostingRegressorParams::default()
        .with_max_iter(3)
        .with_max_leaf_nodes(2)
        .with_min_samples_leaf(1)
}

fn boosting_classifier_params() -> HistGradientBoostingClassifierParams {
    HistGradientBoostingClassifierParams::default()
        .with_max_iter(3)
        .with_max_leaf_nodes(2)
        .with_min_samples_leaf(1)
}

// ---------------------------------------------------------- registered cases

struct RandomForestClassifierCase;

impl ClassifierCase for RandomForestClassifierCase {
    probabilistic_hooks!();

    type Model = RandomForestClassifier;
    const NAME: &'static str = "RandomForestClassifier";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        RandomForestClassifier::fit(&train.view(), &train.labels, forest_classifier_params())
    }

    fn fit_multiclass(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(RandomForestClassifier::fit_multiclass(
            &train.view(),
            &train.class_labels,
            forest_classifier_params(),
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(RandomForestClassifier::fit_weighted(
            &train.view(),
            &train.labels,
            &train.unit_weights(),
            forest_classifier_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| RandomForestClassifier::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarClassifierCase for RandomForestClassifierCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        model.predict_one(row)
    }
}

struct ExtraTreesClassifierCase;

impl ClassifierCase for ExtraTreesClassifierCase {
    probabilistic_hooks!();

    type Model = ExtraTreesClassifier;
    const NAME: &'static str = "ExtraTreesClassifier";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        ExtraTreesClassifier::fit(
            &train.view(),
            &train.labels,
            extra_trees_classifier_params(),
        )
    }

    fn fit_multiclass(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(ExtraTreesClassifier::fit_multiclass(
            &train.view(),
            &train.class_labels,
            extra_trees_classifier_params(),
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(ExtraTreesClassifier::fit_weighted(
            &train.view(),
            &train.labels,
            &train.unit_weights(),
            extra_trees_classifier_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| ExtraTreesClassifier::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarClassifierCase for ExtraTreesClassifierCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        model.predict_one(row)
    }
}

struct ExtraTreesRegressorCase;

impl RegressorCase for ExtraTreesRegressorCase {
    type Model = ExtraTreesRegressor;
    const NAME: &'static str = "ExtraTreesRegressor";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        ExtraTreesRegressor::fit(&train.view(), &train.values, extra_trees_regressor_params())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(ExtraTreesRegressor::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            extra_trees_regressor_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| ExtraTreesRegressor::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarRegressorCase for ExtraTreesRegressorCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct DecisionTreeClassifierCase;

impl ClassifierCase for DecisionTreeClassifierCase {
    probabilistic_hooks!();

    type Model = DecisionTreeClassifier;
    const NAME: &'static str = "DecisionTreeClassifier";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        DecisionTreeClassifier::fit(&train.view(), &train.labels, tree_classifier_params())
    }

    fn fit_multiclass(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(DecisionTreeClassifier::fit_multiclass(
            &train.view(),
            &train.class_labels,
            tree_classifier_params(),
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(DecisionTreeClassifier::fit_weighted(
            &train.view(),
            &train.labels,
            &train.unit_weights(),
            tree_classifier_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| DecisionTreeClassifier::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarClassifierCase for DecisionTreeClassifierCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        model.predict_one(row)
    }
}

struct DecisionTreeRegressorCase;

impl RegressorCase for DecisionTreeRegressorCase {
    type Model = DecisionTreeRegressor;
    const NAME: &'static str = "DecisionTreeRegressor";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        DecisionTreeRegressor::fit(&train.view(), &train.values, tree_regressor_params())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(DecisionTreeRegressor::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            tree_regressor_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| DecisionTreeRegressor::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarRegressorCase for DecisionTreeRegressorCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct LogisticRegressionCase;

impl ClassifierCase for LogisticRegressionCase {
    probabilistic_hooks!();

    type Model = LogisticRegression;
    const NAME: &'static str = "LogisticRegression";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        LogisticRegression::fit(
            &train.view(),
            &train.labels,
            LogisticRegressionParams::default(),
        )
    }

    fn fit_multiclass(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(LogisticRegression::fit_multiclass(
            &train.view(),
            &train.class_labels,
            LogisticRegressionParams::default(),
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(LogisticRegression::fit_weighted(
            &train.view(),
            &train.labels,
            &train.unit_weights(),
            LogisticRegressionParams::default(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| LogisticRegression::from_artifact(bytes, SCHEMA),
        )
    }

    fn decision_function(
        model: &Self::Model,
        data: &MatrixView<'_>,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        Some(model.decision_function(data))
    }

    fn decision_function_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        Some(model.decision_function_into(data, output))
    }
}

impl ScalarClassifierCase for LogisticRegressionCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        model.predict_one(row)
    }
}

macro_rules! any_classifier_case {
    ($case:ident, $inner:ty, $name:literal) => {
        struct $case;

        impl ClassifierCase for $case {
            type Model = AnyClassifier;
            const NAME: &'static str = $name;

            fn predict_proba(
                model: &Self::Model,
                data: &MatrixView<'_>,
            ) -> Option<Result<Vec<f32>, ModelError>> {
                model
                    .as_probabilistic()
                    .map(|model| model.predict_proba(data))
            }

            fn predict_proba_into(
                model: &Self::Model,
                data: &MatrixView<'_>,
                output: &mut [f32],
            ) -> Option<Result<(), ModelError>> {
                model
                    .as_probabilistic()
                    .map(|model| model.predict_proba_into(data, output))
            }

            fn predict_class_proba(
                model: &Self::Model,
                data: &MatrixView<'_>,
                class: u8,
            ) -> Option<Result<Vec<f32>, ModelError>> {
                model
                    .as_probabilistic()
                    .map(|model| model.predict_class_proba(data, class))
            }

            fn predict_class_proba_into(
                model: &Self::Model,
                data: &MatrixView<'_>,
                class: u8,
                output: &mut [f32],
            ) -> Option<Result<(), ModelError>> {
                model
                    .as_probabilistic()
                    .map(|model| model.predict_class_proba_into(data, class, output))
            }

            fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
                <$inner as ClassifierCase>::fit(train, holdout).map(Into::into)
            }

            fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
                round_trip(
                    || model.to_artifact(SCHEMA),
                    |bytes| AnyClassifier::from_artifact(bytes, SCHEMA),
                )
            }
        }
    };
}

any_classifier_case!(
    AnyForestClassifierCase,
    RandomForestClassifierCase,
    "AnyClassifier::RandomForest"
);
any_classifier_case!(
    AnyLogisticClassifierCase,
    LogisticRegressionCase,
    "AnyClassifier::LogisticRegression"
);
any_classifier_case!(
    AnyBoostedClassifierCase,
    HistGradientBoostingClassifierCase,
    "AnyClassifier::HistGradientBoosting"
);

/// The two calibrated compositions, registered like any other classifier.
///
/// The inner forest is fitted on `train` and the calibrator on `holdout`,
/// which is what the composition's own documentation requires: a calibrator
/// fitted on the rows its model already memorised measures that memory rather
/// than the model's probabilities. The battery supplies the second sample, so
/// the registration is honest rather than a structural approximation.
struct IsotonicCalibratedForestCase;

impl ClassifierCase for IsotonicCalibratedForestCase {
    probabilistic_hooks!();

    type Model = CalibratedClassifier<RandomForestClassifier, IsotonicRegression>;
    const NAME: &'static str = "CalibratedClassifier<RandomForestClassifier, IsotonicRegression>";

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        CalibratedClassifier::fit_isotonic(
            RandomForestClassifierCase::fit(train, holdout)?,
            &holdout.view(),
            &holdout.labels,
        )
    }
}

struct PlattCalibratedForestCase;

impl ClassifierCase for PlattCalibratedForestCase {
    probabilistic_hooks!();

    type Model = CalibratedClassifier<RandomForestClassifier, PlattCalibrator>;
    const NAME: &'static str = "CalibratedClassifier<RandomForestClassifier, PlattCalibrator>";

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        CalibratedClassifier::fit_platt(
            RandomForestClassifierCase::fit(train, holdout)?,
            &holdout.view(),
            &holdout.labels,
            PlattParams::default(),
        )
    }

    fn decision_function(
        model: &Self::Model,
        data: &MatrixView<'_>,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        Some(model.decision_function(data))
    }

    fn decision_function_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        Some(model.decision_function_into(data, output))
    }
}

struct RandomForestRegressorCase;

impl RegressorCase for RandomForestRegressorCase {
    type Model = RandomForestRegressor;
    const NAME: &'static str = "RandomForestRegressor";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        RandomForestRegressor::fit(&train.view(), &train.values, forest_regressor_params())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(RandomForestRegressor::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            forest_regressor_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| RandomForestRegressor::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarRegressorCase for RandomForestRegressorCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct LinearRegressionCase;

impl RegressorCase for LinearRegressionCase {
    type Model = LinearRegression;
    const NAME: &'static str = "LinearRegression";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        LinearRegression::fit(
            &train.view(),
            &train.values,
            LinearRegressionParams::default(),
        )
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(LinearRegression::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            LinearRegressionParams::default(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| LinearRegression::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarRegressorCase for LinearRegressionCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct RidgeCase;

impl RegressorCase for RidgeCase {
    type Model = Ridge;
    const NAME: &'static str = "Ridge";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        Ridge::fit(&train.view(), &train.values, RidgeParams::default())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(Ridge::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            RidgeParams::default(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| Ridge::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarRegressorCase for RidgeCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct LassoCase;

impl RegressorCase for LassoCase {
    type Model = Lasso;
    const NAME: &'static str = "Lasso";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        Lasso::fit(&train.view(), &train.values, lasso_params())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(Lasso::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            lasso_params(),
        ))
    }
}

impl ScalarRegressorCase for LassoCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

/// A penalty small enough that the fixture keeps a non-degenerate fit, and a
/// sweep budget large enough that the battery never observes a refusal.
fn lasso_params() -> LassoParams {
    LassoParams::default()
        .with_alpha(0.01)
        .with_max_iter(10_000)
}

struct ElasticNetCase;

impl RegressorCase for ElasticNetCase {
    type Model = ElasticNet;
    const NAME: &'static str = "ElasticNet";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        ElasticNet::fit(&train.view(), &train.values, elastic_net_params())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(ElasticNet::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            elastic_net_params(),
        ))
    }
}

impl ScalarRegressorCase for ElasticNetCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

fn elastic_net_params() -> ElasticNetParams {
    ElasticNetParams::default()
        .with_alpha(0.01)
        .with_max_iter(10_000)
}

struct HistGradientBoostingRegressorCase;

impl RegressorCase for HistGradientBoostingRegressorCase {
    type Model = HistGradientBoostingRegressor;
    const NAME: &'static str = "HistGradientBoostingRegressor";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        HistGradientBoostingRegressor::fit(&train.view(), &train.values, boosting_params())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(HistGradientBoostingRegressor::fit_weighted(
            &train.view(),
            &train.values,
            &train.unit_weights(),
            boosting_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| HistGradientBoostingRegressor::from_artifact(bytes, SCHEMA),
        )
    }
}

impl ScalarRegressorCase for HistGradientBoostingRegressorCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct HistGradientBoostingClassifierCase;

impl ClassifierCase for HistGradientBoostingClassifierCase {
    probabilistic_hooks!();

    type Model = HistGradientBoostingClassifier;
    const NAME: &'static str = "HistGradientBoostingClassifier";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        HistGradientBoostingClassifier::fit(
            &train.view(),
            &train.labels,
            boosting_classifier_params(),
        )
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(HistGradientBoostingClassifier::fit_weighted(
            &train.view(),
            &train.labels,
            &train.unit_weights(),
            boosting_classifier_params(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| HistGradientBoostingClassifier::from_artifact(bytes, SCHEMA),
        )
    }

    fn decision_function(
        model: &Self::Model,
        data: &MatrixView<'_>,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        Some(model.decision_function(data))
    }

    fn decision_function_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        Some(model.decision_function_into(data, output))
    }
}

impl ScalarClassifierCase for HistGradientBoostingClassifierCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        model.predict_one(row)
    }
}

macro_rules! any_regressor_case {
    ($case:ident, $inner:ty, $name:literal) => {
        struct $case;

        impl RegressorCase for $case {
            type Model = AnyRegressor;
            const NAME: &'static str = $name;

            fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
                <$inner as RegressorCase>::fit(train, holdout).map(Into::into)
            }

            fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
                round_trip(
                    || model.to_artifact(SCHEMA),
                    |bytes| AnyRegressor::from_artifact(bytes, SCHEMA),
                )
            }
        }
    };
}

any_regressor_case!(
    AnyForestRegressorCase,
    RandomForestRegressorCase,
    "AnyRegressor::RandomForest"
);
any_regressor_case!(
    AnyLinearRegressorCase,
    LinearRegressionCase,
    "AnyRegressor::LinearRegression"
);
any_regressor_case!(AnyRidgeRegressorCase, RidgeCase, "AnyRegressor::Ridge");
any_regressor_case!(
    AnyBoostedRegressorCase,
    HistGradientBoostingRegressorCase,
    "AnyRegressor::HistGradientBoosting"
);

struct DummyClassifierCase;

impl ClassifierCase for DummyClassifierCase {
    probabilistic_hooks!();

    type Model = DummyClassifier;
    const NAME: &'static str = "DummyClassifier";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        DummyClassifier::fit(&train.view(), &train.labels, DummyClassifierParams)
    }
}

impl ScalarClassifierCase for DummyClassifierCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        model.predict_one(row)
    }
}

struct DummyRegressorCase;

impl RegressorCase for DummyRegressorCase {
    type Model = DummyRegressor;
    const NAME: &'static str = "DummyRegressor";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        DummyRegressor::fit(&train.view(), &train.values, DummyRegressorParams)
    }
}

impl ScalarRegressorCase for DummyRegressorCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

struct StandardScalerCase;

impl TransformerCase for StandardScalerCase {
    type Model = StandardScaler;
    const NAME: &'static str = "StandardScaler";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        StandardScaler::fit(&train.view(), StandardScalerParams::default())
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        Some(StandardScaler::fit_weighted(
            &train.view(),
            &train.unit_weights(),
            StandardScalerParams::default(),
        ))
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| StandardScaler::from_artifact(bytes, SCHEMA, TRANSFORMED_SCHEMA),
        )
    }
}

struct MinMaxScalerCase;

impl TransformerCase for MinMaxScalerCase {
    type Model = MinMaxScaler;
    const NAME: &'static str = "MinMaxScaler";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        MinMaxScaler::fit(&train.view(), MinMaxScalerParams::default())
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| MinMaxScaler::from_artifact(bytes, SCHEMA, TRANSFORMED_SCHEMA),
        )
    }
}

struct MaxAbsScalerCase;

impl TransformerCase for MaxAbsScalerCase {
    type Model = MaxAbsScaler;
    const NAME: &'static str = "MaxAbsScaler";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        MaxAbsScaler::fit(&train.view(), MaxAbsScalerParams)
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| MaxAbsScaler::from_artifact(bytes, SCHEMA, TRANSFORMED_SCHEMA),
        )
    }
}

/// A monotone map of one feature, registered on the univariate fixture.
///
/// It is a real `Regressor` and could not be registered at all while the
/// battery had one eight-by-two dataset: a univariate estimator is required to
/// reject a wider matrix, so the only shape on offer was one it must refuse.
struct RobustScalerCase;

impl TransformerCase for RobustScalerCase {
    type Model = RobustScaler;
    const NAME: &'static str = "RobustScaler";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        RobustScaler::fit(&train.view(), RobustScalerParams::default())
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| RobustScaler::from_artifact(bytes, SCHEMA, TRANSFORMED_SCHEMA),
        )
    }
}

/// Stateless, so it supplies no round-trip hook and must declare none.
struct NormalizerCase;

impl TransformerCase for NormalizerCase {
    type Model = Normalizer;
    const NAME: &'static str = "Normalizer";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        Normalizer::fit(&train.view(), NormalizerParams::default())
    }
}

/// Stateless, so it supplies no round-trip hook and must declare none.
struct BinarizerCase;

impl TransformerCase for BinarizerCase {
    type Model = Binarizer;
    const NAME: &'static str = "Binarizer";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        Binarizer::fit(
            &train.view(),
            BinarizerParams::default().with_threshold(3.5),
        )
    }
}

/// Registered at a named `fn`, so the case type is concrete and the
/// declaration-versus-behaviour check runs exactly as it does for every other
/// transformer. A closure could not be named here at all.
struct FunctionTransformerCase;

fn double(value: f32) -> f32 {
    value * 2.0
}

impl TransformerCase for FunctionTransformerCase {
    type Model = FunctionTransformer;
    const NAME: &'static str = "FunctionTransformer";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        FunctionTransformer::fit(
            &train.view(),
            FunctionTransformerParams::default().with_func(double),
        )
    }
}

struct IsotonicRegressionCase;

impl RegressorCase for IsotonicRegressionCase {
    type Model = IsotonicRegression;
    const NAME: &'static str = "IsotonicRegression";
    const FIXTURE: FixtureShape = FixtureShape::Univariate;

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        IsotonicRegression::fit(&train.view(), &train.values)
    }
}

impl ScalarRegressorCase for IsotonicRegressionCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.predict_one(row)
    }
}

/// One fitted scaler and one fitted classifier, predicted through a workspace.
struct ScaledLogisticPipelineCase;

impl WorkspaceClassifierCase for ScaledLogisticPipelineCase {
    type Model = Pipeline<StandardScaler, LogisticRegression>;
    const NAME: &'static str = "Pipeline<StandardScaler, LogisticRegression>";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        let scaler = StandardScaler::fit(&train.view(), StandardScalerParams::default())?;
        let transformed = scaler.transform(&train.view())?;
        let estimator = LogisticRegression::fit(
            &transformed.as_view(),
            &train.labels,
            LogisticRegressionParams::default(),
        )?;
        Pipeline::new(scaler, estimator)
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| {
                Pipeline::<StandardScaler, LogisticRegression>::from_artifact(
                    bytes,
                    SCHEMA,
                    TRANSFORMED_SCHEMA,
                )
            },
        )
    }

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        model.workspace_len(rows)
    }

    fn classes(model: &Self::Model) -> Vec<u8> {
        model.estimator().classes().to_vec()
    }

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<u8>, ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict(transformed)
        })
    }

    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError> {
        model.predict_into(data, workspace, output)
    }

    fn predict_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Option<Result<Vec<f32>, ModelError>> {
        Some(
            model.with_transformed(data, workspace, |estimator, transformed| {
                estimator.predict_proba(transformed)
            }),
        )
    }

    fn predict_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        Some(model.predict_proba_into(data, workspace, output))
    }

    fn predict_class_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        Some(
            model.with_transformed(data, workspace, |estimator, transformed| {
                ProbabilisticClassifier::predict_class_proba(estimator, transformed, class)
            }),
        )
    }

    fn predict_class_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        Some(model.predict_class_proba_into(data, class, workspace, output))
    }

    fn decision_function(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Option<Result<Vec<f32>, ModelError>> {
        Some(
            model.with_transformed(data, workspace, |estimator, transformed| {
                estimator.decision_function(transformed)
            }),
        )
    }

    fn decision_function_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        Some(model.decision_function_into(data, workspace, output))
    }
}

macro_rules! scaled_regression_pipeline_case {
    ($case:ident, $estimator:ty, $params:expr, $name:literal) => {
        struct $case;

        impl WorkspaceRegressorCase for $case {
            type Model = Pipeline<StandardScaler, $estimator>;
            const NAME: &'static str = $name;

            fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
                let scaler = StandardScaler::fit(&train.view(), StandardScalerParams::default())?;
                let transformed = scaler.transform(&train.view())?;
                let estimator = <$estimator>::fit(&transformed.as_view(), &train.values, $params)?;
                Pipeline::new(scaler, estimator)
            }

            fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
                round_trip(
                    || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
                    |bytes| {
                        Pipeline::<StandardScaler, $estimator>::from_artifact(
                            bytes,
                            SCHEMA,
                            TRANSFORMED_SCHEMA,
                        )
                    },
                )
            }

            fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
                model.workspace_len(rows)
            }

            fn predict(
                model: &Self::Model,
                data: &MatrixView<'_>,
                workspace: &mut [f32],
            ) -> Result<Vec<f32>, ModelError> {
                model.with_transformed(data, workspace, |estimator, transformed| {
                    estimator.predict(transformed)
                })
            }

            fn predict_into(
                model: &Self::Model,
                data: &MatrixView<'_>,
                workspace: &mut [f32],
                output: &mut [f32],
            ) -> Result<(), ModelError> {
                model.predict_into(data, workspace, output)
            }
        }
    };
}

scaled_regression_pipeline_case!(
    ScaledRidgePipelineCase,
    Ridge,
    RidgeParams::default(),
    "Pipeline<StandardScaler, Ridge>"
);
scaled_regression_pipeline_case!(
    ScaledLinearPipelineCase,
    LinearRegression,
    LinearRegressionParams::default(),
    "Pipeline<StandardScaler, LinearRegression>"
);

/// A persisted stage built this sprint, composed beside one that predates it.
///
/// This is what stage tag `4` buys: a robust scaler is a first-class stage, so
/// the whole composition still round-trips through one artifact.
struct RobustStagedPipelineCase;

impl WorkspaceRegressorCase for RobustStagedPipelineCase {
    type Model = StagedPipeline<(RobustScaler, MaxAbsScaler), Ridge>;
    const NAME: &'static str = "StagedPipeline<(RobustScaler, MaxAbsScaler), Ridge>";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        StagedPipeline::fit(
            &train.view(),
            |batch| RobustScaler::fit(batch, RobustScalerParams::default()),
            |batch| MaxAbsScaler::fit(batch, MaxAbsScalerParams),
            |batch| Ridge::fit(batch, &train.values, RidgeParams::default()),
        )
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| {
                StagedPipeline::<(RobustScaler, MaxAbsScaler), Ridge>::from_artifact(
                    bytes,
                    SCHEMA,
                    TRANSFORMED_SCHEMA,
                )
            },
        )
    }

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        model.workspace_len(rows)
    }

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict(transformed)
        })
    }

    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict_into(transformed, output)
        })
    }
}

/// Two fitted transform stages and an estimator, predicted through one
/// workspace split per stage.
struct TwoStagePipelineCase;

impl WorkspaceRegressorCase for TwoStagePipelineCase {
    type Model = StagedPipeline<(MinMaxScaler, StandardScaler), Ridge>;
    const NAME: &'static str = "StagedPipeline<(MinMaxScaler, StandardScaler), Ridge>";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        StagedPipeline::fit(
            &train.view(),
            |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
            |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
            |batch| Ridge::fit(batch, &train.values, RidgeParams::default()),
        )
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| {
                StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                    bytes,
                    SCHEMA,
                    TRANSFORMED_SCHEMA,
                )
            },
        )
    }

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        model.workspace_len(rows)
    }

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict(transformed)
        })
    }

    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict_into(transformed, output)
        })
    }
}

/// The three-stage arity, so both `TransformerStack` implementations are
/// covered by a registration rather than by a bespoke test.
struct ThreeStagePipelineCase;

impl WorkspaceRegressorCase for ThreeStagePipelineCase {
    type Model = StagedPipeline<(StandardScaler, MaxAbsScaler, MinMaxScaler), LinearRegression>;
    const NAME: &'static str =
        "StagedPipeline<(StandardScaler, MaxAbsScaler, MinMaxScaler), LinearRegression>";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        let first = StandardScaler::fit(&train.view(), StandardScalerParams::default())?;
        let after_first = first.transform(&train.view())?;
        let second = MaxAbsScaler::fit(&after_first.as_view(), MaxAbsScalerParams)?;
        let after_second = second.transform(&after_first.as_view())?;
        let third = MinMaxScaler::fit(&after_second.as_view(), MinMaxScalerParams::default())?;
        let transformed = third.transform(&after_second.as_view())?;
        let estimator = LinearRegression::fit(
            &transformed.as_view(),
            &train.values,
            LinearRegressionParams::default(),
        )?;
        StagedPipeline::new((first, second, third), estimator)
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA, TRANSFORMED_SCHEMA),
            |bytes| {
                StagedPipeline::<
                    (StandardScaler, MaxAbsScaler, MinMaxScaler),
                    LinearRegression,
                >::from_artifact(bytes, SCHEMA, TRANSFORMED_SCHEMA)
            },
        )
    }

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        model.workspace_len(rows)
    }

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict(transformed)
        })
    }

    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.with_transformed(data, workspace, |estimator, transformed| {
            estimator.predict_into(transformed, output)
        })
    }
}

/// The ranker, registered as an ordinary member of the battery.
///
/// **Ranking sits inside the battery.** Excluding it is exactly the
/// special-casing the capability descriptor exists to abolish, and every
/// structural obligation the battery states is meaningful for a ranker:
/// metadata, caller-owned output matching the allocating form, feature width
/// and output length validated before any write, scalar and batch agreement,
/// non-finite rejection, a deterministic refit, and an artifact declaration
/// that matches behavior.
///
/// The two things it does not share are absorbed here, at the case boundary,
/// rather than by giving ranking a category of its own:
///
/// - *Error type.* Scoring returns [`PairwiseError`]; this case unwraps its
///   `Model` variant. Any other variant would be a pair-construction error
///   escaping a scoring path, which cannot happen by construction — scoring is
///   a thin wrapper over the linear model — so it is asserted rather than
///   handled.
/// - *Fit input.* Fitting takes pair observations rather than a target vector,
///   so the case builds them from the fixture's monotone values.
///
/// It produces one real-valued score per row through a caller-owned buffer,
/// which is exactly [`WorkspaceRegressorCase`]'s shape, so it needs no
/// workspace at all and no new obligation list. The pairwise surface —
/// `pair_margin`, `compare`, `compare_into`, pair-index validation — is not a
/// shared obligation and stays in the estimator-specific tests below.
struct PairwiseLinearRankerCase;

/// Unwraps the only error variant a scoring path can produce.
fn scoring_error(error: PairwiseError) -> ModelError {
    match error {
        PairwiseError::Model(error) => error,
        other => panic!("scoring produced a non-model pairwise error: {other:?}"),
    }
}

/// Consecutive pairs of the fixture, each preferring the higher-valued row.
///
/// The fixture's targets increase with the row index, so this is a total order
/// the ranker can reproduce, built from the same data every other case uses.
fn fixture_pairs(train: &Sample) -> Vec<PairwiseObservation> {
    (1..train.rows())
        .map(|row| {
            PairwiseObservation::new(
                PairIndex::new(row, row - 1).expect("distinct fixture pair"),
                PairOutcome::LeftPreferred,
                1.0,
            )
            .expect("valid fixture observation")
        })
        .collect()
}

impl WorkspaceRegressorCase for PairwiseLinearRankerCase {
    type Model = PairwiseLinearRanker;
    const NAME: &'static str = "PairwiseLinearRanker";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        PairwiseLinearRanker::fit(
            &train.view(),
            &fixture_pairs(train),
            PairwiseLinearRankerParams::default(),
        )
        .map_err(scoring_error)
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        round_trip(
            || model.to_artifact(SCHEMA),
            |bytes| PairwiseLinearRanker::from_artifact(bytes, SCHEMA),
        )
    }

    fn workspace_len(_model: &Self::Model, _rows: usize) -> Result<usize, ModelError> {
        Ok(0)
    }

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        model.score_items(data).map_err(scoring_error)
    }

    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.score_items_into(data, output).map_err(scoring_error)
    }
}

impl ScalarWorkspaceRegressorCase for PairwiseLinearRankerCase {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        model.score_one(row).map_err(scoring_error)
    }
}

// ----------------------------------------------------- registration list
//
// One line per estimator. `check_batch_only_*` is the weaker entry point and
// exists only for the runtime dispatch enums, which have no scalar path.

#[test]
fn random_forest_classifier_conforms() {
    check_classifier::<RandomForestClassifierCase>();
}

#[test]
fn extra_trees_classifier_conforms() {
    check_classifier::<ExtraTreesClassifierCase>();
}

#[test]
fn extra_trees_regressor_conforms() {
    check_regressor::<ExtraTreesRegressorCase>();
}

#[test]
fn decision_tree_classifier_conforms() {
    check_classifier::<DecisionTreeClassifierCase>();
}

#[test]
fn decision_tree_regressor_conforms() {
    check_regressor::<DecisionTreeRegressorCase>();
}

#[test]
fn logistic_regression_conforms() {
    check_classifier::<LogisticRegressionCase>();
}

#[test]
fn hist_gradient_boosting_classifier_conforms() {
    check_classifier::<HistGradientBoostingClassifierCase>();
}

#[test]
fn any_classifier_conforms_for_every_variant() {
    check_batch_only_classifier::<AnyForestClassifierCase>();
    check_batch_only_classifier::<AnyLogisticClassifierCase>();
    check_batch_only_classifier::<AnyBoostedClassifierCase>();
}

#[test]
fn calibrated_classifiers_conform_for_both_calibrators() {
    check_batch_only_classifier::<IsotonicCalibratedForestCase>();
    check_batch_only_classifier::<PlattCalibratedForestCase>();
}

#[test]
fn random_forest_regressor_conforms() {
    check_regressor::<RandomForestRegressorCase>();
}

#[test]
fn linear_regression_conforms() {
    check_regressor::<LinearRegressionCase>();
}

#[test]
fn ridge_conforms() {
    check_regressor::<RidgeCase>();
}

#[test]
fn lasso_conforms() {
    check_regressor::<LassoCase>();
}

#[test]
fn elastic_net_conforms() {
    check_regressor::<ElasticNetCase>();
}

#[test]
fn hist_gradient_boosting_regressor_conforms() {
    check_regressor::<HistGradientBoostingRegressorCase>();
}

#[test]
fn any_regressor_conforms_for_every_variant() {
    check_batch_only_regressor::<AnyForestRegressorCase>();
    check_batch_only_regressor::<AnyLinearRegressorCase>();
    check_batch_only_regressor::<AnyRidgeRegressorCase>();
    check_batch_only_regressor::<AnyBoostedRegressorCase>();
}

#[test]
fn dummy_classifier_conforms() {
    check_classifier::<DummyClassifierCase>();
}

#[test]
fn dummy_regressor_conforms() {
    check_regressor::<DummyRegressorCase>();
}

#[test]
fn standard_scaler_conforms() {
    check_transformer::<StandardScalerCase>();
}

#[test]
fn min_max_scaler_conforms() {
    check_transformer::<MinMaxScalerCase>();
}

#[test]
fn max_abs_scaler_conforms() {
    check_transformer::<MaxAbsScalerCase>();
}

#[test]
fn robust_scaler_conforms() {
    check_transformer::<RobustScalerCase>();
}

#[test]
fn stateless_transformers_conform() {
    check_transformer::<NormalizerCase>();
    check_transformer::<BinarizerCase>();
    check_transformer::<FunctionTransformerCase>();
}

#[test]
fn isotonic_regression_conforms() {
    check_regressor::<IsotonicRegressionCase>();
}

#[test]
fn fitted_pipelines_conform() {
    check_workspace_classifier::<ScaledLogisticPipelineCase>();
    check_workspace_regressor::<ScaledRidgePipelineCase>();
    check_workspace_regressor::<ScaledLinearPipelineCase>();
}

#[test]
fn pairwise_linear_ranker_conforms() {
    check_scalar_workspace_regressor::<PairwiseLinearRankerCase>();
}

#[test]
fn staged_pipelines_conform_at_both_arities() {
    check_workspace_regressor::<TwoStagePipelineCase>();
    check_workspace_regressor::<ThreeStagePipelineCase>();
    check_workspace_regressor::<RobustStagedPipelineCase>();
}

/// A composition holding a stateless stage transforms and predicts, but cannot
/// be persisted — and cannot be registered in the battery either.
///
/// The reason is worth recording precisely, because it is stronger than "no
/// artifact". `HasCapabilities for StagedPipeline` is itself bounded on the
/// stage-persistence trait, so a stack containing a stage that declares no
/// artifact cannot declare *anything*, and the battery requires a declaring
/// model. The composition is perfectly usable; it simply has no capability
/// vocabulary to be checked against, so its obligations are asserted here
/// directly rather than generically.
#[test]
fn a_stateless_stage_composes_and_predicts_without_being_persistable() {
    let fixture = Fixture::default();
    let train = &fixture.train;
    let first = Normalizer::fit(&train.view(), NormalizerParams::default()).unwrap();
    let after_first = first.transform(&train.view()).unwrap();
    let second = Binarizer::fit(
        &after_first.as_view(),
        BinarizerParams::default().with_threshold(0.5),
    )
    .unwrap();
    let after_second = second.transform(&after_first.as_view()).unwrap();
    let third = FunctionTransformer::fit(
        &after_second.as_view(),
        FunctionTransformerParams::default().with_func(double),
    )
    .unwrap();
    let after_third = third.transform(&after_second.as_view()).unwrap();
    let pipeline: StagedPipeline<(Normalizer, Binarizer, FunctionTransformer), Ridge> =
        StagedPipeline::new(
            (first, second, third),
            Ridge::fit(
                &after_third.as_view(),
                &train.values,
                RidgeParams::default(),
            )
            .unwrap(),
        )
        .expect("a stateless composition composes");

    let view = train.view();
    let workspace_len = pipeline.workspace_len(view.rows()).unwrap();
    let mut workspace = vec![0.0; workspace_len];
    let predictions = pipeline
        .with_transformed(&view, &mut workspace, |estimator, transformed| {
            estimator.predict(transformed)
        })
        .unwrap();
    assert_eq!(predictions.len(), view.rows());
    assert!(predictions.iter().all(|value| value.is_finite()));

    // The two composition hazards the battery would otherwise have covered: a
    // workspace whose length is never checked, and one whose contents leak from
    // one batch into the next.
    let mut short = vec![0.0; workspace_len - 1];
    assert_eq!(
        pipeline
            .with_transformed(&view, &mut short, |estimator, transformed| {
                estimator.predict(transformed)
            })
            .unwrap_err(),
        ModelError::OutputLength {
            expected: workspace_len,
            actual: workspace_len - 1
        }
    );

    let holdout = fixture.holdout.view();
    let mut fresh = vec![0.0; workspace_len];
    let alone = pipeline
        .with_transformed(&holdout, &mut fresh, |estimator, transformed| {
            estimator.predict(transformed)
        })
        .unwrap();
    let mut reused = vec![0.0; workspace_len];
    let _ = pipeline.with_transformed(&view, &mut reused, |estimator, transformed| {
        estimator.predict(transformed)
    });
    let after = pipeline
        .with_transformed(&holdout, &mut reused, |estimator, transformed| {
            estimator.predict(transformed)
        })
        .unwrap();
    assert_eq!(
        alone, after,
        "a reused workspace must carry nothing forward"
    );
}

// Deliberately unregistered, with the reason stated rather than left as a gap:
//
// - `PlattCalibrator` is a `Calibrator`, not an `Estimator`: a fitted map of
//   one score, with no fitted input width, no batch surface, and no capability
//   declaration to check against behavior. Its obligations are in
//   `tests/calibration.rs`.
// - The standalone trees and the randomized ensembles are **not** `AnyClassifier` /
//   `AnyRegressor`
//   variants. Adding a variant is a public-enum change to a shared file with no
//   consumer in this sprint, and the nested-payload dispatch design means it can
//   be added later without touching any existing estimator's bytes. The
//   estimators themselves are registered above, so the omission is a dispatch
//   gap rather than a contract gap.
// - `TransformerStack` tuples are not `Estimator`s either. A stack is reached
//   only through `StagedPipeline`, which is registered above at both arities,
//   so its handoff validation and its disjoint workspace segments are covered
//   by a registration rather than by a bespoke test.

// -------------------------------------------------- estimator-specific tests

fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
    DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
}

fn classifier(
    data: &DenseMatrix,
    labels: Vec<u8>,
    min_samples_split: usize,
) -> RandomForestClassifier {
    RandomForestClassifier::fit(
        &data.as_view(),
        &BinaryTargets::new(labels).unwrap(),
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All)
            .with_min_samples_split(min_samples_split),
    )
    .unwrap()
}

#[test]
fn exact_probability_tie_selects_the_first_smaller_class() {
    // The battery proves labels follow the probability argmax; only a fixture
    // built to tie exactly can prove which side of a tie wins.
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let model = classifier(&data, vec![0, 1, 0, 1], 5);

    assert_eq!(model.predict_proba_one(&[10.0]).unwrap(), vec![0.5, 0.5]);
    assert_eq!(model.predict_one(&[10.0]).unwrap(), 0);
    assert_eq!(model.predict(&data.as_view()).unwrap(), vec![0; 4]);
}

#[test]
fn single_class_models_use_one_probability_column() {
    // Only the forest can be fitted on a single observed class, so this shape
    // cannot be exercised from the shared fixture.
    let data = matrix(&[0.0, 1.0, 2.0], 3, 1);
    for (label, absent) in [(0, 1), (1, 0)] {
        let model = classifier(&data, vec![label; 3], 2);

        assert_eq!(model.classes(), &[label]);
        assert_eq!(model.predict(&data.as_view()).unwrap(), vec![label; 3]);
        assert_eq!(model.predict_proba(&data.as_view()).unwrap(), vec![1.0; 3]);
        assert_eq!(model.predict_proba_one(&[1.5]).unwrap(), vec![1.0]);
        assert_eq!(
            model.predict_class_proba(&data.as_view(), label).unwrap(),
            vec![1.0; 3]
        );
        assert_eq!(
            model
                .predict_class_proba(&data.as_view(), absent)
                .unwrap_err(),
            ModelError::UnknownClass { class: absent }
        );
        assert_eq!(
            model.predict_positive_proba(&[1.5]).unwrap(),
            f32::from(label)
        );
    }
}

#[test]
fn scalar_probability_output_is_validated_before_writing() {
    // `predict_proba_one_into` is a scalar probability path the shared
    // `Classifier` contract does not have, so the battery cannot reach it.
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let model = classifier(&data, vec![0, 0, 1, 1], 2);

    let mut one_row = [9.0_f32; 1];
    assert_eq!(
        model
            .predict_proba_one_into(&[1.0], &mut one_row)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(one_row, [9.0]);
}

#[test]
fn parallel_and_serial_forests_fit_identically() {
    // Thread count is a forest parameter, so this determinism guarantee is
    // narrower than the battery's same-parameters refit check.
    let data = matrix(
        &[
            0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 7.0, 5.0, 6.0, 6.0, 5.0, 7.0, 4.0,
        ],
        8,
        2,
    );
    let targets = BinaryTargets::new(vec![0, 1, 1, 0, 1, 0, 0, 1]).unwrap();
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(31)
        .with_max_depth(Some(8))
        .with_max_features(MaxFeatures::All)
        .with_random_state(123);
    let serial = RandomForestClassifier::fit(&data.as_view(), &targets, params.clone()).unwrap();
    let parallel = RandomForestClassifier::fit(
        &data.as_view(),
        &targets,
        params.with_n_jobs(NJobs::Count(4)),
    )
    .unwrap();

    assert_eq!(
        serial.predict(&data.as_view()).unwrap(),
        parallel.predict(&data.as_view()).unwrap()
    );
    assert_eq!(
        serial.predict_proba(&data.as_view()).unwrap(),
        parallel.predict_proba(&data.as_view()).unwrap()
    );
}

#[test]
fn runtime_dispatch_preserves_predictions_and_parameter_identity() {
    // The battery proves each dispatch variant satisfies the contract on its
    // own; this proves dispatching does not change the answer or lose the
    // concrete parameter type.
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();

    let concrete = classifier(&data, vec![0, 0, 1, 1], 2);
    let expected_labels = concrete.predict(&data.as_view()).unwrap();
    let expected_probabilities = concrete.predict_proba(&data.as_view()).unwrap();
    let dispatched: AnyClassifier = concrete.into();
    let erased: &dyn Classifier = &dispatched;
    assert_eq!(erased.predict(&data.as_view()).unwrap(), expected_labels);
    // Probabilities are reached through the fallible accessor rather than
    // through the dispatch enum's own trait impl; see the shape test below.
    assert_eq!(
        dispatched
            .as_probabilistic()
            .expect("every shipped variant produces probabilities")
            .predict_proba(&data.as_view())
            .unwrap(),
        expected_probabilities
    );
    assert!(matches!(
        dispatched.get_params(),
        AnyClassifierParams::RandomForest(_)
    ));

    let logistic = LogisticRegression::fit(
        &data.as_view(),
        &binary,
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let expected = logistic.predict_proba(&data.as_view()).unwrap();
    let dispatched: AnyClassifier = logistic.into();
    assert_eq!(
        dispatched
            .as_probabilistic()
            .expect("every shipped variant produces probabilities")
            .predict_proba(&data.as_view())
            .unwrap(),
        expected
    );
    assert!(matches!(
        dispatched.get_params(),
        AnyClassifierParams::LogisticRegression(_)
    ));
    assert_eq!(dispatched.n_features_in(), data.columns());

    let forest = RandomForestRegressor::fit(
        &data.as_view(),
        &targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false),
    )
    .unwrap();
    let expected = forest.predict(&data.as_view()).unwrap();
    let dispatched: AnyRegressor = forest.into();
    let erased: &dyn Regressor = &dispatched;
    assert_eq!(erased.predict(&data.as_view()).unwrap(), expected);
    assert!(matches!(
        dispatched.get_params(),
        AnyRegressorParams::RandomForest(_)
    ));

    let linear: AnyRegressor =
        LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
            .unwrap()
            .into();
    assert!(matches!(
        linear.get_params(),
        AnyRegressorParams::LinearRegression(_)
    ));

    let ridge: AnyRegressor = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())
        .unwrap()
        .into();
    assert!(matches!(ridge.get_params(), AnyRegressorParams::Ridge(_)));

    let boosted: AnyRegressor =
        HistGradientBoostingRegressor::fit(&data.as_view(), &targets, boosting_params())
            .unwrap()
            .into();
    assert!(matches!(
        boosted.get_params(),
        AnyRegressorParams::HistGradientBoosting(_)
    ));

    for dispatched in [linear, ridge, boosted] {
        assert!(
            dispatched
                .predict(&data.as_view())
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn pairwise_scores_are_raw_antisymmetric_and_batch_validation_is_atomic() {
    let items = matrix(&[0.0, 0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 2.0], 4, 2);
    let observations = [
        PairwiseObservation::new(
            PairIndex::new(3, 2).unwrap(),
            PairOutcome::LeftPreferred,
            1.0,
        )
        .unwrap(),
        PairwiseObservation::new(
            PairIndex::new(2, 1).unwrap(),
            PairOutcome::LeftPreferred,
            1.0,
        )
        .unwrap(),
        PairwiseObservation::new(
            PairIndex::new(1, 0).unwrap(),
            PairOutcome::LeftPreferred,
            1.0,
        )
        .unwrap(),
    ];
    let model = PairwiseLinearRanker::fit(
        &items.as_view(),
        &observations,
        PairwiseLinearRankerParams::default().with_c(4.0),
    )
    .unwrap();
    let forward = model
        .pair_margin(&items.as_view(), PairIndex::new(3, 0).unwrap())
        .unwrap();
    let reverse = model
        .pair_margin(&items.as_view(), PairIndex::new(0, 3).unwrap())
        .unwrap();
    assert_eq!(forward.to_bits(), (-reverse).to_bits());
    assert!(forward > 1.0);

    let pairs = [PairIndex::new(3, 0).unwrap(), PairIndex::new(0, 7).unwrap()];
    let mut output = [99.0; 2];
    assert_eq!(
        model.pair_margins_into(&items.as_view(), &pairs, &mut output),
        Err(PairwiseError::PairIndexOutOfBounds {
            pair: 1,
            item: 7,
            items: 4,
        })
    );
    assert_eq!(output, [99.0; 2]);

    assert_eq!(
        model.score_one(&[f32::NAN, 0.0]),
        Err(PairwiseError::Model(ModelError::NonFiniteFeature {
            row: 0,
            column: 0,
        }))
    );
}

#[test]
fn finite_inputs_that_overflow_are_reported_rather_than_returned() {
    // The battery proves non-finite *inputs* are rejected. Provoking a
    // non-finite *result* from finite inputs is estimator-specific: the linear
    // models overflow on an extreme feature, the forest on extreme leaves.
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let regression = RegressionTargets::new(vec![0.0, 2.0, 4.0, 6.0]).unwrap();
    let linear = LinearRegression::fit(
        &data.as_view(),
        &regression,
        LinearRegressionParams::default().with_fit_intercept(false),
    )
    .unwrap();
    let ridge = Ridge::fit(
        &data.as_view(),
        &regression,
        RidgeParams::default()
            .with_alpha(0.0)
            .with_fit_intercept(false),
    )
    .unwrap();
    assert_eq!(
        linear.predict_one(&[f32::MAX]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    assert_eq!(
        ridge.predict_one(&[f32::MAX]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    let logistic = LogisticRegression::fit(
        &data.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
        LogisticRegressionParams::default().with_c(100.0),
    )
    .unwrap();
    assert_eq!(
        logistic.decision_function_one(&[f32::NEG_INFINITY]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        logistic.predict_positive_proba(&[f32::NAN]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        logistic.decision_function_one(&[f32::MAX]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    // Averaging extreme leaf values overflows the f32 accumulator, which is
    // the forest's route to a non-finite prediction from finite inputs. Every
    // entry point must surface that as an error rather than an infinity.
    let extreme = RegressionTargets::new(vec![f32::MAX; 4]).unwrap();
    let forest = RandomForestRegressor::fit(
        &data.as_view(),
        &extreme,
        RandomForestRegressorParams::default()
            .with_n_estimators(4)
            .with_bootstrap(false),
    )
    .unwrap();
    assert_eq!(
        forest.predict_one(&[1.0]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    assert_eq!(
        forest.predict(&data.as_view()),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    let mut output = [7.0; 4];
    assert_eq!(
        forest.predict_into(&data.as_view(), &mut output),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    assert_eq!(
        Regressor::predict(&forest, &data.as_view()),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    let erased: AnyRegressor = forest.into();
    assert_eq!(
        erased.predict(&data.as_view()),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    // A finitely-predicting forest keeps returning values, so the check
    // rejects only the overflowing case.
    let ordinary = RandomForestRegressor::fit(
        &data.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap(),
        RandomForestRegressorParams::default()
            .with_n_estimators(4)
            .with_bootstrap(false),
    )
    .unwrap();
    assert!(
        ordinary
            .predict(&data.as_view())
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}
