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
    Regressor,
};
use ferricml::artifact::ArtifactError;
use ferricml::calibration::{
    CalibratedClassifier, IsotonicRegression, PlattCalibrator, PlattParams,
};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::dummy::{
    DummyClassifier, DummyClassifierParams, DummyRegressor, DummyRegressorParams,
};
use ferricml::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures, NJobs,
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LinearRegression, LinearRegressionParams,
    LogisticRegression, LogisticRegressionParams, Ridge, RidgeParams,
};
use ferricml::preprocessing::{
    MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams, StandardScaler,
    StandardScalerParams,
};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseError, PairwiseLinearRanker, PairwiseLinearRankerParams,
    PairwiseObservation,
};

use support::conformance::{
    ClassifierCase, OptionalFit, RegressorCase, RoundTrip, SCHEMA, Sample, ScalarClassifierCase,
    ScalarRegressorCase, TransformerCase, check_batch_only_classifier, check_batch_only_regressor,
    check_classifier, check_regressor, check_transformer,
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

struct LogisticRegressionCase;

impl ClassifierCase for LogisticRegressionCase {
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
/// The battery's one fixture is both the training and the calibration sample
/// here, which is deliberately *not* how a calibrated model should be built —
/// but these obligations are structural, and the fixture is the only data the
/// battery has. Held-out calibration is proven in `tests/calibration.rs`, where
/// there is room for two folds.
struct IsotonicCalibratedForestCase;

impl ClassifierCase for IsotonicCalibratedForestCase {
    type Model = CalibratedClassifier<RandomForestClassifier, IsotonicRegression>;
    const NAME: &'static str = "CalibratedClassifier<RandomForestClassifier, IsotonicRegression>";

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        CalibratedClassifier::fit_isotonic(
            RandomForestClassifierCase::fit(train, holdout)?,
            &train.view(),
            &train.labels,
        )
    }
}

struct PlattCalibratedForestCase;

impl ClassifierCase for PlattCalibratedForestCase {
    type Model = CalibratedClassifier<RandomForestClassifier, PlattCalibrator>;
    const NAME: &'static str = "CalibratedClassifier<RandomForestClassifier, PlattCalibrator>";

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        CalibratedClassifier::fit_platt(
            RandomForestClassifierCase::fit(train, holdout)?,
            &train.view(),
            &train.labels,
            PlattParams::default(),
        )
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

// ----------------------------------------------------- registration list
//
// One line per estimator. `check_batch_only_*` is the weaker entry point and
// exists only for the runtime dispatch enums, which have no scalar path.

#[test]
fn random_forest_classifier_conforms() {
    check_classifier::<RandomForestClassifierCase>();
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
    assert_eq!(
        erased.predict_proba(&data.as_view()).unwrap(),
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
    assert_eq!(dispatched.predict_proba(&data.as_view()).unwrap(), expected);
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
