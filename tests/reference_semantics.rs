//! FerricML conformance against frozen reference outputs.

use ferricml::api::ModelError;
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures, NJobs,
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::metrics::{
    MetricError, accuracy_score, binary_confusion_matrix, brier_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error,
};
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};

#[allow(dead_code, clippy::excessive_precision)]
mod reference {
    include!("fixtures/reference_semantics_v1.rs");
}

const EXACT_TOLERANCE: f32 = 1.0e-6;
const LOGISTIC_TOLERANCE: f32 = 2.0e-5;
const QUALITY_SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
const HGB_QUALITY_SEEDS: [u64; 3] = [11, 22, 33];

fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
    DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
}

fn exact_classifier_params() -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(1)
        .with_max_depth(Some(2))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::All)
        .with_bootstrap(false)
        .with_random_state(0)
        .with_n_jobs(NJobs::Serial)
}

fn exact_regressor_params() -> RandomForestRegressorParams {
    RandomForestRegressorParams::default()
        .with_n_estimators(1)
        .with_max_depth(Some(2))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::All)
        .with_bootstrap(false)
        .with_random_state(0)
        .with_n_jobs(NJobs::Serial)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_close_with_tolerance(actual, expected, EXACT_TOLERANCE);
}

fn assert_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn evaluation_metric_values_and_validation_order_are_frozen() {
    let expected = [0, 0, 1, 1];
    let predicted = [0, 1, 1, 1];
    let probabilities = [0.1, 0.8, 0.7, 0.9];
    let confusion = binary_confusion_matrix(&expected, &predicted).unwrap();
    assert_eq!(
        [
            confusion.true_negatives(),
            confusion.false_positives(),
            confusion.false_negatives(),
            confusion.true_positives(),
        ],
        [1, 1, 0, 2]
    );
    assert_eq!(accuracy_score(&expected, &predicted), Ok(0.75));
    assert_eq!(precision_score(&expected, &predicted), Ok(2.0 / 3.0));
    assert_eq!(recall_score(&expected, &predicted), Ok(1.0));
    assert_eq!(f1_score(&expected, &predicted), Ok(0.8));
    assert_eq!(
        brier_score(&expected, &probabilities),
        Ok(0.187_500_007_823_109_83)
    );
    assert_eq!(
        log_loss(&expected, &probabilities),
        Ok(0.544_208_498_117_417_1)
    );
    assert_eq!(roc_auc_score(&expected, &probabilities), Ok(0.75));

    let regression_expected = [1.0, 2.0, 3.0, 4.0];
    let regression_predicted = [1.0, 3.0, 2.0, 5.0];
    assert_eq!(
        mean_absolute_error(&regression_expected, &regression_predicted),
        Ok(0.75)
    );
    assert_eq!(
        mean_squared_error(&regression_expected, &regression_predicted),
        Ok(0.75)
    );
    assert_eq!(
        root_mean_squared_error(&regression_expected, &regression_predicted),
        Ok(0.75_f64.sqrt())
    );
    assert_eq!(
        r2_score(&regression_expected, &regression_predicted),
        Ok(0.4)
    );

    assert_eq!(
        brier_score(&[2], &[]),
        Err(MetricError::LengthMismatch {
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        roc_auc_score(&[1, 1], &[0.1, 0.9]),
        Err(MetricError::Undefined)
    );
    assert_eq!(
        r2_score(&[2.0, 2.0], &[2.0, 2.0]),
        Err(MetricError::Undefined)
    );
}

#[test]
fn exact_classifier_matches_frozen_reference_outputs() {
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let targets = BinaryTargets::new(reference::EXACT_CLASSIFIER_Y.to_vec()).unwrap();
    let model =
        RandomForestClassifier::fit(&train.as_view(), &targets, exact_classifier_params()).unwrap();

    assert_eq!(model.n_features_in(), 2);
    assert_eq!(model.classes(), reference::EXACT_CLASSES);
    assert_eq!(
        model.predict(&test.as_view()).unwrap(),
        reference::EXACT_LABELS
    );
    let probabilities = model.predict_proba(&test.as_view()).unwrap();
    assert_eq!(probabilities.len(), 5 * model.classes().len());
    assert_close(&probabilities, reference::EXACT_PROBABILITIES);
}

#[test]
fn exact_regressor_matches_frozen_reference_outputs() {
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let targets = RegressionTargets::new(reference::EXACT_REGRESSION_Y.to_vec()).unwrap();
    let model =
        RandomForestRegressor::fit(&train.as_view(), &targets, exact_regressor_params()).unwrap();

    assert_eq!(model.n_features_in(), 2);
    let predictions = model.predict(&test.as_view()).unwrap();
    assert_eq!(predictions.len(), 5);
    assert_close(&predictions, reference::EXACT_REGRESSION);
}

#[test]
fn tie_and_single_class_shapes_match_frozen_reference_outputs() {
    let train = matrix(reference::TIE_TRAIN_X, 4, 1);
    let test = matrix(reference::TIE_TEST_X, 3, 1);
    let tie = RandomForestClassifier::fit(
        &train.as_view(),
        &BinaryTargets::new(reference::TIE_Y.to_vec()).unwrap(),
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_min_samples_split(5)
            .with_max_features(MaxFeatures::All)
            .with_bootstrap(false),
    )
    .unwrap();
    assert_eq!(tie.classes(), reference::TIE_CLASSES);
    assert_eq!(tie.predict(&test.as_view()).unwrap(), reference::TIE_LABELS);
    assert_close(
        &tie.predict_proba(&test.as_view()).unwrap(),
        reference::TIE_PROBABILITIES,
    );

    let single = RandomForestClassifier::fit(
        &train.as_view(),
        &BinaryTargets::new(vec![1; 4]).unwrap(),
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_max_features(MaxFeatures::All)
            .with_bootstrap(false),
    )
    .unwrap();
    assert_eq!(single.classes(), reference::SINGLE_CLASSES);
    assert_eq!(
        single.predict(&test.as_view()).unwrap(),
        reference::SINGLE_LABELS
    );
    let probabilities = single.predict_proba(&test.as_view()).unwrap();
    assert_eq!(probabilities.len(), test.rows());
    assert_close(&probabilities, reference::SINGLE_PROBABILITIES);
}

#[test]
fn logistic_without_intercept_matches_frozen_reference_outputs() {
    let train = matrix(reference::LOGISTIC_NO_INTERCEPT_TRAIN_X, 6, 2);
    let test = matrix(reference::LOGISTIC_NO_INTERCEPT_TEST_X, 5, 2);
    let targets = BinaryTargets::new(reference::LOGISTIC_NO_INTERCEPT_Y.to_vec()).unwrap();
    let model = LogisticRegression::fit(
        &train.as_view(),
        &targets,
        LogisticRegressionParams::default()
            .with_fit_intercept(false)
            .with_tol(1.0e-8),
    )
    .unwrap();

    assert_eq!(model.intercept().to_bits(), 0.0_f32.to_bits());
    assert_close(
        model.coefficients(),
        reference::LOGISTIC_NO_INTERCEPT_COEFFICIENTS,
    );
    assert_close(
        &model.decision_function(&test.as_view()).unwrap(),
        reference::LOGISTIC_NO_INTERCEPT_DECISIONS,
    );
    assert_close(
        &model.predict_proba(&test.as_view()).unwrap(),
        reference::LOGISTIC_NO_INTERCEPT_PROBABILITIES,
    );
}

#[test]
fn weighted_logistic_matches_frozen_reference_outputs() {
    let train = matrix(reference::LOGISTIC_NO_INTERCEPT_TRAIN_X, 6, 2);
    let test = matrix(reference::LOGISTIC_NO_INTERCEPT_TEST_X, 5, 2);
    let targets = BinaryTargets::new(reference::LOGISTIC_NO_INTERCEPT_Y.to_vec()).unwrap();
    let weights = SampleWeights::new(reference::LOGISTIC_WEIGHTS.to_vec()).unwrap();
    let model = LogisticRegression::fit_weighted(
        &train.as_view(),
        &targets,
        &weights,
        LogisticRegressionParams::default()
            .with_c(0.75)
            .with_tol(1.0e-8),
    )
    .unwrap();

    assert_close_with_tolerance(
        model.coefficients(),
        reference::LOGISTIC_WEIGHTED_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[model.intercept()],
        reference::LOGISTIC_WEIGHTED_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &model.decision_function(&test.as_view()).unwrap(),
        reference::LOGISTIC_WEIGHTED_DECISIONS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &model.predict_proba(&test.as_view()).unwrap(),
        reference::LOGISTIC_WEIGHTED_PROBABILITIES,
        LOGISTIC_TOLERANCE,
    );
}

#[test]
fn linear_regression_matches_frozen_reference_outputs() {
    let full_x = matrix(reference::LINEAR_FULL_X, 4, 2);
    let full_y = RegressionTargets::new(reference::LINEAR_FULL_Y.to_vec()).unwrap();
    let test_x = matrix(reference::LINEAR_TEST_X, 3, 2);
    let full = LinearRegression::fit(
        &full_x.as_view(),
        &full_y,
        LinearRegressionParams::default(),
    )
    .unwrap();
    assert_eq!(full.rank(), 2);
    assert_close_with_tolerance(
        full.coefficients(),
        reference::LINEAR_FULL_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[full.intercept()],
        reference::LINEAR_FULL_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &full.predict(&test_x.as_view()).unwrap(),
        reference::LINEAR_FULL_PREDICTIONS,
        LOGISTIC_TOLERANCE,
    );

    let rank_x = matrix(reference::LINEAR_RANK_DEFICIENT_X, 3, 2);
    let rank_y = RegressionTargets::new(reference::LINEAR_RANK_DEFICIENT_Y.to_vec()).unwrap();
    let rank_deficient = LinearRegression::fit(
        &rank_x.as_view(),
        &rank_y,
        LinearRegressionParams::default()
            .with_fit_intercept(false)
            .with_tol(0.0),
    )
    .unwrap();
    assert_eq!(rank_deficient.rank(), 1);
    assert_close_with_tolerance(
        rank_deficient.coefficients(),
        reference::LINEAR_RANK_DEFICIENT_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );

    let weighted_x = matrix(reference::LINEAR_WEIGHTED_X, 4, 1);
    let weighted_y = RegressionTargets::new(reference::LINEAR_WEIGHTED_Y.to_vec()).unwrap();
    let weights = SampleWeights::new(reference::LINEAR_WEIGHTS.to_vec()).unwrap();
    let weighted = LinearRegression::fit_weighted(
        &weighted_x.as_view(),
        &weighted_y,
        &weights,
        LinearRegressionParams::default(),
    )
    .unwrap();
    assert_close_with_tolerance(
        weighted.coefficients(),
        reference::LINEAR_WEIGHTED_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[weighted.intercept()],
        reference::LINEAR_WEIGHTED_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
}

#[test]
fn ridge_matches_frozen_reference_outputs() {
    let full_x = matrix(reference::LINEAR_FULL_X, 4, 2);
    let full_y = RegressionTargets::new(reference::LINEAR_FULL_Y.to_vec()).unwrap();
    let test_x = matrix(reference::LINEAR_TEST_X, 3, 2);
    let full = Ridge::fit(&full_x.as_view(), &full_y, RidgeParams::default()).unwrap();
    assert_close_with_tolerance(
        full.coefficients(),
        reference::RIDGE_FULL_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[full.intercept()],
        reference::RIDGE_FULL_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &full.predict(&test_x.as_view()).unwrap(),
        reference::RIDGE_FULL_PREDICTIONS,
        LOGISTIC_TOLERANCE,
    );

    let rank_x = matrix(reference::LINEAR_RANK_DEFICIENT_X, 3, 2);
    let rank_y = RegressionTargets::new(reference::LINEAR_RANK_DEFICIENT_Y.to_vec()).unwrap();
    let alpha_zero = Ridge::fit(
        &rank_x.as_view(),
        &rank_y,
        RidgeParams::default()
            .with_alpha(0.0)
            .with_fit_intercept(false),
    )
    .unwrap();
    assert_close_with_tolerance(
        alpha_zero.coefficients(),
        reference::RIDGE_ALPHA_ZERO_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );

    let weighted_x = matrix(reference::LINEAR_WEIGHTED_X, 4, 1);
    let weighted_y = RegressionTargets::new(reference::LINEAR_WEIGHTED_Y.to_vec()).unwrap();
    let weights = SampleWeights::new(reference::LINEAR_WEIGHTS.to_vec()).unwrap();
    let weighted = Ridge::fit_weighted(
        &weighted_x.as_view(),
        &weighted_y,
        &weights,
        RidgeParams::default(),
    )
    .unwrap();
    assert_close_with_tolerance(
        weighted.coefficients(),
        reference::RIDGE_WEIGHTED_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[weighted.intercept()],
        reference::RIDGE_WEIGHTED_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
}

#[test]
fn standard_scaler_matches_frozen_reference_outputs() {
    let data = matrix(reference::SCALER_TRAIN_X, 4, 3);
    let default = StandardScaler::fit(&data.as_view(), StandardScalerParams::default()).unwrap();
    assert_close_f64(default.means(), reference::SCALER_DEFAULT_MEAN);
    assert_close_f64(default.variances(), reference::SCALER_DEFAULT_VARIANCE);
    assert_close_f64(default.scales(), reference::SCALER_DEFAULT_SCALE);
    assert_close(
        default.transform(&data.as_view()).unwrap().as_slice(),
        reference::SCALER_DEFAULT_TRANSFORMED,
    );

    let no_mean = StandardScaler::fit(
        &data.as_view(),
        StandardScalerParams::default().with_mean(false),
    )
    .unwrap();
    assert_close(
        no_mean.transform(&data.as_view()).unwrap().as_slice(),
        reference::SCALER_NO_MEAN_TRANSFORMED,
    );

    let no_std = StandardScaler::fit(
        &data.as_view(),
        StandardScalerParams::default().with_std(false),
    )
    .unwrap();
    assert_close(
        no_std.transform(&data.as_view()).unwrap().as_slice(),
        reference::SCALER_NO_STD_TRANSFORMED,
    );

    let weights = SampleWeights::new(reference::SCALER_WEIGHTS.to_vec()).unwrap();
    let weighted =
        StandardScaler::fit_weighted(&data.as_view(), &weights, StandardScalerParams::default())
            .unwrap();
    assert_close_f64(weighted.means(), reference::SCALER_WEIGHTED_MEAN);
    assert_close_f64(weighted.variances(), reference::SCALER_WEIGHTED_VARIANCE);
    assert_close_f64(weighted.scales(), reference::SCALER_WEIGHTED_SCALE);
    assert_close(
        weighted.transform(&data.as_view()).unwrap().as_slice(),
        reference::SCALER_WEIGHTED_TRANSFORMED,
    );
}

#[test]
fn histogram_boosting_matches_frozen_reference_one_step_outputs() {
    let train = matrix(reference::HGB_TRAIN_X, 8, 1);
    let targets = RegressionTargets::new(reference::HGB_TRAIN_Y.to_vec()).unwrap();
    let test = matrix(reference::HGB_TEST_X, 4, 1);
    let model = HistGradientBoostingRegressor::fit(
        &train.as_view(),
        &targets,
        HistGradientBoostingRegressorParams::default()
            .with_learning_rate(1.0)
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1),
    )
    .unwrap();
    assert_eq!(model.n_iter(), 1);
    assert_close(
        &model.predict(&test.as_view()).unwrap(),
        reference::HGB_PREDICTIONS,
    );
}

#[test]
fn supported_defaults_names_and_validation_are_locked() {
    let classifier = RandomForestClassifierParams::default();
    assert_eq!(classifier.n_estimators(), 100);
    assert_eq!(classifier.max_depth(), None);
    assert_eq!(classifier.min_samples_split(), 2);
    assert_eq!(classifier.min_samples_leaf(), 1);
    assert_eq!(classifier.max_features(), MaxFeatures::Sqrt);
    assert!(classifier.bootstrap());
    assert_eq!(classifier.random_state(), 0);
    assert_eq!(classifier.n_jobs(), NJobs::Serial);

    let regressor = RandomForestRegressorParams::default();
    assert_eq!(regressor.n_estimators(), 100);
    assert_eq!(regressor.max_depth(), None);
    assert_eq!(regressor.min_samples_split(), 2);
    assert_eq!(regressor.min_samples_leaf(), 1);
    assert_eq!(regressor.max_features(), MaxFeatures::All);
    assert!(regressor.bootstrap());
    assert_eq!(regressor.random_state(), 0);
    assert_eq!(regressor.n_jobs(), NJobs::Serial);

    let boosting = HistGradientBoostingRegressorParams::default();
    assert_eq!(boosting.learning_rate(), 0.1);
    assert_eq!(boosting.max_iter(), 100);
    assert_eq!(boosting.max_leaf_nodes(), 31);
    assert_eq!(boosting.max_depth(), None);
    assert_eq!(boosting.min_samples_leaf(), 20);
    assert_eq!(boosting.l2_regularization(), 0.0);
    assert_eq!(boosting.max_bins(), 255);

    let train = matrix(&[0.0, 1.0], 2, 1);
    let targets = BinaryTargets::new(vec![0, 1]).unwrap();
    let invalid = [
        (
            RandomForestClassifierParams::default().with_n_estimators(0),
            ModelError::InvalidEstimatorCount,
        ),
        (
            RandomForestClassifierParams::default().with_max_depth(Some(0)),
            ModelError::InvalidMaxDepth,
        ),
        (
            RandomForestClassifierParams::default().with_min_samples_split(1),
            ModelError::InvalidMinSamplesSplit,
        ),
        (
            RandomForestClassifierParams::default().with_min_samples_leaf(0),
            ModelError::InvalidMinSamplesLeaf,
        ),
        (
            RandomForestClassifierParams::default().with_max_features(MaxFeatures::Count(0)),
            ModelError::InvalidMaxFeatures {
                requested: 0,
                available: 1,
            },
        ),
        (
            RandomForestClassifierParams::default().with_n_jobs(NJobs::Count(0)),
            ModelError::InvalidJobCount,
        ),
    ];
    for (params, expected) in invalid {
        assert_eq!(
            RandomForestClassifier::fit(&train.as_view(), &targets, params).unwrap_err(),
            expected
        );
    }

    let model = RandomForestClassifier::fit(
        &train.as_view(),
        &targets,
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false),
    )
    .unwrap();
    let wrong_width = matrix(&[0.0, 0.0, 1.0, 1.0], 2, 2);
    assert_eq!(
        model.predict(&wrong_width.as_view()).unwrap_err(),
        ModelError::FeatureDimension {
            expected: 1,
            actual: 2,
        }
    );
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn signed_unit(&mut self) -> f32 {
        let fraction = (self.next() >> 40) as f32 / (1_u32 << 24) as f32;
        fraction * 2.0 - 1.0
    }
}

fn generated_matrix(rng: &mut SplitMix64, rows: usize, columns: usize) -> DenseMatrix {
    let values = (0..rows * columns).map(|_| rng.signed_unit()).collect();
    DenseMatrix::new(values, rows, columns).unwrap()
}

fn classification_labels(lane: &str, values: &DenseMatrix, seed: u64) -> BinaryTargets {
    let labels = values
        .iter_rows()
        .enumerate()
        .map(|(index, row)| {
            let score = match lane {
                "nonlinear" => {
                    row[0] * row[1] + 0.7 * row[2] * row[2] - 0.45 * row[3] + 0.2 * row[4] * row[5]
                        - 0.15
                }
                "separable" => 1.2 * row[0] - 0.9 * row[1] + 0.5 * row[2],
                "imbalanced" => 1.3 * row[0] + 0.8 * row[1] - 0.35 * row[2] * row[2] - 1.25,
                "noise" => {
                    let noise = (((index as u64 * 1_103_515_245 + seed) & 0xffff) as f64 / 32_768.0
                        - 1.0) as f32;
                    0.25 * row[0] + noise
                }
                _ => panic!("unknown quality lane {lane}"),
            };
            u8::from(score > 0.0)
        })
        .collect();
    BinaryTargets::new(labels).unwrap()
}

fn classification_data(
    lane: &str,
    seed: u64,
) -> (DenseMatrix, BinaryTargets, DenseMatrix, BinaryTargets) {
    let mut rng = SplitMix64::new(seed);
    let train = generated_matrix(&mut rng, 768, 12);
    let test = generated_matrix(&mut rng, 384, 12);
    let train_targets = classification_labels(lane, &train, seed);
    let test_targets = classification_labels(lane, &test, seed);
    (train, train_targets, test, test_targets)
}

fn regression_targets(values: &DenseMatrix, seed: u64) -> RegressionTargets {
    let targets = values
        .iter_rows()
        .enumerate()
        .map(|(index, row)| {
            let noise = (((index as u64 * 214_013 + seed * 2_531_011) & 0xffff) as f64 / 32_768.0
                - 1.0) as f32;
            1.7 * row[0] - 0.8 * row[1] * row[1]
                + 0.6 * row[2] * row[3]
                + 0.3 * row[4]
                + 0.1 * noise
        })
        .collect();
    RegressionTargets::new(targets).unwrap()
}

fn regression_data(
    seed: u64,
) -> (
    DenseMatrix,
    RegressionTargets,
    DenseMatrix,
    RegressionTargets,
) {
    let mut rng = SplitMix64::new(seed);
    let train = generated_matrix(&mut rng, 768, 12);
    let test = generated_matrix(&mut rng, 384, 12);
    let train_targets = regression_targets(&train, seed);
    let test_targets = regression_targets(&test, seed);
    (train, train_targets, test, test_targets)
}

fn accuracy(expected: &[u8], actual: &[u8]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left == right)
        .count() as f64
        / expected.len() as f64
}

fn brier(expected: &[u8], probabilities: &[f32]) -> f64 {
    expected
        .iter()
        .zip(probabilities)
        .map(|(&label, &probability)| (f64::from(probability) - f64::from(label)).powi(2))
        .sum::<f64>()
        / expected.len() as f64
}

fn nrmse(expected: &[f32], actual: &[f32]) -> f64 {
    let mean = expected.iter().copied().map(f64::from).sum::<f64>() / expected.len() as f64;
    let variance = expected
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / expected.len() as f64;
    let mse = expected
        .iter()
        .zip(actual)
        .map(|(&expected, &actual)| (f64::from(actual) - f64::from(expected)).powi(2))
        .sum::<f64>()
        / expected.len() as f64;
    mse.sqrt() / variance.sqrt()
}

#[test]
fn five_seed_classification_quality_stays_within_approved_deltas() {
    for lane in ["nonlinear", "separable", "imbalanced", "noise"] {
        let mut ferric_accuracy = 0.0;
        let mut ferric_brier = 0.0;
        let mut baseline_accuracy = 0.0;
        let mut baseline_brier = 0.0;
        for seed in QUALITY_SEEDS {
            let (train, train_y, test, test_y) = classification_data(lane, seed);
            let model = RandomForestClassifier::fit(
                &train.as_view(),
                &train_y,
                RandomForestClassifierParams::default()
                    .with_n_estimators(64)
                    .with_max_depth(Some(10))
                    .with_min_samples_leaf(2)
                    .with_max_features(MaxFeatures::Sqrt)
                    .with_random_state(seed),
            )
            .unwrap();
            ferric_accuracy +=
                accuracy(test_y.as_slice(), &model.predict(&test.as_view()).unwrap());
            ferric_brier += brier(
                test_y.as_slice(),
                &model.predict_class_proba(&test.as_view(), 1).unwrap(),
            );
            let reference = reference::QUALITY_REFERENCES
                .iter()
                .find(|reference| reference.lane == lane && reference.seed == seed)
                .unwrap();
            baseline_accuracy += reference.accuracy;
            baseline_brier += reference.brier;
        }
        let count = QUALITY_SEEDS.len() as f64;
        ferric_accuracy /= count;
        ferric_brier /= count;
        baseline_accuracy /= count;
        baseline_brier /= count;
        eprintln!(
            "quality {lane}: ferric accuracy={ferric_accuracy:.6} brier={ferric_brier:.6}; baseline accuracy={baseline_accuracy:.6} brier={baseline_brier:.6}"
        );
        assert!(
            ferric_accuracy + 0.02 >= baseline_accuracy,
            "{lane}: FerricML accuracy {ferric_accuracy:.6} trails baseline {baseline_accuracy:.6} by more than 0.02"
        );
        assert!(
            ferric_brier <= baseline_brier + 0.02,
            "{lane}: FerricML Brier {ferric_brier:.6} exceeds baseline {baseline_brier:.6} by more than 0.02"
        );
    }
}

#[test]
fn five_seed_regression_quality_stays_within_approved_delta() {
    let mut ferric_nrmse = 0.0;
    let mut baseline_nrmse = 0.0;
    for seed in QUALITY_SEEDS {
        let (train, train_y, test, test_y) = regression_data(seed);
        let model = RandomForestRegressor::fit(
            &train.as_view(),
            &train_y,
            RandomForestRegressorParams::default()
                .with_n_estimators(64)
                .with_max_depth(Some(10))
                .with_min_samples_leaf(2)
                .with_max_features(MaxFeatures::All)
                .with_random_state(seed),
        )
        .unwrap();
        ferric_nrmse += nrmse(test_y.as_slice(), &model.predict(&test.as_view()).unwrap());
        baseline_nrmse += reference::QUALITY_REFERENCES
            .iter()
            .find(|reference| reference.lane == "regression" && reference.seed == seed)
            .unwrap()
            .nrmse;
    }
    let count = QUALITY_SEEDS.len() as f64;
    ferric_nrmse /= count;
    baseline_nrmse /= count;
    eprintln!(
        "quality regression: ferric nRMSE={ferric_nrmse:.6}; baseline nRMSE={baseline_nrmse:.6}"
    );
    assert!(
        ferric_nrmse <= baseline_nrmse * 1.05,
        "FerricML nRMSE {ferric_nrmse:.6} exceeds baseline {baseline_nrmse:.6} by more than 5%"
    );
}

#[test]
fn histogram_boosting_multi_seed_quality_stays_near_frozen_baseline() {
    let mut ferric_nrmse = 0.0;
    let mut baseline_nrmse = 0.0;
    for (index, seed) in HGB_QUALITY_SEEDS.into_iter().enumerate() {
        let (train, train_y, test, test_y) = regression_data(seed);
        let model = HistGradientBoostingRegressor::fit(
            &train.as_view(),
            &train_y,
            HistGradientBoostingRegressorParams::default()
                .with_learning_rate(0.1)
                .with_max_iter(32)
                .with_max_leaf_nodes(7)
                .with_min_samples_leaf(10)
                .with_max_bins(64),
        )
        .unwrap();
        ferric_nrmse += nrmse(test_y.as_slice(), &model.predict(&test.as_view()).unwrap());
        baseline_nrmse += reference::HGB_QUALITY_NRMSE[index];
    }
    ferric_nrmse /= HGB_QUALITY_SEEDS.len() as f64;
    baseline_nrmse /= HGB_QUALITY_SEEDS.len() as f64;
    eprintln!(
        "quality histogram boosting: ferric nRMSE={ferric_nrmse:.6}; baseline nRMSE={baseline_nrmse:.6}"
    );
    assert!(
        ferric_nrmse <= baseline_nrmse * 1.05,
        "FerricML HGB nRMSE {ferric_nrmse:.6} exceeds baseline {baseline_nrmse:.6} by more than 5%"
    );
}
