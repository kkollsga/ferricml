//! FerricML conformance against frozen reference outputs.

use ferricml::api::ModelError;
use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::datasets::{Dataset, Recipe, ReferenceLane, ReferenceQuality, Source, Target};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, ExtraTreesRegressor,
    ExtraTreesRegressorParams, HistGradientBoostingClassifier,
    HistGradientBoostingClassifierParams, HistGradientBoostingRegressor,
    HistGradientBoostingRegressorParams, NJobs, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor, RandomForestRegressorParams,
};
use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LinearRegression, LinearRegressionParams,
    LogisticRegression, LogisticRegressionParams, LogisticSolver, Ridge, RidgeParams,
};
use ferricml::metrics::{
    MetricError, accuracy_score, binary_confusion_matrix, brier_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error,
};
use ferricml::model_selection::{
    ClassificationScorer, CrossValidationError, HoldoutParams, KFold, RegressionScorer,
    ScorableClassifier, ScoringError, SplitError, SplitPartition, StratifiedKFold, TestSize,
    cross_validate_classifier, cross_validate_regressor, score_classifier, score_regressor,
    stratified_train_test_split, train_test_split,
};
use ferricml::preprocessing::{
    Binarizer, BinarizerParams, MinMaxScaler, MinMaxScalerParams, Norm, Normalizer,
    NormalizerParams, RobustScaler, RobustScalerParams, StandardScaler, StandardScalerParams,
};
use ferricml::tree::MaxFeatures;
use ferricml::tree::{
    DecisionTreeClassifier, DecisionTreeClassifierParams, DecisionTreeRegressor,
    DecisionTreeRegressorParams,
};

#[allow(dead_code, clippy::excessive_precision)]
mod reference {
    include!("fixtures/reference_semantics_v1.rs");
}

#[path = "support/rng.rs"]
mod rng;

use rng::TestRng;

const EXACT_TOLERANCE: f32 = 1.0e-6;
const LOGISTIC_TOLERANCE: f32 = 2.0e-5;
const QUALITY_SEEDS: [u64; 5] = ReferenceQuality::SEEDS;
const HGB_QUALITY_SEEDS: [u64; 3] = [11, 22, 33];

/// The four classification lanes, which name themselves.
///
/// This list used to pair each lane with a separate string literal, on the
/// argument that a fixture's row key is the fixture's vocabulary rather than the
/// generator's. The exchange retired that argument: a derived container records
/// which lane it holds, so the crate owns a written name for a lane whether or
/// not this file supplies one — and two spellings that have to agree are worse
/// than one. `ReferenceLane::label` returns exactly the strings
/// `QUALITY_REFERENCES` keys its rows on, so the lookup below is now provably
/// against the same word rather than visibly against a matching one.
const QUALITY_LANES: [ReferenceLane; 4] = [
    ReferenceLane::NonlinearBinary,
    ReferenceLane::SeparableBinary,
    ReferenceLane::ImbalancedBinary,
    ReferenceLane::NoisyBinary,
];

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
fn deterministic_split_membership_and_validation_are_frozen() {
    let holdout = train_test_split(
        10,
        HoldoutParams::default()
            .with_test_size(TestSize::Count(3))
            .with_random_state(42),
    )
    .unwrap();
    assert_eq!(holdout.train_indices(), &[0, 4, 5, 6, 7, 8, 9]);
    assert_eq!(holdout.test_indices(), &[1, 2, 3]);

    let folds = KFold::new(3)
        .with_shuffle(true)
        .with_random_state(42)
        .split(8)
        .unwrap()
        .map(|split| split.test_indices().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(folds, vec![vec![1, 3, 6], vec![0, 2, 4], vec![5, 7]]);

    let labels = [0, 0, 0, 0, 1, 1, 1, 1];
    let stratified = stratified_train_test_split(
        &labels,
        HoldoutParams::default()
            .with_test_size(TestSize::Count(4))
            .with_shuffle(false),
    )
    .unwrap();
    assert_eq!(stratified.train_indices(), &[0, 1, 4, 5]);
    assert_eq!(stratified.test_indices(), &[2, 3, 6, 7]);
    let stratified_folds = StratifiedKFold::new(2)
        .split(&labels)
        .unwrap()
        .map(|split| split.test_indices().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(stratified_folds, vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]]);

    assert_eq!(
        train_test_split(
            4,
            HoldoutParams::default().with_test_size(TestSize::Fraction(f64::NAN))
        ),
        Err(SplitError::InvalidTestFraction)
    );
    assert_eq!(
        stratified_train_test_split(
            &[0, 0, 1, 1],
            HoldoutParams::default().with_test_size(TestSize::Count(1))
        ),
        Err(SplitError::PartitionTooSmallForClasses {
            partition: SplitPartition::Test,
            rows: 1,
            classes: 2,
        })
    );
}

#[test]
fn direct_estimator_scores_and_errors_are_frozen() {
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let classifier = RandomForestClassifier::fit(
        &train.as_view(),
        &BinaryTargets::new(reference::EXACT_CLASSIFIER_Y.to_vec()).unwrap(),
        exact_classifier_params(),
    )
    .unwrap();
    let expected_labels = BinaryTargets::new(reference::EXACT_LABELS.to_vec()).unwrap();
    assert_eq!(
        score_classifier(
            ScorableClassifier::probabilistic(&classifier),
            &test.as_view(),
            &expected_labels,
            ClassificationScorer::Accuracy,
        ),
        Ok(1.0)
    );
    assert_eq!(
        score_classifier(
            ScorableClassifier::probabilistic(&classifier),
            &test.as_view(),
            &expected_labels,
            ClassificationScorer::Brier,
        ),
        brier_score(expected_labels.as_slice(), &[0.0, 0.0, 0.5, 0.75, 0.75])
            .map_err(ScoringError::Metric)
    );

    let regressor = RandomForestRegressor::fit(
        &train.as_view(),
        &RegressionTargets::new(reference::EXACT_REGRESSION_Y.to_vec()).unwrap(),
        exact_regressor_params(),
    )
    .unwrap();
    let expected_regression = RegressionTargets::new(reference::EXACT_REGRESSION.to_vec()).unwrap();
    assert_eq!(
        score_regressor(
            &regressor,
            &test.as_view(),
            &expected_regression,
            RegressionScorer::MeanSquaredError,
        ),
        Ok(0.0)
    );
    assert_eq!(
        score_regressor(
            &regressor,
            &test.as_view(),
            &RegressionTargets::new(vec![0.0]).unwrap(),
            RegressionScorer::MeanSquaredError,
        ),
        Err(ScoringError::TargetLength {
            rows: 5,
            targets: 1,
        })
    );
}

#[test]
fn cross_validation_fold_scores_and_error_attribution_are_frozen() {
    let data = matrix(&[0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0], 4, 2);
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let classification = cross_validate_classifier(
        &data.as_view(),
        &binary,
        StratifiedKFold::new(2).split(binary.as_slice()).unwrap(),
        ClassificationScorer::Accuracy,
        |train, targets| {
            RandomForestClassifier::fit(
                train,
                targets,
                RandomForestClassifierParams::default()
                    .with_n_estimators(1)
                    .with_max_features(MaxFeatures::All)
                    .with_bootstrap(false),
            )
        },
        |model| ScorableClassifier::probabilistic(model),
    )
    .unwrap();
    assert_eq!(classification.scores(), &[0.5, 1.0]);
    assert_eq!(classification.mean(), 0.75);
    assert_eq!(classification.population_standard_deviation(), 0.25);

    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let regression_result = cross_validate_regressor(
        &data.as_view(),
        &regression,
        KFold::new(2).split(data.rows()).unwrap(),
        RegressionScorer::MeanSquaredError,
        |train, targets| Ridge::fit(train, targets, RidgeParams::default().with_alpha(0.0)),
    )
    .unwrap();
    assert_eq!(
        regression_result.scores(),
        &[0.029_585_803_313_369_685, 5.0]
    );
    assert_eq!(regression_result.mean(), 2.514_792_901_656_685);
    assert_eq!(
        regression_result.population_standard_deviation(),
        2.485_207_098_343_315
    );

    assert_eq!(
        cross_validate_regressor::<Ridge, _, _, _>(
            &data.as_view(),
            &regression,
            KFold::new(2).split(data.rows()).unwrap(),
            RegressionScorer::MeanSquaredError,
            |_train, _targets| Err(ModelError::LinearSolveFailed),
        ),
        Err(CrossValidationError::Fit {
            fold: 0,
            source: ModelError::LinearSolveFailed,
        })
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

/// The claimed standalone-tree subset: the standard split search, every
/// feature, a stated depth and stated node-size bounds.
///
/// Randomized split selection is deliberately absent from every exact fixture.
/// FerricML owns its generator and does not promise randomized tree identity,
/// so a frozen drawn threshold would convert a recorded non-promise into a
/// promise. Extra-trees is held to a quality envelope instead, below.
///
/// Parity is claimed here only **outside** the two recorded divergent regions:
/// `max_features` never binds, so the constant-column quota rule cannot show,
/// and no two values are near-duplicates, so the exact-distinctness rule cannot
/// either. Both regions are covered by first-party assertions in
/// `src/tree/tests.rs`, which state what the reference produces instead.
#[test]
fn exact_standalone_trees_match_frozen_reference_outputs() {
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let labels = BinaryTargets::new(reference::EXACT_CLASSIFIER_Y.to_vec()).unwrap();
    let values = RegressionTargets::new(reference::EXACT_REGRESSION_Y.to_vec()).unwrap();

    let classifier = DecisionTreeClassifier::fit(
        &train.as_view(),
        &labels,
        DecisionTreeClassifierParams::default()
            .with_max_depth(Some(2))
            .with_min_samples_split(2)
            .with_min_samples_leaf(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    assert_eq!(classifier.n_features_in(), 2);
    assert_eq!(classifier.classes(), reference::TREE_CLASSES);
    assert_eq!(
        classifier.predict(&test.as_view()).unwrap(),
        reference::TREE_LABELS
    );
    assert_close(
        &classifier.predict_proba(&test.as_view()).unwrap(),
        reference::TREE_PROBABILITIES,
    );

    let regressor = DecisionTreeRegressor::fit(
        &train.as_view(),
        &values,
        DecisionTreeRegressorParams::default()
            .with_max_depth(Some(2))
            .with_min_samples_split(2)
            .with_min_samples_leaf(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    assert_close(
        &regressor.predict(&test.as_view()).unwrap(),
        reference::TREE_REGRESSION,
    );

    // A second configuration where the node-size bounds bind, so the fixture
    // covers the parameters rather than only the default path. Both bounded
    // arrays differ from their unbounded twins, which is asserted in the
    // generator so this cannot silently become a duplicate.
    let bounded_classifier = DecisionTreeClassifier::fit(
        &train.as_view(),
        &labels,
        DecisionTreeClassifierParams::default()
            .with_max_depth(Some(3))
            .with_min_samples_split(6)
            .with_min_samples_leaf(3)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    assert_eq!(
        bounded_classifier.predict(&test.as_view()).unwrap(),
        reference::TREE_BOUNDED_LABELS
    );
    assert_close(
        &bounded_classifier.predict_proba(&test.as_view()).unwrap(),
        reference::TREE_BOUNDED_PROBABILITIES,
    );
    assert_ne!(
        bounded_classifier.predict_proba(&test.as_view()).unwrap(),
        classifier.predict_proba(&test.as_view()).unwrap()
    );

    let bounded_regressor = DecisionTreeRegressor::fit(
        &train.as_view(),
        &values,
        DecisionTreeRegressorParams::default()
            .with_max_depth(Some(3))
            .with_min_samples_split(6)
            .with_min_samples_leaf(3)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    assert_close(
        &bounded_regressor.predict(&test.as_view()).unwrap(),
        reference::TREE_BOUNDED_REGRESSION,
    );
    assert_ne!(
        bounded_regressor.predict(&test.as_view()).unwrap(),
        regressor.predict(&test.as_view()).unwrap()
    );
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

/// The multiclass exact-tree configuration: one tree, every feature, no
/// bootstrap, so nothing depends on randomized topology.
fn exact_multiclass_forest_params(max_depth: usize) -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(1)
        .with_max_depth(Some(max_depth))
        .with_max_features(MaxFeatures::All)
        .with_bootstrap(false)
        .with_random_state(0)
}

#[test]
fn joint_multinomial_logistic_matches_frozen_reference_outputs() {
    let train = matrix(reference::MULTICLASS_TRAIN_X, 12, 2);
    let test = matrix(reference::MULTICLASS_TEST_X, 4, 2);
    let targets = ClassTargets::new(reference::MULTICLASS_Y.to_vec()).unwrap();
    let model = LogisticRegression::fit_multiclass(
        &train.as_view(),
        &targets,
        LogisticRegressionParams::default()
            .with_max_iter(1000)
            .with_tol(1.0e-9),
    )
    .unwrap();

    assert_eq!(model.classes(), reference::MULTICLASS_CLASSES);
    // One coefficient row and one intercept per class, with no pinned
    // reference class.
    assert_eq!(model.n_decision_columns(), 3);
    assert_close_with_tolerance(
        model.coefficients(),
        reference::MULTINOMIAL_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        model.intercepts(),
        reference::MULTINOMIAL_INTERCEPTS,
        LOGISTIC_TOLERANCE,
    );

    let scores = model.decision_function(&test.as_view()).unwrap();
    assert_eq!(scores.len(), test.rows() * 3);
    assert_close_with_tolerance(
        &scores,
        reference::MULTINOMIAL_DECISIONS,
        LOGISTIC_TOLERANCE,
    );
    let probabilities = model.predict_proba(&test.as_view()).unwrap();
    assert_eq!(probabilities.len(), test.rows() * 3);
    assert_close_with_tolerance(
        &probabilities,
        reference::MULTINOMIAL_PROBABILITIES,
        LOGISTIC_TOLERANCE,
    );
    assert_eq!(
        model.predict(&test.as_view()).unwrap(),
        reference::MULTINOMIAL_LABELS
    );

    // The score rows are centred: they sum to zero to `f32` rounding, which is
    // what "no reference class" means observably.
    for row in scores.chunks_exact(3) {
        let magnitude = row.iter().fold(1.0_f32, |max, value| max.max(value.abs()));
        assert!(
            row.iter().sum::<f32>().abs() <= 3.0 * f32::EPSILON * magnitude,
            "score row {row:?} is not centred"
        );
    }
    // And probability rows sum to one only within the frozen tolerance; the
    // contract is explicitly not exact summation.
    let train_probabilities = model.predict_proba(&train.as_view()).unwrap();
    let mut inexact = 0;
    for row in probabilities
        .chunks_exact(3)
        .chain(train_probabilities.chunks_exact(3))
    {
        let sum = row.iter().sum::<f32>();
        assert!(
            (sum - 1.0).abs() <= 3.0 * f32::EPSILON,
            "row {row:?} sums to {sum}"
        );
        inexact += usize::from(sum != 1.0);
    }
    // The other half of the recorded divergence, and the half the tolerance
    // above cannot state: rows are **not renormalised**. A renormalising
    // implementation satisfies every assertion so far — it lands exactly on
    // 1.0, which is well inside the tolerance — so without this the divergence
    // could become silently false. The probe widens to the training rows
    // purely to give the property enough rows to show on; both matrices are
    // existing fixtures.
    assert!(
        inexact > 0,
        "every probability row summed to exactly 1.0, which is what \
         renormalising would produce; the divergence records that FerricML \
         does not renormalise"
    );
}

#[test]
fn multiclass_columns_follow_relabelled_classes_in_both_estimators() {
    let train = matrix(reference::MULTICLASS_TRAIN_X, 12, 2);
    let test = matrix(reference::MULTICLASS_TEST_X, 4, 2);
    let targets = ClassTargets::new(reference::MULTICLASS_RELABELLED_Y.to_vec()).unwrap();
    assert_eq!(targets.classes(), reference::MULTICLASS_RELABELLED_CLASSES);

    let logistic = LogisticRegression::fit_multiclass(
        &train.as_view(),
        &targets,
        LogisticRegressionParams::default()
            .with_max_iter(1000)
            .with_tol(1.0e-9),
    )
    .unwrap();
    assert_eq!(logistic.classes(), reference::MULTICLASS_RELABELLED_CLASSES);
    assert_close_with_tolerance(
        &logistic.predict_proba(&test.as_view()).unwrap(),
        reference::MULTINOMIAL_RELABELLED_PROBABILITIES,
        LOGISTIC_TOLERANCE,
    );

    let forest = RandomForestClassifier::fit_multiclass(
        &train.as_view(),
        &targets,
        exact_multiclass_forest_params(2),
    )
    .unwrap();
    assert_eq!(forest.classes(), reference::MULTICLASS_RELABELLED_CLASSES);
    assert_close(
        &forest.predict_proba(&test.as_view()).unwrap(),
        reference::FOREST_MULTICLASS_RELABELLED_PROBABILITIES,
    );
}

#[test]
fn natively_multiclass_forests_match_frozen_reference_outputs() {
    let train = matrix(reference::MULTICLASS_TRAIN_X, 12, 2);
    let test = matrix(reference::MULTICLASS_TEST_X, 4, 2);
    let targets = ClassTargets::new(reference::MULTICLASS_Y.to_vec()).unwrap();
    let model = RandomForestClassifier::fit_multiclass(
        &train.as_view(),
        &targets,
        exact_multiclass_forest_params(2),
    )
    .unwrap();
    assert_eq!(model.classes(), reference::MULTICLASS_CLASSES);
    assert_eq!(
        model.predict(&test.as_view()).unwrap(),
        reference::FOREST_MULTICLASS_LABELS
    );
    assert_close(
        &model.predict_proba(&test.as_view()).unwrap(),
        reference::FOREST_MULTICLASS_PROBABILITIES,
    );

    // A depth-one stump: its leaves hold class *distributions*, so the
    // probabilities are fractional where a vote over one tree could only ever
    // produce a zero or a one.
    let stump = RandomForestClassifier::fit_multiclass(
        &train.as_view(),
        &targets,
        exact_multiclass_forest_params(1),
    )
    .unwrap();
    let probabilities = stump.predict_proba(&test.as_view()).unwrap();
    assert!(
        probabilities
            .iter()
            .any(|&value| value != 0.0 && value != 1.0),
        "a stump's leaves must expose fractional class distributions: {probabilities:?}"
    );
    assert_close(&probabilities, reference::FOREST_STUMP_PROBABILITIES);
    assert_eq!(
        stump.predict(&test.as_view()).unwrap(),
        reference::FOREST_STUMP_LABELS
    );

    // A single observed class fits and reports one all-ones column, on a label
    // that is neither zero nor one.
    let single = RandomForestClassifier::fit_multiclass(
        &train.as_view(),
        &ClassTargets::new(vec![7; train.rows()]).unwrap(),
        exact_multiclass_forest_params(2),
    )
    .unwrap();
    assert_eq!(single.classes(), reference::FOREST_SINGLE_CLASS_CLASSES);
    let probabilities = single.predict_proba(&test.as_view()).unwrap();
    assert_eq!(probabilities.len(), test.rows());
    assert_close(&probabilities, reference::FOREST_SINGLE_CLASS_PROBABILITIES);
}

#[test]
fn a_multiclass_fit_never_changes_the_binary_one() {
    // The two entry points are different models on purpose. This asserts the
    // binary fit is untouched by the multiclass one existing: its frozen
    // reference outputs are re-checked here from the same data the multiclass
    // path uses, through the binary API.
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let binary = BinaryTargets::new(reference::EXACT_CLASSIFIER_Y.to_vec()).unwrap();
    let forest =
        RandomForestClassifier::fit(&train.as_view(), &binary, exact_classifier_params()).unwrap();
    assert_close(
        &forest.predict_proba(&test.as_view()).unwrap(),
        reference::EXACT_PROBABILITIES,
    );

    let logistic = LogisticRegression::fit(
        &matrix(reference::LOGISTIC_NO_INTERCEPT_TRAIN_X, 6, 2).as_view(),
        &BinaryTargets::new(reference::LOGISTIC_NO_INTERCEPT_Y.to_vec()).unwrap(),
        LogisticRegressionParams::default()
            .with_fit_intercept(false)
            .with_tol(1.0e-8),
    )
    .unwrap();
    assert_eq!(logistic.n_decision_columns(), 1);
    assert_eq!(
        logistic
            .decision_function(&matrix(reference::LOGISTIC_NO_INTERCEPT_TEST_X, 5, 2).as_view())
            .unwrap()
            .len(),
        5,
        "a binary decision score stays one value per row"
    );
    assert_close_with_tolerance(
        logistic.coefficients(),
        reference::LOGISTIC_NO_INTERCEPT_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
}

/// Parity with the frozen reference is a property of the *model*, not of the
/// solver that reached it.
///
/// The lanes above fit with `LogisticRegressionParams::default()`, so each of
/// them measures whichever solver is the default on the day it runs — and the
/// default moved. That is exactly the situation in which a passing suite says
/// less than it appears to: it would keep passing if one solver drifted, as
/// long as the *other* one were not the default any more.
///
/// So both are checked against the same frozen values, at the same tolerance.
/// The frozen file did not move when the default did, and could not have: it
/// records the external reference's outputs, and the reference's own default is
/// already the matrix-free path. What the flip could have moved is FerricML's
/// distance from those values, which this pins from both sides.
///
/// Measured when this was written: every worst-case residual below is identical
/// between the two solvers to four significant figures — `1.192e-7` on the
/// no-intercept coefficients, `1.812e-5` on the weighted intercept, `5.960e-8`
/// on the multinomial coefficients. Parity held exactly; it neither tightened
/// nor loosened.
#[test]
fn both_logistic_solvers_match_the_frozen_reference_outputs() {
    let train = matrix(reference::LOGISTIC_NO_INTERCEPT_TRAIN_X, 6, 2);
    let test = matrix(reference::LOGISTIC_NO_INTERCEPT_TEST_X, 5, 2);
    let binary = BinaryTargets::new(reference::LOGISTIC_NO_INTERCEPT_Y.to_vec()).unwrap();
    let weights = SampleWeights::new(reference::LOGISTIC_WEIGHTS.to_vec()).unwrap();
    let multiclass_train = matrix(reference::MULTICLASS_TRAIN_X, 12, 2);
    let multiclass_test = matrix(reference::MULTICLASS_TEST_X, 4, 2);
    let classes = ClassTargets::new(reference::MULTICLASS_Y.to_vec()).unwrap();

    for solver in [LogisticSolver::Newton, LogisticSolver::Lbfgs] {
        let no_intercept = LogisticRegression::fit(
            &train.as_view(),
            &binary,
            LogisticRegressionParams::default()
                .with_solver(solver)
                .with_fit_intercept(false)
                .with_tol(1.0e-8),
        )
        .unwrap_or_else(|error| panic!("{solver:?} no-intercept lane: {error:?}"));
        assert_close_with_tolerance(
            no_intercept.coefficients(),
            reference::LOGISTIC_NO_INTERCEPT_COEFFICIENTS,
            LOGISTIC_TOLERANCE,
        );
        assert_close_with_tolerance(
            &no_intercept.predict_proba(&test.as_view()).unwrap(),
            reference::LOGISTIC_NO_INTERCEPT_PROBABILITIES,
            LOGISTIC_TOLERANCE,
        );

        let weighted = LogisticRegression::fit_weighted(
            &train.as_view(),
            &binary,
            &weights,
            LogisticRegressionParams::default()
                .with_solver(solver)
                .with_c(0.75)
                .with_tol(1.0e-8),
        )
        .unwrap_or_else(|error| panic!("{solver:?} weighted lane: {error:?}"));
        assert_close_with_tolerance(
            weighted.coefficients(),
            reference::LOGISTIC_WEIGHTED_COEFFICIENTS,
            LOGISTIC_TOLERANCE,
        );
        assert_close_with_tolerance(
            &[weighted.intercept()],
            reference::LOGISTIC_WEIGHTED_INTERCEPT,
            LOGISTIC_TOLERANCE,
        );

        let joint = LogisticRegression::fit_multiclass(
            &multiclass_train.as_view(),
            &classes,
            LogisticRegressionParams::default()
                .with_solver(solver)
                .with_max_iter(1000)
                .with_tol(1.0e-9),
        )
        .unwrap_or_else(|error| panic!("{solver:?} multinomial lane: {error:?}"));
        assert_close_with_tolerance(
            joint.coefficients(),
            reference::MULTINOMIAL_COEFFICIENTS,
            LOGISTIC_TOLERANCE,
        );
        assert_close_with_tolerance(
            &joint.predict_proba(&multiclass_test.as_view()).unwrap(),
            reference::MULTINOMIAL_PROBABILITIES,
            LOGISTIC_TOLERANCE,
        );
    }
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

/// A tolerance and a sweep budget far past either implementation's default, so
/// the comparison is against the *optimum* both minimize rather than against
/// two different stopping rules.
fn penalized_lasso(alpha: f32) -> LassoParams {
    LassoParams::default()
        .with_alpha(alpha)
        .with_max_iter(1_000_000)
        .with_tol(1.0e-9)
}

fn penalized_elastic(alpha: f32, l1_ratio: f32) -> ElasticNetParams {
    ElasticNetParams::default()
        .with_alpha(alpha)
        .with_l1_ratio(l1_ratio)
        .with_max_iter(1_000_000)
        .with_tol(1.0e-9)
}

#[test]
fn lasso_matches_frozen_reference_outputs() {
    let train_x = matrix(reference::PENALIZED_TRAIN_X, 8, 4);
    let train_y = RegressionTargets::new(reference::PENALIZED_Y.to_vec()).unwrap();
    let test_x = matrix(reference::PENALIZED_TEST_X, 3, 4);

    let sparse = Lasso::fit(&train_x.as_view(), &train_y, penalized_lasso(0.5)).unwrap();
    assert_close_with_tolerance(
        sparse.coefficients(),
        reference::LASSO_SPARSE_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[sparse.intercept()],
        reference::LASSO_SPARSE_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &sparse.predict(&test_x.as_view()).unwrap(),
        reference::LASSO_SPARSE_PREDICTIONS,
        LOGISTIC_TOLERANCE,
    );

    // The zeros are exact on both sides, not merely small. That is the
    // behaviour the penalty exists for, so it is asserted as an equality.
    for (index, (&actual, &expected)) in sparse
        .coefficients()
        .iter()
        .zip(reference::LASSO_SPARSE_COEFFICIENTS)
        .enumerate()
    {
        assert_eq!(
            actual == 0.0,
            expected == 0.0,
            "coefficient {index} disagrees on removal: {actual} vs {expected}"
        );
    }
    assert_eq!(sparse.n_zero_coefficients(), 2);

    let weak = Lasso::fit(&train_x.as_view(), &train_y, penalized_lasso(0.01)).unwrap();
    assert_close_with_tolerance(
        weak.coefficients(),
        reference::LASSO_WEAK_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert!(weak.n_zero_coefficients() < sparse.n_zero_coefficients());

    let no_intercept = Lasso::fit(
        &train_x.as_view(),
        &train_y,
        penalized_lasso(0.5).with_fit_intercept(false),
    )
    .unwrap();
    assert_close_with_tolerance(
        no_intercept.coefficients(),
        reference::LASSO_NO_INTERCEPT_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_eq!(no_intercept.intercept().to_bits(), 0.0_f32.to_bits());

    let weights = SampleWeights::new(reference::PENALIZED_WEIGHTS.to_vec()).unwrap();
    let weighted =
        Lasso::fit_weighted(&train_x.as_view(), &train_y, &weights, penalized_lasso(0.5)).unwrap();
    assert_close_with_tolerance(
        weighted.coefficients(),
        reference::LASSO_WEIGHTED_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[weighted.intercept()],
        reference::LASSO_WEIGHTED_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
}

#[test]
fn a_removed_coefficient_diverges_from_the_reference_only_in_the_sign_of_its_zero() {
    // A recorded divergence, asserted in both directions so neither side can
    // drift silently: the reference stores a negatively signed zero for a
    // coefficient shrunk from below, and FerricML stores a positive one.
    // Mathematically the same model; a different byte pattern in storage.
    let train_x = matrix(reference::PENALIZED_TRAIN_X, 8, 4);
    let train_y = RegressionTargets::new(reference::PENALIZED_Y.to_vec()).unwrap();
    let sparse = Lasso::fit(&train_x.as_view(), &train_y, penalized_lasso(0.5)).unwrap();

    let reference_negative_zeros = reference::LASSO_SPARSE_COEFFICIENTS
        .iter()
        .filter(|value| **value == 0.0 && value.is_sign_negative())
        .count();
    assert!(
        reference_negative_zeros > 0,
        "the divergence is real, not hypothetical"
    );
    for (index, &value) in sparse.coefficients().iter().enumerate() {
        if value == 0.0 {
            assert!(
                value.is_sign_positive(),
                "coefficient {index} is a negative zero"
            );
        }
    }
}

#[test]
fn elastic_net_matches_frozen_reference_outputs() {
    let train_x = matrix(reference::PENALIZED_TRAIN_X, 8, 4);
    let train_y = RegressionTargets::new(reference::PENALIZED_Y.to_vec()).unwrap();
    let test_x = matrix(reference::PENALIZED_TEST_X, 3, 4);

    let mixed = ElasticNet::fit(&train_x.as_view(), &train_y, penalized_elastic(0.5, 0.5)).unwrap();
    assert_close_with_tolerance(
        mixed.coefficients(),
        reference::ELASTIC_NET_MIXED_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &[mixed.intercept()],
        reference::ELASTIC_NET_MIXED_INTERCEPT,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &mixed.predict(&test_x.as_view()).unwrap(),
        reference::ELASTIC_NET_MIXED_PREDICTIONS,
        LOGISTIC_TOLERANCE,
    );

    // A pure L2 mixture shrinks every coefficient and removes none, including
    // the one the sparse lane removes.
    let pure_l2 =
        ElasticNet::fit(&train_x.as_view(), &train_y, penalized_elastic(0.5, 0.0)).unwrap();
    assert_close_with_tolerance(
        pure_l2.coefficients(),
        reference::ELASTIC_NET_PURE_L2_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );

    let weights = SampleWeights::new(reference::PENALIZED_WEIGHTS.to_vec()).unwrap();
    let weighted = ElasticNet::fit_weighted(
        &train_x.as_view(),
        &train_y,
        &weights,
        penalized_elastic(0.5, 0.5),
    )
    .unwrap();
    assert_close_with_tolerance(
        weighted.coefficients(),
        reference::ELASTIC_NET_WEIGHTED_COEFFICIENTS,
        LOGISTIC_TOLERANCE,
    );

    // `l1_ratio = 1` is the lasso at the same alpha, on both sides.
    let unit_ratio =
        ElasticNet::fit(&train_x.as_view(), &train_y, penalized_elastic(0.5, 1.0)).unwrap();
    assert_close_with_tolerance(
        unit_ratio.coefficients(),
        reference::LASSO_SPARSE_COEFFICIENTS,
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

/// Robust scaling against the reference, including the degenerate shapes.
///
/// The quantile definition is the whole point of this fixture: on samples this
/// small the defensible interpolation rules disagree, so agreeing here is what
/// establishes that FerricML implements the same one. `center_` is compared at
/// the same `1e-12` tolerance as every other fitted `f64`, which is what
/// absorbs the one-ulp difference between evaluating the median through the
/// general interpolation expression and averaging the two middle order
/// statistics — FerricML does the former uniformly and deliberately carries the
/// difference here rather than in a special case in the code.
#[test]
fn robust_scaler_matches_frozen_reference_outputs() {
    let data = matrix(reference::ROBUST_TRAIN_X, 4, 3);
    let default = RobustScaler::fit(&data.as_view(), RobustScalerParams::default()).unwrap();
    assert_close_f64(default.centers(), reference::ROBUST_DEFAULT_CENTER);
    assert_close_f64(default.scales(), reference::ROBUST_DEFAULT_SCALE);
    assert_close(
        default.transform(&data.as_view()).unwrap().as_slice(),
        reference::ROBUST_DEFAULT_TRANSFORMED,
    );

    // A non-default percentile pair moves the spread but never the centre.
    let wide = RobustScaler::fit(
        &data.as_view(),
        RobustScalerParams::default().with_quantile_range(10.0, 90.0),
    )
    .unwrap();
    assert_close_f64(wide.centers(), reference::ROBUST_WIDE_CENTER);
    assert_close_f64(wide.scales(), reference::ROBUST_WIDE_SCALE);
    assert_close(
        wide.transform(&data.as_view()).unwrap().as_slice(),
        reference::ROBUST_WIDE_TRANSFORMED,
    );

    let no_centering = RobustScaler::fit(
        &data.as_view(),
        RobustScalerParams::default().with_centering(false),
    )
    .unwrap();
    assert_close(
        no_centering.transform(&data.as_view()).unwrap().as_slice(),
        reference::ROBUST_NO_CENTERING_TRANSFORMED,
    );

    let no_scaling = RobustScaler::fit(
        &data.as_view(),
        RobustScalerParams::default().with_scaling(false),
    )
    .unwrap();
    assert_close(
        no_scaling.transform(&data.as_view()).unwrap().as_slice(),
        reference::ROBUST_NO_SCALING_TRANSFORMED,
    );

    // The third training column is constant, so its divisor is the substituted
    // one and it transforms to zero under both implementations.
    assert_eq!(default.scales()[2], 1.0);
}

/// The zero-spread-but-not-constant column, where the two implementations agree
/// and the divergence FerricML declares does not bind.
///
/// FerricML substitutes a divisor of one at an *exactly* zero spread; the
/// reference substitutes below an absolute threshold. This column's spread is
/// exactly zero, so both substitute and the outputs match — including the tails
/// passing through as raw deviations from the median, which is the surprising
/// part worth freezing. The declared divergence lives at a merely *small*
/// spread, which no fixture pins because the two implementations genuinely
/// differ there by design.
#[test]
fn robust_scaler_degenerate_column_matches_the_reference() {
    let data = matrix(reference::ROBUST_DEGENERATE_X, 9, 1);
    let scaler = RobustScaler::fit(&data.as_view(), RobustScalerParams::default()).unwrap();
    assert_close_f64(scaler.centers(), reference::ROBUST_DEGENERATE_CENTER);
    assert_close_f64(scaler.scales(), reference::ROBUST_DEGENERATE_SCALE);
    assert_eq!(scaler.spreads(), &[0.0], "the raw spread really is zero");
    assert_close(
        scaler.transform(&data.as_view()).unwrap().as_slice(),
        reference::ROBUST_DEGENERATE_TRANSFORMED,
    );
}

/// Row normalization at each supported norm, including a zero row.
#[test]
fn normalizer_matches_frozen_reference_outputs() {
    let data = matrix(reference::NORMALIZER_X, 4, 3);
    for (norm, expected) in [
        (Norm::L1, reference::NORMALIZER_L1),
        (Norm::L2, reference::NORMALIZER_L2),
        (Norm::Max, reference::NORMALIZER_MAX),
    ] {
        let normalizer =
            Normalizer::fit(&data.as_view(), NormalizerParams::default().with_norm(norm)).unwrap();
        assert_close(
            normalizer.transform(&data.as_view()).unwrap().as_slice(),
            expected,
        );
    }
}

/// Thresholding, including the boundary value itself.
///
/// The fixture deliberately contains a value exactly at each threshold, because
/// that is where a strict and a non-strict comparison disagree and everywhere
/// else they do not.
#[test]
fn binarizer_matches_frozen_reference_outputs() {
    let data = matrix(reference::BINARIZER_X, 3, 3);
    for (threshold, expected) in [
        (0.0, reference::BINARIZER_DEFAULT),
        (2.0, reference::BINARIZER_AT_TWO),
    ] {
        let binarizer = Binarizer::fit(
            &data.as_view(),
            BinarizerParams::default().with_threshold(threshold),
        )
        .unwrap();
        assert_close(
            binarizer.transform(&data.as_view()).unwrap().as_slice(),
            expected,
        );
    }
}

/// Min-max scaling onto a non-default output range.
///
/// The constant third column is the case worth pinning: its divisor is the
/// substituted one, so it lands on the range's lower bound rather than on zero.
#[test]
fn min_max_scaler_feature_range_matches_frozen_reference_outputs() {
    let data = matrix(reference::MIN_MAX_RANGE_X, 4, 3);
    let scaler = MinMaxScaler::fit(
        &data.as_view(),
        MinMaxScalerParams::default().with_feature_range(-1.0, 1.0),
    )
    .unwrap();
    assert_close_f64(scaler.scales(), reference::MIN_MAX_RANGE_SCALE);
    assert_close_f64(scaler.offsets(), reference::MIN_MAX_RANGE_MIN);
    assert_close(
        scaler.transform(&data.as_view()).unwrap().as_slice(),
        reference::MIN_MAX_RANGE_TRANSFORMED,
    );
    assert_eq!(
        scaler.transform(&data.as_view()).unwrap().get(0, 2),
        Some(-1.0),
        "a constant column lands on the lower bound, not on zero"
    );
}

/// Weighted tree fitting against the reference, on the exact single-tree
/// configuration where nothing depends on randomized topology.
///
/// **One deliberate divergence, and it is why this fixture keeps
/// `min_samples_leaf` at one.** The reference bounds the minimum split and leaf
/// sizes by the number of *rows* in a node; FerricML bounds them by the node's
/// total *weight*. That is what makes an integer weight the same fitted model
/// as repeating the row unconditionally in FerricML, where the reference only
/// has that equivalence while the constraint does not bind. With the bound at
/// one it never binds, so this compares the weighted impurity and leaf
/// arithmetic alone — which is where the two must agree.
#[test]
fn weighted_forests_match_frozen_reference_outputs() {
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let weights = SampleWeights::new(reference::FOREST_WEIGHTS.to_vec()).unwrap();

    let classifier = RandomForestClassifier::fit_weighted(
        &train.as_view(),
        &BinaryTargets::new(reference::EXACT_CLASSIFIER_Y.to_vec()).unwrap(),
        &weights,
        exact_classifier_params(),
    )
    .unwrap();
    assert_eq!(
        classifier.predict(&test.as_view()).unwrap(),
        reference::FOREST_WEIGHTED_LABELS
    );
    assert_close(
        &classifier.predict_proba(&test.as_view()).unwrap(),
        reference::FOREST_WEIGHTED_PROBABILITIES,
    );

    let regressor = RandomForestRegressor::fit_weighted(
        &train.as_view(),
        &RegressionTargets::new(reference::EXACT_REGRESSION_Y.to_vec()).unwrap(),
        &weights,
        exact_regressor_params(),
    )
    .unwrap();
    assert_close(
        &regressor.predict(&test.as_view()).unwrap(),
        reference::FOREST_WEIGHTED_REGRESSION,
    );
}

/// The recorded node-size divergence, in the one region where it is visible.
///
/// Every other weighted tree fixture holds `min_samples_split` at two and
/// `min_samples_leaf` at one with every weight at least one. In that region the
/// summed-weight rule and the reference's row-count rule are provably the same
/// function — a non-empty node weighs at least one, and a one-row node cannot
/// split under either bound whatever it weighs — so those fixtures pin the
/// weighted impurity arithmetic and say nothing about the bounds. This one pins
/// the bounds, and it is the fixture that fails if FerricML ever adopts row
/// counting.
///
/// The weights straddle one and the split bound is three, so one fitted tree
/// separates the rules in **both** directions:
///
/// * **FerricML is stricter.** The classifier's best second split puts two rows
///   weighing `0.25` each into one child. The reference admits it — two rows is
///   at least one row — and FerricML refuses it, because `0.5` is below a
///   `min_samples_leaf` of one. FerricML takes the next admissible split
///   instead, so the whole left subtree differs. The regressor shows the same
///   direction on the split bound: a four-row node weighing `1.0` is a leaf in
///   FerricML and splits twice more in the reference.
/// * **FerricML is looser.** Two-row nodes weighing `4.0` are leaves in the
///   reference, which sees two rows against a bound of three, and FerricML
///   splits them, because four is not below three.
///
/// Both sets of values are asserted exactly. The reference's come from the
/// frozen fixture, whose generator asserts that both node shapes really occur;
/// FerricML's are written here, because no reference run can produce them.
#[test]
fn fractional_weights_separate_the_weight_bound_from_the_reference_row_bound() {
    let train = matrix(reference::EXACT_TRAIN_X, 8, 2);
    let test = matrix(reference::EXACT_TEST_X, 5, 2);
    let weights = SampleWeights::new(reference::FRACTIONAL_WEIGHTS.to_vec()).unwrap();

    // The straddle is a property of the frozen weights, not of this test.
    assert!(
        reference::FRACTIONAL_WEIGHTS.iter().any(|&w| w < 1.0)
            && reference::FRACTIONAL_WEIGHTS.iter().any(|&w| w > 1.0),
        "the fixture weights must fall on both sides of one, or neither bound diverges"
    );

    let classifier = DecisionTreeClassifier::fit_weighted(
        &train.as_view(),
        &BinaryTargets::new(reference::EXACT_CLASSIFIER_Y.to_vec()).unwrap(),
        &weights,
        DecisionTreeClassifierParams::default()
            .with_max_depth(None)
            .with_min_samples_split(3)
            .with_min_samples_leaf(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    let labels = classifier.predict(&test.as_view()).unwrap();
    let probabilities = classifier.predict_proba(&test.as_view()).unwrap();
    assert_eq!(labels, [1, 1, 1, 1, 1]);
    assert_close(
        &probabilities,
        &[0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.0, 1.0],
    );
    assert_ne!(
        labels.as_slice(),
        reference::FRACTIONAL_ROW_BOUND_LABELS,
        "the row-count rule and the weight rule must disagree here, or this \
         fixture pins nothing the unweighted fixtures do not"
    );
    assert_ne!(
        probabilities.as_slice(),
        reference::FRACTIONAL_ROW_BOUND_PROBABILITIES
    );

    let regressor = DecisionTreeRegressor::fit_weighted(
        &train.as_view(),
        &RegressionTargets::new(reference::EXACT_REGRESSION_Y.to_vec()).unwrap(),
        &weights,
        DecisionTreeRegressorParams::default()
            .with_max_depth(None)
            .with_min_samples_split(3)
            .with_min_samples_leaf(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    let predictions = regressor.predict(&test.as_view()).unwrap();
    // Index 0..3 is the stricter direction: a four-row node weighing 1.0 stays a
    // leaf, so all three rows read its weighted mean of 0.5 rather than the
    // reference's -0.5, -0.5 and 1.5. Indices 3 and 4 are the looser direction:
    // two-row nodes weighing 4.0 split, so each test row reaches a single
    // training target instead of a two-row mean.
    assert_close(&predictions, &[0.5, 0.5, 0.5, 4.0, 7.0]);
    for (index, (&actual, &row_bound)) in predictions
        .iter()
        .zip(reference::FRACTIONAL_ROW_BOUND_REGRESSION)
        .enumerate()
    {
        assert_ne!(
            actual, row_bound,
            "row {index} must separate the two bound rules"
        );
    }
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

/// The same one-step boosted configuration, weighted. The bin grid is fitted
/// from the distinct observed values in both implementations, so weighting
/// moves only the baseline and the leaf arithmetic.
#[test]
fn weighted_histogram_boosting_matches_frozen_reference_outputs() {
    let train = matrix(reference::HGB_TRAIN_X, 8, 1);
    let targets = RegressionTargets::new(reference::HGB_TRAIN_Y.to_vec()).unwrap();
    let test = matrix(reference::HGB_TEST_X, 4, 1);
    let weights = SampleWeights::new(reference::HGB_WEIGHTS.to_vec()).unwrap();
    let model = HistGradientBoostingRegressor::fit_weighted(
        &train.as_view(),
        &targets,
        &weights,
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
        reference::HGB_WEIGHTED_PREDICTIONS,
    );
}

/// One boosting step of the binary log loss, against the reference's public
/// `decision_function`, `predict_proba`, and `predict`.
///
/// The labels are balanced, so the baseline log-odds is exactly zero and every
/// row starts at `p = 0.5`. Each side of the split at 3.5 then carries gradient
/// `+-2` and curvature `1`, which is what makes the fitted score `-+2` an exact
/// value both implementations must reach rather than a number to compare
/// loosely. It is also the case that separates this model from a squared-error
/// fit of the same labels: dividing by the row count instead of the curvature
/// would give `+-0.5`.
#[test]
fn boosted_classification_matches_frozen_reference_one_step_outputs() {
    let train = matrix(reference::HGB_TRAIN_X, 8, 1);
    let targets = BinaryTargets::new(reference::HGBC_TRAIN_Y.to_vec()).unwrap();
    let test = matrix(reference::HGB_TEST_X, 4, 1);
    let model = HistGradientBoostingClassifier::fit(
        &train.as_view(),
        &targets,
        one_step_boosted_classifier_params(),
    )
    .unwrap();
    assert_eq!(model.n_iter(), 1);
    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(model.baseline(), 0.0);
    assert_close_with_tolerance(
        &model.decision_function(&test.as_view()).unwrap(),
        reference::HGBC_DECISIONS,
        LOGISTIC_TOLERANCE,
    );
    assert_close_with_tolerance(
        &model.predict_proba(&test.as_view()).unwrap(),
        reference::HGBC_PROBABILITIES,
        LOGISTIC_TOLERANCE,
    );
    assert_eq!(
        model.predict(&test.as_view()).unwrap(),
        reference::HGBC_LABELS
    );
}

/// The same one-step configuration, weighted.
///
/// The leaf bound is one so it never binds, which is what keeps this comparison
/// meaningful: FerricML bounds summed weight where the reference counts rows,
/// and that recorded divergence is observable only when the bound binds.
#[test]
fn weighted_boosted_classification_matches_frozen_reference_outputs() {
    let train = matrix(reference::HGB_TRAIN_X, 8, 1);
    let targets = BinaryTargets::new(reference::HGBC_TRAIN_Y.to_vec()).unwrap();
    let test = matrix(reference::HGB_TEST_X, 4, 1);
    let weights = SampleWeights::new(reference::HGB_WEIGHTS.to_vec()).unwrap();
    let model = HistGradientBoostingClassifier::fit_weighted(
        &train.as_view(),
        &targets,
        &weights,
        one_step_boosted_classifier_params(),
    )
    .unwrap();
    assert_eq!(model.n_iter(), 1);
    assert_close_with_tolerance(
        &model.decision_function(&test.as_view()).unwrap(),
        reference::HGBC_WEIGHTED_DECISIONS,
        LOGISTIC_TOLERANCE,
    );
    // Weighting is not inert: an unbalanced positive rate moves the baseline
    // off zero, so the two fits are genuinely different models.
    assert_ne!(model.baseline(), 0.0);
}

fn one_step_boosted_classifier_params() -> HistGradientBoostingClassifierParams {
    HistGradientBoostingClassifierParams::default()
        .with_learning_rate(1.0)
        .with_max_iter(1)
        .with_max_leaf_nodes(2)
        .with_min_samples_leaf(1)
}

/// Boosted classification quality against the reference's own boosted
/// classifier, on the same three seeds its regressor sibling uses.
///
/// The allowances are the crate-wide classification ones — at most 0.02 accuracy
/// behind and at most 0.02 Brier above — evaluated on the mean over seeds rather
/// than per seed, because a per-seed comparison of two different tree searches
/// measures tie-breaking rather than quality.
#[test]
fn boosted_classification_multi_seed_quality_stays_near_frozen_baseline() {
    let mut ferric_accuracy = 0.0;
    let mut ferric_brier = 0.0;
    let mut baseline_accuracy = 0.0;
    let mut baseline_brier = 0.0;
    for (index, seed) in HGB_QUALITY_SEEDS.into_iter().enumerate() {
        let (train, train_y, test, test_y) =
            classification_data(ReferenceLane::NonlinearBinary, seed);
        let model = HistGradientBoostingClassifier::fit(
            &train.as_view(),
            &train_y,
            HistGradientBoostingClassifierParams::default()
                .with_learning_rate(0.1)
                .with_max_iter(32)
                .with_max_leaf_nodes(7)
                .with_min_samples_leaf(10)
                .with_max_bins(64),
        )
        .unwrap();
        ferric_accuracy +=
            accuracy_score(test_y.as_slice(), &model.predict(&test.as_view()).unwrap()).unwrap();
        ferric_brier += brier_score(
            test_y.as_slice(),
            &model.predict_class_proba(&test.as_view(), 1).unwrap(),
        )
        .unwrap();
        baseline_accuracy += reference::HGBC_QUALITY_ACCURACY[index];
        baseline_brier += reference::HGBC_QUALITY_BRIER[index];
    }
    let count = HGB_QUALITY_SEEDS.len() as f64;
    ferric_accuracy /= count;
    ferric_brier /= count;
    baseline_accuracy /= count;
    baseline_brier /= count;
    eprintln!(
        "quality boosted classification: ferric accuracy={ferric_accuracy:.6} brier={ferric_brier:.6}; baseline accuracy={baseline_accuracy:.6} brier={baseline_brier:.6}"
    );
    assert!(
        ferric_accuracy + 0.02 >= baseline_accuracy,
        "FerricML boosted accuracy {ferric_accuracy:.6} trails baseline {baseline_accuracy:.6} by more than 0.02"
    );
    assert!(
        ferric_brier <= baseline_brier + 0.02,
        "FerricML boosted Brier {ferric_brier:.6} exceeds baseline {baseline_brier:.6} by more than 0.02"
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

/// The exact stream every quality lane below is generated from.
///
/// This file used to carry a private SplitMix64 whose core was
/// character-identical to `src/numeric/rng.rs`'s and to
/// `tests/artifact_hardening.rs`'s — three copies of one stream. It now draws
/// from the one test-crate generator, and this test is what made that a
/// refactor rather than a fixture change: the literals were captured from the
/// private copy before it was deleted, so they are what the frozen reference
/// outputs in `fixtures/reference_semantics_v1.rs` were recorded against.
///
/// The quality lanes themselves could not have proven this. They compare
/// aggregate accuracy and Brier score against the reference within `0.02`, so a
/// generator emitting a *different but similarly distributed* stream would pass
/// them while silently changing every design matrix. That is exactly the shape
/// of a fixture-moving change disguised as a refactor, and it is why the stream
/// is pinned here by value.
///
/// `TestRng::from_state` is the raw-seed constructor for that reason:
/// `TestRng::new` perturbs the seed and would have moved every fixture.
///
/// The lanes have since moved again, into `ferricml::datasets` — the crate now
/// owns the generator it measures itself against. The literals below did not
/// move with them, and the final assertion is what makes that check-able rather
/// than asserted: the ported `Source::Sampled` design and this file's surviving
/// `TestRng` composition are compared against each other as well as against the
/// values recorded before either existed. `TestRng` itself stays, because five
/// other test binaries sweep and fuzz from it and `test-rng-single-source`
/// requires its markers present.
#[test]
fn the_generated_design_stream_is_frozen_bit_for_bit() {
    let raw: [(u64, [u64; 6]); 2] = [
        (
            0,
            [
                16294208416658607535,
                7960286522194355700,
                487617019471545679,
                17909611376780542444,
                1961750202426094747,
                6038094601263162090,
            ],
        ),
        (
            11,
            [
                5833679380957638813,
                4839782808629744545,
                11769803791402734189,
                9308485889748266480,
                3047264704176347588,
                10181453352864339982,
            ],
        ),
    ];
    for (seed, stream) in raw {
        let mut rng = TestRng::from_state(seed);
        let actual: Vec<u64> = (0..stream.len()).map(|_| rng.next_u64()).collect();
        assert_eq!(actual, stream, "raw stream changed for seed {seed}");
    }

    // The `f32` draw, which is what a design matrix is made of. Compared with
    // `assert_eq!` rather than a tolerance because every operation in it is
    // exact and any difference at all would move a fixture.
    let mut rng = TestRng::from_state(0);
    let signed: Vec<f32> = (0..6).map(|_| rng.signed_unit()).collect();
    assert_eq!(
        signed,
        vec![
            0.7666216,
            -0.13694406,
            -0.94713247,
            0.9417639,
            -0.78730667,
            -0.34534848
        ]
    );

    // And the composition the lanes actually call, at the first quality seed —
    // now reached two ways. The first is what this file did before the lanes
    // moved; the second is the crate's own generator, named on the raw state
    // rather than on a derived one. Both are compared against the literals, and
    // then against each other, so a port that changed the map would have to
    // change three things at once to stay green.
    let mut rng = TestRng::from_state(11);
    let composed: Vec<f32> = (0..6).map(|_| rng.signed_unit()).collect();
    let ported = Recipe::new(2, 3, Source::Sampled { state: 11 })
        .unwrap()
        .design();
    assert_eq!(
        composed,
        [
            -0.36751127,
            -0.4752698,
            0.27608466,
            0.009227991,
            -0.6696149,
            0.10387528
        ]
    );
    assert_eq!(ported.as_slice(), composed);
}

/// One quality lane's split, from the preset that absorbed it.
///
/// The generator this used to call lived in this file; it now lives in
/// `ferricml::datasets`, pinned by value there against literals captured from
/// this file's copy before it was deleted. What remains here is the adapter
/// from the generator's vocabulary to the containers the estimators take.
fn classification_data(
    lane: ReferenceLane,
    seed: u64,
) -> (DenseMatrix, BinaryTargets, DenseMatrix, BinaryTargets) {
    let preset = ReferenceQuality::new(lane, seed);
    let (train, train_targets) = binary_split(preset.train());
    let (test, test_targets) = binary_split(preset.test());
    (train, train_targets, test, test_targets)
}

fn binary_split(dataset: Dataset) -> (DenseMatrix, BinaryTargets) {
    let targets = match dataset.target() {
        Some(Target::Binary(targets)) => targets.clone(),
        other => panic!("a classification lane produced {other:?}"),
    };
    (dataset.into_features(), targets)
}

fn regression_data(
    seed: u64,
) -> (
    DenseMatrix,
    RegressionTargets,
    DenseMatrix,
    RegressionTargets,
) {
    let preset = ReferenceQuality::new(ReferenceLane::Regression, seed);
    let (train, train_targets) = regression_split(preset.train());
    let (test, test_targets) = regression_split(preset.test());
    (train, train_targets, test, test_targets)
}

fn regression_split(dataset: Dataset) -> (DenseMatrix, RegressionTargets) {
    let targets = match dataset.target() {
        Some(Target::Regression(targets)) => targets.clone(),
        other => panic!("the regression lane produced {other:?}"),
    };
    (dataset.into_features(), targets)
}

fn normalized_root_mean_squared_error(expected: &[f32], actual: &[f32]) -> f64 {
    let mean = expected.iter().copied().map(f64::from).sum::<f64>() / expected.len() as f64;
    let variance = expected
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / expected.len() as f64;
    root_mean_squared_error(expected, actual).unwrap() / variance.sqrt()
}

#[test]
fn five_seed_classification_quality_stays_within_approved_deltas() {
    for lane in QUALITY_LANES {
        let name = lane.label();
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
                accuracy_score(test_y.as_slice(), &model.predict(&test.as_view()).unwrap())
                    .unwrap();
            ferric_brier += brier_score(
                test_y.as_slice(),
                &model.predict_class_proba(&test.as_view(), 1).unwrap(),
            )
            .unwrap();
            let reference = reference::QUALITY_REFERENCES
                .iter()
                .find(|reference| reference.lane == name && reference.seed == seed)
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
            "quality {name}: ferric accuracy={ferric_accuracy:.6} brier={ferric_brier:.6}; baseline accuracy={baseline_accuracy:.6} brier={baseline_brier:.6}"
        );
        assert!(
            ferric_accuracy + 0.02 >= baseline_accuracy,
            "{name}: FerricML accuracy {ferric_accuracy:.6} trails baseline {baseline_accuracy:.6} by more than 0.02"
        );
        assert!(
            ferric_brier <= baseline_brier + 0.02,
            "{name}: FerricML Brier {ferric_brier:.6} exceeds baseline {baseline_brier:.6} by more than 0.02"
        );
    }
}

/// Extra-trees is held to a quality envelope and to nothing exact.
///
/// Its thresholds come out of each implementation's own generator, so the two
/// never agree value-for-value and no tolerance would make them. What *is*
/// comparable is whether the randomized ensemble is as good a model, and that
/// is what this checks. It is also the lane that would catch a split search
/// randomized in name only: a `Splitter::Random` that had silently fallen back
/// to the exhaustive sweep would still score well here, but the bitwise
/// equivalence tests in `src/ensemble/equivalence.rs` cover that half — the two
/// mechanisms answer different questions and neither substitutes for the other.
#[test]
fn five_seed_extra_trees_quality_stays_within_approved_deltas() {
    let mut ferric_accuracy = 0.0;
    let mut ferric_brier = 0.0;
    let mut baseline_accuracy = 0.0;
    let mut baseline_brier = 0.0;
    for seed in QUALITY_SEEDS {
        let (train, train_y, test, test_y) =
            classification_data(ReferenceLane::NonlinearBinary, seed);
        let model = ExtraTreesClassifier::fit(
            &train.as_view(),
            &train_y,
            ExtraTreesClassifierParams::default()
                .with_n_estimators(64)
                .with_max_depth(Some(10))
                .with_min_samples_leaf(2)
                .with_max_features(MaxFeatures::Sqrt)
                .with_random_state(seed),
        )
        .unwrap();
        ferric_accuracy +=
            accuracy_score(test_y.as_slice(), &model.predict(&test.as_view()).unwrap()).unwrap();
        ferric_brier += brier_score(
            test_y.as_slice(),
            &model.predict_class_proba(&test.as_view(), 1).unwrap(),
        )
        .unwrap();
        let reference = reference::QUALITY_REFERENCES
            .iter()
            .find(|reference| reference.lane == "extra_trees_nonlinear" && reference.seed == seed)
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
        "quality extra_trees_nonlinear: ferric accuracy={ferric_accuracy:.6} brier={ferric_brier:.6}; baseline accuracy={baseline_accuracy:.6} brier={baseline_brier:.6}"
    );
    assert!(
        ferric_accuracy + 0.02 >= baseline_accuracy,
        "extra-trees accuracy {ferric_accuracy:.6} trails baseline {baseline_accuracy:.6} by more than 0.02"
    );
    assert!(
        ferric_brier <= baseline_brier + 0.02,
        "extra-trees Brier {ferric_brier:.6} exceeds baseline {baseline_brier:.6} by more than 0.02"
    );

    let mut ferric_nrmse = 0.0;
    let mut baseline_nrmse = 0.0;
    for seed in QUALITY_SEEDS {
        let (train, train_y, test, test_y) = regression_data(seed);
        let model = ExtraTreesRegressor::fit(
            &train.as_view(),
            &train_y,
            ExtraTreesRegressorParams::default()
                .with_n_estimators(64)
                .with_max_depth(Some(10))
                .with_min_samples_leaf(2)
                .with_max_features(MaxFeatures::All)
                .with_random_state(seed),
        )
        .unwrap();
        ferric_nrmse += normalized_root_mean_squared_error(
            test_y.as_slice(),
            &model.predict(&test.as_view()).unwrap(),
        );
        baseline_nrmse += reference::QUALITY_REFERENCES
            .iter()
            .find(|reference| reference.lane == "extra_trees_regression" && reference.seed == seed)
            .unwrap()
            .nrmse;
    }
    let count = QUALITY_SEEDS.len() as f64;
    ferric_nrmse /= count;
    baseline_nrmse /= count;
    eprintln!(
        "quality extra_trees_regression: ferric nRMSE={ferric_nrmse:.6}; baseline nRMSE={baseline_nrmse:.6}"
    );
    assert!(
        ferric_nrmse <= baseline_nrmse * 1.05,
        "extra-trees nRMSE {ferric_nrmse:.6} exceeds baseline {baseline_nrmse:.6} by more than 5%"
    );
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
        ferric_nrmse += normalized_root_mean_squared_error(
            test_y.as_slice(),
            &model.predict(&test.as_view()).unwrap(),
        );
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
        ferric_nrmse += normalized_root_mean_squared_error(
            test_y.as_slice(),
            &model.predict(&test.as_view()).unwrap(),
        );
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
