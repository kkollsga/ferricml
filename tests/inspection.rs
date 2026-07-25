use ferricml::api::{AnyRegressor, ModelError, Regressor};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures,
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::inspection::{
    InspectionError, PermutationImportanceParams, permutation_importance_classifier,
    permutation_importance_classifier_into, permutation_importance_regressor,
    permutation_importance_regressor_into,
};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams, Ridge, RidgeParams};
use ferricml::metrics::mean_squared_error;
use ferricml::model_selection::{
    ClassificationScorer, RegressionScore, RegressionScorer, ScorableClassifier, ScoringError,
};

/// Four columns: a dominant signal, a weak signal, a constant, and a copy of
/// the constant. Only the first two can carry information.
fn regression_fixture() -> (DenseMatrix, RegressionTargets) {
    let rows = 64;
    let mut values = Vec::with_capacity(rows * 4);
    let mut targets = Vec::with_capacity(rows);
    for row in 0..rows {
        let dominant = ((row * 37 % 61) as f32 / 30.0) - 1.0;
        let weak = ((row * 17 % 23) as f32 / 11.0) - 1.0;
        values.extend_from_slice(&[dominant, weak, 1.0, 1.0]);
        targets.push(8.0 * dominant + 0.25 * weak);
    }
    (
        DenseMatrix::new(values, rows, 4).unwrap(),
        RegressionTargets::new(targets).unwrap(),
    )
}

fn classification_fixture() -> (DenseMatrix, BinaryTargets) {
    let (data, regression) = regression_fixture();
    let labels = regression
        .as_slice()
        .iter()
        .map(|&value| u8::from(value > 0.0))
        .collect();
    (data, BinaryTargets::new(labels).unwrap())
}

fn forest_regressor(data: &DenseMatrix, targets: &RegressionTargets) -> RandomForestRegressor {
    RandomForestRegressor::fit(
        &data.as_view(),
        targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(12)
            .with_max_features(MaxFeatures::All)
            .with_random_state(4),
    )
    .unwrap()
}

fn params(n_repeats: usize, random_state: u64) -> PermutationImportanceParams {
    PermutationImportanceParams::default()
        .with_n_repeats(n_repeats)
        .with_random_state(random_state)
}

#[test]
fn importance_is_deterministic_for_a_fixed_seed_and_changes_with_it() {
    let (data, targets) = regression_fixture();
    let model = forest_regressor(&data, &targets);
    let run = |seed| {
        permutation_importance_regressor(
            &model,
            &data.as_view(),
            &targets,
            RegressionScorer::MeanSquaredError,
            params(6, seed),
        )
        .unwrap()
    };
    let first = run(7);
    assert_eq!(first, run(7));
    assert_eq!(first.means(), run(7).means());
    assert_eq!(first.std_devs(), run(7).std_devs());
    assert_ne!(first.means(), run(8).means());

    // The allocating and caller-owned entry points agree exactly.
    let mut means = vec![0.0; data.columns()];
    let mut std_devs = vec![0.0; data.columns()];
    permutation_importance_regressor_into(
        &model,
        &data.as_view(),
        &targets,
        RegressionScorer::MeanSquaredError,
        params(6, 7),
        &mut means,
        &mut std_devs,
    )
    .unwrap();
    assert_eq!(means, first.means());
    assert_eq!(std_devs, first.std_devs());
}

#[test]
fn a_dominant_feature_outranks_a_weak_one_and_uninformative_columns_score_near_zero() {
    let (data, targets) = regression_fixture();
    let model = forest_regressor(&data, &targets);
    for scorer in [
        RegressionScorer::MeanSquaredError,
        RegressionScorer::MeanAbsoluteError,
        RegressionScorer::RootMeanSquaredError,
        RegressionScorer::R2,
    ] {
        let importance = permutation_importance_regressor(
            &model,
            &data.as_view(),
            &targets,
            scorer,
            params(8, 11),
        )
        .unwrap();
        assert_eq!(importance.n_features(), 4);
        assert_eq!(
            importance.ranked()[0],
            0,
            "{scorer:?} did not rank the dominant feature first"
        );
        assert!(
            importance.means()[0] > importance.means()[1],
            "{scorer:?}: dominant {} vs weak {}",
            importance.means()[0],
            importance.means()[1]
        );
        // A constant column, and a duplicate of it, cannot be permuted into a
        // different matrix, so their importance is exactly zero.
        assert_eq!(importance.means()[2], 0.0, "{scorer:?} constant feature");
        assert_eq!(importance.means()[3], 0.0, "{scorer:?} duplicated feature");
        assert_eq!(importance.std_devs()[2], 0.0);
        assert_eq!(importance.std_devs()[3], 0.0);
        // Every orientation reports destroying signal as a positive loss.
        assert!(importance.means()[0] > 0.0, "{scorer:?} orientation");
    }
}

#[test]
fn an_ignored_feature_scores_near_zero_even_when_it_varies() {
    // The model is fitted on the informative columns only, then inspected on a
    // matrix whose fourth column is pure noise it never saw as signal.
    let rows = 48;
    let mut values = Vec::with_capacity(rows * 2);
    let mut targets = Vec::with_capacity(rows);
    for row in 0..rows {
        let signal = (row as f32) / 8.0;
        let noise = ((row * 29 % 17) as f32) - 8.0;
        values.extend_from_slice(&[signal, noise]);
        targets.push(3.0 * signal);
    }
    let data = DenseMatrix::new(values, rows, 2).unwrap();
    let targets = RegressionTargets::new(targets).unwrap();
    let model = LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
        .unwrap();
    let importance = permutation_importance_regressor(
        &model,
        &data.as_view(),
        &targets,
        RegressionScorer::MeanSquaredError,
        params(8, 3),
    )
    .unwrap();
    assert_eq!(importance.ranked(), vec![0, 1]);
    assert!(importance.means()[0] > 1.0, "{:?}", importance.means());
    assert!(
        importance.means()[1].abs() < 1.0e-3,
        "{:?}",
        importance.means()
    );
}

#[test]
fn every_estimator_family_is_inspected_through_the_same_contract() {
    let (data, targets) = regression_fixture();
    let forest = forest_regressor(&data, &targets);
    let ridge = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
    let linear =
        LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
            .unwrap();
    let boosted = HistGradientBoostingRegressor::fit(
        &data.as_view(),
        &targets,
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(8)
            .with_max_leaf_nodes(4)
            .with_min_samples_leaf(2),
    )
    .unwrap();
    let erased: AnyRegressor = forest.clone().into();
    let models: [&dyn Regressor; 5] = [&forest, &ridge, &linear, &boosted, &erased];
    for model in models {
        let importance = permutation_importance_regressor(
            model,
            &data.as_view(),
            &targets,
            RegressionScorer::R2,
            params(4, 21),
        )
        .unwrap();
        assert_eq!(importance.ranked()[0], 0);
        assert_eq!(importance.means()[2], 0.0);
    }

    // A runtime-erased model and its concrete original agree exactly.
    let concrete = permutation_importance_regressor(
        &forest,
        &data.as_view(),
        &targets,
        RegressionScorer::R2,
        params(4, 21),
    )
    .unwrap();
    let dispatched = permutation_importance_regressor(
        &erased,
        &data.as_view(),
        &targets,
        RegressionScorer::R2,
        params(4, 21),
    )
    .unwrap();
    assert_eq!(concrete, dispatched);
}

#[test]
fn classifier_importance_covers_label_and_probability_scorers() {
    let (data, targets) = classification_fixture();
    let model = RandomForestClassifier::fit(
        &data.as_view(),
        &targets,
        RandomForestClassifierParams::default()
            .with_n_estimators(12)
            .with_max_features(MaxFeatures::All)
            .with_random_state(9),
    )
    .unwrap();
    for scorer in [
        ClassificationScorer::Accuracy,
        ClassificationScorer::Precision,
        ClassificationScorer::Recall,
        ClassificationScorer::F1,
        ClassificationScorer::Brier,
        ClassificationScorer::LogLoss,
        ClassificationScorer::RocAuc,
    ] {
        let importance = permutation_importance_classifier(
            ScorableClassifier::probabilistic(&model),
            &data.as_view(),
            &targets,
            scorer,
            params(6, 13),
        )
        .unwrap();
        assert_eq!(importance.n_features(), 4);
        assert_eq!(
            importance.ranked()[0],
            0,
            "{scorer:?} ranked {:?}",
            importance.means()
        );
        assert!(importance.means()[0] > 0.0, "{scorer:?} orientation");
        assert_eq!(importance.means()[2], 0.0, "{scorer:?} constant feature");
    }

    // A single-class model exposes a degenerate probability column, and every
    // feature is then equally irrelevant.
    let single = RandomForestClassifier::fit(
        &data.as_view(),
        &BinaryTargets::new(vec![1; data.rows()]).unwrap(),
        RandomForestClassifierParams::default().with_n_estimators(2),
    )
    .unwrap();
    assert_eq!(single.classes(), &[1]);
    let importance = permutation_importance_classifier(
        ScorableClassifier::probabilistic(&single),
        &data.as_view(),
        &targets,
        ClassificationScorer::Brier,
        params(3, 1),
    )
    .unwrap();
    assert!(importance.means().iter().all(|&value| value == 0.0));
}

#[test]
fn shape_and_parameter_problems_are_rejected_before_any_prediction() {
    let (data, targets) = regression_fixture();
    let model = forest_regressor(&data, &targets);
    let short = RegressionTargets::new(targets.as_slice()[..8].to_vec()).unwrap();
    assert_eq!(
        permutation_importance_regressor(
            &model,
            &data.as_view(),
            &short,
            RegressionScorer::R2,
            params(3, 0),
        ),
        Err(InspectionError::Scoring(ScoringError::TargetLength {
            rows: data.rows(),
            targets: 8,
        }))
    );
    assert_eq!(
        permutation_importance_regressor(
            &model,
            &data.as_view(),
            &targets,
            RegressionScorer::R2,
            params(0, 0),
        ),
        Err(InspectionError::InvalidRepeatCount)
    );

    let mut means = [0.0; 3];
    let mut std_devs = [0.0; 4];
    assert_eq!(
        permutation_importance_regressor_into(
            &model,
            &data.as_view(),
            &targets,
            RegressionScorer::R2,
            params(3, 0),
            &mut means,
            &mut std_devs,
        ),
        Err(InspectionError::OutputLength {
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(means, [0.0; 3], "outputs were written before validation");

    // A width mismatch surfaces as the estimator's own prediction error.
    let narrow = DenseMatrix::new(vec![1.0; 8], 8, 1).unwrap();
    let narrow_targets = RegressionTargets::new(vec![1.0; 8]).unwrap();
    assert_eq!(
        permutation_importance_regressor(
            &model,
            &narrow.as_view(),
            &narrow_targets,
            RegressionScorer::R2,
            params(3, 0),
        ),
        Err(InspectionError::Scoring(ScoringError::Prediction(
            ModelError::FeatureDimension {
                expected: 4,
                actual: 1,
            }
        )))
    );

    let mut means = [0.0; 4];
    let mut std_devs = [0.0; 4];
    let (classification_data, labels) = classification_fixture();
    let classifier = RandomForestClassifier::fit(
        &classification_data.as_view(),
        &labels,
        RandomForestClassifierParams::default().with_n_estimators(2),
    )
    .unwrap();
    assert_eq!(
        permutation_importance_classifier_into(
            ScorableClassifier::probabilistic(&classifier),
            &classification_data.as_view(),
            &BinaryTargets::new(vec![0, 1]).unwrap(),
            ClassificationScorer::Accuracy,
            params(2, 0),
            &mut means,
            &mut std_devs,
        ),
        Err(InspectionError::Scoring(ScoringError::TargetLength {
            rows: 64,
            targets: 2,
        }))
    );
}

/// A score FerricML does not enumerate, so importance cannot be reading a
/// private table of built-in scorers.
struct NegatedMeanSquaredError;

impl RegressionScore for NegatedMeanSquaredError {
    fn greater_is_better(&self) -> bool {
        true
    }

    fn score(&self, expected: &[f32], predicted: &[f32]) -> Result<f64, ScoringError> {
        mean_squared_error(expected, predicted)
            .map(|value| -value)
            .map_err(ScoringError::Metric)
    }
}

#[test]
fn a_caller_defined_score_is_inspected_through_the_same_contract() {
    let (data, targets) = regression_fixture();
    let model = forest_regressor(&data, &targets);
    let custom = permutation_importance_regressor(
        &model,
        &data.as_view(),
        &targets,
        NegatedMeanSquaredError,
        params(4, 3),
    )
    .unwrap();
    let built_in = permutation_importance_regressor(
        &model,
        &data.as_view(),
        &targets,
        RegressionScorer::MeanSquaredError,
        params(4, 3),
    )
    .unwrap();

    // Negating a minimized metric turns it into a maximized one; permutation
    // importance reports the same quality loss either way, because the score
    // declares its own orientation.
    assert_eq!(custom.means(), built_in.means());
    assert_eq!(custom.ranked(), built_in.ranked());
    assert!(custom.means()[0] > 0.0);
    assert_eq!(custom.means()[2], 0.0);
}
