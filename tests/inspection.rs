use ferricml::api::{AnyRegressor, Classifier, Estimator, ModelError, Regressor};
use ferricml::data::{
    BinaryTargets, ClassTargets, ClassificationTargets, DenseMatrix, MatrixView, RegressionTargets,
};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor, RandomForestRegressorParams,
};
use ferricml::inspection::{
    InspectionError, PermutationImportance, PermutationImportanceParams,
    permutation_importance_classifier, permutation_importance_classifier_into,
    permutation_importance_regressor, permutation_importance_regressor_into,
};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams, Ridge, RidgeParams};
use ferricml::metrics::mean_squared_error;
use ferricml::model_selection::{
    ClassificationScorer, RegressionScore, RegressionScorer, ScorableClassifier, ScoringError,
};
use ferricml::tree::MaxFeatures;

#[path = "support/rng.rs"]
mod rng;

use rng::TestRng;

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

/// The same four columns, labelled into three neither contiguous nor
/// zero-based classes. The label is decided by the dominant column, so the
/// weak column carries only the little information the tertile boundaries
/// leave it and the last two columns carry none.
fn multiclass_fixture() -> (DenseMatrix, ClassTargets) {
    let (data, regression) = regression_fixture();
    let labels = regression
        .as_slice()
        .iter()
        .map(|&value| match value {
            value if value < -2.7 => 3,
            value if value < 2.7 => 7,
            _ => 10,
        })
        .collect();
    (data, ClassTargets::new(labels).unwrap())
}

/// Permutation importance written once, over any target vocabulary.
///
/// A caller outside the crate can write this because the bound is public and
/// there is one entry point under it; before the widening it needed one copy
/// per target type.
fn classifier_importance<T: ClassificationTargets>(
    model: &RandomForestClassifier,
    data: &DenseMatrix,
    targets: &T,
    scorer: ClassificationScorer,
    seed: u64,
) -> PermutationImportance {
    permutation_importance_classifier(
        ScorableClassifier::probabilistic(model),
        &data.as_view(),
        targets,
        scorer,
        params(6, seed),
    )
    .unwrap()
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
fn a_multiclass_vocabulary_reaches_the_same_permutation_entry_point() {
    let (data, targets) = multiclass_fixture();
    assert_eq!(targets.classes(), &[3, 7, 10]);
    let model = RandomForestClassifier::fit_multiclass(
        &data.as_view(),
        &targets,
        RandomForestClassifierParams::default()
            .with_n_estimators(12)
            .with_max_features(MaxFeatures::All)
            .with_random_state(9),
    )
    .unwrap();
    assert_eq!(model.classes(), &[3, 7, 10]);

    for scorer in [
        ClassificationScorer::Accuracy,
        ClassificationScorer::MulticlassLogLoss,
        ClassificationScorer::MulticlassBrier,
    ] {
        let importance = classifier_importance(&model, &data, &targets, scorer, 13);
        assert_eq!(importance.n_features(), 4);
        assert_eq!(
            importance.ranked()[0],
            0,
            "{scorer:?} ranked {:?}",
            importance.means()
        );
        assert!(importance.means()[0] > 0.0, "{scorer:?} orientation");
        // A constant column, and a duplicate of it, cannot be permuted into a
        // different matrix, so their importance is exactly zero.
        assert_eq!(importance.means()[2], 0.0, "{scorer:?} constant feature");
        assert_eq!(importance.means()[3], 0.0, "{scorer:?} duplicated feature");

        // The caller-owned form widened with it and is the same measurement.
        let mut means = vec![0.0; data.columns()];
        let mut std_devs = vec![0.0; data.columns()];
        permutation_importance_classifier_into(
            ScorableClassifier::probabilistic(&model),
            &data.as_view(),
            &targets,
            scorer,
            params(6, 13),
            &mut means,
            &mut std_devs,
        )
        .unwrap();
        assert_eq!(means, importance.means(), "{scorer:?}");
        assert_eq!(std_devs, importance.std_devs(), "{scorer:?}");
    }

    // Non-vacuity: a positive importance is a claim about this model, not a
    // property of the machinery. A model that observed one class predicts the
    // same distribution whatever the columns say, so the identical call
    // reports exactly zero for every column — including the one asserted
    // positive above.
    let single_labels = ClassTargets::new(vec![7; data.rows()]).unwrap();
    let single = RandomForestClassifier::fit_multiclass(
        &data.as_view(),
        &single_labels,
        RandomForestClassifierParams::default().with_n_estimators(2),
    )
    .unwrap();
    assert_eq!(single.classes(), &[7]);
    let flat = classifier_importance(
        &single,
        &data,
        &single_labels,
        ClassificationScorer::MulticlassLogLoss,
        13,
    );
    assert!(
        flat.means().iter().all(|&value| value == 0.0),
        "{:?}",
        flat.means()
    );

    // Widening the vocabulary bought no leniency: a binary
    // positive-probability metric over three classes is still refused rather
    // than reading one column as "the positive one".
    assert_eq!(
        permutation_importance_classifier(
            ScorableClassifier::probabilistic(&model),
            &data.as_view(),
            &targets,
            ClassificationScorer::Brier,
            params(3, 13),
        ),
        Err(InspectionError::Scoring(ScoringError::UnsupportedClasses)),
    );
}

#[test]
fn widening_binary_targets_to_a_class_set_measures_the_same_thing() {
    let (data, binary) = classification_fixture();
    let model = RandomForestClassifier::fit(
        &data.as_view(),
        &binary,
        RandomForestClassifierParams::default()
            .with_n_estimators(12)
            .with_max_features(MaxFeatures::All)
            .with_random_state(9),
    )
    .unwrap();
    let widened = ClassTargets::from(binary.clone());
    assert_eq!(widened.classes(), &[0, 1]);

    for scorer in [ClassificationScorer::Accuracy, ClassificationScorer::Brier] {
        let from_binary = classifier_importance(&model, &data, &binary, scorer, 13);
        let from_classes = classifier_importance(&model, &data, &widened, scorer, 13);
        assert_eq!(from_binary, from_classes, "{scorer:?}");
        // Non-vacuity: the two agree because the labels and the permutation
        // stream are the same, not because every run of this returns the same
        // numbers — one seed away, they differ.
        assert_ne!(
            from_binary.means(),
            classifier_importance(&model, &data, &binary, scorer, 14).means(),
            "{scorer:?}"
        );
    }
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

// ---------------------------------------------------------------------------
// Permutation importance against an oracle.
//
// Every test above fits a real estimator and then asserts an ordering, which
// can only ever say "this looks right". A model whose dependence on each column
// is a *construction* rather than a fit turns those into exact statements: a
// column the model cannot read must score exactly zero, and — with the targets
// set to the model's own predictions, so the unpermuted score is exactly
// perfect — a column it does read must score strictly positive.
// ---------------------------------------------------------------------------

/// A regressor that reads exactly the columns it is told to.
struct SubsetLinear {
    n_features_in: usize,
    used: Vec<usize>,
    weights: Vec<f64>,
}

impl SubsetLinear {
    fn value(&self, row: &[f32]) -> f32 {
        let mut total = 0.0_f64;
        for (&column, &weight) in self.used.iter().zip(&self.weights) {
            total += weight * f64::from(row[column]);
        }
        total as f32
    }
}

impl Estimator for SubsetLinear {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Regressor for SubsetLinear {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        if data.columns() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: data.columns(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        for (row, slot) in data.iter_rows().zip(output) {
            *slot = self.value(row);
        }
        Ok(())
    }
}

/// The same construction as a label-only classifier.
struct SubsetThreshold {
    inner: SubsetLinear,
    classes: [u8; 2],
}

impl Estimator for SubsetThreshold {
    fn n_features_in(&self) -> usize {
        self.inner.n_features_in()
    }
}

impl Classifier for SubsetThreshold {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        if data.columns() != self.n_features_in() {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in(),
                actual: data.columns(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        for (row, slot) in data.iter_rows().zip(output) {
            *slot = u8::from(self.inner.value(row) > 0.0);
        }
        Ok(())
    }
}

/// One randomized inspection problem.
struct SubsetCase {
    data: DenseMatrix,
    model: SubsetLinear,
    /// Whether each column can change a prediction at all: read by the model,
    /// carrying a non-zero weight, and not constant down the batch.
    effective: Vec<bool>,
    /// The column held constant, when the case has one.
    constant: Option<usize>,
    params: PermutationImportanceParams,
}

fn subset_case(seed: u64) -> SubsetCase {
    let mut rng = TestRng::new(seed);
    let rows = rng.between(24, 48);
    let columns = rng.between(2, 6);

    // A non-empty proper subset, so every case has both a read column and an
    // ignored one.
    let mut used = Vec::new();
    while used.is_empty() || used.len() == columns {
        used = (0..columns).filter(|_| rng.flag()).collect();
    }
    let weights = used
        .iter()
        .map(|_| rng.range(-4.0, 4.0))
        .collect::<Vec<_>>();
    // Sometimes hold one column constant, which cannot matter even when read.
    let constant = if rng.below(3) == 0 {
        Some(rng.below(columns))
    } else {
        None
    };

    let mut values = Vec::with_capacity(rows * columns);
    for _ in 0..rows {
        for column in 0..columns {
            values.push(if constant == Some(column) {
                0.75
            } else {
                rng.range_f32(-3.0, 3.0)
            });
        }
    }

    let effective = (0..columns)
        .map(|column| {
            constant != Some(column)
                && used
                    .iter()
                    .zip(&weights)
                    .any(|(&index, &weight)| index == column && weight != 0.0)
        })
        .collect();

    SubsetCase {
        data: DenseMatrix::new(values, rows, columns).expect("generated shape"),
        model: SubsetLinear {
            n_features_in: columns,
            used,
            weights,
        },
        effective,
        constant,
        params: PermutationImportanceParams::default()
            .with_n_repeats(rng.between(1, 6))
            .with_random_state(rng.next_u64()),
    }
}

#[test]
fn a_column_the_model_cannot_read_scores_exactly_zero_and_one_it_reads_scores_positive() {
    let mut checked_zero = 0_usize;
    let mut checked_positive = 0_usize;
    let mut constant_columns = 0_usize;
    let mut smallest_positive = f64::INFINITY;
    let mut worst_nonzero_on_an_ignored_column = 0.0_f64;
    let mut into_mismatches = 0_usize;
    let mut cases = 0_usize;

    for seed in 0..96_u64 {
        let case = subset_case(0x1a5b_0003_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        // The targets are the model's own predictions, so the unpermuted score
        // is exactly zero error and every reported loss is the permuted score
        // itself rather than a difference of two approximations.
        let predictions = case
            .model
            .predict(&case.data.as_view())
            .expect("batch prediction");
        let targets = RegressionTargets::new(predictions).expect("finite predictions");

        let importance = permutation_importance_regressor(
            &case.model,
            &case.data.as_view(),
            &targets,
            RegressionScorer::MeanSquaredError,
            case.params,
        )
        .expect("inspection");

        let columns = case.data.columns();
        let mut means = vec![0.0; columns];
        let mut std_devs = vec![0.0; columns];
        permutation_importance_regressor_into(
            &case.model,
            &case.data.as_view(),
            &targets,
            RegressionScorer::MeanSquaredError,
            case.params,
            &mut means,
            &mut std_devs,
        )
        .expect("inspection");
        if means
            .iter()
            .zip(importance.means())
            .any(|(left, right)| left.to_bits() != right.to_bits())
            || std_devs
                .iter()
                .zip(importance.std_devs())
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            into_mismatches += 1;
        }

        for column in 0..columns {
            let mean = importance.means()[column];
            if case.effective[column] {
                checked_positive += 1;
                smallest_positive = smallest_positive.min(mean);
                assert!(
                    mean > 0.0,
                    "seed {seed} column {column} is read by the model but scored {mean}"
                );
            } else {
                checked_zero += 1;
                worst_nonzero_on_an_ignored_column =
                    worst_nonzero_on_an_ignored_column.max(mean.abs());
                assert_eq!(
                    mean, 0.0,
                    "seed {seed} column {column} cannot change a prediction but scored {mean}"
                );
                assert_eq!(
                    importance.std_devs()[column],
                    0.0,
                    "seed {seed} column {column} scored a spread over identical repeats"
                );
            }
        }
        if case.constant.is_some() {
            constant_columns += 1;
        }

        // `ranked` must be the means in descending order with ties by index.
        let ranked = importance.ranked();
        for pair in ranked.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            assert!(
                importance.means()[left] > importance.means()[right]
                    || (importance.means()[left] == importance.means()[right] && left < right),
                "seed {seed}: ranked order {ranked:?} disagrees with {:?}",
                importance.means()
            );
        }
        cases += 1;
    }

    println!(
        "inspection: {cases} constructed models, {checked_zero} unreadable columns all \
         exactly zero (worst |mean| = {worst_nonzero_on_an_ignored_column:e}), \
         {checked_positive} readable columns all positive (smallest = {smallest_positive:e})"
    );
    println!(
        "inspection: allocating and `_into` forms disagreed in {into_mismatches} of {cases} \
         cases; {constant_columns} cases held a column constant"
    );

    assert_eq!(into_mismatches, 0, "the two forms must agree bit for bit");
    assert!(
        checked_zero > 0 && checked_positive > 0,
        "both arms must run"
    );
    // Non-vacuity: "exactly zero" must not be what this reports by default.
    // Every readable column scored above zero, and the smallest of them is the
    // margin between the two claims.
    assert!(
        smallest_positive > 0.0,
        "no readable column produced a positive score"
    );
}

#[test]
fn the_classifier_entry_point_answers_the_same_construction() {
    let mut checked_zero = 0_usize;
    let mut checked_positive = 0_usize;
    let mut readable_but_unmoved = 0_usize;
    let mut into_mismatches = 0_usize;
    let mut cases = 0_usize;

    for seed in 0..64_u64 {
        let case = subset_case(0x1a5c_0004_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        let model = SubsetThreshold {
            inner: case.model,
            classes: [0, 1],
        };
        let labels = model
            .predict(&case.data.as_view())
            .expect("batch prediction");
        if labels.iter().all(|&label| label == labels[0]) {
            // One observed class is not a two-class problem; skip rather than
            // pretend the case was informative.
            continue;
        }
        let targets = BinaryTargets::new(labels).expect("two observed classes");

        let importance = permutation_importance_classifier(
            ScorableClassifier::labels_only(&model),
            &case.data.as_view(),
            &targets,
            ClassificationScorer::Accuracy,
            case.params,
        )
        .expect("inspection");

        let columns = case.data.columns();
        let mut means = vec![0.0; columns];
        let mut std_devs = vec![0.0; columns];
        permutation_importance_classifier_into(
            ScorableClassifier::labels_only(&model),
            &case.data.as_view(),
            &targets,
            ClassificationScorer::Accuracy,
            case.params,
            &mut means,
            &mut std_devs,
        )
        .expect("inspection");
        if means
            .iter()
            .zip(importance.means())
            .any(|(left, right)| left.to_bits() != right.to_bits())
            || std_devs
                .iter()
                .zip(importance.std_devs())
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            into_mismatches += 1;
        }

        for column in 0..columns {
            let mean = importance.means()[column];
            if case.effective[column] {
                if mean > 0.0 {
                    checked_positive += 1;
                } else {
                    readable_but_unmoved += 1;
                }
            } else {
                checked_zero += 1;
                assert_eq!(
                    mean, 0.0,
                    "seed {seed} column {column} cannot change a label but scored {mean}"
                );
            }
        }
        cases += 1;
    }

    println!(
        "inspection classifier: {cases} constructed models, {checked_zero} unreadable columns \
         all exactly zero, {checked_positive} readable columns scored positive and \
         {readable_but_unmoved} left the labels unchanged, {into_mismatches} `_into` \
         disagreements"
    );
    assert_eq!(into_mismatches, 0, "the two forms must agree bit for bit");
    assert!(checked_zero > 0, "no unreadable column was checked");
    assert!(
        checked_positive > 0,
        "no readable column degraded accuracy, so the zero claim is not discriminating"
    );
}
