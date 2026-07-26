//! Probability calibration against the rest of the crate.
//!
//! These are the cross-module obligations calibration owes: that Platt scaling
//! really is the one-dimensional logistic fit it claims to be, that a
//! calibrated model composes with the existing scorer and cross-validation
//! paths, and that a calibrator fitted on held-out rows is not the same object
//! as one fitted on the rows the model memorised.

use ferricml::calibration::{Calibrator, PlattCalibrator, PlattParams};
use ferricml::data::{BinaryTargets, DenseMatrix, SampleWeights};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};

/// A deliberately overlapping one-dimensional problem.
fn scores_and_labels() -> (Vec<f32>, BinaryTargets) {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    let mut state = 0x2545_f491_u32;
    for step in 0..120 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let score = (step as f32) / 40.0 - 1.5;
        let noise = (state >> 8) as f32 / (1u32 << 24) as f32;
        labels.push(u8::from(noise < 1.0 / (1.0 + (-1.7 * score).exp())));
        scores.push(score);
    }
    (scores, BinaryTargets::new(labels).unwrap())
}

/// Platt's prior-corrected targets for a label vector.
fn corrected_targets(targets: &BinaryTargets) -> Vec<f32> {
    let positives = targets
        .as_slice()
        .iter()
        .filter(|&&label| label == 1)
        .count();
    let negatives = targets.len() - positives;
    let high = (positives as f32 + 1.0) / (positives as f32 + 2.0);
    let low = 1.0 / (negatives as f32 + 2.0);
    targets
        .as_slice()
        .iter()
        .map(|&label| if label == 1 { high } else { low })
        .collect()
}

#[test]
fn platt_scaling_agrees_with_a_logistic_fit_of_the_same_one_dimensional_problem() {
    let (scores, labels) = scores_and_labels();
    let platt = PlattCalibrator::fit(
        &scores,
        &labels,
        PlattParams::default().with_tol(1.0e-9).with_max_iter(200),
    )
    .unwrap();

    // A fractional target `t` at score `s` contributes
    // `softplus(z) - t * z` to the objective, which is exactly what two
    // weighted rows at `s` contribute: one positive of weight `t` and one
    // negative of weight `1 - t`. So the identical fit is reachable through
    // the crate's own general logistic solver, on the same scores, and the two
    // must land on the same coefficients.
    let targets = corrected_targets(&labels);
    let mut expanded = Vec::with_capacity(scores.len() * 2);
    let mut expanded_labels = Vec::with_capacity(scores.len() * 2);
    let mut weights = Vec::with_capacity(scores.len() * 2);
    for (&score, &target) in scores.iter().zip(&targets) {
        expanded.push(score);
        expanded_labels.push(1);
        weights.push(target);
        expanded.push(score);
        expanded_labels.push(0);
        weights.push(1.0 - target);
    }
    let rows = expanded.len();
    let data = DenseMatrix::new(expanded, rows, 1).unwrap();
    let logistic = LogisticRegression::fit_weighted(
        &data.as_view(),
        &BinaryTargets::new(expanded_labels).unwrap(),
        &SampleWeights::new(weights).unwrap(),
        // A negligible penalty, so the fit is the unregularized maximum
        // likelihood Platt scaling targets.
        LogisticRegressionParams::default()
            .with_c(1.0e8)
            .with_tol(1.0e-9)
            .with_max_iter(200),
    )
    .unwrap();

    let (slope, intercept) = (platt.slope(), platt.intercept());
    assert!(
        (slope - logistic.coefficients()[0]).abs() <= 1.0e-3 * slope.abs().max(1.0),
        "slope {slope} against logistic coefficient {}",
        logistic.coefficients()[0]
    );
    assert!(
        (intercept - logistic.intercept()).abs() <= 1.0e-3 * intercept.abs().max(1.0),
        "intercept {intercept} against logistic intercept {}",
        logistic.intercept()
    );

    // And the maps themselves agree, which is what a caller observes.
    for &score in &scores {
        let calibrated = platt.calibrate(score);
        let reference = logistic.predict_positive_proba_one(&[score]).unwrap();
        assert!(
            (calibrated - reference).abs() <= 1.0e-4,
            "score {score}: {calibrated} against {reference}"
        );
    }
}

#[test]
fn the_prior_correction_is_what_separates_platt_from_a_raw_label_fit() {
    // The same comparison without the correction: fitting the raw labels gives
    // a measurably different, more confident map. This is what makes the
    // previous test a real check rather than one that would pass either way.
    let (scores, labels) = scores_and_labels();
    let platt = PlattCalibrator::fit(
        &scores,
        &labels,
        PlattParams::default().with_tol(1.0e-9).with_max_iter(200),
    )
    .unwrap();
    let rows = scores.len();
    let data = DenseMatrix::new(scores.clone(), rows, 1).unwrap();
    let raw = LogisticRegression::fit(
        &data.as_view(),
        &labels,
        LogisticRegressionParams::default()
            .with_c(1.0e8)
            .with_tol(1.0e-9)
            .with_max_iter(200),
    )
    .unwrap();
    assert!(
        (platt.slope() - raw.coefficients()[0]).abs() > 1.0e-3,
        "the corrected and uncorrected fits coincided at slope {}",
        platt.slope()
    );
    assert!(
        raw.coefficients()[0].abs() > platt.slope().abs(),
        "the uncorrected fit was not the more confident one"
    );
}

// ------------------------------------------------- calibrated classifiers

use ferricml::api::ProbabilisticClassifier;
use ferricml::calibration::{CalibratedClassifier, IsotonicRegression, IsotonicRegressionParams};
use ferricml::ensemble::{RandomForestClassifier, RandomForestClassifierParams};
use ferricml::metrics::{brier_score, log_loss, roc_auc_score};
use ferricml::model_selection::{
    ClassificationScorer, HoldoutParams, KFold, ScorableClassifier, Split, TestSize,
    cross_validate_classifier, score_classifier, stratified_train_test_split,
};
use ferricml::tree::MaxFeatures;

/// A noisy two-feature problem no model can fit perfectly.
///
/// The label noise is what makes calibration meaningful: a model that
/// memorises its training rows is *certain* about rows it cannot actually be
/// certain about, which is exactly the miscalibration to correct.
fn problem(rows: usize, seed: u32) -> (DenseMatrix, BinaryTargets) {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut values = Vec::with_capacity(rows * 2);
    let mut labels = Vec::with_capacity(rows);
    for _ in 0..rows {
        let (first, second) = (next(), next());
        let probability = 1.0 / (1.0 + (-3.0 * (first + second - 1.0)).exp());
        labels.push(u8::from(next() < probability));
        values.push(first);
        values.push(second);
    }
    (
        DenseMatrix::new(values, rows, 2).unwrap(),
        BinaryTargets::new(labels).unwrap(),
    )
}

/// One unpruned tree with no bootstrap: it memorises whatever it is fitted on.
fn memorising_forest(data: &DenseMatrix, labels: &BinaryTargets) -> RandomForestClassifier {
    RandomForestClassifier::fit(
        &data.as_view(),
        labels,
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All)
            .with_random_state(11),
    )
    .unwrap()
}

fn gather(data: &DenseMatrix, indices: &[usize]) -> DenseMatrix {
    data.select_rows(indices).unwrap()
}

#[test]
fn calibrating_on_the_training_fold_is_a_different_model_from_calibrating_on_held_out_rows() {
    let (train, train_labels) = problem(240, 0x1234_5678);
    let (holdout, holdout_labels) = problem(240, 0x2bad_c0de);
    let (evaluation, evaluation_labels) = problem(400, 0x0f0f_1234);
    let forest = memorising_forest(&train, &train_labels);

    // The forest reproduces its training labels exactly, so every training
    // score is 0 or 1 and there is nothing for a calibrator to learn from.
    let in_fold_scores = forest.predict_class_proba(&train.as_view(), 1).unwrap();
    assert!(
        in_fold_scores
            .iter()
            .all(|score| *score == 0.0 || *score == 1.0),
        "the fixture forest did not memorise its training fold"
    );

    let leaky = CalibratedClassifier::fit_isotonic(
        forest.clone(),
        &train.as_view(),
        &train_labels,
        IsotonicRegressionParams,
    )
    .unwrap();
    let honest = CalibratedClassifier::fit_isotonic(
        forest.clone(),
        &holdout.as_view(),
        &holdout_labels,
        IsotonicRegressionParams,
    )
    .unwrap();

    // The two fitted maps are not the same object, and they differ in the way
    // the leak predicts: the in-fold map is the identity on {0, 1}, while the
    // held-out map has genuinely interior values.
    assert_ne!(leaky.calibrator(), honest.calibrator());
    assert_eq!(leaky.calibrator().values(), &[0.0, 1.0]);
    assert!(
        honest
            .calibrator()
            .values()
            .iter()
            .all(|value| *value > 0.0 && *value < 1.0),
        "held-out calibration produced a saturated map: {:?}",
        honest.calibrator().values()
    );

    // And the leak is measurable on rows neither of them saw.
    let view = evaluation.as_view();
    let expected = evaluation_labels.as_slice();
    let leaky_probabilities = leaky.predict_class_proba(&view, 1).unwrap();
    let honest_probabilities = honest.predict_class_proba(&view, 1).unwrap();
    let raw_probabilities = forest.predict_class_proba(&view, 1).unwrap();
    let leaky_loss = log_loss(expected, &leaky_probabilities).unwrap();
    let honest_loss = log_loss(expected, &honest_probabilities).unwrap();
    let raw_loss = log_loss(expected, &raw_probabilities).unwrap();
    assert_eq!(
        leaky_probabilities, raw_probabilities,
        "calibrating on the training fold changed nothing, as it cannot"
    );
    assert!(
        honest_loss < leaky_loss,
        "held-out log loss {honest_loss} did not beat in-fold {leaky_loss}"
    );
    assert!(
        honest_loss < raw_loss,
        "held-out calibration did not improve on the raw model"
    );
    assert!(
        brier_score(expected, &honest_probabilities).unwrap()
            < brier_score(expected, &raw_probabilities).unwrap(),
        "held-out calibration did not improve the Brier score"
    );
}

#[test]
fn platt_calibration_also_separates_the_training_fold_from_a_held_out_one() {
    let (train, train_labels) = problem(240, 0x1234_5678);
    let (holdout, holdout_labels) = problem(240, 0x2bad_c0de);
    let (evaluation, evaluation_labels) = problem(400, 0x0f0f_1234);
    let forest = memorising_forest(&train, &train_labels);

    let leaky = CalibratedClassifier::fit_platt(
        forest.clone(),
        &train.as_view(),
        &train_labels,
        PlattParams::default(),
    )
    .unwrap();
    let honest = CalibratedClassifier::fit_platt(
        forest.clone(),
        &holdout.as_view(),
        &holdout_labels,
        PlattParams::default(),
    )
    .unwrap();
    assert_ne!(leaky.calibrator(), honest.calibrator());
    assert!(
        leaky.calibrator().slope() > honest.calibrator().slope(),
        "the in-fold fit was not the more confident one: {} against {}",
        leaky.calibrator().slope(),
        honest.calibrator().slope()
    );

    let view = evaluation.as_view();
    let expected = evaluation_labels.as_slice();
    let leaky_loss = log_loss(expected, &leaky.predict_class_proba(&view, 1).unwrap()).unwrap();
    let honest_loss = log_loss(expected, &honest.predict_class_proba(&view, 1).unwrap()).unwrap();
    let raw_loss = log_loss(expected, &forest.predict_class_proba(&view, 1).unwrap()).unwrap();
    assert!(
        honest_loss < leaky_loss,
        "{honest_loss} against {leaky_loss}"
    );
    assert!(honest_loss < raw_loss, "{honest_loss} against {raw_loss}");
}

#[test]
fn calibration_preserves_the_ranking_it_recalibrates() {
    let (train, train_labels) = problem(240, 0x1234_5678);
    let (holdout, holdout_labels) = problem(240, 0x2bad_c0de);
    let (evaluation, evaluation_labels) = problem(400, 0x0f0f_1234);
    let forest = memorising_forest(&train, &train_labels);
    let view = evaluation.as_view();
    let expected = evaluation_labels.as_slice();
    let raw = forest.predict_class_proba(&view, 1).unwrap();

    let platt = CalibratedClassifier::fit_platt(
        forest.clone(),
        &holdout.as_view(),
        &holdout_labels,
        PlattParams::default(),
    )
    .unwrap();
    let calibrated = platt.predict_class_proba(&view, 1).unwrap();

    // A *strictly increasing* map cannot reorder two rows, so every
    // threshold-sweeping score is unchanged. The slope's sign is the whole
    // condition, which is why it is asserted first rather than assumed:
    // `a_held_out_fold_that_disagrees_with_the_model_inverts_its_ranking`
    // holds the other branch.
    assert!(platt.calibrator().slope() > 0.0);
    for (left, right) in (0..raw.len()).zip(1..raw.len()) {
        assert_eq!(
            raw[left].partial_cmp(&raw[right]),
            calibrated[left].partial_cmp(&calibrated[right]),
            "rows {left} and {right} were reordered"
        );
    }
    let raw_auc = roc_auc_score(expected, &raw).unwrap();
    assert!((roc_auc_score(expected, &calibrated).unwrap() - raw_auc).abs() <= 1.0e-12);

    // Isotonic is monotone but not strict: it may merge two scores into one
    // value, so it can lose ordering information but never invert it.
    let isotonic = CalibratedClassifier::fit_isotonic(
        forest,
        &holdout.as_view(),
        &holdout_labels,
        IsotonicRegressionParams,
    )
    .unwrap();
    let stepped = isotonic.predict_class_proba(&view, 1).unwrap();
    for (left, right) in (0..raw.len()).zip(1..raw.len()) {
        if raw[left] < raw[right] {
            assert!(stepped[left] <= stepped[right], "rows {left}, {right}");
        }
    }
    assert!(roc_auc_score(expected, &stepped).unwrap() <= raw_auc + 1.0e-12);
}

/// The mean score over positive rows, minus the mean over negative rows.
fn class_mean_gap(scores: &[f32], targets: &BinaryTargets) -> f64 {
    let mut sums = [0.0_f64; 2];
    let mut counts = [0_usize; 2];
    for (&score, &label) in scores.iter().zip(targets.as_slice()) {
        sums[usize::from(label)] += f64::from(score);
        counts[usize::from(label)] += 1;
    }
    sums[1] / counts[1] as f64 - sums[0] / counts[0] as f64
}

#[test]
fn a_held_out_fold_that_disagrees_with_the_model_inverts_its_ranking() {
    // The other branch of `calibration_preserves_the_ranking_it_recalibrates`,
    // and the case the module's own documentation recommends: the calibration
    // rows are held out, so nothing stops them from disagreeing with the model.
    // Six rows are enough, and small folds are exactly where this happens.
    let (train, train_labels) = problem(240, 0x1234_5678);
    let (holdout, holdout_labels) = problem(6, 0xa);
    let (evaluation, evaluation_labels) = problem(400, 0x0f0f_1234);
    let forest = memorising_forest(&train, &train_labels);
    let view = evaluation.as_view();
    let expected = evaluation_labels.as_slice();
    let raw = forest.predict_class_proba(&view, 1).unwrap();
    let raw_auc = roc_auc_score(expected, &raw).unwrap();
    assert!(
        raw_auc > 0.5,
        "the fixture forest does not rank the evaluation rows: {raw_auc}"
    );

    // The mechanism, stated before its consequence: on this fold the forest's
    // positive rows carry a *lower* mean score than its negative rows. The
    // maximum-likelihood slope takes the sign of that gap and nothing else.
    let fold_scores = forest.predict_class_proba(&holdout.as_view(), 1).unwrap();
    let gap = class_mean_gap(&fold_scores, &holdout_labels);
    assert!(
        gap < 0.0,
        "the fold's class mean gap was not negative: {gap}"
    );

    let platt = CalibratedClassifier::fit_platt(
        forest.clone(),
        &holdout.as_view(),
        &holdout_labels,
        PlattParams::default(),
    )
    .unwrap();
    assert!(
        platt.calibrator().slope() < 0.0,
        "slope {} did not follow the fold's negative mean gap",
        platt.calibrator().slope()
    );

    // A strictly decreasing map is still monotone, and it reverses every
    // pairwise comparison rather than preserving it. Nothing errors: the
    // calibrated model is a working, silently anti-ranking classifier.
    let calibrated = platt.predict_class_proba(&view, 1).unwrap();
    for (left, right) in (0..raw.len()).zip(1..raw.len()) {
        assert_eq!(
            raw[left].partial_cmp(&raw[right]),
            calibrated[left]
                .partial_cmp(&calibrated[right])
                .map(std::cmp::Ordering::reverse),
            "rows {left} and {right} were not inverted"
        );
    }
    let calibrated_auc = roc_auc_score(expected, &calibrated).unwrap();
    assert!(
        (calibrated_auc - (1.0 - raw_auc)).abs() <= 1.0e-12,
        "AUC {calibrated_auc} is not the exact inverse of {raw_auc}"
    );

    // Isotonic on the same fold pools every observation into one block, so the
    // map is constant: it loses the ordering rather than inverting it, and the
    // calibrated AUC is exactly the coin-flip value.
    let isotonic = CalibratedClassifier::fit_isotonic(
        forest,
        &holdout.as_view(),
        &holdout_labels,
        IsotonicRegressionParams,
    )
    .unwrap();
    let values = isotonic.calibrator().values();
    assert!(
        values.windows(2).all(|pair| pair[0] == pair[1]),
        "the fold did not pool to a constant map: {values:?}"
    );
    let stepped = isotonic.predict_class_proba(&view, 1).unwrap();
    assert!(stepped.iter().all(|value| *value == stepped[0]));
    assert_eq!(roc_auc_score(expected, &stepped).unwrap(), 0.5);
}

#[test]
fn a_calibrated_model_scores_and_cross_validates_through_the_existing_paths() {
    let (data, labels) = problem(240, 0x51de_beef);
    let (holdout, holdout_labels) = problem(240, 0x2bad_c0de);
    let forest = memorising_forest(&data, &labels);
    let calibrated = CalibratedClassifier::fit_isotonic(
        forest,
        &holdout.as_view(),
        &holdout_labels,
        IsotonicRegressionParams,
    )
    .unwrap();

    // The scorer takes it as any other classifier, with no calibration-aware
    // branch anywhere in the scoring path.
    let probabilities = calibrated.predict_class_proba(&data.as_view(), 1).unwrap();
    assert_eq!(
        score_classifier(
            ScorableClassifier::probabilistic(&calibrated),
            &data.as_view(),
            &labels,
            ClassificationScorer::Brier
        ),
        Ok(brier_score(labels.as_slice(), &probabilities).unwrap())
    );

    // Cross-validation fits a calibrated model per fold, each one calibrated on
    // rows held out of its own training half, and every fold's score equals
    // scoring that fold directly. One evaluation implementation, not two.
    let splits: Vec<Split> = KFold::new(3).split(data.rows()).unwrap().collect();
    let result = cross_validate_classifier(
        &data.as_view(),
        &labels,
        splits.clone(),
        ClassificationScorer::LogLoss,
        fit_calibrated_fold,
        |model| ScorableClassifier::probabilistic(model),
    )
    .unwrap();
    assert_eq!(result.len(), 3);

    for (fold, split) in splits.iter().enumerate() {
        let train = gather(&data, split.train_indices());
        let train_labels = labels.select(split.train_indices()).unwrap();
        let model = fit_calibrated_fold(&train.as_view(), &train_labels).unwrap();
        let test = gather(&data, split.test_indices());
        let test_labels = labels.select(split.test_indices()).unwrap();
        assert_eq!(
            score_classifier(
                ScorableClassifier::probabilistic(&model),
                &test.as_view(),
                &test_labels,
                ClassificationScorer::LogLoss
            ),
            Ok(result.scores()[fold]),
            "fold {fold}"
        );
    }
}

/// Splits a training fold again, fits on one half and calibrates on the other.
fn fit_calibrated_fold(
    data: &ferricml::data::MatrixView<'_>,
    labels: &BinaryTargets,
) -> Result<
    CalibratedClassifier<RandomForestClassifier, IsotonicRegression>,
    ferricml::api::ModelError,
> {
    let inner = stratified_train_test_split(
        labels.as_slice(),
        HoldoutParams::default()
            .with_test_size(TestSize::Fraction(0.4))
            .with_random_state(3),
    )
    .expect("inner calibration split");
    let owned = DenseMatrix::new(
        data.iter_rows().flatten().copied().collect(),
        data.rows(),
        data.columns(),
    )
    .expect("fold matrix");
    let fit_rows = owned.select_rows(inner.train_indices()).expect("fit rows");
    let fit_labels = labels.select(inner.train_indices()).expect("fit labels");
    let calibration_rows = owned
        .select_rows(inner.test_indices())
        .expect("calibration rows");
    let calibration_labels = labels
        .select(inner.test_indices())
        .expect("calibration labels");
    let forest = memorising_forest(&fit_rows, &fit_labels);
    CalibratedClassifier::fit_isotonic(
        forest,
        &calibration_rows.as_view(),
        &calibration_labels,
        IsotonicRegressionParams,
    )
}
