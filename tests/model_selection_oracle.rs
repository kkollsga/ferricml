//! Cross-entry-point identity for grid search, against a hand-rolled loop.
//!
//! `grid_search_regressor` documents itself as a loop over cross-validation
//! rather than a second evaluation path, and the unit tests prove that against
//! `cross_validate_regressor` — which is the same implementation, so the
//! comparison cannot see a defect the two share. This binary compares it with a
//! loop that shares nothing: it selects fold rows itself, fits through the
//! public estimator entry point, predicts through the public batch contract,
//! and scores with metrics written here from their mathematical definitions.
//!
//! Sizes come from `FERRICML_ORACLE_SWEEP` (searches per arm) so the in-gate
//! run stays cheap and a recorded sweep can be many times larger:
//!
//! ```text
//! FERRICML_ORACLE_SWEEP=400 cargo test --release --test model_selection_oracle -- --nocapture
//! ```

use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams, Ridge, RidgeParams};
use ferricml::model_selection::{
    ClassificationScorer, KFold, ParameterGrid, RegressionScore, RegressionScorer,
    ScorableClassifier, Split, StratifiedKFold, grid_search_classifier, grid_search_regressor,
};

#[path = "support/rng.rs"]
mod rng;

use rng::TestRng;

/// Searches per arm when the environment does not ask for more.
const DEFAULT_SEARCHES: usize = 60;

fn searches() -> usize {
    std::env::var("FERRICML_ORACLE_SWEEP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SEARCHES)
}

// ---------------------------------------------------------------------------
// Metrics, written from their definitions rather than reached through the
// crate. These are the oracle; if one of them is wrong the sweep reports a
// disagreement, which is the outcome an oracle exists to produce.
// ---------------------------------------------------------------------------

/// `mean((predicted - expected)^2)`, accumulated in input order.
fn oracle_mean_squared_error(expected: &[f32], predicted: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for (&want, &got) in expected.iter().zip(predicted) {
        let error = f64::from(got) - f64::from(want);
        total += error * error;
    }
    total / expected.len() as f64
}

/// `mean(|predicted - expected|)`, accumulated in input order.
fn oracle_mean_absolute_error(expected: &[f32], predicted: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for (&want, &got) in expected.iter().zip(predicted) {
        total += (f64::from(got) - f64::from(want)).abs();
    }
    total / expected.len() as f64
}

/// `1 - residual sum of squares / total sum of squares`.
fn oracle_r2(expected: &[f32], predicted: &[f32]) -> f64 {
    let mut mean = 0.0_f64;
    for &want in expected {
        mean += f64::from(want);
    }
    mean /= expected.len() as f64;
    let mut residual = 0.0_f64;
    let mut total = 0.0_f64;
    for (&want, &got) in expected.iter().zip(predicted) {
        let error = f64::from(got) - f64::from(want);
        residual += error * error;
        let deviation = f64::from(want) - mean;
        total += deviation * deviation;
    }
    1.0 - residual / total
}

/// `matches / rows`.
fn oracle_accuracy(expected: &[u8], predicted: &[u8]) -> f64 {
    let matched = expected
        .iter()
        .zip(predicted)
        .filter(|&(&want, &got)| want == got)
        .count();
    matched as f64 / expected.len() as f64
}

/// The four regression scores this sweep uses, paired with their orientation.
#[derive(Clone, Copy, Debug)]
enum Oracle {
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    R2,
}

impl Oracle {
    fn greater_is_better(self) -> bool {
        matches!(self, Self::R2)
    }

    fn score(self, expected: &[f32], predicted: &[f32]) -> f64 {
        match self {
            Self::MeanSquaredError => oracle_mean_squared_error(expected, predicted),
            Self::MeanAbsoluteError => oracle_mean_absolute_error(expected, predicted),
            Self::RootMeanSquaredError => oracle_mean_squared_error(expected, predicted).sqrt(),
            Self::R2 => oracle_r2(expected, predicted),
        }
    }

    fn crate_scorer(self) -> RegressionScorer {
        match self {
            Self::MeanSquaredError => RegressionScorer::MeanSquaredError,
            Self::MeanAbsoluteError => RegressionScorer::MeanAbsoluteError,
            Self::RootMeanSquaredError => RegressionScorer::RootMeanSquaredError,
            Self::R2 => RegressionScorer::R2,
        }
    }
}

/// Fixed-order mean, which is what a fold summary is.
fn mean_in_order(scores: &[f64]) -> f64 {
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Strictly-better selection over candidate means, keeping the earliest on a
/// tie. Written here so the winner is decided by this file's own rule.
fn select_best(means: &[f64], greater_is_better: bool) -> usize {
    let mut best = 0;
    for (candidate, &mean) in means.iter().enumerate().skip(1) {
        let improved = if greater_is_better {
            mean > means[best]
        } else {
            mean < means[best]
        };
        if improved {
            best = candidate;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Randomized problem generation.
// ---------------------------------------------------------------------------

struct RegressionCase {
    data: DenseMatrix,
    targets: RegressionTargets,
    splits: Vec<Split>,
    alphas: Vec<f32>,
    oracle: Oracle,
    /// True when the candidate list deliberately repeats a value so two
    /// candidates tie exactly.
    forced_tie: bool,
}

fn regression_case(seed: u64) -> RegressionCase {
    let mut rng = TestRng::new(seed);
    let rows = rng.between(18, 48);
    let columns = rng.between(1, 4);
    let folds = rng.between(2, 5);

    let mut values = Vec::with_capacity(rows * columns);
    let mut weights = Vec::with_capacity(columns);
    for _ in 0..columns {
        weights.push(rng.range(-3.0, 3.0));
    }
    let mut targets = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut target = rng.range(-0.5, 0.5);
        for weight in &weights {
            let value = rng.range_f32(-2.0, 2.0);
            target += weight * f64::from(value);
            values.push(value);
        }
        targets.push(target as f32);
    }

    let candidates = rng.between(1, 5);
    let mut alphas = Vec::with_capacity(candidates);
    for _ in 0..candidates {
        alphas.push(10.0_f32.powf(rng.range_f32(-3.0, 3.0)));
    }
    // A third of the cases repeat a candidate so the tie rule is exercised on
    // means that are equal bit for bit rather than merely close.
    let forced_tie = alphas.len() > 1 && rng.below(3) == 0;
    if forced_tie {
        let source = rng.below(alphas.len());
        let target = rng.below(alphas.len());
        alphas[target] = alphas[source];
    }

    let oracle = [
        Oracle::MeanSquaredError,
        Oracle::MeanAbsoluteError,
        Oracle::RootMeanSquaredError,
        Oracle::R2,
    ][rng.below(4)];

    let splits = KFold::new(folds)
        .with_shuffle(rng.flag())
        .with_random_state(rng.next_u64())
        .split(rows)
        .expect("fold count is at most the row count")
        .collect::<Vec<_>>();

    RegressionCase {
        data: DenseMatrix::new(values, rows, columns).expect("generated shape"),
        targets: RegressionTargets::new(targets).expect("finite targets"),
        splits,
        alphas,
        oracle,
        forced_tie,
    }
}

/// Fits, predicts and scores every candidate on every fold without touching
/// `cross_validate_*` or `grid_search_*`.
fn hand_rolled_regression(case: &RegressionCase) -> Vec<Vec<f64>> {
    let mut per_candidate = Vec::with_capacity(case.alphas.len());
    for &alpha in &case.alphas {
        let mut folds = Vec::with_capacity(case.splits.len());
        for split in &case.splits {
            let train = case
                .data
                .select_rows(split.train_indices())
                .expect("validated split");
            let train_targets = case
                .targets
                .select(split.train_indices())
                .expect("validated split");
            let model = Ridge::fit(
                &train.as_view(),
                &train_targets,
                RidgeParams::default().with_alpha(alpha),
            )
            .expect("ridge fits a well-posed penalized system");
            let test = case
                .data
                .select_rows(split.test_indices())
                .expect("validated split");
            let test_targets = case
                .targets
                .select(split.test_indices())
                .expect("validated split");
            let predicted = model.predict(&test.as_view()).expect("batch prediction");
            folds.push(case.oracle.score(test_targets.as_slice(), &predicted));
        }
        per_candidate.push(folds);
    }
    per_candidate
}

/// Scores each candidate on the rows it was *trained* on. Used only as a
/// control: it must disagree with the real fold scores, or the comparison above
/// is not looking at the held-out fold at all.
fn hand_rolled_on_training_rows(case: &RegressionCase) -> Vec<Vec<f64>> {
    let mut per_candidate = Vec::with_capacity(case.alphas.len());
    for &alpha in &case.alphas {
        let mut folds = Vec::with_capacity(case.splits.len());
        for split in &case.splits {
            let train = case
                .data
                .select_rows(split.train_indices())
                .expect("validated split");
            let train_targets = case
                .targets
                .select(split.train_indices())
                .expect("validated split");
            let model = Ridge::fit(
                &train.as_view(),
                &train_targets,
                RidgeParams::default().with_alpha(alpha),
            )
            .expect("ridge fit");
            let predicted = model.predict(&train.as_view()).expect("batch prediction");
            folds.push(case.oracle.score(train_targets.as_slice(), &predicted));
        }
        per_candidate.push(folds);
    }
    per_candidate
}

/// Counts of one sweep, so the assertion and the printed record agree.
#[derive(Default)]
struct Tally {
    searches: usize,
    fold_scores: usize,
    worst_delta: f64,
    winner_disagreements: usize,
    forced_ties: usize,
    /// Searches whose winning mean was achieved by more than one candidate, so
    /// the tie rule — not the comparison — decided the answer.
    ties_at_the_winner: usize,
    control_orientation_flips: usize,
    control_training_row_disagreements: usize,
}

#[test]
fn grid_search_selects_what_a_hand_rolled_loop_selects() {
    let mut tally = Tally::default();
    for seed in 0..searches() as u64 {
        let case = regression_case(0x5eed_0001 ^ seed.wrapping_mul(0x9e37_79b9));
        let grid = ParameterGrid::from_candidates(
            case.alphas
                .iter()
                .map(|&alpha| RidgeParams::default().with_alpha(alpha))
                .collect(),
        );
        let searched = grid_search_regressor(
            &case.data.as_view(),
            &case.targets,
            case.splits.clone(),
            &grid,
            case.oracle.crate_scorer(),
            |train, train_targets, params| Ridge::fit(train, train_targets, params.clone()),
        )
        .expect("a well-posed search");

        let expected = hand_rolled_regression(&case);
        assert_eq!(searched.len(), expected.len(), "candidate count");
        let mut means = Vec::with_capacity(expected.len());
        for (candidate, folds) in expected.iter().enumerate() {
            let observed = searched.candidates()[candidate].folds().scores();
            assert_eq!(observed.len(), folds.len(), "fold count");
            for (fold, (&want, &got)) in folds.iter().zip(observed).enumerate() {
                let delta = (want - got).abs();
                tally.worst_delta = tally.worst_delta.max(delta);
                tally.fold_scores += 1;
                assert!(
                    delta == 0.0,
                    "seed {seed} candidate {candidate} fold {fold}: \
                     hand-rolled {want}, search {got}"
                );
            }
            means.push(mean_in_order(folds));
        }

        let greater_is_better = case.oracle.crate_scorer().greater_is_better();
        assert_eq!(greater_is_better, case.oracle.greater_is_better());
        let want_best = select_best(&means, greater_is_better);
        if want_best != searched.best_index() {
            tally.winner_disagreements += 1;
        }
        assert_eq!(
            want_best,
            searched.best_index(),
            "seed {seed}: winner disagreement over means {means:?}"
        );
        if case.forced_tie {
            tally.forced_ties += 1;
        }
        if means
            .iter()
            .filter(|&&mean| mean == means[want_best])
            .count()
            > 1
        {
            tally.ties_at_the_winner += 1;
            assert!(
                means[..want_best]
                    .iter()
                    .all(|&mean| mean != means[want_best]),
                "seed {seed}: a tie must go to the earliest candidate, got {want_best} \
                 over means {means:?}"
            );
        }

        // Control 1: the orientation actually decides the winner. Reversing it
        // has to move the answer whenever the candidates are distinguishable.
        if select_best(&means, !greater_is_better) != want_best {
            tally.control_orientation_flips += 1;
        }
        // Control 2: the fold scores are the held-out ones. Scoring the trained
        // rows instead has to produce different numbers.
        let training = hand_rolled_on_training_rows(&case);
        if training
            .iter()
            .zip(&expected)
            .any(|(left, right)| left.iter().zip(right).any(|(a, b)| a != b))
        {
            tally.control_training_row_disagreements += 1;
        }
        tally.searches += 1;
    }

    println!(
        "model_selection: {} searches, {} candidate-fold scores, worst |delta| = {:e}, \
         {} winner disagreements, {} forced-tie grids, {} exact ties at the winner",
        tally.searches,
        tally.fold_scores,
        tally.worst_delta,
        tally.winner_disagreements,
        tally.forced_ties,
        tally.ties_at_the_winner,
    );
    println!(
        "model_selection controls: orientation flips the winner in {} of {} searches, \
         training-row scoring disagrees in {} of {}",
        tally.control_orientation_flips,
        tally.searches,
        tally.control_training_row_disagreements,
        tally.searches,
    );

    assert!(tally.fold_scores >= tally.searches, "the sweep must run");
    assert!(
        tally.forced_ties > 0,
        "no grid in this sweep repeated a candidate, so the tie rule was never reached"
    );
    assert!(
        tally.ties_at_the_winner > 0,
        "no search in this sweep was decided by the tie rule"
    );
    // Non-vacuity: both controls must fire, or an equality above could hold for
    // reasons that have nothing to do with search being correct.
    assert!(
        tally.control_orientation_flips * 2 > tally.searches,
        "reversing the orientation moved the winner in only {} of {} searches",
        tally.control_orientation_flips,
        tally.searches,
    );
    assert_eq!(
        tally.control_training_row_disagreements, tally.searches,
        "scoring the training rows produced the held-out numbers, so the comparison \
         is not testing which rows were scored"
    );
}

#[test]
fn classifier_grid_search_selects_what_a_hand_rolled_loop_selects() {
    let mut searches_run = 0_usize;
    let mut fold_scores = 0_usize;
    let mut worst_delta = 0.0_f64;
    let mut flips = 0_usize;

    for seed in 0..(searches() / 2).max(8) as u64 {
        let mut rng = TestRng::new(0xc1a5_0002 ^ seed.wrapping_mul(0x9e37_79b9));
        let rows = rng.between(24, 48);
        let columns = rng.between(1, 3);
        let mut values = Vec::with_capacity(rows * columns);
        let mut labels = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut signal = 0.0_f64;
            for _ in 0..columns {
                let value = rng.range_f32(-2.0, 2.0);
                signal += f64::from(value);
                values.push(value);
            }
            // A label that is mostly but not perfectly explained by the rows,
            // and is guaranteed to contain both classes.
            let label = if row < 2 {
                row as u8
            } else {
                u8::from(signal + rng.range(-1.0, 1.0) > 0.0)
            };
            labels.push(label);
        }
        let data = DenseMatrix::new(values, rows, columns).expect("generated shape");
        let targets = BinaryTargets::new(labels).expect("two observed classes");
        let splits = StratifiedKFold::new(2)
            .with_shuffle(rng.flag())
            .with_random_state(rng.next_u64())
            .split(targets.as_slice())
            .expect("both classes present")
            .collect::<Vec<_>>();

        let cs = [rng.range_f32(0.05, 0.5), rng.range_f32(1.0, 20.0)];
        let grid = ParameterGrid::from_candidates(
            cs.iter()
                .map(|&c| {
                    LogisticRegressionParams::default()
                        .with_c(c)
                        .with_max_iter(200)
                })
                .collect(),
        );
        let searched = grid_search_classifier(
            &data.as_view(),
            &targets,
            splits.clone(),
            &grid,
            ClassificationScorer::Accuracy,
            |train, train_targets, params| {
                LogisticRegression::fit(train, train_targets, params.clone())
            },
            |model| ScorableClassifier::probabilistic(model),
        )
        .expect("a well-posed search");

        let mut means = Vec::with_capacity(cs.len());
        for (candidate, &c) in cs.iter().enumerate() {
            let mut folds = Vec::with_capacity(splits.len());
            for split in &splits {
                let train = data
                    .select_rows(split.train_indices())
                    .expect("validated split");
                let train_targets = targets
                    .select(split.train_indices())
                    .expect("validated split");
                let model = LogisticRegression::fit(
                    &train.as_view(),
                    &train_targets,
                    LogisticRegressionParams::default()
                        .with_c(c)
                        .with_max_iter(200),
                )
                .expect("logistic fit");
                let test = data
                    .select_rows(split.test_indices())
                    .expect("validated split");
                let test_targets = targets
                    .select(split.test_indices())
                    .expect("validated split");
                let predicted = model.predict(&test.as_view()).expect("batch prediction");
                folds.push(oracle_accuracy(test_targets.as_slice(), &predicted));
            }
            let observed = searched.candidates()[candidate].folds().scores();
            for (fold, (&want, &got)) in folds.iter().zip(observed).enumerate() {
                let delta = (want - got).abs();
                worst_delta = worst_delta.max(delta);
                fold_scores += 1;
                assert!(
                    delta == 0.0,
                    "seed {seed} candidate {candidate} fold {fold}: \
                     hand-rolled {want}, search {got}"
                );
            }
            means.push(mean_in_order(&folds));
        }
        assert_eq!(
            select_best(&means, true),
            searched.best_index(),
            "seed {seed}: winner disagreement over means {means:?}"
        );
        if select_best(&means, false) != select_best(&means, true) {
            flips += 1;
        }
        searches_run += 1;
    }

    println!(
        "model_selection classifier: {searches_run} searches, {fold_scores} candidate-fold \
         scores, worst |delta| = {worst_delta:e}, orientation flips the winner in {flips}"
    );
    assert!(fold_scores > 0, "the sweep must run");
    assert!(
        flips > 0,
        "no accuracy grid in this sweep had a distinguishable winner, so the selection \
         assertion could not have failed"
    );
}
