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
        let reference = logistic.predict_positive_proba(&[score]).unwrap();
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
