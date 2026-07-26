//! Post-hoc probability calibration around an already-fitted classifier.
//!
//! FerricML gates model quality on Brier score and log loss, which is a claim
//! that probability quality matters. This module is the remedy for a model that
//! ranks well but whose probabilities are miscalibrated — forests and boosted
//! ensembles in particular.
//!
//! # What a calibrator is
//!
//! A [`Calibrator`] is a fitted monotone map of one raw model score onto a
//! probability. Two are shipped: [`IsotonicRegression`], which is
//! non-parametric and assumes only monotonicity, and [`PlattCalibrator`], which
//! is a one-dimensional logistic fit through the crate's shared objective
//! contract. Both are deterministic functions of the calibration sample alone.
//!
//! # Boundaries
//!
//! - Calibration reaches a wrapped model only through the public
//!   [`ProbabilisticClassifier`](crate::api::ProbabilisticClassifier) contract,
//!   so it works for a model FerricML does not ship and names no estimator
//!   family. `scripts/check_source_layout.py` enforces that mechanically.
//! - The score being calibrated is the wrapped model's **positive-class
//!   probability**, which is the one score that contract requires. A classifier
//!   that declares no probability is outside the wrapper's bound entirely, so
//!   there is no case where it has to invent one. Calibrating a raw decision
//!   function instead is the classical formulation, but `decision_function` is
//!   an inherent method of one estimator rather than part of any classifier
//!   contract, so a generic wrapper cannot reach it.
//!   [`Capabilities::decision_function`] records which types have one, which is
//!   what a consumer selecting its behavior at compile time reads.
//!
//! [`Capabilities::decision_function`]: crate::api::Capabilities::decision_function
//! - The calibration sample is supplied by the caller and is never taken from
//!   the wrapped model's own training rows implicitly. A calibrator fitted on
//!   the rows its model was trained on measures the model's memory, not its
//!   probabilities, and FerricML makes that the caller's explicit choice.

mod classifier;
mod isotonic;
mod platt;

pub use classifier::CalibratedClassifier;
pub use isotonic::{IsotonicRegression, IsotonicRegressionParams};
pub use platt::{PlattCalibrator, PlattParams};

use crate::api::ModelError;
use crate::data::BinaryTargets;

/// A fitted monotone map of one model score onto a calibrated probability.
///
/// Implementations are stateless functions of their fitted parameters: the same
/// score always produces the same probability, with no interior mutability and
/// no dependence on call order. The map must be monotone in the score, which is
/// what preserves the wrapped model's ranking — calibration changes *how
/// confident* a prediction is, never *which way round* two rows are ordered.
///
/// The trait is open so a caller can supply a calibration family FerricML does
/// not ship. Such a calibrator composes with [`CalibratedClassifier`] for
/// prediction; the capability declaration on the composition is written per
/// shipped calibrator, because what a composition can do depends on which
/// calibrator it holds, so a caller-defined one carries no declaration.
pub trait Calibrator {
    /// Maps one raw model score onto a calibrated probability in `0.0..=1.0`.
    fn calibrate(&self, score: f32) -> f32;

    /// Maps a batch of scores in place.
    ///
    /// This is the allocation-free path prediction uses: the caller's buffer
    /// arrives holding the wrapped model's scores and leaves holding calibrated
    /// probabilities, with no second buffer in between.
    fn calibrate_in_place(&self, scores: &mut [f32]) {
        for score in scores.iter_mut() {
            *score = self.calibrate(*score);
        }
    }
}

impl<K: Calibrator + ?Sized> Calibrator for &K {
    fn calibrate(&self, score: f32) -> f32 {
        (**self).calibrate(score)
    }

    fn calibrate_in_place(&self, scores: &mut [f32]) {
        (**self).calibrate_in_place(scores);
    }
}

/// Validates a `(scores, labels)` calibration sample before any fitting work.
///
/// Stated once here so both calibrators reject the same undefined cases in the
/// same order: empty, mismatched, non-finite, single-class.
fn validate_calibration_sample(scores: &[f32], targets: &BinaryTargets) -> Result<(), ModelError> {
    if scores.is_empty() {
        return Err(ModelError::EmptyData);
    }
    if scores.len() != targets.len() {
        return Err(ModelError::TargetLength {
            rows: scores.len(),
            targets: targets.len(),
        });
    }
    if let Some(row) = scores.iter().position(|score| !score.is_finite()) {
        return Err(ModelError::NonFiniteFeature { row, column: 0 });
    }
    let positives = targets
        .as_slice()
        .iter()
        .filter(|&&label| label == 1)
        .count();
    if positives == 0 || positives == targets.len() {
        return Err(ModelError::RequiresTwoClasses);
    }
    Ok(())
}
