//! Platt scaling: a one-dimensional logistic fit of model scores onto labels.

use crate::api::ModelError;
use crate::data::BinaryTargets;
use crate::loss::{BinaryLogLoss, accumulate_newton_row, raw_score};
use crate::numeric::sigmoid_f32;

use super::Calibrator;

/// Fit parameters for [`PlattCalibrator`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlattParams {
    max_iter: usize,
    tol: f32,
}

impl Default for PlattParams {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1.0e-6,
        }
    }
}

impl PlattParams {
    /// Sets the maximum number of Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance on the largest parameter update.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns the maximum number of Newton iterations.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    pub const fn tol(&self) -> f32 {
        self.tol
    }
}

/// A fitted logistic map of one model score onto a calibrated probability.
///
/// The map is `sigmoid(slope * score + intercept)`, and the two parameters are
/// the maximum-likelihood fit of exactly the binary log-loss objective every
/// other FerricML solver minimizes — this consumes the crate's shared objective
/// contract rather than carrying a third logistic solver.
///
/// # Prior-corrected targets
///
/// The fit does **not** regress on the raw `0`/`1` labels. It regresses on
/// Platt's prior-corrected targets: a positive row targets
/// `(n_pos + 1) / (n_pos + 2)` and a negative row targets `1 / (n_neg + 2)`.
/// That is not a smoothing convenience. With raw labels a perfectly separating
/// score has no finite maximum-likelihood fit at all — the slope runs to
/// infinity and the calibrated probabilities collapse to `0` and `1`, which is
/// precisely the overconfidence calibration exists to remove. Targets strictly
/// inside `(0, 1)` give the objective a finite minimizer for every input,
/// which is why no regularization term is needed here.
///
/// The per-sample objective accepts a fractional target unchanged, because it
/// is written in raw-score space as `softplus(raw) - target * raw`; nothing
/// about it assumed an integral label.
///
/// ```
/// use ferricml::calibration::{Calibrator, PlattCalibrator, PlattParams};
/// use ferricml::data::BinaryTargets;
///
/// // Scores that separate the classes perfectly. With raw labels this
/// // problem has no finite maximum-likelihood solution; Platt's
/// // prior-corrected targets are what keep the fit finite.
/// let scores = [-3.0_f32, -2.0, -1.0, 1.0, 2.0, 3.0];
/// let labels = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;
///
/// let calibrator = PlattCalibrator::fit(&scores, &labels, PlattParams::default())?;
///
/// // A probability, never the exact certainty calibration exists to remove.
/// let high = calibrator.calibrate(3.0);
/// assert!(high > 0.5 && high < 1.0);
/// assert!(calibrator.calibrate(-3.0) < 0.5);
///
/// // This sample fitted a positive slope, so the map is strictly increasing
/// // and cannot reorder two rows. The sign is the condition, and it belongs
/// // to the calibration sample: see `slope` below.
/// assert!(calibrator.slope() > 0.0);
/// assert!(calibrator.calibrate(1.0) < calibrator.calibrate(2.0));
///
/// // A sample whose positive rows score below its negative rows fits the
/// // mirror image, and that map reverses every pairwise comparison.
/// let mirrored = PlattCalibrator::fit(
///     &[3.0_f32, 2.0, 1.0, -1.0, -2.0, -3.0],
///     &labels,
///     PlattParams::default(),
/// )?;
/// assert!(mirrored.slope() < 0.0);
/// assert!(mirrored.calibrate(1.0) > mirrored.calibrate(2.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PlattCalibrator {
    slope: f32,
    intercept: f32,
    params: PlattParams,
    iterations: usize,
}

impl PlattCalibrator {
    /// Fits the two-parameter logistic map from scores onto observed labels.
    ///
    /// Both labels must be observed. A single-class calibration sample gives
    /// the objective no slope to identify and is
    /// [`ModelError::RequiresTwoClasses`].
    pub fn fit(
        scores: &[f32],
        targets: &BinaryTargets,
        params: PlattParams,
    ) -> Result<Self, ModelError> {
        validate_params(&params)?;
        super::validate_calibration_sample(scores, targets)?;

        let positives = targets
            .as_slice()
            .iter()
            .filter(|&&label| label == 1)
            .count();
        let negatives = targets.len() - positives;
        let high = (positives as f64 + 1.0) / (positives as f64 + 2.0);
        let low = 1.0 / (negatives as f64 + 2.0);

        // `theta` is [slope, intercept] and each design row is [score, 1.0], so
        // the intercept is parameter index 1 exactly as the linear seam
        // expects.
        let mut theta = [0.0_f64; PARAMETERS];
        let mut gradient = [0.0_f64; PARAMETERS];
        let mut hessian = [0.0_f64; PARAMETERS * PARAMETERS];
        let mut iterations = 0;
        for iteration in 0..params.max_iter {
            gradient.fill(0.0);
            hessian.fill(0.0);
            for (&score, &label) in scores.iter().zip(targets.as_slice()) {
                let design_row = [f64::from(score), 1.0];
                let target = if label == 1 { high } else { low };
                let raw = raw_score(&theta, &design_row, 1, Some(1));
                accumulate_newton_row::<BinaryLogLoss>(
                    &design_row,
                    raw,
                    target,
                    1.0,
                    &mut gradient,
                    &mut hessian,
                );
            }
            let update = solve_symmetric_2x2(&hessian, &gradient)?;
            let max_update = update
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            for (value, step) in theta.iter_mut().zip(update) {
                *value -= step;
            }
            iterations = iteration + 1;
            if max_update <= f64::from(params.tol) {
                break;
            }
        }

        let slope = theta[0] as f32;
        let intercept = theta[1] as f32;
        if !slope.is_finite() || !intercept.is_finite() {
            return Err(ModelError::LinearSolveFailed);
        }
        Ok(Self {
            slope,
            intercept,
            params,
            iterations,
        })
    }

    /// Returns the fitted slope on the model score.
    ///
    /// **This sign is the ranking contract.** The map is monotone either way,
    /// and monotone in the decreasing direction reverses every pairwise
    /// comparison:
    ///
    /// - `slope > 0`: strictly increasing. No two rows are reordered, so a
    ///   threshold-sweeping score such as ROC AUC is unchanged.
    /// - `slope < 0`: strictly decreasing. Every pair is inverted, and a model
    ///   with ROC AUC `auc` becomes one with `1.0 - auc`.
    /// - `slope == 0`: constant. All ordering is gone and ROC AUC is `0.5`.
    ///
    /// The sign is not the wrapped model's overall quality; it is a property of
    /// the calibration sample, and it is the sign of that sample's class mean
    /// gap — the mean score over its positive rows minus the mean over its
    /// negative rows. Because the objective is strictly convex, the fit's slope
    /// takes the sign of that gap and nothing else, so a small held-out fold
    /// whose few positive rows happen to score low fits a negative slope even
    /// for a model that ranks well everywhere else. Note that this is a
    /// *mean* comparison, not a rank one: a calibration sample can have ROC AUC
    /// above `0.5` and still fit a negative slope, if one high-scoring negative
    /// row outweighs the ranking. A perfectly separated sample cannot, because
    /// separation forces the mean gap positive.
    ///
    /// A negative slope is therefore the correct maximum-likelihood answer for
    /// the sample it was given, not a solver failure, and it is reported rather
    /// than rejected. Callers whose downstream use depends on ranking should
    /// check the sign; a negative one usually means the calibration fold is too
    /// small or unrepresentative.
    pub const fn slope(&self) -> f32 {
        self.slope
    }

    /// Returns the fitted intercept.
    pub const fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns the exact fit parameters.
    pub const fn get_params(&self) -> &PlattParams {
        &self.params
    }

    /// Returns the number of Newton iterations performed.
    pub const fn n_iter(&self) -> usize {
        self.iterations
    }

    /// Returns the raw calibrated score whose sigmoid is the probability.
    ///
    /// This is the calibrated model's decision function: a real-valued score,
    /// monotone in the input score, that a threshold-sweeping consumer can read
    /// without the sigmoid's saturation flattening the extremes.
    pub fn decision_score(&self, score: f32) -> f32 {
        self.slope.mul_add(score, self.intercept)
    }
}

/// Slope and intercept: the whole parameter vector of a Platt fit.
const PARAMETERS: usize = 2;

fn validate_params(params: &PlattParams) -> Result<(), ModelError> {
    if params.max_iter == 0 {
        return Err(ModelError::InvalidIterationCount);
    }
    if !params.tol.is_finite() || params.tol <= 0.0 {
        return Err(ModelError::InvalidTolerance);
    }
    Ok(())
}

/// Solves the symmetric two-parameter Newton system by its closed form.
///
/// Only the lower triangle is written by the shared accumulator, so the
/// off-diagonal is read once from index `2`. A non-positive or non-finite
/// determinant means the curvature collapsed and the step is undefined, which
/// is reported rather than turned into an arbitrary direction.
fn solve_symmetric_2x2(hessian: &[f64], gradient: &[f64]) -> Result<[f64; 2], ModelError> {
    let (a, b, d) = (hessian[0], hessian[2], hessian[3]);
    let determinant = a * d - b * b;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(ModelError::LinearSolveFailed);
    }
    let update = [
        (d * gradient[0] - b * gradient[1]) / determinant,
        (a * gradient[1] - b * gradient[0]) / determinant,
    ];
    if update.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::LinearSolveFailed);
    }
    Ok(update)
}

impl Calibrator for PlattCalibrator {
    fn calibrate(&self, score: f32) -> f32 {
        sigmoid_f32(self.decision_score(score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::Objective;

    fn sample() -> (Vec<f32>, BinaryTargets) {
        // Deliberately overlapping, so the objective has an interior optimum
        // that a finite-difference check can confirm.
        let scores = vec![
            0.05_f32, 0.12, 0.18, 0.25, 0.31, 0.44, 0.52, 0.58, 0.63, 0.71, 0.77, 0.84, 0.90, 0.95,
        ];
        let labels = vec![0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1];
        (scores, BinaryTargets::new(labels).unwrap())
    }

    /// The exact objective the fit claims to minimize.
    fn objective(scores: &[f32], targets: &[f64], slope: f64, intercept: f64) -> f64 {
        scores
            .iter()
            .zip(targets)
            .map(|(&score, &target)| {
                BinaryLogLoss::value(slope * f64::from(score) + intercept, target)
            })
            .sum()
    }

    fn corrected_targets(targets: &BinaryTargets) -> Vec<f64> {
        let positives = targets
            .as_slice()
            .iter()
            .filter(|&&label| label == 1)
            .count();
        let negatives = targets.len() - positives;
        let high = (positives as f64 + 1.0) / (positives as f64 + 2.0);
        let low = 1.0 / (negatives as f64 + 2.0);
        targets
            .as_slice()
            .iter()
            .map(|&label| if label == 1 { high } else { low })
            .collect()
    }

    #[test]
    fn the_fit_is_a_stationary_point_of_the_objective_it_claims_to_minimize() {
        let (scores, targets) = sample();
        let fitted = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
        let corrected = corrected_targets(&targets);
        let (slope, intercept) = (f64::from(fitted.slope()), f64::from(fitted.intercept()));
        let step = 1.0e-5;
        let base = objective(&scores, &corrected, slope, intercept);
        let d_slope = (objective(&scores, &corrected, slope + step, intercept)
            - objective(&scores, &corrected, slope - step, intercept))
            / (2.0 * step);
        let d_intercept = (objective(&scores, &corrected, slope, intercept + step)
            - objective(&scores, &corrected, slope, intercept - step))
            / (2.0 * step);
        assert!(d_slope.abs() <= 1.0e-4, "slope gradient {d_slope}");
        assert!(
            d_intercept.abs() <= 1.0e-4,
            "intercept gradient {d_intercept}"
        );

        // And it is a minimum, not merely stationary: every neighbour is worse.
        for (delta_slope, delta_intercept) in [
            (1.0e-2, 0.0),
            (-1.0e-2, 0.0),
            (0.0, 1.0e-2),
            (0.0, -1.0e-2),
            (1.0e-2, 1.0e-2),
            (-1.0e-2, 1.0e-2),
        ] {
            let neighbour = objective(
                &scores,
                &corrected,
                slope + delta_slope,
                intercept + delta_intercept,
            );
            assert!(
                neighbour > base,
                "neighbour ({delta_slope}, {delta_intercept}) scored {neighbour} against {base}"
            );
        }
    }

    #[test]
    fn a_separable_sample_still_has_a_finite_fit() {
        // With raw labels this problem has no finite optimum. The prior
        // correction is what keeps it well posed, so the fit must converge and
        // must not produce a saturated probability.
        let scores: Vec<f32> = (0..20).map(|step| step as f32 * 0.1).collect();
        let labels: Vec<u8> = (0..20).map(|step| u8::from(step >= 10)).collect();
        let targets = BinaryTargets::new(labels).unwrap();
        let fitted = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
        assert!(fitted.slope().is_finite() && fitted.intercept().is_finite());
        assert!(
            fitted.slope() > 0.0,
            "slope {} is not positive",
            fitted.slope()
        );
        assert!(fitted.n_iter() < fitted.get_params().max_iter());
        for &score in &scores {
            let probability = fitted.calibrate(score);
            assert!(
                probability > 0.0 && probability < 1.0,
                "score {score} calibrated to {probability}"
            );
        }
    }

    #[test]
    fn the_calibrated_map_is_monotone_in_the_score() {
        let (scores, targets) = sample();
        let fitted = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
        assert!(fitted.slope() > 0.0);
        let mut previous = f32::NEG_INFINITY;
        for step in -100..=200 {
            let probability = fitted.calibrate(step as f32 * 0.01);
            assert!(probability >= previous, "map decreased at {step}");
            assert!((0.0..=1.0).contains(&probability));
            previous = probability;
        }

        // A score anti-correlated with the label fits a negative slope, and the
        // map is then monotone the other way rather than being wrong.
        let flipped: Vec<f32> = scores.iter().map(|score| -score).collect();
        let mirrored = PlattCalibrator::fit(&flipped, &targets, PlattParams::default()).unwrap();
        assert!(mirrored.slope() < 0.0);
        assert!((mirrored.slope() + fitted.slope()).abs() <= 1.0e-4);
    }

    #[test]
    fn the_slope_takes_the_sign_of_the_calibration_sample_s_class_mean_gap() {
        // The documented ranking condition, enumerated rather than sampled:
        // every label assignment over one fixed score set that has both
        // classes. The slope's sign is what decides whether calibration
        // preserves or reverses the wrapped model's ranking, and the claim is
        // that it is decided by this sample statistic alone — the mean score
        // over positive rows minus the mean over negative rows — because the
        // objective is strictly convex and its slope gradient at the profile
        // optimum for a zero slope is that gap up to a positive factor.
        let scores = [0.05_f32, 0.2, 0.4, 0.55, 0.7, 0.9];
        let mut fitted = 0;
        let mut positive = 0;
        let mut negative = 0;
        for mask in 1_u32..(1 << scores.len()) - 1 {
            let labels: Vec<u8> = (0..scores.len())
                .map(|index| u8::from(mask >> index & 1 == 1))
                .collect();
            let targets = BinaryTargets::new(labels.clone()).unwrap();
            let fit = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
            let mut sums = [0.0_f64; 2];
            let mut counts = [0_usize; 2];
            for (&score, &label) in scores.iter().zip(&labels) {
                sums[usize::from(label)] += f64::from(score);
                counts[usize::from(label)] += 1;
            }
            let gap = sums[1] / counts[1] as f64 - sums[0] / counts[0] as f64;
            fitted += 1;
            assert_eq!(
                gap > 0.0,
                fit.slope() > 0.0,
                "labels {labels:?}: mean gap {gap} against slope {}",
                fit.slope()
            );
            if gap > 0.0 {
                positive += 1;
            } else {
                negative += 1;
            }
        }

        // Both branches have to occur, or the assertion above proves nothing
        // about the one that inverts a model's ranking.
        assert_eq!(fitted, (1 << scores.len()) - 2);
        assert!(
            positive > 0 && negative > 0,
            "{positive} against {negative}"
        );
    }

    #[test]
    fn refitting_the_same_sample_reproduces_the_same_parameters() {
        let (scores, targets) = sample();
        let first = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
        let second = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.slope().to_bits(), second.slope().to_bits());
        assert_eq!(first.intercept().to_bits(), second.intercept().to_bits());
    }

    #[test]
    fn undefined_and_invalid_inputs_are_reported_rather_than_fitted() {
        let (scores, targets) = sample();
        assert_eq!(
            PlattCalibrator::fit(&scores, &targets, PlattParams::default().with_max_iter(0)),
            Err(ModelError::InvalidIterationCount)
        );
        assert_eq!(
            PlattCalibrator::fit(&scores, &targets, PlattParams::default().with_tol(0.0)),
            Err(ModelError::InvalidTolerance)
        );
        assert_eq!(
            PlattCalibrator::fit(&scores, &targets, PlattParams::default().with_tol(f32::NAN)),
            Err(ModelError::InvalidTolerance)
        );
        assert_eq!(
            PlattCalibrator::fit(
                &[0.1, 0.2, 0.3],
                &BinaryTargets::new(vec![1, 1, 1]).unwrap(),
                PlattParams::default()
            ),
            Err(ModelError::RequiresTwoClasses)
        );
        assert_eq!(
            PlattCalibrator::fit(
                &[0.1, 0.2],
                &BinaryTargets::new(vec![0, 1, 1]).unwrap(),
                PlattParams::default()
            ),
            Err(ModelError::TargetLength {
                rows: 2,
                targets: 3,
            })
        );
        assert_eq!(
            PlattCalibrator::fit(
                &[],
                &BinaryTargets::new(vec![0, 1]).unwrap(),
                PlattParams::default()
            ),
            Err(ModelError::EmptyData)
        );
        assert_eq!(
            PlattCalibrator::fit(
                &[0.1, f32::INFINITY],
                &BinaryTargets::new(vec![0, 1]).unwrap(),
                PlattParams::default()
            ),
            Err(ModelError::NonFiniteFeature { row: 1, column: 0 })
        );
    }

    #[test]
    fn a_constant_score_column_is_reported_instead_of_solved() {
        // Every design row is identical, so the two-parameter system is
        // singular: there is no slope the data can identify.
        let scores = [0.5_f32; 8];
        let targets = BinaryTargets::new(vec![0, 1, 0, 1, 0, 1, 0, 1]).unwrap();
        assert_eq!(
            PlattCalibrator::fit(&scores, &targets, PlattParams::default()),
            Err(ModelError::LinearSolveFailed)
        );
    }

    #[test]
    fn the_decision_score_is_the_logit_of_the_calibrated_probability() {
        let (scores, targets) = sample();
        let fitted = PlattCalibrator::fit(&scores, &targets, PlattParams::default()).unwrap();
        for &score in &scores {
            let raw = fitted.decision_score(score);
            assert_eq!(fitted.calibrate(score), sigmoid_f32(raw));
            let probability = f64::from(fitted.calibrate(score));
            assert!(
                (f64::from(raw) - (probability / (1.0 - probability)).ln()).abs() <= 1.0e-5,
                "score {score}"
            );
        }
    }
}
