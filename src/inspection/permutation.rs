use std::error::Error;
use std::fmt;

use crate::api::{Classifier, Regressor};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};
use crate::metrics::{
    accuracy_score, brier_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
};
use crate::model_selection::{ClassificationScorer, RegressionScorer, ScoringError};
use crate::numeric::OwnedRng;

/// Parameters for a permutation-importance run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PermutationImportanceParams {
    n_repeats: usize,
    random_state: u64,
}

impl Default for PermutationImportanceParams {
    fn default() -> Self {
        Self {
            n_repeats: 5,
            random_state: 0,
        }
    }
}

impl PermutationImportanceParams {
    /// Sets how many independent permutations each feature receives.
    #[must_use]
    pub fn with_n_repeats(mut self, n_repeats: usize) -> Self {
        self.n_repeats = n_repeats;
        self
    }

    /// Sets the deterministic permutation seed.
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// Returns the number of permutations per feature.
    pub const fn n_repeats(&self) -> usize {
        self.n_repeats
    }

    /// Returns the deterministic permutation seed.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }
}

/// Per-feature permutation importance.
///
/// Each value is the loss of scorer quality caused by permuting that feature,
/// so a larger number always means a more important feature whether the
/// underlying metric is maximized (accuracy, R²) or minimized (log loss, mean
/// squared error). Both slices are indexed by input feature.
#[derive(Clone, Debug, PartialEq)]
pub struct PermutationImportance {
    means: Vec<f64>,
    std_devs: Vec<f64>,
}

impl PermutationImportance {
    /// Number of inspected features.
    pub fn n_features(&self) -> usize {
        self.means.len()
    }

    /// Mean quality loss per feature, in scorer units.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Population standard deviation of the per-repeat quality loss.
    pub fn std_devs(&self) -> &[f64] {
        &self.std_devs
    }

    /// Feature indices ordered by decreasing mean importance.
    ///
    /// Equal means keep their natural feature order, so the ranking is
    /// deterministic for every input.
    pub fn ranked(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.means.len()).collect();
        order.sort_by(|&left, &right| {
            self.means[right]
                .total_cmp(&self.means[left])
                .then(left.cmp(&right))
        });
        order
    }
}

/// Errors produced while inspecting a fitted model.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum InspectionError {
    /// The requested permutation count was zero.
    InvalidRepeatCount,
    /// A caller-owned output slice did not have one entry per feature.
    OutputLength {
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Scoring the fitted model failed.
    Scoring(ScoringError),
}

impl fmt::Display for InspectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRepeatCount => f.write_str("permutation repeats must be at least one"),
            Self::OutputLength { expected, actual } => write!(
                f,
                "importance output length {actual} does not match {expected} features"
            ),
            Self::Scoring(error) => write!(f, "scoring failed: {error}"),
        }
    }
}

impl Error for InspectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Scoring(error) => Some(error),
            _ => None,
        }
    }
}

/// Measures permutation importance for a fitted classifier.
pub fn permutation_importance_classifier(
    classifier: &dyn Classifier,
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    scorer: ClassificationScorer,
    params: PermutationImportanceParams,
) -> Result<PermutationImportance, InspectionError> {
    let mut means = vec![0.0; data.columns()];
    let mut std_devs = vec![0.0; data.columns()];
    permutation_importance_classifier_into(
        classifier,
        data,
        targets,
        scorer,
        params,
        &mut means,
        &mut std_devs,
    )?;
    Ok(PermutationImportance { means, std_devs })
}

/// Measures permutation importance for a fitted classifier into caller-owned
/// storage.
///
/// Both output slices must have one entry per input feature. Prediction and
/// permutation workspace is allocated once, so the cost of extra repeats is
/// scoring alone.
pub fn permutation_importance_classifier_into(
    classifier: &dyn Classifier,
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    scorer: ClassificationScorer,
    params: PermutationImportanceParams,
    means: &mut [f64],
    std_devs: &mut [f64],
) -> Result<(), InspectionError> {
    validate(data, targets.len(), params, means, std_devs)?;
    let mut labels = vec![0_u8; if uses_labels(scorer) { data.rows() } else { 0 }];
    let mut probabilities = vec![0.0_f32; if uses_labels(scorer) { 0 } else { data.rows() }];
    let mut score = |view: &MatrixView<'_>| {
        classification_score(
            classifier,
            view,
            targets.as_slice(),
            scorer,
            &mut labels,
            &mut probabilities,
        )
    };
    run_permutations(
        data,
        params,
        classification_greater_is_better(scorer),
        means,
        std_devs,
        &mut score,
    )
}

/// Measures permutation importance for a fitted regressor.
pub fn permutation_importance_regressor(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: RegressionScorer,
    params: PermutationImportanceParams,
) -> Result<PermutationImportance, InspectionError> {
    let mut means = vec![0.0; data.columns()];
    let mut std_devs = vec![0.0; data.columns()];
    permutation_importance_regressor_into(
        regressor,
        data,
        targets,
        scorer,
        params,
        &mut means,
        &mut std_devs,
    )?;
    Ok(PermutationImportance { means, std_devs })
}

/// Measures permutation importance for a fitted regressor into caller-owned
/// storage.
///
/// Both output slices must have one entry per input feature. Prediction and
/// permutation workspace is allocated once, so the cost of extra repeats is
/// scoring alone.
pub fn permutation_importance_regressor_into(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: RegressionScorer,
    params: PermutationImportanceParams,
    means: &mut [f64],
    std_devs: &mut [f64],
) -> Result<(), InspectionError> {
    validate(data, targets.len(), params, means, std_devs)?;
    let mut predictions = vec![0.0_f32; data.rows()];
    let mut score = |view: &MatrixView<'_>| {
        regression_score(
            regressor,
            view,
            targets.as_slice(),
            scorer,
            &mut predictions,
        )
    };
    run_permutations(
        data,
        params,
        regression_greater_is_better(scorer),
        means,
        std_devs,
        &mut score,
    )
}

/// Rejects shape and parameter problems before any prediction work happens.
fn validate(
    data: &MatrixView<'_>,
    targets: usize,
    params: PermutationImportanceParams,
    means: &[f64],
    std_devs: &[f64],
) -> Result<(), InspectionError> {
    if data.rows() != targets {
        return Err(InspectionError::Scoring(ScoringError::TargetLength {
            rows: data.rows(),
            targets,
        }));
    }
    for output in [means, std_devs] {
        if output.len() != data.columns() {
            return Err(InspectionError::OutputLength {
                expected: data.columns(),
                actual: output.len(),
            });
        }
    }
    if params.n_repeats == 0 {
        return Err(InspectionError::InvalidRepeatCount);
    }
    Ok(())
}

/// Scores the unpermuted data once, then each feature `n_repeats` times.
///
/// Features are visited in ascending index order and repeats in ascending
/// order within a feature, drawing from one seeded stream, which is what makes
/// the result reproducible for a given seed.
fn run_permutations(
    data: &MatrixView<'_>,
    params: PermutationImportanceParams,
    greater_is_better: bool,
    means: &mut [f64],
    std_devs: &mut [f64],
    score: &mut dyn FnMut(&MatrixView<'_>) -> Result<f64, ScoringError>,
) -> Result<(), InspectionError> {
    let rows = data.rows();
    let columns = data.columns();
    let baseline = score(data).map_err(InspectionError::Scoring)?;

    let mut values: Vec<f32> = data.iter_rows().flatten().copied().collect();
    let mut order: Vec<usize> = (0..rows).collect();
    let mut original = vec![0.0_f32; rows];
    let mut rng = OwnedRng::new(params.random_state);

    for feature in 0..columns {
        for (row, slot) in original.iter_mut().enumerate() {
            *slot = values[row * columns + feature];
        }
        let mut total = 0.0_f64;
        let mut total_squares = 0.0_f64;
        for _ in 0..params.n_repeats {
            shuffle(&mut order, &mut rng);
            for row in 0..rows {
                values[row * columns + feature] = original[order[row]];
            }
            let view = MatrixView::new(&values, rows, columns)
                .expect("permuting a column preserves the validated matrix");
            let permuted = score(&view).map_err(InspectionError::Scoring)?;
            let delta = if greater_is_better {
                baseline - permuted
            } else {
                permuted - baseline
            };
            total += delta;
            total_squares += delta * delta;
        }
        for (row, &value) in original.iter().enumerate() {
            values[row * columns + feature] = value;
        }

        let repeats = params.n_repeats as f64;
        let mean = total / repeats;
        means[feature] = mean;
        // Population variance, clamped against cancellation noise.
        std_devs[feature] = (total_squares / repeats - mean * mean).max(0.0).sqrt();
    }
    Ok(())
}

fn shuffle(order: &mut [usize], rng: &mut OwnedRng) {
    for index in (1..order.len()).rev() {
        order.swap(index, rng.index(index + 1));
    }
}

const fn uses_labels(scorer: ClassificationScorer) -> bool {
    matches!(
        scorer,
        ClassificationScorer::Accuracy
            | ClassificationScorer::Precision
            | ClassificationScorer::Recall
            | ClassificationScorer::F1
    )
}

const fn classification_greater_is_better(scorer: ClassificationScorer) -> bool {
    !matches!(
        scorer,
        ClassificationScorer::Brier | ClassificationScorer::LogLoss
    )
}

const fn regression_greater_is_better(scorer: RegressionScorer) -> bool {
    matches!(scorer, RegressionScorer::R2)
}

/// Scores a classifier through the allocation-free batch contract.
fn classification_score(
    classifier: &dyn Classifier,
    data: &MatrixView<'_>,
    targets: &[u8],
    scorer: ClassificationScorer,
    labels: &mut [u8],
    probabilities: &mut [f32],
) -> Result<f64, ScoringError> {
    if uses_labels(scorer) {
        classifier
            .predict_into(data, labels)
            .map_err(ScoringError::Prediction)?;
        let labels = &*labels;
        return match scorer {
            ClassificationScorer::Accuracy => accuracy_score(targets, labels),
            ClassificationScorer::Precision => precision_score(targets, labels),
            ClassificationScorer::Recall => recall_score(targets, labels),
            _ => f1_score(targets, labels),
        }
        .map_err(ScoringError::Metric);
    }
    match classifier.classes() {
        [0] => probabilities.fill(0.0),
        [1] => probabilities.fill(1.0),
        [0, 1] => classifier
            .predict_class_proba_into(data, 1, probabilities)
            .map_err(ScoringError::Prediction)?,
        _ => return Err(ScoringError::UnsupportedClasses),
    }
    let probabilities = &*probabilities;
    match scorer {
        ClassificationScorer::Brier => brier_score(targets, probabilities),
        ClassificationScorer::LogLoss => log_loss(targets, probabilities),
        _ => roc_auc_score(targets, probabilities),
    }
    .map_err(ScoringError::Metric)
}

/// Scores a regressor through the allocation-free batch contract.
fn regression_score(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &[f32],
    scorer: RegressionScorer,
    predictions: &mut [f32],
) -> Result<f64, ScoringError> {
    regressor
        .predict_into(data, predictions)
        .map_err(ScoringError::Prediction)?;
    let predictions = &*predictions;
    match scorer {
        RegressionScorer::MeanAbsoluteError => mean_absolute_error(targets, predictions),
        RegressionScorer::MeanSquaredError => mean_squared_error(targets, predictions),
        RegressionScorer::RootMeanSquaredError => root_mean_squared_error(targets, predictions),
        _ => r2_score(targets, predictions),
    }
    .map_err(ScoringError::Metric)
}
