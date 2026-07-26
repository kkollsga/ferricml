use std::error::Error;
use std::fmt;

use crate::api::Regressor;
use crate::data::{ClassificationTargets, MatrixView, RegressionTargets};
use crate::model_selection::{
    ClassificationScore, RegressionScore, ScorableClassifier, ScoringError, ScoringWorkspace,
    score_labelled, score_regressor_with,
};
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
///
/// This is the crate's only classifier permutation-importance entry point, over
/// **any** [`ClassificationTargets`] vocabulary. Shuffling a column and
/// re-scoring says how much the model relied on it; how many classes the labels
/// name is a property of the *metric*, so
/// [`ClassTargets`](crate::data::ClassTargets) is inspected here exactly as
/// [`BinaryTargets`](crate::data::BinaryTargets) is. A score reading
/// [`ClassifierOutputKind::ProbabilityMatrix`](crate::model_selection::ClassifierOutputKind::ProbabilityMatrix)
/// works for any observed class set, while the binary positive-probability
/// layouts still refuse a wider one with
/// [`ScoringError::UnsupportedClasses`] rather than reinterpreting a column.
///
/// Values are quality **losses**, oriented so a larger number always means a
/// more important feature — whichever direction the underlying metric improves
/// in. The orientation is taken from the score's own declaration rather than
/// assumed.
///
/// ```
/// use ferricml::data::{ClassTargets, DenseMatrix};
/// use ferricml::inspection::{PermutationImportanceParams, permutation_importance_classifier};
/// use ferricml::model_selection::{ClassificationScorer, ScorableClassifier};
/// use ferricml::tree::{DecisionTreeClassifier, DecisionTreeClassifierParams};
///
/// // Column 0 determines one of three classes; column 1 alternates and says
/// // nothing. The labels are neither contiguous nor zero-based.
/// let mut values = Vec::new();
/// let mut labels = Vec::new();
/// for index in 0..30_usize {
///     values.push((index % 3) as f32);
///     values.push(if index % 2 == 0 { 1.0 } else { -1.0 });
///     labels.push([3, 7, 10][index % 3]);
/// }
/// let data = DenseMatrix::new(values, 30, 2)?;
/// let targets = ClassTargets::new(labels)?;
/// assert_eq!(targets.classes(), &[3, 7, 10]);
///
/// let model = DecisionTreeClassifier::fit_multiclass(
///     &data.as_view(),
///     &targets,
///     DecisionTreeClassifierParams::default().with_min_samples_leaf(1),
/// )?;
///
/// let importance = permutation_importance_classifier(
///     ScorableClassifier::probabilistic(&model),
///     &data.as_view(),
///     &targets,
///     ClassificationScorer::MulticlassLogLoss,
///     PermutationImportanceParams::default().with_random_state(5),
/// )?;
///
/// // The deciding column matters; the alternating one does not.
/// assert_eq!(importance.ranked()[0], 0);
/// assert!(importance.means()[0] > importance.means()[1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn permutation_importance_classifier<T: ClassificationTargets, S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &T,
    scorer: S,
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
pub fn permutation_importance_classifier_into<T: ClassificationTargets, S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &T,
    scorer: S,
    params: PermutationImportanceParams,
    means: &mut [f64],
    std_devs: &mut [f64],
) -> Result<(), InspectionError> {
    let labels = targets.as_slice();
    validate(data, labels.len(), params, means, std_devs)?;
    let greater_is_better = scorer.greater_is_better();
    let mut workspace = ScoringWorkspace::new();
    let mut score =
        |view: &MatrixView<'_>| score_labelled(classifier, view, labels, &scorer, &mut workspace);
    run_permutations(data, params, greater_is_better, means, std_devs, &mut score)
}

/// Measures permutation importance for a fitted regressor.
///
/// Shuffling a column and re-scoring says how much the model relied on it.
/// This works through the public batch prediction and scoring contracts alone,
/// so it needs no cooperation from the estimator and exposes no internals.
///
/// Values are quality **losses**, oriented so a larger number always means a
/// more important feature — whichever direction the underlying metric improves
/// in. The orientation is taken from the score's own declaration rather than
/// assumed.
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::inspection::{PermutationImportanceParams, permutation_importance_regressor};
/// use ferricml::linear_model::{LinearRegression, LinearRegressionParams};
/// use ferricml::model_selection::RegressionScorer;
///
/// // Column 0 drives the target; column 1 is noise.
/// let mut values = Vec::new();
/// let mut targets = Vec::new();
/// for index in 0..24 {
///     let signal = index as f32;
///     values.push(signal);
///     values.push(if index % 2 == 0 { 1.0 } else { -1.0 });
///     targets.push(2.0 * signal);
/// }
/// let data = DenseMatrix::new(values, 24, 2)?;
/// let targets = RegressionTargets::new(targets)?;
///
/// let model = LinearRegression::fit(
///     &data.as_view(),
///     &targets,
///     LinearRegressionParams::default(),
/// )?;
///
/// let importance = permutation_importance_regressor(
///     &model,
///     &data.as_view(),
///     &targets,
///     RegressionScorer::MeanSquaredError,
///     PermutationImportanceParams::default().with_random_state(4),
/// )?;
///
/// assert_eq!(importance.n_features(), 2);
/// // The signal column matters; the noise column does not.
/// assert!(importance.means()[0] > importance.means()[1]);
/// // `ranked` orders features most important first.
/// assert_eq!(importance.ranked()[0], 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn permutation_importance_regressor<S: RegressionScore>(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: S,
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
pub fn permutation_importance_regressor_into<S: RegressionScore>(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: S,
    params: PermutationImportanceParams,
    means: &mut [f64],
    std_devs: &mut [f64],
) -> Result<(), InspectionError> {
    validate(data, targets.len(), params, means, std_devs)?;
    let greater_is_better = scorer.greater_is_better();
    let mut workspace = ScoringWorkspace::new();
    let mut score = |view: &MatrixView<'_>| {
        score_regressor_with(regressor, view, targets, &scorer, &mut workspace)
    };
    run_permutations(data, params, greater_is_better, means, std_devs, &mut score)
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
