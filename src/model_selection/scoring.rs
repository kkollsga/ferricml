use std::error::Error;
use std::fmt;

use crate::api::{Classifier, ModelError, ProbabilisticClassifier, Regressor};
use crate::data::{BinaryTargets, ClassTargets, MatrixView, RegressionTargets};
use crate::metrics::{
    MetricError, accuracy_score, brier_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, multiclass_brier_score, multiclass_log_loss, precision_score, r2_score,
    recall_score, roc_auc_score, root_mean_squared_error,
};

/// Built-in scores for fitted binary classifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClassificationScorer {
    /// Exact label accuracy.
    Accuracy,
    /// Positive predictive value.
    Precision,
    /// Positive-class recall.
    Recall,
    /// Harmonic mean of precision and recall.
    F1,
    /// Mean squared positive-probability error.
    Brier,
    /// Mean binary logarithmic loss.
    LogLoss,
    /// Area under the receiver-operating-characteristic curve.
    RocAuc,
    // New variants are appended rather than inserted: this enum is
    // `#[non_exhaustive]`, but moving an existing variant's discriminant is
    // still a reported change and buys nothing.
    /// Mean cross-entropy of a whole probability matrix.
    ///
    /// Unlike [`LogLoss`](Self::LogLoss) this reads every probability column
    /// and therefore scores an arbitrary observed class set. On a two-class
    /// classifier the two agree.
    MulticlassLogLoss,
    /// Mean squared error of a whole probability row, over every class.
    ///
    /// Deliberately *not* interchangeable with [`Brier`](Self::Brier), which
    /// squares the positive column alone; on the same two-class predictions
    /// this is exactly twice that value.
    MulticlassBrier,
}

/// Built-in scores for fitted regressors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegressionScorer {
    /// Mean absolute error.
    MeanAbsoluteError,
    /// Mean squared error.
    MeanSquaredError,
    /// Root mean squared error.
    RootMeanSquaredError,
    /// Coefficient of determination.
    R2,
}

/// Which batch output a classification score reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClassifierOutputKind {
    /// One predicted label per row.
    Labels,
    /// One positive-class probability per row.
    PositiveProbabilities,
    /// A row-major probability matrix with one column per observed class.
    ProbabilityMatrix,
}

/// One batch of classifier output, ready to score.
///
/// Producing this is the caller's job, not the score's: a score declares what
/// it reads through [`ClassificationScore::output_kind`] and receives exactly
/// that, so the prediction call happens once per batch however many scores
/// consume it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ClassifierOutput<'a> {
    /// Predicted labels, one per row.
    Labels(&'a [u8]),
    /// Positive-class probabilities, one per row.
    PositiveProbabilities(&'a [f32]),
    /// A whole probability matrix and the class list naming its columns.
    ///
    /// This is the variant that makes probability scoring independent of the
    /// binary class layouts: the columns are identified by the classifier's own
    /// sorted class list rather than assumed to be `[0, 1]`, so a score reading
    /// it works for any observed class set.
    ProbabilityMatrix {
        /// The classifier's sorted class labels, one per column.
        classes: &'a [u8],
        /// Row-major probabilities, `classes.len()` per row.
        probabilities: &'a [f32],
    },
}

impl ClassifierOutput<'_> {
    /// Which kind of output this batch holds.
    pub const fn kind(&self) -> ClassifierOutputKind {
        match self {
            Self::Labels(_) => ClassifierOutputKind::Labels,
            Self::PositiveProbabilities(_) => ClassifierOutputKind::PositiveProbabilities,
            Self::ProbabilityMatrix { .. } => ClassifierOutputKind::ProbabilityMatrix,
        }
    }
}

/// A score over one fitted classifier's batch output.
///
/// This is the seam that makes scoring open: cross-validation, batch scoring,
/// and permutation importance all consume this trait, so a caller can score on
/// a metric FerricML has not enumerated without reimplementing the prediction
/// or class-layout handling. The built-in [`ClassificationScorer`] is one
/// implementation of it, not a privileged one.
pub trait ClassificationScore {
    /// Which batch output [`ClassificationScore::score`] expects.
    fn output_kind(&self) -> ClassifierOutputKind;

    /// Whether a larger value means a better model.
    ///
    /// Consumers that measure degradation — permutation importance is the one
    /// in the crate — need this to orient their result, so it belongs with the
    /// score rather than being re-derived per consumer.
    fn greater_is_better(&self) -> bool;

    /// Scores one batch against its expected labels.
    ///
    /// Implementations receive the output kind they declared. Any other kind
    /// is [`ScoringError::UnsupportedOutput`], never a substituted value.
    fn score(&self, expected: &[u8], output: ClassifierOutput<'_>) -> Result<f64, ScoringError>;
}

/// A score over one fitted regressor's batch output.
pub trait RegressionScore {
    /// Whether a larger value means a better model.
    fn greater_is_better(&self) -> bool;

    /// Scores predicted values against their expected values.
    fn score(&self, expected: &[f32], predicted: &[f32]) -> Result<f64, ScoringError>;
}

impl<S: ClassificationScore + ?Sized> ClassificationScore for &S {
    fn output_kind(&self) -> ClassifierOutputKind {
        (**self).output_kind()
    }

    fn greater_is_better(&self) -> bool {
        (**self).greater_is_better()
    }

    fn score(&self, expected: &[u8], output: ClassifierOutput<'_>) -> Result<f64, ScoringError> {
        (**self).score(expected, output)
    }
}

impl<S: RegressionScore + ?Sized> RegressionScore for &S {
    fn greater_is_better(&self) -> bool {
        (**self).greater_is_better()
    }

    fn score(&self, expected: &[f32], predicted: &[f32]) -> Result<f64, ScoringError> {
        (**self).score(expected, predicted)
    }
}

impl ClassificationScore for ClassificationScorer {
    fn output_kind(&self) -> ClassifierOutputKind {
        match self {
            Self::Accuracy | Self::Precision | Self::Recall | Self::F1 => {
                ClassifierOutputKind::Labels
            }
            Self::Brier | Self::LogLoss | Self::RocAuc => {
                ClassifierOutputKind::PositiveProbabilities
            }
            Self::MulticlassLogLoss | Self::MulticlassBrier => {
                ClassifierOutputKind::ProbabilityMatrix
            }
        }
    }

    fn greater_is_better(&self) -> bool {
        !matches!(
            self,
            Self::Brier | Self::LogLoss | Self::MulticlassLogLoss | Self::MulticlassBrier
        )
    }

    fn score(&self, expected: &[u8], output: ClassifierOutput<'_>) -> Result<f64, ScoringError> {
        let value = match (self, output) {
            (Self::Accuracy, ClassifierOutput::Labels(predicted)) => {
                accuracy_score(expected, predicted)
            }
            (Self::Precision, ClassifierOutput::Labels(predicted)) => {
                precision_score(expected, predicted)
            }
            (Self::Recall, ClassifierOutput::Labels(predicted)) => {
                recall_score(expected, predicted)
            }
            (Self::F1, ClassifierOutput::Labels(predicted)) => f1_score(expected, predicted),
            (Self::Brier, ClassifierOutput::PositiveProbabilities(probabilities)) => {
                brier_score(expected, probabilities)
            }
            (Self::LogLoss, ClassifierOutput::PositiveProbabilities(probabilities)) => {
                log_loss(expected, probabilities)
            }
            (Self::RocAuc, ClassifierOutput::PositiveProbabilities(probabilities)) => {
                roc_auc_score(expected, probabilities)
            }
            (
                Self::MulticlassLogLoss,
                ClassifierOutput::ProbabilityMatrix {
                    classes,
                    probabilities,
                },
            ) => multiclass_log_loss(expected, classes, probabilities),
            (
                Self::MulticlassBrier,
                ClassifierOutput::ProbabilityMatrix {
                    classes,
                    probabilities,
                },
            ) => multiclass_brier_score(expected, classes, probabilities),
            (_, supplied) => {
                return Err(ScoringError::UnsupportedOutput {
                    required: self.output_kind(),
                    supplied: supplied.kind(),
                });
            }
        };
        value.map_err(ScoringError::Metric)
    }
}

impl RegressionScore for RegressionScorer {
    fn greater_is_better(&self) -> bool {
        matches!(self, Self::R2)
    }

    fn score(&self, expected: &[f32], predicted: &[f32]) -> Result<f64, ScoringError> {
        match self {
            Self::MeanAbsoluteError => mean_absolute_error(expected, predicted),
            Self::MeanSquaredError => mean_squared_error(expected, predicted),
            Self::RootMeanSquaredError => root_mean_squared_error(expected, predicted),
            Self::R2 => r2_score(expected, predicted),
        }
        .map_err(ScoringError::Metric)
    }
}

/// Reusable prediction storage for scoring one fitted model repeatedly.
///
/// Scoring writes the model's batch output here instead of allocating it, so a
/// caller that scores the same shape many times — cross-validation across
/// folds, permutation importance across repeats — allocates on the first call
/// and never again.
#[derive(Clone, Debug, Default)]
pub struct ScoringWorkspace {
    labels: Vec<u8>,
    values: Vec<f32>,
}

impl ScoringWorkspace {
    /// Creates an empty workspace that sizes itself on first use.
    pub fn new() -> Self {
        Self::default()
    }

    fn labels(&mut self, rows: usize) -> &mut [u8] {
        self.labels.resize(rows, 0);
        &mut self.labels
    }

    fn values(&mut self, rows: usize) -> &mut [f32] {
        self.values.resize(rows, 0.0);
        &mut self.values
    }

    /// Storage for a whole probability matrix, reusing the value buffer.
    ///
    /// A probability matrix is `rows * columns` values, so it shares the
    /// workspace's value allocation rather than adding a second one; the
    /// buffer grows to the largest shape any call has needed and never
    /// reallocates for a repeat of that shape.
    fn matrix(&mut self, rows: usize, columns: usize) -> Result<&mut [f32], ScoringError> {
        let len = rows.checked_mul(columns).ok_or(ScoringError::Prediction(
            ModelError::OutputShapeOverflow { rows, columns },
        ))?;
        Ok(self.values(len))
    }
}

/// Errors produced while scoring a fitted estimator.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ScoringError {
    /// Target count did not match the evaluated row count.
    TargetLength {
        /// Evaluated rows.
        rows: usize,
        /// Supplied targets.
        targets: usize,
    },
    /// A classifier exposed classes outside the supported binary layouts.
    UnsupportedClasses,
    /// A score received a batch output other than the one it declared.
    UnsupportedOutput {
        /// Output the score declared it reads.
        required: ClassifierOutputKind,
        /// Output it was given.
        supplied: ClassifierOutputKind,
    },
    /// Batch prediction failed.
    Prediction(ModelError),
    /// The selected metric rejected or could not score the predictions.
    Metric(MetricError),
}

impl fmt::Display for ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetLength { rows, targets } => {
                write!(
                    f,
                    "target length {targets} does not match {rows} evaluated rows"
                )
            }
            Self::UnsupportedClasses => {
                f.write_str("binary scoring requires classes [0], [1], or [0, 1]")
            }
            Self::UnsupportedOutput { required, supplied } => {
                write!(f, "score reads {required:?} but was given {supplied:?}")
            }
            Self::Prediction(error) => write!(f, "prediction failed: {error}"),
            Self::Metric(error) => write!(f, "metric failed: {error}"),
        }
    }
}

impl Error for ScoringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Prediction(error) => Some(error),
            Self::Metric(error) => Some(error),
            _ => None,
        }
    }
}

/// A fitted classifier as the scoring layer sees it.
///
/// Every classifier predicts labels; only some produce probabilities. A score
/// declares which it reads through
/// [`ClassificationScore::output_kind`], so the requirement is a property of
/// the *metric*, not of the entry point — accuracy and log loss reach the same
/// function. This type carries exactly what the held classifier can do, so a
/// label-only estimator is scorable on a label metric and is refused, with
/// [`ScoringError::UnsupportedOutput`], on a probability metric.
///
/// One type rather than a parallel family of `*_labels` functions: it keeps
/// one way to score, and it makes "the labels and the probabilities come from
/// the same model" true by construction rather than by convention.
///
/// ```
/// use ferricml::api::Classifier;
/// use ferricml::data::{BinaryTargets, DenseMatrix};
/// use ferricml::dummy::{DummyClassifier, DummyClassifierParams};
/// use ferricml::model_selection::{
///     ClassificationScorer, ScorableClassifier, score_classifier,
/// };
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
/// let targets = BinaryTargets::new(vec![0, 0, 1, 1])?;
/// let model = DummyClassifier::fit(&data.as_view(), &targets, DummyClassifierParams)?;
///
/// // This model does produce probabilities, so either view scores on accuracy.
/// let view = ScorableClassifier::probabilistic(&model);
/// let accuracy = score_classifier(view, &data.as_view(), &targets, ClassificationScorer::Accuracy)?;
/// assert!((0.0..=1.0).contains(&accuracy));
///
/// // Viewed as labels only, a probability metric is refused rather than guessed.
/// let labels_only = ScorableClassifier::labels_only(&model as &dyn Classifier);
/// assert!(
///     score_classifier(labels_only, &data.as_view(), &targets, ClassificationScorer::LogLoss)
///         .is_err()
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy)]
pub struct ScorableClassifier<'a> {
    labels: &'a dyn Classifier,
    probabilities: Option<&'a dyn ProbabilisticClassifier>,
}

impl<'a> ScorableClassifier<'a> {
    /// Views a classifier that produces probabilities, so every score applies.
    #[must_use]
    pub fn probabilistic(model: &'a dyn ProbabilisticClassifier) -> Self {
        Self {
            labels: model,
            probabilities: Some(model),
        }
    }

    /// Views a classifier by its labels alone.
    ///
    /// A probability metric applied to this is
    /// [`ScoringError::UnsupportedOutput`] — never a substituted value, and
    /// never a compile error the caller cannot work around.
    #[must_use]
    pub const fn labels_only(model: &'a dyn Classifier) -> Self {
        Self {
            labels: model,
            probabilities: None,
        }
    }

    /// The probabilities this view offers, or the error naming what is missing.
    fn probabilities(
        self,
        required: ClassifierOutputKind,
    ) -> Result<&'a dyn ProbabilisticClassifier, ScoringError> {
        self.probabilities.ok_or(ScoringError::UnsupportedOutput {
            required,
            supplied: ClassifierOutputKind::Labels,
        })
    }
}

/// Scores one fitted classifier through a single batch prediction call.
///
/// The score decides what it reads: a label metric works for any classifier, a
/// probability metric needs a [`ScorableClassifier::probabilistic`] view and is
/// otherwise refused with [`ScoringError::UnsupportedOutput`].
pub fn score_classifier<S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    scorer: S,
) -> Result<f64, ScoringError> {
    score_classifier_with(
        classifier,
        data,
        targets,
        scorer,
        &mut ScoringWorkspace::new(),
    )
}

/// Scores one fitted classifier into caller-owned prediction storage.
///
/// This is the allocation-free form: reusing one workspace across calls of the
/// same shape allocates on the first call only. The class layouts `[0]`, `[1]`,
/// and `[0, 1]` are handled here, once, so no consumer re-derives them.
///
/// A label-only classifier reaches this through
/// [`ScorableClassifier::labels_only`] and is scored on any label metric.
pub fn score_classifier_with<S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    scorer: S,
    workspace: &mut ScoringWorkspace,
) -> Result<f64, ScoringError> {
    score_labelled(classifier, data, targets.as_slice(), scorer, workspace)
}

/// Scores one fitted classifier against an arbitrary observed class set.
///
/// This is [`score_classifier`] over [`ClassTargets`] instead of
/// [`BinaryTargets`]. It runs the same single batch prediction call and the
/// same score contract; only the target vocabulary is wider. Scores reading
/// [`ClassifierOutputKind::ProbabilityMatrix`] work here for any number of
/// classes, while the binary positive-probability layouts remain what they were.
pub fn score_multiclass_classifier<S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &ClassTargets,
    scorer: S,
) -> Result<f64, ScoringError> {
    score_multiclass_classifier_with(
        classifier,
        data,
        targets,
        scorer,
        &mut ScoringWorkspace::new(),
    )
}

/// Scores one fitted classifier over a class set into caller-owned storage.
///
/// The allocation-free form of [`score_multiclass_classifier`].
pub fn score_multiclass_classifier_with<S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &ClassTargets,
    scorer: S,
    workspace: &mut ScoringWorkspace,
) -> Result<f64, ScoringError> {
    score_labelled(classifier, data, targets.as_slice(), scorer, workspace)
}

/// The one classifier scoring implementation, over already-validated labels.
///
/// Every target vocabulary reaches the model through this function, so the
/// prediction call, the class-layout handling, and the workspace reuse exist
/// exactly once. Cross-validation calls it directly rather than through one of
/// the two public wrappers: it is generic over the target vocabulary and
/// already holds a validated fold of it, so routing through a wrapper would
/// mean choosing a vocabulary the loop deliberately does not know.
pub(super) fn score_labelled<S: ClassificationScore>(
    classifier: ScorableClassifier<'_>,
    data: &MatrixView<'_>,
    targets: &[u8],
    scorer: S,
    workspace: &mut ScoringWorkspace,
) -> Result<f64, ScoringError> {
    validate_target_length(data.rows(), targets.len())?;
    let output = match scorer.output_kind() {
        ClassifierOutputKind::Labels => {
            let labels = workspace.labels(data.rows());
            classifier
                .labels
                .predict_into(data, labels)
                .map_err(ScoringError::Prediction)?;
            ClassifierOutput::Labels(labels)
        }
        ClassifierOutputKind::PositiveProbabilities => {
            let model = classifier.probabilities(ClassifierOutputKind::PositiveProbabilities)?;
            let probabilities = workspace.values(data.rows());
            match model.classes() {
                [0] => probabilities.fill(0.0),
                [1] => probabilities.fill(1.0),
                [0, 1] => model
                    .predict_class_proba_into(data, 1, probabilities)
                    .map_err(ScoringError::Prediction)?,
                _ => return Err(ScoringError::UnsupportedClasses),
            }
            ClassifierOutput::PositiveProbabilities(probabilities)
        }
        ClassifierOutputKind::ProbabilityMatrix => {
            let model = classifier.probabilities(ClassifierOutputKind::ProbabilityMatrix)?;
            let classes = model.classes();
            let probabilities = workspace.matrix(data.rows(), classes.len())?;
            model
                .predict_proba_into(data, probabilities)
                .map_err(ScoringError::Prediction)?;
            ClassifierOutput::ProbabilityMatrix {
                classes,
                probabilities,
            }
        }
    };
    scorer.score(targets, output)
}

/// Scores one fitted regressor through a single batch prediction call.
pub fn score_regressor<S: RegressionScore>(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: S,
) -> Result<f64, ScoringError> {
    score_regressor_with(
        regressor,
        data,
        targets,
        scorer,
        &mut ScoringWorkspace::new(),
    )
}

/// Scores one fitted regressor into caller-owned prediction storage.
///
/// This is the allocation-free form: reusing one workspace across calls of the
/// same shape allocates on the first call only.
pub fn score_regressor_with<S: RegressionScore>(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: S,
    workspace: &mut ScoringWorkspace,
) -> Result<f64, ScoringError> {
    validate_target_length(data.rows(), targets.len())?;
    let predicted = workspace.values(data.rows());
    regressor
        .predict_into(data, predicted)
        .map_err(ScoringError::Prediction)?;
    scorer.score(targets.as_slice(), predicted)
}

fn validate_target_length(rows: usize, targets: usize) -> Result<(), ScoringError> {
    if rows != targets {
        return Err(ScoringError::TargetLength { rows, targets });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{AnyClassifier, AnyRegressor};
    use crate::data::{ClassTargets, DenseMatrix};
    use crate::ensemble::{
        MaxFeatures, RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
        RandomForestRegressorParams,
    };
    use crate::metrics::{Average, ConfusionMatrix};

    fn matrix() -> DenseMatrix {
        DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap()
    }

    fn classifier(targets: &BinaryTargets) -> RandomForestClassifier {
        RandomForestClassifier::fit(
            &matrix().as_view(),
            targets,
            RandomForestClassifierParams::default()
                .with_n_estimators(1)
                .with_bootstrap(false)
                .with_max_features(MaxFeatures::All),
        )
        .unwrap()
    }

    /// A score FerricML does not enumerate, written only against the trait.
    struct MacroF1;

    impl ClassificationScore for MacroF1 {
        fn output_kind(&self) -> ClassifierOutputKind {
            ClassifierOutputKind::Labels
        }

        fn greater_is_better(&self) -> bool {
            true
        }

        fn score(
            &self,
            expected: &[u8],
            output: ClassifierOutput<'_>,
        ) -> Result<f64, ScoringError> {
            let ClassifierOutput::Labels(predicted) = output else {
                return Err(ScoringError::UnsupportedOutput {
                    required: self.output_kind(),
                    supplied: output.kind(),
                });
            };
            ConfusionMatrix::new(expected, predicted)
                .and_then(|matrix| matrix.f1(Average::Macro))
                .map_err(ScoringError::Metric)
        }
    }

    #[test]
    fn classifier_scorers_equal_direct_metric_calls_for_concrete_and_erased_models() {
        let data = matrix();
        let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let concrete = classifier(&targets);
        let labels = concrete.predict(&data.as_view()).unwrap();
        let probabilities = concrete.predict_class_proba(&data.as_view(), 1).unwrap();
        assert_eq!(
            score_classifier(
                ScorableClassifier::probabilistic(&concrete),
                &data.as_view(),
                &targets,
                ClassificationScorer::Accuracy
            ),
            accuracy_score(targets.as_slice(), &labels).map_err(ScoringError::Metric)
        );
        assert_eq!(
            score_classifier(
                ScorableClassifier::probabilistic(&concrete),
                &data.as_view(),
                &targets,
                ClassificationScorer::Brier
            ),
            brier_score(targets.as_slice(), &probabilities).map_err(ScoringError::Metric)
        );

        let erased: AnyClassifier = concrete.into();
        for scorer in [
            ClassificationScorer::Accuracy,
            ClassificationScorer::Precision,
            ClassificationScorer::Recall,
            ClassificationScorer::F1,
            ClassificationScorer::Brier,
            ClassificationScorer::LogLoss,
            ClassificationScorer::RocAuc,
        ] {
            assert!(
                score_classifier(
                    ScorableClassifier::probabilistic(
                        erased
                            .as_probabilistic()
                            .expect("every shipped variant produces probabilities")
                    ),
                    &data.as_view(),
                    &targets,
                    scorer
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn singleton_class_probability_scores_are_explicit() {
        let data = matrix();
        for (label, expected_brier) in [(0, 0.0), (1, 0.0)] {
            let targets = BinaryTargets::new(vec![label; 4]).unwrap();
            let model = classifier(&targets);
            assert_eq!(
                score_classifier(
                    ScorableClassifier::probabilistic(&model),
                    &data.as_view(),
                    &targets,
                    ClassificationScorer::Brier
                ),
                Ok(expected_brier)
            );
            assert_eq!(
                score_classifier(
                    ScorableClassifier::probabilistic(&model),
                    &data.as_view(),
                    &targets,
                    ClassificationScorer::RocAuc
                ),
                Err(ScoringError::Metric(MetricError::Undefined))
            );
        }
    }

    #[test]
    fn regressor_scorers_equal_direct_metrics_and_runtime_dispatch() {
        let data = matrix();
        let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
        let concrete = RandomForestRegressor::fit(
            &data.as_view(),
            &targets,
            RandomForestRegressorParams::default()
                .with_n_estimators(1)
                .with_bootstrap(false),
        )
        .unwrap();
        let erased: AnyRegressor = concrete.into();
        for scorer in [
            RegressionScorer::MeanAbsoluteError,
            RegressionScorer::MeanSquaredError,
            RegressionScorer::RootMeanSquaredError,
            RegressionScorer::R2,
        ] {
            assert!(score_regressor(&erased, &data.as_view(), &targets, scorer).is_ok());
        }
    }

    #[test]
    fn scoring_validates_targets_before_prediction_and_preserves_metric_errors() {
        let data = matrix();
        let targets = BinaryTargets::new(vec![0, 1]).unwrap();
        let fitted_targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let model = classifier(&fitted_targets);
        assert_eq!(
            score_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &targets,
                ClassificationScorer::Accuracy
            ),
            Err(ScoringError::TargetLength {
                rows: 4,
                targets: 2,
            })
        );

        let all_negative = BinaryTargets::new(vec![0; 4]).unwrap();
        assert_eq!(
            score_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &all_negative,
                ClassificationScorer::Recall
            ),
            Err(ScoringError::Metric(MetricError::Undefined))
        );
    }

    #[test]
    fn the_workspace_form_agrees_with_the_allocating_one_and_can_be_reused() {
        let data = matrix();
        let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let model = classifier(&binary);
        let mut workspace = ScoringWorkspace::new();
        for scorer in [
            ClassificationScorer::Accuracy,
            ClassificationScorer::F1,
            ClassificationScorer::Brier,
            ClassificationScorer::RocAuc,
        ] {
            let allocating = score_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &binary,
                scorer,
            );
            let reused = score_classifier_with(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &binary,
                scorer,
                &mut workspace,
            );
            assert_eq!(allocating, reused, "{scorer:?}");
            // A second pass through the same workspace repeats exactly.
            assert_eq!(
                reused,
                score_classifier_with(
                    ScorableClassifier::probabilistic(&model),
                    &data.as_view(),
                    &binary,
                    scorer,
                    &mut workspace
                )
            );
        }

        let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
        let regressor = RandomForestRegressor::fit(
            &data.as_view(),
            &regression,
            RandomForestRegressorParams::default()
                .with_n_estimators(1)
                .with_bootstrap(false),
        )
        .unwrap();
        for scorer in [RegressionScorer::MeanSquaredError, RegressionScorer::R2] {
            assert_eq!(
                score_regressor(&regressor, &data.as_view(), &regression, scorer),
                score_regressor_with(
                    &regressor,
                    &data.as_view(),
                    &regression,
                    scorer,
                    &mut workspace
                )
            );
        }
    }

    #[test]
    fn a_caller_defined_score_is_consumed_exactly_like_a_built_in_one() {
        let data = matrix();
        let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let model = classifier(&binary);
        let labels = model.predict(&data.as_view()).unwrap();
        let expected = ConfusionMatrix::new(binary.as_slice(), &labels)
            .unwrap()
            .f1(Average::Macro)
            .unwrap();

        assert_eq!(
            score_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &binary,
                MacroF1
            ),
            Ok(expected)
        );
        // A reference to a score is a score, so it need not be moved.
        let scorer = MacroF1;
        assert_eq!(
            score_classifier_with(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &binary,
                &scorer,
                &mut ScoringWorkspace::new()
            ),
            Ok(expected)
        );
    }

    #[test]
    fn orientation_is_declared_by_the_score_itself() {
        for scorer in [
            ClassificationScorer::Accuracy,
            ClassificationScorer::Precision,
            ClassificationScorer::Recall,
            ClassificationScorer::F1,
            ClassificationScorer::RocAuc,
        ] {
            assert!(scorer.greater_is_better(), "{scorer:?}");
        }
        for scorer in [ClassificationScorer::Brier, ClassificationScorer::LogLoss] {
            assert!(!scorer.greater_is_better(), "{scorer:?}");
        }
        assert!(RegressionScorer::R2.greater_is_better());
        for scorer in [
            RegressionScorer::MeanAbsoluteError,
            RegressionScorer::MeanSquaredError,
            RegressionScorer::RootMeanSquaredError,
        ] {
            assert!(!scorer.greater_is_better(), "{scorer:?}");
        }
    }

    #[test]
    fn a_score_given_the_wrong_output_reports_it_instead_of_guessing() {
        let labels = [0_u8, 1];
        let probabilities = [0.2_f32, 0.9];
        assert_eq!(
            ClassificationScorer::Accuracy.score(
                &labels,
                ClassifierOutput::PositiveProbabilities(&probabilities)
            ),
            Err(ScoringError::UnsupportedOutput {
                required: ClassifierOutputKind::Labels,
                supplied: ClassifierOutputKind::PositiveProbabilities,
            })
        );
        assert_eq!(
            ClassificationScorer::RocAuc.score(&labels, ClassifierOutput::Labels(&labels)),
            Err(ScoringError::UnsupportedOutput {
                required: ClassifierOutputKind::PositiveProbabilities,
                supplied: ClassifierOutputKind::Labels,
            })
        );
        assert_eq!(
            ClassifierOutput::Labels(&labels).kind(),
            ClassifierOutputKind::Labels
        );
        let classes = [0_u8, 1];
        assert_eq!(
            ClassificationScorer::MulticlassLogLoss
                .score(&labels, ClassifierOutput::Labels(&labels)),
            Err(ScoringError::UnsupportedOutput {
                required: ClassifierOutputKind::ProbabilityMatrix,
                supplied: ClassifierOutputKind::Labels,
            })
        );
        assert_eq!(
            ClassificationScorer::Brier.score(
                &labels,
                ClassifierOutput::ProbabilityMatrix {
                    classes: &classes,
                    probabilities: &[0.8, 0.2, 0.1, 0.9],
                }
            ),
            Err(ScoringError::UnsupportedOutput {
                required: ClassifierOutputKind::PositiveProbabilities,
                supplied: ClassifierOutputKind::ProbabilityMatrix,
            })
        );
    }

    fn multiclass_forest(targets: &ClassTargets) -> RandomForestClassifier {
        RandomForestClassifier::fit_multiclass(
            &matrix().as_view(),
            targets,
            RandomForestClassifierParams::default()
                .with_n_estimators(3)
                .with_bootstrap(false)
                .with_max_features(MaxFeatures::All),
        )
        .unwrap()
    }

    #[test]
    fn probability_matrix_scores_read_every_column_for_any_class_set() {
        let data = matrix();
        // Deliberately non-contiguous, non-zero-based labels.
        let targets = ClassTargets::new(vec![3, 3, 7, 10]).unwrap();
        let model = multiclass_forest(&targets);
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        assert_eq!(model.classes(), &[3, 7, 10]);

        for (scorer, expected) in [
            (
                ClassificationScorer::MulticlassLogLoss,
                multiclass_log_loss(targets.as_slice(), model.classes(), &probabilities),
            ),
            (
                ClassificationScorer::MulticlassBrier,
                multiclass_brier_score(targets.as_slice(), model.classes(), &probabilities),
            ),
        ] {
            assert_eq!(
                score_multiclass_classifier(
                    ScorableClassifier::probabilistic(&model),
                    &data.as_view(),
                    &targets,
                    scorer
                ),
                expected.map_err(ScoringError::Metric),
                "{scorer:?}"
            );
        }

        // The positive-probability layouts still refuse a wider class set,
        // rather than reinterpreting one of its columns.
        assert_eq!(
            score_multiclass_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &targets,
                ClassificationScorer::Brier
            ),
            Err(ScoringError::UnsupportedClasses)
        );
    }

    #[test]
    fn probability_matrix_scores_reach_binary_targets_through_the_same_path() {
        let data = matrix();
        let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let model = classifier(&binary);
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        let positive = model.predict_class_proba(&data.as_view(), 1).unwrap();

        // Cross-entropy reads only the true class's column, so the multiclass
        // and binary log losses agree on two classes.
        let multiclass = score_classifier(
            ScorableClassifier::probabilistic(&model),
            &data.as_view(),
            &binary,
            ClassificationScorer::MulticlassLogLoss,
        )
        .unwrap();
        let binary_log_loss = score_classifier(
            ScorableClassifier::probabilistic(&model),
            &data.as_view(),
            &binary,
            ClassificationScorer::LogLoss,
        )
        .unwrap();
        assert!((multiclass - binary_log_loss).abs() <= 1.0e-12);
        assert_eq!(
            multiclass,
            multiclass_log_loss(binary.as_slice(), model.classes(), &probabilities).unwrap()
        );

        // The Brier scores deliberately differ by exactly a factor of two.
        let multiclass_brier = score_classifier(
            ScorableClassifier::probabilistic(&model),
            &data.as_view(),
            &binary,
            ClassificationScorer::MulticlassBrier,
        )
        .unwrap();
        let binary_brier = brier_score(binary.as_slice(), &positive).unwrap();
        assert!((multiclass_brier - 2.0 * binary_brier).abs() <= 1.0e-12);
    }

    #[test]
    fn the_probability_matrix_workspace_is_reused_and_agrees_with_the_allocating_form() {
        let data = matrix();
        let targets = ClassTargets::new(vec![3, 3, 7, 10]).unwrap();
        let model = multiclass_forest(&targets);
        let mut workspace = ScoringWorkspace::new();
        for scorer in [
            ClassificationScorer::MulticlassLogLoss,
            ClassificationScorer::MulticlassBrier,
        ] {
            let allocating = score_multiclass_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &targets,
                scorer,
            );
            let reused = score_multiclass_classifier_with(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &targets,
                scorer,
                &mut workspace,
            );
            assert_eq!(allocating, reused, "{scorer:?}");
            // One workspace serves a narrower output kind straight afterwards.
            assert_eq!(
                reused,
                score_multiclass_classifier_with(
                    ScorableClassifier::probabilistic(&model),
                    &data.as_view(),
                    &targets,
                    scorer,
                    &mut workspace
                )
            );
        }
        assert_eq!(
            score_multiclass_classifier_with(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &targets,
                ClassificationScorer::Accuracy,
                &mut workspace
            ),
            score_multiclass_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &targets,
                ClassificationScorer::Accuracy
            )
        );
    }

    #[test]
    fn multiclass_scoring_validates_targets_before_prediction() {
        let data = matrix();
        let fitted = ClassTargets::new(vec![3, 3, 7, 10]).unwrap();
        let model = multiclass_forest(&fitted);
        let short = ClassTargets::new(vec![3, 7]).unwrap();
        assert_eq!(
            score_multiclass_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &short,
                ClassificationScorer::MulticlassLogLoss
            ),
            Err(ScoringError::TargetLength {
                rows: 4,
                targets: 2,
            })
        );

        // A label with no probability column is reported, never scored.
        let unknown = ClassTargets::new(vec![3, 3, 7, 11]).unwrap();
        assert_eq!(
            score_multiclass_classifier(
                ScorableClassifier::probabilistic(&model),
                &data.as_view(),
                &unknown,
                ClassificationScorer::MulticlassLogLoss
            ),
            Err(ScoringError::Metric(MetricError::UnknownClass {
                index: 3,
                value: 11,
            }))
        );
    }
}
