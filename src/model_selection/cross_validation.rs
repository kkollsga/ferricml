use std::error::Error;
use std::fmt;

use crate::api::{ModelError, Regressor};
use crate::data::{MatrixView, RegressionTargets};
use crate::metrics::MetricError;

use super::scoring::score_labelled;
use super::{
    ClassificationScore, ClassificationTargets, ClassifierOutputKind, RegressionScore,
    ScorableClassifier, ScoringError, ScoringWorkspace, Split, score_regressor_with,
};

/// Errors produced while running serial cross-validation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CrossValidationError {
    /// Target count did not match the source row count.
    TargetLength {
        /// Source rows.
        rows: usize,
        /// Supplied targets.
        targets: usize,
    },
    /// The split iterator produced no folds.
    NoSplits,
    /// A fold was built for a different sample count.
    SplitSampleCount {
        /// Zero-based fold index.
        fold: usize,
        /// Required source rows.
        expected: usize,
        /// Rows covered by the split.
        actual: usize,
    },
    /// Fitting failed for one fold.
    Fit {
        /// Zero-based fold index.
        fold: usize,
        /// Original estimator error.
        source: ModelError,
    },
    /// Batch prediction failed for one fold.
    Prediction {
        /// Zero-based fold index.
        fold: usize,
        /// Original estimator error.
        source: ModelError,
    },
    /// A metric was undefined or rejected one fold's predictions.
    Metric {
        /// Zero-based fold index.
        fold: usize,
        /// Original metric error.
        source: MetricError,
    },
    /// A classifier exposed an unsupported class layout in one fold.
    UnsupportedClasses {
        /// Zero-based fold index.
        fold: usize,
    },
    /// A score received a batch output other than the one it declared.
    UnsupportedOutput {
        /// Zero-based fold index.
        fold: usize,
        /// Output the score declared it reads.
        required: ClassifierOutputKind,
        /// Output it was given.
        supplied: ClassifierOutputKind,
    },
}

impl fmt::Display for CrossValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetLength { rows, targets } => {
                write!(
                    f,
                    "target length {targets} does not match {rows} source rows"
                )
            }
            Self::NoSplits => f.write_str("cross-validation requires at least one split"),
            Self::SplitSampleCount {
                fold,
                expected,
                actual,
            } => write!(
                f,
                "fold {fold} covers {actual} samples, expected {expected}"
            ),
            Self::Fit { fold, source } => write!(f, "fold {fold} fit failed: {source}"),
            Self::Prediction { fold, source } => {
                write!(f, "fold {fold} prediction failed: {source}")
            }
            Self::Metric { fold, source } => write!(f, "fold {fold} metric failed: {source}"),
            Self::UnsupportedClasses { fold } => {
                write!(f, "fold {fold} exposed unsupported classifier classes")
            }
            Self::UnsupportedOutput {
                fold,
                required,
                supplied,
            } => write!(
                f,
                "fold {fold} score reads {required:?} but was given {supplied:?}"
            ),
        }
    }
}

impl Error for CrossValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Fit { source, .. } | Self::Prediction { source, .. } => Some(source),
            Self::Metric { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Ordered fold scores from one deterministic cross-validation run.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossValidationResult {
    scores: Vec<f64>,
}

impl CrossValidationResult {
    /// Raw scores in split iteration order.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Number of evaluated folds.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns whether the result contains no scores.
    ///
    /// Successful cross-validation results are never empty; this method keeps
    /// the collection-style interface explicit.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Fixed-order arithmetic mean of all fold scores.
    pub fn mean(&self) -> f64 {
        self.scores.iter().sum::<f64>() / self.scores.len() as f64
    }

    /// Fixed-order population standard deviation of all fold scores.
    pub fn population_standard_deviation(&self) -> f64 {
        let mean = self.mean();
        let variance = self
            .scores
            .iter()
            .map(|score| {
                let delta = score - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.scores.len() as f64;
        variance.sqrt()
    }
}

/// Fits and scores one classifier per supplied split, serially and in order.
///
/// This is the crate's only classifier cross-validation entry point, over
/// **any** [`ClassificationTargets`] vocabulary. The loop branches on
/// classifier-versus-regressor and on nothing else: how many classes the labels
/// name is a property of the *metric*, which already declares what it reads
/// through
/// [`ClassificationScore::output_kind`](super::ClassificationScore::output_kind),
/// so [`ClassTargets`](crate::data::ClassTargets) folds here exactly as
/// [`BinaryTargets`](crate::data::BinaryTargets) does. A score reading
/// [`ClassifierOutputKind::ProbabilityMatrix`] works for any observed class set;
/// the binary positive-probability layouts still refuse a wider one with
/// [`CrossValidationError::UnsupportedClasses`] rather than reinterpreting a
/// column.
///
/// `view` says how each fold's fitted model presents itself to the scoring
/// layer: [`ScorableClassifier::probabilistic`] for a model that produces
/// probabilities, [`ScorableClassifier::labels_only`] for one that does not.
/// That is the same mechanism [`score_classifier`](super::score_classifier) and
/// permutation importance take, so `model_selection` answers "does this
/// classifier give probabilities?" in exactly one way — with a value, rather
/// than with a second entry point per answer. A probability metric applied to a
/// labels-only view is [`CrossValidationError::UnsupportedOutput`], never a
/// substituted value.
///
/// The view cannot simply be a `ScorableClassifier` argument: the fitting
/// closure returns an owned model *per fold*, and a `ScorableClassifier`
/// borrows the model it wraps, so the borrow has to be taken inside the loop.
/// Passing the constructor is what lets it be taken there.
///
/// The two choices are therefore independent: the target vocabulary is a type
/// parameter, the model's scoring capability is a value, and neither multiplies
/// the other into extra entry points.
///
/// ```
/// use ferricml::data::{BinaryTargets, DenseMatrix};
/// use ferricml::dummy::{DummyClassifier, DummyClassifierParams};
/// use ferricml::model_selection::{
///     ClassificationScorer, KFold, ScorableClassifier, cross_validate_classifier,
/// };
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
/// let targets = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;
///
/// let folds = cross_validate_classifier(
///     &data.as_view(),
///     &targets,
///     KFold::new(3).split(data.rows())?,
///     ClassificationScorer::Accuracy,
///     |train, train_targets| {
///         DummyClassifier::fit(train, train_targets, DummyClassifierParams)
///     },
///     |model| ScorableClassifier::probabilistic(model),
/// )?;
/// assert_eq!(folds.len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn cross_validate_classifier<M, T, I, F, S, V>(
    data: &MatrixView<'_>,
    targets: &T,
    splits: I,
    scorer: S,
    mut fit: F,
    view: V,
) -> Result<CrossValidationResult, CrossValidationError>
where
    T: ClassificationTargets,
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &T) -> Result<M, ModelError>,
    S: ClassificationScore,
    V: for<'m> Fn(&'m M) -> ScorableClassifier<'m>,
{
    validate_target_length(data.rows(), targets.as_slice().len())?;
    let mut feature_buffer = Vec::new();
    let mut workspace = ScoringWorkspace::new();
    let mut scores = Vec::new();
    for (fold, split) in splits.into_iter().enumerate() {
        validate_split_sample_count(fold, data.rows(), &split)?;
        let model = {
            let train = gather_rows(data, split.train_indices(), &mut feature_buffer);
            let train_targets = targets
                .select(split.train_indices())
                .expect("validated split contains non-empty in-bounds indices");
            fit(&train, &train_targets)
                .map_err(|source| CrossValidationError::Fit { fold, source })?
        };
        let test = gather_rows(data, split.test_indices(), &mut feature_buffer);
        let test_targets = targets
            .select(split.test_indices())
            .expect("validated split contains non-empty in-bounds indices");
        let score = score_labelled(
            view(&model),
            &test,
            test_targets.as_slice(),
            &scorer,
            &mut workspace,
        )
        .map_err(|error| map_scoring_error(fold, error))?;
        scores.push(score);
    }
    finish(scores)
}

/// Fits and scores one regressor per supplied split, serially and in order.
pub fn cross_validate_regressor<M, I, F, S>(
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    splits: I,
    scorer: S,
    mut fit: F,
) -> Result<CrossValidationResult, CrossValidationError>
where
    M: Regressor,
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &RegressionTargets) -> Result<M, ModelError>,
    S: RegressionScore,
{
    validate_target_length(data.rows(), targets.len())?;
    let mut feature_buffer = Vec::new();
    let mut workspace = ScoringWorkspace::new();
    let mut scores = Vec::new();
    for (fold, split) in splits.into_iter().enumerate() {
        validate_split_sample_count(fold, data.rows(), &split)?;
        let model = {
            let train = gather_rows(data, split.train_indices(), &mut feature_buffer);
            let train_targets = targets
                .select(split.train_indices())
                .expect("validated split contains non-empty in-bounds indices");
            fit(&train, &train_targets)
                .map_err(|source| CrossValidationError::Fit { fold, source })?
        };
        let test = gather_rows(data, split.test_indices(), &mut feature_buffer);
        let test_targets = targets
            .select(split.test_indices())
            .expect("validated split contains non-empty in-bounds indices");
        let score = score_regressor_with(&model, &test, &test_targets, &scorer, &mut workspace)
            .map_err(|error| map_scoring_error(fold, error))?;
        scores.push(score);
    }
    finish(scores)
}

pub(super) fn validate_target_length(
    rows: usize,
    targets: usize,
) -> Result<(), CrossValidationError> {
    if rows != targets {
        return Err(CrossValidationError::TargetLength { rows, targets });
    }
    Ok(())
}

pub(super) fn validate_split_sample_count(
    fold: usize,
    expected: usize,
    split: &Split,
) -> Result<(), CrossValidationError> {
    let actual = split.sample_count();
    if actual != expected {
        return Err(CrossValidationError::SplitSampleCount {
            fold,
            expected,
            actual,
        });
    }
    Ok(())
}

fn gather_rows<'buffer>(
    data: &MatrixView<'_>,
    indices: &[usize],
    buffer: &'buffer mut Vec<f32>,
) -> MatrixView<'buffer> {
    let output_len = indices
        .len()
        .checked_mul(data.columns())
        .expect("a subset of a validated matrix cannot overflow its source shape");
    buffer.clear();
    buffer.reserve(output_len);
    for &index in indices {
        buffer.extend_from_slice(data.row(index).expect("split index was validated"));
    }
    MatrixView::new(buffer, indices.len(), data.columns())
        .expect("selected rows preserve validated matrix invariants")
}

fn map_scoring_error(fold: usize, error: ScoringError) -> CrossValidationError {
    match error {
        ScoringError::Prediction(source) => CrossValidationError::Prediction { fold, source },
        ScoringError::Metric(source) => CrossValidationError::Metric { fold, source },
        ScoringError::UnsupportedClasses => CrossValidationError::UnsupportedClasses { fold },
        ScoringError::UnsupportedOutput { required, supplied } => {
            CrossValidationError::UnsupportedOutput {
                fold,
                required,
                supplied,
            }
        }
        ScoringError::TargetLength { .. } => {
            unreachable!("selected rows and targets have identical validated indices")
        }
    }
}

fn finish(scores: Vec<f64>) -> Result<CrossValidationResult, CrossValidationError> {
    if scores.is_empty() {
        return Err(CrossValidationError::NoSplits);
    }
    Ok(CrossValidationResult { scores })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Estimator;
    use crate::data::{BinaryTargets, ClassTargets, DenseMatrix};
    use crate::linear_model::{LogisticRegression, LogisticRegressionParams, Ridge, RidgeParams};
    use crate::model_selection::{
        ClassificationScorer, KFold, RegressionScorer, StratifiedKFold, TimeSeriesSplit,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    fn data() -> DenseMatrix {
        DenseMatrix::new(
            (0..12)
                .flat_map(|row| [row as f32, (row * row) as f32])
                .collect(),
            12,
            2,
        )
        .unwrap()
    }

    #[test]
    fn real_classifier_and_regressor_cv_are_repeatable() {
        let data = data();
        let binary = BinaryTargets::new((0..12).map(|row| u8::from(row >= 6)).collect()).unwrap();
        let classifier_splits = StratifiedKFold::new(3)
            .with_shuffle(true)
            .with_random_state(17)
            .split(binary.as_slice())
            .unwrap();
        let classifier = cross_validate_classifier(
            &data.as_view(),
            &binary,
            classifier_splits,
            ClassificationScorer::Accuracy,
            |train, targets| {
                LogisticRegression::fit(train, targets, LogisticRegressionParams::default())
            },
            |model| ScorableClassifier::probabilistic(model),
        )
        .unwrap();
        assert_eq!(classifier.len(), 3);
        assert!(!classifier.is_empty());
        assert!(
            classifier
                .scores()
                .iter()
                .all(|score| (0.0..=1.0).contains(score))
        );
        assert!(classifier.mean().is_finite());
        assert!(classifier.population_standard_deviation().is_finite());

        let regression =
            RegressionTargets::new((0..12).map(|row| (row * row) as f32).collect()).unwrap();
        let run = || {
            cross_validate_regressor(
                &data.as_view(),
                &regression,
                KFold::new(4)
                    .with_shuffle(true)
                    .with_random_state(23)
                    .split(data.rows())
                    .unwrap(),
                RegressionScorer::MeanSquaredError,
                |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
            )
            .unwrap()
        };
        assert_eq!(run(), run());
    }

    /// Twelve rows over three deliberately non-contiguous, non-zero-based
    /// labels, four of each.
    fn multiclass_targets() -> ClassTargets {
        ClassTargets::new((0..12).map(|row| [3_u8, 7, 10][row % 3]).collect()).unwrap()
    }

    /// The arity unification: an arbitrary observed class set folds through the
    /// *same* entry point binary targets use, and what is admissible is decided
    /// by the metric rather than by which function was called.
    #[test]
    fn an_arbitrary_class_set_cross_validates_through_the_same_entry_point() {
        let data = data();
        let targets = multiclass_targets();
        assert_eq!(targets.classes(), &[3, 7, 10]);
        let splits = StratifiedKFold::new(2)
            .split(targets.as_slice())
            .unwrap()
            .collect::<Vec<_>>();
        let params = LogisticRegressionParams::default().with_max_iter(200);

        let folds = cross_validate_classifier(
            &data.as_view(),
            &targets,
            splits.iter().cloned(),
            ClassificationScorer::MulticlassLogLoss,
            |train, train_targets| {
                LogisticRegression::fit_multiclass(train, train_targets, params.clone())
            },
            |model| ScorableClassifier::probabilistic(model),
        )
        .unwrap();
        assert_eq!(folds.len(), 2);
        assert!(folds.scores().iter().all(|score| score.is_finite()));

        // Every fold score equals scoring that fold directly, so the wider
        // vocabulary reaches the model through the one scoring implementation
        // rather than through a second loop.
        for (fold, split) in splits.iter().enumerate() {
            let train = data.select_rows(split.train_indices()).unwrap();
            let train_targets = targets.select(split.train_indices()).unwrap();
            let model = LogisticRegression::fit_multiclass(
                &train.as_view(),
                &train_targets,
                params.clone(),
            )
            .unwrap();
            let test = data.select_rows(split.test_indices()).unwrap();
            let test_targets = targets.select(split.test_indices()).unwrap();
            assert_eq!(
                Ok(folds.scores()[fold]),
                super::super::score_multiclass_classifier(
                    ScorableClassifier::probabilistic(&model),
                    &test.as_view(),
                    &test_targets,
                    ClassificationScorer::MulticlassLogLoss
                ),
                "fold {fold}"
            );
        }

        // And the entry point did not simply become permissive: a binary
        // positive-probability metric is still refused on a three-class model,
        // with the fold that refused it. Arity lives in the metric.
        assert_eq!(
            cross_validate_classifier(
                &data.as_view(),
                &targets,
                splits,
                ClassificationScorer::Brier,
                |train, train_targets| {
                    LogisticRegression::fit_multiclass(train, train_targets, params.clone())
                },
                |model| ScorableClassifier::probabilistic(model),
            ),
            Err(CrossValidationError::UnsupportedClasses { fold: 0 })
        );
    }

    /// The setup guards are the ones finding 5 said a hand-rolled fold loop
    /// gives up, so the wider vocabulary owes proof that it reaches them.
    #[test]
    fn the_wider_vocabulary_reaches_the_same_setup_guards() {
        let data = data();
        let targets = multiclass_targets();
        let params = LogisticRegressionParams::default();
        let run = |targets: &ClassTargets, splits: Vec<Split>| {
            cross_validate_classifier(
                &data.as_view(),
                targets,
                splits,
                ClassificationScorer::MulticlassLogLoss,
                |train, train_targets| {
                    LogisticRegression::fit_multiclass(train, train_targets, params.clone())
                },
                |model| ScorableClassifier::probabilistic(model),
            )
        };

        assert_eq!(
            run(&ClassTargets::new(vec![3, 7]).unwrap(), Vec::new()),
            Err(CrossValidationError::TargetLength {
                rows: 12,
                targets: 2,
            })
        );
        assert_eq!(
            &run(&targets, Vec::new()),
            &Err(CrossValidationError::NoSplits)
        );
        assert_eq!(
            run(
                &targets,
                vec![Split::new(4, vec![0, 1], vec![2, 3]).unwrap()]
            ),
            Err(CrossValidationError::SplitSampleCount {
                fold: 0,
                expected: 12,
                actual: 4,
            })
        );
    }

    #[test]
    fn fit_runs_once_per_fold_and_never_observes_held_out_rows() {
        let data = data();
        let targets = RegressionTargets::new((0..12).map(|row| row as f32).collect()).unwrap();
        let splits = KFold::new(3).split(12).unwrap().collect::<Vec<_>>();
        let calls = Rc::new(Cell::new(0_usize));
        let fit_calls = Rc::clone(&calls);
        let result = cross_validate_regressor(
            &data.as_view(),
            &targets,
            splits.clone(),
            RegressionScorer::MeanAbsoluteError,
            move |train, train_targets| {
                let fold = fit_calls.get();
                fit_calls.set(fold + 1);
                for &held_out in splits[fold].test_indices() {
                    assert!(!train.iter_rows().any(|row| row[0] == held_out as f32));
                }
                assert!(
                    train
                        .iter_rows()
                        .zip(train_targets.as_slice())
                        .all(|(row, &target)| row[0] == target)
                );
                Ridge::fit(train, train_targets, RidgeParams::default())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 3);
        assert_eq!(result.len(), 3);
    }

    /// A score FerricML does not enumerate, to prove cross-validation reaches
    /// the scorer through the trait rather than through a private table.
    struct HalvedMeanSquaredError;

    impl RegressionScore for HalvedMeanSquaredError {
        fn greater_is_better(&self) -> bool {
            false
        }

        fn score(&self, expected: &[f32], predicted: &[f32]) -> Result<f64, ScoringError> {
            crate::metrics::mean_squared_error(expected, predicted)
                .map(|value| value / 2.0)
                .map_err(ScoringError::Metric)
        }
    }

    #[test]
    fn each_fold_score_equals_scoring_that_fold_directly() {
        let data = data();
        let targets =
            RegressionTargets::new((0..12).map(|row| (row * row) as f32).collect()).unwrap();
        let splits = KFold::new(3)
            .with_shuffle(true)
            .with_random_state(5)
            .split(12)
            .unwrap()
            .collect::<Vec<_>>();

        for scorer in [RegressionScorer::MeanSquaredError, RegressionScorer::R2] {
            let folds = cross_validate_regressor(
                &data.as_view(),
                &targets,
                splits.clone(),
                scorer,
                |train, train_targets| Ridge::fit(train, train_targets, RidgeParams::default()),
            )
            .unwrap();

            for (fold, split) in splits.iter().enumerate() {
                let train = data.select_rows(split.train_indices()).unwrap();
                let train_targets = targets.select(split.train_indices()).unwrap();
                let model =
                    Ridge::fit(&train.as_view(), &train_targets, RidgeParams::default()).unwrap();
                let test = data.select_rows(split.test_indices()).unwrap();
                let test_targets = targets.select(split.test_indices()).unwrap();
                assert_eq!(
                    Ok(folds.scores()[fold]),
                    super::super::score_regressor(&model, &test.as_view(), &test_targets, scorer),
                    "{scorer:?} fold {fold}"
                );
            }
        }

        // The same holds for a score the crate does not enumerate.
        let custom = cross_validate_regressor(
            &data.as_view(),
            &targets,
            splits.clone(),
            HalvedMeanSquaredError,
            |train, train_targets| Ridge::fit(train, train_targets, RidgeParams::default()),
        )
        .unwrap();
        let built_in = cross_validate_regressor(
            &data.as_view(),
            &targets,
            splits,
            RegressionScorer::MeanSquaredError,
            |train, train_targets| Ridge::fit(train, train_targets, RidgeParams::default()),
        )
        .unwrap();
        for (custom, built_in) in custom.scores().iter().zip(built_in.scores()) {
            assert_eq!(*custom, built_in / 2.0);
        }
    }

    #[test]
    fn a_partial_time_series_split_never_lets_a_fold_train_on_its_future() {
        let data = data();
        let targets = RegressionTargets::new((0..12).map(|row| row as f32).collect()).unwrap();
        let splits = TimeSeriesSplit::new(3)
            .split(12)
            .unwrap()
            .collect::<Vec<_>>();
        assert!(splits.iter().any(|split| split.covered_samples() < 12));

        let calls = Rc::new(Cell::new(0_usize));
        let fit_calls = Rc::clone(&calls);
        let result = cross_validate_regressor(
            &data.as_view(),
            &targets,
            splits.clone(),
            RegressionScorer::MeanAbsoluteError,
            move |train, train_targets| {
                let fold = fit_calls.get();
                fit_calls.set(fold + 1);
                let first_test = splits[fold].test_indices()[0] as f32;
                assert!(
                    train.iter_rows().all(|row| row[0] < first_test),
                    "fold {fold} trained on a row at or after {first_test}"
                );
                Ridge::fit(train, train_targets, RidgeParams::default())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn validation_and_fit_errors_identify_the_exact_fold() {
        let data = data();
        let targets = RegressionTargets::new((0..12).map(|row| row as f32).collect()).unwrap();
        assert_eq!(
            cross_validate_regressor::<Ridge, _, _, _>(
                &data.as_view(),
                &RegressionTargets::new(vec![0.0]).unwrap(),
                KFold::new(2).split(12).unwrap(),
                RegressionScorer::MeanSquaredError,
                |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
            ),
            Err(CrossValidationError::TargetLength {
                rows: 12,
                targets: 1,
            })
        );
        assert_eq!(
            cross_validate_regressor::<Ridge, _, _, _>(
                &data.as_view(),
                &targets,
                std::iter::empty(),
                RegressionScorer::MeanSquaredError,
                |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
            ),
            Err(CrossValidationError::NoSplits)
        );
        assert_eq!(
            cross_validate_regressor::<Ridge, _, _, _>(
                &data.as_view(),
                &targets,
                vec![Split::new(4, vec![0, 1], vec![2, 3]).unwrap()],
                RegressionScorer::MeanSquaredError,
                |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
            ),
            Err(CrossValidationError::SplitSampleCount {
                fold: 0,
                expected: 12,
                actual: 4,
            })
        );

        let calls = Cell::new(0_usize);
        assert_eq!(
            cross_validate_regressor::<Ridge, _, _, _>(
                &data.as_view(),
                &targets,
                KFold::new(3).split(12).unwrap(),
                RegressionScorer::MeanSquaredError,
                |train, targets| {
                    let fold = calls.get();
                    calls.set(fold + 1);
                    if fold == 1 {
                        Err(ModelError::LinearSolveFailed)
                    } else {
                        Ridge::fit(train, targets, RidgeParams::default())
                    }
                },
            ),
            Err(CrossValidationError::Fit {
                fold: 1,
                source: ModelError::LinearSolveFailed,
            })
        );
    }

    #[derive(Clone, Debug)]
    struct FailingRegressor {
        features: usize,
    }

    impl Estimator for FailingRegressor {
        fn n_features_in(&self) -> usize {
            self.features
        }
    }

    impl Regressor for FailingRegressor {
        fn predict_into(
            &self,
            _data: &MatrixView<'_>,
            _output: &mut [f32],
        ) -> Result<(), ModelError> {
            Err(ModelError::NonFinitePrediction { row: 0 })
        }
    }

    #[test]
    fn prediction_and_metric_errors_remain_typed_and_fold_attributed() {
        let data = data();
        let targets = RegressionTargets::new((0..12).map(|row| row as f32).collect()).unwrap();
        assert_eq!(
            cross_validate_regressor(
                &data.as_view(),
                &targets,
                KFold::new(3).split(12).unwrap(),
                RegressionScorer::MeanSquaredError,
                |train, _| Ok(FailingRegressor {
                    features: train.columns()
                }),
            ),
            Err(CrossValidationError::Prediction {
                fold: 0,
                source: ModelError::NonFinitePrediction { row: 0 },
            })
        );

        let constant_targets = RegressionTargets::new(vec![1.0; 12]).unwrap();
        assert_eq!(
            cross_validate_regressor(
                &data.as_view(),
                &constant_targets,
                KFold::new(3).split(12).unwrap(),
                RegressionScorer::R2,
                |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
            ),
            Err(CrossValidationError::Metric {
                fold: 0,
                source: MetricError::Undefined,
            })
        );
    }
}
