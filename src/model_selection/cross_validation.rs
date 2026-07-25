use std::error::Error;
use std::fmt;

use crate::api::{Classifier, ModelError, ProbabilisticClassifier, Regressor};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};
use crate::metrics::MetricError;

use super::{
    ClassificationScore, ClassifierOutputKind, RegressionScore, ScorableClassifier, ScoringError,
    ScoringWorkspace, Split, score_classifier_with, score_regressor_with,
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
pub fn cross_validate_classifier<M, I, F, S>(
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    splits: I,
    scorer: S,
    fit: F,
) -> Result<CrossValidationResult, CrossValidationError>
where
    M: ProbabilisticClassifier,
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &BinaryTargets) -> Result<M, ModelError>,
    S: ClassificationScore,
{
    cross_validate_folds(data, targets, splits, scorer, fit, |model| {
        ScorableClassifier::probabilistic(model)
    })
}

/// Fits and scores one label-producing classifier per supplied split.
///
/// This is [`cross_validate_classifier`] for a classifier that produces no
/// probabilities. Only label metrics apply: a probability metric returns
/// [`ScoringError::UnsupportedOutput`] through
/// [`CrossValidationError::Scoring`] rather than a substituted value.
pub fn cross_validate_classifier_labels<M, I, F, S>(
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    splits: I,
    scorer: S,
    fit: F,
) -> Result<CrossValidationResult, CrossValidationError>
where
    M: Classifier,
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &BinaryTargets) -> Result<M, ModelError>,
    S: ClassificationScore,
{
    cross_validate_folds(data, targets, splits, scorer, fit, |model| {
        ScorableClassifier::labels_only(model)
    })
}

/// The one cross-validation loop, over whichever view the caller's model has.
fn cross_validate_folds<M, I, F, S, V>(
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    splits: I,
    scorer: S,
    mut fit: F,
    view: V,
) -> Result<CrossValidationResult, CrossValidationError>
where
    I: IntoIterator<Item = Split>,
    F: FnMut(&MatrixView<'_>, &BinaryTargets) -> Result<M, ModelError>,
    S: ClassificationScore,
    V: for<'m> Fn(&'m M) -> ScorableClassifier<'m>,
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
        let score =
            score_classifier_with(view(&model), &test, &test_targets, &scorer, &mut workspace)
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
    use crate::data::DenseMatrix;
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
