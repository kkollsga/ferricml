use std::error::Error;
use std::fmt;

use crate::api::{Classifier, ModelError, Regressor};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};
use crate::metrics::{
    MetricError, accuracy_score, brier_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    root_mean_squared_error,
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

/// Scores one fitted classifier through a single batch prediction call.
pub fn score_classifier(
    classifier: &dyn Classifier,
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
    scorer: ClassificationScorer,
) -> Result<f64, ScoringError> {
    validate_target_length(data.rows(), targets.len())?;
    match scorer {
        ClassificationScorer::Accuracy => {
            let predicted = classifier.predict(data).map_err(ScoringError::Prediction)?;
            accuracy_score(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        ClassificationScorer::Precision => {
            let predicted = classifier.predict(data).map_err(ScoringError::Prediction)?;
            precision_score(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        ClassificationScorer::Recall => {
            let predicted = classifier.predict(data).map_err(ScoringError::Prediction)?;
            recall_score(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        ClassificationScorer::F1 => {
            let predicted = classifier.predict(data).map_err(ScoringError::Prediction)?;
            f1_score(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        ClassificationScorer::Brier => {
            let probabilities = positive_probabilities(classifier, data)?;
            brier_score(targets.as_slice(), &probabilities).map_err(ScoringError::Metric)
        }
        ClassificationScorer::LogLoss => {
            let probabilities = positive_probabilities(classifier, data)?;
            log_loss(targets.as_slice(), &probabilities).map_err(ScoringError::Metric)
        }
        ClassificationScorer::RocAuc => {
            let probabilities = positive_probabilities(classifier, data)?;
            roc_auc_score(targets.as_slice(), &probabilities).map_err(ScoringError::Metric)
        }
    }
}

/// Scores one fitted regressor through a single batch prediction call.
pub fn score_regressor(
    regressor: &dyn Regressor,
    data: &MatrixView<'_>,
    targets: &RegressionTargets,
    scorer: RegressionScorer,
) -> Result<f64, ScoringError> {
    validate_target_length(data.rows(), targets.len())?;
    let predicted = regressor.predict(data).map_err(ScoringError::Prediction)?;
    match scorer {
        RegressionScorer::MeanAbsoluteError => {
            mean_absolute_error(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        RegressionScorer::MeanSquaredError => {
            mean_squared_error(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        RegressionScorer::RootMeanSquaredError => {
            root_mean_squared_error(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
        RegressionScorer::R2 => {
            r2_score(targets.as_slice(), &predicted).map_err(ScoringError::Metric)
        }
    }
}

fn validate_target_length(rows: usize, targets: usize) -> Result<(), ScoringError> {
    if rows != targets {
        return Err(ScoringError::TargetLength { rows, targets });
    }
    Ok(())
}

fn positive_probabilities(
    classifier: &dyn Classifier,
    data: &MatrixView<'_>,
) -> Result<Vec<f32>, ScoringError> {
    match classifier.classes() {
        [0] => Ok(vec![0.0; data.rows()]),
        [1] => Ok(vec![1.0; data.rows()]),
        [0, 1] => classifier
            .predict_class_proba(data, 1)
            .map_err(ScoringError::Prediction),
        _ => Err(ScoringError::UnsupportedClasses),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{AnyClassifier, AnyRegressor};
    use crate::data::DenseMatrix;
    use crate::ensemble::{
        MaxFeatures, RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
        RandomForestRegressorParams,
    };

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

    #[test]
    fn classifier_scorers_equal_direct_metric_calls_for_concrete_and_erased_models() {
        let data = matrix();
        let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let concrete = classifier(&targets);
        let labels = concrete.predict(&data.as_view()).unwrap();
        let probabilities = concrete.predict_class_proba(&data.as_view(), 1).unwrap();
        assert_eq!(
            score_classifier(
                &concrete,
                &data.as_view(),
                &targets,
                ClassificationScorer::Accuracy
            ),
            accuracy_score(targets.as_slice(), &labels).map_err(ScoringError::Metric)
        );
        assert_eq!(
            score_classifier(
                &concrete,
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
            assert!(score_classifier(&erased, &data.as_view(), &targets, scorer).is_ok());
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
                    &model,
                    &data.as_view(),
                    &targets,
                    ClassificationScorer::Brier
                ),
                Ok(expected_brier)
            );
            assert_eq!(
                score_classifier(
                    &model,
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
                &model,
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
                &model,
                &data.as_view(),
                &all_negative,
                ClassificationScorer::Recall
            ),
            Err(ScoringError::Metric(MetricError::Undefined))
        );
    }
}
