//! A classifier that produces labels but no probabilities stays usable.
//!
//! This is the shape D11 exists for, and the shape G8's `RidgeClassifier` will
//! have: a margin-based estimator whose natural output is a score, which
//! therefore implements [`Classifier`] and **not**
//! [`ProbabilisticClassifier`]. Nothing in FerricML ships in that shape yet, so
//! the property is pinned here against a probe rather than left as a promise
//! for the sprint that first needs it.
//!
//! The obligations proven are the four abstraction points that reach a
//! classifier without knowing its concrete type — scoring, cross-validation,
//! grid search, and permutation importance. Each must:
//!
//! - accept a label-only classifier on a **label** metric, and
//! - refuse a **probability** metric with a typed error naming what was
//!   required and what was supplied — never a compile error the caller cannot
//!   work around, and never a substituted value.
//!
//! # Why this is not a conformance-battery obligation
//!
//! The battery states obligations an *estimator* owes. This is a property of
//! the *scoring layer*: that it accepts a weaker classifier and reports an
//! honest error for a metric it cannot serve. The subject is the entry point,
//! not the model, so it is a direct test.

use ferricml::api::{Classifier, Estimator, ModelError};
use ferricml::data::{BinaryTargets, DenseMatrix, MatrixView};
use ferricml::inspection::{PermutationImportanceParams, permutation_importance_classifier};
use ferricml::model_selection::{
    ClassificationScorer, ClassifierOutputKind, CrossValidationError, KFold, ParameterGrid,
    ScorableClassifier, ScoringError, SearchError, cross_validate_classifier_labels,
    grid_search_classifier_labels, score_classifier,
};

/// A margin-based classifier: a threshold on one feature, and no probability.
///
/// Deliberately implements [`Classifier`] only. If it also implemented
/// [`ProbabilisticClassifier`](ferricml::api::ProbabilisticClassifier) this
/// file would prove nothing, so the absence is the point.
#[derive(Clone, Debug)]
struct MarginClassifier {
    features: usize,
    threshold: f32,
    classes: Vec<u8>,
}

impl MarginClassifier {
    fn fit(data: &MatrixView<'_>, threshold: f32) -> Result<Self, ModelError> {
        Ok(Self {
            features: data.columns(),
            threshold,
            classes: vec![0, 1],
        })
    }
}

impl Estimator for MarginClassifier {
    fn n_features_in(&self) -> usize {
        self.features
    }
}

impl Classifier for MarginClassifier {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        if data.columns() != self.features {
            return Err(ModelError::FeatureDimension {
                expected: self.features,
                actual: data.columns(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            *slot = u8::from(row[0] > self.threshold);
        }
        Ok(())
    }
}

fn fixture() -> (DenseMatrix, BinaryTargets) {
    (
        DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 8, 1)
            .expect("fixture matrix"),
        BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).expect("fixture labels"),
    )
}

/// The error a probability metric must produce, rather than a guessed value.
fn refused(error: &ScoringError, required: ClassifierOutputKind) -> bool {
    matches!(
        error,
        ScoringError::UnsupportedOutput {
            required: found,
            supplied: ClassifierOutputKind::Labels,
        } if *found == required
    )
}

#[test]
fn a_label_only_classifier_is_scorable_on_a_label_metric() {
    let (data, targets) = fixture();
    let model = MarginClassifier::fit(&data.as_view(), 3.5).unwrap();
    let view = ScorableClassifier::labels_only(&model);

    let accuracy = score_classifier(
        view,
        &data.as_view(),
        &targets,
        ClassificationScorer::Accuracy,
    )
    .expect("a label metric needs no probabilities");
    assert_eq!(accuracy, 1.0);
}

#[test]
fn a_label_only_classifier_refuses_a_probability_metric_with_a_typed_error() {
    let (data, targets) = fixture();
    let model = MarginClassifier::fit(&data.as_view(), 3.5).unwrap();
    let view = ScorableClassifier::labels_only(&model);

    for (scorer, required) in [
        (
            ClassificationScorer::LogLoss,
            ClassifierOutputKind::PositiveProbabilities,
        ),
        (
            ClassificationScorer::Brier,
            ClassifierOutputKind::PositiveProbabilities,
        ),
        (
            ClassificationScorer::RocAuc,
            ClassifierOutputKind::PositiveProbabilities,
        ),
    ] {
        let error = score_classifier(view, &data.as_view(), &targets, scorer)
            .expect_err("a probability metric cannot be served by labels alone");
        assert!(
            refused(&error, required),
            "{scorer:?} returned {error:?} instead of naming the missing output"
        );
    }
}

#[test]
fn a_label_only_classifier_cross_validates_on_a_label_metric() {
    let (data, targets) = fixture();
    let splits: Vec<_> = KFold::new(4).split(data.rows()).unwrap().collect();

    let folds = cross_validate_classifier_labels(
        &data.as_view(),
        &targets,
        splits.iter().cloned(),
        ClassificationScorer::Accuracy,
        |train, _| MarginClassifier::fit(train, 3.5),
    )
    .expect("a label metric cross-validates without probabilities");
    assert_eq!(folds.scores().len(), 4);

    let error = cross_validate_classifier_labels(
        &data.as_view(),
        &targets,
        splits,
        ClassificationScorer::LogLoss,
        |train, _| MarginClassifier::fit(train, 3.5),
    )
    .expect_err("a probability metric cannot be served by labels alone");
    assert!(
        matches!(
            &error,
            CrossValidationError::UnsupportedOutput {
                required: ClassifierOutputKind::PositiveProbabilities,
                supplied: ClassifierOutputKind::Labels,
                ..
            }
        ),
        "cross-validation returned {error:?} instead of naming the missing output"
    );
}

#[test]
fn a_label_only_classifier_is_grid_searchable_on_a_label_metric() {
    let (data, targets) = fixture();
    let grid = ParameterGrid::from_candidates(vec![1.5_f32, 3.5, 5.5]);
    let splits: Vec<_> = KFold::new(4).split(data.rows()).unwrap().collect();

    let result = grid_search_classifier_labels(
        &data.as_view(),
        &targets,
        splits.iter().cloned(),
        &grid,
        ClassificationScorer::Accuracy,
        |train, _, threshold| MarginClassifier::fit(train, *threshold),
    )
    .expect("a label metric searches without probabilities");
    assert_eq!(result.candidates().len(), 3);

    let error = grid_search_classifier_labels(
        &data.as_view(),
        &targets,
        splits,
        &grid,
        ClassificationScorer::Brier,
        |train, _, threshold| MarginClassifier::fit(train, *threshold),
    )
    .expect_err("a probability metric cannot be served by labels alone");
    assert!(
        matches!(&error, SearchError::Candidate { .. }),
        "search returned {error:?} instead of failing the candidate"
    );
}

#[test]
fn a_label_only_classifier_supports_permutation_importance_on_a_label_metric() {
    let (data, targets) = fixture();
    let model = MarginClassifier::fit(&data.as_view(), 3.5).unwrap();
    let view = ScorableClassifier::labels_only(&model);

    let importance = permutation_importance_classifier(
        view,
        &data.as_view(),
        &targets,
        ClassificationScorer::Accuracy,
        PermutationImportanceParams::default(),
    )
    .expect("a label metric needs no probabilities");
    assert_eq!(importance.means().len(), data.columns());

    let error = permutation_importance_classifier(
        view,
        &data.as_view(),
        &targets,
        ClassificationScorer::LogLoss,
        PermutationImportanceParams::default(),
    )
    .expect_err("a probability metric cannot be served by labels alone");
    assert!(
        format!("{error:?}").contains("UnsupportedOutput"),
        "permutation importance returned {error:?} instead of naming the missing output"
    );
}

/// A probabilistic classifier still reaches every metric.
///
/// Without this the tests above would pass against a scoring layer that had
/// simply stopped serving probability metrics at all.
#[test]
fn a_probabilistic_classifier_still_serves_probability_metrics() {
    use ferricml::dummy::{DummyClassifier, DummyClassifierParams};

    let (data, targets) = fixture();
    let model = DummyClassifier::fit(&data.as_view(), &targets, DummyClassifierParams).unwrap();
    let view = ScorableClassifier::probabilistic(&model);

    assert!(
        score_classifier(
            view,
            &data.as_view(),
            &targets,
            ClassificationScorer::LogLoss
        )
        .is_ok()
    );
    assert!(
        score_classifier(
            view,
            &data.as_view(),
            &targets,
            ClassificationScorer::Accuracy
        )
        .is_ok()
    );
}
