use crate::data::MatrixView;
use crate::ensemble::{RandomForestClassifier, RandomForestClassifierParams};
use crate::linear_model::{LogisticRegression, LogisticRegressionParams};

use super::super::{Classifier, Estimator, ModelError};

/// Parameters retained by a fitted [`AnyClassifier`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyClassifierParams<'a> {
    /// Random-forest classifier parameters.
    RandomForest(&'a RandomForestClassifierParams),
    /// Logistic-regression classifier parameters.
    LogisticRegression(&'a LogisticRegressionParams),
}

/// An owned fitted classifier selected at runtime.
///
/// Matching happens once for each batch call; traversal remains statically
/// dispatched inside the concrete model.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyClassifier {
    /// A fitted random-forest classifier.
    RandomForest(RandomForestClassifier),
    /// A fitted logistic-regression classifier.
    LogisticRegression(LogisticRegression),
}

impl AnyClassifier {
    /// Returns the feature width required by this model.
    pub fn n_features_in(&self) -> usize {
        <Self as Estimator>::n_features_in(self)
    }

    /// Returns sorted class labels observed during fitting.
    pub fn classes(&self) -> &[u8] {
        <Self as Classifier>::classes(self)
    }

    /// Returns the concrete fitted parameters without erasing their type.
    pub fn get_params(&self) -> AnyClassifierParams<'_> {
        match self {
            Self::RandomForest(model) => AnyClassifierParams::RandomForest(model.get_params()),
            Self::LogisticRegression(model) => {
                AnyClassifierParams::LogisticRegression(model.get_params())
            }
        }
    }

    /// Predicts one label per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Predicts one label per row without allocating.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        <Self as Classifier>::predict_into(self, data, output)
    }

    /// Predicts row-major probabilities, allocating the output.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities without allocating.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Classifier>::predict_proba_into(self, data, output)
    }

    /// Predicts one fitted-class probability column without allocating.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Classifier>::predict_class_proba_into(self, data, class, output)
    }

    /// Predicts one fitted-class probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_class_proba(self, data, class)
    }
}

impl From<RandomForestClassifier> for AnyClassifier {
    fn from(model: RandomForestClassifier) -> Self {
        Self::RandomForest(model)
    }
}

impl From<LogisticRegression> for AnyClassifier {
    fn from(model: LogisticRegression) -> Self {
        Self::LogisticRegression(model)
    }
}

impl Estimator for AnyClassifier {
    fn n_features_in(&self) -> usize {
        match self {
            Self::RandomForest(model) => model.n_features_in(),
            Self::LogisticRegression(model) => model.n_features_in(),
        }
    }
}

impl Classifier for AnyClassifier {
    fn classes(&self) -> &[u8] {
        match self {
            Self::RandomForest(model) => model.classes(),
            Self::LogisticRegression(model) => model.classes(),
        }
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_into(data, output),
            Self::LogisticRegression(model) => model.predict_into(data, output),
        }
    }

    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_proba_into(data, output),
            Self::LogisticRegression(model) => model.predict_proba_into(data, output),
        }
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_class_proba_into(data, class, output),
            Self::LogisticRegression(model) => model.predict_class_proba_into(data, class, output),
        }
    }
}
