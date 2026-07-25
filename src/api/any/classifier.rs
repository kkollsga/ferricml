use crate::data::MatrixView;
use crate::ensemble::{RandomForestClassifier, RandomForestClassifierParams};
use crate::linear_model::{LogisticRegression, LogisticRegressionParams};

use super::super::{Capabilities, Classifier, Estimator, HasCapabilities, ModelError};

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

    /// Returns the capabilities of the estimator type this value holds.
    ///
    /// [`HasCapabilities::CAPABILITIES`] is the intersection over every
    /// variant, which is what batch dispatch can rely on without inspecting
    /// the value. This reports the selected variant instead, which is what a
    /// caller needs before deciding whether this particular fitted model can
    /// be refitted with weights or persisted.
    pub fn capabilities(&self) -> Capabilities {
        match self {
            Self::RandomForest(_) => RandomForestClassifier::CAPABILITIES,
            Self::LogisticRegression(_) => LogisticRegression::CAPABILITIES,
        }
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

/// Declares only what holds for every variant, so a caller that has not
/// inspected the runtime variant is never promised more than it gets.
///
/// Weighted and multiclass fitting are both declared away structurally rather
/// than composed: the enum owns fitted models and no fitting entry point, so it
/// could accept neither weights nor a class set even though every variant can.
/// An intersection would have declared multiclass fitting the enum does not
/// offer. It still *holds* and serves a multiclass model — `classes()` and
/// `predict_proba` are already shaped by the fitted model — which is a property
/// of the value, not a capability of this type.
impl HasCapabilities for AnyClassifier {
    const CAPABILITIES: Capabilities = RandomForestClassifier::CAPABILITIES
        .intersection(LogisticRegression::CAPABILITIES)
        .with_sample_weights(false)
        .with_multiclass(false);
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
