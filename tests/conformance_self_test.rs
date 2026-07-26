//! Proves the conformance battery by construction.
//!
//! An obligation that cannot fail is not a test. Every obligation the battery
//! declares is asserted here against a deliberately broken in-test estimator
//! that violates exactly that obligation, and a final check asserts that no
//! obligation is left without one. Without this file, collapsing the
//! per-estimator contract tests into the battery would be an act of faith.
//!
//! Each probe is honest except for one const-selected fault, so the expected
//! violation set is exact rather than "at least". Over-reporting is a defect
//! too: a battery that flags everything for every fault would tell a reviewer
//! nothing.

mod support;

use ferricml::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, ModelError, ProbabilisticClassifier,
    Regressor, Transformer,
};
use ferricml::data::{DenseMatrix, MatrixView};
use std::cell::Cell;
use std::collections::BTreeSet;

use support::conformance::{
    CLASSIFIER_OBLIGATIONS, ClassifierCase, OptionalFit, REGRESSOR_OBLIGATIONS, RegressorCase,
    Report, RoundTrip, SCALAR_CLASSIFIER_OBLIGATIONS, SCALAR_REGRESSOR_OBLIGATIONS, Sample,
    ScalarClassifierCase, ScalarRegressorCase, TRANSFORMER_OBLIGATIONS, TransformerCase,
    WORKSPACE_OBLIGATIONS, WorkspaceRegressorCase, batch_classifier_report, classifier_report,
    regressor_report, transformer_report, workspace_regressor_report,
};

// Faults, one per obligation the battery declares. `HONEST` must produce a
// clean report; every other value must produce exactly the violation named in
// the tables at the bottom of this file.
const HONEST: u8 = 0;
const PREDICT_FAILS: u8 = 1;
const WRONG_METADATA: u8 = 2;
const INTO_DISAGREES: u8 = 3;
const WRONG_PROBABILITY_COLUMN: u8 = 4;
const LABEL_IGNORES_PROBABILITY: u8 = 5;
const WIDTH_UNCHECKED: u8 = 6;
const LENGTH_UNCHECKED: u8 = 7;
const UNKNOWN_CLASS_ACCEPTED: u8 = 8;
const NONDETERMINISTIC_FIT: u8 = 9;
const WEIGHTS_DECLARED_WITHOUT_HOOK: u8 = 10;
const WEIGHT_HOOK_WITHOUT_DECLARATION: u8 = 11;
const WEIGHTED_FIT_DIFFERS: u8 = 12;
const ARTIFACT_DECLARED_WITHOUT_HOOK: u8 = 13;
const ARTIFACT_HOOK_WITHOUT_DECLARATION: u8 = 14;
const ARTIFACT_DECODES_DIFFERENTLY: u8 = 15;
const SCALAR_DISAGREES: u8 = 16;
const NON_FINITE_ACCEPTED: u8 = 17;
const SHAPE_IS_WRONG: u8 = 18;
const MULTICLASS_DECLARED_WITHOUT_HOOK: u8 = 19;
const MULTICLASS_HOOK_WITHOUT_DECLARATION: u8 = 20;
const MULTICLASS_COLLAPSES_CLASSES: u8 = 21;
const FIT_FAILS: u8 = 22;

// Faults of the workspace probe, which is a different family: only a model
// predicted through caller-owned scratch storage can have them.
const WORKSPACE_LENGTH_UNCHECKED: u8 = 23;
const WORKSPACE_LEAKS: u8 = 24;

// Faults of the one declared capability that shipped with no behavioral check.
const DECISION_FUNCTION_DECLARED_WITHOUT_HOOK: u8 = 25;
const DECISION_FUNCTION_HOOK_WITHOUT_DECLARATION: u8 = 26;
const DECISION_FUNCTION_CONTRADICTS_PROBABILITY: u8 = 27;

// Faults of the capability D11 made varying: producing probabilities at all.
const PROBABILITY_DECLARED_WITHOUT_HOOK: u8 = 28;
const PROBABILITY_HOOK_WITHOUT_DECLARATION: u8 = 29;

const BASE_THRESHOLD: f32 = 3.5;
const BASE_OFFSET: f32 = 0.0;
const BASE_SCALE: f32 = 2.0;
const FABRICATED_ARTIFACT: &[u8] = b"probe";

const fn probe_capabilities(fault: u8) -> Capabilities {
    Capabilities::NONE
        .with_sample_weights(fault != WEIGHT_HOOK_WITHOUT_DECLARATION)
        .with_artifact(fault != ARTIFACT_HOOK_WITHOUT_DECLARATION)
        // Multiclass is the one capability the honest probe does *not* have, so
        // it is opted into by fault rather than out of.
        .with_multiclass(
            fault == MULTICLASS_DECLARED_WITHOUT_HOOK || fault == MULTICLASS_COLLAPSES_CLASSES,
        )
        .with_decision_function(fault != DECISION_FUNCTION_HOOK_WITHOUT_DECLARATION)
        .with_probability(fault != PROBABILITY_HOOK_WITHOUT_DECLARATION)
}

thread_local! {
    /// How many times a `NONDETERMINISTIC_FIT` probe has been fitted *within
    /// the report currently running on this thread*.
    ///
    /// Manufactured nondeterminism has to be scoped to one report. A
    /// process-global counter would let two concurrently running tests that
    /// both build such a probe advance each other's drift, which makes the
    /// expected violation set depend on interleaving: `fit_weighted` reproduces
    /// the *first* fit of a report, so it only stays consistent while the
    /// count it is compared against is the one this report produced. Thread
    /// local storage isolates concurrent tests, and [`nondeterministic`]
    /// resets the cell so the outcome never depends on how many probes this
    /// thread has already fitted.
    static FIT_DRIFT: Cell<f32> = const { Cell::new(0.0) };
}

/// Returns the drift for this fit and advances it for the next one.
fn next_drift() -> f32 {
    FIT_DRIFT.with(|drift| {
        let current = drift.get();
        drift.set(current + 1.0);
        current
    })
}

/// The drift every `fit_weighted` reproduces: that of a report's first fit.
const FIRST_DRIFT: f32 = 0.0;

/// Runs one report for a probe whose fit is deliberately nondeterministic.
///
/// Resetting here is what makes the fault local to this invocation: the report
/// always sees drift 0 for its first fit, 1 for its refit, and `fit_weighted`
/// reproducing the first, whatever else the process has run. The trailing
/// assertion keeps the fault from going quiet — a battery that stopped
/// refitting would otherwise turn this probe into a test of nothing.
fn nondeterministic(run: impl FnOnce() -> Report) -> Report {
    FIT_DRIFT.with(|drift| drift.set(FIRST_DRIFT));
    let report = run();
    assert!(
        FIT_DRIFT.with(Cell::get) >= 2.0,
        "the nondeterministic probe was fitted fewer than twice, so its fault never applied"
    );
    report
}

fn sorted(report: &Report) -> Vec<&'static str> {
    report
        .names()
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The error a `FIT_FAILS` probe refuses the fixture with.
fn refusal() -> ModelError {
    ModelError::EmptyData
}

// ---------------------------------------------------------------- classifier

#[derive(Clone, Debug, PartialEq)]
struct ClassifierProbe<const FAULT: u8> {
    features: usize,
    classes: Vec<u8>,
    threshold: f32,
}

impl<const FAULT: u8> ClassifierProbe<FAULT> {
    fn with_threshold(data: &MatrixView<'_>, threshold: f32) -> Self {
        Self::with_classes(data, threshold, vec![0, 1])
    }

    fn with_classes(data: &MatrixView<'_>, threshold: f32, classes: Vec<u8>) -> Self {
        Self {
            features: data.columns(),
            classes,
            threshold,
        }
    }

    /// One row of probabilities over however many classes this probe has.
    ///
    /// The favoured class takes `0.75` and the rest split `0.25` evenly, which
    /// is exactly the two-class `[0.25, 0.75]` the binary probe used and sums
    /// to one exactly at every width, so the row-sum obligation measures the
    /// battery rather than this arithmetic.
    fn class_probabilities(&self, row: &[f32]) -> Vec<f32> {
        let classes = self.classes.len();
        if classes == 1 {
            return vec![1.0];
        }
        let favoured = if row[0] > self.threshold {
            classes - 1
        } else {
            0
        };
        let rest = 0.25 / (classes - 1) as f32;
        (0..classes)
            .map(|class| if class == favoured { 0.75 } else { rest })
            .collect()
    }

    fn favoured(&self, row: &[f32]) -> usize {
        let probabilities = self.class_probabilities(row);
        let mut best = 0;
        for class in 1..probabilities.len() {
            if probabilities[class] > probabilities[best] {
                best = class;
            }
        }
        best
    }

    fn label(&self, row: &[f32]) -> u8 {
        let honest = self.favoured(row);
        if FAULT == LABEL_IGNORES_PROBABILITY {
            self.classes[(honest + 1) % self.classes.len()]
        } else {
            self.classes[honest]
        }
    }

    /// A raw score whose ordering agrees with the positive-class probability.
    ///
    /// The probability is a step at `threshold`, so any function increasing in
    /// `row[0]` is rank-consistent with it. The contradicting fault reverses
    /// the sign, which passes every shape check and fails only the ordering.
    fn score(&self, row: &[f32]) -> f32 {
        let honest = row[0] - self.threshold;
        if FAULT == DECISION_FUNCTION_CONTRADICTS_PROBABILITY {
            -honest
        } else {
            honest
        }
    }

    fn check_width(&self, columns: usize) -> Result<(), ModelError> {
        if FAULT == WIDTH_UNCHECKED || columns == self.features {
            return Ok(());
        }
        Err(ModelError::FeatureDimension {
            expected: self.features,
            actual: columns,
        })
    }

    fn check_len(&self, actual: usize, expected: usize) -> Result<(), ModelError> {
        if FAULT == LENGTH_UNCHECKED || actual == expected {
            return Ok(());
        }
        Err(ModelError::OutputLength { expected, actual })
    }
}

impl<const FAULT: u8> Estimator for ClassifierProbe<FAULT> {
    fn n_features_in(&self) -> usize {
        if FAULT == WRONG_METADATA {
            self.features + 1
        } else {
            self.features
        }
    }
}

impl<const FAULT: u8> HasCapabilities for ClassifierProbe<FAULT> {
    const CAPABILITIES: Capabilities = probe_capabilities(FAULT);
}

impl<const FAULT: u8> Classifier for ClassifierProbe<FAULT> {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        if FAULT == PREDICT_FAILS {
            return Err(ModelError::EmptyData);
        }
        self.check_width(data.columns())?;
        Ok(data.iter_rows().map(|row| self.label(row)).collect())
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        self.check_width(data.columns())?;
        self.check_len(output.len(), data.rows())?;
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            *slot = if FAULT == INTO_DISAGREES {
                self.classes[0]
            } else {
                self.label(row)
            };
        }
        Ok(())
    }
}

impl<const FAULT: u8> ProbabilisticClassifier for ClassifierProbe<FAULT> {
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.check_width(data.columns())?;
        self.check_len(output.len(), data.rows() * self.classes.len())?;
        for (slot, row) in output.chunks_mut(self.classes.len()).zip(data.iter_rows()) {
            if slot.len() == self.classes.len() {
                slot.copy_from_slice(&self.class_probabilities(row));
            }
        }
        Ok(())
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.check_width(data.columns())?;
        if FAULT != UNKNOWN_CLASS_ACCEPTED && !self.classes.contains(&class) {
            return Err(ModelError::UnknownClass { class });
        }
        self.check_len(output.len(), data.rows())?;
        let column = self.classes.iter().position(|&known| known == class);
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            let honest = column.map_or(0.0, |column| self.class_probabilities(row)[column]);
            *slot = if FAULT == WRONG_PROBABILITY_COLUMN {
                1.0 - honest
            } else {
                honest
            };
        }
        Ok(())
    }
}

struct ClassifierProbeCase<const FAULT: u8>;

impl<const FAULT: u8> ClassifierCase for ClassifierProbeCase<FAULT> {
    type Model = ClassifierProbe<FAULT>;
    const NAME: &'static str = "classifier probe";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        if FAULT == FIT_FAILS {
            return Err(refusal());
        }
        let drift = if FAULT == NONDETERMINISTIC_FIT {
            next_drift()
        } else {
            FIRST_DRIFT
        };
        Ok(ClassifierProbe::with_threshold(
            &train.view(),
            BASE_THRESHOLD + drift,
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        match FAULT {
            WEIGHTS_DECLARED_WITHOUT_HOOK => None,
            WEIGHTED_FIT_DIFFERS => Some(Ok(ClassifierProbe::with_threshold(&train.view(), 0.0))),
            _ => Some(Ok(ClassifierProbe::with_threshold(
                &train.view(),
                BASE_THRESHOLD + FIRST_DRIFT,
            ))),
        }
    }

    fn fit_multiclass(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        match FAULT {
            MULTICLASS_DECLARED_WITHOUT_HOOK => None,
            // A hook that quietly returns a two-class model: the shape checks
            // all pass, and only the class set gives it away.
            MULTICLASS_COLLAPSES_CLASSES => Some(Ok(ClassifierProbe::with_threshold(
                &train.view(),
                BASE_THRESHOLD + FIRST_DRIFT,
            ))),
            MULTICLASS_HOOK_WITHOUT_DECLARATION => Some(Ok(ClassifierProbe::with_classes(
                &train.view(),
                BASE_THRESHOLD + FIRST_DRIFT,
                train.class_labels.classes().to_vec(),
            ))),
            _ => None,
        }
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        match FAULT {
            ARTIFACT_DECLARED_WITHOUT_HOOK => None,
            ARTIFACT_DECODES_DIFFERENTLY => Some(Ok((
                FABRICATED_ARTIFACT.to_vec(),
                ClassifierProbe {
                    threshold: 0.0,
                    ..model.clone()
                },
            ))),
            _ => Some(Ok((FABRICATED_ARTIFACT.to_vec(), model.clone()))),
        }
    }

    fn predict_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        (FAULT != PROBABILITY_DECLARED_WITHOUT_HOOK)
            .then(|| ProbabilisticClassifier::predict_proba(model, data))
    }

    fn predict_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        (FAULT != PROBABILITY_DECLARED_WITHOUT_HOOK)
            .then(|| ProbabilisticClassifier::predict_proba_into(model, data, output))
    }

    fn predict_class_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        (FAULT != PROBABILITY_DECLARED_WITHOUT_HOOK)
            .then(|| ProbabilisticClassifier::predict_class_proba(model, data, class))
    }

    fn predict_class_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        (FAULT != PROBABILITY_DECLARED_WITHOUT_HOOK)
            .then(|| ProbabilisticClassifier::predict_class_proba_into(model, data, class, output))
    }

    fn decision_function(
        model: &Self::Model,
        data: &MatrixView<'_>,
    ) -> Option<Result<Vec<f32>, ModelError>> {
        if FAULT == DECISION_FUNCTION_DECLARED_WITHOUT_HOOK {
            return None;
        }
        Some(Ok(data.iter_rows().map(|row| model.score(row)).collect()))
    }

    fn decision_function_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Option<Result<(), ModelError>> {
        if FAULT == DECISION_FUNCTION_DECLARED_WITHOUT_HOOK {
            return None;
        }
        if output.len() != data.rows() {
            return Some(Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            }));
        }
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            *slot = model.score(row);
        }
        Some(Ok(()))
    }
}

impl<const FAULT: u8> ScalarClassifierCase for ClassifierProbeCase<FAULT> {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        if row.len() != model.features {
            return Err(ModelError::FeatureDimension {
                expected: model.features,
                actual: row.len(),
            });
        }
        if FAULT != NON_FINITE_ACCEPTED
            && let Some(column) = row.iter().position(|value| !value.is_finite())
        {
            return Err(ModelError::NonFiniteFeature { row: 0, column });
        }
        let label = model.label(row);
        Ok(if FAULT == SCALAR_DISAGREES {
            1 - label
        } else {
            label
        })
    }
}

// ----------------------------------------------------------------- regressor

#[derive(Clone, Debug, PartialEq)]
struct RegressorProbe<const FAULT: u8> {
    features: usize,
    offset: f32,
}

impl<const FAULT: u8> RegressorProbe<FAULT> {
    fn with_offset(data: &MatrixView<'_>, offset: f32) -> Self {
        Self {
            features: data.columns(),
            offset,
        }
    }

    fn value(&self, row: &[f32]) -> f32 {
        row[0] + self.offset
    }

    fn check_width(&self, columns: usize) -> Result<(), ModelError> {
        if FAULT == WIDTH_UNCHECKED || columns == self.features {
            return Ok(());
        }
        Err(ModelError::FeatureDimension {
            expected: self.features,
            actual: columns,
        })
    }
}

impl<const FAULT: u8> Estimator for RegressorProbe<FAULT> {
    fn n_features_in(&self) -> usize {
        if FAULT == WRONG_METADATA {
            self.features + 1
        } else {
            self.features
        }
    }
}

impl<const FAULT: u8> HasCapabilities for RegressorProbe<FAULT> {
    const CAPABILITIES: Capabilities = probe_capabilities(FAULT);
}

impl<const FAULT: u8> Regressor for RegressorProbe<FAULT> {
    fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        if FAULT == PREDICT_FAILS {
            return Err(ModelError::EmptyData);
        }
        self.check_width(data.columns())?;
        Ok(data.iter_rows().map(|row| self.value(row)).collect())
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        self.check_width(data.columns())?;
        if FAULT != LENGTH_UNCHECKED && output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            *slot = if FAULT == INTO_DISAGREES {
                0.0
            } else {
                self.value(row)
            };
        }
        Ok(())
    }
}

struct RegressorProbeCase<const FAULT: u8>;

impl<const FAULT: u8> RegressorCase for RegressorProbeCase<FAULT> {
    type Model = RegressorProbe<FAULT>;
    const NAME: &'static str = "regressor probe";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        if FAULT == FIT_FAILS {
            return Err(refusal());
        }
        let drift = if FAULT == NONDETERMINISTIC_FIT {
            next_drift()
        } else {
            FIRST_DRIFT
        };
        Ok(RegressorProbe::with_offset(
            &train.view(),
            BASE_OFFSET + drift,
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        match FAULT {
            WEIGHTS_DECLARED_WITHOUT_HOOK => None,
            WEIGHTED_FIT_DIFFERS => Some(Ok(RegressorProbe::with_offset(&train.view(), 1.0))),
            _ => Some(Ok(RegressorProbe::with_offset(
                &train.view(),
                BASE_OFFSET + FIRST_DRIFT,
            ))),
        }
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        match FAULT {
            ARTIFACT_DECLARED_WITHOUT_HOOK => None,
            ARTIFACT_DECODES_DIFFERENTLY => Some(Ok((
                FABRICATED_ARTIFACT.to_vec(),
                RegressorProbe {
                    offset: 1.0,
                    ..model.clone()
                },
            ))),
            _ => Some(Ok((FABRICATED_ARTIFACT.to_vec(), model.clone()))),
        }
    }
}

impl<const FAULT: u8> ScalarRegressorCase for RegressorProbeCase<FAULT> {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        if row.len() != model.features {
            return Err(ModelError::FeatureDimension {
                expected: model.features,
                actual: row.len(),
            });
        }
        if FAULT != NON_FINITE_ACCEPTED
            && let Some(column) = row.iter().position(|value| !value.is_finite())
        {
            return Err(ModelError::NonFiniteFeature { row: 0, column });
        }
        let value = model.value(row);
        Ok(if FAULT == SCALAR_DISAGREES {
            value + 1.0
        } else {
            value
        })
    }
}

// --------------------------------------------------------------- transformer

#[derive(Clone, Debug, PartialEq)]
struct TransformerProbe<const FAULT: u8> {
    features: usize,
    scale: f32,
}

impl<const FAULT: u8> TransformerProbe<FAULT> {
    fn with_scale(data: &MatrixView<'_>, scale: f32) -> Self {
        Self {
            features: data.columns(),
            scale,
        }
    }

    fn check_width(&self, columns: usize) -> Result<(), ModelError> {
        if FAULT == WIDTH_UNCHECKED || columns == self.features {
            return Ok(());
        }
        Err(ModelError::FeatureDimension {
            expected: self.features,
            actual: columns,
        })
    }
}

impl<const FAULT: u8> Estimator for TransformerProbe<FAULT> {
    fn n_features_in(&self) -> usize {
        if FAULT == WRONG_METADATA {
            self.features + 1
        } else {
            self.features
        }
    }
}

impl<const FAULT: u8> HasCapabilities for TransformerProbe<FAULT> {
    const CAPABILITIES: Capabilities = probe_capabilities(FAULT);
}

impl<const FAULT: u8> Transformer for TransformerProbe<FAULT> {
    fn n_features_out(&self) -> usize {
        self.features
    }

    /// Overridden so that a caller-owned fault cannot hide behind the
    /// allocating convenience method delegating to it.
    fn transform(&self, data: &MatrixView<'_>) -> Result<DenseMatrix, ModelError> {
        if FAULT == PREDICT_FAILS {
            return Err(ModelError::EmptyData);
        }
        self.check_width(data.columns())?;
        let values: Vec<f32> = data
            .as_slice()
            .iter()
            .map(|value| value * self.scale)
            .collect();
        DenseMatrix::new(values, data.rows(), self.features).map_err(|_| ModelError::EmptyData)
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        if FAULT == PREDICT_FAILS {
            return Err(ModelError::EmptyData);
        }
        self.check_width(data.columns())?;
        let expected = data.rows() * self.features;
        if FAULT != LENGTH_UNCHECKED && output.len() != expected {
            return Err(ModelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        let written = output.len().min(expected);
        for (slot, &value) in output.iter_mut().zip(data.as_slice()) {
            *slot = if FAULT == INTO_DISAGREES {
                0.0
            } else {
                value * self.scale
            };
        }
        // A shape fault has to return a view that is itself valid, or it would
        // be caught as a failed transform rather than as a wrong shape.
        let (rows, columns) = if FAULT == SHAPE_IS_WRONG {
            (written, 1)
        } else {
            (written / self.features, self.features)
        };
        MatrixView::new(&output[..written], rows, columns).map_err(|_| ModelError::EmptyData)
    }
}

struct TransformerProbeCase<const FAULT: u8>;

impl<const FAULT: u8> TransformerCase for TransformerProbeCase<FAULT> {
    type Model = TransformerProbe<FAULT>;
    const NAME: &'static str = "transformer probe";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        if FAULT == FIT_FAILS {
            return Err(refusal());
        }
        let drift = if FAULT == NONDETERMINISTIC_FIT {
            next_drift()
        } else {
            FIRST_DRIFT
        };
        Ok(TransformerProbe::with_scale(
            &train.view(),
            BASE_SCALE + drift,
        ))
    }

    fn fit_weighted(train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        match FAULT {
            WEIGHTS_DECLARED_WITHOUT_HOOK => None,
            WEIGHTED_FIT_DIFFERS => Some(Ok(TransformerProbe::with_scale(&train.view(), 3.0))),
            _ => Some(Ok(TransformerProbe::with_scale(
                &train.view(),
                BASE_SCALE + FIRST_DRIFT,
            ))),
        }
    }

    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        match FAULT {
            ARTIFACT_DECLARED_WITHOUT_HOOK => None,
            ARTIFACT_DECODES_DIFFERENTLY => Some(Ok((
                FABRICATED_ARTIFACT.to_vec(),
                TransformerProbe {
                    scale: 3.0,
                    ..model.clone()
                },
            ))),
            _ => Some(Ok((FABRICATED_ARTIFACT.to_vec(), model.clone()))),
        }
    }
}

// ----------------------------------------------------------------- workspace

/// A model that predicts through caller-owned scratch storage.
///
/// Deliberately shaped like the compositions this exists for: it stages the
/// batch into the workspace and predicts from the staged copy, so a workspace
/// that is the wrong length or that still holds the previous batch changes the
/// answer. It declares nothing, so its weighted and artifact obligations pass
/// without a hook.
#[derive(Clone, Debug, PartialEq)]
struct WorkspaceProbe<const FAULT: u8> {
    features: usize,
    offset: f32,
}

impl<const FAULT: u8> WorkspaceProbe<FAULT> {
    fn staged_len(&self, rows: usize) -> usize {
        rows * self.features
    }

    fn predict_into(
        &self,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        if data.columns() != self.features {
            return Err(ModelError::FeatureDimension {
                expected: self.features,
                actual: data.columns(),
            });
        }
        let expected = self.staged_len(data.rows());
        if FAULT != WORKSPACE_LENGTH_UNCHECKED && workspace.len() != expected {
            return Err(ModelError::OutputLength {
                expected,
                actual: workspace.len(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        // Staging. A leaking probe treats an already-written slot as state and
        // declines to restage it, which is exactly what mistaking scratch for
        // state looks like: correct on a fresh buffer, stale on a reused one.
        // The fault is deliberately confined to *reuse* — restaging the same
        // batch is idempotent — so the probe violates one obligation, not two.
        for (slot, &value) in workspace.iter_mut().zip(data.as_slice()) {
            if FAULT == WORKSPACE_LEAKS && *slot != 0.0 {
                continue;
            }
            *slot = value;
        }
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = workspace[index * self.features] + self.offset;
        }
        Ok(())
    }
}

impl<const FAULT: u8> Estimator for WorkspaceProbe<FAULT> {
    fn n_features_in(&self) -> usize {
        self.features
    }
}

impl<const FAULT: u8> HasCapabilities for WorkspaceProbe<FAULT> {
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

struct WorkspaceProbeCase<const FAULT: u8>;

impl<const FAULT: u8> WorkspaceRegressorCase for WorkspaceProbeCase<FAULT> {
    type Model = WorkspaceProbe<FAULT>;
    const NAME: &'static str = "workspace probe";

    fn fit(train: &Sample, _holdout: &Sample) -> Result<Self::Model, ModelError> {
        Ok(WorkspaceProbe {
            features: train.columns(),
            offset: BASE_OFFSET,
        })
    }

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        Ok(model.staged_len(rows))
    }

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        let mut output = vec![0.0; data.rows()];
        model.predict_into(data, workspace, &mut output)?;
        Ok(output)
    }

    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.predict_into(data, workspace, output)
    }
}

// -------------------------------------------------------------------- proofs

#[test]
fn an_honest_estimator_violates_nothing() {
    assert_eq!(
        sorted(&classifier_report::<ClassifierProbeCase<HONEST>>()),
        Vec::<&str>::new()
    );
    assert_eq!(
        sorted(&regressor_report::<RegressorProbeCase<HONEST>>()),
        Vec::<&str>::new()
    );
    assert_eq!(
        sorted(&transformer_report::<TransformerProbeCase<HONEST>>()),
        Vec::<&str>::new()
    );
    assert_eq!(
        sorted(&batch_classifier_report::<ClassifierProbeCase<HONEST>>()),
        Vec::<&str>::new()
    );
    assert_eq!(
        sorted(&workspace_regressor_report::<WorkspaceProbeCase<HONEST>>()),
        Vec::<&str>::new()
    );
}

macro_rules! violates {
    ($name:ident, $report:expr, $expected:expr) => {
        #[test]
        fn $name() {
            assert_eq!(sorted(&$report), $expected);
        }
    };
}

violates!(
    classifier_fit_failure,
    classifier_report::<ClassifierProbeCase<FIT_FAILS>>(),
    ["fits_the_fixture"]
);
violates!(
    classifier_predict_failure,
    classifier_report::<ClassifierProbeCase<PREDICT_FAILS>>(),
    ["predicts_the_fixture"]
);
// Two obligations, from one lie. The probe overrides `predict` but not
// `predict_proba`, so the allocating `ProbabilisticClassifier::predict_proba`
// default runs — and that default now checks the batch width against
// `n_features_in` before allocating. A probe whose `n_features_in` disagrees
// with the width its own `_into` accepts therefore fails the declared
// probability path as well as the metadata check. That is the fault showing
// through a second surface rather than a second fault.
violates!(
    classifier_wrong_metadata,
    classifier_report::<ClassifierProbeCase<WRONG_METADATA>>(),
    ["metadata", "probability_declaration_matches_behavior"]
);
violates!(
    classifier_into_disagrees,
    classifier_report::<ClassifierProbeCase<INTO_DISAGREES>>(),
    ["into_matches_allocating"]
);
violates!(
    classifier_wrong_probability_column,
    classifier_report::<ClassifierProbeCase<WRONG_PROBABILITY_COLUMN>>(),
    ["probability_columns_follow_classes"]
);
violates!(
    classifier_label_ignores_probability,
    classifier_report::<ClassifierProbeCase<LABEL_IGNORES_PROBABILITY>>(),
    ["label_matches_probability_argmax"]
);
violates!(
    classifier_width_unchecked,
    classifier_report::<ClassifierProbeCase<WIDTH_UNCHECKED>>(),
    ["feature_width_validated_before_write"]
);
violates!(
    classifier_length_unchecked,
    classifier_report::<ClassifierProbeCase<LENGTH_UNCHECKED>>(),
    ["output_length_validated_before_write"]
);
violates!(
    classifier_unknown_class_accepted,
    classifier_report::<ClassifierProbeCase<UNKNOWN_CLASS_ACCEPTED>>(),
    ["unknown_class_rejected"]
);
violates!(
    classifier_nondeterministic_fit,
    nondeterministic(classifier_report::<ClassifierProbeCase<NONDETERMINISTIC_FIT>>),
    ["refit_is_deterministic"]
);
violates!(
    classifier_weights_declared_without_hook,
    classifier_report::<ClassifierProbeCase<WEIGHTS_DECLARED_WITHOUT_HOOK>>(),
    ["sample_weight_declaration_matches_behavior"]
);
violates!(
    classifier_weight_hook_without_declaration,
    classifier_report::<ClassifierProbeCase<WEIGHT_HOOK_WITHOUT_DECLARATION>>(),
    ["sample_weight_declaration_matches_behavior"]
);
violates!(
    classifier_weighted_fit_differs,
    classifier_report::<ClassifierProbeCase<WEIGHTED_FIT_DIFFERS>>(),
    ["sample_weight_declaration_matches_behavior"]
);
violates!(
    classifier_artifact_declared_without_hook,
    classifier_report::<ClassifierProbeCase<ARTIFACT_DECLARED_WITHOUT_HOOK>>(),
    ["artifact_declaration_matches_behavior"]
);
violates!(
    classifier_artifact_hook_without_declaration,
    classifier_report::<ClassifierProbeCase<ARTIFACT_HOOK_WITHOUT_DECLARATION>>(),
    ["artifact_declaration_matches_behavior"]
);
violates!(
    classifier_artifact_decodes_differently,
    classifier_report::<ClassifierProbeCase<ARTIFACT_DECODES_DIFFERENTLY>>(),
    ["artifact_declaration_matches_behavior"]
);
violates!(
    classifier_multiclass_declared_without_hook,
    classifier_report::<ClassifierProbeCase<MULTICLASS_DECLARED_WITHOUT_HOOK>>(),
    ["multiclass_declaration_matches_behavior"]
);
violates!(
    classifier_multiclass_hook_without_declaration,
    classifier_report::<ClassifierProbeCase<MULTICLASS_HOOK_WITHOUT_DECLARATION>>(),
    ["multiclass_declaration_matches_behavior"]
);
violates!(
    classifier_multiclass_collapses_the_class_set,
    classifier_report::<ClassifierProbeCase<MULTICLASS_COLLAPSES_CLASSES>>(),
    ["multiclass_declaration_matches_behavior"]
);
violates!(
    classifier_probability_declared_without_hook,
    classifier_report::<ClassifierProbeCase<PROBABILITY_DECLARED_WITHOUT_HOOK>>(),
    ["probability_declaration_matches_behavior"]
);
violates!(
    classifier_probability_hook_without_declaration,
    classifier_report::<ClassifierProbeCase<PROBABILITY_HOOK_WITHOUT_DECLARATION>>(),
    ["probability_declaration_matches_behavior"]
);
violates!(
    classifier_decision_function_declared_without_hook,
    classifier_report::<ClassifierProbeCase<DECISION_FUNCTION_DECLARED_WITHOUT_HOOK>>(),
    ["decision_function_declaration_matches_behavior"]
);
violates!(
    classifier_decision_function_hook_without_declaration,
    classifier_report::<ClassifierProbeCase<DECISION_FUNCTION_HOOK_WITHOUT_DECLARATION>>(),
    ["decision_function_declaration_matches_behavior"]
);
violates!(
    classifier_decision_function_contradicts_probability,
    classifier_report::<ClassifierProbeCase<DECISION_FUNCTION_CONTRADICTS_PROBABILITY>>(),
    ["decision_function_declaration_matches_behavior"]
);
violates!(
    classifier_scalar_disagrees,
    classifier_report::<ClassifierProbeCase<SCALAR_DISAGREES>>(),
    ["scalar_matches_batch"]
);
violates!(
    classifier_non_finite_accepted,
    classifier_report::<ClassifierProbeCase<NON_FINITE_ACCEPTED>>(),
    ["non_finite_scalar_rejected"]
);

violates!(
    regressor_fit_failure,
    regressor_report::<RegressorProbeCase<FIT_FAILS>>(),
    ["fits_the_fixture"]
);
violates!(
    regressor_predict_failure,
    regressor_report::<RegressorProbeCase<PREDICT_FAILS>>(),
    ["predicts_the_fixture"]
);
violates!(
    regressor_wrong_metadata,
    regressor_report::<RegressorProbeCase<WRONG_METADATA>>(),
    ["metadata"]
);
violates!(
    regressor_into_disagrees,
    regressor_report::<RegressorProbeCase<INTO_DISAGREES>>(),
    ["into_matches_allocating"]
);
violates!(
    regressor_width_unchecked,
    regressor_report::<RegressorProbeCase<WIDTH_UNCHECKED>>(),
    ["feature_width_validated_before_write"]
);
violates!(
    regressor_length_unchecked,
    regressor_report::<RegressorProbeCase<LENGTH_UNCHECKED>>(),
    ["output_length_validated_before_write"]
);
violates!(
    regressor_nondeterministic_fit,
    nondeterministic(regressor_report::<RegressorProbeCase<NONDETERMINISTIC_FIT>>),
    ["refit_is_deterministic"]
);
violates!(
    regressor_weights_declared_without_hook,
    regressor_report::<RegressorProbeCase<WEIGHTS_DECLARED_WITHOUT_HOOK>>(),
    ["sample_weight_declaration_matches_behavior"]
);
violates!(
    regressor_artifact_decodes_differently,
    regressor_report::<RegressorProbeCase<ARTIFACT_DECODES_DIFFERENTLY>>(),
    ["artifact_declaration_matches_behavior"]
);
violates!(
    regressor_scalar_disagrees,
    regressor_report::<RegressorProbeCase<SCALAR_DISAGREES>>(),
    ["scalar_matches_batch"]
);
violates!(
    regressor_non_finite_accepted,
    regressor_report::<RegressorProbeCase<NON_FINITE_ACCEPTED>>(),
    ["non_finite_scalar_rejected"]
);

violates!(
    transformer_fit_failure,
    transformer_report::<TransformerProbeCase<FIT_FAILS>>(),
    ["fits_the_fixture"]
);
violates!(
    transformer_transform_failure,
    transformer_report::<TransformerProbeCase<PREDICT_FAILS>>(),
    ["transforms_the_fixture"]
);
violates!(
    transformer_wrong_metadata,
    transformer_report::<TransformerProbeCase<WRONG_METADATA>>(),
    ["metadata"]
);
violates!(
    transformer_into_disagrees,
    transformer_report::<TransformerProbeCase<INTO_DISAGREES>>(),
    ["into_matches_allocating"]
);
violates!(
    transformer_shape_is_wrong,
    transformer_report::<TransformerProbeCase<SHAPE_IS_WRONG>>(),
    ["transformed_shape_is_preserved"]
);
violates!(
    transformer_width_unchecked,
    transformer_report::<TransformerProbeCase<WIDTH_UNCHECKED>>(),
    ["feature_width_validated_before_write"]
);
violates!(
    transformer_length_unchecked,
    transformer_report::<TransformerProbeCase<LENGTH_UNCHECKED>>(),
    ["output_length_validated_before_write"]
);
violates!(
    transformer_nondeterministic_fit,
    nondeterministic(transformer_report::<TransformerProbeCase<NONDETERMINISTIC_FIT>>),
    ["refit_is_deterministic"]
);
violates!(
    transformer_weights_declared_without_hook,
    transformer_report::<TransformerProbeCase<WEIGHTS_DECLARED_WITHOUT_HOOK>>(),
    ["sample_weight_declaration_matches_behavior"]
);
violates!(
    transformer_artifact_decodes_differently,
    transformer_report::<TransformerProbeCase<ARTIFACT_DECODES_DIFFERENTLY>>(),
    ["artifact_declaration_matches_behavior"]
);

violates!(
    workspace_length_unchecked,
    workspace_regressor_report::<WorkspaceProbeCase<WORKSPACE_LENGTH_UNCHECKED>>(),
    ["workspace_length_validated_before_write"]
);
violates!(
    workspace_leaks_between_batches,
    workspace_regressor_report::<WorkspaceProbeCase<WORKSPACE_LEAKS>>(),
    ["workspace_reuse_is_independent"]
);

/// A nondeterministic probe must report the same violation every time.
///
/// Regression test for a real defect in this file: the manufactured drift was
/// a process-global counter, so the first fit of a *later* report no longer
/// matched the fixed model `fit_weighted` returns, and
/// `sample_weight_declaration_matches_behavior` tripped alongside
/// `refit_is_deterministic`. Two tests building the probe concurrently hit the
/// same cause through interleaving, which cannot be reproduced on demand;
/// repeating the report on one thread reproduces it deterministically.
#[test]
fn a_nondeterministic_probe_reports_the_same_violations_on_every_invocation() {
    for _ in 0..4 {
        assert_eq!(
            sorted(&nondeterministic(
                classifier_report::<ClassifierProbeCase<NONDETERMINISTIC_FIT>>,
            )),
            ["refit_is_deterministic"]
        );
        assert_eq!(
            sorted(&nondeterministic(
                regressor_report::<RegressorProbeCase<NONDETERMINISTIC_FIT>>,
            )),
            ["refit_is_deterministic"]
        );
        assert_eq!(
            sorted(&nondeterministic(
                transformer_report::<TransformerProbeCase<NONDETERMINISTIC_FIT>>,
            )),
            ["refit_is_deterministic"]
        );
    }
}

/// Every declared obligation must have a probe that trips it.
///
/// This is the check that makes the battery honest as it grows: adding an
/// obligation without a probe fails here, exactly as adding a source-layout
/// rule without a synthetic violation fails that checker's self-test.
///
/// The workspace obligations are checked as their own set. Both the classifier
/// and the regressor battery reach one shared implementation of them, so a
/// probe tripping them on the regressor side proves the code path for both —
/// the same reasoning that lets one `check_output_length` serve three
/// categories.
#[test]
fn every_declared_obligation_has_a_probe_that_trips_it() {
    let classifier: BTreeSet<&str> = CLASSIFIER_OBLIGATIONS
        .iter()
        .chain(SCALAR_CLASSIFIER_OBLIGATIONS)
        .copied()
        .collect();
    let regressor: BTreeSet<&str> = REGRESSOR_OBLIGATIONS
        .iter()
        .chain(SCALAR_REGRESSOR_OBLIGATIONS)
        .copied()
        .collect();
    let transformer: BTreeSet<&str> = TRANSFORMER_OBLIGATIONS.iter().copied().collect();
    // A workspace-shaped case owes the regressor obligations *and* the two
    // that only caller-owned scratch storage can violate, so its probes are
    // allowed to report from either list.
    let workspace: BTreeSet<&str> = WORKSPACE_OBLIGATIONS.iter().copied().collect();
    let workspace_reportable: BTreeSet<&str> = workspace.union(&regressor).copied().collect();

    let mut tripped_classifier = BTreeSet::new();
    for names in [
        sorted(&classifier_report::<ClassifierProbeCase<FIT_FAILS>>()),
        sorted(&classifier_report::<ClassifierProbeCase<PREDICT_FAILS>>()),
        sorted(&classifier_report::<ClassifierProbeCase<WRONG_METADATA>>()),
        sorted(&classifier_report::<ClassifierProbeCase<INTO_DISAGREES>>()),
        sorted(&classifier_report::<
            ClassifierProbeCase<WRONG_PROBABILITY_COLUMN>,
        >()),
        sorted(&classifier_report::<
            ClassifierProbeCase<LABEL_IGNORES_PROBABILITY>,
        >()),
        sorted(&classifier_report::<ClassifierProbeCase<WIDTH_UNCHECKED>>()),
        sorted(&classifier_report::<ClassifierProbeCase<LENGTH_UNCHECKED>>()),
        sorted(&classifier_report::<
            ClassifierProbeCase<UNKNOWN_CLASS_ACCEPTED>,
        >()),
        sorted(&nondeterministic(
            classifier_report::<ClassifierProbeCase<NONDETERMINISTIC_FIT>>,
        )),
        sorted(&classifier_report::<
            ClassifierProbeCase<WEIGHTS_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&classifier_report::<
            ClassifierProbeCase<ARTIFACT_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&classifier_report::<
            ClassifierProbeCase<MULTICLASS_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&classifier_report::<
            ClassifierProbeCase<DECISION_FUNCTION_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&classifier_report::<
            ClassifierProbeCase<PROBABILITY_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&classifier_report::<ClassifierProbeCase<SCALAR_DISAGREES>>()),
        sorted(&classifier_report::<ClassifierProbeCase<NON_FINITE_ACCEPTED>>()),
    ] {
        tripped_classifier.extend(names);
    }

    let mut tripped_regressor = BTreeSet::new();
    for names in [
        sorted(&regressor_report::<RegressorProbeCase<FIT_FAILS>>()),
        sorted(&regressor_report::<RegressorProbeCase<PREDICT_FAILS>>()),
        sorted(&regressor_report::<RegressorProbeCase<WRONG_METADATA>>()),
        sorted(&regressor_report::<RegressorProbeCase<INTO_DISAGREES>>()),
        sorted(&regressor_report::<RegressorProbeCase<WIDTH_UNCHECKED>>()),
        sorted(&regressor_report::<RegressorProbeCase<LENGTH_UNCHECKED>>()),
        sorted(&nondeterministic(
            regressor_report::<RegressorProbeCase<NONDETERMINISTIC_FIT>>,
        )),
        sorted(&regressor_report::<
            RegressorProbeCase<WEIGHTS_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&regressor_report::<
            RegressorProbeCase<ARTIFACT_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&regressor_report::<RegressorProbeCase<SCALAR_DISAGREES>>()),
        sorted(&regressor_report::<RegressorProbeCase<NON_FINITE_ACCEPTED>>()),
    ] {
        tripped_regressor.extend(names);
    }

    let mut tripped_transformer = BTreeSet::new();
    for names in [
        sorted(&transformer_report::<TransformerProbeCase<FIT_FAILS>>()),
        sorted(&transformer_report::<TransformerProbeCase<PREDICT_FAILS>>()),
        sorted(&transformer_report::<TransformerProbeCase<WRONG_METADATA>>()),
        sorted(&transformer_report::<TransformerProbeCase<INTO_DISAGREES>>()),
        sorted(&transformer_report::<TransformerProbeCase<SHAPE_IS_WRONG>>()),
        sorted(&transformer_report::<TransformerProbeCase<WIDTH_UNCHECKED>>()),
        sorted(&transformer_report::<TransformerProbeCase<LENGTH_UNCHECKED>>()),
        sorted(&nondeterministic(
            transformer_report::<TransformerProbeCase<NONDETERMINISTIC_FIT>>,
        )),
        sorted(&transformer_report::<
            TransformerProbeCase<WEIGHTS_DECLARED_WITHOUT_HOOK>,
        >()),
        sorted(&transformer_report::<
            TransformerProbeCase<ARTIFACT_DECODES_DIFFERENTLY>,
        >()),
    ] {
        tripped_transformer.extend(names);
    }

    let mut tripped_workspace = BTreeSet::new();
    for names in [
        sorted(&workspace_regressor_report::<
            WorkspaceProbeCase<WORKSPACE_LENGTH_UNCHECKED>,
        >()),
        sorted(&workspace_regressor_report::<
            WorkspaceProbeCase<WORKSPACE_LEAKS>,
        >()),
    ] {
        tripped_workspace.extend(names);
    }

    assert_eq!(
        classifier
            .difference(&tripped_classifier)
            .collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "classifier obligations with no probe that trips them"
    );
    assert_eq!(
        regressor.difference(&tripped_regressor).collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "regressor obligations with no probe that trips them"
    );
    assert_eq!(
        transformer
            .difference(&tripped_transformer)
            .collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "transformer obligations with no probe that trips them"
    );
    assert_eq!(
        workspace.difference(&tripped_workspace).collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "workspace obligations with no probe that trips them"
    );
    assert_eq!(
        tripped_classifier
            .difference(&classifier)
            .collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "the battery reported obligations it never declared"
    );
    assert_eq!(
        tripped_regressor.difference(&regressor).collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "the battery reported obligations it never declared"
    );
    assert_eq!(
        tripped_transformer
            .difference(&transformer)
            .collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "the battery reported obligations it never declared"
    );
    assert_eq!(
        tripped_workspace
            .difference(&workspace_reportable)
            .collect::<Vec<_>>(),
        Vec::<&&str>::new(),
        "the battery reported obligations it never declared"
    );
}
