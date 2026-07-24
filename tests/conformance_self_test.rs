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
    Capabilities, Classifier, Estimator, HasCapabilities, ModelError, Regressor, Transformer,
};
use ferricml::data::{BinaryTargets, DenseMatrix, MatrixView, RegressionTargets, SampleWeights};
use std::cell::Cell;
use std::collections::BTreeSet;

use support::conformance::{
    CLASSIFIER_OBLIGATIONS, ClassifierCase, REGRESSOR_OBLIGATIONS, RegressorCase, Report,
    RoundTrip, SCALAR_CLASSIFIER_OBLIGATIONS, SCALAR_REGRESSOR_OBLIGATIONS, ScalarClassifierCase,
    ScalarRegressorCase, TRANSFORMER_OBLIGATIONS, TransformerCase, batch_classifier_report,
    classifier_report, regressor_report, transformer_report,
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

const BASE_THRESHOLD: f32 = 3.5;
const BASE_OFFSET: f32 = 0.0;
const BASE_SCALE: f32 = 2.0;
const FABRICATED_ARTIFACT: &[u8] = b"probe";

const fn probe_capabilities(fault: u8) -> Capabilities {
    Capabilities::NONE
        .with_sample_weights(fault != WEIGHT_HOOK_WITHOUT_DECLARATION)
        .with_artifact(fault != ARTIFACT_HOOK_WITHOUT_DECLARATION)
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

// ---------------------------------------------------------------- classifier

#[derive(Clone, Debug, PartialEq)]
struct ClassifierProbe<const FAULT: u8> {
    features: usize,
    classes: Vec<u8>,
    threshold: f32,
}

impl<const FAULT: u8> ClassifierProbe<FAULT> {
    fn with_threshold(data: &MatrixView<'_>, threshold: f32) -> Self {
        Self {
            features: data.columns(),
            classes: vec![0, 1],
            threshold,
        }
    }

    fn positive(&self, row: &[f32]) -> f32 {
        if row[0] > self.threshold { 0.75 } else { 0.25 }
    }

    fn label(&self, row: &[f32]) -> u8 {
        let honest = u8::from(self.positive(row) > 0.5);
        if FAULT == LABEL_IGNORES_PROBABILITY {
            1 - honest
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

    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.check_width(data.columns())?;
        self.check_len(output.len(), data.rows() * self.classes.len())?;
        for (slot, row) in output.chunks_mut(self.classes.len()).zip(data.iter_rows()) {
            let positive = self.positive(row);
            if slot.len() == self.classes.len() {
                slot[0] = 1.0 - positive;
                slot[1] = positive;
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
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            let positive = self.positive(row);
            let honest = if class == 1 { positive } else { 1.0 - positive };
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

    fn fit(data: &MatrixView<'_>, _labels: &BinaryTargets) -> Self::Model {
        let drift = if FAULT == NONDETERMINISTIC_FIT {
            next_drift()
        } else {
            FIRST_DRIFT
        };
        ClassifierProbe::with_threshold(data, BASE_THRESHOLD + drift)
    }

    fn fit_weighted(
        data: &MatrixView<'_>,
        _labels: &BinaryTargets,
        _weights: &SampleWeights,
    ) -> Option<Self::Model> {
        match FAULT {
            WEIGHTS_DECLARED_WITHOUT_HOOK => None,
            WEIGHTED_FIT_DIFFERS => Some(ClassifierProbe::with_threshold(data, 0.0)),
            _ => Some(ClassifierProbe::with_threshold(
                data,
                BASE_THRESHOLD + FIRST_DRIFT,
            )),
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

    fn fit(data: &MatrixView<'_>, _values: &RegressionTargets) -> Self::Model {
        let drift = if FAULT == NONDETERMINISTIC_FIT {
            next_drift()
        } else {
            FIRST_DRIFT
        };
        RegressorProbe::with_offset(data, BASE_OFFSET + drift)
    }

    fn fit_weighted(
        data: &MatrixView<'_>,
        _values: &RegressionTargets,
        _weights: &SampleWeights,
    ) -> Option<Self::Model> {
        match FAULT {
            WEIGHTS_DECLARED_WITHOUT_HOOK => None,
            WEIGHTED_FIT_DIFFERS => Some(RegressorProbe::with_offset(data, 1.0)),
            _ => Some(RegressorProbe::with_offset(data, BASE_OFFSET + FIRST_DRIFT)),
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

    fn fit(data: &MatrixView<'_>) -> Self::Model {
        let drift = if FAULT == NONDETERMINISTIC_FIT {
            next_drift()
        } else {
            FIRST_DRIFT
        };
        TransformerProbe::with_scale(data, BASE_SCALE + drift)
    }

    fn fit_weighted(data: &MatrixView<'_>, _weights: &SampleWeights) -> Option<Self::Model> {
        match FAULT {
            WEIGHTS_DECLARED_WITHOUT_HOOK => None,
            WEIGHTED_FIT_DIFFERS => Some(TransformerProbe::with_scale(data, 3.0)),
            _ => Some(TransformerProbe::with_scale(data, BASE_SCALE + FIRST_DRIFT)),
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

// ------------------------------------------------------------------- proofs

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
    classifier_predict_failure,
    classifier_report::<ClassifierProbeCase<PREDICT_FAILS>>(),
    ["predicts_the_fixture"]
);
violates!(
    classifier_wrong_metadata,
    classifier_report::<ClassifierProbeCase<WRONG_METADATA>>(),
    ["metadata"]
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

    let mut tripped_classifier = BTreeSet::new();
    for names in [
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
        sorted(&classifier_report::<ClassifierProbeCase<SCALAR_DISAGREES>>()),
        sorted(&classifier_report::<ClassifierProbeCase<NON_FINITE_ACCEPTED>>()),
    ] {
        tripped_classifier.extend(names);
    }

    let mut tripped_regressor = BTreeSet::new();
    for names in [
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
}
