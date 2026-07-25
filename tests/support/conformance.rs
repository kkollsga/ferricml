//! The generic estimator conformance battery.
//!
//! Every fitted estimator owes the same structural obligations: shapes and
//! column order, validation before any write, caller-owned output identical to
//! the allocating convenience method, atomic batch failure, scalar and batch
//! agreement, non-finite rejection, and a deterministic refit. Writing those
//! per estimator makes each new estimator an opportunity to skip one silently.
//! This module states them once, generically, and every estimator is
//! registered into it.
//!
//! # Scope
//!
//! Structural obligations only. Numerical quality and FerricML's frozen
//! reference semantics stay with `reference_semantics.rs` and the artifact
//! fingerprint fixtures; this battery deliberately asserts nothing about
//! whether a prediction is *good*, only that the contract around it holds.
//!
//! # Registering an estimator
//!
//! Implement [`ClassifierCase`], [`RegressorCase`], or [`TransformerCase`] for
//! a unit type describing how to fit the estimator, add the scalar extension
//! trait if it has a scalar prediction path, and add one line to the
//! registration list calling [`check_classifier`], [`check_regressor`], or
//! [`check_transformer`].
//!
//! Optional obligations are not opt-in. They are selected by the estimator's
//! own [`Capabilities`] declaration: a case that declares a capability without
//! supplying the corresponding hook fails, and so does a case that supplies a
//! hook it never declared. A declaration that disagrees with behavior is
//! therefore a test failure rather than a comment nobody rechecks.

use std::fmt::Write as _;

use ferricml::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, ModelError, Regressor, Transformer,
};
use ferricml::artifact::ArtifactError;
use ferricml::data::{
    BinaryTargets, ClassTargets, DenseMatrix, MatrixView, RegressionTargets, SampleWeights,
};

/// Schema identity used for every artifact obligation in the battery.
pub const SCHEMA: [u8; 32] = [42; 32];

/// Obligations every registered classifier owes.
pub const CLASSIFIER_OBLIGATIONS: &[&str] = &[
    "predicts_the_fixture",
    "metadata",
    "into_matches_allocating",
    "probability_columns_follow_classes",
    "label_matches_probability_argmax",
    "feature_width_validated_before_write",
    "output_length_validated_before_write",
    "unknown_class_rejected",
    "refit_is_deterministic",
    "sample_weight_declaration_matches_behavior",
    "artifact_declaration_matches_behavior",
    "multiclass_declaration_matches_behavior",
];

/// Additional obligations owed by a classifier with a scalar prediction path.
pub const SCALAR_CLASSIFIER_OBLIGATIONS: &[&str] =
    &["scalar_matches_batch", "non_finite_scalar_rejected"];

/// Obligations every registered regressor owes.
pub const REGRESSOR_OBLIGATIONS: &[&str] = &[
    "predicts_the_fixture",
    "metadata",
    "into_matches_allocating",
    "feature_width_validated_before_write",
    "output_length_validated_before_write",
    "refit_is_deterministic",
    "sample_weight_declaration_matches_behavior",
    "artifact_declaration_matches_behavior",
];

/// Additional obligations owed by a regressor with a scalar prediction path.
pub const SCALAR_REGRESSOR_OBLIGATIONS: &[&str] =
    &["scalar_matches_batch", "non_finite_scalar_rejected"];

/// Obligations every registered transformer owes.
pub const TRANSFORMER_OBLIGATIONS: &[&str] = &[
    "transforms_the_fixture",
    "metadata",
    "into_matches_allocating",
    "transformed_shape_is_preserved",
    "feature_width_validated_before_write",
    "output_length_validated_before_write",
    "refit_is_deterministic",
    "sample_weight_declaration_matches_behavior",
    "artifact_declaration_matches_behavior",
];

/// A model's artifact round trip: the encoded bytes and the decoded model.
///
/// `None` means the case supplies no round trip, which must agree with the
/// model's [`Capabilities::artifact`] declaration.
pub type RoundTrip<M> = Option<Result<(Vec<u8>, M), ArtifactError>>;

/// One violated obligation and why it was violated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    /// The obligation's name, as listed in the obligation constants.
    pub obligation: &'static str,
    /// What was observed instead.
    pub detail: String,
}

/// Violated obligations collected while running the battery.
#[derive(Debug, Default)]
pub struct Report {
    violations: Vec<Violation>,
}

impl Report {
    fn record(&mut self, obligation: &'static str, detail: String) {
        self.violations.push(Violation { obligation, detail });
    }

    fn require(&mut self, obligation: &'static str, held: bool, detail: impl FnOnce() -> String) {
        if !held {
            self.record(obligation, detail());
        }
    }

    /// Names of every violated obligation, in the order they were found.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.violations
            .iter()
            .map(|violation| violation.obligation)
            .collect()
    }

    /// Whether every obligation held.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    fn assert_clean(&self, name: &str) {
        if self.violations.is_empty() {
            return;
        }
        let mut message = format!("{name} violated {} obligations:", self.violations.len());
        for violation in &self.violations {
            let _ = write!(
                message,
                "\n  - {}: {}",
                violation.obligation, violation.detail
            );
        }
        panic!("{message}");
    }
}

/// The single small dataset every registered estimator is exercised on.
pub struct Fixture {
    /// Eight rows of two well-separated, non-constant features.
    pub data: DenseMatrix,
    /// Balanced binary labels, separable on the fixture.
    pub labels: BinaryTargets,
    /// Three non-contiguous, non-zero-based labels over the same rows.
    ///
    /// The labels are deliberately `{3, 7, 10}` rather than `{0, 1, 2}`, so a
    /// classifier that assumed contiguous or zero-based classes fails here
    /// instead of passing by coincidence.
    pub class_labels: ClassTargets,
    /// Monotone regression targets.
    pub values: RegressionTargets,
}

impl Fixture {
    /// Builds the fixture.
    ///
    /// # Panics
    ///
    /// If the hard-coded fixture stops being a valid dense matrix or target
    /// vector, which is a defect in this file rather than in an estimator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: DenseMatrix::new(
                vec![
                    0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 7.0, 5.0, 6.0, 6.0, 5.0, 7.0, 4.0,
                ],
                8,
                2,
            )
            .expect("fixture matrix"),
            labels: BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).expect("fixture labels"),
            class_labels: ClassTargets::new(vec![7, 7, 3, 3, 10, 10, 3, 7])
                .expect("fixture class labels"),
            values: RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
                .expect("fixture values"),
        }
    }

    /// A matrix with the fixture's row count but a different feature width.
    ///
    /// # Panics
    ///
    /// If the hard-coded matrix stops being valid.
    #[must_use]
    pub fn wrong_width(&self) -> DenseMatrix {
        DenseMatrix::new(vec![1.0; self.data.rows() * 3], self.data.rows(), 3)
            .expect("wrong-width matrix")
    }

    /// Unit sample weights, one per fixture row.
    ///
    /// # Panics
    ///
    /// If the hard-coded weights stop being valid.
    #[must_use]
    pub fn unit_weights(&self) -> SampleWeights {
        SampleWeights::new(vec![1.0; self.data.rows()]).expect("unit weights")
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

/// A classifier registered into the battery.
pub trait ClassifierCase {
    /// The fitted classifier type under test.
    type Model: Classifier + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Fits the model on the battery fixture.
    fn fit(data: &MatrixView<'_>, labels: &BinaryTargets) -> Self::Model;

    /// Fits with sample weights.
    ///
    /// Required exactly when the model declares
    /// [`Capabilities::sample_weights`]; supplying it otherwise is a violation.
    fn fit_weighted(
        _data: &MatrixView<'_>,
        _labels: &BinaryTargets,
        _weights: &SampleWeights,
    ) -> Option<Self::Model> {
        None
    }

    /// Fits over an arbitrary observed class set.
    ///
    /// Required exactly when the model declares
    /// [`Capabilities::multiclass`]; supplying it otherwise is a violation.
    fn fit_multiclass(_data: &MatrixView<'_>, _labels: &ClassTargets) -> Option<Self::Model> {
        None
    }

    /// Encodes and decodes the model through its artifact.
    ///
    /// Required exactly when the model declares [`Capabilities::artifact`];
    /// supplying it otherwise is a violation.
    fn round_trip(_model: &Self::Model) -> RoundTrip<Self::Model> {
        None
    }
}

/// A registered classifier that also offers a scalar prediction path.
///
/// Runtime dispatch enums deliberately have none: they match once per batch.
/// Everything else does, and wiring it here is what keeps the scalar
/// obligations from being skipped by omission.
pub trait ScalarClassifierCase: ClassifierCase {
    /// Predicts one label for one row.
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError>;
}

/// A regressor registered into the battery.
pub trait RegressorCase {
    /// The fitted regressor type under test.
    type Model: Regressor + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Fits the model on the battery fixture.
    fn fit(data: &MatrixView<'_>, values: &RegressionTargets) -> Self::Model;

    /// Fits with sample weights. Required exactly when declared.
    fn fit_weighted(
        _data: &MatrixView<'_>,
        _values: &RegressionTargets,
        _weights: &SampleWeights,
    ) -> Option<Self::Model> {
        None
    }

    /// Encodes and decodes the model. Required exactly when declared.
    fn round_trip(_model: &Self::Model) -> RoundTrip<Self::Model> {
        None
    }
}

/// A registered regressor that also offers a scalar prediction path.
pub trait ScalarRegressorCase: RegressorCase {
    /// Predicts one value for one row.
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError>;
}

/// A transformer registered into the battery.
pub trait TransformerCase {
    /// The fitted transformer type under test.
    type Model: Transformer + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Fits the transformer on the battery fixture.
    fn fit(data: &MatrixView<'_>) -> Self::Model;

    /// Fits with sample weights. Required exactly when declared.
    fn fit_weighted(_data: &MatrixView<'_>, _weights: &SampleWeights) -> Option<Self::Model> {
        None
    }

    /// Encodes and decodes the transformer. Required exactly when declared.
    fn round_trip(_model: &Self::Model) -> RoundTrip<Self::Model> {
        None
    }
}

/// Runs every obligation a classifier with a scalar path owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_classifier<C: ScalarClassifierCase>() {
    let report = classifier_report::<C>();
    report.assert_clean(C::NAME);
}

/// Runs every obligation a classifier without a scalar path owes.
///
/// Prefer [`check_classifier`]. This weaker entry point exists only for the
/// runtime dispatch enums, which have no scalar prediction path at all.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_batch_only_classifier<C: ClassifierCase>() {
    let report = batch_classifier_report::<C>();
    report.assert_clean(C::NAME);
}

/// Runs every obligation a regressor with a scalar path owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_regressor<R: ScalarRegressorCase>() {
    let report = regressor_report::<R>();
    report.assert_clean(R::NAME);
}

/// Runs every obligation a regressor without a scalar path owes.
///
/// Prefer [`check_regressor`]; see [`check_batch_only_classifier`].
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_batch_only_regressor<R: RegressorCase>() {
    let report = batch_regressor_report::<R>();
    report.assert_clean(R::NAME);
}

/// Runs every obligation a transformer owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_transformer<T: TransformerCase>() {
    let report = transformer_report::<T>();
    report.assert_clean(T::NAME);
}

/// Collects violations for a classifier with a scalar path.
#[must_use]
pub fn classifier_report<C: ScalarClassifierCase>() -> Report {
    let fixture = Fixture::new();
    let mut report = batch_classifier_report::<C>();
    let model = C::fit(&fixture.data.as_view(), &fixture.labels);
    scalar_classifier_obligations::<C>(&mut report, &fixture, &model);
    report
}

/// Collects violations for a classifier without a scalar path.
#[must_use]
pub fn batch_classifier_report<C: ClassifierCase>() -> Report {
    let fixture = Fixture::new();
    let mut report = Report::default();
    let model = C::fit(&fixture.data.as_view(), &fixture.labels);
    let view = fixture.data.as_view();

    let classes = model.classes().to_vec();
    report.require(
        "metadata",
        model.n_features_in() == fixture.data.columns(),
        || {
            format!(
                "n_features_in is {} but the fixture has {} columns",
                model.n_features_in(),
                fixture.data.columns()
            )
        },
    );
    report.require("metadata", !classes.is_empty(), || {
        "classes() is empty".to_owned()
    });
    report.require(
        "metadata",
        classes.windows(2).all(|pair| pair[0] < pair[1]),
        || format!("classes() is not strictly ascending: {classes:?}"),
    );

    let labels = model.predict(&view);
    let probabilities = model.predict_proba(&view);
    let (Ok(labels), Ok(probabilities)) = (&labels, &probabilities) else {
        report.record(
            "predicts_the_fixture",
            format!("predict returned {labels:?} and predict_proba returned {probabilities:?}"),
        );
        return report;
    };
    report.require("predicts_the_fixture", labels.len() == view.rows(), || {
        format!(
            "predict returned {} values for {} rows",
            labels.len(),
            view.rows()
        )
    });
    let expected_probabilities = view.rows() * classes.len();
    report.require(
        "predicts_the_fixture",
        probabilities.len() == expected_probabilities,
        || {
            format!(
                "predict_proba returned {} values, expected {expected_probabilities}",
                probabilities.len()
            )
        },
    );
    if labels.len() != view.rows() || probabilities.len() != expected_probabilities {
        return report;
    }

    let mut labels_into = vec![u8::MAX; view.rows()];
    let written = model.predict_into(&view, &mut labels_into);
    report.require(
        "into_matches_allocating",
        written.is_ok() && &labels_into == labels,
        || {
            format!(
                "predict_into returned {written:?} and wrote {labels_into:?}, expected {labels:?}"
            )
        },
    );
    let mut probabilities_into = vec![f32::MAX; expected_probabilities];
    let written = model.predict_proba_into(&view, &mut probabilities_into);
    report.require(
        "into_matches_allocating",
        written.is_ok() && &probabilities_into == probabilities,
        || format!("predict_proba_into returned {written:?} and disagreed with predict_proba"),
    );
    for (column, &class) in classes.iter().enumerate() {
        let allocating = model.predict_class_proba(&view, class);
        let mut into = vec![f32::MAX; view.rows()];
        let written = model.predict_class_proba_into(&view, class, &mut into);
        report.require(
            "into_matches_allocating",
            written.is_ok() && allocating.as_ref() == Ok(&into),
            || format!("class {class} disagreed between predict_class_proba and its _into form"),
        );
        let expected: Vec<f32> = probabilities
            .chunks_exact(classes.len())
            .map(|row| row[column])
            .collect();
        report.require(
            "probability_columns_follow_classes",
            allocating.as_ref() == Ok(&expected),
            || {
                format!(
                    "class {class} is not column {column} of predict_proba: {allocating:?} vs {expected:?}"
                )
            },
        );
    }
    for (index, row) in probabilities.chunks_exact(classes.len()).enumerate() {
        let total: f32 = row.iter().sum();
        report.require(
            "probability_columns_follow_classes",
            (total - 1.0).abs() <= 1.0e-6,
            || format!("row {index} probabilities sum to {total}: {row:?}"),
        );
    }

    for (index, row) in probabilities.chunks_exact(classes.len()).enumerate() {
        let expected = classes[argmax_first_wins(row)];
        report.require(
            "label_matches_probability_argmax",
            labels[index] == expected,
            || {
                format!(
                    "row {index} predicted {} but its probabilities {row:?} favor {expected}",
                    labels[index]
                )
            },
        );
    }

    let wrong_width = fixture.wrong_width();
    let mut untouched = vec![7_u8; view.rows()];
    let rejected = model.predict_into(&wrong_width.as_view(), &mut untouched);
    report.require(
        "feature_width_validated_before_write",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: fixture.data.columns(),
                actual: 3,
            })
            && untouched == vec![7_u8; view.rows()],
        || {
            format!(
                "predict_into on a wrong-width batch returned {rejected:?} and wrote {untouched:?}"
            )
        },
    );
    let mut untouched = vec![7.0_f32; expected_probabilities];
    let rejected = model.predict_proba_into(&wrong_width.as_view(), &mut untouched);
    report.require(
        "feature_width_validated_before_write",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: fixture.data.columns(),
                actual: 3,
            })
            && untouched == vec![7.0_f32; expected_probabilities],
        || format!("predict_proba_into on a wrong-width batch returned {rejected:?}"),
    );

    check_output_length::<u8>(&mut report, view.rows(), 7, |output| {
        model.predict_into(&view, output)
    });
    check_output_length::<f32>(&mut report, expected_probabilities, 7.0, |output| {
        model.predict_proba_into(&view, output)
    });
    check_output_length::<f32>(&mut report, view.rows(), 7.0, |output| {
        model.predict_class_proba_into(&view, classes[0], output)
    });

    let absent = absent_class(&classes);
    let mut untouched = vec![7.0_f32; view.rows()];
    let rejected = model.predict_class_proba_into(&view, absent, &mut untouched);
    report.require(
        "unknown_class_rejected",
        rejected == Err(ModelError::UnknownClass { class: absent })
            && untouched == vec![7.0_f32; view.rows()],
        || format!("class {absent} returned {rejected:?} and wrote {untouched:?}"),
    );
    let rejected = model.predict_class_proba(&view, absent);
    report.require(
        "unknown_class_rejected",
        rejected == Err(ModelError::UnknownClass { class: absent }),
        || {
            format!(
                "allocating predict_class_proba for absent class {absent} returned {rejected:?}"
            )
        },
    );

    let refitted = C::fit(&view, &fixture.labels);
    report.require(
        "refit_is_deterministic",
        refitted.predict(&view).as_ref() == Ok(labels)
            && refitted.predict_proba(&view).as_ref() == Ok(probabilities),
        || "refitting the same data and parameters changed the predictions".to_owned(),
    );

    let weighted = C::fit_weighted(&view, &fixture.labels, &fixture.unit_weights());
    check_weight_declaration(
        &mut report,
        C::Model::CAPABILITIES,
        weighted.is_some(),
        weighted.map(|weighted| {
            (
                weighted.predict(&view).as_ref() == Ok(labels)
                    && weighted.predict_proba(&view).as_ref() == Ok(probabilities),
                "unit-weighted fit predicts differently from the unweighted fit".to_owned(),
            )
        }),
    );

    check_multiclass_declaration::<C>(&mut report, &fixture);

    let round_tripped = C::round_trip(&model);
    check_artifact_declaration(
        &mut report,
        C::Model::CAPABILITIES,
        round_tripped.is_some(),
        round_tripped.map(|outcome| match (outcome, C::round_trip(&model)) {
            (Ok((bytes, decoded)), Some(Ok((again, _)))) => (
                bytes == again
                    && decoded.predict(&view).as_ref() == Ok(labels)
                    && decoded.predict_proba(&view).as_ref() == Ok(probabilities),
                "the artifact re-encoded differently or the decoded model predicts differently"
                    .to_owned(),
            ),
            (Err(error), _) => (false, format!("the artifact round trip failed: {error:?}")),
            (Ok(_), _) => (
                false,
                "the artifact round trip did not repeat successfully".to_owned(),
            ),
        }),
    );
    check_multiclass_artifact_declaration::<C>(&mut report, &fixture);

    report
}

/// A declared capability holds for every fit the type offers.
///
/// `artifact` is a property of the estimator *type*, not of one fitted value,
/// so a classifier declaring both persistence and multiclass fitting owes a
/// working round trip for a multiclass fit too. Without this the flag could
/// quietly mean "some fits persist", and a caller reading it before choosing an
/// estimator would be told something untrue of the model it is about to train.
fn check_multiclass_artifact_declaration<C: ClassifierCase>(
    report: &mut Report,
    fixture: &Fixture,
) {
    let capabilities = C::Model::CAPABILITIES;
    if !(capabilities.artifact() && capabilities.multiclass()) {
        return;
    }
    let view = fixture.data.as_view();
    let Some(model) = C::fit_multiclass(&view, &fixture.class_labels) else {
        return;
    };
    let (Ok(labels), Ok(probabilities)) = (model.predict(&view), model.predict_proba(&view)) else {
        return;
    };
    let detail = match (C::round_trip(&model), C::round_trip(&model)) {
        (Some(Ok((bytes, decoded))), Some(Ok((again, _)))) => {
            if bytes != again {
                Some("the multiclass artifact did not re-encode to the same bytes".to_owned())
            } else if decoded.classes() != model.classes() {
                Some(format!(
                    "the decoded multiclass model reports classes {:?} instead of {:?}",
                    decoded.classes(),
                    model.classes()
                ))
            } else if decoded.predict(&view).as_ref() != Ok(&labels)
                || decoded.predict_proba(&view).as_ref() != Ok(&probabilities)
            {
                Some("the decoded multiclass model predicts differently".to_owned())
            } else {
                None
            }
        }
        (Some(Err(error)), _) => Some(format!(
            "artifact is declared but a multiclass fit does not persist: {error:?}"
        )),
        _ => Some("the multiclass artifact round trip did not repeat successfully".to_owned()),
    };
    report.require(
        "artifact_declaration_matches_behavior",
        detail.is_none(),
        || detail.unwrap_or_default(),
    );
}

fn scalar_classifier_obligations<C: ScalarClassifierCase>(
    report: &mut Report,
    fixture: &Fixture,
    model: &C::Model,
) {
    let view = fixture.data.as_view();
    let Ok(labels) = model.predict(&view) else {
        return;
    };
    for (index, row) in view.iter_rows().enumerate() {
        let scalar = C::predict_one(model, row);
        report.require("scalar_matches_batch", scalar == Ok(labels[index]), || {
            format!(
                "row {index} predicted {scalar:?} scalar but {} in batch",
                labels[index]
            )
        });
    }
    check_non_finite_scalar(report, fixture.data.columns(), |row| {
        C::predict_one(model, row).map(|_| ())
    });
}

/// Collects violations for a regressor with a scalar path.
#[must_use]
pub fn regressor_report<R: ScalarRegressorCase>() -> Report {
    let fixture = Fixture::new();
    let mut report = batch_regressor_report::<R>();
    let model = R::fit(&fixture.data.as_view(), &fixture.values);
    let view = fixture.data.as_view();
    if let Ok(values) = model.predict(&view) {
        for (index, row) in view.iter_rows().enumerate() {
            let scalar = R::predict_one(&model, row);
            report.require("scalar_matches_batch", scalar == Ok(values[index]), || {
                format!(
                    "row {index} predicted {scalar:?} scalar but {} in batch",
                    values[index]
                )
            });
        }
    }
    check_non_finite_scalar(&mut report, fixture.data.columns(), |row| {
        R::predict_one(&model, row).map(|_| ())
    });
    report
}

/// Collects violations for a regressor without a scalar path.
#[must_use]
pub fn batch_regressor_report<R: RegressorCase>() -> Report {
    let fixture = Fixture::new();
    let mut report = Report::default();
    let model = R::fit(&fixture.data.as_view(), &fixture.values);
    let view = fixture.data.as_view();

    report.require(
        "metadata",
        model.n_features_in() == fixture.data.columns(),
        || {
            format!(
                "n_features_in is {} but the fixture has {} columns",
                model.n_features_in(),
                fixture.data.columns()
            )
        },
    );

    let values = model.predict(&view);
    let Ok(values) = &values else {
        report.record(
            "predicts_the_fixture",
            format!("predict returned {values:?}"),
        );
        return report;
    };
    report.require("predicts_the_fixture", values.len() == view.rows(), || {
        format!(
            "predict returned {} values for {} rows",
            values.len(),
            view.rows()
        )
    });
    if values.len() != view.rows() {
        return report;
    }

    let mut into = vec![f32::MAX; view.rows()];
    let written = model.predict_into(&view, &mut into);
    report.require(
        "into_matches_allocating",
        written.is_ok() && &into == values,
        || format!("predict_into returned {written:?} and disagreed with predict"),
    );

    let wrong_width = fixture.wrong_width();
    let mut untouched = vec![7.0_f32; view.rows()];
    let rejected = model.predict_into(&wrong_width.as_view(), &mut untouched);
    report.require(
        "feature_width_validated_before_write",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: fixture.data.columns(),
                actual: 3,
            })
            && untouched == vec![7.0_f32; view.rows()],
        || {
            format!(
                "predict_into on a wrong-width batch returned {rejected:?} and wrote {untouched:?}"
            )
        },
    );

    check_output_length::<f32>(&mut report, view.rows(), 7.0, |output| {
        model.predict_into(&view, output)
    });

    let refitted = R::fit(&view, &fixture.values);
    report.require(
        "refit_is_deterministic",
        refitted.predict(&view).as_ref() == Ok(values),
        || "refitting the same data and parameters changed the predictions".to_owned(),
    );

    let weighted = R::fit_weighted(&view, &fixture.values, &fixture.unit_weights());
    check_weight_declaration(
        &mut report,
        R::Model::CAPABILITIES,
        weighted.is_some(),
        weighted.map(|weighted| {
            (
                weighted.predict(&view).as_ref() == Ok(values),
                "unit-weighted fit predicts differently from the unweighted fit".to_owned(),
            )
        }),
    );

    let round_tripped = R::round_trip(&model);
    check_artifact_declaration(
        &mut report,
        R::Model::CAPABILITIES,
        round_tripped.is_some(),
        round_tripped.map(|outcome| match (outcome, R::round_trip(&model)) {
            (Ok((bytes, decoded)), Some(Ok((again, _)))) => (
                bytes == again && decoded.predict(&view).as_ref() == Ok(values),
                "the artifact re-encoded differently or the decoded model predicts differently"
                    .to_owned(),
            ),
            (Err(error), _) => (false, format!("the artifact round trip failed: {error:?}")),
            (Ok(_), _) => (
                false,
                "the artifact round trip did not repeat successfully".to_owned(),
            ),
        }),
    );

    report
}

/// Collects violations for a transformer.
#[must_use]
pub fn transformer_report<T: TransformerCase>() -> Report {
    let fixture = Fixture::new();
    let mut report = Report::default();
    let model = T::fit(&fixture.data.as_view());
    let view = fixture.data.as_view();
    let columns = model.n_features_out();

    report.require(
        "metadata",
        model.n_features_in() == fixture.data.columns(),
        || {
            format!(
                "n_features_in is {} but the fixture has {} columns",
                model.n_features_in(),
                fixture.data.columns()
            )
        },
    );
    report.require("metadata", columns > 0, || {
        "n_features_out is zero".to_owned()
    });

    let allocating = model.transform(&view);
    let Ok(allocating) = &allocating else {
        report.record(
            "transforms_the_fixture",
            format!("transform returned {allocating:?}"),
        );
        return report;
    };
    let transformed = allocating.as_slice().to_vec();
    report.require(
        "transformed_shape_is_preserved",
        allocating.rows() == view.rows() && allocating.columns() == columns,
        || {
            format!(
                "transform produced {}x{}, expected {}x{columns}",
                allocating.rows(),
                allocating.columns(),
                view.rows()
            )
        },
    );

    let expected_len = view.rows() * columns;
    let mut into = vec![f32::MAX; expected_len];
    let written = model.transform_into(&view, &mut into);
    match &written {
        Ok(matrix) => {
            report.require(
                "into_matches_allocating",
                matrix.as_slice() == transformed.as_slice(),
                || "transform_into wrote different values from transform".to_owned(),
            );
            report.require(
                "transformed_shape_is_preserved",
                matrix.rows() == view.rows() && matrix.columns() == columns,
                || {
                    format!(
                        "transform_into returned a {}x{} view, expected {}x{columns}",
                        matrix.rows(),
                        matrix.columns(),
                        view.rows()
                    )
                },
            );
        }
        Err(_) => report.record(
            "transforms_the_fixture",
            format!("transform_into returned {written:?}"),
        ),
    }

    let wrong_width = fixture.wrong_width();
    let mut untouched = vec![7.0_f32; expected_len];
    let rejected = model
        .transform_into(&wrong_width.as_view(), &mut untouched)
        .map(|_| ());
    report.require(
        "feature_width_validated_before_write",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: fixture.data.columns(),
                actual: 3,
            })
            && untouched == vec![7.0_f32; expected_len],
        || format!("transform_into on a wrong-width batch returned {rejected:?}"),
    );

    check_output_length::<f32>(&mut report, expected_len, 7.0, |output| {
        model.transform_into(&view, output).map(|_| ())
    });

    let refitted = T::fit(&view);
    report.require(
        "refit_is_deterministic",
        refitted
            .transform(&view)
            .map(|matrix| matrix.as_slice().to_vec())
            == Ok(transformed.clone()),
        || "refitting the same data and parameters changed the transform".to_owned(),
    );

    let weighted = T::fit_weighted(&view, &fixture.unit_weights());
    check_weight_declaration(
        &mut report,
        T::Model::CAPABILITIES,
        weighted.is_some(),
        weighted.map(|weighted| {
            (
                weighted
                    .transform(&view)
                    .map(|matrix| matrix.as_slice().to_vec())
                    == Ok(transformed.clone()),
                "unit-weighted fit transforms differently from the unweighted fit".to_owned(),
            )
        }),
    );

    let round_tripped = T::round_trip(&model);
    check_artifact_declaration(
        &mut report,
        T::Model::CAPABILITIES,
        round_tripped.is_some(),
        round_tripped.map(|outcome| match (outcome, T::round_trip(&model)) {
            (Ok((bytes, decoded)), Some(Ok((again, _)))) => (
                bytes == again
                    && decoded
                        .transform(&view)
                        .map(|matrix| matrix.as_slice().to_vec())
                        == Ok(transformed.clone()),
                "the artifact re-encoded differently or the decoded transformer differs".to_owned(),
            ),
            (Err(error), _) => (false, format!("the artifact round trip failed: {error:?}")),
            (Ok(_), _) => (
                false,
                "the artifact round trip did not repeat successfully".to_owned(),
            ),
        }),
    );

    report
}

fn check_output_length<T: Clone + PartialEq + std::fmt::Debug>(
    report: &mut Report,
    expected: usize,
    sentinel: T,
    mut call: impl FnMut(&mut [T]) -> Result<(), ModelError>,
) {
    for actual in [expected.saturating_sub(1), expected + 1] {
        if actual == expected {
            continue;
        }
        let mut output = vec![sentinel.clone(); actual];
        let rejected = call(&mut output);
        report.require(
            "output_length_validated_before_write",
            rejected == Err(ModelError::OutputLength { expected, actual })
                && output == vec![sentinel.clone(); actual],
            || {
                format!(
                    "an output buffer of {actual} for {expected} values returned {rejected:?} \
                     and left {output:?}"
                )
            },
        );
    }
}

fn check_non_finite_scalar(
    report: &mut Report,
    columns: usize,
    mut call: impl FnMut(&[f32]) -> Result<(), ModelError>,
) {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for column in 0..columns {
            let mut row = vec![1.0_f32; columns];
            row[column] = value;
            let rejected = call(&row);
            report.require(
                "non_finite_scalar_rejected",
                rejected == Err(ModelError::NonFiniteFeature { row: 0, column }),
                || format!("a {value} in column {column} returned {rejected:?}"),
            );
        }
    }
    let rejected = call(&vec![1.0_f32; columns + 1]);
    report.require(
        "non_finite_scalar_rejected",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: columns,
                actual: columns + 1,
            }),
        || format!("a scalar row of the wrong width returned {rejected:?}"),
    );
}

/// A declared multiclass fit must exist and must be genuinely multiclass.
///
/// The obligation is not "does it return something": it is that the fitted
/// model reports exactly the fixture's observed labels in sorted order, that
/// its probability matrix has one column per label, that each column is the
/// one `predict_class_proba` returns for that label, that every row sums to
/// one within the documented tolerance, and that the label is the argmax of
/// that row. A model that quietly collapsed the class set, permuted the
/// columns, or renormalized would fail one of those.
fn check_multiclass_declaration<C: ClassifierCase>(report: &mut Report, fixture: &Fixture) {
    let view = fixture.data.as_view();
    let declared = C::Model::CAPABILITIES.multiclass();
    let fitted = C::fit_multiclass(&view, &fixture.class_labels);
    report.require(
        "multiclass_declaration_matches_behavior",
        declared == fitted.is_some(),
        || {
            format!(
                "declares multiclass = {declared} but {} a multiclass-fit hook",
                if fitted.is_some() {
                    "supplies"
                } else {
                    "supplies no"
                }
            )
        },
    );
    let Some(model) = fitted.filter(|_| declared) else {
        return;
    };

    let classes = model.classes().to_vec();
    report.require(
        "multiclass_declaration_matches_behavior",
        classes == fixture.class_labels.classes(),
        || {
            format!(
                "multiclass fit reports classes {classes:?}, expected {:?}",
                fixture.class_labels.classes()
            )
        },
    );
    let (Ok(labels), Ok(probabilities)) = (model.predict(&view), model.predict_proba(&view)) else {
        report.record(
            "multiclass_declaration_matches_behavior",
            "the multiclass fit failed to predict on the data it was fitted on".to_owned(),
        );
        return;
    };
    if probabilities.len() != view.rows() * classes.len() || labels.len() != view.rows() {
        report.record(
            "multiclass_declaration_matches_behavior",
            format!(
                "multiclass fit produced {} labels and {} probabilities for {} rows and {} classes",
                labels.len(),
                probabilities.len(),
                view.rows(),
                classes.len()
            ),
        );
        return;
    }
    for (column, &class) in classes.iter().enumerate() {
        let expected: Vec<f32> = probabilities
            .chunks_exact(classes.len())
            .map(|row| row[column])
            .collect();
        report.require(
            "multiclass_declaration_matches_behavior",
            model.predict_class_proba(&view, class).as_ref() == Ok(&expected),
            || format!("multiclass class {class} is not column {column} of predict_proba"),
        );
    }
    let tolerance = classes.len() as f32 * f32::EPSILON;
    for (index, row) in probabilities.chunks_exact(classes.len()).enumerate() {
        let total: f32 = row.iter().sum();
        report.require(
            "multiclass_declaration_matches_behavior",
            (total - 1.0).abs() <= tolerance,
            || format!("multiclass row {index} sums to {total}: {row:?}"),
        );
        report.require(
            "multiclass_declaration_matches_behavior",
            labels[index] == classes[argmax_first_wins(row)],
            || {
                format!(
                    "multiclass row {index} predicted {} but its probabilities {row:?} favor {}",
                    labels[index],
                    classes[argmax_first_wins(row)]
                )
            },
        );
    }
}

fn check_weight_declaration(
    report: &mut Report,
    capabilities: Capabilities,
    supplied: bool,
    outcome: Option<(bool, String)>,
) {
    report.require(
        "sample_weight_declaration_matches_behavior",
        capabilities.sample_weights() == supplied,
        || {
            format!(
                "declares sample_weights = {} but {} a weighted-fit hook",
                capabilities.sample_weights(),
                if supplied { "supplies" } else { "supplies no" }
            )
        },
    );
    if let Some((held, detail)) = outcome
        && capabilities.sample_weights()
    {
        report.require("sample_weight_declaration_matches_behavior", held, || {
            detail
        });
    }
}

fn check_artifact_declaration(
    report: &mut Report,
    capabilities: Capabilities,
    supplied: bool,
    outcome: Option<(bool, String)>,
) {
    report.require(
        "artifact_declaration_matches_behavior",
        capabilities.artifact() == supplied,
        || {
            format!(
                "declares artifact = {} but {} a round-trip hook",
                capabilities.artifact(),
                if supplied { "supplies" } else { "supplies no" }
            )
        },
    );
    if let Some((held, detail)) = outcome
        && capabilities.artifact()
    {
        report.require("artifact_declaration_matches_behavior", held, || detail);
    }
}

fn argmax_first_wins(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in row.iter().enumerate().skip(1) {
        if *value > row[best] {
            best = index;
        }
    }
    best
}

fn absent_class(classes: &[u8]) -> u8 {
    (0..=u8::MAX)
        .find(|candidate| !classes.contains(candidate))
        .expect("the fixture never observes every u8 label")
}
