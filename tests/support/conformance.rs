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
//! A model whose prediction path takes a caller-owned workspace — a
//! `Pipeline`, a `StagedPipeline` — implements [`WorkspaceClassifierCase`] or
//! [`WorkspaceRegressorCase`] instead, and is registered through
//! [`check_workspace_classifier`] or [`check_workspace_regressor`]. Those
//! traits relax the model bound to [`Estimator`] and carry the batch surface
//! themselves, because a fitted composition deliberately has no trait-shaped
//! prediction method: its intermediate batch lives in storage the caller owns
//! and reuses, which is the whole point of the composition.
//!
//! # Fixtures
//!
//! A case declares which [`FixtureShape`] it is exercised on, defaulting to the
//! two-feature one. Each shape supplies a `train` and a genuinely disjoint
//! `holdout` [`Sample`], and each sample carries features together with every
//! target flavour over the same rows. That is what lets one `fit` signature
//! serve an unsupervised transformer, a supervised one, a regressor, and a
//! meta-estimator whose second fitting stage needs rows its inner model has
//! never seen.
//!
//! # Optional obligations are declaration-selected
//!
//! Optional obligations are not opt-in. They are selected by the estimator's
//! own [`Capabilities`] declaration: a case that declares a capability without
//! supplying the corresponding hook fails, and so does a case that supplies a
//! hook it never declared. A declaration that disagrees with behavior is
//! therefore a test failure rather than a comment nobody rechecks.
//! `check_declaration` states that pattern once, so a capability added later
//! is one call rather than a new bespoke check.
//!
//! # A note for whoever answers D11
//!
//! Probability production is currently *mandatory*: `predict_proba_into` is a
//! required method of [`Classifier`] with no default body, so every classifier
//! has one and no declaration could vary. Everything that follows from it is
//! deliberately grouped in `probability_obligations`, and reaches the model
//! only through the driver's four probability methods. If probabilities ever
//! stop being mandatory, those four become the group a declaration selects and
//! that one function gains the gate. Nothing else in this file moves.

use std::fmt::Write as _;
use std::marker::PhantomData;

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
    "fits_the_fixture",
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
    "fits_the_fixture",
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
    "fits_the_fixture",
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

/// Additional obligations owed by a model predicted through a caller-owned
/// workspace.
///
/// These are the composition hazards no trait-shaped estimator can have: a
/// workspace whose length is never checked, and one whose contents leak from
/// one batch into the next.
pub const WORKSPACE_OBLIGATIONS: &[&str] = &[
    "workspace_length_validated_before_write",
    "workspace_reuse_is_independent",
];

/// A model's artifact round trip: the encoded bytes and the decoded model.
///
/// `None` means the case supplies no round trip, which must agree with the
/// model's [`Capabilities::artifact`] declaration.
pub type RoundTrip<M> = Option<Result<(Vec<u8>, M), ArtifactError>>;

/// A fit a case supplies only when its model declares the matching capability.
///
/// `None` means "no such entry point". `Some(Err(_))` means the entry point
/// exists but refused the fixture, which is a violation rather than a silent
/// skip.
pub type OptionalFit<M> = Option<Result<M, ModelError>>;

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

// ------------------------------------------------------------------ fixtures

/// Which of the battery's datasets a case is exercised on.
///
/// One fixture shape is itself a constraint on what the battery can hold: a
/// univariate estimator has no row in an eight-by-two world, and could only be
/// registered by handing it a width it is required to reject.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixtureShape {
    /// Eight rows of two well-separated, non-constant features.
    Wide,
    /// Eight rows of one strictly increasing feature.
    Univariate,
}

/// One dataset: features together with every target flavour over its rows.
///
/// Carrying all the targets beside the features is what lets a single `fit`
/// signature serve an unsupervised transformer, a *supervised* one — a feature
/// selector or a discriminant projection fits with a target and transforms
/// without one — a regressor, and a classifier. A case takes the targets it
/// needs and ignores the rest.
pub struct Sample {
    /// The feature matrix.
    pub data: DenseMatrix,
    /// Balanced binary labels, separable on the sample.
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

impl Sample {
    fn new(
        data: Vec<f32>,
        rows: usize,
        columns: usize,
        values: Vec<f32>,
        classes: Vec<u8>,
    ) -> Self {
        Self {
            data: DenseMatrix::new(data, rows, columns).expect("fixture matrix"),
            labels: BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).expect("fixture labels"),
            class_labels: ClassTargets::new(classes).expect("fixture class labels"),
            values: RegressionTargets::new(values).expect("fixture values"),
        }
    }

    /// A borrowed view of the feature matrix.
    #[must_use]
    pub fn view(&self) -> MatrixView<'_> {
        self.data.as_view()
    }

    /// Rows in this sample.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.data.rows()
    }

    /// Feature columns in this sample.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.data.columns()
    }

    /// A matrix with this sample's row count but a different feature width.
    ///
    /// # Panics
    ///
    /// If the derived matrix stops being valid.
    #[must_use]
    pub fn wrong_width(&self) -> DenseMatrix {
        let columns = self.columns() + 1;
        DenseMatrix::new(vec![1.0; self.rows() * columns], self.rows(), columns)
            .expect("wrong-width matrix")
    }

    /// Unit sample weights, one per row.
    ///
    /// # Panics
    ///
    /// If the derived weights stop being valid.
    #[must_use]
    pub fn unit_weights(&self) -> SampleWeights {
        SampleWeights::new(vec![1.0; self.rows()]).expect("unit weights")
    }
}

/// The datasets every registered estimator is exercised on.
///
/// `holdout` shares `train`'s width and observed class set but none of its
/// rows, so a meta-estimator with a held-out component — a calibrated
/// classifier is the shipped case — is fitted the way its own documentation
/// says it must be, rather than on the only rows the battery happened to have.
pub struct Fixture {
    /// The rows every model is fitted on.
    pub train: Sample,
    /// Disjoint rows of the same width, for a second fitting stage.
    pub holdout: Sample,
}

impl Fixture {
    /// Builds the fixture for one shape.
    ///
    /// # Panics
    ///
    /// If the hard-coded fixture stops being a valid dense matrix or target
    /// vector, which is a defect in this file rather than in an estimator.
    #[must_use]
    pub fn new(shape: FixtureShape) -> Self {
        match shape {
            FixtureShape::Wide => Self {
                train: Sample::new(
                    vec![
                        0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 7.0, 5.0, 6.0, 6.0, 5.0, 7.0,
                        4.0,
                    ],
                    8,
                    2,
                    vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
                    vec![7, 7, 3, 3, 10, 10, 3, 7],
                ),
                holdout: Sample::new(
                    vec![
                        0.5, 3.5, 1.5, 2.5, 2.5, 1.5, 3.5, 0.5, 4.5, 7.5, 5.5, 6.5, 6.5, 5.5, 7.5,
                        4.5,
                    ],
                    8,
                    2,
                    vec![0.5, 1.5, 4.5, 9.5, 16.5, 25.5, 36.5, 49.5],
                    vec![3, 7, 10, 7, 3, 10, 7, 3],
                ),
            },
            FixtureShape::Univariate => Self {
                train: Sample::new(
                    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    8,
                    1,
                    vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
                    vec![7, 7, 3, 3, 10, 10, 3, 7],
                ),
                holdout: Sample::new(
                    vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                    8,
                    1,
                    vec![0.5, 1.5, 4.5, 9.5, 16.5, 25.5, 36.5, 49.5],
                    vec![3, 7, 10, 7, 3, 10, 7, 3],
                ),
            },
        }
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new(FixtureShape::Wide)
    }
}

// --------------------------------------------------------------- case traits

/// A classifier registered into the battery.
pub trait ClassifierCase {
    /// The fitted classifier type under test.
    type Model: Classifier + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Which fixture shape this estimator is exercised on.
    const FIXTURE: FixtureShape = FixtureShape::Wide;

    /// Fits the model on the battery fixture.
    ///
    /// `holdout` is disjoint from `train` and exists for a meta-estimator with
    /// a second fitting stage; an ordinary estimator ignores it.
    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;

    /// Fits with sample weights.
    ///
    /// Required exactly when the model declares
    /// [`Capabilities::sample_weights`]; supplying it otherwise is a violation.
    fn fit_weighted(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        None
    }

    /// Fits over an arbitrary observed class set.
    ///
    /// Required exactly when the model declares [`Capabilities::multiclass`];
    /// supplying it otherwise is a violation.
    fn fit_multiclass(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
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

    /// Which fixture shape this estimator is exercised on.
    const FIXTURE: FixtureShape = FixtureShape::Wide;

    /// Fits the model on the battery fixture.
    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;

    /// Fits with sample weights. Required exactly when declared.
    fn fit_weighted(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
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
///
/// `fit` receives whole samples rather than a bare matrix, so a *supervised*
/// fitted transformer — one that fits with a target and transforms without one
/// — registers with the same case shape as an unsupervised one. That is the
/// whole accommodation such a transformer needs: FerricML's traits describe
/// fitted models, never fitting, so its fitted form is an ordinary
/// [`Transformer`] and only the battery had to learn where the target comes
/// from.
pub trait TransformerCase {
    /// The fitted transformer type under test.
    type Model: Transformer + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Which fixture shape this transformer is exercised on.
    const FIXTURE: FixtureShape = FixtureShape::Wide;

    /// Fits the transformer on the battery fixture.
    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;

    /// Fits with sample weights. Required exactly when declared.
    fn fit_weighted(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        None
    }

    /// Encodes and decodes the transformer. Required exactly when declared.
    fn round_trip(_model: &Self::Model) -> RoundTrip<Self::Model> {
        None
    }
}

/// A classifier-shaped model predicted through a caller-owned workspace.
///
/// This is the composition shape. The model bound is only [`Estimator`],
/// because a fitted composition has no trait-shaped prediction method: its
/// intermediate batch lives in storage the caller owns and reuses, which
/// [`Classifier`] cannot express. The case therefore carries the batch surface
/// itself, and every entry point takes the workspace.
pub trait WorkspaceClassifierCase {
    /// The fitted composition under test.
    type Model: Estimator + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Which fixture shape this composition is exercised on.
    const FIXTURE: FixtureShape = FixtureShape::Wide;

    /// Fits the composition on the battery fixture.
    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;

    /// Fits with sample weights. Required exactly when declared.
    fn fit_weighted(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        None
    }

    /// Fits over an arbitrary observed class set. Required exactly when
    /// declared.
    fn fit_multiclass(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        None
    }

    /// Encodes and decodes the composition. Required exactly when declared.
    fn round_trip(_model: &Self::Model) -> RoundTrip<Self::Model> {
        None
    }

    /// `f32` values this model needs to predict a batch of `rows` rows.
    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError>;

    /// Sorted class labels observed during fitting.
    fn classes(model: &Self::Model) -> Vec<u8>;

    /// Predicts labels, allocating the output.
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<u8>, ModelError>;

    /// Predicts labels into caller-owned output.
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError>;

    /// Predicts row-major probabilities, allocating the output.
    fn predict_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError>;

    /// Predicts row-major probabilities into caller-owned output.
    fn predict_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError>;

    /// Predicts one probability column, allocating the output.
    fn predict_class_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
    ) -> Result<Vec<f32>, ModelError>;

    /// Predicts one probability column into caller-owned output.
    fn predict_class_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError>;
}

/// A real-valued-score model predicted through a caller-owned workspace.
///
/// The composition shape for anything producing one value per row: a fitted
/// regression pipeline, a staged pipeline, or an estimator whose scoring path
/// is inherent rather than trait-shaped.
pub trait WorkspaceRegressorCase {
    /// The fitted model under test.
    type Model: Estimator + HasCapabilities;

    /// Name used in failure messages.
    const NAME: &'static str;

    /// Which fixture shape this model is exercised on.
    const FIXTURE: FixtureShape = FixtureShape::Wide;

    /// Fits the model on the battery fixture.
    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;

    /// Fits with sample weights. Required exactly when declared.
    fn fit_weighted(_train: &Sample, _holdout: &Sample) -> OptionalFit<Self::Model> {
        None
    }

    /// Encodes and decodes the model. Required exactly when declared.
    fn round_trip(_model: &Self::Model) -> RoundTrip<Self::Model> {
        None
    }

    /// `f32` values this model needs to predict a batch of `rows` rows.
    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError>;

    /// Predicts values, allocating the output.
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError>;

    /// Predicts values into caller-owned output.
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError>;
}

// ------------------------------------------------------------- driver traits
//
// The obligations are written once, against these. Two adapter types carry the
// two case shapes into them: a single blanket implementation over the *model*
// type is impossible, because the compiler cannot prove that a composition is
// not also a `Regressor`, while two blanket implementations over two distinct
// wrapper types are coherent. That is what keeps registration one line for an
// ordinary estimator while making a composition expressible at all.

trait ClassifierUnderTest {
    type Model: Estimator + HasCapabilities;
    const NAME: &'static str;
    const FIXTURE: FixtureShape;

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;
    fn fit_weighted(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model>;
    fn fit_multiclass(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model>;
    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model>;

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError>;
    fn classes(model: &Self::Model) -> Vec<u8>;

    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<u8>, ModelError>;
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError>;

    // The probability group; see the module note on D11.
    fn predict_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError>;
    fn predict_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError>;
    fn predict_class_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
    ) -> Result<Vec<f32>, ModelError>;
    fn predict_class_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError>;
}

trait ScalarClassifierUnderTest: ClassifierUnderTest {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError>;
}

trait RegressorUnderTest {
    type Model: Estimator + HasCapabilities;
    const NAME: &'static str;
    const FIXTURE: FixtureShape;

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError>;
    fn fit_weighted(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model>;
    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model>;

    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError>;
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError>;
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError>;
}

trait ScalarRegressorUnderTest: RegressorUnderTest {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError>;
}

/// Carries a trait-shaped case into the drivers, ignoring the workspace.
struct TraitShaped<C>(PhantomData<C>);

/// Carries a workspace-shaped case into the drivers.
struct WorkspaceShaped<C>(PhantomData<C>);

impl<C: ClassifierCase> ClassifierUnderTest for TraitShaped<C> {
    type Model = C::Model;
    const NAME: &'static str = C::NAME;
    const FIXTURE: FixtureShape = C::FIXTURE;

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        C::fit(train, holdout)
    }
    fn fit_weighted(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model> {
        C::fit_weighted(train, holdout)
    }
    fn fit_multiclass(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model> {
        C::fit_multiclass(train, holdout)
    }
    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        C::round_trip(model)
    }
    fn workspace_len(_model: &Self::Model, _rows: usize) -> Result<usize, ModelError> {
        Ok(0)
    }
    fn classes(model: &Self::Model) -> Vec<u8> {
        model.classes().to_vec()
    }
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
    ) -> Result<Vec<u8>, ModelError> {
        model.predict(data)
    }
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError> {
        model.predict_into(data, output)
    }
    fn predict_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        model.predict_proba(data)
    }
    fn predict_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.predict_proba_into(data, output)
    }
    fn predict_class_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        model.predict_class_proba(data, class)
    }
    fn predict_class_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.predict_class_proba_into(data, class, output)
    }
}

impl<C: ScalarClassifierCase> ScalarClassifierUnderTest for TraitShaped<C> {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<u8, ModelError> {
        C::predict_one(model, row)
    }
}

impl<C: WorkspaceClassifierCase> ClassifierUnderTest for WorkspaceShaped<C> {
    type Model = C::Model;
    const NAME: &'static str = C::NAME;
    const FIXTURE: FixtureShape = C::FIXTURE;

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        C::fit(train, holdout)
    }
    fn fit_weighted(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model> {
        C::fit_weighted(train, holdout)
    }
    fn fit_multiclass(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model> {
        C::fit_multiclass(train, holdout)
    }
    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        C::round_trip(model)
    }
    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        C::workspace_len(model, rows)
    }
    fn classes(model: &Self::Model) -> Vec<u8> {
        C::classes(model)
    }
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<u8>, ModelError> {
        C::predict(model, data, workspace)
    }
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError> {
        C::predict_into(model, data, workspace, output)
    }
    fn predict_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        C::predict_proba(model, data, workspace)
    }
    fn predict_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        C::predict_proba_into(model, data, workspace, output)
    }
    fn predict_class_proba(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        C::predict_class_proba(model, data, workspace, class)
    }
    fn predict_class_proba_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        C::predict_class_proba_into(model, data, workspace, class, output)
    }
}

impl<R: RegressorCase> RegressorUnderTest for TraitShaped<R> {
    type Model = R::Model;
    const NAME: &'static str = R::NAME;
    const FIXTURE: FixtureShape = R::FIXTURE;

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        R::fit(train, holdout)
    }
    fn fit_weighted(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model> {
        R::fit_weighted(train, holdout)
    }
    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        R::round_trip(model)
    }
    fn workspace_len(_model: &Self::Model, _rows: usize) -> Result<usize, ModelError> {
        Ok(0)
    }
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        model.predict(data)
    }
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        _workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        model.predict_into(data, output)
    }
}

impl<R: ScalarRegressorCase> ScalarRegressorUnderTest for TraitShaped<R> {
    fn predict_one(model: &Self::Model, row: &[f32]) -> Result<f32, ModelError> {
        R::predict_one(model, row)
    }
}

impl<R: WorkspaceRegressorCase> RegressorUnderTest for WorkspaceShaped<R> {
    type Model = R::Model;
    const NAME: &'static str = R::NAME;
    const FIXTURE: FixtureShape = R::FIXTURE;

    fn fit(train: &Sample, holdout: &Sample) -> Result<Self::Model, ModelError> {
        R::fit(train, holdout)
    }
    fn fit_weighted(train: &Sample, holdout: &Sample) -> OptionalFit<Self::Model> {
        R::fit_weighted(train, holdout)
    }
    fn round_trip(model: &Self::Model) -> RoundTrip<Self::Model> {
        R::round_trip(model)
    }
    fn workspace_len(model: &Self::Model, rows: usize) -> Result<usize, ModelError> {
        R::workspace_len(model, rows)
    }
    fn predict(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
    ) -> Result<Vec<f32>, ModelError> {
        R::predict(model, data, workspace)
    }
    fn predict_into(
        model: &Self::Model,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        R::predict_into(model, data, workspace, output)
    }
}

// -------------------------------------------------------------- entry points

/// Runs every obligation a classifier with a scalar path owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_classifier<C: ScalarClassifierCase>() {
    classifier_report::<C>().assert_clean(C::NAME);
}

/// Runs every obligation a classifier without a scalar path owes.
///
/// Prefer [`check_classifier`]. This weaker entry point exists only for the
/// runtime dispatch enums and fitted compositions, which have no scalar
/// prediction path at all.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_batch_only_classifier<C: ClassifierCase>() {
    batch_classifier_report::<C>().assert_clean(C::NAME);
}

/// Runs every obligation a workspace-predicted classifier owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_workspace_classifier<C: WorkspaceClassifierCase>() {
    workspace_classifier_report::<C>().assert_clean(C::NAME);
}

/// Runs every obligation a regressor with a scalar path owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_regressor<R: ScalarRegressorCase>() {
    regressor_report::<R>().assert_clean(R::NAME);
}

/// Runs every obligation a regressor without a scalar path owes.
///
/// Prefer [`check_regressor`]; see [`check_batch_only_classifier`].
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_batch_only_regressor<R: RegressorCase>() {
    batch_regressor_report::<R>().assert_clean(R::NAME);
}

/// Runs every obligation a workspace-predicted regressor owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_workspace_regressor<R: WorkspaceRegressorCase>() {
    workspace_regressor_report::<R>().assert_clean(R::NAME);
}

/// Runs every obligation a transformer owes.
///
/// # Panics
///
/// With one line per violated obligation.
pub fn check_transformer<T: TransformerCase>() {
    transformer_report::<T>().assert_clean(T::NAME);
}

/// Collects violations for a classifier with a scalar path.
#[must_use]
pub fn classifier_report<C: ScalarClassifierCase>() -> Report {
    scalar_classifier_obligations::<TraitShaped<C>>()
}

/// Collects violations for a classifier without a scalar path.
#[must_use]
pub fn batch_classifier_report<C: ClassifierCase>() -> Report {
    classifier_obligations::<TraitShaped<C>>()
}

/// Collects violations for a workspace-predicted classifier.
#[must_use]
pub fn workspace_classifier_report<C: WorkspaceClassifierCase>() -> Report {
    classifier_obligations::<WorkspaceShaped<C>>()
}

/// Collects violations for a regressor with a scalar path.
#[must_use]
pub fn regressor_report<R: ScalarRegressorCase>() -> Report {
    scalar_regressor_obligations::<TraitShaped<R>>()
}

/// Collects violations for a regressor without a scalar path.
#[must_use]
pub fn batch_regressor_report<R: RegressorCase>() -> Report {
    regressor_obligations::<TraitShaped<R>>()
}

/// Collects violations for a workspace-predicted regressor.
#[must_use]
pub fn workspace_regressor_report<R: WorkspaceRegressorCase>() -> Report {
    regressor_obligations::<WorkspaceShaped<R>>()
}

// -------------------------------------------------------- classifier battery

fn scalar_classifier_obligations<C: ScalarClassifierUnderTest>() -> Report {
    let fixture = Fixture::new(C::FIXTURE);
    let mut report = classifier_obligations::<C>();
    let Ok(model) = C::fit(&fixture.train, &fixture.holdout) else {
        return report;
    };
    let view = fixture.train.view();
    let mut workspace = workspace_for(C::workspace_len(&model, view.rows()));
    let Ok(labels) = C::predict(&model, &view, &mut workspace) else {
        return report;
    };
    for (index, row) in view.iter_rows().enumerate() {
        let scalar = C::predict_one(&model, row);
        report.require("scalar_matches_batch", scalar == Ok(labels[index]), || {
            format!(
                "row {index} predicted {scalar:?} scalar but {} in batch",
                labels[index]
            )
        });
    }
    check_non_finite_scalar(&mut report, fixture.train.columns(), |row| {
        C::predict_one(&model, row).map(|_| ())
    });
    report
}

fn classifier_obligations<C: ClassifierUnderTest>() -> Report {
    let fixture = Fixture::new(C::FIXTURE);
    let mut report = Report::default();
    let train = &fixture.train;
    let view = train.view();

    let model = match C::fit(train, &fixture.holdout) {
        Ok(model) => model,
        Err(error) => {
            report.record(
                "fits_the_fixture",
                format!("fitting the fixture returned {error:?}"),
            );
            return report;
        }
    };
    let workspace_len = match C::workspace_len(&model, view.rows()) {
        Ok(len) => len,
        Err(error) => {
            report.record(
                "fits_the_fixture",
                format!("the fitted model could not size a workspace: {error:?}"),
            );
            return report;
        }
    };
    let mut workspace = vec![0.0; workspace_len];

    let classes = C::classes(&model);
    report.require("metadata", model.n_features_in() == train.columns(), || {
        format!(
            "n_features_in is {} but the fixture has {} columns",
            model.n_features_in(),
            train.columns()
        )
    });
    report.require("metadata", !classes.is_empty(), || {
        "classes() is empty".to_owned()
    });
    report.require(
        "metadata",
        classes.windows(2).all(|pair| pair[0] < pair[1]),
        || format!("classes() is not strictly ascending: {classes:?}"),
    );

    let labels = C::predict(&model, &view, &mut workspace);
    let probabilities = C::predict_proba(&model, &view, &mut workspace);
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
    let written = C::predict_into(&model, &view, &mut workspace, &mut labels_into);
    report.require(
        "into_matches_allocating",
        written.is_ok() && &labels_into == labels,
        || {
            format!(
                "predict_into returned {written:?} and wrote {labels_into:?}, expected {labels:?}"
            )
        },
    );

    probability_obligations::<C>(&mut report, &model, &view, &mut workspace, &classes, labels);

    let wrong_width = train.wrong_width();
    let width_error = ModelError::FeatureDimension {
        expected: train.columns(),
        actual: wrong_width.columns(),
    };
    let mut untouched = vec![7_u8; view.rows()];
    let rejected = C::predict_into(
        &model,
        &wrong_width.as_view(),
        &mut workspace,
        &mut untouched,
    );
    report.require(
        "feature_width_validated_before_write",
        rejected == Err(width_error.clone()) && untouched == vec![7_u8; view.rows()],
        || {
            format!(
                "predict_into on a wrong-width batch returned {rejected:?} and wrote {untouched:?}"
            )
        },
    );
    let mut untouched = vec![7.0_f32; expected_probabilities];
    let rejected = C::predict_proba_into(
        &model,
        &wrong_width.as_view(),
        &mut workspace,
        &mut untouched,
    );
    report.require(
        "feature_width_validated_before_write",
        rejected == Err(width_error) && untouched == vec![7.0_f32; expected_probabilities],
        || format!("predict_proba_into on a wrong-width batch returned {rejected:?}"),
    );

    check_output_length::<u8>(&mut report, view.rows(), 7, |output| {
        C::predict_into(&model, &view, &mut workspace, output)
    });
    check_output_length::<f32>(&mut report, expected_probabilities, 7.0, |output| {
        C::predict_proba_into(&model, &view, &mut workspace, output)
    });
    check_output_length::<f32>(&mut report, view.rows(), 7.0, |output| {
        C::predict_class_proba_into(&model, &view, &mut workspace, classes[0], output)
    });

    let absent = absent_class(&classes);
    let mut untouched = vec![7.0_f32; view.rows()];
    let rejected =
        C::predict_class_proba_into(&model, &view, &mut workspace, absent, &mut untouched);
    report.require(
        "unknown_class_rejected",
        rejected == Err(ModelError::UnknownClass { class: absent })
            && untouched == vec![7.0_f32; view.rows()],
        || format!("class {absent} returned {rejected:?} and wrote {untouched:?}"),
    );
    let rejected = C::predict_class_proba(&model, &view, &mut workspace, absent);
    report.require(
        "unknown_class_rejected",
        rejected == Err(ModelError::UnknownClass { class: absent }),
        || {
            format!(
                "allocating predict_class_proba for absent class {absent} returned {rejected:?}"
            )
        },
    );

    workspace_obligations(
        &mut report,
        &model,
        &fixture,
        workspace_len,
        |model, data, workspace, output: &mut [u8]| C::predict_into(model, data, workspace, output),
        |model, data, workspace| C::predict(model, data, workspace),
        u8::MAX,
    );

    match C::fit(train, &fixture.holdout) {
        Ok(refitted) => {
            let mut refit_workspace = workspace_for(C::workspace_len(&refitted, view.rows()));
            report.require(
                "refit_is_deterministic",
                C::predict(&refitted, &view, &mut refit_workspace).as_ref() == Ok(labels)
                    && C::predict_proba(&refitted, &view, &mut refit_workspace).as_ref()
                        == Ok(probabilities),
                || "refitting the same data and parameters changed the predictions".to_owned(),
            );
        }
        Err(error) => report.record(
            "refit_is_deterministic",
            format!("refitting the same data and parameters failed: {error:?}"),
        ),
    }

    let weighted = C::fit_weighted(train, &fixture.holdout);
    check_declaration(
        &mut report,
        "sample_weight_declaration_matches_behavior",
        "sample_weights",
        C::Model::CAPABILITIES.sample_weights(),
        weighted.is_some(),
        weighted.map(|weighted| match weighted {
            Ok(weighted) => {
                let mut weighted_workspace =
                    workspace_for(C::workspace_len(&weighted, view.rows()));
                (
                    C::predict(&weighted, &view, &mut weighted_workspace).as_ref() == Ok(labels)
                        && C::predict_proba(&weighted, &view, &mut weighted_workspace).as_ref()
                            == Ok(probabilities),
                    "unit-weighted fit predicts differently from the unweighted fit".to_owned(),
                )
            }
            Err(error) => (
                false,
                format!("the declared weighted fit failed: {error:?}"),
            ),
        }),
    );

    check_multiclass_declaration::<C>(&mut report, &fixture);

    let round_tripped = C::round_trip(&model);
    check_declaration(
        &mut report,
        "artifact_declaration_matches_behavior",
        "artifact",
        C::Model::CAPABILITIES.artifact(),
        round_tripped.is_some(),
        round_tripped.map(|outcome| match (outcome, C::round_trip(&model)) {
            (Ok((bytes, decoded)), Some(Ok((again, _)))) => {
                let mut decoded_workspace = workspace_for(C::workspace_len(&decoded, view.rows()));
                (
                    bytes == again
                        && C::predict(&decoded, &view, &mut decoded_workspace).as_ref()
                            == Ok(labels)
                        && C::predict_proba(&decoded, &view, &mut decoded_workspace).as_ref()
                            == Ok(probabilities),
                    "the artifact re-encoded differently or the decoded model predicts differently"
                        .to_owned(),
                )
            }
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

/// Everything that follows from a classifier producing probabilities.
///
/// Grouped deliberately; see the module note on D11. Probability production is
/// mandatory on `Classifier` today, so this runs unconditionally.
fn probability_obligations<C: ClassifierUnderTest>(
    report: &mut Report,
    model: &C::Model,
    view: &MatrixView<'_>,
    workspace: &mut [f32],
    classes: &[u8],
    labels: &[u8],
) {
    let Ok(probabilities) = C::predict_proba(model, view, workspace) else {
        return;
    };
    let mut probabilities_into = vec![f32::MAX; probabilities.len()];
    let written = C::predict_proba_into(model, view, workspace, &mut probabilities_into);
    report.require(
        "into_matches_allocating",
        written.is_ok() && probabilities_into == probabilities,
        || format!("predict_proba_into returned {written:?} and disagreed with predict_proba"),
    );

    for (column, &class) in classes.iter().enumerate() {
        let allocating = C::predict_class_proba(model, view, workspace, class);
        let mut into = vec![f32::MAX; view.rows()];
        let written = C::predict_class_proba_into(model, view, workspace, class, &mut into);
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
}

/// A declared capability holds for every fit the type offers.
///
/// `artifact` is a property of the estimator *type*, not of one fitted value,
/// so a classifier declaring both persistence and multiclass fitting owes a
/// working round trip for a multiclass fit too. Without this the flag could
/// quietly mean "some fits persist", and a caller reading it before choosing an
/// estimator would be told something untrue of the model it is about to train.
fn check_multiclass_artifact_declaration<C: ClassifierUnderTest>(
    report: &mut Report,
    fixture: &Fixture,
) {
    let capabilities = C::Model::CAPABILITIES;
    if !(capabilities.artifact() && capabilities.multiclass()) {
        return;
    }
    let view = fixture.train.view();
    let Some(Ok(model)) = C::fit_multiclass(&fixture.train, &fixture.holdout) else {
        return;
    };
    let mut workspace = workspace_for(C::workspace_len(&model, view.rows()));
    let (Ok(labels), Ok(probabilities)) = (
        C::predict(&model, &view, &mut workspace),
        C::predict_proba(&model, &view, &mut workspace),
    ) else {
        return;
    };
    let detail = match (C::round_trip(&model), C::round_trip(&model)) {
        (Some(Ok((bytes, decoded))), Some(Ok((again, _)))) => {
            let mut decoded_workspace = workspace_for(C::workspace_len(&decoded, view.rows()));
            if bytes != again {
                Some("the multiclass artifact did not re-encode to the same bytes".to_owned())
            } else if C::classes(&decoded) != C::classes(&model) {
                Some(format!(
                    "the decoded multiclass model reports classes {:?} instead of {:?}",
                    C::classes(&decoded),
                    C::classes(&model)
                ))
            } else if C::predict(&decoded, &view, &mut decoded_workspace).as_ref() != Ok(&labels)
                || C::predict_proba(&decoded, &view, &mut decoded_workspace).as_ref()
                    != Ok(&probabilities)
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

/// A declared multiclass fit must exist and must be genuinely multiclass.
///
/// The obligation is not "does it return something": it is that the fitted
/// model reports exactly the fixture's observed labels in sorted order, that
/// its probability matrix has one column per label, that each column is the
/// one `predict_class_proba` returns for that label, that every row sums to
/// one within the documented tolerance, and that the label is the argmax of
/// that row. A model that quietly collapsed the class set, permuted the
/// columns, or renormalized would fail one of those.
fn check_multiclass_declaration<C: ClassifierUnderTest>(report: &mut Report, fixture: &Fixture) {
    let view = fixture.train.view();
    let declared = C::Model::CAPABILITIES.multiclass();
    let fitted = C::fit_multiclass(&fixture.train, &fixture.holdout);
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
    let Some(fitted) = fitted.filter(|_| declared) else {
        return;
    };
    let model = match fitted {
        Ok(model) => model,
        Err(error) => {
            report.record(
                "multiclass_declaration_matches_behavior",
                format!("the declared multiclass fit failed: {error:?}"),
            );
            return;
        }
    };
    let mut workspace = workspace_for(C::workspace_len(&model, view.rows()));

    let classes = C::classes(&model);
    report.require(
        "multiclass_declaration_matches_behavior",
        classes == fixture.train.class_labels.classes(),
        || {
            format!(
                "multiclass fit reports classes {classes:?}, expected {:?}",
                fixture.train.class_labels.classes()
            )
        },
    );
    let (Ok(labels), Ok(probabilities)) = (
        C::predict(&model, &view, &mut workspace),
        C::predict_proba(&model, &view, &mut workspace),
    ) else {
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
            C::predict_class_proba(&model, &view, &mut workspace, class).as_ref() == Ok(&expected),
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

// --------------------------------------------------------- regressor battery

fn scalar_regressor_obligations<R: ScalarRegressorUnderTest>() -> Report {
    let fixture = Fixture::new(R::FIXTURE);
    let mut report = regressor_obligations::<R>();
    let Ok(model) = R::fit(&fixture.train, &fixture.holdout) else {
        return report;
    };
    let view = fixture.train.view();
    let mut workspace = workspace_for(R::workspace_len(&model, view.rows()));
    if let Ok(values) = R::predict(&model, &view, &mut workspace) {
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
    check_non_finite_scalar(&mut report, fixture.train.columns(), |row| {
        R::predict_one(&model, row).map(|_| ())
    });
    report
}

fn regressor_obligations<R: RegressorUnderTest>() -> Report {
    let fixture = Fixture::new(R::FIXTURE);
    let mut report = Report::default();
    let train = &fixture.train;
    let view = train.view();

    let model = match R::fit(train, &fixture.holdout) {
        Ok(model) => model,
        Err(error) => {
            report.record(
                "fits_the_fixture",
                format!("fitting the fixture returned {error:?}"),
            );
            return report;
        }
    };
    let workspace_len = match R::workspace_len(&model, view.rows()) {
        Ok(len) => len,
        Err(error) => {
            report.record(
                "fits_the_fixture",
                format!("the fitted model could not size a workspace: {error:?}"),
            );
            return report;
        }
    };
    let mut workspace = vec![0.0; workspace_len];

    report.require("metadata", model.n_features_in() == train.columns(), || {
        format!(
            "n_features_in is {} but the fixture has {} columns",
            model.n_features_in(),
            train.columns()
        )
    });

    let values = R::predict(&model, &view, &mut workspace);
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
    let written = R::predict_into(&model, &view, &mut workspace, &mut into);
    report.require(
        "into_matches_allocating",
        written.is_ok() && &into == values,
        || format!("predict_into returned {written:?} and disagreed with predict"),
    );

    let wrong_width = train.wrong_width();
    let mut untouched = vec![7.0_f32; view.rows()];
    let rejected = R::predict_into(
        &model,
        &wrong_width.as_view(),
        &mut workspace,
        &mut untouched,
    );
    report.require(
        "feature_width_validated_before_write",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: train.columns(),
                actual: wrong_width.columns(),
            })
            && untouched == vec![7.0_f32; view.rows()],
        || {
            format!(
                "predict_into on a wrong-width batch returned {rejected:?} and wrote {untouched:?}"
            )
        },
    );

    check_output_length::<f32>(&mut report, view.rows(), 7.0, |output| {
        R::predict_into(&model, &view, &mut workspace, output)
    });

    workspace_obligations(
        &mut report,
        &model,
        &fixture,
        workspace_len,
        |model, data, workspace, output: &mut [f32]| {
            R::predict_into(model, data, workspace, output)
        },
        |model, data, workspace| R::predict(model, data, workspace),
        f32::MAX,
    );

    match R::fit(train, &fixture.holdout) {
        Ok(refitted) => {
            let mut refit_workspace = workspace_for(R::workspace_len(&refitted, view.rows()));
            report.require(
                "refit_is_deterministic",
                R::predict(&refitted, &view, &mut refit_workspace).as_ref() == Ok(values),
                || "refitting the same data and parameters changed the predictions".to_owned(),
            );
        }
        Err(error) => report.record(
            "refit_is_deterministic",
            format!("refitting the same data and parameters failed: {error:?}"),
        ),
    }

    let weighted = R::fit_weighted(train, &fixture.holdout);
    check_declaration(
        &mut report,
        "sample_weight_declaration_matches_behavior",
        "sample_weights",
        R::Model::CAPABILITIES.sample_weights(),
        weighted.is_some(),
        weighted.map(|weighted| match weighted {
            Ok(weighted) => {
                let mut weighted_workspace =
                    workspace_for(R::workspace_len(&weighted, view.rows()));
                (
                    R::predict(&weighted, &view, &mut weighted_workspace).as_ref() == Ok(values),
                    "unit-weighted fit predicts differently from the unweighted fit".to_owned(),
                )
            }
            Err(error) => (
                false,
                format!("the declared weighted fit failed: {error:?}"),
            ),
        }),
    );

    let round_tripped = R::round_trip(&model);
    check_declaration(
        &mut report,
        "artifact_declaration_matches_behavior",
        "artifact",
        R::Model::CAPABILITIES.artifact(),
        round_tripped.is_some(),
        round_tripped.map(|outcome| match (outcome, R::round_trip(&model)) {
            (Ok((bytes, decoded)), Some(Ok((again, _)))) => {
                let mut decoded_workspace = workspace_for(R::workspace_len(&decoded, view.rows()));
                (
                    bytes == again
                        && R::predict(&decoded, &view, &mut decoded_workspace).as_ref()
                            == Ok(values),
                    "the artifact re-encoded differently or the decoded model predicts differently"
                        .to_owned(),
                )
            }
            (Err(error), _) => (false, format!("the artifact round trip failed: {error:?}")),
            (Ok(_), _) => (
                false,
                "the artifact round trip did not repeat successfully".to_owned(),
            ),
        }),
    );

    report
}

// ------------------------------------------------------- transformer battery

/// Collects violations for a transformer.
#[must_use]
pub fn transformer_report<T: TransformerCase>() -> Report {
    let fixture = Fixture::new(T::FIXTURE);
    let mut report = Report::default();
    let train = &fixture.train;
    let view = train.view();

    let model = match T::fit(train, &fixture.holdout) {
        Ok(model) => model,
        Err(error) => {
            report.record(
                "fits_the_fixture",
                format!("fitting the fixture returned {error:?}"),
            );
            return report;
        }
    };
    let columns = model.n_features_out();

    report.require("metadata", model.n_features_in() == train.columns(), || {
        format!(
            "n_features_in is {} but the fixture has {} columns",
            model.n_features_in(),
            train.columns()
        )
    });
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

    let wrong_width = train.wrong_width();
    let mut untouched = vec![7.0_f32; expected_len];
    let rejected = model
        .transform_into(&wrong_width.as_view(), &mut untouched)
        .map(|_| ());
    report.require(
        "feature_width_validated_before_write",
        rejected
            == Err(ModelError::FeatureDimension {
                expected: train.columns(),
                actual: wrong_width.columns(),
            })
            && untouched == vec![7.0_f32; expected_len],
        || format!("transform_into on a wrong-width batch returned {rejected:?}"),
    );

    check_output_length::<f32>(&mut report, expected_len, 7.0, |output| {
        model.transform_into(&view, output).map(|_| ())
    });

    match T::fit(train, &fixture.holdout) {
        Ok(refitted) => report.require(
            "refit_is_deterministic",
            refitted
                .transform(&view)
                .map(|matrix| matrix.as_slice().to_vec())
                == Ok(transformed.clone()),
            || "refitting the same data and parameters changed the transform".to_owned(),
        ),
        Err(error) => report.record(
            "refit_is_deterministic",
            format!("refitting the same data and parameters failed: {error:?}"),
        ),
    }

    let weighted = T::fit_weighted(train, &fixture.holdout);
    check_declaration(
        &mut report,
        "sample_weight_declaration_matches_behavior",
        "sample_weights",
        T::Model::CAPABILITIES.sample_weights(),
        weighted.is_some(),
        weighted.map(|weighted| match weighted {
            Ok(weighted) => (
                weighted
                    .transform(&view)
                    .map(|matrix| matrix.as_slice().to_vec())
                    == Ok(transformed.clone()),
                "unit-weighted fit transforms differently from the unweighted fit".to_owned(),
            ),
            Err(error) => (
                false,
                format!("the declared weighted fit failed: {error:?}"),
            ),
        }),
    );

    let round_tripped = T::round_trip(&model);
    check_declaration(
        &mut report,
        "artifact_declaration_matches_behavior",
        "artifact",
        T::Model::CAPABILITIES.artifact(),
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

// -------------------------------------------------------------- shared checks

fn workspace_for<E>(len: Result<usize, E>) -> Vec<f32> {
    vec![0.0; len.unwrap_or(0)]
}

/// The two obligations a caller-owned workspace adds.
///
/// Skipped when the model needs no workspace: there is nothing to validate and
/// nothing to leak, and a trait-shaped estimator would otherwise be asked to
/// reject a buffer it never reads.
///
/// Both the classifier and the regressor battery reach this one
/// implementation, so a probe tripping it on either side proves the code path
/// for both — exactly as `check_output_length` is one implementation shared by
/// three categories.
fn workspace_obligations<M, T: Clone + PartialEq + std::fmt::Debug>(
    report: &mut Report,
    model: &M,
    fixture: &Fixture,
    workspace_len: usize,
    mut predict_into: impl FnMut(&M, &MatrixView<'_>, &mut [f32], &mut [T]) -> Result<(), ModelError>,
    mut predict: impl FnMut(&M, &MatrixView<'_>, &mut [f32]) -> Result<Vec<T>, ModelError>,
    sentinel: T,
) {
    if workspace_len == 0 {
        return;
    }
    let view = fixture.train.view();

    for actual in [workspace_len - 1, workspace_len + 1] {
        let mut workspace = vec![0.0; actual];
        let mut output = vec![sentinel.clone(); view.rows()];
        let rejected = predict_into(model, &view, &mut workspace, &mut output);
        report.require(
            "workspace_length_validated_before_write",
            rejected
                == Err(ModelError::OutputLength {
                    expected: workspace_len,
                    actual,
                })
                && output == vec![sentinel.clone(); view.rows()],
            || {
                format!(
                    "a workspace of {actual} for {workspace_len} values returned {rejected:?} \
                     and left {output:?}"
                )
            },
        );
    }

    // Reusing one workspace across batches must not carry anything from the
    // previous batch into the next: what a composition parks there is scratch,
    // not state.
    let holdout = fixture.holdout.view();
    let mut fresh = vec![0.0; workspace_len];
    let alone = predict(model, &holdout, &mut fresh);
    let mut reused = vec![0.0; workspace_len];
    let _ = predict(model, &view, &mut reused);
    let after = predict(model, &holdout, &mut reused);
    report.require("workspace_reuse_is_independent", alone == after, || {
        format!("predicting after a previous batch returned {after:?}, alone it returned {alone:?}")
    });
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

/// A declaration and the hook it selects must agree, in both directions.
///
/// Stated once so that declaring a further capability later is one call rather
/// than a new bespoke check: the caller supplies the obligation name, the
/// capability's name for the failure message, what the type declared, whether
/// the case supplied the hook, and — when it did — whether the hook behaved.
fn check_declaration(
    report: &mut Report,
    obligation: &'static str,
    capability: &'static str,
    declared: bool,
    supplied: bool,
    outcome: Option<(bool, String)>,
) {
    report.require(obligation, declared == supplied, || {
        format!(
            "declares {capability} = {declared} but {} a matching hook",
            if supplied { "supplies" } else { "supplies no" }
        )
    });
    if let Some((held, detail)) = outcome
        && declared
    {
        report.require(obligation, held, || detail);
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

/// Silences the unused-import warning for a type only the case traits name.
const _: Option<Capabilities> = None;
