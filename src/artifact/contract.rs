//! The persistence contract every fitted type declares by implementing it.
//!
//! # Why persistence is a trait rather than an inherent method
//!
//! A fitted type used to declare persistence *twice*: once by writing an
//! inherent `to_artifact`, and again by being added to a hand-maintained list
//! of composition members. Because the second declaration was a separate act of
//! remembering, seven estimators shipped a working, fuzz-tested encoder and were
//! still invisible to the type system — and every composition ending in one of
//! them fell out of the conformance battery, because the battery reaches a
//! composition through its capability declaration and that declaration was
//! gated on the list.
//!
//! Implementing one of the traits below *is* writing the encoder, so the two
//! declarations became one. A new estimator that persists cannot be silently
//! unpersistable, because there is no second place left to omit it from. This
//! follows the rule `crate::api` already states: a concrete method stays
//! inherent until its shape semantics are shared by every implementation, and
//! fourteen estimators sharing one signature exactly is that condition being
//! met.
//!
//! # Two arities, because there are two kinds of schema binding
//!
//! A final estimator is bound to the features it was fitted on: one schema.
//! Anything that *transforms* is bound to what it consumed and what it produced:
//! two schemas. Those are the only two shapes in the crate, and they are the
//! two traits here.
//!
//! # Kinds, tags, and why they are separate numbers
//!
//! [`ModelArtifact::ARTIFACT_KIND`] names the type in the on-disk envelope.
//! [`ModelArtifact::MODEL_TAG`] names it inside a *composed* payload. They are
//! different namespaces, and both are permanent: changing either changes what
//! previously written artifacts decode as.
//!
//! The tag therefore *defaults* to the kind, which is already unique and
//! already permanent. A new estimator gets a composition identity for free and
//! has nothing extra to assign — the failure this whole contract exists to
//! prevent.
//!
//! Six types were tagged before that default existed and their tags do not
//! equal their kinds. Rather than have those six override the default — six
//! chances to write the wrong number, in six files, forever — the exceptions
//! live in the two closed tables below, beside the kinds they translate. No
//! estimator anywhere declares a tag. The tables cannot grow: a type whose tag
//! already equals its kind needs no entry, and every future type gets the
//! default. Every resulting tag is pinned by number in an integration test, and
//! the adversarial corpus records two of the six as frozen bytes.

use super::ArtifactError;
use super::envelope::{
    HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND, MAX_ABS_SCALER_ARTIFACT_KIND,
    MIN_MAX_SCALER_ARTIFACT_KIND, RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
    ROBUST_SCALER_ARTIFACT_KIND, STANDARD_SCALER_ARTIFACT_KIND,
};

/// Composition tags assigned to final estimators before the tag defaulted to
/// the kind. Closed: every other estimator's tag *is* its kind.
const LEGACY_MODEL_TAGS: [(u16, u16); 2] = [
    (RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND, 4),
    (HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND, 5),
];

/// Composition tags assigned to transform stages before the same default.
/// Closed for the same reason.
const LEGACY_STAGE_TAGS: [(u16, u16); 4] = [
    (STANDARD_SCALER_ARTIFACT_KIND, 1),
    (MIN_MAX_SCALER_ARTIFACT_KIND, 2),
    (MAX_ABS_SCALER_ARTIFACT_KIND, 3),
    (ROBUST_SCALER_ARTIFACT_KIND, 4),
];

const fn translate(table: &[(u16, u16)], kind: u16) -> u16 {
    let mut index = 0;
    while index < table.len() {
        if table[index].0 == kind {
            return table[index].1;
        }
        index += 1;
    }
    kind
}

/// The composition tag a final estimator of this kind carries.
#[must_use]
pub const fn model_tag_for_kind(kind: u16) -> u16 {
    translate(&LEGACY_MODEL_TAGS, kind)
}

/// The composition tag a transform stage of this kind carries.
#[must_use]
pub const fn stage_tag_for_kind(kind: u16) -> u16 {
    translate(&LEGACY_STAGE_TAGS, kind)
}

/// A fitted estimator that persists against the features it was fitted on.
///
/// Implementing this is the *whole* declaration: it supplies the encoder, the
/// envelope identity, and the composition identity together, so none of the
/// three can be present without the others.
///
/// A type implementing this must also declare
/// [`Capabilities::artifact`](crate::api::Capabilities::artifact), and a type
/// declaring that capability must implement this. Neither direction is left to
/// convention: an integration test closes them against each other through the
/// frozen public-API and capability snapshots.
pub trait ModelArtifact: Sized {
    /// Permanent identity of this estimator in the on-disk envelope.
    const ARTIFACT_KIND: u16;

    /// Permanent identity of this estimator inside a composed payload.
    ///
    /// Derived from [`Self::ARTIFACT_KIND`]; never declared by an
    /// implementation. Do not override it — the two estimators whose tag
    /// differs from their kind are handled by the closed table this reads.
    const MODEL_TAG: u16 = model_tag_for_kind(Self::ARTIFACT_KIND);

    /// Encodes this estimator against its fitted feature schema.
    fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError>;

    /// Decodes this estimator, checking the fitted feature schema.
    fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError>;
}

/// A fitted type that persists against both an input and an output schema.
///
/// This covers transform stages and the compositions built from them: both
/// consume one feature schema and produce another, and both must refuse bytes
/// written against a different pair.
///
/// The tag identifies a stage inside a composed payload, and carries the same
/// permanence and the same default as [`ModelArtifact::MODEL_TAG`]. A
/// composition implements this too — it is schema-bound in exactly the same
/// shape a stage is — and simply never appears in another payload's tag list.
pub trait StageArtifact: Sized {
    /// Permanent identity of this type in the on-disk envelope.
    const ARTIFACT_KIND: u16;

    /// Permanent identity of this stage inside a composed payload.
    ///
    /// Derived from [`Self::ARTIFACT_KIND`] by the same closed table, and
    /// likewise never declared by an implementation.
    const STAGE_TAG: u16 = stage_tag_for_kind(Self::ARTIFACT_KIND);

    /// Encodes this type against the input and output schemas it spans.
    fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError>;

    /// Decodes this type, checking both schemas.
    fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError>;
}
