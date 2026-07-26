//! Persistence for multi-stage compositions.
//!
//! A composition persists exactly when every one of its parts does. Rather
//! than reserving one artifact kind per concrete composition — which does not
//! scale past a handful and is what [`Pipeline`](super::Pipeline) already
//! spent three kinds on — a [`StagedPipeline`](super::StagedPipeline) uses one
//! kind and records *which* concrete parts it holds as tags inside the
//! payload. Decoding into a different composition then fails on the tags
//! rather than silently reinterpreting another model's bytes.
//!
//! The stack trait below is the composition contract, so it lives with the
//! composition. The per-type persistence contracts it builds on —
//! [`ModelArtifact`](crate::artifact::ModelArtifact) and
//! [`StageArtifact`](crate::artifact::StageArtifact) — live in
//! [`crate::artifact`], because a fitted type persists whether or not it is
//! ever composed.

use crate::artifact::{ArtifactError, StageArtifact};

/// A stack of transform stages that persists as an ordered component list.
///
/// [`PersistedStack::STAGE_TAGS`] is the stack's identity: two stacks holding
/// the same stage types in a different order have different tag sequences, so
/// one never decodes as the other.
pub trait PersistedStack: Sized {
    /// Stage-type tags, in application order.
    const STAGE_TAGS: &'static [u16];

    /// Encodes every stage in order.
    fn encode_stages(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<Vec<u8>>, ArtifactError>;

    /// Decodes every stage from its already-extracted component payload.
    ///
    /// `components` has exactly [`PersistedStack::STAGE_TAGS`] entries; the
    /// caller has already checked that count and the recorded tags.
    fn decode_stages(
        components: &[&[u8]],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError>;
}

impl<A, B> PersistedStack for (A, B)
where
    A: StageArtifact,
    B: StageArtifact,
{
    const STAGE_TAGS: &'static [u16] = &[A::STAGE_TAG, B::STAGE_TAG];

    fn encode_stages(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<Vec<u8>>, ArtifactError> {
        Ok(vec![
            self.0.to_artifact(input_schema, transformed_schema)?,
            self.1.to_artifact(input_schema, transformed_schema)?,
        ])
    }

    fn decode_stages(
        components: &[&[u8]],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let [first, second] = components else {
            return Err(ArtifactError::InvalidPayload);
        };
        Ok((
            A::from_artifact(first, input_schema, transformed_schema)?,
            B::from_artifact(second, input_schema, transformed_schema)?,
        ))
    }
}

impl<A, B, C> PersistedStack for (A, B, C)
where
    A: StageArtifact,
    B: StageArtifact,
    C: StageArtifact,
{
    const STAGE_TAGS: &'static [u16] = &[A::STAGE_TAG, B::STAGE_TAG, C::STAGE_TAG];

    fn encode_stages(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<Vec<u8>>, ArtifactError> {
        Ok(vec![
            self.0.to_artifact(input_schema, transformed_schema)?,
            self.1.to_artifact(input_schema, transformed_schema)?,
            self.2.to_artifact(input_schema, transformed_schema)?,
        ])
    }

    fn decode_stages(
        components: &[&[u8]],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let [first, second, third] = components else {
            return Err(ArtifactError::InvalidPayload);
        };
        Ok((
            A::from_artifact(first, input_schema, transformed_schema)?,
            B::from_artifact(second, input_schema, transformed_schema)?,
            C::from_artifact(third, input_schema, transformed_schema)?,
        ))
    }
}
