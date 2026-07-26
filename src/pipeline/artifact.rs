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
//! [`StageArtifact`] — live in
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

/// Implements the persisted-stack contract for one flat tuple arity.
///
/// Decoding walks the component list in order rather than destructuring a
/// fixed-length pattern, because tuple expressions evaluate left to right and
/// that is the same order the tags were written in. The length is checked
/// first, so a payload carrying the wrong number of stages is refused before
/// any stage decodes.
macro_rules! impl_persisted_stack {
    ($($stage:ident $index:tt),+) => {
        impl<$($stage),+> PersistedStack for ($($stage,)+)
        where
            $($stage: StageArtifact,)+
        {
            const STAGE_TAGS: &'static [u16] = &[$($stage::STAGE_TAG),+];

            fn encode_stages(
                &self,
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Vec<Vec<u8>>, ArtifactError> {
                Ok(vec![
                    $(self.$index.to_artifact(input_schema, transformed_schema)?,)+
                ])
            }

            fn decode_stages(
                components: &[&[u8]],
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Self, ArtifactError> {
                if components.len() != Self::STAGE_TAGS.len() {
                    return Err(ArtifactError::InvalidPayload);
                }
                let mut remaining = components.iter();
                Ok((
                    $(
                        $stage::from_artifact(
                            remaining.next().ok_or(ArtifactError::InvalidPayload)?,
                            input_schema,
                            transformed_schema,
                        )?,
                    )+
                ))
            }
        }
    };
}

impl_persisted_stack!(A 0);
impl_persisted_stack!(A 0, B 1);
impl_persisted_stack!(A 0, B 1, C 2);
impl_persisted_stack!(A 0, B 1, C 2, D 3);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
impl_persisted_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
