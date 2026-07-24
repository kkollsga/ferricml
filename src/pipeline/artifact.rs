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
//! The three traits below are the composition contract, so they live with the
//! composition. Each simply names the schema-bound artifact a fitted type
//! already has, plus the tag that identifies it inside a composed payload.

use crate::artifact::ArtifactError;
use crate::ensemble::{HistGradientBoostingRegressor, RandomForestRegressor};
use crate::linear_model::{LinearRegression, LogisticRegression, Ridge};
use crate::preprocessing::{MaxAbsScaler, MinMaxScaler, StandardScaler};

/// A fitted transform stage that can be persisted inside a composition.
///
/// The tag identifies the concrete stage type within a composed payload. Tags
/// are stable: changing one changes what previously written artifacts decode
/// as, so a new stage type takes the next unused value instead.
pub trait StageArtifact: Sized {
    /// Stable identity of this stage type inside a composed payload.
    const STAGE_TAG: u16;

    /// Encodes this stage against the composition's input and output schemas.
    fn to_stage_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError>;

    /// Decodes this stage, checking both schemas.
    fn from_stage_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError>;
}

/// A fitted final estimator that can be persisted inside a composition.
pub trait ModelArtifact: Sized {
    /// Stable identity of this estimator type inside a composed payload.
    const MODEL_TAG: u16;

    /// Encodes this estimator against the transformed-feature schema.
    fn to_model_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError>;

    /// Decodes this estimator, checking the transformed-feature schema.
    fn from_model_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError>;
}

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

macro_rules! impl_stage_artifact {
    ($stage:ty, $tag:expr) => {
        impl StageArtifact for $stage {
            const STAGE_TAG: u16 = $tag;

            fn to_stage_artifact(
                &self,
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Vec<u8>, ArtifactError> {
                self.to_artifact(input_schema, transformed_schema)
            }

            fn from_stage_artifact(
                bytes: &[u8],
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Self, ArtifactError> {
                Self::from_artifact(bytes, input_schema, transformed_schema)
            }
        }
    };
}

impl_stage_artifact!(StandardScaler, 1);
impl_stage_artifact!(MinMaxScaler, 2);
impl_stage_artifact!(MaxAbsScaler, 3);

macro_rules! impl_model_artifact {
    ($model:ty, $tag:expr) => {
        impl ModelArtifact for $model {
            const MODEL_TAG: u16 = $tag;

            fn to_model_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
                self.to_artifact(schema)
            }

            fn from_model_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
                Self::from_artifact(bytes, schema)
            }
        }
    };
}

impl_model_artifact!(LogisticRegression, 1);
impl_model_artifact!(LinearRegression, 2);
impl_model_artifact!(Ridge, 3);
impl_model_artifact!(RandomForestRegressor, 4);
impl_model_artifact!(HistGradientBoostingRegressor, 5);

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
            self.0.to_stage_artifact(input_schema, transformed_schema)?,
            self.1.to_stage_artifact(input_schema, transformed_schema)?,
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
            A::from_stage_artifact(first, input_schema, transformed_schema)?,
            B::from_stage_artifact(second, input_schema, transformed_schema)?,
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
            self.0.to_stage_artifact(input_schema, transformed_schema)?,
            self.1.to_stage_artifact(input_schema, transformed_schema)?,
            self.2.to_stage_artifact(input_schema, transformed_schema)?,
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
            A::from_stage_artifact(first, input_schema, transformed_schema)?,
            B::from_stage_artifact(second, input_schema, transformed_schema)?,
            C::from_stage_artifact(third, input_schema, transformed_schema)?,
        ))
    }
}
