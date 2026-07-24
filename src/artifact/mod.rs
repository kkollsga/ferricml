//! Stable, bounded model-artifact errors and format identity.

mod cursor;
mod envelope;
mod error;
mod logical_tree;

pub(crate) use cursor::ArtifactCursor;
pub(crate) use envelope::{
    ANY_REGRESSOR_ARTIFACT_KIND, ArtifactPayloadWriter,
    HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND, LINEAR_REGRESSION_ARTIFACT_KIND,
    LOGISTIC_ARTIFACT_KIND, PAIRWISE_LINEAR_RANKER_ARTIFACT_KIND,
    RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND, RIDGE_ARTIFACT_KIND, STANDARD_SCALER_ARTIFACT_KIND,
    STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND, STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND,
    STANDARD_SCALER_RIDGE_PIPELINE_ARTIFACT_KIND, SchemaRole, artifact_version, decode_component,
    decode_legacy_envelope, decode_v2_envelope, encode_component, encode_v2_envelope,
};
pub use error::ArtifactError;
pub(crate) use logical_tree::{LogicalTreeNode, decode_logical_tree, encode_logical_tree};

/// Current FerricML binary model artifact version.
pub const MODEL_ARTIFACT_VERSION: u16 = 2;

/// Maximum accepted size of one encoded model artifact.
pub const MAX_MODEL_ARTIFACT_BYTES: usize = 32 * 1024 * 1024;
