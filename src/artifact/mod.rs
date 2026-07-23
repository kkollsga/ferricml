//! Stable, bounded model-artifact errors and format identity.

mod cursor;
mod envelope;
mod logical_tree;

use std::error::Error;
use std::fmt;

pub(crate) use cursor::ArtifactCursor;
pub(crate) use envelope::{
    ArtifactPayloadWriter, HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND,
    LINEAR_REGRESSION_ARTIFACT_KIND, LOGISTIC_ARTIFACT_KIND, PAIRWISE_LINEAR_RANKER_ARTIFACT_KIND,
    RIDGE_ARTIFACT_KIND, STANDARD_SCALER_ARTIFACT_KIND,
    STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND, STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND,
    STANDARD_SCALER_RIDGE_PIPELINE_ARTIFACT_KIND, SchemaRole, artifact_version, decode_component,
    decode_legacy_envelope, decode_v2_envelope, encode_component, encode_v2_envelope,
};
pub(crate) use logical_tree::{decode_logical_tree, encode_logical_tree};

/// Current FerricML binary model artifact version.
pub const MODEL_ARTIFACT_VERSION: u16 = 2;

/// Maximum accepted size of one encoded model artifact.
pub const MAX_MODEL_ARTIFACT_BYTES: usize = 32 * 1024 * 1024;

/// Errors encountered while decoding or validating a model artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArtifactError {
    /// The artifact is shorter than the fixed envelope.
    Truncated,
    /// The artifact magic does not identify FerricML.
    InvalidMagic,
    /// The artifact version is unsupported.
    UnsupportedVersion { found: u16 },
    /// The estimator payload version is unsupported.
    UnsupportedPayloadVersion { found: u16 },
    /// Required envelope flags are not understood by this reader.
    UnsupportedRequiredFlags { found: u16 },
    /// The model kind is unsupported by the requested decoder.
    UnsupportedModelKind { found: u16 },
    /// The SHA-256 integrity footer does not match the payload.
    ChecksumMismatch,
    /// The embedded feature schema differs from the caller's requirement.
    FeatureSchemaMismatch,
    /// A count, flag, parameter, or floating-point value is invalid.
    InvalidPayload,
    /// Bytes remain after the complete model payload.
    TrailingBytes,
    /// The encoded artifact exceeds the hard reader limit.
    SizeLimitExceeded { limit: usize, actual: usize },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("model artifact is truncated"),
            Self::InvalidMagic => f.write_str("model artifact has invalid magic"),
            Self::UnsupportedVersion { found } => {
                write!(f, "unsupported model artifact version {found}")
            }
            Self::UnsupportedPayloadVersion { found } => {
                write!(f, "unsupported model payload version {found}")
            }
            Self::UnsupportedRequiredFlags { found } => {
                write!(f, "unsupported required artifact flags {found:#06x}")
            }
            Self::UnsupportedModelKind { found } => {
                write!(f, "unsupported model artifact kind {found}")
            }
            Self::ChecksumMismatch => f.write_str("model artifact checksum mismatch"),
            Self::FeatureSchemaMismatch => f.write_str("model artifact feature schema mismatch"),
            Self::InvalidPayload => f.write_str("model artifact payload is invalid"),
            Self::TrailingBytes => f.write_str("model artifact contains trailing bytes"),
            Self::SizeLimitExceeded { limit, actual } => {
                write!(f, "model artifact size {actual} exceeds limit {limit}")
            }
        }
    }
}

impl Error for ArtifactError {}
