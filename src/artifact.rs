//! Stable, bounded model-artifact errors and format identity.

use std::error::Error;
use std::fmt;

/// Current FerricML binary model artifact version.
pub const MODEL_ARTIFACT_VERSION: u16 = 1;

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
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("model artifact is truncated"),
            Self::InvalidMagic => f.write_str("model artifact has invalid magic"),
            Self::UnsupportedVersion { found } => {
                write!(f, "unsupported model artifact version {found}")
            }
            Self::UnsupportedModelKind { found } => {
                write!(f, "unsupported model artifact kind {found}")
            }
            Self::ChecksumMismatch => f.write_str("model artifact checksum mismatch"),
            Self::FeatureSchemaMismatch => f.write_str("model artifact feature schema mismatch"),
            Self::InvalidPayload => f.write_str("model artifact payload is invalid"),
            Self::TrailingBytes => f.write_str("model artifact contains trailing bytes"),
        }
    }
}

impl Error for ArtifactError {}
