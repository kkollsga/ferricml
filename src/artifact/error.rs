use std::error::Error;
use std::fmt;

/// Errors encountered while decoding or validating a model artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArtifactError {
    /// The artifact is shorter than the fixed envelope.
    Truncated,
    /// The artifact magic does not identify FerricML.
    InvalidMagic,
    /// The artifact version is unsupported.
    UnsupportedVersion {
        /// Envelope version read from the artifact.
        found: u16,
    },
    /// The estimator payload version is unsupported.
    UnsupportedPayloadVersion {
        /// Payload version read from the artifact.
        found: u16,
    },
    /// Required envelope flags are not understood by this reader.
    UnsupportedRequiredFlags {
        /// The required-flag bits read from the artifact.
        found: u16,
    },
    /// The model kind is unsupported by the requested decoder.
    UnsupportedModelKind {
        /// Model-kind code read from the artifact.
        found: u16,
    },
    /// The SHA-256 integrity footer does not match the payload.
    ChecksumMismatch,
    /// The embedded feature schema differs from the caller's requirement.
    FeatureSchemaMismatch,
    /// A count, flag, parameter, or floating-point value is invalid.
    InvalidPayload,
    /// Bytes remain after the complete model payload.
    TrailingBytes,
    /// The encoded artifact exceeds the hard reader limit.
    SizeLimitExceeded {
        /// Hard byte limit the reader enforces.
        limit: usize,
        /// Byte length the artifact actually has.
        actual: usize,
    },
    /// The fitted model holds state this artifact schema cannot represent.
    ///
    /// Encoding refuses instead of writing bytes that would decode as a
    /// different model. A schema that gains the missing state is a new payload
    /// version, never a reinterpretation of the current one.
    UnsupportedModelState,
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
            Self::UnsupportedModelState => {
                f.write_str("fitted model state is unsupported by this artifact schema")
            }
        }
    }
}

impl Error for ArtifactError {}
