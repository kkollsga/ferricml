use super::{ArtifactCursor, ArtifactError, MAX_MODEL_ARTIFACT_BYTES, MODEL_ARTIFACT_VERSION};
use sha2::{Digest, Sha256};

const ARTIFACT_MAGIC: &[u8; 8] = b"FERRICML";
const LEGACY_ARTIFACT_VERSION: u16 = 1;
const ARTIFACT_CHECKSUM_BYTES: usize = 32;
const LEGACY_HEADER_BYTES: usize = 8 + 2 + 2 + 32;
const V2_FIXED_HEADER_BYTES: usize = 8 + 2 + 2 + 2 + 2 + 4 + 2 + 2;
const SCHEMA_RECORD_BYTES: usize = 2 + 2 + 32;
const COMPONENT_HEADER_BYTES: usize = 2 + 2 + 4;

pub(crate) const LOGISTIC_ARTIFACT_KIND: u16 = 1;
pub(crate) const LINEAR_REGRESSION_ARTIFACT_KIND: u16 = 2;
pub(crate) const RIDGE_ARTIFACT_KIND: u16 = 3;
pub(crate) const STANDARD_SCALER_ARTIFACT_KIND: u16 = 4;
pub(crate) const STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND: u16 = 5;
pub(crate) const STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND: u16 = 6;
pub(crate) const STANDARD_SCALER_RIDGE_PIPELINE_ARTIFACT_KIND: u16 = 7;
pub(crate) const PAIRWISE_LINEAR_RANKER_ARTIFACT_KIND: u16 = 8;
pub(crate) const HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND: u16 = 9;
pub(crate) const RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND: u16 = 10;
pub(crate) const ANY_REGRESSOR_ARTIFACT_KIND: u16 = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SchemaRole {
    Input,
    Transformed,
}

impl SchemaRole {
    const fn code(self) -> u16 {
        match self {
            Self::Input => 1,
            Self::Transformed => 2,
        }
    }
}

#[derive(Default)]
pub(crate) struct ArtifactPayloadWriter {
    bytes: Vec<u8>,
}

impl ArtifactPayloadWriter {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

pub(crate) fn artifact_version(bytes: &[u8]) -> Result<u16, ArtifactError> {
    if bytes.len() < ARTIFACT_MAGIC.len() + 2 {
        return Err(ArtifactError::Truncated);
    }
    Ok(u16::from_le_bytes(
        bytes[ARTIFACT_MAGIC.len()..ARTIFACT_MAGIC.len() + 2]
            .try_into()
            .expect("exact length"),
    ))
}

pub(crate) fn encode_component(
    kind: u16,
    version: u16,
    payload: &[u8],
) -> Result<Vec<u8>, ArtifactError> {
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: payload.len(),
        })?;
    let total_len = COMPONENT_HEADER_BYTES.checked_add(payload.len()).ok_or(
        ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: usize::MAX,
        },
    )?;
    if total_len > MAX_MODEL_ARTIFACT_BYTES {
        return Err(ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: total_len,
        });
    }
    let mut bytes = Vec::with_capacity(total_len);
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

pub(crate) fn decode_component<'a>(
    cursor: &mut ArtifactCursor<'a>,
    expected_kind: u16,
    expected_version: u16,
) -> Result<ArtifactCursor<'a>, ArtifactError> {
    let kind = cursor.u16()?;
    if kind != expected_kind {
        return Err(ArtifactError::InvalidPayload);
    }
    let version = cursor.u16()?;
    if version != expected_version {
        return Err(ArtifactError::UnsupportedPayloadVersion { found: version });
    }
    let payload_len = cursor.u32()? as usize;
    Ok(ArtifactCursor::new(cursor.take(payload_len)?))
}

pub(crate) fn encode_v2_envelope(
    kind: u16,
    payload_version: u16,
    schemas: &[(SchemaRole, [u8; 32])],
    payload: &[u8],
) -> Result<Vec<u8>, ArtifactError> {
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: payload.len(),
        })?;
    let schema_count = u16::try_from(schemas.len()).map_err(|_| ArtifactError::InvalidPayload)?;
    let schema_bytes = schemas
        .len()
        .checked_mul(SCHEMA_RECORD_BYTES)
        .ok_or(ArtifactError::InvalidPayload)?;
    let total_len = V2_FIXED_HEADER_BYTES
        .checked_add(schema_bytes)
        .and_then(|length| length.checked_add(payload.len()))
        .and_then(|length| length.checked_add(ARTIFACT_CHECKSUM_BYTES))
        .ok_or(ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: usize::MAX,
        })?;
    if total_len > MAX_MODEL_ARTIFACT_BYTES {
        return Err(ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: total_len,
        });
    }

    let mut bytes = Vec::with_capacity(total_len);
    bytes.extend_from_slice(ARTIFACT_MAGIC);
    bytes.extend_from_slice(&MODEL_ARTIFACT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&payload_version.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(&schema_count.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    for &(role, hash) in schemas {
        bytes.extend_from_slice(&role.code().to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&hash);
    }
    bytes.extend_from_slice(payload);
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    Ok(bytes)
}

pub(crate) fn decode_v2_envelope<'a>(
    bytes: &'a [u8],
    expected_kind: u16,
    expected_payload_version: u16,
    expected_schemas: &[(SchemaRole, [u8; 32])],
) -> Result<ArtifactCursor<'a>, ArtifactError> {
    if bytes.len() > MAX_MODEL_ARTIFACT_BYTES {
        return Err(ArtifactError::SizeLimitExceeded {
            limit: MAX_MODEL_ARTIFACT_BYTES,
            actual: bytes.len(),
        });
    }
    if bytes.len() < V2_FIXED_HEADER_BYTES + ARTIFACT_CHECKSUM_BYTES {
        return Err(ArtifactError::Truncated);
    }
    let (checksummed, checksum) = bytes.split_at(bytes.len() - ARTIFACT_CHECKSUM_BYTES);
    if &Sha256::digest(checksummed)[..] != checksum {
        return Err(ArtifactError::ChecksumMismatch);
    }

    let mut cursor = ArtifactCursor::new(checksummed);
    if cursor.take(ARTIFACT_MAGIC.len())? != ARTIFACT_MAGIC {
        return Err(ArtifactError::InvalidMagic);
    }
    let version = cursor.u16()?;
    if version != MODEL_ARTIFACT_VERSION {
        return Err(ArtifactError::UnsupportedVersion { found: version });
    }
    let kind = cursor.u16()?;
    if kind != expected_kind {
        return Err(ArtifactError::UnsupportedModelKind { found: kind });
    }
    let payload_version = cursor.u16()?;
    if payload_version != expected_payload_version {
        return Err(ArtifactError::UnsupportedPayloadVersion {
            found: payload_version,
        });
    }
    let required_flags = cursor.u16()?;
    if required_flags != 0 {
        return Err(ArtifactError::UnsupportedRequiredFlags {
            found: required_flags,
        });
    }
    let payload_len = cursor.u32()? as usize;
    let schema_count = cursor.u16()? as usize;
    if cursor.u16()? != 0 || schema_count != expected_schemas.len() {
        return Err(ArtifactError::InvalidPayload);
    }
    for &(expected_role, expected_hash) in expected_schemas {
        let role = cursor.u16()?;
        let schema_flags = cursor.u16()?;
        if role != expected_role.code() || schema_flags != 0 {
            return Err(ArtifactError::InvalidPayload);
        }
        if cursor.take(32)? != expected_hash {
            return Err(ArtifactError::FeatureSchemaMismatch);
        }
    }
    let payload = cursor.take(payload_len)?;
    if !cursor.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(ArtifactCursor::new(payload))
}

pub(crate) fn decode_legacy_envelope<'a>(
    bytes: &'a [u8],
    expected_kind: u16,
    expected_feature_schema_sha256: [u8; 32],
    minimum_payload_bytes: usize,
) -> Result<ArtifactCursor<'a>, ArtifactError> {
    let minimum_len = LEGACY_HEADER_BYTES
        .checked_add(minimum_payload_bytes)
        .and_then(|length| length.checked_add(ARTIFACT_CHECKSUM_BYTES))
        .ok_or(ArtifactError::InvalidPayload)?;
    if bytes.len() < minimum_len {
        return Err(ArtifactError::Truncated);
    }
    let (payload, checksum) = bytes.split_at(bytes.len() - ARTIFACT_CHECKSUM_BYTES);
    if &Sha256::digest(payload)[..] != checksum {
        return Err(ArtifactError::ChecksumMismatch);
    }

    let mut cursor = ArtifactCursor::new(payload);
    if cursor.take(ARTIFACT_MAGIC.len())? != ARTIFACT_MAGIC {
        return Err(ArtifactError::InvalidMagic);
    }
    let version = cursor.u16()?;
    if version != LEGACY_ARTIFACT_VERSION {
        return Err(ArtifactError::UnsupportedVersion { found: version });
    }
    let kind = cursor.u16()?;
    if kind != expected_kind {
        return Err(ArtifactError::UnsupportedModelKind { found: kind });
    }
    if cursor.take(32)? != expected_feature_schema_sha256 {
        return Err(ArtifactError::FeatureSchemaMismatch);
    }
    Ok(cursor)
}
