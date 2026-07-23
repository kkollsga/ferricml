use super::{ArtifactCursor, ArtifactError, MODEL_ARTIFACT_VERSION};
use sha2::{Digest, Sha256};

const ARTIFACT_MAGIC: &[u8; 8] = b"FERRICML";
const ARTIFACT_CHECKSUM_BYTES: usize = 32;
const LEGACY_HEADER_BYTES: usize = 8 + 2 + 2 + 32;

pub(crate) const LOGISTIC_ARTIFACT_KIND: u16 = 1;

pub(crate) struct LegacyArtifactWriter {
    bytes: Vec<u8>,
}

impl LegacyArtifactWriter {
    pub(crate) fn new(kind: u16, feature_schema_sha256: [u8; 32], capacity: usize) -> Self {
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(ARTIFACT_MAGIC);
        bytes.extend_from_slice(&MODEL_ARTIFACT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&feature_schema_sha256);
        Self { bytes }
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        let checksum = Sha256::digest(&self.bytes);
        self.bytes.extend_from_slice(&checksum);
        self.bytes
    }
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
    if version != MODEL_ARTIFACT_VERSION {
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
