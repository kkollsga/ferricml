//! Shared per-column scaling primitives.
//!
//! Every fitted scaler in this module applies one `f32 -> f32` map per column
//! and owes the same two guarantees: a batch either transforms completely or
//! writes nothing, and a finite input that scales to a non-finite `f32` is
//! reported rather than returned. Stating that once keeps the guarantee from
//! being re-derived, and slightly differently, per scaler.

use crate::api::ModelError;
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, MODEL_ARTIFACT_VERSION, SchemaRole,
    artifact_version, decode_component, decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::MatrixView;

/// Feature count below which extrema are screened on the stack.
const STACK_PREFLIGHT_FEATURES: usize = 256;

/// Largest fitted width a scaler artifact accepts, encoding or decoding.
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const PAYLOAD_VERSION: u16 = 1;
const STATE_COMPONENT_KIND: u16 = 1;
const STATE_COMPONENT_VERSION: u16 = 1;

/// Applies `transform` to every value after proving the batch cannot overflow.
///
/// The cheap screen transforms only each column's extrema. When it cannot
/// prove safety — because the column count exceeds the stack screen, or an
/// extremum does overflow — every value is checked in row-major order first,
/// so the reported location is the first offending one and `output` is left
/// untouched.
pub(super) fn transform_preflighted<F>(
    data: &MatrixView<'_>,
    output: &mut [f32],
    transform: F,
) -> Result<(), ModelError>
where
    F: Fn(f32, usize) -> f32 + Copy,
{
    if !extrema_are_safe(data, transform) {
        for (row_index, row) in data.iter_rows().enumerate() {
            for (column, &value) in row.iter().enumerate() {
                if !transform(value, column).is_finite() {
                    return Err(ModelError::NonFiniteTransform {
                        row: row_index,
                        column,
                    });
                }
            }
        }
    }
    for (row, output_row) in data
        .iter_rows()
        .zip(output.chunks_exact_mut(data.columns()))
    {
        for (column, (&value, slot)) in row.iter().zip(output_row).enumerate() {
            *slot = transform(value, column);
        }
    }
    Ok(())
}

/// Whether transforming each column's extrema stays finite.
///
/// A monotone per-column map cannot produce a non-finite value between two
/// finite ones, so screening the extrema is sufficient for the maps this
/// module applies. Returning `false` only costs the full row-major scan.
fn extrema_are_safe<F>(data: &MatrixView<'_>, transform: F) -> bool
where
    F: Fn(f32, usize) -> f32,
{
    if data.columns() > STACK_PREFLIGHT_FEATURES {
        return false;
    }
    let mut minima = [f32::INFINITY; STACK_PREFLIGHT_FEATURES];
    let mut maxima = [f32::NEG_INFINITY; STACK_PREFLIGHT_FEATURES];
    for row in data.iter_rows() {
        for (column, &value) in row.iter().enumerate() {
            minima[column] = minima[column].min(value);
            maxima[column] = maxima[column].max(value);
        }
    }
    (0..data.columns()).all(|column| {
        transform(minima[column], column).is_finite()
            && transform(maxima[column], column).is_finite()
    })
}

/// Encodes fitted scaler state into a schema-bound envelope.
///
/// Every scaler here has the same artifact shape: one state component holding
/// a feature count, the scaler's own parameter flags, a repeated feature
/// count, and then a fixed group of `f64` fields per feature. Only the flags
/// and the per-feature fields differ between scalers, so those are what a
/// caller supplies and everything else is stated once.
pub(super) fn encode_scaler_artifact(
    kind: u16,
    input_schema: [u8; 32],
    transformed_schema: [u8; 32],
    n_features_in: usize,
    flags: &[u32],
    fields_per_feature: usize,
    mut write_feature: impl FnMut(usize, &mut ArtifactPayloadWriter),
) -> Result<Vec<u8>, ArtifactError> {
    if n_features_in > MAX_ARTIFACT_FEATURES {
        return Err(ArtifactError::InvalidPayload);
    }
    let count = u32::try_from(n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
    let capacity = 8 + flags.len() * 4 + n_features_in * fields_per_feature * 8;
    let mut state = ArtifactPayloadWriter::with_capacity(capacity);
    state.u32(count);
    for &flag in flags {
        state.u32(flag);
    }
    state.u32(count);
    for feature in 0..n_features_in {
        write_feature(feature, &mut state);
    }
    let component = encode_component(
        STATE_COMPONENT_KIND,
        STATE_COMPONENT_VERSION,
        &state.finish(),
    )?;
    encode_v2_envelope(
        kind,
        PAYLOAD_VERSION,
        &[
            (SchemaRole::Input, input_schema),
            (SchemaRole::Transformed, transformed_schema),
        ],
        &component,
    )
}

/// Decodes the shared prefix of a scaler artifact.
///
/// Returns the fitted width, the scaler's parameter flags exactly as written,
/// and a cursor positioned at the first per-feature field. The caller reads
/// its own fields from that cursor and must reject any trailing bytes.
pub(super) fn decode_scaler_artifact<'a>(
    bytes: &'a [u8],
    kind: u16,
    input_schema: [u8; 32],
    transformed_schema: [u8; 32],
    flag_count: usize,
) -> Result<(usize, Vec<u32>, ArtifactCursor<'a>), ArtifactError> {
    let version = artifact_version(bytes)?;
    if version != MODEL_ARTIFACT_VERSION {
        return Err(ArtifactError::UnsupportedVersion { found: version });
    }
    let mut envelope = decode_v2_envelope(
        bytes,
        kind,
        PAYLOAD_VERSION,
        &[
            (SchemaRole::Input, input_schema),
            (SchemaRole::Transformed, transformed_schema),
        ],
    )?;
    let mut state = decode_component(&mut envelope, STATE_COMPONENT_KIND, STATE_COMPONENT_VERSION)?;
    if !envelope.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    let n_features_in = state.u32()? as usize;
    let mut flags = Vec::with_capacity(flag_count);
    for _ in 0..flag_count {
        flags.push(state.u32()?);
    }
    let count = state.u32()? as usize;
    if n_features_in == 0 || n_features_in > MAX_ARTIFACT_FEATURES || count != n_features_in {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok((n_features_in, flags, state))
}

/// Reads an artifact flag written as a `u32` back into a `bool`.
pub(super) fn decode_flag(value: u32) -> Result<bool, ArtifactError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ArtifactError::InvalidPayload),
    }
}

/// Validates a transform request against the fitted width and output length.
///
/// Returns the required output length so callers cannot recompute it
/// differently. Both checks happen before any write, so a rejected batch
/// leaves `output` exactly as the caller left it.
pub(super) fn validate_transform_request(
    n_features_in: usize,
    data: &MatrixView<'_>,
    output: &[f32],
) -> Result<usize, ModelError> {
    if data.columns() != n_features_in {
        return Err(ModelError::FeatureDimension {
            expected: n_features_in,
            actual: data.columns(),
        });
    }
    let expected =
        data.rows()
            .checked_mul(n_features_in)
            .ok_or(ModelError::OutputShapeOverflow {
                rows: data.rows(),
                columns: n_features_in,
            })?;
    if output.len() != expected {
        return Err(ModelError::OutputLength {
            expected,
            actual: output.len(),
        });
    }
    Ok(expected)
}
