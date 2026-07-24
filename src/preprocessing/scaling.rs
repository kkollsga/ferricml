//! Shared per-column scaling primitives.
//!
//! Every fitted scaler in this module applies one `f32 -> f32` map per column
//! and owes the same two guarantees: a batch either transforms completely or
//! writes nothing, and a finite input that scales to a non-finite `f32` is
//! reported rather than returned. Stating that once keeps the guarantee from
//! being re-derived, and slightly differently, per scaler.

use crate::api::ModelError;
use crate::data::MatrixView;

/// Feature count below which extrema are screened on the stack.
const STACK_PREFLIGHT_FEATURES: usize = 256;

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
