//! Shared primitives for transforms that change the feature width.
//!
//! [`super::scaling`] is the sibling of this module and covers every transform
//! whose output is as wide as its input. Neither of its two central assumptions
//! survives a width change: it chunks the output buffer by `data.columns()`,
//! and it proves a batch finite by screening each column's extrema under a map
//! that is monotone in one input value. A width-changing transform has neither
//! property — one output column can read every input column, and the map from
//! a value to an output cell need not be monotone in anything.
//!
//! What is deliberately *not* different is the contract. A batch still
//! transforms completely or writes nothing, an output that is not finite is
//! still reported with its location rather than returned, and both shape checks
//! still happen before the first write. Those are tested guarantees on the
//! width-preserving side, and a width change is no reason to weaken them.
//!
//! # Why the callback is per cell
//!
//! [`expand_preflighted`] takes `Fn(&[f32], usize) -> f32` — one input row and
//! one output column — rather than a row-at-a-time writer. That shape is what
//! makes "write nothing on failure" free. A row-at-a-time writer needs
//! somewhere to put a row it has not yet proven finite, which is either an
//! allocation on the hot path or a partial write into the caller's buffer; a
//! per-cell callback can be *evaluated* without being stored, so the validation
//! pass needs no storage at all and the write pass is reached only once nothing
//! can fail. It is also exactly the shape
//! [`super::scaling::transform_preflighted`] already uses, so the two seams
//! read as one contract with two column rules rather than as two designs.
//!
//! The cost is that a caller which could share partial products between output
//! columns — a polynomial expansion is the obvious one — recomputes them per
//! cell. That is a real cost and it is taken deliberately: it buys the
//! guarantee above, the Criterion lanes make it visible, and per `CLAUDE.md` a
//! faster shape is a change to make when a measurement asks for it rather than
//! when a reading of the code suggests it might.

use crate::api::ModelError;
use crate::data::{DenseMatrix, MatrixView};

/// Validates an expansion request against the fitted widths.
///
/// Returns the required output length so callers cannot recompute it
/// differently. The input width, the length overflow, and the output length are
/// all checked before any write, so a rejected batch leaves `output` exactly as
/// the caller left it.
pub(super) fn validate_expansion_request(
    n_features_in: usize,
    n_features_out: usize,
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
            .checked_mul(n_features_out)
            .ok_or(ModelError::OutputShapeOverflow {
                rows: data.rows(),
                columns: n_features_out,
            })?;
    if output.len() != expected {
        return Err(ModelError::OutputLength {
            expected,
            actual: output.len(),
        });
    }
    Ok(expected)
}

/// Allocates an expansion output and fills it through `expand`.
///
/// The overflow check on the output length is the part worth stating once
/// rather than once per width-changing transformer.
pub(super) fn expand_allocating(
    n_features_out: usize,
    data: &MatrixView<'_>,
    expand: impl FnOnce(&MatrixView<'_>, &mut [f32]) -> Result<(), ModelError>,
) -> Result<DenseMatrix, ModelError> {
    let len = data
        .rows()
        .checked_mul(n_features_out)
        .ok_or(ModelError::OutputShapeOverflow {
            rows: data.rows(),
            columns: n_features_out,
        })?;
    let mut output = vec![0.0; len];
    expand(data, &mut output)?;
    Ok(DenseMatrix::from_validated_parts(
        output,
        data.rows(),
        n_features_out,
    ))
}

/// Writes `n_features_out` values per input row after proving every one of them
/// finite, and returns a validated view over exactly the values written.
///
/// `proven_finite` is the caller's own sufficient screen: `true` asserts that
/// no cell this batch produces can be non-finite, which lets the values be
/// written in one pass. It is a claim the caller has to have earned — a
/// polynomial expansion earns it by bounding each monomial with the product of
/// its factors' magnitudes, a spline basis under constant extrapolation earns
/// it by being a partition of unity, and a caller with no such argument passes
/// `false` and pays a validation pass. Passing `true` without the argument
/// would not corrupt memory, but it would let a non-finite value reach the
/// output, which is the one thing this function exists to prevent.
///
/// Returning the view rather than `()` is what keeps the finiteness guarantee
/// structural. This function is the code that proves it, so it is the only
/// place among the width-changing transformers that may reach
/// [`MatrixView::from_validated_parts`]; a caller wrapping `output` itself
/// would be restating a claim rather than making one.
pub(super) fn expand_preflighted<'output, F>(
    data: &MatrixView<'_>,
    output: &'output mut [f32],
    n_features_out: usize,
    proven_finite: bool,
    value_at: F,
) -> Result<MatrixView<'output>, ModelError>
where
    F: Fn(&[f32], usize) -> f32 + Copy,
{
    if !proven_finite {
        // Row-major order, so the location reported is the first offending one
        // a reader scanning the output would meet, and `output` is untouched.
        for (row_index, row) in data.iter_rows().enumerate() {
            for column in 0..n_features_out {
                if !value_at(row, column).is_finite() {
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
        .zip(output.chunks_exact_mut(n_features_out))
    {
        for (column, slot) in output_row.iter_mut().enumerate() {
            *slot = value_at(row, column);
        }
    }
    Ok(MatrixView::from_validated_parts(
        output,
        data.rows(),
        n_features_out,
    ))
}

/// Feature count below which a batch's magnitudes are bounded on the stack.
///
/// The same number [`super::scaling`] screens with, for the same reason: a
/// fixed array keeps the screen free of an allocation on the transform path,
/// and a batch wider than it simply pays the validation pass instead.
pub(super) const STACK_SCREEN_FEATURES: usize = 256;

/// The largest magnitude each input column carries, or `None` if the batch is
/// wider than the stack screen.
///
/// This is the input half of every sufficient screen a width-changing
/// transformer can offer: `|f(x)|` is bounded by `f` evaluated on these
/// magnitudes whenever `f` is a product of the row's values, which is what a
/// monomial is and what a spline basis is not.
pub(super) fn column_magnitude_bounds(
    data: &MatrixView<'_>,
) -> Option<[f64; STACK_SCREEN_FEATURES]> {
    if data.columns() > STACK_SCREEN_FEATURES {
        return None;
    }
    let mut bounds = [0.0_f64; STACK_SCREEN_FEATURES];
    for row in data.iter_rows() {
        for (column, &value) in row.iter().enumerate() {
            bounds[column] = bounds[column].max(f64::from(value.abs()));
        }
    }
    Some(bounds)
}

/// Largest expanded width FerricML will build.
///
/// A width is not merely a number here: a fitted expansion reserves a term per
/// output column, and a transformed batch reserves a value per output column
/// *per row*. So an unbounded width is an unbounded allocation, and the bound
/// has to exist for the failure to be a typed error rather than an abort.
///
/// The value matches [`super::scaling`]'s own artifact ceiling, deliberately,
/// so the crate has one answer to "how wide may a fitted transformer be"
/// instead of one per module. At this width a single thousand-row batch is
/// already four gigabytes of output, which is the honest scale of what is being
/// refused.
pub(super) const MAX_EXPANDED_FEATURES: usize = 1_000_000;

/// The binomial coefficient `C(n, k)`, or `None` if it overflows.
///
/// Multiply-then-divide with the smaller of `k` and `n - k`, which keeps every
/// intermediate an exact integer: after step `i` the running value is
/// `C(n - k + i, i)`, so the division never truncates. An intermediate can
/// still overflow where the final value would have fitted, which is why this
/// returns an option rather than saturating — the caller turns it into the same
/// typed refusal a genuinely too-wide expansion gets, and a width that only
/// *nearly* overflows `usize` is far past the ceiling above regardless.
pub(super) fn binomial(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for step in 1..=k {
        result = result.checked_mul(n - k + step)?;
        result /= step;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_binomial_coefficient_is_exact_where_it_fits() {
        assert_eq!(binomial(0, 0), Some(1));
        assert_eq!(binomial(5, 0), Some(1));
        assert_eq!(binomial(5, 5), Some(1));
        assert_eq!(binomial(5, 2), Some(10));
        assert_eq!(binomial(4, 2), Some(6));
        // The width of a degree-4 expansion over 8 features, and of a degree-10
        // one over 50 — the second being the request that made this bound
        // necessary.
        assert_eq!(binomial(12, 4), Some(495));
        assert_eq!(binomial(60, 10), Some(75_394_027_566));
        assert_eq!(binomial(3, 5), Some(0), "asking for more than exists");
    }

    #[test]
    fn the_binomial_coefficient_reports_overflow_rather_than_wrapping() {
        // Far past `usize`, so no arrangement of the multiply-divide order
        // brings it back.
        assert_eq!(binomial(100_000, 5_000), None);
        // And the symmetric argument is used, so a large `k` is as cheap as its
        // small complement rather than overflowing on the way.
        assert_eq!(binomial(60, 50), Some(75_394_027_566));
    }
}
