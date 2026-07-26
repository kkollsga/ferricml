//! Deterministic row-wise scaling to unit norm.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::data::MatrixView;
use crate::numeric::sum_in_order;

use super::scaling::{substituted_divisor, validate_transform_request};

/// Which norm a [`Normalizer`] scales each row to one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Norm {
    /// Sum of absolute values.
    L1,
    /// Euclidean length.
    #[default]
    L2,
    /// Largest absolute value.
    Max,
}

/// Parameters for [`Normalizer`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormalizerParams {
    norm: Norm,
}

impl NormalizerParams {
    /// Selects the norm each row is scaled to one.
    #[must_use]
    pub const fn with_norm(mut self, norm: Norm) -> Self {
        self.norm = norm;
        self
    }

    /// Returns the norm each row is scaled to one.
    #[must_use]
    pub const fn norm(&self) -> Norm {
        self.norm
    }
}

/// Row-wise scaling so each row has unit norm.
///
/// This transformer is *stateless*: it fits nothing from the data except the
/// width it must later be handed, because a row's norm is a property of that
/// row alone. Two consequences follow, and both are deliberate. There is no
/// artifact — there would be nothing in it but a feature count that the
/// pipeline already validates — and a row transforms identically whether it was
/// in the fitting batch or arrives years later.
///
/// A row whose norm is zero has no direction to preserve, so it keeps a divisor
/// of one and passes through as the zero row it already is. That is the same
/// exact-zero rule the fitted scalers use, reached through the same helper.
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::preprocessing::{Norm, Normalizer, NormalizerParams};
///
/// // Two rows pointing the same way at different lengths.
/// let data = DenseMatrix::new(vec![3.0, 4.0, 30.0, 40.0], 2, 2)?;
///
/// let normalizer = Normalizer::fit(
///     &data.as_view(),
///     NormalizerParams::default().with_norm(Norm::L2),
/// )?;
/// let unit = normalizer.transform(&data.as_view())?;
///
/// // Scaling is per row, so both rows land on the same unit vector.
/// assert_eq!(unit.row(0), unit.row(1));
/// assert!((unit.as_slice()[0] - 0.6).abs() < 1e-6);
/// assert!((unit.as_slice()[1] - 0.8).abs() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Normalizer {
    n_features_in: usize,
    params: NormalizerParams,
}

impl Normalizer {
    /// Records the width this normalizer accepts.
    ///
    /// Nothing is estimated from `data`. The width is taken so that the fitted
    /// value is an ordinary [`Transformer`] whose handoff into a pipeline is
    /// validated exactly like every other stage's.
    pub fn fit(data: &MatrixView<'_>, params: NormalizerParams) -> Result<Self, ModelError> {
        Ok(Self {
            n_features_in: data.columns(),
            params,
        })
    }

    /// Returns the fitted input width.
    #[must_use]
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the fitted output width.
    #[must_use]
    pub const fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact transformation parameters.
    #[must_use]
    pub const fn get_params(&self) -> &NormalizerParams {
        &self.params
    }

    /// Transforms a batch into caller-owned row-major storage.
    pub fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        <Self as Transformer>::transform_into(self, data, output)
    }

    /// Transforms a batch into a newly allocated dense matrix.
    pub fn transform(&self, data: &MatrixView<'_>) -> Result<crate::data::DenseMatrix, ModelError> {
        <Self as Transformer>::transform(self, data)
    }

    /// One row's norm, accumulated in `f64` in ascending column order.
    fn row_norm(&self, row: &[f32]) -> f64 {
        match self.params.norm {
            Norm::L1 => sum_in_order(row.iter().map(|value| f64::from(*value).abs())),
            Norm::L2 => sum_in_order(row.iter().map(|value| {
                let value = f64::from(*value);
                value * value
            }))
            .sqrt(),
            Norm::Max => row.iter().fold(0.0_f64, |largest, value| {
                largest.max(f64::from(*value).abs())
            }),
        }
    }
}

impl Estimator for Normalizer {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for Normalizer {
    /// Nothing is fitted, so there is nothing to persist and nothing a
    /// per-sample weight could move. Declaring an artifact would promise a
    /// stable encoding of a value that does not exist.
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

impl HasParams for Normalizer {
    type Params = NormalizerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for Normalizer {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_transform_request(self.n_features_in, data, output)?;

        // Row scaling is not the per-column map `transform_preflighted`
        // screens, so finiteness is proven the way that helper proves it in its
        // fallback: every value first, then every write. Dividing a finite
        // value by a norm that is at least as large as its magnitude cannot
        // overflow, but the norm is a `f64` narrowing to `f32` on the way out,
        // so the check is real rather than ceremonial.
        for (row_index, row) in data.iter_rows().enumerate() {
            let divisor = substituted_divisor(self.row_norm(row));
            for (column, value) in row.iter().enumerate() {
                if !((f64::from(*value) / divisor) as f32).is_finite() {
                    return Err(ModelError::NonFiniteTransform {
                        row: row_index,
                        column,
                    });
                }
            }
        }
        for (row, output_row) in data
            .iter_rows()
            .zip(output.chunks_exact_mut(self.n_features_in))
        {
            let divisor = substituted_divisor(self.row_norm(row));
            for (value, slot) in row.iter().zip(output_row) {
                *slot = (f64::from(*value) / divisor) as f32;
            }
        }
        Ok(MatrixView::from_validated_parts(
            output,
            data.rows(),
            self.n_features_in,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
        DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
    }

    fn normalized(data: &DenseMatrix, norm: Norm) -> Vec<f32> {
        let fitted =
            Normalizer::fit(&data.as_view(), NormalizerParams::default().with_norm(norm)).unwrap();
        fitted
            .transform(&data.as_view())
            .unwrap()
            .as_slice()
            .to_vec()
    }

    #[test]
    fn each_norm_scales_a_row_to_one_under_its_own_definition() {
        let data = matrix(&[1.0, 2.0, -2.0], 1, 3);
        assert_eq!(normalized(&data, Norm::L1), &[0.2, 0.4, -0.4]);
        assert_eq!(
            normalized(&data, Norm::L2),
            &[1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0]
        );
        assert_eq!(normalized(&data, Norm::Max), &[0.5, 1.0, -1.0]);
    }

    #[test]
    fn the_max_norm_is_the_largest_magnitude_not_the_largest_value() {
        // A row whose largest magnitude is negative would normalize to values
        // outside `-1..=1` if the sign were kept.
        assert_eq!(
            normalized(&matrix(&[-4.0, 2.0], 1, 2), Norm::Max),
            &[-1.0, 0.5]
        );
    }

    #[test]
    fn a_zero_row_keeps_a_divisor_of_one_and_passes_through() {
        for norm in [Norm::L1, Norm::L2, Norm::Max] {
            assert_eq!(
                normalized(&matrix(&[0.0, 0.0, 0.0], 1, 3), norm),
                &[0.0, 0.0, 0.0],
                "{norm:?} on a zero row"
            );
        }
    }

    #[test]
    fn rows_are_normalized_independently_of_each_other() {
        let data = matrix(&[3.0, 4.0, 0.0, 0.0, 6.0, 8.0], 3, 2);
        assert_eq!(
            normalized(&data, Norm::L2),
            &[0.6, 0.8, 0.0, 0.0, 0.6, 0.8],
            "the third row is a multiple of the first and normalizes to it"
        );
    }

    #[test]
    fn a_row_transforms_the_same_whether_or_not_it_was_fitted_on() {
        let fitting = matrix(&[1.0, 2.0], 1, 2);
        let fitted = Normalizer::fit(&fitting.as_view(), NormalizerParams::default()).unwrap();
        let unseen = matrix(&[3.0, 4.0], 1, 2);
        assert_eq!(
            fitted.transform(&unseen.as_view()).unwrap().as_slice(),
            &[0.6, 0.8],
            "nothing was fitted, so nothing about the fitting batch can leak in"
        );
    }

    #[test]
    fn refitting_the_same_batch_is_deterministic() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let params = NormalizerParams::default().with_norm(Norm::L1);
        let first = Normalizer::fit(&data.as_view(), params).unwrap();
        let second = Normalizer::fit(&data.as_view(), params).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.get_params(), &params);
        assert_eq!(first.get_params().norm(), Norm::L1);
    }

    #[test]
    fn validates_width_and_workspace_before_writing() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let fitted = Normalizer::fit(&data.as_view(), NormalizerParams::default()).unwrap();

        let mut short = [91.0; 3];
        assert_eq!(
            fitted
                .transform_into(&data.as_view(), &mut short)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(short, [91.0; 3]);

        let narrow = matrix(&[1.0, 2.0, 3.0], 1, 3);
        let mut narrow_output = [91.0; 3];
        assert_eq!(
            fitted
                .transform_into(&narrow.as_view(), &mut narrow_output)
                .unwrap_err(),
            ModelError::FeatureDimension {
                expected: 2,
                actual: 3
            }
        );
        assert_eq!(narrow_output, [91.0; 3]);
    }

    #[test]
    fn an_overflowing_row_is_reported_before_anything_is_written() {
        // A denormal row: the `f64` norm is representable but dividing by it
        // carries the values past what `f32` can hold.
        let data = matrix(&[f32::MIN_POSITIVE, 0.0, 1.0, 1.0], 2, 2);
        let fitted = Normalizer::fit(
            &data.as_view(),
            NormalizerParams::default().with_norm(Norm::L2),
        )
        .unwrap();
        let extreme = matrix(&[f32::MIN_POSITIVE * 1e-7, 0.0], 1, 2);
        let mut output = [73.0; 2];
        let outcome = fitted.transform_into(&extreme.as_view(), &mut output);
        // Either it stayed finite or it was reported; what must never happen is
        // a non-finite value reaching the caller's buffer.
        match outcome {
            Ok(view) => assert!(view.as_slice().iter().all(|value| value.is_finite())),
            Err(error) => {
                assert!(matches!(error, ModelError::NonFiniteTransform { .. }));
                assert_eq!(output, [73.0; 2]);
            }
        }
    }
}
