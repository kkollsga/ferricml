//! Deterministic thresholding of dense values to zero or one.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::data::MatrixView;

use super::scaling::{transform_preflighted, validate_transform_request};

/// Parameters for [`Binarizer`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BinarizerParams {
    threshold: f32,
}

impl BinarizerParams {
    /// Sets the value a feature must exceed to become one.
    ///
    /// The comparison is strict, so a value exactly at the threshold becomes
    /// `0.0`. The default threshold is `0.0`, which splits on sign and sends an
    /// exact zero to `0.0`.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Returns the value a feature must exceed to become one.
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Rejects a threshold no comparison could be meaningful against.
    fn validate(&self) -> Result<(), ModelError> {
        if self.threshold.is_finite() {
            Ok(())
        } else {
            Err(ModelError::InvalidThreshold)
        }
    }
}

/// Maps every value to `1.0` when it exceeds a threshold and `0.0` otherwise.
///
/// Stateless, like [`Normalizer`](super::Normalizer): the threshold is a
/// parameter the caller chose, not a statistic estimated from data, so nothing
/// about the fitting batch can influence a later one. There is consequently no
/// artifact — the only fitted value is the width a pipeline already validates.
///
/// The comparison is strictly greater-than. A value exactly at the threshold
/// becomes `0.0`, which makes the two output classes `(-inf, t]` and
/// `(t, +inf)` rather than leaving the boundary to rounding.
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::preprocessing::{Binarizer, BinarizerParams};
///
/// let data = DenseMatrix::new(vec![-1.0, 0.0, 1.0, 2.0], 4, 1)?;
/// let binarizer = Binarizer::fit(
///     &data.as_view(),
///     BinarizerParams::default().with_threshold(1.0),
/// )?;
///
/// // Strictly greater than: a value exactly at the threshold becomes 0.0.
/// assert_eq!(
///     binarizer.transform(&data.as_view())?.as_slice(),
///     &[0.0, 0.0, 0.0, 1.0],
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binarizer {
    n_features_in: usize,
    params: BinarizerParams,
}

impl Binarizer {
    /// Records the width this binarizer accepts, after validating the threshold.
    pub fn fit(data: &MatrixView<'_>, params: BinarizerParams) -> Result<Self, ModelError> {
        params.validate()?;
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
    pub const fn get_params(&self) -> &BinarizerParams {
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
}

impl Estimator for Binarizer {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for Binarizer {
    /// Nothing is fitted, so there is nothing to persist and nothing a
    /// per-sample weight could move.
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

impl HasParams for Binarizer {
    type Params = BinarizerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for Binarizer {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_transform_request(self.n_features_in, data, output)?;
        let threshold = self.params.threshold;
        // The map is monotone non-decreasing per column and lands in `{0, 1}`,
        // so the preflight can never fire; it runs anyway because the guarantee
        // belongs to the seam rather than to each scaler's own reasoning.
        transform_preflighted(data, output, |value, _| {
            f32::from(u8::from(value > threshold))
        })?;
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

    fn binarized(values: &[f32], threshold: f32) -> Vec<f32> {
        let data = matrix(values, 1, values.len());
        let fitted = Binarizer::fit(
            &data.as_view(),
            BinarizerParams::default().with_threshold(threshold),
        )
        .unwrap();
        fitted
            .transform(&data.as_view())
            .unwrap()
            .as_slice()
            .to_vec()
    }

    #[test]
    fn values_above_the_threshold_become_one_and_the_rest_become_zero() {
        assert_eq!(
            binarized(&[1.0, 2.0, 3.0, 10.0], 2.0),
            &[0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn the_comparison_is_strict_at_the_threshold_itself() {
        // The boundary is a contract, not a rounding accident: exactly the
        // threshold is below it.
        let just_above = 2.0_f32 + f32::EPSILON * 2.0;
        assert_eq!(binarized(&[2.0, just_above, 1.999], 2.0), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn the_default_threshold_splits_on_sign_and_sends_zero_low() {
        let data = matrix(&[-1.0, -0.0, 0.0, 1.0], 1, 4);
        let fitted = Binarizer::fit(&data.as_view(), BinarizerParams::default()).unwrap();
        assert_eq!(
            fitted.transform(&data.as_view()).unwrap().as_slice(),
            &[0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn a_negative_threshold_works_the_same_way() {
        assert_eq!(binarized(&[-1.0, -2.0, 0.0], -1.0), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_non_finite_threshold_is_rejected_before_any_width_is_recorded() {
        let data = matrix(&[1.0, 2.0], 1, 2);
        for threshold in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                Binarizer::fit(
                    &data.as_view(),
                    BinarizerParams::default().with_threshold(threshold)
                )
                .unwrap_err(),
                ModelError::InvalidThreshold,
                "threshold {threshold}"
            );
        }
    }

    #[test]
    fn every_output_is_exactly_zero_or_one() {
        let data = matrix(&[-1e30, -1.0, 0.0, 1e-30, 1.0, 1e30], 2, 3);
        let fitted = Binarizer::fit(&data.as_view(), BinarizerParams::default()).unwrap();
        let transformed = fitted.transform(&data.as_view()).unwrap();
        assert!(
            transformed
                .as_slice()
                .iter()
                .all(|value| *value == 0.0 || *value == 1.0)
        );
    }

    #[test]
    fn refitting_the_same_batch_is_deterministic() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let params = BinarizerParams::default().with_threshold(2.5);
        let first = Binarizer::fit(&data.as_view(), params).unwrap();
        let second = Binarizer::fit(&data.as_view(), params).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.get_params().threshold(), 2.5);
    }

    #[test]
    fn validates_width_and_workspace_before_writing() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let fitted = Binarizer::fit(&data.as_view(), BinarizerParams::default()).unwrap();

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
}
