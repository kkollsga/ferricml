//! Deterministic dense scaling by each feature's largest magnitude.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::artifact::{ArtifactError, MAX_ABS_SCALER_ARTIFACT_KIND};
use crate::data::MatrixView;

use super::scaling::{
    decode_scaler_artifact, encode_scaler_artifact, transform_preflighted,
    validate_transform_request,
};

/// Parameters for [`MaxAbsScaler`].
///
/// Dividing by the largest fitted magnitude has nothing to tune. This type
/// exists so the scaler is fitted exactly like every other FerricML
/// transformer, and so a later option can be added without changing `fit`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaxAbsScalerParams;

/// Fitted per-feature scaling by the largest observed magnitude.
///
/// Each column is divided by `max(|x|)`, so fitted values land in
/// `-1.0..=1.0` and a zero stays a zero. A column that is zero everywhere has
/// no magnitude to divide by: it keeps a divisor of one and passes through
/// unchanged rather than producing a non-finite value.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxAbsScaler {
    n_features_in: usize,
    params: MaxAbsScalerParams,
    max_abs: Vec<f64>,
    scales: Vec<f64>,
}

impl MaxAbsScaler {
    /// Fits per-feature maximum magnitudes in fixed row order.
    pub fn fit(data: &MatrixView<'_>, params: MaxAbsScalerParams) -> Result<Self, ModelError> {
        let columns = data.columns();
        let mut max_abs = vec![0.0_f64; columns];
        for row in data.iter_rows() {
            for (largest, &value) in max_abs.iter_mut().zip(row) {
                *largest = largest.max(f64::from(value).abs());
            }
        }
        let scales = max_abs.iter().copied().map(derive_scale).collect();

        Ok(Self {
            n_features_in: columns,
            params,
            max_abs,
            scales,
        })
    }

    /// Returns the fitted per-feature maximum magnitudes as `f64` values.
    pub fn max_abs(&self) -> &[f64] {
        &self.max_abs
    }

    /// Returns the fitted divisors; an all-zero column uses one.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Returns the fitted input width.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the fitted output width.
    pub const fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact transformation parameters.
    pub const fn get_params(&self) -> &MaxAbsScalerParams {
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

    /// Encodes fitted scaling state with explicit input and transformed schemas.
    pub fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        encode_scaler_artifact(
            MAX_ABS_SCALER_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            self.n_features_in,
            &[],
            1,
            |feature, state| state.f64(self.max_abs[feature]),
        )
    }

    /// Decodes fitted scaling state after checking both schemas.
    pub fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let (n_features_in, _, mut state) = decode_scaler_artifact(
            bytes,
            MAX_ABS_SCALER_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            0,
        )?;
        let mut max_abs = Vec::with_capacity(n_features_in);
        let mut scales = Vec::with_capacity(n_features_in);
        for _ in 0..n_features_in {
            let largest = state.f64()?;
            if !largest.is_finite() || largest < 0.0 {
                return Err(ArtifactError::InvalidPayload);
            }
            max_abs.push(largest);
            scales.push(derive_scale(largest));
        }
        if !state.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in,
            params: MaxAbsScalerParams,
            max_abs,
            scales,
        })
    }
}

/// The divisor that maps a column's largest magnitude onto one.
///
/// A column that is zero everywhere has no magnitude to normalize by, so it
/// keeps a divisor of one and passes through unchanged. This is the documented
/// reference treatment of an all-zero column and what keeps a division by zero
/// out of the transform.
fn derive_scale(max_abs: f64) -> f64 {
    if max_abs == 0.0 { 1.0 } else { max_abs }
}

impl Estimator for MaxAbsScaler {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for MaxAbsScaler {
    /// The largest magnitude is an order statistic: a per-sample weight cannot
    /// move it, so there is no weighted entry point to declare.
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_artifact(true);
}

impl HasParams for MaxAbsScaler {
    type Params = MaxAbsScalerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for MaxAbsScaler {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_transform_request(self.n_features_in, data, output)?;
        transform_preflighted(data, output, |value, column| {
            (f64::from(value) / self.scales[column]) as f32
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

    fn matrix() -> DenseMatrix {
        DenseMatrix::new(vec![1.0, -4.0, 0.0, -2.0, 2.0, 0.0, 4.0, 1.0, 0.0], 3, 3).unwrap()
    }

    #[test]
    fn fits_magnitudes_and_maps_them_into_the_signed_unit_interval() {
        let scaler = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        assert_eq!(scaler.max_abs(), &[4.0, 4.0, 0.0]);
        let transformed = scaler.transform(&matrix().as_view()).unwrap();
        assert_eq!(transformed.get(0, 0), Some(0.25));
        assert_eq!(transformed.get(2, 0), Some(1.0));
        assert_eq!(transformed.get(0, 1), Some(-1.0));
    }

    #[test]
    fn an_all_zero_column_keeps_a_divisor_of_one_and_passes_through() {
        let scaler = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        assert_eq!(scaler.scales()[2], 1.0);
        for row in 0..3 {
            assert_eq!(
                scaler.transform(&matrix().as_view()).unwrap().get(row, 2),
                Some(0.0)
            );
        }
    }

    #[test]
    fn a_negative_only_column_uses_its_magnitude_and_keeps_its_sign() {
        let negative = DenseMatrix::new(vec![-1.0, -5.0], 2, 1).unwrap();
        let scaler = MaxAbsScaler::fit(&negative.as_view(), MaxAbsScalerParams).unwrap();
        assert_eq!(scaler.max_abs(), &[5.0]);
        assert_eq!(
            scaler.transform(&negative.as_view()).unwrap().as_slice(),
            &[-0.2, -1.0]
        );
    }

    #[test]
    fn refitting_the_same_batch_is_deterministic() {
        let first = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        let second = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn validates_width_and_workspace_before_writing() {
        let scaler = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        let mut output = [91.0; 8];
        assert_eq!(
            scaler
                .transform_into(&matrix().as_view(), &mut output)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 9,
                actual: 8
            }
        );
        assert_eq!(output, [91.0; 8]);

        let narrow = DenseMatrix::new(vec![1.0, 2.0], 1, 2).unwrap();
        let mut narrow_output = [91.0; 2];
        assert_eq!(
            scaler
                .transform_into(&narrow.as_view(), &mut narrow_output)
                .unwrap_err(),
            ModelError::FeatureDimension {
                expected: 3,
                actual: 2
            }
        );
        assert_eq!(narrow_output, [91.0; 2]);
    }

    #[test]
    fn overflow_reports_the_first_row_major_location_without_partial_writes() {
        let tiny = DenseMatrix::new(vec![f32::MIN_POSITIVE], 1, 1).unwrap();
        let scaler = MaxAbsScaler::fit(&tiny.as_view(), MaxAbsScalerParams).unwrap();
        // Row 0 divides to a large but finite value; row 1 is the first that
        // overflows, and nothing is written because it is found first.
        let extreme = DenseMatrix::new(vec![1.0, f32::MAX], 2, 1).unwrap();
        let mut output = [73.0; 2];
        assert_eq!(
            scaler
                .transform_into(&extreme.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 1, column: 0 }
        );
        assert_eq!(output, [73.0; 2]);
    }

    #[test]
    fn artifact_is_deterministic_and_schema_bound() {
        let scaler = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(bytes, scaler.to_artifact([1; 32], [2; 32]).unwrap());
        assert_eq!(
            MaxAbsScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap(),
            scaler
        );
        assert_eq!(
            MaxAbsScaler::from_artifact(&bytes, [1; 32], [3; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
    }

    #[test]
    fn artifact_kinds_do_not_decode_as_each_other() {
        use super::super::{MinMaxScaler, MinMaxScalerParams};

        let scaler = MaxAbsScaler::fit(&matrix().as_view(), MaxAbsScalerParams).unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert!(matches!(
            MinMaxScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::UnsupportedModelKind { .. }
        ));

        let other = MinMaxScaler::fit(&matrix().as_view(), MinMaxScalerParams::default()).unwrap();
        let other_bytes = other.to_artifact([1; 32], [2; 32]).unwrap();
        assert!(matches!(
            MaxAbsScaler::from_artifact(&other_bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::UnsupportedModelKind { .. }
        ));
    }

    #[test]
    fn artifact_rejects_a_negative_magnitude() {
        let invalid = MaxAbsScaler {
            n_features_in: 1,
            params: MaxAbsScalerParams,
            max_abs: vec![-1.0],
            scales: vec![-1.0],
        };
        let bytes = invalid.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            MaxAbsScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }
}
