//! Deterministic dense scaling onto the unit interval.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::artifact::{ArtifactError, MIN_MAX_SCALER_ARTIFACT_KIND};
use crate::data::MatrixView;

use super::scaling::{
    decode_flag, decode_scaler_artifact, encode_scaler_artifact, transform_preflighted,
    validate_transform_request,
};

/// Parameters for [`MinMaxScaler`].
///
/// FerricML claims the default output range only. A configurable range needs a
/// validated parameter type and its own error, so it is deliberately left out
/// until a caller needs it rather than guessed at now.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MinMaxScalerParams {
    clip: bool,
}

impl MinMaxScalerParams {
    /// Enables or disables clamping transformed values into `0.0..=1.0`.
    ///
    /// Fitted minima and maxima come from the training batch, so a later batch
    /// containing more extreme values transforms outside the unit interval.
    /// Clipping is off by default, which reports those values as they are.
    #[must_use]
    pub const fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Returns whether transformed values are clamped into `0.0..=1.0`.
    #[must_use]
    pub const fn clip_enabled(&self) -> bool {
        self.clip
    }
}

/// Fitted per-feature scaling onto `0.0..=1.0`.
///
/// Each column is mapped so the smallest fitted value becomes `0.0` and the
/// largest becomes `1.0`. A column with no spread has no such map: it keeps a
/// divisor of one and transforms to `0.0`, so a constant feature stays
/// constant instead of producing a non-finite value.
#[derive(Clone, Debug, PartialEq)]
pub struct MinMaxScaler {
    n_features_in: usize,
    params: MinMaxScalerParams,
    data_min: Vec<f64>,
    data_max: Vec<f64>,
    scales: Vec<f64>,
    offsets: Vec<f64>,
}

impl MinMaxScaler {
    /// Fits per-feature minima and maxima in fixed row order.
    pub fn fit(data: &MatrixView<'_>, params: MinMaxScalerParams) -> Result<Self, ModelError> {
        let columns = data.columns();
        let mut data_min = vec![f64::INFINITY; columns];
        let mut data_max = vec![f64::NEG_INFINITY; columns];
        for row in data.iter_rows() {
            for ((minimum, maximum), &value) in data_min.iter_mut().zip(&mut data_max).zip(row) {
                let value = f64::from(value);
                *minimum = minimum.min(value);
                *maximum = maximum.max(value);
            }
        }

        let mut scales = Vec::with_capacity(columns);
        let mut offsets = Vec::with_capacity(columns);
        for (&minimum, &maximum) in data_min.iter().zip(&data_max) {
            let scale = derive_scale(minimum, maximum);
            if !scale.is_finite() {
                return Err(ModelError::NumericalOverflow);
            }
            offsets.push(-minimum * scale);
            scales.push(scale);
        }

        Ok(Self {
            n_features_in: columns,
            params,
            data_min,
            data_max,
            scales,
            offsets,
        })
    }

    /// Returns the fitted per-feature minima as `f64` values.
    pub fn data_min(&self) -> &[f64] {
        &self.data_min
    }

    /// Returns the fitted per-feature maxima as `f64` values.
    pub fn data_max(&self) -> &[f64] {
        &self.data_max
    }

    /// Returns the fitted multipliers; a column with no spread uses one.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Returns the fitted per-feature offsets applied after scaling.
    pub fn offsets(&self) -> &[f64] {
        &self.offsets
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
    pub const fn get_params(&self) -> &MinMaxScalerParams {
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
            MIN_MAX_SCALER_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            self.n_features_in,
            &[u32::from(self.params.clip)],
            2,
            |feature, state| {
                state.f64(self.data_min[feature]);
                state.f64(self.data_max[feature]);
            },
        )
    }

    /// Decodes fitted scaling state after checking both schemas.
    pub fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let (n_features_in, flags, mut state) = decode_scaler_artifact(
            bytes,
            MIN_MAX_SCALER_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            1,
        )?;
        let clip = decode_flag(flags[0])?;
        // Two `f64` fields per feature: the reservation is clamped to the
        // bytes actually present, never to the declared width alone.
        let capacity = state.bounded_capacity(n_features_in, 2 * 8);
        let mut data_min = Vec::with_capacity(capacity);
        let mut data_max = Vec::with_capacity(capacity);
        let mut scales = Vec::with_capacity(capacity);
        let mut offsets = Vec::with_capacity(capacity);
        for _ in 0..n_features_in {
            let minimum = state.f64()?;
            let maximum = state.f64()?;
            if !minimum.is_finite() || !maximum.is_finite() || maximum < minimum {
                return Err(ArtifactError::InvalidPayload);
            }
            let scale = derive_scale(minimum, maximum);
            let offset = -minimum * scale;
            if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
                return Err(ArtifactError::InvalidPayload);
            }
            data_min.push(minimum);
            data_max.push(maximum);
            scales.push(scale);
            offsets.push(offset);
        }
        if !state.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in,
            params: MinMaxScalerParams { clip },
            data_min,
            data_max,
            scales,
            offsets,
        })
    }
}

/// The multiplier that maps `[minimum, maximum]` onto `0.0..=1.0`.
///
/// A column with no spread has no such multiplier, so it keeps one and is
/// carried to `0.0` by its offset alone. This is the documented reference
/// treatment of a constant or zero-range column, and it is what keeps a
/// division by zero out of the transform.
fn derive_scale(minimum: f64, maximum: f64) -> f64 {
    let range = maximum - minimum;
    if range == 0.0 { 1.0 } else { 1.0 / range }
}

impl Estimator for MinMaxScaler {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for MinMaxScaler {
    /// Minima and maxima are order statistics: a per-sample weight cannot move
    /// them, so there is no weighted entry point to declare.
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_artifact(true);
}

impl HasParams for MinMaxScaler {
    type Params = MinMaxScalerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for MinMaxScaler {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_transform_request(self.n_features_in, data, output)?;

        if self.params.clip {
            transform_preflighted(data, output, |value, column| {
                ((f64::from(value) * self.scales[column] + self.offsets[column]) as f32)
                    .clamp(0.0, 1.0)
            })?;
        } else {
            transform_preflighted(data, output, |value, column| {
                (f64::from(value) * self.scales[column] + self.offsets[column]) as f32
            })?;
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

    fn matrix() -> DenseMatrix {
        DenseMatrix::new(vec![1.0, 2.0, 5.0, 3.0, 4.0, 5.0, 5.0, 6.0, 5.0], 3, 3).unwrap()
    }

    #[test]
    fn fits_extrema_and_maps_them_onto_the_unit_interval() {
        let scaler = MinMaxScaler::fit(&matrix().as_view(), MinMaxScalerParams::default()).unwrap();
        assert_eq!(scaler.data_min(), &[1.0, 2.0, 5.0]);
        assert_eq!(scaler.data_max(), &[5.0, 6.0, 5.0]);
        let transformed = scaler.transform(&matrix().as_view()).unwrap();
        assert_eq!(transformed.get(0, 0), Some(0.0));
        assert_eq!(transformed.get(1, 0), Some(0.5));
        assert_eq!(transformed.get(2, 0), Some(1.0));
    }

    #[test]
    fn a_zero_range_column_keeps_a_divisor_of_one_and_transforms_to_zero() {
        let scaler = MinMaxScaler::fit(&matrix().as_view(), MinMaxScalerParams::default()).unwrap();
        assert_eq!(scaler.scales()[2], 1.0);
        let transformed = scaler.transform(&matrix().as_view()).unwrap();
        for row in 0..3 {
            assert_eq!(transformed.get(row, 2), Some(0.0));
        }
    }

    #[test]
    fn a_single_row_batch_is_constant_in_every_column() {
        let single = DenseMatrix::new(vec![7.0, -3.0], 1, 2).unwrap();
        let scaler = MinMaxScaler::fit(&single.as_view(), MinMaxScalerParams::default()).unwrap();
        assert_eq!(scaler.scales(), &[1.0, 1.0]);
        assert_eq!(
            scaler.transform(&single.as_view()).unwrap().as_slice(),
            &[0.0, 0.0]
        );
    }

    #[test]
    fn refitting_the_same_batch_is_deterministic() {
        let params = MinMaxScalerParams::default().with_clip(true);
        let first = MinMaxScaler::fit(&matrix().as_view(), params).unwrap();
        let second = MinMaxScaler::fit(&matrix().as_view(), params).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.get_params(), &params);
    }

    #[test]
    fn clipping_bounds_values_outside_the_fitted_range() {
        let data = matrix();
        let open = MinMaxScaler::fit(&data.as_view(), MinMaxScalerParams::default()).unwrap();
        let clipped = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_clip(true),
        )
        .unwrap();
        let later = DenseMatrix::new(vec![-3.0, 10.0, 5.0], 1, 3).unwrap();
        assert_eq!(
            open.transform(&later.as_view()).unwrap().as_slice(),
            &[-1.0, 2.0, 0.0]
        );
        assert_eq!(
            clipped.transform(&later.as_view()).unwrap().as_slice(),
            &[0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn validates_width_and_workspace_before_writing() {
        let scaler = MinMaxScaler::fit(&matrix().as_view(), MinMaxScalerParams::default()).unwrap();
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
        let tiny = DenseMatrix::new(vec![0.0, f32::MIN_POSITIVE], 2, 1).unwrap();
        let scaler = MinMaxScaler::fit(&tiny.as_view(), MinMaxScalerParams::default()).unwrap();
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
        let scaler = MinMaxScaler::fit(
            &matrix().as_view(),
            MinMaxScalerParams::default().with_clip(true),
        )
        .unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(bytes, scaler.to_artifact([1; 32], [2; 32]).unwrap());
        assert_eq!(
            MinMaxScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap(),
            scaler
        );
        assert_eq!(
            MinMaxScaler::from_artifact(&bytes, [3; 32], [2; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
    }

    #[test]
    fn artifact_rejects_an_inverted_range_and_trailing_bytes() {
        let scaler = MinMaxScaler::fit(&matrix().as_view(), MinMaxScalerParams::default()).unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            MinMaxScaler::from_artifact(&bytes[..bytes.len() - 1], [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );

        let inverted = MinMaxScaler {
            n_features_in: 1,
            params: MinMaxScalerParams::default(),
            data_min: vec![5.0],
            data_max: vec![1.0],
            scales: vec![1.0],
            offsets: vec![-5.0],
        };
        let bytes = inverted.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            MinMaxScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }
}
