//! Deterministic dense standardization.

use crate::api::{Estimator, HasParams, ModelError, Transformer};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, MODEL_ARTIFACT_VERSION, STANDARD_SCALER_ARTIFACT_KIND,
    SchemaRole, artifact_version, decode_component, decode_v2_envelope, encode_component,
    encode_v2_envelope,
};
use crate::data::{MatrixView, SampleWeights};

const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const PAYLOAD_VERSION: u16 = 1;
const STATE_COMPONENT_KIND: u16 = 1;
const STATE_COMPONENT_VERSION: u16 = 1;

/// Parameters for [`StandardScaler`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StandardScalerParams {
    with_mean: bool,
    with_std: bool,
}

impl Default for StandardScalerParams {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }
}

impl StandardScalerParams {
    /// Enables or disables centering during transformation.
    #[must_use]
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Enables or disables population-standard-deviation scaling.
    #[must_use]
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    /// Returns whether transformed values are centered.
    pub const fn mean_enabled(&self) -> bool {
        self.with_mean
    }

    /// Returns whether transformed values are divided by fitted scales.
    pub const fn std_enabled(&self) -> bool {
        self.with_std
    }
}

/// Fitted per-feature population standardization state.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardScaler {
    n_features_in: usize,
    params: StandardScalerParams,
    means: Vec<f64>,
    variances: Vec<f64>,
    scales: Vec<f64>,
}

impl StandardScaler {
    /// Fits unweighted population statistics in fixed row order.
    pub fn fit(data: &MatrixView<'_>, params: StandardScalerParams) -> Result<Self, ModelError> {
        Self::fit_internal(data, None, params)
    }

    /// Fits weighted population statistics in fixed row order.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        sample_weights: &SampleWeights,
        params: StandardScalerParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        sample_weights: Option<&SampleWeights>,
        params: StandardScalerParams,
    ) -> Result<Self, ModelError> {
        if let Some(weights) = sample_weights
            && weights.len() != data.rows()
        {
            return Err(ModelError::SampleWeightLength {
                rows: data.rows(),
                weights: weights.len(),
            });
        }

        let columns = data.columns();
        let total_weight = sample_weights
            .map(SampleWeights::total)
            .unwrap_or(data.rows() as f64);
        let mut means = vec![0.0_f64; columns];
        for (row_index, row) in data.iter_rows().enumerate() {
            let weight = sample_weights
                .map(|weights| f64::from(weights.as_slice()[row_index]))
                .unwrap_or(1.0);
            for (mean, &value) in means.iter_mut().zip(row) {
                *mean += weight * f64::from(value);
            }
        }
        for mean in &mut means {
            *mean /= total_weight;
        }

        let mut variances = vec![0.0_f64; columns];
        for (row_index, row) in data.iter_rows().enumerate() {
            let weight = sample_weights
                .map(|weights| f64::from(weights.as_slice()[row_index]))
                .unwrap_or(1.0);
            for ((variance, &mean), &value) in variances.iter_mut().zip(&means).zip(row) {
                let difference = f64::from(value) - mean;
                *variance += weight * difference * difference;
            }
        }
        for variance in &mut variances {
            *variance /= total_weight;
        }
        let scales = variances
            .iter()
            .map(|&variance| {
                if variance == 0.0 {
                    1.0
                } else {
                    variance.sqrt()
                }
            })
            .collect();

        Ok(Self {
            n_features_in: columns,
            params,
            means,
            variances,
            scales,
        })
    }

    /// Returns fitted feature means as `f64` values.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Returns fitted population variances as `f64` values.
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }

    /// Returns fitted standard-deviation divisors; constant columns use one.
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
    pub const fn get_params(&self) -> &StandardScalerParams {
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
        if self.n_features_in > MAX_ARTIFACT_FEATURES {
            return Err(ArtifactError::InvalidPayload);
        }
        let count = u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut state = ArtifactPayloadWriter::with_capacity(16 + self.n_features_in * 24);
        state.u32(count);
        state.u32(u32::from(self.params.with_mean));
        state.u32(u32::from(self.params.with_std));
        state.u32(count);
        for ((&mean, &variance), &scale) in self.means.iter().zip(&self.variances).zip(&self.scales)
        {
            state.f64(mean);
            state.f64(variance);
            state.f64(scale);
        }
        let component = encode_component(
            STATE_COMPONENT_KIND,
            STATE_COMPONENT_VERSION,
            &state.finish(),
        )?;
        encode_v2_envelope(
            STANDARD_SCALER_ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[
                (SchemaRole::Input, input_schema),
                (SchemaRole::Transformed, transformed_schema),
            ],
            &component,
        )
    }

    /// Decodes fitted scaling state after checking both schemas.
    pub fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let version = artifact_version(bytes)?;
        if version != MODEL_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedVersion { found: version });
        }
        let mut envelope = decode_v2_envelope(
            bytes,
            STANDARD_SCALER_ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[
                (SchemaRole::Input, input_schema),
                (SchemaRole::Transformed, transformed_schema),
            ],
        )?;
        let mut state =
            decode_component(&mut envelope, STATE_COMPONENT_KIND, STATE_COMPONENT_VERSION)?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        let n_features_in = state.u32()? as usize;
        let with_mean = decode_bool(state.u32()?)?;
        let with_std = decode_bool(state.u32()?)?;
        let count = state.u32()? as usize;
        if n_features_in == 0 || n_features_in > MAX_ARTIFACT_FEATURES || count != n_features_in {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut means = Vec::with_capacity(count);
        let mut variances = Vec::with_capacity(count);
        let mut scales = Vec::with_capacity(count);
        for _ in 0..count {
            let mean = state.f64()?;
            let variance = state.f64()?;
            let scale = state.f64()?;
            if !mean.is_finite()
                || !variance.is_finite()
                || variance < 0.0
                || !scale.is_finite()
                || scale <= 0.0
                || (variance == 0.0 && scale != 1.0)
                || (variance > 0.0 && scale != variance.sqrt())
            {
                return Err(ArtifactError::InvalidPayload);
            }
            means.push(mean);
            variances.push(variance);
            scales.push(scale);
        }
        if !state.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in,
            params: StandardScalerParams {
                with_mean,
                with_std,
            },
            means,
            variances,
            scales,
        })
    }

    #[cfg(test)]
    fn transformed_value(&self, value: f32, column: usize) -> f32 {
        let mut transformed = f64::from(value);
        if self.params.with_mean {
            transformed -= self.means[column];
        }
        if self.params.with_std {
            transformed /= self.scales[column];
        }
        transformed as f32
    }

    fn transform_checked<F>(
        data: &MatrixView<'_>,
        output: &mut [f32],
        transform: F,
    ) -> Result<(), ModelError>
    where
        F: Fn(f32, usize) -> f32 + Copy,
    {
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
}

fn decode_bool(value: u32) -> Result<bool, ArtifactError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ArtifactError::InvalidPayload),
    }
}

impl Estimator for StandardScaler {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for StandardScaler {
    type Params = StandardScalerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for StandardScaler {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        if data.columns() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: data.columns(),
            });
        }
        let expected =
            data.rows()
                .checked_mul(self.n_features_in)
                .ok_or(ModelError::OutputShapeOverflow {
                    rows: data.rows(),
                    columns: self.n_features_in,
                })?;
        if output.len() != expected {
            return Err(ModelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }

        match (self.params.with_mean, self.params.with_std) {
            (false, false) => Self::transform_checked(data, output, |value, _| value)?,
            (true, false) => Self::transform_checked(data, output, |value, column| {
                (f64::from(value) - self.means[column]) as f32
            })?,
            (false, true) => Self::transform_checked(data, output, |value, column| {
                (f64::from(value) / self.scales[column]) as f32
            })?,
            (true, true) => Self::transform_checked(data, output, |value, column| {
                ((f64::from(value) - self.means[column]) / self.scales[column]) as f32
            })?,
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

    fn legacy_transform(scaler: &StandardScaler, data: &MatrixView<'_>) -> Vec<f32> {
        data.as_slice()
            .iter()
            .enumerate()
            .map(|(index, &value)| scaler.transformed_value(value, index % data.columns()))
            .collect()
    }

    #[test]
    fn fits_population_statistics_and_constant_scale() {
        let scaler =
            StandardScaler::fit(&matrix().as_view(), StandardScalerParams::default()).unwrap();
        assert_eq!(scaler.means(), &[3.0, 4.0, 5.0]);
        assert_eq!(scaler.variances(), &[8.0 / 3.0, 8.0 / 3.0, 0.0]);
        assert_eq!(scaler.scales()[2], 1.0);
        let transformed = scaler.transform(&matrix().as_view()).unwrap();
        assert_eq!(transformed.get(1, 0), Some(0.0));
        assert_eq!(transformed.get(0, 2), Some(0.0));
    }

    #[test]
    fn weighted_statistics_and_parameter_toggles_are_deterministic() {
        let weights = SampleWeights::new(vec![1.0, 2.0, 1.0]).unwrap();
        let params = StandardScalerParams::default()
            .with_mean(false)
            .with_std(true);
        let first =
            StandardScaler::fit_weighted(&matrix().as_view(), &weights, params.clone()).unwrap();
        let second = StandardScaler::fit_weighted(&matrix().as_view(), &weights, params).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.means(), &[3.0, 4.0, 5.0]);
        assert_eq!(first.variances(), &[2.0, 2.0, 0.0]);
        assert_eq!(
            first.transform(&matrix().as_view()).unwrap().get(0, 0),
            Some(1.0 / 2.0_f32.sqrt())
        );
    }

    #[test]
    fn every_transform_mode_matches_the_legacy_bit_pattern() {
        let data = matrix();
        for with_mean in [false, true] {
            for with_std in [false, true] {
                let scaler = StandardScaler::fit(
                    &data.as_view(),
                    StandardScalerParams::default()
                        .with_mean(with_mean)
                        .with_std(with_std),
                )
                .unwrap();
                let expected = legacy_transform(&scaler, &data.as_view());
                let actual = scaler.transform(&data.as_view()).unwrap();
                assert_eq!(
                    actual
                        .as_slice()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn validates_workspace_before_writing() {
        let scaler =
            StandardScaler::fit(&matrix().as_view(), StandardScalerParams::default()).unwrap();
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
    }

    #[test]
    fn near_constant_columns_are_scaled_but_overflow_is_reported_before_writes() {
        let near = DenseMatrix::new(vec![1.0, 1.0 + f32::EPSILON], 2, 1).unwrap();
        let scaler = StandardScaler::fit(&near.as_view(), StandardScalerParams::default()).unwrap();
        assert!(scaler.variances()[0] > 0.0);
        assert_ne!(scaler.scales()[0], 1.0);

        let extreme = DenseMatrix::new(vec![f32::MAX], 1, 1).unwrap();
        let mut output = [73.0];
        assert_eq!(
            scaler
                .transform_into(&extreme.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 0, column: 0 }
        );
        assert_eq!(output, [73.0]);
    }

    #[test]
    fn overflow_reports_the_first_row_major_location_without_partial_writes() {
        let fitted =
            DenseMatrix::new(vec![1.0, 1.0, 1.0 + f32::EPSILON, 1.0 + f32::EPSILON], 2, 2).unwrap();
        let scaler =
            StandardScaler::fit(&fitted.as_view(), StandardScalerParams::default()).unwrap();
        let extreme = DenseMatrix::new(vec![1.0, 1.0, 1.0, f32::MAX], 2, 2).unwrap();
        let mut output = [73.0; 4];
        assert_eq!(
            scaler
                .transform_into(&extreme.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 1, column: 1 }
        );
        assert_eq!(output, [73.0; 4]);
    }

    #[test]
    fn artifact_is_deterministic_and_schema_bound() {
        let scaler =
            StandardScaler::fit(&matrix().as_view(), StandardScalerParams::default()).unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(bytes, scaler.to_artifact([1; 32], [2; 32]).unwrap());
        assert_eq!(
            StandardScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap(),
            scaler
        );
        assert_eq!(
            StandardScaler::from_artifact(&bytes, [3; 32], [2; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
    }
}
