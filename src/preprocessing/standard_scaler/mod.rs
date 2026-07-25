//! Deterministic dense standardization.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::artifact::{ArtifactError, STANDARD_SCALER_ARTIFACT_KIND};
use crate::data::{MatrixView, SampleWeights};

use super::scaling::{
    ScalerHeader, ScalerParameters, decode_flag, decode_scaler_artifact, encode_scaler_artifact,
    substituted_divisor, transform_preflighted, validate_transform_request,
};

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
        // A zero-variance column keeps a divisor of one through the crate-wide
        // degeneracy rule; the square root is what is particular to
        // standardization. Substituting before the root rather than after is
        // the same value either way, since `1.0.sqrt()` is `1.0`.
        let scales = variances
            .iter()
            .map(|&variance| substituted_divisor(variance).sqrt())
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
        encode_scaler_artifact(
            STANDARD_SCALER_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            self.n_features_in,
            ScalerParameters {
                flags: &[
                    u32::from(self.params.with_mean),
                    u32::from(self.params.with_std),
                ],
                reals: &[],
            },
            3,
            |feature, state| {
                state.f64(self.means[feature]);
                state.f64(self.variances[feature]);
                state.f64(self.scales[feature]);
            },
        )
    }

    /// Decodes fitted scaling state after checking both schemas.
    pub fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let ScalerHeader {
            n_features_in,
            flags,
            mut state,
            ..
        } = decode_scaler_artifact(
            bytes,
            STANDARD_SCALER_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            2,
            0,
        )?;
        let with_mean = decode_flag(flags[0])?;
        let with_std = decode_flag(flags[1])?;
        // Three `f64` fields per feature: the reservation is clamped to the
        // bytes actually present, never to the declared width alone.
        let capacity = state.bounded_capacity(n_features_in, 3 * 8);
        let mut means = Vec::with_capacity(capacity);
        let mut variances = Vec::with_capacity(capacity);
        let mut scales = Vec::with_capacity(capacity);
        for _ in 0..n_features_in {
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
}

impl Estimator for StandardScaler {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for StandardScaler {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
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
        validate_transform_request(self.n_features_in, data, output)?;

        match (self.params.with_mean, self.params.with_std) {
            (false, false) => transform_preflighted(data, output, |value, _| value)?,
            (true, false) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) - self.means[column]) as f32
            })?,
            (false, true) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) / self.scales[column]) as f32
            })?,
            (true, true) => transform_preflighted(data, output, |value, column| {
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
    fn unsafe_extrema_fall_back_to_the_first_row_major_error() {
        let fitted =
            DenseMatrix::new(vec![1.0, 1.0, 1.0 + f32::EPSILON, 1.0 + f32::EPSILON], 2, 2).unwrap();
        let scaler =
            StandardScaler::fit(&fitted.as_view(), StandardScalerParams::default()).unwrap();
        let extreme = DenseMatrix::new(vec![1.0, f32::MAX, f32::MAX, 1.0], 2, 2).unwrap();
        let mut output = [41.0; 4];
        assert_eq!(
            scaler
                .transform_into(&extreme.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 0, column: 1 }
        );
        assert_eq!(output, [41.0; 4]);
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
