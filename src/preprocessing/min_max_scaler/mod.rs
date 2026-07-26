//! Deterministic dense scaling onto the unit interval.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::artifact::{
    ArtifactError, MIN_MAX_SCALER_ARTIFACT_KIND, StageArtifact, artifact_payload_version,
};
use crate::data::MatrixView;

use super::scaling::{
    BASE_PAYLOAD_VERSION, ScalerHeader, ScalerParameters, decode_flag, decode_scaler_artifact,
    encode_scaler_artifact, inverse_transform_allocating, substituted_divisor,
    transform_preflighted, validate_inverse_request, validate_transform_request,
};

/// Parameters for [`MinMaxScaler`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxScalerParams {
    clip: bool,
    feature_min: f64,
    feature_max: f64,
}

/// The output range a scaler maps onto unless the caller chooses another.
const DEFAULT_FEATURE_RANGE: (f64, f64) = (0.0, 1.0);

/// The payload version carrying an explicit output range.
const FEATURE_RANGE_PAYLOAD_VERSION: u16 = 2;

impl Default for MinMaxScalerParams {
    fn default() -> Self {
        Self {
            clip: false,
            feature_min: DEFAULT_FEATURE_RANGE.0,
            feature_max: DEFAULT_FEATURE_RANGE.1,
        }
    }
}

impl MinMaxScalerParams {
    /// Sets the interval each column's fitted range is mapped onto.
    ///
    /// The smallest fitted value becomes `min` and the largest becomes `max`.
    /// `min` must be strictly below `max` and both must be finite; an empty or
    /// inverted range is rejected when a scaler is fitted, before any
    /// allocation. Equality is rejected rather than accepted as "map everything
    /// to one value", because that is a constant, not a scaling.
    #[must_use]
    pub const fn with_feature_range(mut self, min: f64, max: f64) -> Self {
        self.feature_min = min;
        self.feature_max = max;
        self
    }

    /// Returns the interval each column's fitted range is mapped onto.
    #[must_use]
    pub const fn feature_range(&self) -> (f64, f64) {
        (self.feature_min, self.feature_max)
    }

    /// Whether this is the range older artifacts could already express.
    fn range_is_default(&self) -> bool {
        self.feature_range() == DEFAULT_FEATURE_RANGE
    }

    /// Rejects an output interval that is not a range.
    fn validate(&self) -> Result<(), ModelError> {
        if !self.feature_min.is_finite()
            || !self.feature_max.is_finite()
            || self.feature_min >= self.feature_max
        {
            return Err(ModelError::InvalidFeatureRange);
        }
        Ok(())
    }

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
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::preprocessing::{MinMaxScaler, MinMaxScalerParams};
///
/// let data = DenseMatrix::new(vec![10.0, 20.0, 30.0, 50.0], 4, 1)?;
/// let scaler = MinMaxScaler::fit(&data.as_view(), MinMaxScalerParams::default())?;
///
/// let scaled = scaler.transform(&data.as_view())?;
/// assert_eq!(scaled.as_slice()[0], 0.0);
/// assert_eq!(scaled.as_slice()[3], 1.0);
///
/// // The map is invertible, so the original values come back.
/// let restored = scaler.inverse_transform(&scaled.as_view())?;
/// assert!((restored.as_slice()[2] - 30.0).abs() < 1e-4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Note that it fits order statistics — a minimum and a maximum — which no
/// per-sample weight can move. That is why it declares no weighted entry point
/// rather than offering one that would do nothing.
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
        params.validate()?;
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
            let scale = derive_scale(minimum, maximum, &params);
            let offset = params.feature_min - minimum * scale;
            if !scale.is_finite() || !offset.is_finite() {
                return Err(ModelError::NumericalOverflow);
            }
            offsets.push(offset);
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

    /// Undoes [`MinMaxScaler::transform`] into caller-owned storage.
    ///
    /// The inverse of `x * scale + offset` is `(x - offset) / scale`.
    ///
    /// # Exactness
    ///
    /// Clipping is **not** invertible: it is a projection, so a value that was
    /// clamped no longer records where it came from, and inverting a clipped
    /// batch recovers the bound rather than the original. With clipping off,
    /// the round trip is exact on a degenerate column whose divisor was
    /// substituted to one, and otherwise exact only when the arithmetic happens
    /// to be.
    pub fn inverse_transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_inverse_request(self.n_features_in, data, output)?;
        transform_preflighted(data, output, |value, column| {
            ((f64::from(value) - self.offsets[column]) / self.scales[column]) as f32
        })
    }

    /// Undoes [`MinMaxScaler::transform`], allocating the output matrix.
    pub fn inverse_transform(
        &self,
        data: &MatrixView<'_>,
    ) -> Result<crate::data::DenseMatrix, ModelError> {
        inverse_transform_allocating(self.n_features_in, data, |batch, output| {
            self.inverse_transform_into(batch, output).map(|_| ())
        })
    }
}

impl StageArtifact for MinMaxScaler {
    const ARTIFACT_KIND: u16 = MIN_MAX_SCALER_ARTIFACT_KIND;

    /// Encodes fitted scaling state with explicit input and transformed schemas.
    fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        // The output range is written only when it is one an older reader
        // could not have assumed. A default-configured scaler therefore emits
        // exactly the bytes it emitted before this parameter existed, so no
        // already-frozen artifact moves; and because the version is a function
        // of the parameters rather than a choice, each fitted model still has
        // exactly one valid encoding.
        let (feature_min, feature_max) = self.params.feature_range();
        let default_range = self.params.range_is_default();
        let range = [feature_min, feature_max];
        encode_scaler_artifact(
            Self::ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            self.n_features_in,
            ScalerParameters {
                version: if default_range {
                    BASE_PAYLOAD_VERSION
                } else {
                    FEATURE_RANGE_PAYLOAD_VERSION
                },
                flags: &[u32::from(self.params.clip)],
                reals: if default_range { &[] } else { &range },
            },
            2,
            |feature, state| {
                state.f64(self.data_min[feature]);
                state.f64(self.data_max[feature]);
            },
        )
    }

    /// Decodes fitted scaling state after checking both schemas.
    fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        // Version 1 predates the configurable output range and is still
        // written by every default-configured scaler, so it is read rather than
        // rejected; its range is the default one by definition.
        let version = artifact_payload_version(bytes)?;
        let reals = match version {
            BASE_PAYLOAD_VERSION => 0,
            FEATURE_RANGE_PAYLOAD_VERSION => 2,
            found => return Err(ArtifactError::UnsupportedPayloadVersion { found }),
        };
        let ScalerHeader {
            n_features_in,
            flags,
            parameters,
            mut state,
        } = decode_scaler_artifact(
            bytes,
            Self::ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            version,
            1,
            reals,
        )?;
        let clip = decode_flag(flags[0])?;
        let params = match parameters.as_slice() {
            [] => MinMaxScalerParams::default().with_clip(clip),
            [feature_min, feature_max] => MinMaxScalerParams::default()
                .with_clip(clip)
                .with_feature_range(*feature_min, *feature_max),
            _ => return Err(ArtifactError::InvalidPayload),
        };
        // A stored range that fitting would have refused, or a default range
        // written at the newer version, describes a model no writer produces.
        if params.validate().is_err() || (reals == 2 && params.range_is_default()) {
            return Err(ArtifactError::InvalidPayload);
        }
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
            let scale = derive_scale(minimum, maximum, &params);
            let offset = params.feature_min - minimum * scale;
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
            params,
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
/// carried to `0.0` by its offset alone. The degeneracy test itself lives in
/// [`substituted_divisor`], which every scaler here shares, so this function
/// only expresses what is particular to min-max scaling: the multiplier is the
/// reciprocal of the range.
fn derive_scale(minimum: f64, maximum: f64, params: &MinMaxScalerParams) -> f64 {
    let (feature_min, feature_max) = params.feature_range();
    (feature_max - feature_min) / substituted_divisor(maximum - minimum)
}

impl Estimator for MinMaxScaler {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for MinMaxScaler {
    /// The fitted range persists; weighted fitting is genuinely unavailable.
    ///
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
            let low = self.params.feature_min as f32;
            let high = self.params.feature_max as f32;
            transform_preflighted(data, output, |value, column| {
                ((f64::from(value) * self.scales[column] + self.offsets[column]) as f32)
                    .clamp(low, high)
            })
        } else {
            transform_preflighted(data, output, |value, column| {
                (f64::from(value) * self.scales[column] + self.offsets[column]) as f32
            })
        }
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
    fn the_inverse_round_trips_and_clipping_is_deliberately_not_invertible() {
        let data = matrix();
        let scaler = MinMaxScaler::fit(&data.as_view(), MinMaxScalerParams::default()).unwrap();
        let transformed = scaler.transform(&data.as_view()).unwrap();
        let recovered = scaler.inverse_transform(&transformed.as_view()).unwrap();
        for (original, recovered) in data.as_slice().iter().zip(recovered.as_slice()) {
            let tolerance = 8.0 * f32::EPSILON * original.abs().max(1.0);
            assert!(
                (original - recovered).abs() <= tolerance,
                "{original} recovered as {recovered}"
            );
        }

        // A zero-range column transforms to zero and inverts back to its single
        // fitted value exactly, because its divisor is the substituted one.
        assert_eq!(recovered.get(0, 2), Some(5.0));

        // Clipping is a projection: a clamped value no longer records where it
        // came from, so inverting recovers the bound rather than the original.
        let clipped = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_clip(true),
        )
        .unwrap();
        let beyond = DenseMatrix::new(vec![-3.0, 10.0, 5.0], 1, 3).unwrap();
        let forward = clipped.transform(&beyond.as_view()).unwrap();
        let back = clipped.inverse_transform(&forward.as_view()).unwrap();
        assert_eq!(
            back.as_slice(),
            &[1.0, 6.0, 5.0],
            "the fitted bounds come back, not the out-of-range inputs"
        );

        let mut short = [91.0; 8];
        assert_eq!(
            scaler
                .inverse_transform_into(&data.as_view(), &mut short)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 9,
                actual: 8
            }
        );
        assert_eq!(short, [91.0; 8]);
    }

    #[test]
    fn a_custom_feature_range_maps_the_fitted_extrema_onto_its_bounds() {
        let data = matrix();
        let scaler = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_feature_range(-1.0, 1.0),
        )
        .unwrap();
        let transformed = scaler.transform(&data.as_view()).unwrap();
        assert_eq!(
            transformed.get(0, 0),
            Some(-1.0),
            "the smallest fitted value"
        );
        assert_eq!(transformed.get(1, 0), Some(0.0), "the midpoint");
        assert_eq!(transformed.get(2, 0), Some(1.0), "the largest fitted value");
    }

    #[test]
    fn a_zero_range_column_lands_on_the_lower_bound_of_the_output_range() {
        // The degeneracy rule supplies a divisor of one, so the column keeps
        // its offset alone — which under a custom range is the lower bound
        // rather than zero.
        let data = matrix();
        let scaler = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_feature_range(-1.0, 1.0),
        )
        .unwrap();
        let transformed = scaler.transform(&data.as_view()).unwrap();
        for row in 0..3 {
            assert_eq!(transformed.get(row, 2), Some(-1.0));
        }
    }

    #[test]
    fn an_invalid_feature_range_is_rejected_before_any_allocation() {
        let data = matrix();
        for (low, high) in [
            (1.0, 1.0),
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, 0.0),
        ] {
            assert_eq!(
                MinMaxScaler::fit(
                    &data.as_view(),
                    MinMaxScalerParams::default().with_feature_range(low, high)
                )
                .unwrap_err(),
                ModelError::InvalidFeatureRange,
                "range ({low}, {high})"
            );
        }
    }

    #[test]
    fn clipping_bounds_values_into_the_configured_range() {
        let data = matrix();
        let clipped = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default()
                .with_feature_range(-1.0, 1.0)
                .with_clip(true),
        )
        .unwrap();
        let later = DenseMatrix::new(vec![-3.0, 10.0, 5.0], 1, 3).unwrap();
        assert_eq!(
            clipped.transform(&later.as_view()).unwrap().as_slice(),
            &[-1.0, 1.0, -1.0]
        );
    }

    #[test]
    fn a_default_range_still_writes_the_original_payload_version() {
        // The compatibility promise, as a test rather than a claim: adding the
        // output range moved no already-frozen artifact, because a
        // default-configured scaler writes exactly what it wrote before.
        let data = matrix();
        let scaler = MinMaxScaler::fit(&data.as_view(), MinMaxScalerParams::default()).unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            artifact_payload_version(&bytes).unwrap(),
            BASE_PAYLOAD_VERSION
        );

        let custom = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_feature_range(-1.0, 1.0),
        )
        .unwrap();
        let custom_bytes = custom.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            artifact_payload_version(&custom_bytes).unwrap(),
            FEATURE_RANGE_PAYLOAD_VERSION,
            "only a range an older reader could not assume raises the version"
        );
        assert!(
            custom_bytes.len() > bytes.len(),
            "the newer payload carries the two extra values"
        );
    }

    #[test]
    fn a_version_one_payload_decodes_to_an_identical_model() {
        // A byte string produced before the output range existed must decode to
        // exactly the model it described, with the default range supplied.
        let data = matrix();
        let scaler = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_clip(true),
        )
        .unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            artifact_payload_version(&bytes).unwrap(),
            BASE_PAYLOAD_VERSION,
            "this fixture must exercise the older layout"
        );

        let decoded = MinMaxScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap();
        assert_eq!(decoded, scaler);
        assert_eq!(decoded.get_params().feature_range(), (0.0, 1.0));
        assert_eq!(
            decoded.transform(&data.as_view()).unwrap().as_slice(),
            scaler.transform(&data.as_view()).unwrap().as_slice()
        );
    }

    #[test]
    fn a_custom_range_round_trips_and_a_degenerate_one_is_rejected() {
        let data = matrix();
        let scaler = MinMaxScaler::fit(
            &data.as_view(),
            MinMaxScalerParams::default().with_feature_range(-5.0, 2.5),
        )
        .unwrap();
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            MinMaxScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap(),
            scaler
        );

        // A default range written at the newer version is a byte string no
        // writer produces, so accepting it would give one model two encodings.
        let smuggled = MinMaxScaler {
            n_features_in: 1,
            params: MinMaxScalerParams::default(),
            data_min: vec![0.0],
            data_max: vec![1.0],
            scales: vec![1.0],
            offsets: vec![0.0],
        };
        let mut forged = smuggled.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            artifact_payload_version(&forged).unwrap(),
            BASE_PAYLOAD_VERSION
        );
        // Rewrite only the payload version field to the newer one.
        forged[12..14].copy_from_slice(&FEATURE_RANGE_PAYLOAD_VERSION.to_le_bytes());
        assert!(MinMaxScaler::from_artifact(&forged, [1; 32], [2; 32]).is_err());
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
