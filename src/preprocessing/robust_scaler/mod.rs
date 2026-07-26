//! Deterministic dense scaling by robust order statistics.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::artifact::{ArtifactError, ROBUST_SCALER_ARTIFACT_KIND, StageArtifact};
use crate::data::MatrixView;
use crate::numeric::{QuantileRule, quantile_sorted, sort_for_quantiles};

use super::scaling::{
    BASE_PAYLOAD_VERSION, ScalerHeader, ScalerParameters, decode_flag, decode_scaler_artifact,
    encode_scaler_artifact, inverse_transform_allocating, substituted_divisor,
    transform_preflighted, validate_inverse_request, validate_transform_request,
};

/// The quantile definition every fitted [`RobustScaler`] statistic is taken
/// under.
///
/// Fixed rather than configurable: a scaler whose centre and spread could be
/// read under different definitions would report two different models under one
/// name, and the fitted values are frozen against this one.
const RULE: QuantileRule = QuantileRule::Linear;

/// Parameters for [`RobustScaler`].
///
/// FerricML claims the quantile range and the two toggles. It deliberately does
/// **not** claim scaling the spread to the corresponding spread of a standard
/// normal distribution: that needs an inverse-normal-CDF primitive with its own
/// accuracy contract, and inventing one to serve a single optional flag would
/// be the wrong trade. It is an unclaimed parameter rather than a difference of
/// opinion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustScalerParams {
    quantile_low: f64,
    quantile_high: f64,
    with_centering: bool,
    with_scaling: bool,
}

impl Default for RobustScalerParams {
    fn default() -> Self {
        Self {
            quantile_low: 25.0,
            quantile_high: 75.0,
            with_centering: true,
            with_scaling: true,
        }
    }
}

impl RobustScalerParams {
    /// Sets the percentile pair whose difference becomes each column's spread.
    ///
    /// Both values are percentiles in `0.0..=100.0`, not fractions, and the
    /// default pair `(25.0, 75.0)` makes the spread the interquartile range.
    /// The pair is validated when a scaler is fitted, not here, so a parameter
    /// value is never rejected before the caller has finished describing it.
    #[must_use]
    pub const fn with_quantile_range(mut self, low: f64, high: f64) -> Self {
        self.quantile_low = low;
        self.quantile_high = high;
        self
    }

    /// Enables or disables subtracting each column's median.
    #[must_use]
    pub const fn with_centering(mut self, with_centering: bool) -> Self {
        self.with_centering = with_centering;
        self
    }

    /// Enables or disables dividing by each column's quantile spread.
    #[must_use]
    pub const fn with_scaling(mut self, with_scaling: bool) -> Self {
        self.with_scaling = with_scaling;
        self
    }

    /// Returns the percentile pair whose difference becomes the spread.
    #[must_use]
    pub const fn quantile_range(&self) -> (f64, f64) {
        (self.quantile_low, self.quantile_high)
    }

    /// Returns whether transformed values have their median removed.
    #[must_use]
    pub const fn centering_enabled(&self) -> bool {
        self.with_centering
    }

    /// Returns whether transformed values are divided by the fitted spread.
    #[must_use]
    pub const fn scaling_enabled(&self) -> bool {
        self.with_scaling
    }

    /// Rejects a percentile pair that does not describe a range.
    ///
    /// The accepted shape is `0 <= low <= high <= 100`. Equal percentiles are
    /// **accepted** and produce a spread of exactly zero, which the degeneracy
    /// rule then carries; that is a legitimate request for centering without
    /// scaling expressed through the range, not an error.
    fn validate(&self) -> Result<(), ModelError> {
        let (low, high) = self.quantile_range();
        if !(low.is_finite() && high.is_finite())
            || !(0.0..=100.0).contains(&low)
            || !(0.0..=100.0).contains(&high)
            || low > high
        {
            return Err(ModelError::InvalidQuantileRange);
        }
        Ok(())
    }
}

/// Fitted per-feature scaling by a median and a quantile spread.
///
/// Each column has its median removed and is divided by the difference between
/// two fitted percentiles — the interquartile range by default. Both statistics
/// are order statistics, so a handful of extreme rows move them far less than
/// they move a mean and a standard deviation, which is the entire reason to
/// prefer this scaler.
///
/// A column with no spread has nothing to divide by: it keeps a divisor of one
/// and is centred alone, so a constant feature stays finite instead of
/// producing a non-finite value. The test for that is exact equality with zero.
/// A column whose spread is merely *small* is real data and is scaled normally;
/// if that overflows `f32`, the batch is rejected with the offending location
/// before anything is written, rather than being silently left unscaled.
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::preprocessing::{
///     RobustScaler, RobustScalerParams, StandardScaler, StandardScalerParams,
/// };
///
/// // Eight ordinary values and one extreme outlier.
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1000.0];
/// let data = DenseMatrix::new(values, 9, 1)?;
///
/// let robust = RobustScaler::fit(&data.as_view(), RobustScalerParams::default())?;
/// let standard = StandardScaler::fit(&data.as_view(), StandardScalerParams::default())?;
///
/// // The outlier inflates the standard deviation, so every ordinary value is
/// // squashed toward zero. The interquartile range barely notices it.
/// let robust_scaled = robust.transform(&data.as_view())?;
/// let standard_scaled = standard.transform(&data.as_view())?;
/// assert!(robust_scaled.as_slice()[0].abs() > standard_scaled.as_slice()[0].abs());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Quantiles use linear interpolation between the two bracketing order
/// statistics, applied uniformly including at the median. Small samples do not
/// contain the value a percentile asks for, so the interpolation rule is a
/// documented semantic choice rather than an implementation detail.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustScaler {
    n_features_in: usize,
    params: RobustScalerParams,
    centers: Vec<f64>,
    spreads: Vec<f64>,
    scales: Vec<f64>,
}

impl RobustScaler {
    /// Fits per-feature medians and quantile spreads in fixed column order.
    ///
    /// Both statistics come from one sorted copy of each column, so a column is
    /// ordered once rather than once per statistic. Parameters are validated
    /// before that copy is allocated.
    pub fn fit(data: &MatrixView<'_>, params: RobustScalerParams) -> Result<Self, ModelError> {
        params.validate()?;

        let (low, high) = params.quantile_range();
        let columns = data.columns();
        let mut column = vec![0.0_f64; data.rows()];
        let mut centers = Vec::with_capacity(columns);
        let mut spreads = Vec::with_capacity(columns);
        let mut scales = Vec::with_capacity(columns);

        for index in 0..columns {
            for (slot, row) in column.iter_mut().zip(data.iter_rows()) {
                *slot = f64::from(row[index]);
            }
            sort_for_quantiles(&mut column);

            // The median is evaluated through the same general expression as
            // every other percentile; there is no midpoint special case.
            centers.push(quantile_sorted(&column, 50.0, RULE));
            let spread = quantile_sorted(&column, high, RULE) - quantile_sorted(&column, low, RULE);
            spreads.push(spread);
            scales.push(substituted_divisor(spread));
        }

        Ok(Self {
            n_features_in: columns,
            params,
            centers,
            spreads,
            scales,
        })
    }

    /// Returns the fitted per-feature medians as `f64` values.
    #[must_use]
    pub fn centers(&self) -> &[f64] {
        &self.centers
    }

    /// Returns each column's raw quantile spread, before any substitution.
    ///
    /// This is the difference of the two fitted percentiles exactly as
    /// measured, so a degenerate column reports the `0.0` it really has.
    /// [`RobustScaler::scales`] reports what the transform divides by.
    #[must_use]
    pub fn spreads(&self) -> &[f64] {
        &self.spreads
    }

    /// Returns the fitted divisors; a column with no spread uses one.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
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
    pub const fn get_params(&self) -> &RobustScalerParams {
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

    /// Undoes [`RobustScaler::transform`] into caller-owned storage.
    ///
    /// The inverse of `(x - center) / scale` is `x * scale + center`, applied
    /// only through the toggles that were enabled at fit time.
    ///
    /// # Exactness
    ///
    /// The round trip is **exact by construction** only when both statistics
    /// are disabled, and on a degenerate column whose divisor was substituted
    /// to one. Everywhere else it is exact only when the arithmetic happens to
    /// be: dividing by a spread and multiplying back is not an identity in
    /// floating point, and neither is subtracting a centre and adding it back.
    /// A caller who needs the original values keeps them.
    pub fn inverse_transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_inverse_request(self.n_features_in, data, output)?;
        match (self.params.with_centering, self.params.with_scaling) {
            (false, false) => transform_preflighted(data, output, |value, _| value),
            (true, false) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) + self.centers[column]) as f32
            }),
            (false, true) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) * self.scales[column]) as f32
            }),
            (true, true) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) * self.scales[column] + self.centers[column]) as f32
            }),
        }
    }

    /// Undoes [`RobustScaler::transform`], allocating the output matrix.
    pub fn inverse_transform(
        &self,
        data: &MatrixView<'_>,
    ) -> Result<crate::data::DenseMatrix, ModelError> {
        inverse_transform_allocating(self.n_features_in, data, |batch, output| {
            self.inverse_transform_into(batch, output).map(|_| ())
        })
    }
}

impl StageArtifact for RobustScaler {
    const ARTIFACT_KIND: u16 = ROBUST_SCALER_ARTIFACT_KIND;

    /// Encodes fitted scaling state with explicit input and transformed schemas.
    ///
    /// The raw spread is stored and the divisor is recomputed on decode, so a
    /// fitted model has exactly one valid byte string: a writer cannot choose
    /// to store a substituted divisor beside the spread it was substituted for.
    fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        let (low, high) = self.params.quantile_range();
        encode_scaler_artifact(
            Self::ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            self.n_features_in,
            ScalerParameters {
                version: BASE_PAYLOAD_VERSION,
                flags: &[
                    u32::from(self.params.with_centering),
                    u32::from(self.params.with_scaling),
                ],
                reals: &[low, high],
            },
            2,
            |feature, state| {
                state.f64(self.centers[feature]);
                state.f64(self.spreads[feature]);
            },
        )
    }

    /// Decodes fitted scaling state after checking both schemas.
    fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
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
            BASE_PAYLOAD_VERSION,
            2,
            2,
        )?;
        let params = RobustScalerParams {
            quantile_low: parameters[0],
            quantile_high: parameters[1],
            with_centering: decode_flag(flags[0])?,
            with_scaling: decode_flag(flags[1])?,
        };
        // A stored range that fitting would have refused describes a model that
        // could not have been produced, so it is rejected on the way back in
        // rather than trusted because it is already encoded.
        params
            .validate()
            .map_err(|_| ArtifactError::InvalidPayload)?;

        // Two `f64` fields per feature: the reservation is clamped to the
        // bytes actually present, never to the declared width alone.
        let capacity = state.bounded_capacity(n_features_in, 2 * 8);
        let mut centers = Vec::with_capacity(capacity);
        let mut spreads = Vec::with_capacity(capacity);
        let mut scales = Vec::with_capacity(capacity);
        for _ in 0..n_features_in {
            let center = state.f64()?;
            let spread = state.f64()?;
            // A negative spread is not reachable from any fit: the two
            // percentiles are ordered and quantiles are monotone in the
            // percentile, so the difference cannot be below zero.
            if !center.is_finite() || !spread.is_finite() || spread < 0.0 {
                return Err(ArtifactError::InvalidPayload);
            }
            centers.push(center);
            spreads.push(spread);
            scales.push(substituted_divisor(spread));
        }
        if !state.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in,
            params,
            centers,
            spreads,
            scales,
        })
    }
}

impl Estimator for RobustScaler {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for RobustScaler {
    /// A median and a quantile spread are order statistics: a per-sample weight
    /// cannot move them without a weighted quantile rule, and the linear rule
    /// this scaler is frozen against has no weighted form. There is therefore
    /// no weighted entry point to declare.
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_artifact(true);
}

impl HasParams for RobustScaler {
    type Params = RobustScalerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for RobustScaler {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_transform_request(self.n_features_in, data, output)?;

        // Every arm is affine in the value with a strictly positive multiplier,
        // so the map stays monotone per column and screening each column's
        // extrema is sufficient to prove the whole batch finite.
        match (self.params.with_centering, self.params.with_scaling) {
            (false, false) => transform_preflighted(data, output, |value, _| value),
            (true, false) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) - self.centers[column]) as f32
            }),
            (false, true) => transform_preflighted(data, output, |value, column| {
                (f64::from(value) / self.scales[column]) as f32
            }),
            (true, true) => transform_preflighted(data, output, |value, column| {
                ((f64::from(value) - self.centers[column]) / self.scales[column]) as f32
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
        DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
    }

    /// One column per green-room worked example, fitted together.
    fn worked_examples() -> DenseMatrix {
        matrix(
            &[
                0.0, 0.0, -3.0, //
                1.0, 1.0, 0.5, //
                1.0, 10.0, 2.0, //
                1.0, 10.0, 11.0,
            ],
            4,
            3,
        )
    }

    fn fitted(data: &DenseMatrix, params: RobustScalerParams) -> RobustScaler {
        RobustScaler::fit(&data.as_view(), params).unwrap()
    }

    #[test]
    fn fits_the_median_and_the_interquartile_range_per_column() {
        let data = matrix(&[0.0, 1.0, 2.0, 10.0], 4, 1);
        let scaler = fitted(&data, RobustScalerParams::default());
        assert_eq!(scaler.centers(), &[1.5]);
        assert_eq!(scaler.spreads(), &[3.25]);
        assert_eq!(scaler.scales(), &[3.25]);
        assert_eq!(
            scaler.transform(&data.as_view()).unwrap().as_slice(),
            &[-0.461_538_46, -0.153_846_16, 0.153_846_16, 2.615_384_6_f32]
        );
    }

    #[test]
    fn columns_are_fitted_independently_of_each_other() {
        let scaler = fitted(&worked_examples(), RobustScalerParams::default());
        assert_eq!(scaler.centers(), &[1.0, 5.5, 1.25]);
        assert_eq!(scaler.spreads(), &[0.25, 9.25, 4.625]);

        // Fitting one column alone reproduces its statistics exactly, and
        // scaling another column by a million leaves it untouched.
        let alone = fitted(&matrix(&[-3.0, 0.5, 2.0, 11.0], 4, 1), Default::default());
        assert_eq!(alone.centers(), &[scaler.centers()[2]]);
        assert_eq!(alone.spreads(), &[scaler.spreads()[2]]);
    }

    #[test]
    fn the_quantile_range_selects_which_spread_is_removed() {
        let data = matrix(&[0.0, 1.0, 2.0, 10.0], 4, 1);
        let full = fitted(
            &data,
            RobustScalerParams::default().with_quantile_range(0.0, 100.0),
        );
        assert_eq!(full.spreads(), &[10.0], "the full range is the extrema");

        let narrow = fitted(
            &data,
            RobustScalerParams::default().with_quantile_range(40.0, 60.0),
        );
        assert_eq!(narrow.centers(), &[1.5]);
        assert!(narrow.spreads()[0] < full.spreads()[0]);
    }

    #[test]
    fn an_empty_quantile_range_is_accepted_and_leaves_the_column_uncentred_only() {
        // Equal percentiles are a legitimate request: the spread is exactly
        // zero, so the degeneracy rule supplies a divisor of one and the
        // transform is a pure centring.
        let data = matrix(&[0.0, 1.0, 2.0, 10.0], 4, 1);
        let scaler = fitted(
            &data,
            RobustScalerParams::default().with_quantile_range(50.0, 50.0),
        );
        assert_eq!(scaler.spreads(), &[0.0]);
        assert_eq!(scaler.scales(), &[1.0]);
        assert_eq!(
            scaler.transform(&data.as_view()).unwrap().as_slice(),
            &[-1.5, -0.5, 0.5, 8.5]
        );
    }

    #[test]
    fn an_invalid_quantile_range_is_rejected_before_any_column_work() {
        let data = worked_examples();
        for (low, high) in [
            (75.0, 25.0),
            (-1.0, 50.0),
            (50.0, 101.0),
            (f64::NAN, 50.0),
            (0.0, f64::INFINITY),
        ] {
            let params = RobustScalerParams::default().with_quantile_range(low, high);
            assert_eq!(
                RobustScaler::fit(&data.as_view(), params).unwrap_err(),
                ModelError::InvalidQuantileRange,
                "range ({low}, {high})"
            );
        }
        // The permitted boundary is inclusive at both ends.
        assert!(
            RobustScaler::fit(
                &data.as_view(),
                RobustScalerParams::default().with_quantile_range(0.0, 100.0)
            )
            .is_ok()
        );
    }

    #[test]
    fn a_constant_column_keeps_a_divisor_of_one_and_transforms_to_zero() {
        let data = matrix(&[3.0, 3.0, 3.0, 3.0], 4, 1);
        let scaler = fitted(&data, RobustScalerParams::default());
        assert_eq!(scaler.centers(), &[3.0]);
        assert_eq!(scaler.spreads(), &[0.0]);
        assert_eq!(scaler.scales(), &[1.0]);
        assert_eq!(
            scaler.transform(&data.as_view()).unwrap().as_slice(),
            &[0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn a_zero_spread_column_that_is_not_constant_passes_its_tails_through_raw() {
        // The sharpest degenerate shape: the interquartile range is exactly
        // zero while the column plainly varies, so the tails survive as raw
        // deviations from the median. Stating it in a test is what keeps it
        // from being discovered as a surprise.
        let values = [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0];
        let data = matrix(&values, 9, 1);
        let scaler = fitted(&data, RobustScalerParams::default());
        assert_eq!(scaler.centers(), &[5.0]);
        assert_eq!(scaler.spreads(), &[0.0]);
        assert_eq!(scaler.scales(), &[1.0]);
        let transformed = scaler.transform(&data.as_view()).unwrap();
        assert_eq!(transformed.get(0, 0), Some(-5.0));
        assert_eq!(transformed.get(8, 0), Some(95.0));
    }

    #[test]
    fn a_single_row_fit_is_constant_in_every_column() {
        let data = matrix(&[7.0, -3.0], 1, 2);
        let scaler = fitted(&data, RobustScalerParams::default());
        assert_eq!(scaler.centers(), &[7.0, -3.0]);
        assert_eq!(scaler.scales(), &[1.0, 1.0]);
        assert_eq!(
            scaler.transform(&data.as_view()).unwrap().as_slice(),
            &[0.0, 0.0]
        );
    }

    #[test]
    fn a_repeated_single_value_is_degenerate_however_many_rows_it_has() {
        let data = matrix(&[7.5; 64], 64, 1);
        let scaler = fitted(&data, RobustScalerParams::default());
        assert_eq!(scaler.centers(), &[7.5]);
        assert_eq!(scaler.scales(), &[1.0]);
    }

    #[test]
    fn a_tiny_spread_is_scaled_rather_than_substituted() {
        // The divergence that matters: a legitimately tiny-scaled column is
        // real data. It is scaled normally, and only an *exactly* zero spread
        // is substituted.
        let step = f32::MIN_POSITIVE;
        let data = matrix(&[0.0, step, 2.0 * step, 3.0 * step], 4, 1);
        let scaler = fitted(&data, RobustScalerParams::default());
        assert!(scaler.spreads()[0] > 0.0, "the spread is small, not zero");
        assert_eq!(scaler.scales()[0], scaler.spreads()[0]);
        let transformed = scaler.transform(&data.as_view()).unwrap();
        assert!(
            transformed.as_slice().iter().all(|value| value.is_finite()),
            "a small spread still produces finite values here"
        );
    }

    #[test]
    fn an_overflowing_scale_is_reported_before_anything_is_written() {
        let tiny = matrix(&[0.0, f32::MIN_POSITIVE, f32::MIN_POSITIVE, 0.0], 4, 1);
        let scaler = fitted(&tiny, RobustScalerParams::default());
        let extreme = matrix(&[0.0, f32::MAX], 2, 1);
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
    fn both_toggles_select_which_statistic_is_removed() {
        let data = matrix(&[0.0, 1.0, 2.0, 10.0], 4, 1);
        let base = RobustScalerParams::default();
        let raw = [0.0_f32, 1.0, 2.0, 10.0];

        let centred = fitted(&data, base.with_scaling(false));
        assert_eq!(
            centred.transform(&data.as_view()).unwrap().as_slice(),
            &[-1.5, -0.5, 0.5, 8.5]
        );

        let scaled = fitted(&data, base.with_centering(false));
        let expected: Vec<f32> = raw.iter().map(|value| value / 3.25).collect();
        assert_eq!(
            scaled.transform(&data.as_view()).unwrap().as_slice(),
            expected.as_slice()
        );

        let neither = fitted(&data, base.with_centering(false).with_scaling(false));
        assert_eq!(
            neither.transform(&data.as_view()).unwrap().as_slice(),
            &raw,
            "with both statistics disabled the transform is the identity"
        );

        // The statistics are fitted either way; the toggles only choose what
        // the transform removes.
        assert_eq!(neither.centers(), centred.centers());
        assert_eq!(neither.scales(), scaled.scales());
    }

    #[test]
    fn refitting_the_same_batch_is_deterministic() {
        let data = worked_examples();
        let params = RobustScalerParams::default()
            .with_quantile_range(10.0, 90.0)
            .with_centering(false);
        let first = fitted(&data, params);
        let second = fitted(&data, params);
        assert_eq!(first, second);
        assert_eq!(first.get_params(), &params);
    }

    #[test]
    fn row_order_does_not_change_the_fitted_statistics() {
        let forward = fitted(&worked_examples(), RobustScalerParams::default());
        let reversed = matrix(
            &[
                1.0, 10.0, 11.0, //
                1.0, 10.0, 2.0, //
                1.0, 1.0, 0.5, //
                0.0, 0.0, -3.0,
            ],
            4,
            3,
        );
        let backward = fitted(&reversed, RobustScalerParams::default());
        assert_eq!(forward.centers(), backward.centers());
        assert_eq!(forward.spreads(), backward.spreads());
    }

    #[test]
    fn the_inverse_recovers_out_of_sample_rows_and_round_trips_them_exactly() {
        let scaler = fitted(&worked_examples(), RobustScalerParams::default());
        // Values never seen at fit time: the inverse is a map, not a lookup.
        let unseen = matrix(&[100.0, -50.0, 7.5, 0.25, 0.0, -1.0], 2, 3);
        let recovered = scaler.inverse_transform(&unseen.as_view()).unwrap();
        let back = scaler.transform(&recovered.as_view()).unwrap();
        assert_eq!(
            back.as_slice(),
            unseen.as_slice(),
            "transform after inverse_transform is exact on this probe"
        );
    }

    #[test]
    fn the_round_trip_is_exact_wherever_the_arithmetic_is() {
        let data = worked_examples();
        let base = RobustScalerParams::default();
        // With both statistics disabled the transform is the identity, so the
        // round trip is exact by construction rather than by luck. Note what is
        // *not* claimed: disabling only centring still divides and multiplies
        // back, which is not an identity in floating point, and this data
        // demonstrates it — see the bounded-envelope test.
        let params = base.with_centering(false).with_scaling(false);
        let scaler = fitted(&data, params);
        let transformed = scaler.transform(&data.as_view()).unwrap();
        let recovered = scaler.inverse_transform(&transformed.as_view()).unwrap();
        assert_eq!(recovered.as_slice(), data.as_slice());

        let constant = matrix(&[3.0, 3.0, 3.0, 3.0], 4, 1);
        let scaler = fitted(&constant, base);
        let transformed = scaler.transform(&constant.as_view()).unwrap();
        let recovered = scaler.inverse_transform(&transformed.as_view()).unwrap();
        assert_eq!(
            recovered.as_slice(),
            constant.as_slice(),
            "a degenerate column divides by the substituted one, so it round \
             trips exactly"
        );
    }

    #[test]
    fn dividing_and_multiplying_back_is_not_an_identity() {
        // The negative case for the exactness claim above, pinned so the
        // documentation cannot quietly overstate what the inverse guarantees.
        let data = worked_examples();
        let scaler = fitted(&data, RobustScalerParams::default().with_centering(false));
        let transformed = scaler.transform(&data.as_view()).unwrap();
        let recovered = scaler.inverse_transform(&transformed.as_view()).unwrap();
        assert_ne!(
            recovered.as_slice(),
            data.as_slice(),
            "scaling alone loses bits on this data, and the docs say so"
        );
    }

    #[test]
    fn a_general_round_trip_stays_within_a_bounded_envelope() {
        // With a general spread the round trip is not an identity, so the
        // honest assertion is a bound rather than equality.
        let data = worked_examples();
        let scaler = fitted(&data, RobustScalerParams::default());
        let transformed = scaler.transform(&data.as_view()).unwrap();
        let recovered = scaler.inverse_transform(&transformed.as_view()).unwrap();
        for (original, recovered) in data.as_slice().iter().zip(recovered.as_slice()) {
            let tolerance = 8.0 * f32::EPSILON * original.abs().max(1.0);
            assert!(
                (original - recovered).abs() <= tolerance,
                "{original} recovered as {recovered}"
            );
        }
    }

    #[test]
    fn the_inverse_validates_width_and_output_before_writing() {
        let scaler = fitted(&worked_examples(), RobustScalerParams::default());
        let mut short = [91.0; 11];
        assert_eq!(
            scaler
                .inverse_transform_into(&worked_examples().as_view(), &mut short)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 12,
                actual: 11
            }
        );
        assert_eq!(short, [91.0; 11]);
    }

    #[test]
    fn validates_width_and_workspace_before_writing() {
        let data = worked_examples();
        let scaler = fitted(&data, RobustScalerParams::default());

        let mut short = [91.0; 11];
        assert_eq!(
            scaler
                .transform_into(&data.as_view(), &mut short)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 12,
                actual: 11
            }
        );
        assert_eq!(short, [91.0; 11]);

        let narrow = matrix(&[1.0, 2.0], 1, 2);
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
    fn artifact_is_deterministic_and_schema_bound() {
        let scaler = fitted(
            &worked_examples(),
            RobustScalerParams::default()
                .with_quantile_range(10.0, 90.0)
                .with_centering(false),
        );
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(bytes, scaler.to_artifact([1; 32], [2; 32]).unwrap());
        assert_eq!(
            RobustScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap(),
            scaler
        );
        assert_eq!(
            RobustScaler::from_artifact(&bytes, [3; 32], [2; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        assert_eq!(
            RobustScaler::from_artifact(&bytes, [1; 32], [9; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
    }

    #[test]
    fn a_degenerate_column_round_trips_as_the_spread_it_really_had() {
        // The substituted divisor is never stored, so the decoded model
        // reports the same raw zero spread the fit measured.
        let data = matrix(&[3.0, 3.0, 3.0, 3.0], 4, 1);
        let scaler = fitted(&data, RobustScalerParams::default());
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        let decoded = RobustScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap();
        assert_eq!(decoded.spreads(), &[0.0]);
        assert_eq!(decoded.scales(), &[1.0]);
        assert_eq!(decoded, scaler);
    }

    #[test]
    fn artifact_rejects_a_truncation_and_a_range_no_fit_could_produce() {
        let scaler = fitted(&worked_examples(), RobustScalerParams::default());
        let bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            RobustScaler::from_artifact(&bytes[..bytes.len() - 1], [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );

        let inverted = RobustScaler {
            n_features_in: 1,
            params: RobustScalerParams {
                quantile_low: 75.0,
                quantile_high: 25.0,
                with_centering: true,
                with_scaling: true,
            },
            centers: vec![1.0],
            spreads: vec![2.0],
            scales: vec![2.0],
        };
        let bytes = inverted.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            RobustScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let negative = RobustScaler {
            n_features_in: 1,
            params: RobustScalerParams::default(),
            centers: vec![1.0],
            spreads: vec![-2.0],
            scales: vec![-2.0],
        };
        let bytes = negative.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            RobustScaler::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }

    #[test]
    fn the_caller_owned_path_matches_the_allocating_one() {
        let data = worked_examples();
        let scaler = fitted(&data, RobustScalerParams::default());
        let allocating = scaler.transform(&data.as_view()).unwrap();
        let mut into = vec![f32::MAX; allocating.as_slice().len()];
        let view = scaler.transform_into(&data.as_view(), &mut into).unwrap();
        assert_eq!(view.as_slice(), allocating.as_slice());
        assert_eq!(view.rows(), 4);
        assert_eq!(view.columns(), 3);
    }
}
