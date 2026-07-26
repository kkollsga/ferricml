//! Deterministic pool-adjacent-violators isotonic regression.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};

use super::Calibrator;

/// Parameters for [`IsotonicRegression`].
///
/// Pool-adjacent-violators has nothing to tune. This type exists so the
/// estimator is fitted exactly like every other FerricML estimator, and so a
/// later option — an out-of-range policy, or a decreasing direction — can be
/// added without changing the `fit` signature.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IsotonicRegressionParams;

/// A fitted non-decreasing step-and-interpolate map of one input.
///
/// This is the non-parametric calibrator: it assumes nothing about the shape of
/// the relationship except that it is monotone. It is also a useful monotone
/// regressor on its own, which is why it is a full [`Regressor`] over a
/// single-column matrix rather than a private helper.
///
/// # The fitted map
///
/// Fitting sorts the observations by input, pools every observation sharing one
/// input value, runs pool-adjacent-violators over the pooled means, and keeps
/// one `(threshold, value)` pair per distinct input. Prediction interpolates
/// linearly between consecutive pairs and **clamps** to the end values outside
/// the fitted range, so the map is total, finite, and non-decreasing everywhere.
///
/// # Non-decreasing is weaker than ranking-preserving
///
/// Pooling is what makes the map non-decreasing rather than strictly
/// increasing, and the difference is visible in any threshold-sweeping score.
/// Two distinct inputs inside one pooled block leave with the same value: the
/// pair is never *inverted*, but it is tied, and a tied pair no longer
/// contributes a full correct ordering to ROC AUC. In the extreme — a
/// calibration sample whose labels run opposite to its scores —
/// pool-adjacent-violators collapses everything into one block, the map is
/// constant, and ROC AUC becomes `0.5`. [`values`](Self::values) is how a
/// caller sees which happened; a single fitted value is that constant map.
///
/// # Tie convention, stated rather than inherited
///
/// Observations that share an input value are pooled into their weighted mean
/// *before* pool-adjacent-violators runs. That is forced rather than chosen: a
/// function of one input can only take one value at one input, so any other
/// handling would depend on the order the tied rows happened to arrive in. The
/// consequence is worth stating explicitly — **the fit does not depend on the
/// order of the observations at all**, only on the multiset of pairs. Equality
/// is IEEE equality, so `-0.0` and `0.0` are the same input.
///
/// ```
/// use ferricml::calibration::{IsotonicRegression, IsotonicRegressionParams};
/// use ferricml::data::{DenseMatrix, RegressionTargets};
///
/// // Three observations at x = 1 disagree; they pool to their mean.
/// let x = DenseMatrix::new(vec![0.0, 1.0, 1.0, 1.0, 2.0], 5, 1)?;
/// let y = RegressionTargets::new(vec![0.0, 1.0, 0.0, 0.0, 1.0])?;
/// let fitted = IsotonicRegression::fit(
///     &x.as_view(),
///     &y,
///     IsotonicRegressionParams::default(),
/// )?;
/// assert_eq!(fitted.thresholds(), &[0.0, 1.0, 2.0]);
/// assert_eq!(fitted.values(), &[0.0, 1.0 / 3.0, 1.0]);
/// // Outside the fitted range the end values are held, never extrapolated.
/// assert_eq!(fitted.map(-5.0), 0.0);
/// assert_eq!(fitted.map(5.0), 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct IsotonicRegression {
    /// Strictly increasing pooled inputs.
    thresholds: Vec<f32>,
    /// Non-decreasing fitted values, one per threshold.
    values: Vec<f32>,
    /// The parameters this fit was given.
    params: IsotonicRegressionParams,
}

impl IsotonicRegression {
    /// Fits a monotone map of one feature onto continuous targets.
    ///
    /// The matrix must have exactly one column: this is a univariate estimator,
    /// and a wider input is [`ModelError::FeatureDimension`] rather than a
    /// silently ignored remainder.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: IsotonicRegressionParams,
    ) -> Result<Self, ModelError> {
        validate_univariate(data, targets.len())?;
        let scores: Vec<f32> = data.iter_rows().map(|row| row[0]).collect();
        Ok(Self::fit_pairs(&scores, targets.as_slice(), params))
    }

    /// Fits a calibration map of raw model scores onto observed labels.
    ///
    /// The fitted values are means of `0`/`1` labels, so every calibrated
    /// output lies in `0.0..=1.0` by construction rather than by clamping.
    /// Both labels must be observed: a single-class calibration set determines
    /// no map and is [`ModelError::RequiresTwoClasses`].
    pub fn fit_calibration(
        scores: &[f32],
        targets: &BinaryTargets,
        params: IsotonicRegressionParams,
    ) -> Result<Self, ModelError> {
        super::validate_calibration_sample(scores, targets)?;
        Ok(Self::fit_pairs(scores, targets.as_slice(), params))
    }

    /// Pool, run pool-adjacent-violators, and keep one pair per distinct input.
    ///
    /// Inputs are already validated non-empty, equal in length, and finite by
    /// the public boundary that called this.
    fn fit_pairs<T: Copy + Into<f64>>(
        scores: &[f32],
        targets: &[T],
        params: IsotonicRegressionParams,
    ) -> Self {
        // A stable sort on the total order makes the visit order a function of
        // the input values alone. Equal inputs land adjacent, which is what
        // lets the pooling below be one linear pass.
        let mut order: Vec<u32> = (0..scores.len() as u32).collect();
        order.sort_by(|&left, &right| scores[left as usize].total_cmp(&scores[right as usize]));

        // Pool observations that share an input. This happens before pool
        // adjacent violators runs, which is what makes the fit independent of
        // the order tied observations arrived in.
        let mut inputs: Vec<Block> = Vec::new();
        for &index in &order {
            let score = scores[index as usize];
            let target: f64 = targets[index as usize].into();
            match inputs.last_mut() {
                Some(last) if last.score == score => {
                    last.count += 1.0;
                    last.total += target;
                }
                _ => inputs.push(Block {
                    score,
                    count: 1.0,
                    total: target,
                }),
            }
        }
        let thresholds: Vec<f32> = inputs.iter().map(|block| block.score).collect();

        // Pool adjacent violators: merge backwards while the previous block's
        // mean exceeds this one's. `span` records how many distinct inputs each
        // surviving block covers, so the fitted values can be expanded back to
        // one per threshold without a second search.
        let mut pooled: Vec<(Block, usize)> = Vec::with_capacity(inputs.len());
        for block in inputs {
            let mut merged = block;
            let mut span = 1;
            while let Some((previous, _)) = pooled.last() {
                if previous.mean() <= merged.mean() {
                    break;
                }
                let (previous, previous_span) = pooled.pop().expect("the block was just inspected");
                span += previous_span;
                merged = Block {
                    score: previous.score,
                    count: previous.count + merged.count,
                    total: previous.total + merged.total,
                };
            }
            pooled.push((merged, span));
        }

        let mut values = Vec::with_capacity(thresholds.len());
        for (block, span) in pooled {
            // Narrowing is monotone, so a non-decreasing sequence of `f64`
            // block means stays non-decreasing in `f32`.
            let value = block.mean() as f32;
            values.extend(std::iter::repeat_n(value, span));
        }
        debug_assert_eq!(values.len(), thresholds.len());
        Self {
            thresholds,
            values,
            params,
        }
    }

    /// Returns the feature width required by this model.
    ///
    /// Always one: this is a univariate estimator by construction.
    pub const fn n_features_in(&self) -> usize {
        1
    }

    /// Returns the exact fitted parameters.
    pub const fn get_params(&self) -> &IsotonicRegressionParams {
        &self.params
    }

    /// Returns the strictly increasing fitted input thresholds.
    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Returns the non-decreasing fitted values, one per threshold.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Maps one input onto its fitted value.
    ///
    /// This is total: an input below the fitted range returns the first fitted
    /// value and one above it returns the last, rather than extrapolating a
    /// trend the fit never observed or returning a non-finite value.
    pub fn map(&self, score: f32) -> f32 {
        let thresholds = &self.thresholds;
        let last = thresholds.len() - 1;
        // A NaN is impossible through the crate's own validated boundaries but
        // is representable in the `f32` a caller may hand this method directly,
        // so it lands on the first fitted value rather than falling through the
        // search with an unordered comparison.
        if score.is_nan() || score <= thresholds[0] {
            return self.values[0];
        }
        if score >= thresholds[last] {
            return self.values[last];
        }
        // `thresholds[0] < score < thresholds[last]`, so the partition point is
        // in `1..=last` and `index - 1` is in range.
        let index = thresholds.partition_point(|&threshold| threshold <= score);
        let (low, high) = (index - 1, index);
        let span = f64::from(thresholds[high]) - f64::from(thresholds[low]);
        let offset = f64::from(score) - f64::from(thresholds[low]);
        let rise = f64::from(self.values[high]) - f64::from(self.values[low]);
        let interpolated = f64::from(self.values[low]) + rise * (offset / span);
        (interpolated as f32).clamp(self.values[low], self.values[high])
    }

    /// Predicts the fitted value for one single-feature row.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        if row.len() != 1 {
            return Err(ModelError::FeatureDimension {
                expected: 1,
                actual: row.len(),
            });
        }
        if !row[0].is_finite() {
            return Err(ModelError::NonFiniteFeature { row: 0, column: 0 });
        }
        Ok(self.map(row[0]))
    }

    /// Predicts one value per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Writes one predicted value per row into a caller-owned buffer.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Regressor>::predict_into(self, data, output)
    }
}

/// One pooled group of observations sharing an input value.
#[derive(Clone, Copy, Debug)]
struct Block {
    score: f32,
    count: f64,
    total: f64,
}

impl Block {
    fn mean(&self) -> f64 {
        self.total / self.count
    }
}

fn validate_univariate(data: &MatrixView<'_>, targets: usize) -> Result<(), ModelError> {
    if data.rows() == 0 || data.columns() == 0 {
        return Err(ModelError::EmptyData);
    }
    if data.columns() != 1 {
        return Err(ModelError::FeatureDimension {
            expected: 1,
            actual: data.columns(),
        });
    }
    if data.rows() != targets {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets,
        });
    }
    Ok(())
}

impl Estimator for IsotonicRegression {
    fn n_features_in(&self) -> usize {
        self.n_features_in()
    }
}

impl HasParams for IsotonicRegression {
    type Params = IsotonicRegressionParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

/// Declares nothing, and every absence is genuine rather than unfinished.
///
/// Pool-adjacent-violators pools by *input value*, so a per-sample weight would
/// have to enter the pooled mean — but the fitted map is frozen against an
/// unweighted rule and there is no `SampleWeights` entry point for a caller to
/// reach, so `sample_weights` would promise an argument that does not exist.
/// A monotone map of one input onto a scalar has no class set, so `multiclass`
/// and `probability` have no meaning here; `fit_calibration` happens to produce
/// values in `0.0..=1.0`, but that is a property of averaging `0`/`1` labels
/// rather than a probability contract this type offers through
/// [`ProbabilisticClassifier`](crate::api::ProbabilisticClassifier).
/// `decision_function` records that a *classifier* exposes a raw score whose
/// squashing is its probability, which a regressor does not have. `artifact` is
/// the one absence that is a gap rather than a meaning: the fitted map is two
/// parallel `f32` vectors and would encode cleanly, but it owns no artifact
/// kind, so nothing may claim it persists.
impl HasCapabilities for IsotonicRegression {
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

impl Regressor for IsotonicRegression {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        if data.columns() != 1 {
            return Err(ModelError::FeatureDimension {
                expected: 1,
                actual: data.columns(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
            *slot = self.map(row[0]);
        }
        Ok(())
    }
}

impl Calibrator for IsotonicRegression {
    fn calibrate(&self, score: f32) -> f32 {
        self.map(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn fit(x: &[f32], y: &[f32]) -> IsotonicRegression {
        let data = DenseMatrix::new(x.to_vec(), x.len(), 1).unwrap();
        let targets = RegressionTargets::new(y.to_vec()).unwrap();
        IsotonicRegression::fit(&data.as_view(), &targets, IsotonicRegressionParams).unwrap()
    }

    #[test]
    fn an_already_monotone_sample_is_reproduced_exactly() {
        let fitted = fit(&[0.0, 1.0, 2.0, 3.0], &[0.0, 0.0, 1.0, 1.0]);
        assert_eq!(fitted.thresholds(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(fitted.values(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn violating_pairs_are_pooled_into_their_mean() {
        let fitted = fit(&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(fitted.values(), &[0.0, 0.5, 0.5, 1.0, 1.0]);
        let fitted = fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 1.0, 0.0, 0.0]);
        assert_eq!(fitted.values(), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn tied_inputs_pool_first_and_the_fit_ignores_observation_order() {
        let scores = [0.0_f32, 1.0, 1.0, 1.0, 2.0];
        let values = [0.0_f32, 1.0, 0.0, 0.0, 1.0];
        let sorted = fit(&scores, &values);
        assert_eq!(sorted.thresholds(), &[0.0, 1.0, 2.0]);
        assert_eq!(sorted.values(), &[0.0, 1.0 / 3.0, 1.0]);

        for permutation in [[2, 0, 4, 3, 1], [4, 3, 2, 1, 0], [1, 3, 0, 2, 4]] {
            let permuted_scores: Vec<f32> = permutation.iter().map(|&i| scores[i]).collect();
            let permuted_values: Vec<f32> = permutation.iter().map(|&i| values[i]).collect();
            let permuted = fit(&permuted_scores, &permuted_values);
            assert_eq!(permuted, sorted, "permutation {permutation:?}");
        }
    }

    #[test]
    fn signed_zeros_are_one_input() {
        let fitted = fit(&[-0.0, 0.0, 1.0], &[0.0, 1.0, 1.0]);
        assert_eq!(fitted.thresholds().len(), 2);
        assert_eq!(fitted.values(), &[0.5, 1.0]);
    }

    #[test]
    fn prediction_interpolates_between_thresholds_and_clamps_outside_them() {
        let fitted = fit(&[0.0, 1.0, 2.0, 3.0], &[0.0, 0.0, 1.0, 1.0]);
        for (input, expected) in [
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (1.25, 0.25),
            (1.5, 0.5),
            (1.75, 0.75),
            (2.0, 1.0),
            (2.5, 1.0),
            (3.0, 1.0),
            (4.0, 1.0),
        ] {
            assert_eq!(fitted.map(input), expected, "at {input}");
        }
    }

    #[test]
    fn a_single_distinct_input_maps_everything_to_its_mean() {
        let fitted = fit(&[1.0, 1.0, 1.0], &[0.0, 1.0, 1.0]);
        assert_eq!(fitted.thresholds(), &[1.0]);
        for input in [-1.0_f32, 1.0, 5.0] {
            assert_eq!(fitted.map(input), 2.0 / 3.0);
        }
    }

    #[test]
    fn the_fitted_map_is_non_decreasing_everywhere() {
        let scores: Vec<f32> = (0..64).map(|step| (step as f32) * 0.37 - 7.0).collect();
        let values: Vec<f32> = (0..64)
            .map(|step| ((step * 37) % 11) as f32 / 10.0)
            .collect();
        let fitted = fit(&scores, &values);
        assert!(
            fitted.values().windows(2).all(|pair| pair[0] <= pair[1]),
            "fitted values are not non-decreasing: {:?}",
            fitted.values()
        );
        assert!(
            fitted.thresholds().windows(2).all(|pair| pair[0] < pair[1]),
            "thresholds are not strictly increasing"
        );
        let mut previous = f32::NEG_INFINITY;
        for step in -300..=300 {
            let mapped = fitted.map(step as f32 * 0.05);
            assert!(mapped >= previous, "map decreased at {step}");
            previous = mapped;
        }
    }

    #[test]
    fn calibration_fitting_needs_both_labels_and_matching_lengths() {
        let scores = [0.1_f32, 0.4, 0.8];
        assert_eq!(
            IsotonicRegression::fit_calibration(
                &scores,
                &BinaryTargets::new(vec![0, 0, 0]).unwrap(),
                IsotonicRegressionParams,
            ),
            Err(ModelError::RequiresTwoClasses)
        );
        assert_eq!(
            IsotonicRegression::fit_calibration(
                &scores,
                &BinaryTargets::new(vec![0, 1]).unwrap(),
                IsotonicRegressionParams,
            ),
            Err(ModelError::TargetLength {
                rows: 3,
                targets: 2,
            })
        );
        assert_eq!(
            IsotonicRegression::fit_calibration(
                &[],
                &BinaryTargets::new(vec![0, 1]).unwrap(),
                IsotonicRegressionParams,
            ),
            Err(ModelError::EmptyData)
        );
        assert_eq!(
            IsotonicRegression::fit_calibration(
                &[0.1, f32::NAN, 0.8],
                &BinaryTargets::new(vec![0, 1, 1]).unwrap(),
                IsotonicRegressionParams,
            ),
            Err(ModelError::NonFiniteFeature { row: 1, column: 0 })
        );
    }

    #[test]
    fn calibration_values_stay_inside_the_probability_range() {
        let scores = [0.9_f32, 0.1, 0.5, 0.3, 0.7];
        let targets = BinaryTargets::new(vec![0, 1, 0, 1, 1]).unwrap();
        let fitted =
            IsotonicRegression::fit_calibration(&scores, &targets, IsotonicRegressionParams)
                .unwrap();
        assert!(
            fitted
                .values()
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        for step in -20..=20 {
            let mapped = fitted.map(step as f32 * 0.1);
            assert!((0.0..=1.0).contains(&mapped), "at {step}: {mapped}");
        }
    }

    #[test]
    fn the_regressor_path_validates_width_and_output_length_before_writing() {
        let fitted = fit(&[0.0, 1.0, 2.0], &[0.0, 0.5, 1.0]);
        let wide = DenseMatrix::new(vec![0.0; 6], 3, 2).unwrap();
        let mut sentinel = [f32::MAX; 3];
        assert_eq!(
            fitted.predict_into(&wide.as_view(), &mut sentinel),
            Err(ModelError::FeatureDimension {
                expected: 1,
                actual: 2,
            })
        );
        let narrow = DenseMatrix::new(vec![0.0, 1.0, 2.0], 3, 1).unwrap();
        assert_eq!(
            fitted.predict_into(&narrow.as_view(), &mut sentinel[..2]),
            Err(ModelError::OutputLength {
                expected: 3,
                actual: 2,
            })
        );
        assert_eq!(sentinel, [f32::MAX; 3]);
        assert_eq!(fitted.predict(&narrow.as_view()), Ok(vec![0.0, 0.5, 1.0]));
        assert_eq!(fitted.predict_one(&[1.0]), Ok(0.5));
        assert_eq!(
            fitted.predict_one(&[1.0, 2.0]),
            Err(ModelError::FeatureDimension {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn fitting_validates_shape_before_any_work() {
        let wide = DenseMatrix::new(vec![0.0; 6], 3, 2).unwrap();
        let targets = RegressionTargets::new(vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            IsotonicRegression::fit(&wide.as_view(), &targets, IsotonicRegressionParams),
            Err(ModelError::FeatureDimension {
                expected: 1,
                actual: 2,
            })
        );
        let narrow = DenseMatrix::new(vec![0.0, 1.0, 2.0], 3, 1).unwrap();
        assert_eq!(
            IsotonicRegression::fit(
                &narrow.as_view(),
                &RegressionTargets::new(vec![0.0, 1.0]).unwrap(),
                IsotonicRegressionParams,
            ),
            Err(ModelError::TargetLength {
                rows: 3,
                targets: 2,
            })
        );
    }

    /// The four conventions this estimator used to be the sole exception to.
    ///
    /// Each half is asserted through the surface a caller actually reaches: the
    /// params type exists and round-trips through both the inherent accessor
    /// and [`HasParams`], and the batch prediction pair is reachable without
    /// importing [`Regressor`] — which is what "inherent" means here and what a
    /// trait-only surface silently failed to provide.
    #[test]
    fn the_crate_wide_estimator_conventions_hold_here_too() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
        let targets = RegressionTargets::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let fitted =
            IsotonicRegression::fit(&data.as_view(), &targets, IsotonicRegressionParams).unwrap();

        assert_eq!(fitted.get_params(), &IsotonicRegressionParams);
        assert_eq!(
            <IsotonicRegression as HasParams>::get_params(&fitted),
            &IsotonicRegressionParams
        );
        assert_eq!(fitted.n_features_in(), 1);

        let allocated = fitted.predict(&data.as_view()).unwrap();
        let mut buffer = vec![f32::NAN; data.rows()];
        fitted.predict_into(&data.as_view(), &mut buffer).unwrap();
        assert_eq!(allocated, buffer);
        assert_eq!(allocated, vec![0.0, 0.0, 1.0, 1.0]);

        // The inherent forms are forwarders, not a second implementation.
        assert_eq!(
            allocated,
            <IsotonicRegression as Regressor>::predict(&fitted, &data.as_view()).unwrap()
        );
    }

    /// The calibration entry points take parameters for the same reason the
    /// regression one does: an option added later must not break either.
    #[test]
    fn every_fitting_entry_point_takes_the_params_type() {
        let scores = [0.1_f32, 0.4, 0.6, 0.9];
        let labels = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
        let calibrator =
            IsotonicRegression::fit_calibration(&scores, &labels, IsotonicRegressionParams)
                .unwrap();
        assert_eq!(calibrator.get_params(), &IsotonicRegressionParams);
    }

    #[test]
    fn refitting_the_same_sample_reproduces_the_same_model() {
        let scores: Vec<f32> = (0..40).map(|step| (step % 7) as f32 * 0.5).collect();
        let values: Vec<f32> = (0..40).map(|step| ((step * 13) % 5) as f32).collect();
        assert_eq!(fit(&scores, &values), fit(&scores, &values));
    }
}
