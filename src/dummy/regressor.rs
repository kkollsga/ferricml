use crate::api::{
    Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor, validate_scalar_row,
};
use crate::data::{MatrixView, RegressionTargets};

/// Parameters for [`DummyRegressor`].
///
/// The mean baseline has nothing to tune. This type exists so the baseline is
/// fitted exactly like every other FerricML estimator, and so a future strategy
/// choice can be added without changing the `fit` signature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DummyRegressorParams;

/// A regressor that ignores its features and predicts the training mean.
///
/// This is the quality floor a real regressor has to beat, and the value an
/// R² of zero corresponds to by definition.
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::dummy::{DummyRegressor, DummyRegressorParams};
/// use ferricml::metrics::r2_score;
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
/// let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;
///
/// let baseline = DummyRegressor::fit(
///     &data.as_view(),
///     &targets,
///     DummyRegressorParams::default(),
/// )?;
///
/// // The training mean, for every row.
/// let predictions = baseline.predict(&data.as_view())?;
/// assert_eq!(predictions, vec![2.5, 2.5, 2.5, 2.5]);
///
/// // Which is exactly what an R-squared of zero means.
/// assert!(r2_score(targets.as_slice(), &predictions)?.abs() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DummyRegressor {
    n_features_in: usize,
    params: DummyRegressorParams,
    mean: f32,
}

impl DummyRegressor {
    /// Fits the mean of the training targets.
    ///
    /// The mean accumulates in `f64` and is rejected if it does not fit a
    /// finite `f32`, so a fitted baseline never predicts a non-finite value.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: DummyRegressorParams,
    ) -> Result<Self, ModelError> {
        if data.rows() == 0 || data.columns() == 0 {
            return Err(ModelError::EmptyData);
        }
        if targets.len() != data.rows() {
            return Err(ModelError::TargetLength {
                rows: data.rows(),
                targets: targets.len(),
            });
        }

        let total: f64 = targets
            .as_slice()
            .iter()
            .map(|&value| f64::from(value))
            .sum();
        let mean = (total / targets.len() as f64) as f32;
        if !mean.is_finite() {
            return Err(ModelError::NumericalOverflow);
        }

        Ok(Self {
            n_features_in: data.columns(),
            params,
            mean,
        })
    }

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fitted parameters.
    pub const fn get_params(&self) -> &DummyRegressorParams {
        &self.params
    }

    /// Returns the fitted training mean.
    pub const fn mean(&self) -> f32 {
        self.mean
    }

    /// Predicts the training mean for one validated row.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        validate_scalar_row(row, self.n_features_in)?;
        Ok(self.mean)
    }

    /// Predicts one value per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predicts one value per row into caller-owned storage.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Regressor>::predict_into(self, data, output)
    }
}

impl Estimator for DummyRegressor {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for DummyRegressor {
    type Params = DummyRegressorParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

/// Declares nothing, and that is the complete and intended declaration.
///
/// The same sentence above `DummyClassifier` was a defect, because that
/// estimator does produce probabilities; the wording is kept here deliberately
/// rather than left looking like the correction was missed. `DummyRegressor`
/// predicts a scalar and has no probability to declare, its baseline is
/// refitted rather than persisted, and it has no weighted entry point — so
/// every capability in the vocabulary is genuinely absent.
impl HasCapabilities for DummyRegressor {
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

impl Regressor for DummyRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        if data.columns() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: data.columns(),
            });
        }
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        output.fill(self.mean);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn data() -> DenseMatrix {
        DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3, 2).unwrap()
    }

    #[test]
    fn predicts_the_training_mean_for_every_row() {
        let model = DummyRegressor::fit(
            &data().as_view(),
            &RegressionTargets::new(vec![1.0, 2.0, 6.0]).unwrap(),
            DummyRegressorParams,
        )
        .unwrap();

        assert_eq!(model.mean(), 3.0);
        assert_eq!(model.predict(&data().as_view()).unwrap(), vec![3.0; 3]);
        assert_eq!(model.predict_one(&[9.0, 9.0]).unwrap(), 3.0);
    }

    #[test]
    fn extreme_targets_average_without_overflowing() {
        let model = DummyRegressor::fit(
            &data().as_view(),
            &RegressionTargets::new(vec![f32::MAX; 3]).unwrap(),
            DummyRegressorParams,
        )
        .unwrap();

        assert_eq!(model.mean(), f32::MAX);
    }

    #[test]
    fn fitting_rejects_mismatched_targets_before_any_work() {
        assert_eq!(
            DummyRegressor::fit(
                &data().as_view(),
                &RegressionTargets::new(vec![1.0, 2.0]).unwrap(),
                DummyRegressorParams,
            )
            .unwrap_err(),
            ModelError::TargetLength {
                rows: 3,
                targets: 2
            }
        );
    }
}
