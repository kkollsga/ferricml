use crate::data::MatrixView;
use crate::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use crate::linear_model::{LinearRegression, LinearRegressionParams, Ridge, RidgeParams};

use super::super::{Estimator, ModelError, Regressor};

/// Parameters retained by a fitted [`AnyRegressor`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyRegressorParams<'a> {
    /// Random-forest regressor parameters.
    RandomForest(&'a RandomForestRegressorParams),
    /// Ordinary least-squares regressor parameters.
    LinearRegression(&'a LinearRegressionParams),
    /// Ridge-regression parameters.
    Ridge(&'a RidgeParams),
    /// Histogram gradient-boosting parameters.
    HistGradientBoosting(&'a HistGradientBoostingRegressorParams),
}

/// An owned fitted regressor selected at runtime.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyRegressor {
    /// A fitted random-forest regressor.
    RandomForest(RandomForestRegressor),
    /// A fitted ordinary least-squares regressor.
    LinearRegression(LinearRegression),
    /// A fitted ridge regressor.
    Ridge(Ridge),
    /// A fitted histogram gradient-boosted regressor.
    HistGradientBoosting(HistGradientBoostingRegressor),
}

impl AnyRegressor {
    /// Returns the feature width required by this model.
    pub fn n_features_in(&self) -> usize {
        <Self as Estimator>::n_features_in(self)
    }

    /// Returns the concrete fitted parameters without erasing their type.
    pub fn get_params(&self) -> AnyRegressorParams<'_> {
        match self {
            Self::RandomForest(model) => AnyRegressorParams::RandomForest(model.get_params()),
            Self::LinearRegression(model) => {
                AnyRegressorParams::LinearRegression(model.get_params())
            }
            Self::Ridge(model) => AnyRegressorParams::Ridge(model.get_params()),
            Self::HistGradientBoosting(model) => {
                AnyRegressorParams::HistGradientBoosting(model.get_params())
            }
        }
    }

    /// Predicts one value per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predicts one value per row without allocating.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Regressor>::predict_into(self, data, output)
    }
}

impl From<RandomForestRegressor> for AnyRegressor {
    fn from(model: RandomForestRegressor) -> Self {
        Self::RandomForest(model)
    }
}

impl From<LinearRegression> for AnyRegressor {
    fn from(model: LinearRegression) -> Self {
        Self::LinearRegression(model)
    }
}

impl From<Ridge> for AnyRegressor {
    fn from(model: Ridge) -> Self {
        Self::Ridge(model)
    }
}

impl From<HistGradientBoostingRegressor> for AnyRegressor {
    fn from(model: HistGradientBoostingRegressor) -> Self {
        Self::HistGradientBoosting(model)
    }
}

impl Estimator for AnyRegressor {
    fn n_features_in(&self) -> usize {
        match self {
            Self::RandomForest(model) => model.n_features_in(),
            Self::LinearRegression(model) => model.n_features_in(),
            Self::Ridge(model) => model.n_features_in(),
            Self::HistGradientBoosting(model) => model.n_features_in(),
        }
    }
}

impl Regressor for AnyRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_into(data, output),
            Self::LinearRegression(model) => model.predict_into(data, output),
            Self::Ridge(model) => model.predict_into(data, output),
            Self::HistGradientBoosting(model) => model.predict_into(data, output),
        }
    }
}
