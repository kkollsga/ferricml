//! Linear estimators with stable fit and prediction semantics.

use crate::api::ModelError;
use crate::data::{MatrixView, SampleWeights};

mod coordinate_descent;
mod elastic_net;
mod lasso;
mod least_squares;
mod linear_regression;
mod logistic;
mod ridge;

pub use elastic_net::{ElasticNet, ElasticNetParams};
pub use lasso::{Lasso, LassoParams};
pub use linear_regression::{LinearRegression, LinearRegressionParams};
pub use logistic::{LogisticRegression, LogisticRegressionParams, LogisticSolver};
pub use ridge::{Ridge, RidgeParams};

/// Shared boundary validation for the penalized dense regressors.
///
/// Stated once for the family so the two estimators cannot disagree on the
/// order their errors are reported in — which is part of the frozen contract,
/// not an implementation detail. Every check happens before any allocation or
/// fitting work.
fn validate_penalized_fit(
    data: &MatrixView<'_>,
    target_len: usize,
    sample_weights: Option<&SampleWeights>,
    alpha: f32,
    l1_ratio: Option<f32>,
    max_iter: usize,
    tol: f32,
) -> Result<(), ModelError> {
    if data.rows() != target_len {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: target_len,
        });
    }
    if let Some(sample_weights) = sample_weights
        && data.rows() != sample_weights.len()
    {
        return Err(ModelError::SampleWeightLength {
            rows: data.rows(),
            weights: sample_weights.len(),
        });
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(ModelError::InvalidPenaltyAlpha);
    }
    if let Some(l1_ratio) = l1_ratio
        && (!l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio))
    {
        return Err(ModelError::InvalidL1Ratio);
    }
    if max_iter == 0 {
        return Err(ModelError::InvalidIterationCount);
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(ModelError::InvalidTolerance);
    }
    Ok(())
}

/// Narrows a `f64` dense fit to the storage type, refusing a non-finite result.
///
/// The check happens *after* narrowing, so a coefficient that is finite at
/// double precision but overflows `f32` is a failed fit rather than a stored
/// infinity that only surfaces as a non-finite prediction later.
fn narrow_dense_fit(coefficients: Vec<f64>, intercept: f64) -> Result<(Vec<f32>, f32), ModelError> {
    let narrowed = coefficients
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<f32>>();
    let intercept = intercept as f32;
    if narrowed.iter().any(|value| !value.is_finite()) || !intercept.is_finite() {
        return Err(ModelError::LinearSolveFailed);
    }
    Ok((narrowed, intercept))
}

/// One dense linear score.
///
/// The intercept seeds the accumulator and the feature terms follow in
/// ascending column order, which is the crate's fixed evaluation order for a
/// bounded inference reduction. See the accumulation policy in
/// [`crate::numeric`].
#[inline]
fn dense_prediction(row: &[f32], coefficients: &[f32], intercept: f32) -> f32 {
    row.iter()
        .zip(coefficients)
        .fold(intercept, |sum, (&value, &coefficient)| {
            sum + value * coefficient
        })
}

/// Validated, allocation-free dense batch prediction.
fn predict_dense_into(
    data: &MatrixView<'_>,
    output: &mut [f32],
    n_features_in: usize,
    coefficients: &[f32],
    intercept: f32,
) -> Result<(), ModelError> {
    if data.columns() != n_features_in {
        return Err(ModelError::FeatureDimension {
            expected: n_features_in,
            actual: data.columns(),
        });
    }
    if output.len() != data.rows() {
        return Err(ModelError::OutputLength {
            expected: data.rows(),
            actual: output.len(),
        });
    }
    for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
        *slot = crate::api::validate_prediction(
            dense_prediction(row, coefficients, intercept),
            row_index,
        )?;
    }
    Ok(())
}
