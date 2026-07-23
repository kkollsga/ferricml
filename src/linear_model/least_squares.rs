use crate::api::ModelError;
use crate::data::{MatrixView, SampleWeights};
use nalgebra::{DMatrix, DVector};

pub(super) struct LeastSquaresFit {
    pub(super) coefficients: Vec<f64>,
    pub(super) intercept: f64,
    pub(super) rank: usize,
}

pub(super) fn fit_dense(
    data: &MatrixView<'_>,
    targets: &[f32],
    sample_weights: Option<&SampleWeights>,
    fit_intercept: bool,
    tolerance: f32,
) -> Result<LeastSquaresFit, ModelError> {
    debug_assert_eq!(data.rows(), targets.len());
    debug_assert!(sample_weights.is_none_or(|weights| weights.len() == data.rows()));

    let rows = data.rows();
    let columns = data.columns();
    let total_weight = sample_weights.map_or(rows as f64, SampleWeights::total);
    let mut feature_means = vec![0.0_f64; columns];
    let mut target_mean = 0.0_f64;
    if fit_intercept {
        for (row_index, (row, &target)) in data.iter_rows().zip(targets).enumerate() {
            let weight = sample_weight(sample_weights, row_index);
            for (column, &value) in row.iter().enumerate() {
                feature_means[column] += weight * f64::from(value);
            }
            target_mean += weight * f64::from(target);
        }
        for mean in &mut feature_means {
            *mean /= total_weight;
        }
        target_mean /= total_weight;
    }

    let mut matrix_values = Vec::with_capacity(rows * columns);
    let mut target_values = Vec::with_capacity(rows);
    for (row_index, (row, &target)) in data.iter_rows().zip(targets).enumerate() {
        let weight_sqrt = sample_weight(sample_weights, row_index).sqrt();
        for (column, &value) in row.iter().enumerate() {
            matrix_values.push(weight_sqrt * (f64::from(value) - feature_means[column]));
        }
        target_values.push(weight_sqrt * (f64::from(target) - target_mean));
    }

    let matrix = DMatrix::from_row_slice(rows, columns, &matrix_values);
    let svd = matrix.svd(true, true);
    let largest_singular = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    if !largest_singular.is_finite() {
        return Err(ModelError::LinearSolveFailed);
    }
    let numerical_floor = f64::EPSILON * rows.max(columns) as f64 * largest_singular;
    let cutoff = (f64::from(tolerance) * largest_singular).max(numerical_floor);
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&value| value > cutoff)
        .count();
    let coefficients = if rank == 0 {
        vec![0.0; columns]
    } else {
        svd.solve(&DVector::from_vec(target_values), cutoff)
            .map_err(|_| ModelError::LinearSolveFailed)?
            .iter()
            .copied()
            .collect::<Vec<_>>()
    };
    let intercept = target_mean
        - feature_means
            .iter()
            .zip(&coefficients)
            .map(|(&mean, &coefficient)| mean * coefficient)
            .sum::<f64>();
    if coefficients.iter().any(|value| !value.is_finite()) || !intercept.is_finite() {
        return Err(ModelError::LinearSolveFailed);
    }
    Ok(LeastSquaresFit {
        coefficients,
        intercept,
        rank,
    })
}

fn sample_weight(sample_weights: Option<&SampleWeights>, row: usize) -> f64 {
    sample_weights.map_or(1.0, |weights| f64::from(weights.as_slice()[row]))
}
