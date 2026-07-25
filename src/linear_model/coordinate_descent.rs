//! Cyclic coordinate descent for the elastic-net objective.
//!
//! # Why this solver and not the quasi-Newton seam
//!
//! An L1 penalty is not differentiable at zero, so a gradient method has no
//! gradient exactly where the interesting behaviour is — a coefficient landing
//! *on* zero and staying there. Coordinate descent sidesteps that entirely:
//! holding every other coefficient fixed leaves a one-variable penalized
//! quadratic whose minimizer has a closed form
//! ([`ElasticNetPenalty::coordinate_minimizer`]). Each step is exact, and the
//! zeros it produces are exact.
//!
//! # The objective, stated
//!
//! ```text
//! (1 / (2 * W)) * sum_i w_i * (y_i - b0 - x_i . b)^2
//!   + alpha * l1_ratio * ||b||_1
//!   + 0.5 * alpha * (1 - l1_ratio) * ||b||_2^2
//! ```
//!
//! with `W` the total sample weight, which is the row count when no weights are
//! supplied. This matches the reference contract's documented parametrization,
//! including the `1 / (2W)` on the data term — which is why an `alpha` here is
//! not the same quantity as [`Ridge`](super::Ridge)'s.
//!
//! # Standardization, frozen
//!
//! **The penalty applies to raw-scale coefficients.** Fitting *centers* the
//! design and the target by their weighted means when an intercept is
//! requested, and does not rescale the columns. Centering does not change a
//! slope, so the coefficients the penalty sees are the coefficients the caller
//! gets, and a feature's units decide how strongly it is penalized. That is a
//! deliberate, documented choice and not an oversight: rescaling internally
//! would make `alpha` mean something different from what the contract says,
//! and a caller who wants scale-free penalization composes a
//! [`StandardScaler`](crate::preprocessing::StandardScaler) in front, where the
//! transformation is visible and persists with the model.
//!
//! The intercept is never penalized. It is recovered from the centering after
//! the sweep, exactly as the ordinary and ridge fits recover theirs.
//!
//! # Convergence
//!
//! One sweep visits every coordinate in ascending column order. The fit has
//! converged when the largest absolute coefficient change across a whole sweep
//! is at most `tol`. That is FerricML's own criterion, stated rather than
//! inherited: the reference tests a duality gap scaled by the target norm,
//! which is a different quantity, so the two agree on the *minimizer* and not
//! on the iteration at which they stop. Exhausting `max_iter` is
//! [`ModelError::SolverDidNotConverge`], never a silently truncated fit.

use super::least_squares::preprocess;
use crate::api::ModelError;
use crate::data::{MatrixView, SampleWeights};
use crate::loss::ElasticNetPenalty;
use crate::numeric::sum_in_order;

/// One converged coordinate-descent fit.
#[derive(Debug)]
pub(super) struct ElasticNetFit {
    pub(super) coefficients: Vec<f64>,
    pub(super) intercept: f64,
    /// Sweeps performed, counting the one whose changes fell below `tol`.
    pub(super) sweeps: usize,
}

/// Fits the elastic-net objective by cyclic coordinate descent.
pub(super) fn fit_elastic_net_dense(
    data: &MatrixView<'_>,
    targets: &[f32],
    sample_weights: Option<&SampleWeights>,
    fit_intercept: bool,
    penalty: ElasticNetPenalty,
    max_iter: usize,
    tol: f32,
) -> Result<ElasticNetFit, ModelError> {
    debug_assert_eq!(data.rows(), targets.len());
    debug_assert!(sample_weights.is_none_or(|weights| weights.len() == data.rows()));

    let rows = data.rows();
    let columns = data.columns();
    let total_weight = sample_weights.map_or(rows as f64, SampleWeights::total);
    let preprocessed = preprocess(data, targets, sample_weights, fit_intercept);
    // Column-major, so one coordinate's column is contiguous.
    let design = preprocessed.matrix.as_slice();
    let mut residual = preprocessed.targets.iter().copied().collect::<Vec<_>>();

    // A coordinate's unpenalized curvature, in the objective's own scaling.
    // Constant once the design is built, so the sweep computes it never.
    let inverse_total_weight = if total_weight > 0.0 {
        1.0 / total_weight
    } else {
        return Err(ModelError::LinearSolveFailed);
    };
    let curvatures = (0..columns)
        .map(|column| {
            let values = &design[column * rows..(column + 1) * rows];
            inverse_total_weight * sum_in_order(values.iter().map(|&value| value * value))
        })
        .collect::<Vec<_>>();

    let mut coefficients = vec![0.0_f64; columns];
    let tolerance = f64::from(tol);
    let mut converged = false;
    let mut sweeps = 0;
    for sweep in 0..max_iter {
        let mut largest_change = 0.0_f64;
        for column in 0..columns {
            let values = &design[column * rows..(column + 1) * rows];
            let current = coefficients[column];
            // The coordinate's unpenalized optimum, read off the *partial*
            // residual — the residual with this coordinate's own contribution
            // added back. Forming it inside the reduction rather than as a
            // separate pass keeps one fixed-order sum per coordinate.
            let target = inverse_total_weight
                * sum_in_order(
                    values
                        .iter()
                        .zip(&residual)
                        .map(|(&value, &residual)| value * (residual + value * current)),
                );
            let updated = penalty.coordinate_minimizer(target, curvatures[column]);
            let change = updated - current;
            if change != 0.0 {
                for (slot, &value) in residual.iter_mut().zip(values) {
                    *slot -= change * value;
                }
                coefficients[column] = updated;
                largest_change = largest_change.max(change.abs());
            }
        }
        sweeps = sweep + 1;
        if largest_change <= tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(ModelError::SolverDidNotConverge { iterations: sweeps });
    }

    let intercept = preprocessed.target_mean
        - sum_in_order(
            preprocessed
                .feature_means
                .iter()
                .zip(&coefficients)
                .map(|(&mean, &coefficient)| mean * coefficient),
        );
    Ok(ElasticNetFit {
        coefficients,
        intercept,
        sweeps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn fit(
        values: &[f32],
        rows: usize,
        columns: usize,
        targets: &[f32],
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
    ) -> Result<ElasticNetFit, ModelError> {
        let data = DenseMatrix::new(values.to_vec(), rows, columns).expect("matrix");
        fit_elastic_net_dense(
            &data.as_view(),
            targets,
            None,
            fit_intercept,
            ElasticNetPenalty::new(alpha, l1_ratio),
            10_000,
            1.0e-12,
        )
    }

    /// Four columns, two of which carry the signal.
    fn sparse_problem() -> (Vec<f32>, Vec<f32>, usize, usize) {
        let rows = 20;
        let columns = 4;
        let mut state = 0x51d_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1_u32 << 31) as f32) * 2.0 - 1.0
        };
        let values = (0..rows * columns).map(|_| next()).collect::<Vec<f32>>();
        let targets = (0..rows)
            .map(|row| 3.0 * values[row * columns] - 1.5 * values[row * columns + 2] + 0.25)
            .collect::<Vec<f32>>();
        (values, targets, rows, columns)
    }

    #[test]
    fn a_zero_penalty_reproduces_the_least_squares_solution() {
        let (values, targets, rows, columns) = sparse_problem();
        let fitted = fit(&values, rows, columns, &targets, 0.0, 1.0, true).expect("fit");
        // The generating model is exactly representable, so an unpenalized fit
        // has to recover it.
        for (index, expected) in [3.0_f64, 0.0, -1.5, 0.0].iter().enumerate() {
            assert!(
                (fitted.coefficients[index] - expected).abs() <= 1.0e-4,
                "coefficient {index}: {:?}",
                fitted.coefficients
            );
        }
        assert!((fitted.intercept - 0.25).abs() <= 1.0e-4);
    }

    #[test]
    fn a_large_enough_penalty_removes_every_coefficient_exactly() {
        let (values, targets, rows, columns) = sparse_problem();
        let fitted = fit(&values, rows, columns, &targets, 100.0, 1.0, true).expect("fit");
        for &coefficient in &fitted.coefficients {
            assert_eq!(coefficient, 0.0);
            assert!(
                coefficient.is_sign_positive(),
                "a removed coefficient is +0"
            );
        }
        // With every slope removed the intercept is the target mean.
        let mean = f64::from(targets.iter().sum::<f32>()) / rows as f64;
        assert!((fitted.intercept - mean).abs() <= 1.0e-6);
    }

    #[test]
    fn increasing_the_penalty_never_increases_a_coefficient_magnitude_sum() {
        // The defining behaviour of an L1 path: shrinkage is monotone in
        // `alpha`, and the number of exact zeros never decreases.
        let (values, targets, rows, columns) = sparse_problem();
        let mut previous_magnitude = f64::INFINITY;
        let mut previous_zeros = 0;
        for step in 0..=20 {
            let alpha = f64::from(step) / 20.0;
            let fitted = fit(&values, rows, columns, &targets, alpha, 1.0, true).expect("fit");
            let magnitude = fitted
                .coefficients
                .iter()
                .map(|value| value.abs())
                .sum::<f64>();
            let zeros = fitted.coefficients.iter().filter(|v| **v == 0.0).count();
            assert!(
                magnitude <= previous_magnitude + 1.0e-9,
                "alpha {alpha}: {magnitude} exceeds {previous_magnitude}"
            );
            assert!(
                zeros >= previous_zeros,
                "alpha {alpha} recovered a coefficient"
            );
            previous_magnitude = magnitude;
            previous_zeros = zeros;
        }
        assert_eq!(previous_zeros, columns);
    }

    #[test]
    fn tightening_the_tolerance_only_ever_decreases_the_objective() {
        // Coordinate descent on a convex objective with a separable penalty is
        // a descent method, so stopping later can only land lower. Driven
        // through the tolerance rather than through `max_iter` because a
        // budget-limited run is a typed error here, not a partial answer —
        // which is the property the solver is built around.
        let (values, targets, rows, columns) = sparse_problem();
        let data = DenseMatrix::new(values.clone(), rows, columns).expect("matrix");
        let penalty = ElasticNetPenalty::new(0.05, 0.7);
        let mut previous = f64::INFINITY;
        let mut previous_sweeps = 0;
        let mut observed = 0;
        for exponent in 1..=10 {
            let tol = 10.0_f32.powi(-exponent);
            let fitted =
                fit_elastic_net_dense(&data.as_view(), &targets, None, true, penalty, 10_000, tol)
                    .expect("fit");
            let objective = objective_value(&values, &targets, rows, columns, &fitted, penalty);
            assert!(
                objective <= previous + 1.0e-15,
                "tol {tol:e}: {objective} exceeds {previous}"
            );
            assert!(
                fitted.sweeps >= previous_sweeps,
                "tol {tol:e} converged in fewer sweeps than a looser one"
            );
            previous = objective;
            previous_sweeps = fitted.sweeps;
            observed += 1;
        }
        assert!(observed == 10 && previous_sweeps > 1, "{previous_sweeps}");
    }

    fn objective_value(
        values: &[f32],
        targets: &[f32],
        rows: usize,
        columns: usize,
        fitted: &ElasticNetFit,
        penalty: ElasticNetPenalty,
    ) -> f64 {
        let squared = sum_in_order((0..rows).map(|row| {
            let prediction = fitted.intercept
                + (0..columns)
                    .map(|column| {
                        f64::from(values[row * columns + column]) * fitted.coefficients[column]
                    })
                    .sum::<f64>();
            let residual = f64::from(targets[row]) - prediction;
            residual * residual
        }));
        squared / (2.0 * rows as f64) + penalty.value(&fitted.coefficients)
    }

    #[test]
    fn a_pure_l2_penalty_agrees_with_ridge_at_the_documented_scaling() {
        // `ElasticNet(alpha, l1_ratio = 0)` and `Ridge(alpha * total_weight)`
        // are the same objective. Stating it as a test is what keeps the
        // `1 / (2W)` in the parametrization from quietly drifting.
        let (values, targets, rows, columns) = sparse_problem();
        let data = DenseMatrix::new(values.clone(), rows, columns).expect("matrix");
        let alpha = 0.3_f64;
        let elastic = fit(&values, rows, columns, &targets, alpha, 0.0, true).expect("fit");
        let ridge = super::super::least_squares::fit_ridge_dense(
            &data.as_view(),
            &targets,
            None,
            true,
            (alpha * rows as f64) as f32,
        )
        .expect("ridge fit");
        for (index, (left, right)) in elastic
            .coefficients
            .iter()
            .zip(&ridge.coefficients)
            .enumerate()
        {
            assert!(
                (left - right).abs() <= 1.0e-6,
                "coefficient {index}: elastic {left}, ridge {right}"
            );
        }
        assert!((elastic.intercept - ridge.intercept).abs() <= 1.0e-6);
    }

    #[test]
    fn an_exhausted_sweep_budget_is_reported_rather_than_returned() {
        let (values, targets, rows, columns) = sparse_problem();
        let data = DenseMatrix::new(values, rows, columns).expect("matrix");
        assert_eq!(
            fit_elastic_net_dense(
                &data.as_view(),
                &targets,
                None,
                true,
                ElasticNetPenalty::new(0.001, 1.0),
                1,
                1.0e-14,
            )
            .unwrap_err(),
            ModelError::SolverDidNotConverge { iterations: 1 }
        );
    }

    #[test]
    fn a_constant_column_is_removed_rather_than_dividing_by_zero() {
        // Centering zeroes a constant column, leaving a coordinate with no
        // curvature at all.
        let values = vec![
            1.0_f32, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0, 5.0, 5.0, 6.0, 5.0,
        ];
        let targets = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let fitted = fit(&values, 6, 2, &targets, 0.01, 1.0, true).expect("fit");
        assert!(fitted.coefficients.iter().all(|value| value.is_finite()));
        assert_eq!(fitted.coefficients[1], 0.0);
        assert!(fitted.coefficients[0] > 0.9);
    }

    #[test]
    fn without_an_intercept_nothing_is_centered() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets = [2.0_f32, 4.0, 6.0];
        let fitted = fit(&values, 3, 2, &targets, 0.0, 1.0, false).expect("fit");
        assert_eq!(fitted.intercept, 0.0);
        let prediction = 1.0 * fitted.coefficients[0] + 2.0 * fitted.coefficients[1];
        assert!(
            (prediction - 2.0).abs() <= 1.0e-6,
            "{:?}",
            fitted.coefficients
        );
    }

    #[test]
    fn refitting_the_same_inputs_reproduces_the_same_bits() {
        let (values, targets, rows, columns) = sparse_problem();
        let first = fit(&values, rows, columns, &targets, 0.02, 0.5, true).expect("fit");
        let second = fit(&values, rows, columns, &targets, 0.02, 0.5, true).expect("refit");
        assert_eq!(
            first
                .coefficients
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .coefficients
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(first.intercept.to_bits(), second.intercept.to_bits());
    }

    #[test]
    fn integer_sample_weights_match_replicating_the_rows() {
        // The weighted convention, stated as an equality rather than a
        // description: a weight of `k` is the row appearing `k` times.
        let values = [0.0_f32, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0];
        let targets = [1.0_f32, 2.0, 4.0, 7.0];
        let weights = [1.0_f32, 2.0, 1.0, 3.0];
        let data = DenseMatrix::new(values.to_vec(), 4, 2).expect("matrix");
        let weighted = fit_elastic_net_dense(
            &data.as_view(),
            &targets,
            Some(&SampleWeights::new(weights.to_vec()).expect("weights")),
            true,
            ElasticNetPenalty::new(0.05, 0.6),
            10_000,
            1.0e-12,
        )
        .expect("weighted fit");

        let mut replicated_values = Vec::new();
        let mut replicated_targets = Vec::new();
        for row in 0..4 {
            for _ in 0..weights[row] as usize {
                replicated_values.extend_from_slice(&values[row * 2..row * 2 + 2]);
                replicated_targets.push(targets[row]);
            }
        }
        let rows = replicated_targets.len();
        let replicated = DenseMatrix::new(replicated_values, rows, 2).expect("matrix");
        let unweighted = fit_elastic_net_dense(
            &replicated.as_view(),
            &replicated_targets,
            None,
            true,
            ElasticNetPenalty::new(0.05, 0.6),
            10_000,
            1.0e-12,
        )
        .expect("replicated fit");
        for (index, (left, right)) in weighted
            .coefficients
            .iter()
            .zip(&unweighted.coefficients)
            .enumerate()
        {
            assert!(
                (left - right).abs() <= 1.0e-9,
                "coefficient {index}: weighted {left}, replicated {right}"
            );
        }
        assert!((weighted.intercept - unweighted.intercept).abs() <= 1.0e-9);
    }

    #[test]
    fn scaling_every_sample_weight_leaves_the_fit_unchanged() {
        // Only relative weights matter, because the data term is divided by the
        // total weight. A fit that moved under a global rescale would mean
        // `alpha` silently depended on the units of the weights.
        let values = [0.0_f32, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0];
        let targets = [1.0_f32, 2.0, 4.0, 7.0];
        let data = DenseMatrix::new(values.to_vec(), 4, 2).expect("matrix");
        let run = |scale: f32| {
            fit_elastic_net_dense(
                &data.as_view(),
                &targets,
                Some(
                    &SampleWeights::new(vec![scale, 2.0 * scale, scale, 3.0 * scale])
                        .expect("weights"),
                ),
                true,
                ElasticNetPenalty::new(0.05, 0.6),
                10_000,
                1.0e-12,
            )
            .expect("fit")
        };
        let base = run(1.0);
        for scale in [0.25_f32, 4.0, 100.0] {
            let scaled = run(scale);
            for (index, (left, right)) in base
                .coefficients
                .iter()
                .zip(&scaled.coefficients)
                .enumerate()
            {
                assert!(
                    (left - right).abs() <= 1.0e-9,
                    "scale {scale} coefficient {index}: {left} vs {right}"
                );
            }
            assert!((base.intercept - scaled.intercept).abs() <= 1.0e-9);
        }
    }
}
