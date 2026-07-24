//! Mapping an objective into linear-coefficient space.
//!
//! A generalized linear model reaches its objective only through a raw score
//! `theta . x`. This module owns that mapping and the second-order system it
//! produces, so a linear solver holds the design matrix, the regularizer, and
//! the factorization while knowing nothing about which loss it is minimizing.

use super::objective::Objective;

/// Raw linear score for one design row.
///
/// The intercept coefficient seeds the accumulator and the feature terms follow
/// in ascending column order. That order is contractual, not incidental: the
/// fitted coefficients are the fixed point of an iteration over these scores,
/// so reordering the reduction would move the fitted artifact. See the
/// accumulation policy in [`crate::numeric`].
#[inline]
pub(crate) fn raw_score(
    theta: &[f64],
    design_row: &[f64],
    columns: usize,
    intercept_index: Option<usize>,
) -> f64 {
    let mut score = intercept_index.map_or(0.0, |index| theta[index]);
    for column in 0..columns {
        score += theta[column] * design_row[column];
    }
    score
}

/// Adds one weighted row to a Newton system in coefficient space.
///
/// The system is the objective's second-order expansion around the current
/// coefficients: `gradient` accumulates `w * loss'(raw) * x` and `hessian`
/// accumulates `w * curvature(raw) * x x'`. Only the lower triangle of the
/// hessian is written, because the caller factorizes it as a symmetric matrix.
///
/// `design_row` carries one entry per fitted parameter, including the constant
/// `1.0` of an intercept column, so its length defines the system's size.
pub(crate) fn accumulate_newton_row<O: Objective>(
    design_row: &[f64],
    raw: f64,
    target: f64,
    sample_weight: f64,
    gradient: &mut [f64],
    hessian: &mut [f64],
) {
    const {
        assert!(
            !O::IS_MULTICLASS,
            "a coefficient-space Newton system carries one raw score per row"
        );
        assert!(
            O::APPROX_HESSIAN == (O::CURVATURE_FLOOR > 0.0),
            "an approximate hessian is declared exactly when the curvature is floored"
        );
    }
    let parameters = design_row.len();
    let (slope, curvature) = O::gradient_and_curvature(raw, target);
    let residual = sample_weight * slope;
    let curvature = sample_weight * curvature;
    for left in 0..parameters {
        let left_value = design_row[left];
        gradient[left] += residual * left_value;
        let scaled_left = curvature * left_value;
        let hessian_row = &mut hessian[left * parameters..left * parameters + left + 1];
        for (slot, &right_value) in hessian_row.iter_mut().zip(design_row) {
            *slot += scaled_left * right_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::BinaryLogLoss;

    #[test]
    fn raw_score_seeds_the_accumulator_with_the_intercept() {
        let theta = [2.0, -3.0, 0.5];
        let row = [1.5, 4.0, 1.0];
        assert_eq!(
            raw_score(&theta, &row, 2, Some(2)),
            0.5 + 2.0 * 1.5 + -3.0 * 4.0
        );
        assert_eq!(
            raw_score(&theta, &row[..2], 2, None),
            2.0 * 1.5 + -3.0 * 4.0
        );
        assert_eq!(raw_score(&[], &[], 0, None), 0.0);
    }

    #[test]
    fn one_row_builds_the_lower_triangle_of_the_weighted_outer_product() {
        let row = [2.0, 1.0];
        let mut gradient = [0.0; 2];
        let mut hessian = [0.0; 4];
        accumulate_newton_row::<BinaryLogLoss>(&row, 0.0, 1.0, 3.0, &mut gradient, &mut hessian);

        // sigmoid(0) = 0.5, so the gradient term is 3 * (0.5 - 1) and the
        // curvature term is 3 * 0.25.
        assert_eq!(gradient, [-3.0, -1.5]);
        assert_eq!(hessian, [3.0, 0.0, 1.5, 0.75]);
    }

    #[test]
    fn rows_accumulate_and_a_zero_weight_row_changes_nothing() {
        let rows = [[1.0, 1.0], [-2.0, 1.0]];
        let mut gradient = [0.0; 2];
        let mut hessian = [0.0; 4];
        for row in &rows {
            accumulate_newton_row::<BinaryLogLoss>(
                row,
                0.25,
                0.0,
                1.0,
                &mut gradient,
                &mut hessian,
            );
        }
        let expected_gradient = gradient;
        let expected_hessian = hessian;

        accumulate_newton_row::<BinaryLogLoss>(
            &rows[0],
            0.25,
            0.0,
            0.0,
            &mut gradient,
            &mut hessian,
        );
        assert_eq!(gradient, expected_gradient);
        assert_eq!(hessian, expected_hessian);
    }
}
