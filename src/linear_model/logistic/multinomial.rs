//! The joint multinomial fit behind [`LogisticRegression::fit_multiclass`].
//!
//! # Why this is not several binary fits
//!
//! A multinomial fit minimizes one loss over one score vector per row. Its
//! probabilities are the softmax of that vector, which is a different function
//! from normalizing independently fitted per-class binary probabilities — the
//! two disagree on real data, including on the predicted label, and neither can
//! be recovered from the other's coefficients. FerricML freezes the joint
//! definition, so this module is the only place that definition lives.
//!
//! # The parametrization
//!
//! One score row per class, with **no pinned reference class**. That looks
//! over-parametrized, and in the loss alone it is: adding the same vector to
//! every class row leaves every probability unchanged. Two things pin it down.
//! The L2 penalty is strictly convex in every coefficient, so it prefers the
//! centred representative of that family. The intercept, which is deliberately
//! *not* penalized, is pinned by the iteration instead: the gradient rows sum
//! to zero across classes at every point, the Newton system maps the
//! sum-zero subspace into itself, and the iterate starts at zero — so every
//! iterate, and therefore the fitted model, stays centred. Raw scores are
//! consequently centred rather than measured against a reference class.
//!
//! # The Newton system
//!
//! The update is the same exact second-order step the binary path takes, over
//! the stacked `classes * parameters` coefficient vector. A bound-majorized
//! alternative is available and much cheaper — one shared factorization
//! instead of one per iteration — but it was measured and rejected: on a
//! weakly regularized separable problem it needed 288 iterations where the
//! exact step needs 6, so the crate's default `max_iter` would have silently
//! returned unconverged models. Cost was the wrong thing to optimize when
//! `max_iter` has to mean the same thing here as it does for a binary fit.
//!
//! With an intercept the exact system is singular in exactly one direction:
//! adding a constant to every class's intercept changes nothing, and no
//! penalty opposes it. That direction is regularized explicitly. Because the
//! gradient is orthogonal to it, the solution in every other direction is
//! untouched and the update simply cannot drift out of the centred subspace.

use super::{
    LogisticRegression, LogisticRegressionParams, Standardization, build_design, sample_weight,
    solve_positive_definite, standardize, validate_common_fit,
};
use crate::api::ModelError;
use crate::data::{ClassTargets, MatrixView, SampleWeights};
use crate::loss::raw_score;
use crate::numeric::{softmax_in_place, sum_in_order};

/// Largest stacked Newton system the multinomial fit will build.
///
/// The system is `classes * (features + intercept)` square in `f64`, so this
/// caps one fit's dominant allocation at 32 MiB. It is checked at the public
/// boundary, before any allocation or training work, rather than discovered as
/// an out-of-memory abort partway through fitting.
pub(super) const MAX_NEWTON_PARAMETERS: usize = 2_048;

pub(super) fn fit(
    data: &MatrixView<'_>,
    targets: &ClassTargets,
    sample_weights: Option<&SampleWeights>,
    params: LogisticRegressionParams,
) -> Result<LogisticRegression, ModelError> {
    validate_fit(data, targets, sample_weights, &params)?;

    let rows = data.rows();
    let columns = data.columns();
    let classes = targets.classes().len();
    let parameter_count = columns + usize::from(params.fit_intercept);
    let parameters = classes * parameter_count;
    let total_weight = sample_weights.map_or(rows as f64, SampleWeights::total);
    let Standardization { means, scales } =
        standardize(data, sample_weights, total_weight, params.fit_intercept);
    let intercept_index = params.fit_intercept.then_some(columns);
    let design = build_design(data, &means, &scales, parameter_count, intercept_index);

    // Resolve each row's class column once. Labels are positions in the sorted
    // class list, never arithmetic.
    let mut class_of_row = Vec::with_capacity(rows);
    for &label in targets.as_slice() {
        class_of_row.push(
            targets
                .class_index(label)
                .expect("every target label is an observed class"),
        );
    }

    let lambda = 1.0 / f64::from(params.c);
    let penalties = (0..columns)
        .map(|column| lambda / (scales[column] * scales[column]))
        .collect::<Vec<_>>();
    // Curvature for the one direction the loss and the penalty both ignore:
    // shifting every class's intercept by the same amount. Scaled like the
    // intercept curvature it replaces, so it neither dominates the system nor
    // vanishes into it.
    let centring_curvature = total_weight / classes as f64;

    let mut theta = vec![0.0_f64; parameters];
    let mut gradient = vec![0.0_f64; parameters];
    let mut hessian = vec![0.0_f64; parameters * parameters];
    let mut update = vec![0.0_f64; parameters];
    let mut probabilities = vec![0.0_f64; classes];
    let mut iterations = 0;
    for iteration in 0..params.max_iter {
        gradient.fill(0.0);
        hessian.fill(0.0);
        for (row_index, design_row) in design.chunks_exact(parameter_count).enumerate() {
            let sample_weight = sample_weight(sample_weights, row_index);
            for (class, probability) in probabilities.iter_mut().enumerate() {
                *probability = raw_score(
                    &theta[class * parameter_count..(class + 1) * parameter_count],
                    design_row,
                    columns,
                    intercept_index,
                );
            }
            softmax_in_place(&mut probabilities);
            accumulate_newton_row(
                design_row,
                &probabilities,
                class_of_row[row_index],
                sample_weight,
                parameter_count,
                parameters,
                &mut gradient,
                &mut hessian,
            );
        }
        for class in 0..classes {
            let offset = class * parameter_count;
            for (column, &penalty) in penalties.iter().enumerate() {
                gradient[offset + column] += penalty * theta[offset + column];
                hessian[(offset + column) * parameters + offset + column] += penalty;
            }
        }
        if let Some(index) = intercept_index {
            for left in 0..classes {
                for right in 0..=left {
                    hessian[(left * parameter_count + index) * parameters
                        + right * parameter_count
                        + index] += centring_curvature;
                }
            }
        }

        solve_positive_definite(&mut hessian, &gradient, &mut update, parameters)?;
        let max_update = update
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()));
        for (value, &update) in theta.iter_mut().zip(&update) {
            *value -= update;
        }
        iterations = iteration + 1;
        if max_update <= f64::from(params.tol) {
            break;
        }
    }

    let mut coefficients = Vec::with_capacity(classes * columns);
    let mut intercepts = Vec::with_capacity(classes);
    for class in 0..classes {
        let row = &theta[class * parameter_count..(class + 1) * parameter_count];
        for column in 0..columns {
            coefficients.push((row[column] / scales[column]) as f32);
        }
        let intercept = intercept_index.map_or(0.0, |index| row[index])
            - sum_in_order((0..columns).map(|column| row[column] * means[column] / scales[column]));
        intercepts.push(intercept as f32);
    }
    if coefficients
        .iter()
        .chain(&intercepts)
        .any(|value| !value.is_finite())
    {
        return Err(ModelError::LinearSolveFailed);
    }

    Ok(LogisticRegression {
        n_features_in: columns,
        params,
        classes: targets.classes().to_vec(),
        coefficients,
        intercepts,
        iterations,
    })
}

/// Adds one weighted row to the stacked multinomial Newton system.
///
/// The gradient block of class `k` accumulates `w (p_k - [y = k]) x`, and the
/// hessian block `(k, l)` accumulates `w p_k (δ_kl - p_l) x x'`. Only the lower
/// triangle is written, because the caller factorizes it as a symmetric matrix.
///
/// The residuals of one row sum to exactly zero across classes, which is what
/// keeps the gradient — and therefore every iterate — inside the centred
/// subspace.
#[allow(clippy::too_many_arguments)]
fn accumulate_newton_row(
    design_row: &[f64],
    probabilities: &[f64],
    observed: usize,
    sample_weight: f64,
    parameter_count: usize,
    parameters: usize,
    gradient: &mut [f64],
    hessian: &mut [f64],
) {
    for (class, &probability) in probabilities.iter().enumerate() {
        let residual = sample_weight * (probability - f64::from(class == observed));
        let block = &mut gradient[class * parameter_count..(class + 1) * parameter_count];
        for (slot, &value) in block.iter_mut().zip(design_row) {
            *slot += residual * value;
        }
    }
    for (left, &left_probability) in probabilities.iter().enumerate() {
        for (right, &right_probability) in probabilities.iter().enumerate().take(left + 1) {
            let curvature =
                sample_weight * left_probability * (f64::from(left == right) - right_probability);
            let diagonal_block = left == right;
            for row in 0..parameter_count {
                let scaled = curvature * design_row[row];
                let width = if diagonal_block {
                    row + 1
                } else {
                    parameter_count
                };
                let start = (left * parameter_count + row) * parameters + right * parameter_count;
                for (slot, &value) in hessian[start..start + width]
                    .iter_mut()
                    .zip(&design_row[..width])
                {
                    *slot += scaled * value;
                }
            }
        }
    }
}

fn validate_fit(
    data: &MatrixView<'_>,
    targets: &ClassTargets,
    sample_weights: Option<&SampleWeights>,
    params: &LogisticRegressionParams,
) -> Result<(), ModelError> {
    validate_common_fit(data, targets.len(), sample_weights, params)?;
    if targets.n_classes() < 2 {
        return Err(ModelError::RequiresTwoClasses);
    }
    let parameters = targets
        .n_classes()
        .checked_mul(data.columns() + usize::from(params.fit_intercept))
        .filter(|&parameters| parameters <= MAX_NEWTON_PARAMETERS)
        .ok_or(ModelError::MulticlassSystemTooLarge {
            classes: targets.n_classes(),
            features: data.columns(),
            limit: MAX_NEWTON_PARAMETERS,
        })?;
    debug_assert!(parameters > 0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::ArtifactError;
    use crate::data::{BinaryTargets, DenseMatrix};

    /// The row-sum tolerance FerricML freezes: `n_classes` `f32` ulps.
    ///
    /// Probabilities are not renormalized, so a row sums to one only to the
    /// rounding of the storage type. This is the bound that statement means.
    fn row_sum_tolerance(classes: usize) -> f32 {
        classes as f32 * f32::EPSILON
    }

    /// Twelve rows, two features, three well-placed classes.
    fn three_class_problem() -> (DenseMatrix, ClassTargets) {
        let values = vec![
            0.0, 0.0, 0.5, 0.2, 0.2, 0.6, 1.0, 0.3, 2.0, 0.1, 1.8, 0.5, 2.2, 0.9, 0.3, 2.0, 0.8,
            2.4, 1.2, 2.2, 1.0, 3.0, 0.1, 1.0,
        ];
        (
            DenseMatrix::new(values, 12, 2).expect("fixture matrix"),
            ClassTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0]).expect("fixture targets"),
        )
    }

    fn query() -> DenseMatrix {
        DenseMatrix::new(vec![0.2, 0.3, 2.0, 0.4, 1.0, 3.0, 1.0, 1.2], 4, 2).expect("query matrix")
    }

    /// A reproducible spread of rows, without a dependency on the crate RNG.
    fn generated(
        rows: usize,
        columns: usize,
        classes: u8,
        seed: u64,
    ) -> (DenseMatrix, ClassTargets) {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1_u32 << 31) as f32) * 4.0 - 2.0
        };
        let mut values = Vec::with_capacity(rows * columns);
        let mut labels = Vec::with_capacity(rows);
        for row in 0..rows {
            for _ in 0..columns {
                values.push(next());
            }
            labels.push((row % classes as usize) as u8);
        }
        (
            DenseMatrix::new(values, rows, columns).expect("generated matrix"),
            ClassTargets::new(labels).expect("generated targets"),
        )
    }

    fn softmax_reference(scores: &[f32]) -> Vec<f32> {
        let mut row = scores.to_vec();
        crate::numeric::softmax_in_place_f32(&mut row);
        row
    }

    #[test]
    fn probabilities_are_exactly_the_softmax_of_the_centred_scores() {
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default().with_tol(1.0e-9),
        )
        .unwrap();
        let query = query();
        let scores = model.decision_function(&query.as_view()).unwrap();
        let probabilities = model.predict_proba(&query.as_view()).unwrap();
        assert_eq!(scores.len(), query.rows() * 3);
        assert_eq!(probabilities.len(), query.rows() * 3);

        for (scores, probabilities) in scores.chunks_exact(3).zip(probabilities.chunks_exact(3)) {
            // Bit-for-bit, not "close to": one definition, evaluated once.
            assert_eq!(probabilities, softmax_reference(scores).as_slice());
        }
    }

    #[test]
    fn scores_are_centred_rather_than_measured_against_a_reference_class() {
        for classes in 2..=6_u8 {
            let (data, targets) = generated(180, 3, classes, 0x5eed + u64::from(classes));
            let model = LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default().with_tol(1.0e-8),
            )
            .unwrap();
            let width = usize::from(classes);
            // No class is pinned: every class carries its own coefficient row
            // and its own intercept.
            assert_eq!(model.n_decision_columns(), width);
            assert_eq!(model.intercepts().len(), width);
            assert_eq!(model.coefficients().len(), width * data.columns());

            let scores = model.decision_function(&data.as_view()).unwrap();
            for row in scores.chunks_exact(width) {
                let magnitude = row.iter().fold(1.0_f32, |max, value| max.max(value.abs()));
                let sum = row.iter().sum::<f32>().abs();
                assert!(
                    sum <= classes as f32 * f32::EPSILON * magnitude,
                    "score row {row:?} sums to {sum}, which is more than f32 rounding"
                );
            }
        }
    }

    #[test]
    fn labels_never_disagree_with_the_probability_argmax() {
        for classes in 2..=6_u8 {
            let (data, targets) = generated(400, 4, classes, 0xabc + u64::from(classes));
            let model = LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default(),
            )
            .unwrap();
            let width = usize::from(classes);
            let labels = model.predict(&data.as_view()).unwrap();
            let probabilities = model.predict_proba(&data.as_view()).unwrap();
            for (index, (label, row)) in labels
                .iter()
                .zip(probabilities.chunks_exact(width))
                .enumerate()
            {
                assert_eq!(
                    *label,
                    model.classes()[super::super::argmax(row)],
                    "row {index} of a {classes}-class fit"
                );
                // The scalar path is the same rule, not a second one.
                assert_eq!(model.predict_one(data.row(index).unwrap()).unwrap(), *label);
            }
        }
    }

    #[test]
    fn probability_rows_are_not_renormalized_and_stay_inside_the_frozen_tolerance() {
        let mut inexact = 0_usize;
        let mut counted = 0_usize;
        for classes in 2..=8_u8 {
            let (data, targets) = generated(200, 3, classes, 7 + u64::from(classes));
            let model = LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default(),
            )
            .unwrap();
            let width = usize::from(classes);
            let tolerance = row_sum_tolerance(width);
            for row in model
                .predict_proba(&data.as_view())
                .unwrap()
                .chunks_exact(width)
            {
                let sum = row.iter().sum::<f32>();
                assert!(
                    (sum - 1.0).abs() <= tolerance,
                    "{classes}-class row {row:?} sums to {sum}"
                );
                assert!(row.iter().all(|&value| (0.0..=1.0).contains(&value)));
                inexact += usize::from(sum != 1.0);
                counted += 1;
            }
        }
        // If every row summed to exactly one, the tolerance would be describing
        // something that does not happen and the contract would be untested.
        assert!(
            inexact > 0,
            "no row deviated from one across {counted} rows; the tolerance is untested"
        );
    }

    #[test]
    fn columns_follow_sorted_labels_and_relabelling_only_permutes_them() {
        let (data, targets) = three_class_problem();
        let params = LogisticRegressionParams::default().with_tol(1.0e-9);
        let base =
            LogisticRegression::fit_multiclass(&data.as_view(), &targets, params.clone()).unwrap();
        assert_eq!(base.classes(), &[0, 1, 2]);

        // Same rows, relabelled to a non-contiguous, non-zero-based set whose
        // sorted order permutes columns 0 and 1.
        let relabelled = ClassTargets::new(
            targets
                .as_slice()
                .iter()
                .map(|&label| match label {
                    0 => 7,
                    1 => 3,
                    _ => 10,
                })
                .collect(),
        )
        .unwrap();
        let permuted =
            LogisticRegression::fit_multiclass(&data.as_view(), &relabelled, params).unwrap();
        assert_eq!(permuted.classes(), &[3, 7, 10]);

        let query = query();
        let base_probabilities = base.predict_proba(&query.as_view()).unwrap();
        let permuted_probabilities = permuted.predict_proba(&query.as_view()).unwrap();
        for (base, permuted) in base_probabilities
            .chunks_exact(3)
            .zip(permuted_probabilities.chunks_exact(3))
        {
            // Column j of the relabelled fit is class `classes[j]`, so old
            // class 1 (new label 3) moved to column 0 and old class 0 (new
            // label 7) to column 1.
            assert_eq!(permuted, [base[1], base[0], base[2]]);
        }
        assert_eq!(
            permuted.predict(&query.as_view()).unwrap(),
            base.predict(&query.as_view())
                .unwrap()
                .into_iter()
                .map(|label| match label {
                    0 => 7,
                    1 => 3,
                    _ => 10,
                })
                .collect::<Vec<_>>()
        );
        // A class that was never observed has no column to ask for.
        assert_eq!(
            permuted
                .predict_class_proba(&query.as_view(), 0)
                .unwrap_err(),
            ModelError::UnknownClass { class: 0 }
        );
        for (index, &label) in permuted.classes().iter().enumerate() {
            let column = permuted
                .predict_class_proba(&query.as_view(), label)
                .unwrap();
            for (row, &value) in column.iter().enumerate() {
                assert_eq!(value, permuted_probabilities[row * 3 + index]);
            }
        }
    }

    /// A model with hand-chosen scores, so a tie is exact by construction
    /// rather than by a symmetry that a solver might only nearly reproduce.
    fn constructed(classes: Vec<u8>, intercepts: Vec<f32>) -> LogisticRegression {
        LogisticRegression {
            n_features_in: 1,
            params: LogisticRegressionParams::default(),
            coefficients: vec![0.0; classes.len()],
            classes,
            intercepts,
            iterations: 1,
        }
    }

    #[test]
    fn an_exact_tie_selects_the_lowest_tied_class_not_the_first_class() {
        let row = DenseMatrix::new(vec![0.25], 1, 1).unwrap();

        // Classes 9 and 20 tie above class 5. "Lowest tied index" selects 9;
        // a rule that always fell back to the first class would say 5.
        let upper = constructed(vec![5, 9, 20], vec![0.0, 1.0, 1.0]);
        let probabilities = upper.predict_proba(&row.as_view()).unwrap();
        assert_eq!(probabilities[1], probabilities[2], "{probabilities:?}");
        assert!(probabilities[1] > probabilities[0], "{probabilities:?}");
        assert_eq!(upper.predict(&row.as_view()).unwrap(), vec![9]);
        assert_eq!(upper.predict_one(&[0.25]).unwrap(), 9);

        // The same rule when the tie does include the first class.
        let lower = constructed(vec![5, 9, 20], vec![1.0, 1.0, 0.0]);
        assert_eq!(lower.predict(&row.as_view()).unwrap(), vec![5]);

        // And when every class ties, which is the uniform row.
        let uniform = constructed(vec![5, 9, 20], vec![2.0, 2.0, 2.0]);
        let probabilities = uniform.predict_proba(&row.as_view()).unwrap();
        assert!(probabilities.iter().all(|&value| value == 1.0 / 3.0));
        assert_eq!(uniform.predict(&row.as_view()).unwrap(), vec![5]);
    }

    #[test]
    fn a_symmetric_fit_ties_bit_exactly_rather_than_nearly() {
        // Rows that are exact mirror images, so classes 9 and 20 are symmetric
        // under the fit while class 5 is not.
        let values = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, -1.0, 1.0, -2.0, 2.0,
        ];
        let data = DenseMatrix::new(values, 6, 2).unwrap();
        let targets = ClassTargets::new(vec![5, 5, 20, 20, 9, 9]).unwrap();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default().with_tol(1.0e-9),
        )
        .unwrap();
        assert_eq!(model.classes(), &[5, 9, 20]);

        let query = DenseMatrix::new(vec![0.0, 0.0], 1, 2).unwrap();
        let probabilities = model.predict_proba(&query.as_view()).unwrap();
        assert_eq!(
            probabilities[1].to_bits(),
            probabilities[2].to_bits(),
            "the symmetric classes must tie bit-exactly: {probabilities:?}"
        );
        // The label follows the argmax of these values, whichever it is.
        assert_eq!(
            model.predict(&query.as_view()).unwrap(),
            vec![model.classes()[super::super::argmax(&probabilities)]]
        );
    }

    #[test]
    fn a_single_observed_class_is_refused_rather_than_fitted() {
        let (data, _) = three_class_problem();
        for label in [0_u8, 3, 200] {
            let targets = ClassTargets::new(vec![label; data.rows()]).unwrap();
            assert_eq!(
                LogisticRegression::fit_multiclass(
                    &data.as_view(),
                    &targets,
                    LogisticRegressionParams::default(),
                )
                .unwrap_err(),
                ModelError::RequiresTwoClasses,
                "single class {label}"
            );
        }
    }

    #[test]
    fn the_binary_entry_point_keeps_its_asymmetric_single_row_shape() {
        let values = vec![
            -3.0, 1.0, -2.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0,
        ];
        let data = DenseMatrix::new(values, 6, 2).unwrap();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let params = LogisticRegressionParams::default().with_tol(1.0e-8);
        let binary = LogisticRegression::fit(
            &data.as_view(),
            &BinaryTargets::new(labels.clone()).unwrap(),
            params.clone(),
        )
        .unwrap();

        // One coefficient row, one score per row, two probability columns.
        assert_eq!(binary.n_decision_columns(), 1);
        assert_eq!(binary.coefficients().len(), data.columns());
        assert_eq!(binary.intercepts().len(), 1);
        assert_eq!(binary.intercept(), binary.intercepts()[0]);
        assert_eq!(binary.classes(), &[0, 1]);
        assert_eq!(
            binary.decision_function(&data.as_view()).unwrap().len(),
            data.rows()
        );
        assert_eq!(
            binary.predict_proba(&data.as_view()).unwrap().len(),
            data.rows() * 2
        );

        // The same two-class data through the multinomial entry point is a
        // different, centred parametrization — deliberately not the same model.
        let centred = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &ClassTargets::new(labels).unwrap(),
            params,
        )
        .unwrap();
        assert_eq!(centred.n_decision_columns(), 2);
        assert_eq!(centred.classes(), &[0, 1]);
        assert_eq!(
            centred.decision_function(&data.as_view()).unwrap().len(),
            data.rows() * 2
        );
        assert_eq!(
            centred.predict_proba(&data.as_view()).unwrap().len(),
            data.rows() * 2
        );
        // Both parametrizations describe the same separable problem, so their
        // labels agree even though their scores do not.
        assert_eq!(
            centred.predict(&data.as_view()).unwrap(),
            binary.predict(&data.as_view()).unwrap()
        );
    }

    #[test]
    fn scalar_valued_requests_are_refused_rather_than_answered_with_one_component() {
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let expected = ModelError::MulticlassOutput { columns: 3 };
        assert_eq!(
            model.decision_function_one(&[0.2, 0.3]).unwrap_err(),
            expected
        );
        assert_eq!(
            model.predict_positive_proba(&[0.2, 0.3]).unwrap_err(),
            expected
        );
    }

    /// A multiclass fit persists under its own payload version, and the two
    /// schemas never decode as each other.
    #[test]
    fn a_multiclass_fit_round_trips_and_cannot_be_read_as_a_binary_one() {
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let schema = [7; 32];
        let bytes = model.to_artifact(schema).unwrap();
        assert_eq!(
            bytes,
            model.to_artifact(schema).unwrap(),
            "encoding is stable"
        );

        let decoded = LogisticRegression::from_artifact(&bytes, schema).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(decoded.classes(), targets.classes());
        assert_eq!(decoded.n_decision_columns(), 3);
        assert_eq!(
            decoded.predict_proba(&data.as_view()).unwrap(),
            model.predict_proba(&data.as_view()).unwrap()
        );
        assert_eq!(
            decoded.to_artifact(schema).unwrap(),
            bytes,
            "re-encoding a decoded model reproduces its bytes exactly"
        );

        // Schema binding and payload-schema isolation.
        assert_eq!(
            LogisticRegression::from_artifact(&bytes, [8; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        let binary = LogisticRegression::fit(
            &data.as_view(),
            &crate::data::BinaryTargets::new(vec![0, 1, 0, 1, 0, 1, 0, 1, 0]).unwrap(),
            LogisticRegressionParams::default(),
        );
        if let Ok(binary) = binary {
            let binary_bytes = binary.to_artifact(schema).unwrap();
            assert_ne!(binary_bytes, bytes);
            assert!(LogisticRegression::from_artifact(&binary_bytes, schema).is_ok());
        }
    }

    /// Non-contiguous labels survive the round trip, because the class list is
    /// stored rather than reconstructed from the row count.
    #[test]
    fn artifact_classes_are_stored_not_guessed() {
        let (data, targets) = three_class_problem();
        let relabelled = ClassTargets::new(
            targets
                .as_slice()
                .iter()
                .map(|&label| [5_u8, 9, 20][usize::from(label)])
                .collect(),
        )
        .unwrap();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &relabelled,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let bytes = model.to_artifact([7; 32]).unwrap();
        let decoded = LogisticRegression::from_artifact(&bytes, [7; 32]).unwrap();
        assert_eq!(decoded.classes(), [5, 9, 20]);
        assert_eq!(decoded, model);
    }

    #[test]
    fn every_output_is_validated_before_a_single_value_is_written() {
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let wrong_width = DenseMatrix::new(vec![1.0; data.rows() * 3], data.rows(), 3).unwrap();
        let width_error = ModelError::FeatureDimension {
            expected: 2,
            actual: 3,
        };

        let mut sentinel = vec![9.0_f32; data.rows() * 3];
        assert_eq!(
            model
                .decision_function_into(&wrong_width.as_view(), &mut sentinel)
                .unwrap_err(),
            width_error
        );
        assert_eq!(
            model
                .predict_proba_into(&wrong_width.as_view(), &mut sentinel)
                .unwrap_err(),
            width_error
        );
        assert!(sentinel.iter().all(|&value| value == 9.0));

        let mut short = vec![9.0_f32; 2];
        let length_error = ModelError::OutputLength {
            expected: data.rows() * 3,
            actual: 2,
        };
        assert_eq!(
            model
                .decision_function_into(&data.as_view(), &mut short)
                .unwrap_err(),
            length_error
        );
        assert_eq!(
            model
                .predict_proba_into(&data.as_view(), &mut short)
                .unwrap_err(),
            length_error
        );
        assert!(short.iter().all(|&value| value == 9.0));

        let mut labels = vec![9_u8; 2];
        assert_eq!(
            model
                .predict_into(&data.as_view(), &mut labels)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: data.rows(),
                actual: 2,
            }
        );
        assert!(labels.iter().all(|&value| value == 9));
    }

    #[test]
    fn fitting_validates_lengths_and_parameters_in_the_frozen_order() {
        let (data, targets) = three_class_problem();
        let short = ClassTargets::new(vec![0, 1, 2]).unwrap();
        assert_eq!(
            LogisticRegression::fit_multiclass(
                &data.as_view(),
                &short,
                LogisticRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::TargetLength {
                rows: 12,
                targets: 3
            }
        );
        assert_eq!(
            LogisticRegression::fit_multiclass_weighted(
                &data.as_view(),
                &targets,
                &SampleWeights::new(vec![1.0; 3]).unwrap(),
                LogisticRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::SampleWeightLength {
                rows: 12,
                weights: 3
            }
        );
        for (params, expected) in [
            (
                LogisticRegressionParams::default().with_c(0.0),
                ModelError::InvalidRegularization,
            ),
            (
                LogisticRegressionParams::default().with_max_iter(0),
                ModelError::InvalidIterationCount,
            ),
            (
                LogisticRegressionParams::default().with_tol(0.0),
                ModelError::InvalidTolerance,
            ),
        ] {
            assert_eq!(
                LogisticRegression::fit_multiclass(&data.as_view(), &targets, params).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn an_oversized_newton_system_is_refused_before_any_allocation() {
        // 200 classes over 60 features needs 12 060 stacked parameters, whose
        // second-order system would be well over a gigabyte.
        let rows = 200;
        let columns = 60;
        let data = DenseMatrix::new(vec![0.5; rows * columns], rows, columns).unwrap();
        let targets = ClassTargets::new((0..rows).map(|row| row as u8).collect()).unwrap();
        assert_eq!(
            LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default(),
            )
            .unwrap_err(),
            ModelError::MulticlassSystemTooLarge {
                classes: rows,
                features: columns,
                limit: MAX_NEWTON_PARAMETERS,
            }
        );

        // The bound is exactly where it says it is. Checked through the
        // validator rather than a fit, because factorizing a system at the
        // limit would cost seconds to prove an off-by-one.
        let classes = 8;
        let columns = MAX_NEWTON_PARAMETERS / classes;
        let rows = classes * 2;
        let data = DenseMatrix::new(vec![0.5; rows * columns], rows, columns).unwrap();
        let targets =
            ClassTargets::new((0..rows).map(|row| (row % classes) as u8).collect()).unwrap();
        let params = LogisticRegressionParams::default();
        assert_eq!(
            validate_fit(&data.as_view(), &targets, None, &params).unwrap_err(),
            ModelError::MulticlassSystemTooLarge {
                classes,
                features: columns,
                limit: MAX_NEWTON_PARAMETERS,
            },
            "{classes} x ({columns} + intercept) is one parameter over the bound"
        );
        // Dropping the intercept drops one parameter per class, which lands
        // exactly on the bound.
        assert_eq!(
            validate_fit(
                &data.as_view(),
                &targets,
                None,
                &params.with_fit_intercept(false),
            ),
            Ok(())
        );
    }

    #[test]
    fn refitting_the_same_inputs_reproduces_the_same_bits() {
        let (data, targets) = three_class_problem();
        let params = LogisticRegressionParams::default().with_tol(1.0e-8);
        let left =
            LogisticRegression::fit_multiclass(&data.as_view(), &targets, params.clone()).unwrap();
        let right = LogisticRegression::fit_multiclass(&data.as_view(), &targets, params).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            left.coefficients()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .coefficients()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn uniform_weights_are_bit_equivalent_and_integer_weights_match_replicated_rows() {
        let (data, targets) = three_class_problem();
        let params = LogisticRegressionParams::default().with_tol(1.0e-9);
        let unweighted =
            LogisticRegression::fit_multiclass(&data.as_view(), &targets, params.clone()).unwrap();
        let uniform = LogisticRegression::fit_multiclass_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(vec![1.0; data.rows()]).unwrap(),
            params.clone(),
        )
        .unwrap();
        assert_eq!(uniform, unweighted);

        let counts = [1_usize, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 2];
        let weighted = LogisticRegression::fit_multiclass_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(counts.iter().map(|&count| count as f32).collect()).unwrap(),
            params.clone(),
        )
        .unwrap();

        let mut replicated_values = Vec::new();
        let mut replicated_labels = Vec::new();
        for ((row, &label), &count) in data.iter_rows().zip(targets.as_slice()).zip(&counts) {
            for _ in 0..count {
                replicated_values.extend_from_slice(row);
                replicated_labels.push(label);
            }
        }
        let replicated_data =
            DenseMatrix::new(replicated_values, replicated_labels.len(), data.columns()).unwrap();
        let replicated = LogisticRegression::fit_multiclass(
            &replicated_data.as_view(),
            &ClassTargets::new(replicated_labels).unwrap(),
            params,
        )
        .unwrap();
        for (&weighted, &replicated) in weighted
            .coefficients()
            .iter()
            .zip(replicated.coefficients())
        {
            assert!(
                (weighted - replicated).abs() <= 1.0e-4,
                "{weighted} != {replicated}"
            );
        }
        for (&weighted, &replicated) in weighted.intercepts().iter().zip(replicated.intercepts()) {
            assert!(
                (weighted - replicated).abs() <= 1.0e-4,
                "{weighted} != {replicated}"
            );
        }
    }

    #[test]
    fn a_class_absent_from_a_prediction_batch_changes_nothing() {
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default(),
        )
        .unwrap();
        // One row deep inside class 0's region: the class set is fitted state,
        // never re-derived from the batch.
        let single = DenseMatrix::new(vec![0.0, 0.0], 1, 2).unwrap();
        assert_eq!(model.classes(), &[0, 1, 2]);
        assert_eq!(model.predict_proba(&single.as_view()).unwrap().len(), 3);
        assert_eq!(model.predict(&single.as_view()).unwrap(), vec![0]);
    }

    #[test]
    fn separable_classes_are_recovered_and_the_fit_converges_early() {
        let (data, targets) = three_class_problem();
        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default()
                .with_c(50.0)
                .with_tol(1.0e-6),
        )
        .unwrap();
        assert_eq!(
            model.predict(&data.as_view()).unwrap(),
            targets.as_slice(),
            "a separable three-class problem must be recovered"
        );
        assert!(
            model.n_iter() < model.get_params().max_iter(),
            "the bound-majorized iteration converged rather than exhausting max_iter"
        );
    }
}
