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
//! That argument is about the exact system, and it is only as good as the
//! *factorized* one. A shared direction the assembled matrix barely sees is a
//! direction in which rounding decides the update, and an update with a shared
//! component moves the iterate out of the centred subspace — which is how a
//! frozen semantic turns into a rounding artefact. The Newton section below is
//! what makes the argument true of the arithmetic as well.
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
//! The exact system is singular on a whole subspace, not in one direction.
//! Adding the same vector to every class's parameter row leaves every
//! probability unchanged, so the loss curvature annihilates every stacked
//! vector of the form "one shared row, repeated per class" — one direction per
//! parameter coordinate, feature coordinates included, whether or not an
//! intercept is fitted.
//!
//! Only the intercept direction is *exactly* singular in the assembled system;
//! the feature directions carry the L2 penalty `1/(C * scale^2)` and nothing
//! else. That is identification on paper and not in arithmetic: on a weakly
//! regularized, badly scaled design the penalty is `1e-11` or smaller against a
//! largest curvature of order the sample weight, so the factorization's pivot
//! in that direction is decided by rounding and the solve refuses. Assembling
//! the system was the whole cost, and it was being thrown away for a direction
//! the answer does not even use.
//!
//! So the curvature is supplied for the whole subspace rather than for the one
//! member of it that no penalty reaches. Each shared coordinate gets a rank-one
//! block across classes, sized to the curvature that coordinate carries at
//! uniform probabilities. This changes no fitted value in exact arithmetic and
//! is not a modelling choice: the loss curvature, the penalty and the added
//! blocks all preserve the split between the shared subspace and its orthogonal
//! complement, the gradient lies in the complement at every iterate, and the
//! iterate starts at zero — so the added blocks multiply a zero component at
//! every iteration. What they change is the conditioning of the matrix that
//! component is solved against.
//!
//! No reference class is pinned, and none can be: the centred representative is
//! the crate's frozen semantic for this fit.

use super::{
    LogisticRegression, LogisticRegressionParams, LogisticSolver, Standardization, build_design,
    sample_weight, solve_positive_definite, standardize, validate_common_fit,
};
use crate::api::ModelError;
use crate::data::{ClassTargets, MatrixView, SampleWeights};
use crate::loss::{newton_decrement, raw_score};
use crate::numeric::{log_sum_exp, softmax_in_place};
use crate::optimize::armijo_backtracking;

/// Largest stacked Newton system the multinomial fit will build.
///
/// The system is `classes * (features + intercept)` square in `f64`, so this
/// caps one fit's dominant allocation at 32 MiB. It is checked at the public
/// boundary, before any allocation or training work, rather than discovered as
/// an out-of-memory abort partway through fitting.
pub(super) const MAX_NEWTON_PARAMETERS: usize = 2_048;

/// Largest stacked system the matrix-free multinomial fit will accept.
///
/// The same 32 MiB storage budget, applied to what
/// [`LogisticSolver::Lbfgs`] actually allocates.
/// Its workspace is `4 + 2 * memory` `f64` per parameter — four working vectors
/// plus one iterate difference and one gradient difference per stored
/// correction pair — which is 20 `f64`, or 160 bytes, at the default memory.
/// That puts this bound at 20 MiB, comfortably inside the same envelope the
/// exact path is held to.
///
/// It is sixty-four times the exact bound because the cost is *linear* here
/// rather than quadratic. That is the whole reason this solver exists, and the
/// two constants sitting next to each other is the clearest statement of it.
pub(super) const MAX_MATRIX_FREE_PARAMETERS: usize = 131_072;

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

    let penalties = super::lbfgs::scaled_penalties(&scales, params.c);
    if params.solver == LogisticSolver::Lbfgs {
        let (theta, iterations) = super::lbfgs::fit_multinomial(
            super::lbfgs::DesignView {
                design: &design,
                sample_weights,
                penalties: &penalties,
                columns,
                parameter_count,
                intercept_index,
                inverse_total_weight: 1.0 / total_weight,
            },
            &class_of_row,
            classes,
            &params,
        )?;
        return LogisticRegression::from_standardized(
            &theta,
            &means,
            &scales,
            columns,
            intercept_index,
            targets.classes().to_vec(),
            params,
            iterations,
        );
    }
    // Curvature for the directions the loss ignores entirely: adding the same
    // vector to every class's parameter row. There is one such direction per
    // parameter coordinate, not one in total — see the module documentation.
    //
    // The size is the curvature a standardized coordinate carries at uniform
    // probabilities. `standardize` gives every design column, including the
    // intercept's constant one, a weighted second moment of `total_weight`, and
    // at `p = 1/classes` the per-coordinate curvature block is
    // `total_weight / classes * (I - 1 1' / classes)`. So one constant is the
    // right scale for every coordinate: it neither dominates the system nor
    // vanishes into it.
    let centring_curvature = total_weight / classes as f64;

    let mut theta = vec![0.0_f64; parameters];
    let mut gradient = vec![0.0_f64; parameters];
    let mut hessian = vec![0.0_f64; parameters * parameters];
    let mut update = vec![0.0_f64; parameters];
    let mut trial = vec![0.0_f64; parameters];
    let mut probabilities = vec![0.0_f64; classes];
    // The step length's own score row, kept separate from the one the Newton
    // system accumulates through so neither has to be restored around the other.
    let mut trial_scores = vec![0.0_f64; classes];
    // The objective the step length is judged against: the same penalized
    // cross-entropy the stacked Newton system expands, in the same standardized
    // coordinates. The centring curvature added to the hessian below is *not*
    // part of it — that term regularizes a direction the objective genuinely
    // does not see, and adding it here would make the step length answer to a
    // quantity the fit does not minimize.
    let penalized_value = |theta: &[f64], scores: &mut [f64]| -> f64 {
        let mut total = 0.0_f64;
        for (row_index, design_row) in design.chunks_exact(parameter_count).enumerate() {
            for (class, slot) in scores.iter_mut().enumerate() {
                *slot = raw_score(
                    &theta[class * parameter_count..(class + 1) * parameter_count],
                    design_row,
                    columns,
                    intercept_index,
                );
            }
            total += sample_weight(sample_weights, row_index)
                * (log_sum_exp(scores) - scores[class_of_row[row_index]]);
        }
        for class in 0..classes {
            let offset = class * parameter_count;
            for (column, &penalty) in penalties.iter().enumerate() {
                total += 0.5 * penalty * theta[offset + column] * theta[offset + column];
            }
        }
        total
    };
    let mut value = penalized_value(&theta, &mut trial_scores);
    let mut iterations = 0;
    let mut converged = false;
    // Refuse by default, exactly as the binary path does: only a step the loop
    // actually took can certify an exhausted budget, and `validate_fit` has
    // already guaranteed the loop below runs at least once.
    let mut decrement = f64::INFINITY;
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
        // One rank-one block per coordinate, each supported on the classes that
        // share it. Only the lower triangle is written, matching the
        // factorization's reader.
        for coordinate in 0..parameter_count {
            for left in 0..classes {
                for right in 0..=left {
                    hessian[(left * parameter_count + coordinate) * parameters
                        + right * parameter_count
                        + coordinate] += centring_curvature;
                }
            }
        }

        solve_positive_definite(&mut hessian, &gradient, &mut update, parameters)?;
        let max_update = update.iter().fold(0.0_f64, |max, step| max.max(step.abs()));
        decrement = newton_decrement(&gradient, &update);
        iterations = iteration + 1;
        // The full step first, exactly as the binary path takes it, so a fit
        // whose exact step already descends keeps its iterate sequence.
        let Some(damped) =
            armijo_backtracking(&mut theta, &update, &mut trial, value, decrement, |probe| {
                penalized_value(probe, &mut trial_scores)
            })
        else {
            break;
        };
        value = damped.value;
        // The exact step's size is the convergence test here too.
        if max_update <= f64::from(params.tol) {
            converged = true;
            break;
        }
    }
    // The same acceptance the binary path applies, over the stacked system.
    // The centring curvature added above is part of the matrix the step was
    // solved against, so the decrement is affine invariant here too; the
    // gradient is orthogonal to the direction that curvature regularizes, so
    // that direction contributes nothing to the product.
    let certified = decrement <= f64::from(params.tol);
    if !converged && !certified {
        return Err(ModelError::SolverDidNotConverge { iterations });
    }

    LogisticRegression::from_standardized(
        &theta,
        &means,
        &scales,
        columns,
        intercept_index,
        targets.classes().to_vec(),
        params,
        iterations,
    )
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
    // The bound is a property of the solver's storage, not of the model, so it
    // is read from the selected solver rather than fixed at the worst case.
    // Every shape the exact path accepted before still takes the exact path and
    // still produces the identical fit; the matrix-free path widens the domain
    // rather than reinterpreting any part of it.
    let limit = match params.solver() {
        LogisticSolver::Newton => MAX_NEWTON_PARAMETERS,
        LogisticSolver::Lbfgs => MAX_MATRIX_FREE_PARAMETERS,
    };
    let parameters = targets
        .n_classes()
        .checked_mul(data.columns() + usize::from(params.fit_intercept))
        .filter(|&parameters| parameters <= limit)
        .ok_or(ModelError::MulticlassSystemTooLarge {
            classes: targets.n_classes(),
            features: data.columns(),
            limit,
        })?;
    debug_assert!(parameters > 0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ArtifactError, ModelArtifact};
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
            model.predict_positive_proba_one(&[0.2, 0.3]).unwrap_err(),
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
                // The refusal belongs to the exact path, which is no longer the
                // default: the matrix-free path fits this shape.
                LogisticRegressionParams::default().with_solver(LogisticSolver::Newton),
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
        let params = LogisticRegressionParams::default().with_solver(LogisticSolver::Newton);
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
    fn the_matrix_free_solver_fits_shapes_the_exact_one_refuses() {
        // The whole reason the seam exists. This shape needs 2408 stacked
        // parameters, so the exact path refuses it before allocating a 46 MB
        // system; the matrix-free path holds 20 f64 per parameter and fits.
        let (classes, columns, rows) = (8_usize, 300_usize, 64_usize);
        let mut state = 0x5eed_u64;
        let values = (0..rows * columns)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1_u32 << 31) as f32) * 2.0 - 1.0
            })
            .collect::<Vec<_>>();
        let data = DenseMatrix::new(values, rows, columns).unwrap();
        let targets =
            ClassTargets::new((0..rows).map(|row| (row % classes) as u8).collect()).unwrap();

        assert_eq!(
            LogisticRegression::fit_multiclass(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default().with_solver(LogisticSolver::Newton),
            )
            .unwrap_err(),
            ModelError::MulticlassSystemTooLarge {
                classes,
                features: columns,
                limit: MAX_NEWTON_PARAMETERS,
            },
            "the exact path's refusal is unchanged"
        );

        let model = LogisticRegression::fit_multiclass(
            &data.as_view(),
            &targets,
            LogisticRegressionParams::default()
                .with_solver(LogisticSolver::Lbfgs)
                .with_max_iter(500),
        )
        .expect("the matrix-free path fits it");
        assert_eq!(model.n_decision_columns(), classes);
        assert_eq!(model.coefficients().len(), classes * columns);
        // Fitted, not merely returned: every training row is classified
        // correctly on a problem this over-parameterized, and the score rows
        // are still centred.
        let labels = model.predict(&data.as_view()).expect("labels");
        assert_eq!(labels, targets.as_slice());
        let scores = model.decision_function(&data.as_view()).expect("scores");
        for row in scores.chunks_exact(classes) {
            assert!(row.iter().sum::<f32>().abs() <= 1.0e-3, "{row:?}");
        }
    }

    #[test]
    fn the_matrix_free_bound_is_enforced_where_it_says_it_is() {
        // Checked through the validator: the point is that the refusal happens
        // before allocation, so building the fit to observe it would defeat it.
        let classes = 256_usize;
        let columns = MAX_MATRIX_FREE_PARAMETERS / classes;
        let rows = classes;
        let data = DenseMatrix::new(vec![0.5; rows * columns], rows, columns).unwrap();
        let targets = ClassTargets::new((0..rows).map(|row| row as u8).collect()).unwrap();
        let params = LogisticRegressionParams::default().with_solver(LogisticSolver::Lbfgs);
        assert_eq!(
            validate_fit(&data.as_view(), &targets, None, &params).unwrap_err(),
            ModelError::MulticlassSystemTooLarge {
                classes,
                features: columns,
                limit: MAX_MATRIX_FREE_PARAMETERS,
            },
            "{classes} x ({columns} + intercept) is one parameter over the bound"
        );
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

    // ------------------------------------------------------- convergence rule

    /// The multiclass twin of the binary ill-conditioned region.
    ///
    /// The stacked system carries the binary path's conditioning trouble plus
    /// its own: the loss curvature annihilates every direction that shifts all
    /// classes alike, one per parameter coordinate, so on a badly scaled design
    /// the only thing left holding those directions up used to be a penalty of
    /// order `1e-12`. That is why this region refused 191 of its 576 cases with
    /// `LinearSolveFailed` until the curvature for that subspace was supplied
    /// directly.
    fn ill_conditioned_neighbourhood() -> Vec<(DenseMatrix, ClassTargets, LogisticRegressionParams)>
    {
        let mut state = 0x0517_2026_0726_2001_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from((state >> 33) as u32) / f64::from(1_u32 << 31) / 2.0
        };
        let mut normal = move || (0..12).map(|_| next()).sum::<f64>() - 6.0;
        let mut cases = Vec::new();
        for _ in 0..6 {
            for &rows in &[12_usize, 30] {
                for &columns in &[2_usize, 4] {
                    for &classes in &[3_usize, 5] {
                        for &column_scale in &[30.0_f64, 1.0e3] {
                            for &separation in &[0.0_f64, 1.5] {
                                let mut values = Vec::with_capacity(rows * columns);
                                let mut labels = Vec::with_capacity(rows);
                                for row in 0..rows {
                                    let class = row % classes;
                                    labels.push(class as u8);
                                    for column in 0..columns {
                                        let shift = if column == 0 {
                                            separation
                                                * (class as f64 - (classes as f64 - 1.0) / 2.0)
                                        } else {
                                            0.0
                                        };
                                        let base = normal() + shift;
                                        values
                                            .push((base * column_scale.powi(column as i32)) as f32);
                                    }
                                }
                                let data = DenseMatrix::new(values, rows, columns).unwrap();
                                let targets = ClassTargets::new(labels).unwrap();
                                for &c in &[1.0e3_f32, 1.0e6, 1.0e9] {
                                    cases.push((
                                        data.clone(),
                                        targets.clone(),
                                        // As on the binary side: this region is
                                        // about the exact step's damping and
                                        // its acceptance certificate, so it
                                        // names that solver rather than
                                        // following the default.
                                        LogisticRegressionParams::default()
                                            .with_c(c)
                                            .with_solver(LogisticSolver::Newton),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        cases
    }

    /// The multinomial objective, at `f64`, in the caller's own feature space.
    fn penalized_objective(
        data: &DenseMatrix,
        class_of_row: &[usize],
        classes: usize,
        coefficients: &[f32],
        intercepts: &[f32],
        c: f64,
    ) -> f64 {
        let view = data.as_view();
        let columns = view.columns();
        let mut scores = vec![0.0_f64; classes];
        let mut total = 0.0;
        for (row_index, row) in view.iter_rows().enumerate() {
            for (class, slot) in scores.iter_mut().enumerate() {
                let mut raw = f64::from(intercepts[class]);
                for (column, &value) in row.iter().enumerate() {
                    raw += f64::from(coefficients[class * columns + column]) * f64::from(value);
                }
                *slot = raw;
            }
            total += crate::numeric::log_sum_exp(&scores) - scores[class_of_row[row_index]];
        }
        total
            + 0.5 / c
                * coefficients
                    .iter()
                    .map(|&value| f64::from(value) * f64::from(value))
                    .sum::<f64>()
    }

    /// Every single-coordinate neighbour of the fit costs at least as much,
    /// probed with a *relative* perturbation for the reason the binary twin
    /// states: these coefficient vectors span many orders of magnitude and a
    /// fixed absolute step measures the arithmetic rather than the fit.
    fn is_a_local_minimum(
        data: &DenseMatrix,
        class_of_row: &[usize],
        classes: usize,
        coefficients: &[f32],
        intercepts: &[f32],
        c: f64,
    ) -> bool {
        let base = penalized_objective(data, class_of_row, classes, coefficients, intercepts, c);
        let slack = 1.0e-7 * base.abs().max(1.0);
        let mut probe = coefficients.to_vec();
        for index in 0..coefficients.len() {
            let value = coefficients[index];
            if value == 0.0 {
                continue;
            }
            for direction in [1.0_f32, -1.0] {
                probe[index] = value * (1.0 + direction * 1.0e-3);
                let neighbour =
                    penalized_objective(data, class_of_row, classes, &probe, intercepts, c);
                probe[index] = value;
                if neighbour < base - slack {
                    return false;
                }
            }
        }
        true
    }

    fn class_positions(targets: &ClassTargets) -> Vec<usize> {
        targets
            .as_slice()
            .iter()
            .map(|&label| targets.class_index(label).expect("observed class"))
            .collect()
    }

    // ------------------------------------------------- shared-subspace curvature

    /// A design whose shared-coordinate penalty is far below the arithmetic.
    ///
    /// Column 1 is column 0 scaled by `1e5`, so `standardize` gives it a scale
    /// of about `1e5` and the L2 penalty it carries in standardized space is
    /// `1/(C * scale^2)`. At `C = 1e6` that is around `1e-16`, against a largest
    /// curvature of order the row count. Nothing else opposes the direction that
    /// adds the same amount to that coordinate in every class, because the loss
    /// is exactly flat along it — so before the curvature for that subspace was
    /// supplied, the pivot there was decided by rounding and the factorization
    /// refused.
    fn shared_subspace_stress() -> (DenseMatrix, ClassTargets) {
        let classes = 3_usize;
        let rows = 30_usize;
        let mut values = Vec::with_capacity(rows * 2);
        let mut labels = Vec::with_capacity(rows);
        for row in 0..rows {
            let class = row % classes;
            labels.push(class as u8);
            let base = (row as f32) * 0.37 - 5.0 + class as f32 * 1.5;
            values.push(base);
            values.push(base * 1.0e5);
        }
        (
            DenseMatrix::new(values, rows, 2).expect("stress matrix"),
            ClassTargets::new(labels).expect("stress targets"),
        )
    }

    /// The shared-coordinate subspace is held up by curvature, not by a penalty.
    ///
    /// This is the smallest statement of the defect the region test below
    /// measures in bulk. Restricting the added curvature to the intercept
    /// coordinate — which is what the crate did through 0.2.0 — makes both
    /// assertions here fail with `LinearSolveFailed`, and the second one fails
    /// even with the intercept case restored, because without an intercept the
    /// old code added no curvature at all.
    #[test]
    fn a_shared_coordinate_the_penalty_barely_reaches_is_still_fitted() {
        let (data, targets) = shared_subspace_stress();
        let params = LogisticRegressionParams::default().with_c(1.0e6);
        let model = LogisticRegression::fit_multiclass(&data.as_view(), &targets, params.clone())
            .expect("the stacked system is identified in every direction");
        assert!(
            is_a_local_minimum(
                &data,
                &class_positions(&targets),
                targets.classes().len(),
                model.coefficients(),
                model.intercepts(),
                f64::from(params.c()),
            ),
            "the fit was returned away from the minimum"
        );

        // Without an intercept the shared subspace is still there — it is a
        // property of the softmax, not of the intercept column.
        let no_intercept = params.with_fit_intercept(false);
        let model =
            LogisticRegression::fit_multiclass(&data.as_view(), &targets, no_intercept.clone())
                .expect("an intercept-free fit is identified too");
        assert!(model.intercepts().iter().all(|&value| value == 0.0));
        assert!(is_a_local_minimum(
            &data,
            &class_positions(&targets),
            targets.classes().len(),
            model.coefficients(),
            model.intercepts(),
            f64::from(no_intercept.c()),
        ));
    }

    /// The whole ill-conditioned region, fitted rather than refused.
    ///
    /// The binary twin of this test requires every case to fit, and that is the
    /// standard this one is measured against rather than a weaker one invented
    /// for it. Supplying the shared subspace's curvature took the region from
    /// **385 fits and 191 `LinearSolveFailed` refusals** to **563 fits and 13**,
    /// and all 563 land at a local minimum of the penalized objective in the
    /// caller's own feature space — measured both ways, as the damping work was.
    ///
    /// The 13 that remain are a different failure: every one of them is twelve
    /// rows over four columns, mostly across five classes, so the design has two
    /// or three rows per class and the curvature that collapses is the *data's*
    /// rather than the parametrization's. That is the same class of refusal the
    /// binary path can produce, and it is left refusing.
    ///
    /// The bounds below are what makes this non-vacuous. Restricting the added
    /// curvature to the intercept coordinate returns 385 fits and 191 refusals,
    /// which fails the first two; the last one fails if the generator ever stops
    /// producing the region these numbers were measured on. None of them asserts
    /// that a refusal still happens, so a later fix for the twelve-row residue
    /// is not obstructed by this test.
    #[test]
    fn the_ill_conditioned_multinomial_region_is_fitted_rather_than_refused() {
        let cases = ill_conditioned_neighbourhood();
        let mut fitted = 0_usize;
        let mut singular = 0_usize;
        for (index, (data, targets, params)) in cases.iter().enumerate() {
            let model = match LogisticRegression::fit_multiclass(
                &data.as_view(),
                targets,
                params.clone(),
            ) {
                Ok(model) => model,
                Err(ModelError::LinearSolveFailed) => {
                    singular += 1;
                    continue;
                }
                Err(other) => panic!("case {index} was refused with {other:?}"),
            };
            assert!(
                is_a_local_minimum(
                    data,
                    &class_positions(targets),
                    targets.classes().len(),
                    model.coefficients(),
                    model.intercepts(),
                    f64::from(params.c()),
                ),
                "case {index} was returned away from the minimum"
            );
            fitted += 1;
        }
        assert_eq!(fitted + singular, cases.len());
        assert!(
            fitted >= 550,
            "only {fitted} of {} cases were fitted; the shared-coordinate subspace is \
             being left to the penalty again",
            cases.len()
        );
        assert!(
            singular <= 20,
            "{singular} cases collapsed; the recorded residue is 13 and they are all the \
             same twelve-row shape"
        );
        assert!(
            cases.len() >= 576,
            "the region shrank to {} cases and no longer covers the conditioning it was \
             built from",
            cases.len()
        );
    }

    /// The centred representative survives the arithmetic, not only the algebra.
    ///
    /// "One centred coefficient row and intercept per class with no pinned
    /// reference class" is the crate's frozen semantic for this fit, and the
    /// argument for it is that the gradient lies in the sum-zero subspace, the
    /// Newton system maps that subspace into itself, and the iterate starts
    /// there. Every step of that is true of the exact system — and it was being
    /// used to conclude something about the *factorized* one.
    ///
    /// It does not survive the factorization when the shared-coordinate
    /// directions are held up only by a `1e-12` penalty. Rounding in the pivot
    /// then puts an arbitrary shared component into the update, the iterate
    /// leaves the centred subspace, and the fit is returned as some other
    /// representative of the same probabilities. Restricting the added curvature
    /// to the intercept coordinate makes **231 of this region's 385 fits** fail
    /// the bound below, the worst of them off by twice its own largest
    /// coefficient — so this was a live violation of a frozen semantic on every
    /// ill-conditioned design, not a hypothetical one, and nothing detected it
    /// because [`scores_are_centred_rather_than_measured_against_a_reference_class`]
    /// only ever asked well-conditioned data.
    ///
    /// The bound is the one that test uses, applied per coordinate rather than
    /// per score row: `n_classes` `f32` ulps of the largest value sharing the
    /// coordinate. All 563 fits clear it with a factor of three to spare.
    #[test]
    fn every_ill_conditioned_multinomial_fit_is_returned_centred() {
        let cases = ill_conditioned_neighbourhood();
        let mut checked = 0_usize;
        for (index, (data, targets, params)) in cases.iter().enumerate() {
            let Ok(model) =
                LogisticRegression::fit_multiclass(&data.as_view(), targets, params.clone())
            else {
                continue;
            };
            let classes = targets.classes().len();
            let columns = data.columns();
            let tolerance = row_sum_tolerance(classes);
            let mut sums = Vec::with_capacity(columns + 1);
            for column in 0..columns {
                sums.push(
                    (0..classes)
                        .map(|class| model.coefficients()[class * columns + column])
                        .collect::<Vec<_>>(),
                );
            }
            sums.push(model.intercepts().to_vec());
            for (coordinate, shared) in sums.iter().enumerate() {
                let magnitude = shared
                    .iter()
                    .fold(1.0_f32, |max, value| max.max(value.abs()));
                let sum = shared.iter().sum::<f32>().abs();
                assert!(
                    sum <= tolerance * magnitude,
                    "case {index} coordinate {coordinate}: {shared:?} sums to {sum}, which \
                     is more than {classes} f32 ulps of {magnitude} — the fit drifted out \
                     of the centred subspace"
                );
            }
            checked += 1;
        }
        assert!(
            checked >= 550,
            "only {checked} of {} cases produced a fit to check",
            cases.len()
        );
    }

    /// An exhausted multinomial budget that reached the minimum is returned.
    ///
    /// The multiclass half of the same contract the binary path carries:
    /// refusing on plain exhaustion would convert every fit counted here into a
    /// spurious error. What changed with the damped step is *how* the exhausted
    /// population is reached. This region used to exhaust the default budget on
    /// 52 of its fits, because the undamped step overshot and never settled;
    /// damped, the whole region converges well inside the hundred iterations it
    /// is given — at most 57 — so the default budget no longer produces an
    /// exhausted fit at all and asserting that it does would be asserting a
    /// defect.
    ///
    /// **That is the whole reason this construction exists, and it is not
    /// evidence the certificate is dead.** `max_iter` is a public parameter: a
    /// caller who budgets fewer iterations than a design needs lands in exactly
    /// the state judged here, and this test is that caller. Surveyed after the
    /// conditioning fix, 11 520 multinomial designs spanning one to four
    /// columns, two to eight classes, column scales to `1e5`, separations to
    /// `20` and `C` to `1e12` produced **no** default-budget exhaustion at all,
    /// with a worst case of 74 iterations of 100 — so the certificate's
    /// reachability on this path rests on the budget being a caller's to set,
    /// not on a region that happens to be slow. It is the same rule the binary
    /// path applies, where 139 of 704 designs do exhaust the default budget and
    /// are accepted; giving the two shapes different acceptance rules because
    /// one sample did not reach 100 iterations is how the defect that 0.2.0
    /// closed comes back on the first design that needs 101.
    ///
    /// The budget is therefore set to one iteration short of what each fit
    /// needs. The loop then runs out without the tolerance break ever firing,
    /// and the construction exercises both answers: 500 of the 563 are accepted
    /// because the Newton decrement says they are already at the minimum — and
    /// each of those is checked against the objective in the caller's own
    /// feature space — while 63 are refused because they are not. Deleting the
    /// certificate refuses all 563; making it unconditional accepts all 563.
    #[test]
    fn an_exhausted_multinomial_budget_that_reached_its_minimum_is_fitted_not_refused() {
        let cases = ill_conditioned_neighbourhood();
        let mut fitted = 0_usize;
        let mut singular = 0_usize;
        let mut probed = 0_usize;
        let mut accepted = 0_usize;
        let mut refused = 0_usize;
        for (index, (data, targets, params)) in cases.iter().enumerate() {
            let model = match LogisticRegression::fit_multiclass(
                &data.as_view(),
                targets,
                params.clone(),
            ) {
                Ok(model) => {
                    fitted += 1;
                    model
                }
                // The stacked curvature genuinely collapsed; that is the
                // neighbouring contract and not what this test is about.
                Err(ModelError::LinearSolveFailed) => {
                    singular += 1;
                    continue;
                }
                Err(other) => panic!("case {index} was refused with {other:?}"),
            };
            // Damped, every fit here converges well inside the default budget,
            // so the acceptance path has to be reached deliberately.
            assert!(
                model.n_iter() < params.max_iter(),
                "case {index} still exhausts its default budget"
            );
            if model.n_iter() < 2 {
                continue;
            }
            probed += 1;
            let short = params.clone().with_max_iter(model.n_iter() - 1);
            match LogisticRegression::fit_multiclass(&data.as_view(), targets, short.clone()) {
                Ok(model) => {
                    accepted += 1;
                    assert_eq!(model.n_iter(), short.max_iter());
                    assert!(
                        is_a_local_minimum(
                            data,
                            &class_positions(targets),
                            targets.classes().len(),
                            model.coefficients(),
                            model.intercepts(),
                            f64::from(params.c()),
                        ),
                        "case {index} was accepted at an exhausted budget away from the minimum"
                    );
                }
                Err(ModelError::SolverDidNotConverge { iterations }) => {
                    assert_eq!(iterations, short.max_iter());
                    refused += 1;
                }
                Err(other) => panic!("case {index} was refused with {other:?}"),
            }
        }
        // As generated the region is 576 cases: 563 fits and 13
        // collapsed-curvature refusals, with no non-convergence refusal left.
        assert_eq!(fitted + singular, cases.len());
        assert_eq!(accepted + refused, probed);
        // Both answers have to occur, or this constrains the certificate from
        // one side only.
        assert!(
            accepted * 4 >= probed * 3,
            "only {accepted} of {probed} penultimate-budget fits were accepted; a \
             certificate this strict would refuse fits that are at the minimum"
        );
        assert!(
            refused > 0,
            "no penultimate-budget fit was refused, so nothing here distinguishes the \
             certificate from an unconditional yes"
        );
    }

    /// And the same region is refused when the budget really is too short.
    #[test]
    fn the_same_multinomial_region_is_refused_when_the_budget_really_is_too_short() {
        let cases = ill_conditioned_neighbourhood();
        let mut reachable = 0_usize;
        let mut refused = 0_usize;
        for (data, targets, params) in &cases {
            if LogisticRegression::fit_multiclass(&data.as_view(), targets, params.clone()).is_err()
            {
                continue;
            }
            reachable += 1;
            if let Err(error) = LogisticRegression::fit_multiclass(
                &data.as_view(),
                targets,
                params.clone().with_max_iter(1),
            ) {
                assert_eq!(error, ModelError::SolverDidNotConverge { iterations: 1 });
                refused += 1;
            }
        }
        // All 563 are refused, for the reason the binary twin records.
        assert!(reachable > 0, "the region produced no fits to starve");
        assert!(
            refused * 10 >= reachable * 9,
            "only {refused} of {reachable} single-iteration multinomial fits were \
             refused; an acceptance rule this permissive would accept anything"
        );
    }

    /// A weighted multinomial fit is held to the same rule.
    #[test]
    fn a_starved_weighted_multinomial_fit_is_refused() {
        let (data, targets) = three_class_problem();
        let weights = SampleWeights::new(vec![
            0.5, 2.0, 1.0, 1.0, 3.0, 0.25, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0,
        ])
        .unwrap();
        assert_eq!(
            LogisticRegression::fit_multiclass_weighted(
                &data.as_view(),
                &targets,
                &weights,
                LogisticRegressionParams::default()
                    .with_max_iter(1)
                    .with_tol(1.0e-12),
            ),
            Err(ModelError::SolverDidNotConverge { iterations: 1 })
        );
        let model = LogisticRegression::fit_multiclass_weighted(
            &data.as_view(),
            &targets,
            &weights,
            LogisticRegressionParams::default(),
        )
        .expect("the default budget fits");
        assert!(model.n_iter() < 100);
    }
}
