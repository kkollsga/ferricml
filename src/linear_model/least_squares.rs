use crate::api::ModelError;
use crate::data::{MatrixView, SampleWeights};
use crate::numeric::sum_in_order;
use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef, Par, Side};

/// Forces the backend's global parallelism to sequential, at every fit.
///
/// FerricML's accumulation policy admits the backend's blocked kernels — see
/// rule 2 in `src/numeric/mod.rs` — on the strict condition that their order
/// never depends on how work was scheduled, and sequential execution is what
/// makes that condition hold. The setting is process-global, so this pins it
/// rather than reading it: FerricML does not enable the backend's `rayon`
/// feature, but Cargo unifies features across a whole dependency graph, so a
/// consumer who reaches a threaded `faer` from somewhere else could otherwise
/// change FerricML's fitted values without touching FerricML. One relaxed
/// atomic store per fit is not measurable against a decomposition, and the
/// alternative — trusting a global this crate does not own — is the kind of
/// assumption the policy exists to refuse.
fn pin_sequential() {
    faer::set_global_parallelism(Par::Seq);
    debug_assert!(
        matches!(faer::get_global_parallelism(), Par::Seq),
        "the backend refused a sequential pin, so reduction order is no longer fixed"
    );
}

pub(super) struct LeastSquaresFit {
    pub(super) coefficients: Vec<f64>,
    pub(super) intercept: f64,
    pub(super) rank: usize,
}

pub(super) struct DenseLinearFit {
    pub(super) coefficients: Vec<f64>,
    pub(super) intercept: f64,
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

    pin_sequential();
    let preprocessed = preprocess(data, targets, sample_weights, fit_intercept);
    let rows = data.rows();
    let columns = data.columns();
    let svd = preprocessed
        .matrix_ref()
        .thin_svd()
        .map_err(|_| ModelError::LinearSolveFailed)?;
    let singular = svd.S();
    let singular = singular.column_vector();
    let largest_singular = singular.iter().copied().fold(0.0_f64, f64::max);
    if !largest_singular.is_finite() {
        return Err(ModelError::LinearSolveFailed);
    }
    let numerical_floor = f64::EPSILON * rows.max(columns) as f64 * largest_singular;
    let cutoff = (f64::from(tolerance) * largest_singular).max(numerical_floor);
    let rank = singular.iter().filter(|&&value| value > cutoff).count();

    // The minimum-norm solution, `V * S⁺ * Uᵀ * b`, written out rather than
    // delegated. Two reasons, and neither is taste. The pseudo-inverse's
    // treatment of a singular value at the cutoff *is* the rank contract
    // `LinearRegression::rank()` reports, so the same `> cutoff` predicate has
    // to decide both, in one place; and both reductions here are FerricML's
    // own, so they owe the unamended rule 2 — ascending index order through
    // `sum_in_order` — rather than the carve-out that admits the decomposition
    // above.
    let left = svd.U();
    let right = svd.V();
    let mut projected = vec![0.0_f64; singular.nrows()];
    for (index, projection) in projected.iter_mut().enumerate() {
        let value = singular[index];
        if value > cutoff {
            *projection =
                sum_in_order((0..rows).map(|row| left[(row, index)] * preprocessed.targets[row]))
                    / value;
        }
    }
    let coefficients = (0..columns)
        .map(|column| {
            sum_in_order(
                projected
                    .iter()
                    .enumerate()
                    .map(|(index, &projection)| right[(column, index)] * projection),
            )
        })
        .collect::<Vec<_>>();
    let intercept = preprocessed.target_mean
        - preprocessed
            .feature_means
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

pub(super) fn fit_ridge_dense(
    data: &MatrixView<'_>,
    targets: &[f32],
    sample_weights: Option<&SampleWeights>,
    fit_intercept: bool,
    alpha: f32,
) -> Result<DenseLinearFit, ModelError> {
    if alpha == 0.0 {
        let fit = fit_dense(data, targets, sample_weights, fit_intercept, 0.0)?;
        return Ok(DenseLinearFit {
            coefficients: fit.coefficients,
            intercept: fit.intercept,
        });
    }
    pin_sequential();
    let preprocessed = preprocess(data, targets, sample_weights, fit_intercept);
    let design = preprocessed.matrix_ref();
    let mut gram: Mat<f64> = design.transpose() * design;
    for index in 0..gram.nrows() {
        gram[(index, index)] += f64::from(alpha);
    }
    let right: Mat<f64> = design.transpose() * preprocessed.targets_ref();
    let solution = gram
        .llt(Side::Lower)
        .map_err(|_| ModelError::LinearSolveFailed)?
        .solve(&right);
    let coefficients = (0..solution.nrows())
        .map(|index| solution[(index, 0)])
        .collect::<Vec<_>>();
    let intercept = preprocessed.target_mean
        - preprocessed
            .feature_means
            .iter()
            .zip(&coefficients)
            .map(|(&mean, &coefficient)| mean * coefficient)
            .sum::<f64>();
    if coefficients.iter().any(|value| !value.is_finite()) || !intercept.is_finite() {
        return Err(ModelError::LinearSolveFailed);
    }
    Ok(DenseLinearFit {
        coefficients,
        intercept,
    })
}

/// Centered, weight-scaled dense storage shared by every dense linear fit.
///
/// `matrix` is a plain `Vec<f64>` of exactly `rows * columns` values in
/// **column-major order with no padding**, and that is a declared layout rather
/// than a borrowed one. The distinction is the whole point of storing it this
/// way: `coordinate_descent` slices this buffer by column and would silently
/// transpose the design — not fail to compile — if the layout it assumed and
/// the layout the producer built ever disagreed. A backend matrix type is free
/// to pad its column stride for alignment, so leaving the buffer in one would
/// make an ABI out of a dependency's internal choice. FerricML owns the buffer
/// and hands the backend a borrowed view at the two call sites that decompose
/// it, which keeps the layout a property of this struct.
pub(super) struct PreprocessedDense {
    pub(super) matrix: Vec<f64>,
    pub(super) targets: Vec<f64>,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) feature_means: Vec<f64>,
    pub(super) target_mean: f64,
}

impl PreprocessedDense {
    /// Borrows the design as a backend matrix, without copying or reshaping.
    fn matrix_ref(&self) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&self.matrix, self.rows, self.columns)
    }

    /// Borrows the centered targets as a single-column backend matrix.
    fn targets_ref(&self) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&self.targets, self.rows, 1)
    }
}

/// Centers by the weighted means when an intercept is fitted and scales each
/// row by the square root of its weight.
///
/// Stated once for the whole family. Sharing it is what makes "the penalty
/// applies to raw-scale coefficients, and the intercept is recovered from the
/// centering" one definition rather than one per estimator.
pub(super) fn preprocess(
    data: &MatrixView<'_>,
    targets: &[f32],
    sample_weights: Option<&SampleWeights>,
    fit_intercept: bool,
) -> PreprocessedDense {
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
    let weight_sqrts = sample_weights.map(|weights| {
        weights
            .as_slice()
            .iter()
            .map(|&weight| f64::from(weight).sqrt())
            .collect::<Vec<_>>()
    });
    let row_scale = |row: usize| weight_sqrts.as_ref().map_or(1.0, |weights| weights[row]);
    let mut matrix_values = Vec::with_capacity(rows * columns);
    for (column, &mean) in feature_means.iter().enumerate() {
        for row in 0..rows {
            let value = data.as_slice()[row * columns + column];
            matrix_values.push(row_scale(row) * (f64::from(value) - mean));
        }
    }
    let target_vector = targets
        .iter()
        .enumerate()
        .map(|(row, &target)| row_scale(row) * (f64::from(target) - target_mean))
        .collect::<Vec<_>>();
    PreprocessedDense {
        matrix: matrix_values,
        targets: target_vector,
        rows,
        columns,
        feature_means,
        target_mean,
    }
}

fn sample_weight(sample_weights: Option<&SampleWeights>, row: usize) -> f64 {
    sample_weights.map_or(1.0, |weights| f64::from(weights.as_slice()[row]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::numeric::OwnedRng;

    /// The buffer layout `coordinate_descent` slices by column, pinned exactly.
    ///
    /// This is the one place in the crate where two components agree on a
    /// memory layout without a type enforcing it: `preprocess` writes the
    /// design and `coordinate_descent` reads `column * rows .. (column + 1) *
    /// rows` out of it. Disagreeing would not fail to compile — it would
    /// transpose the design and quietly fit a different model, on estimators
    /// with frozen fixtures. A non-square shape is deliberate: on a square one
    /// a transposed read is still in bounds and still plausible.
    #[test]
    fn preprocess_writes_a_dense_column_major_buffer_the_backend_reads_the_same_way() {
        // 3 rows, 2 columns, distinct in both directions.
        let data = DenseMatrix::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        let targets = [1.0_f32, 2.0, 3.0];
        let preprocessed = preprocess(&data.as_view(), &targets, None, false);

        assert_eq!(preprocessed.rows, 3);
        assert_eq!(preprocessed.columns, 2);
        // Exactly `rows * columns` values and no padding, column by column.
        assert_eq!(preprocessed.matrix, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);

        // And the backend view agrees with that indexing, rather than the
        // transposed one. Asserting both halves matters: the first pins what
        // FerricML writes, the second pins what the dependency reads out of it.
        let view = preprocessed.matrix_ref();
        assert_eq!(view.nrows(), 3);
        assert_eq!(view.ncols(), 2);
        for row in 0..3 {
            for column in 0..2 {
                assert_eq!(
                    view[(row, column)],
                    preprocessed.matrix[column * 3 + row],
                    "element ({row}, {column}) is not where the column-major slice puts it"
                );
                assert_eq!(
                    view[(row, column)],
                    f64::from(data.as_slice()[row * 2 + column])
                );
            }
        }
    }

    /// A duplicated column is exactly rank-deficient, and the answer is a
    /// least-squares answer anyway.
    ///
    /// This is a defect-class sweep, not an example. The previous backend
    /// returned, for this shape, a factorization that does not reconstruct its
    /// own input — measured at 55 of 300 draws at 64x3 and 146 of 300 at
    /// 1024x6, worst relative reconstruction error 18.5 — and the coefficients
    /// it produced were consequently not least-squares solutions at all. No
    /// single fixture would have caught that, because whether a draw is
    /// corrupted depends on the draw.
    ///
    /// Three properties are asserted per draw, and each fails independently:
    /// the normal-equation gradient must vanish (it *is* a least-squares
    /// solution), the reported rank must be the true one, and the two identical
    /// columns must split their effect evenly (the minimum-norm choice among
    /// the infinitely many solutions).
    #[test]
    fn exactly_rank_deficient_tall_designs_get_a_minimum_norm_least_squares_answer() {
        let mut rng = OwnedRng::new(0x5EED_1234);
        let mut worst_gradient = 0.0_f64;
        let mut worst_split = 0.0_f64;
        for &(rows, columns) in &[(64_usize, 3_usize), (256, 4), (300, 6)] {
            for _ in 0..25 {
                // Independent columns, then a final column duplicating the
                // first exactly, so the true rank is `columns - 1`.
                let mut values = vec![0.0_f32; rows * columns];
                for row in 0..rows {
                    for column in 0..columns - 1 {
                        values[row * columns + column] = (rng.unit_f64() * 2.0 - 1.0) as f32;
                    }
                    values[row * columns + columns - 1] = values[row * columns];
                }
                let targets = (0..rows)
                    .map(|_| (rng.unit_f64() * 2.0 - 1.0) as f32)
                    .collect::<Vec<f32>>();
                let data = DenseMatrix::new(values.clone(), rows, columns).unwrap();
                let fit = fit_dense(&data.as_view(), &targets, None, false, 0.0).unwrap();

                assert_eq!(
                    fit.rank,
                    columns - 1,
                    "a duplicated column leaves rank {} at {rows}x{columns}",
                    columns - 1
                );

                // ‖Xᵀ(Xb − y)‖∞, scaled by ‖X‖∞‖y‖∞ so the bound is unitless.
                let residual = (0..rows)
                    .map(|row| {
                        let predicted = sum_in_order((0..columns).map(|column| {
                            f64::from(values[row * columns + column]) * fit.coefficients[column]
                        }));
                        predicted - f64::from(targets[row])
                    })
                    .collect::<Vec<_>>();
                let scale = values
                    .iter()
                    .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()))
                    * targets
                        .iter()
                        .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()))
                    * rows as f64;
                for column in 0..columns {
                    let gradient = sum_in_order(
                        (0..rows)
                            .map(|row| f64::from(values[row * columns + column]) * residual[row]),
                    );
                    worst_gradient = worst_gradient.max(gradient.abs() / scale);
                }
                let split = (fit.coefficients[0] - fit.coefficients[columns - 1]).abs();
                worst_split = worst_split.max(split);
            }
        }
        assert!(
            worst_gradient < 1.0e-12,
            "the fit is not a least-squares solution: worst scaled normal-equation \
             gradient {worst_gradient:e}"
        );
        assert!(
            worst_split < 1.0e-9,
            "duplicated columns did not split evenly, so the solution is not the \
             minimum-norm one: worst gap {worst_split:e}"
        );
    }

    /// The rank cutoff has margin on the design the reference suite freezes.
    ///
    /// `LinearRegression::rank()` is public API and a float comparison decides
    /// it, so the interesting number is not that the rank is `1` but how far
    /// the vanishing singular value sits from the floor that rounds it away. A
    /// backend that left it a factor of two below the cutoff would pass the
    /// rank assertion today and fail it on the next dependency bump.
    #[test]
    fn the_vanishing_singular_value_clears_the_rank_floor_with_margin() {
        pin_sequential();
        // The 3x2 design `tests/reference_semantics.rs` pins: column 1 is twice
        // column 0, so σ₂ is exactly zero in exact arithmetic.
        let data = DenseMatrix::new(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2).unwrap();
        let targets = [1.0_f32, 2.0, 3.0];
        let preprocessed = preprocess(&data.as_view(), &targets, None, false);
        let svd = preprocessed.matrix_ref().thin_svd().unwrap();
        let singular = svd.S();
        let singular = singular.column_vector();
        let largest = singular.iter().copied().fold(0.0_f64, f64::max);
        let floor = f64::EPSILON * 3.0 * largest;
        assert!(
            singular[1] < floor / 4.0,
            "σ₂ = {} is within a factor of four of the rank floor {floor}, which \
             leaves no margin for the comparison `rank()` reports",
            singular[1]
        );
        assert!(
            singular[0] > floor * 1.0e12,
            "σ₁ = {} does not clear the floor by a wide margin",
            singular[0]
        );
    }
}
