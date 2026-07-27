use crate::api::ModelError;
use crate::data::{MatrixView, SampleWeights};
use crate::numeric::sum_in_order;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_transpose_on_the_left_in_place_scratch,
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
};
use faer::linalg::solvers::Solve;
use faer::{Conj, Mat, MatRef, Par, Side};

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
    fit_dense_by(
        data,
        targets,
        sample_weights,
        fit_intercept,
        tolerance,
        reduce_before_decomposing(data.rows(), data.columns()),
    )
}

/// [`fit_dense`] with the decomposition path named rather than derived.
///
/// The choice is a *performance* choice between two routes to the same
/// contract, so the tests have to be able to run either one on any design and
/// compare — a battery that only ever exercises whichever path the shape guard
/// picks cannot tell agreement from a shape guard that never fires.
fn fit_dense_by(
    data: &MatrixView<'_>,
    targets: &[f32],
    sample_weights: Option<&SampleWeights>,
    fit_intercept: bool,
    tolerance: f32,
    reduce: bool,
) -> Result<LeastSquaresFit, ModelError> {
    debug_assert_eq!(data.rows(), targets.len());
    debug_assert!(sample_weights.is_none_or(|weights| weights.len() == data.rows()));

    pin_sequential();
    let preprocessed = preprocess(data, targets, sample_weights, fit_intercept);
    let rows = data.rows();
    let columns = data.columns();
    let (coefficients, rank) = if reduce {
        let (upper, projected) = reduce_by_qr(&preprocessed)?;
        solve_min_norm(upper.as_ref(), &projected, rows, columns, tolerance)?
    } else {
        solve_min_norm(
            preprocessed.matrix_ref(),
            &preprocessed.targets,
            rows,
            columns,
            tolerance,
        )?
    };
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

/// Whether a design is tall enough that reducing it before the SVD pays.
///
/// `rows >= 1.25 * columns`, written in integers so no rounding decides it.
///
/// **This is a tuning constant, and an earlier attempt at this reduction
/// deliberately refused to have one.** That attempt gated on `rows > columns`
/// instead, on two grounds: a ratio threshold reintroduces a shape-dependent
/// cliff, and — under the previous backend — the reduced path was also the
/// *more accurate* one, because a smaller matrix reached less of an upstream
/// SVD defect on exactly rank-deficient input. Under that second condition a
/// shape just below any constant would silently receive the worse
/// decomposition, which is not a trade a speed argument can buy.
///
/// **The second ground no longer holds, and that is measured rather than
/// assumed.** That backend is gone and the defect it worked around went with
/// it. Across well-conditioned, ill-conditioned and exactly rank-deficient
/// designs at `rows == columns`, `rows == columns + 1` and well above, weighted
/// and unweighted, with and without an intercept, the two paths report the
/// *same rank every time*, and their coefficients agree to `1e-15` relative on
/// tall designs (worst `2.0e-14`) and `9.2e-14` on near-square ones. Only a
/// design conditioned at about `1e8` separates them further, at `1.1e-8` — which
/// is `cond * eps`, the spread any two backward-stable solves are entitled to,
/// and not a quality ordering: the *fit* still agrees there, with a worst scaled
/// normal-equation gradient of `1.8e-14` over the whole corpus. See
/// `the_two_paths_agree_on_well_conditioned_ill_conditioned_and_deficient_designs`.
///
/// So neither path is the more accurate one, and what is left below the
/// constant is a pure speed question — where the reduction *loses*. At
/// `rows == columns` the QR is redundant work ahead of a decomposition that was
/// already square, and costs a stable 10 percent: 600x600, 74.8 ms direct
/// against 82.6 ms reduced; 1000x1000, 314.9 ms against 350.1 ms. Sweeping
/// `rows` at `columns == 300` puts the break-even at `rows/columns ≈ 1.20`
/// (13.33 ms direct against 13.36 ms reduced), with the reduction 10 percent
/// behind at 1.00 and 5 percent ahead by 1.30.
///
/// `1.25` is therefore one notch on the safe side of break-even, and it is not
/// bought at the cost of a saving: at the break-even ratio itself there is
/// nothing to save, and by `1.25` the reduction is already 3 percent ahead
/// (13.81 ms against 13.45 ms). The one honest debit is that break-even drifts
/// with `columns` — at `columns == 100` it sits nearer `1.5`, so a design
/// between `1.25` and `1.5` there pays up to 4 percent. That is 4 percent of
/// 1.2 ms. Making the constant a function of `columns` to recover 0.05 ms would
/// be a worse trade than the cliff it removes.
///
/// The first ground — that a threshold is a cliff — stands on its own, but it
/// is now a cliff between two answers of equal quality, so it is a cliff in
/// *cost* only, and paying 10 percent on every square design to avoid it is the
/// worse deal.
///
/// The guard is therefore defensible on speed alone in a way it was not before.
/// Do not restore the unconditional `rows > columns` rule, and do not move this
/// constant, without re-running the agreement measurement above — it is that
/// measurement, not the timings, that licenses having a constant at all.
fn reduce_before_decomposing(rows: usize, columns: usize) -> bool {
    4 * rows >= 5 * columns
}

/// Reduces a tall design to the `p`-by-`p` `R` of its thin QR, and the target
/// to `Qᵀb`.
///
/// This is the first half of Chan's R-SVD. `A = QR` with `Q` an isometry, so
/// `R` and `A` have identical singular values — the rank cutoff is applied to
/// the same numbers either way — and `(QR)⁺ = R⁺Qᵀ`, so the minimum-norm
/// least-squares solution of `Rx = Qᵀb` *is* the minimum-norm least-squares
/// solution of `Ax = b`. Rank deficiency needs no special case: an unpivoted
/// QR factors a rank-deficient `A` exactly all the same.
///
/// **`Q` is never materialized, and that is the whole reduction.** The obvious
/// spelling — form the thin `Q`, then multiply `Qᵀb` — was measured against
/// applying the stored Householder sequence to the single right-hand side, and
/// it does not merely cost more, it *gives back the entire saving*: 50000x50
/// goes 25.4 ms to 38.0 ms against a direct SVD's 38.3, and 50000x300 goes
/// 302 ms to 620 ms against 640. Forming an `n`-by-`p` `Q` is the same order of
/// work as decomposing the `n`-by-`p` design, so a reduction that materializes
/// it has reduced nothing. Applying the sequence to one column is `O(np)`.
///
/// Only the first `columns` entries of `Qᵀb` are kept, because only the thin
/// `Q` multiplies `R`; the tail is the residual component orthogonal to the
/// column space and contributes nothing to the solution.
///
/// **Requires at least as many rows as columns.** An underdetermined design has
/// no square `R` to reduce to, and the minimum-norm solution it needs lives
/// precisely in the part a square `R` would discard. Nothing here checks that
/// at runtime, because nothing has to: `reduce_before_decomposing` cannot
/// return `true` below `rows == 1.25 * columns`, and
/// `the_shape_guard_never_routes_an_underdetermined_design_into_the_reduction`
/// pins that implication rather than leaving it to be re-derived.
fn reduce_by_qr(preprocessed: &PreprocessedDense) -> Result<(Mat<f64>, Vec<f64>), ModelError> {
    let rows = preprocessed.rows;
    let columns = preprocessed.columns;
    debug_assert!(
        rows >= columns,
        "the R-SVD reduction was handed an underdetermined design, whose \
         minimum-norm solution lives in the part `R` discards"
    );
    let qr = preprocessed.matrix_ref().qr();

    // `R` arrives already square and already upper triangular — pinned by
    // `the_backends_qr_returns_a_square_upper_triangular_r`, because it is a
    // property of a dependency that this code would silently misuse if it ever
    // changed. It is nonetheless *copied* rather than decomposed where it lies:
    // the borrowed view is a `columns`-by-`columns` window on storage whose
    // column stride is the design's `rows`, so on a 50000-by-50 fit consecutive
    // columns of a 50-by-50 matrix sit 400 KB apart, and the SVD would spend
    // itself on cache misses. The copy is `p²` against the SVD's `p³`.
    let packed = qr.R();
    let upper = Mat::from_fn(columns, columns, |row, column| packed[(row, column)]);

    let basis = qr.Q_basis();
    let coefficients = qr.Q_coeff();
    let mut rhs = Mat::from_fn(rows, 1, |row, _| preprocessed.targets[row]);
    let mut buffer = MemBuffer::new(
        apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(
            rows,
            coefficients.nrows(),
            1,
        ),
    );
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        basis,
        coefficients,
        Conj::No,
        rhs.as_mut(),
        Par::Seq,
        MemStack::new(&mut buffer),
    );
    let projected = (0..columns).map(|row| rhs[(row, 0)]).collect::<Vec<_>>();
    if projected.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::LinearSolveFailed);
    }
    Ok((upper, projected))
}

/// Minimum-norm least-squares solution of `matrix * x ≈ targets`, and the rank.
///
/// `rows` and `columns` describe the **design the caller is solving**, which is
/// not always `matrix`'s own shape: on the reduced path `matrix` is the square
/// `R`, while the numerical floor under the rank cutoff scales with the
/// design's larger dimension and has to keep scaling with it however the solve
/// reaches the singular values. Passing `matrix`'s dimensions here would shrink
/// the floor by the aspect ratio — a factor of 1000 on a 50000-by-50 design —
/// and silently change the rank a tall design reports.
fn solve_min_norm(
    matrix: MatRef<'_, f64>,
    targets: &[f64],
    rows: usize,
    columns: usize,
    tolerance: f32,
) -> Result<(Vec<f64>, usize), ModelError> {
    debug_assert_eq!(matrix.nrows(), targets.len());
    let svd = matrix
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
                sum_in_order((0..matrix.nrows()).map(|row| left[(row, index)] * targets[row]))
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
    Ok((coefficients, rank))
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

    /// How a corpus design is conditioned, which is what decides how closely
    /// two arithmetically different routes to the same answer may be asked to
    /// agree.
    #[derive(Clone, Copy, Debug)]
    enum Conditioning {
        Well,
        /// Columns scaled geometrically down to `1e-8`, so the design's
        /// condition number is about `1e8` and a coefficient is determined to
        /// only about `cond * eps` by *any* backward-stable solve.
        Ill,
        /// The last column duplicates the first exactly, so the true rank is
        /// `columns - 1` and the solution is one point in an affine set.
        Deficient,
    }

    /// Builds one corpus design and its targets, deterministically.
    fn corpus_design(
        rows: usize,
        columns: usize,
        conditioning: Conditioning,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut rng = OwnedRng::new(seed);
        let mut values = vec![0.0_f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                values[row * columns + column] = (rng.unit_f64() * 2.0 - 1.0) as f32;
            }
        }
        match conditioning {
            Conditioning::Well => {}
            Conditioning::Ill => {
                for column in 0..columns {
                    let scale =
                        10.0_f64.powf(-8.0 * column as f64 / (columns - 1).max(1) as f64) as f32;
                    for row in 0..rows {
                        values[row * columns + column] *= scale;
                    }
                }
            }
            Conditioning::Deficient => {
                for row in 0..rows {
                    values[row * columns + columns - 1] = values[row * columns];
                }
            }
        }
        let targets = (0..rows)
            .map(|_| (rng.unit_f64() * 2.0 - 1.0) as f32)
            .collect::<Vec<f32>>();
        (values, targets)
    }

    /// `‖Xᵀ(Xb − y)‖∞`, weighted, and scaled by `‖X‖∞‖y‖∞·rows` so the bound is
    /// unitless. Zero exactly when `b` solves the normal equations — that is,
    /// exactly when `b` minimises the residual.
    fn scaled_normal_equation_gradient(
        values: &[f32],
        targets: &[f32],
        rows: usize,
        columns: usize,
        coefficients: &[f64],
        intercept: f64,
        weights: Option<&[f32]>,
    ) -> f64 {
        let weight = |row: usize| weights.map_or(1.0, |w| f64::from(w[row]));
        let residual = (0..rows)
            .map(|row| {
                let predicted = sum_in_order((0..columns).map(|column| {
                    f64::from(values[row * columns + column]) * coefficients[column]
                })) + intercept;
                weight(row) * (predicted - f64::from(targets[row]))
            })
            .collect::<Vec<_>>();
        let scale = values
            .iter()
            .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()))
            * targets
                .iter()
                .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()))
            * rows as f64;
        let mut worst = 0.0_f64;
        for column in 0..columns {
            let gradient = sum_in_order(
                (0..rows).map(|row| f64::from(values[row * columns + column]) * residual[row]),
            );
            worst = worst.max(gradient.abs() / scale);
        }
        worst
    }

    /// The shape guard implies the reduction's precondition, for every shape.
    ///
    /// [`reduce_by_qr`] needs `rows >= columns` and does not enforce it, on the
    /// grounds that `4 * rows >= 5 * columns` already implies it. That is a
    /// coupling between two conditions written in two places, and the kind that
    /// survives a plausible edit: widening the guard to `rows > columns` or to
    /// `rows >= columns` keeps it true, while relaxing it to admit a square-ish
    /// or underdetermined design silently hands the reduction a case whose
    /// minimum-norm solution it cannot represent. So the implication is checked
    /// here over every small shape rather than argued about at the call site.
    #[test]
    fn the_shape_guard_never_routes_an_underdetermined_design_into_the_reduction() {
        let mut reduced_shapes = 0_usize;
        for columns in 1..48_usize {
            for rows in 0..96_usize {
                if reduce_before_decomposing(rows, columns) {
                    reduced_shapes += 1;
                    assert!(
                        rows >= columns,
                        "the guard routes a {rows}x{columns} design into a reduction \
                         that discards the part its minimum-norm solution lives in"
                    );
                }
            }
        }
        assert!(
            reduced_shapes > 1000,
            "only {reduced_shapes} shapes reach the reduction at all, so this \
             implication is close to vacuous"
        );
    }

    /// The backend's `R` is square and upper triangular, which
    /// [`reduce_by_qr`] copies without masking.
    ///
    /// This is a dependency's property that FerricML relies on and cannot see.
    /// A `QR` implementation is equally entitled to hand back the *packed*
    /// factor — `rows`-by-`columns`, with the Householder vectors living in the
    /// strict lower triangle where the zeros would be — and this crate's copy
    /// would then carry those vectors into the SVD and fit a different model
    /// without failing anything structural. Masking the lower triangle
    /// defensively would hide that behind a branch no test could reach, so the
    /// assumption is pinned here instead, where a dependency bump breaks it
    /// loudly.
    ///
    /// The shape is asserted too: a `rows`-by-`columns` `R` would make the copy
    /// a silent truncation rather than a copy.
    #[test]
    fn the_backends_qr_returns_a_square_upper_triangular_r() {
        pin_sequential();
        // Tall and non-square, with no zeros of its own anywhere.
        let rows = 17_usize;
        let columns = 5_usize;
        let mut rng = OwnedRng::new(0xD1CE_0007);
        let values = (0..rows * columns)
            .map(|_| (rng.unit_f64() * 2.0 - 1.0) as f32 + 1.5)
            .collect::<Vec<f32>>();
        let data = DenseMatrix::new(values, rows, columns).unwrap();
        let targets = vec![1.0_f32; rows];
        let preprocessed = preprocess(&data.as_view(), &targets, None, false);
        let factor = preprocessed.matrix_ref().qr();
        let upper = factor.R();

        assert_eq!(
            (upper.nrows(), upper.ncols()),
            (columns, columns),
            "the backend's `R` is no longer square, so `reduce_by_qr` is \
             truncating it rather than copying it"
        );
        for row in 0..columns {
            for column in 0..row {
                assert_eq!(
                    upper[(row, column)],
                    0.0,
                    "the backend's `R` has {} below the diagonal at ({row}, \
                     {column}), so it is the packed factor and `reduce_by_qr` is \
                     feeding Householder vectors to the SVD",
                    upper[(row, column)]
                );
            }
            assert_ne!(
                upper[(row, row)],
                0.0,
                "the backend's `R` has a zero diagonal entry at {row} on a design \
                 with no zeros in it, so this is not the factor it claims to be"
            );
        }
    }

    /// The two decomposition routes answer the same question, on every kind of
    /// design the guard can route either way.
    ///
    /// This is the measurement the shape guard on
    /// [`reduce_before_decomposing`] rests on, and it is deliberately a
    /// *measurement* rather than an inherited claim. The predecessor of this
    /// reduction refused a ratio threshold partly because, under the previous
    /// backend, the reduced path was the more accurate one, so any design below
    /// the constant would have been silently downgraded. Whether that is still
    /// true is not a matter of reasoning about `Q` being an isometry — it is a
    /// matter of what the two paths actually return — so it is asserted here,
    /// across well-conditioned, ill-conditioned and exactly rank-deficient
    /// designs, at `rows == columns`, `rows == columns + 1` and well above,
    /// weighted and unweighted, with and without an intercept.
    ///
    /// Ranks must agree *exactly*: the rank rule is public API and reads the
    /// same singular values either way, because `A = QR` with `Q` an isometry
    /// gives `R` the singular values of `A`.
    ///
    /// Coefficients are compared relative to the coefficient scale. The bound
    /// is loosened for an ill-conditioned design, and that is not a concession:
    /// at a condition number of `1e8`, `cond * eps ≈ 1e-8` is the spread any two
    /// backward-stable solves are entitled to, and demanding better would be
    /// asserting a coincidence. The *fit* is held to the tight bound there
    /// instead, through the normal-equation gradient, which is what actually has
    /// to agree.
    #[test]
    fn the_two_paths_agree_on_well_conditioned_ill_conditioned_and_deficient_designs() {
        let shapes = [
            (500_usize, 8_usize, Conditioning::Well),
            (400, 6, Conditioning::Ill),
            (300, 5, Conditioning::Deficient),
            (40, 40, Conditioning::Well),
            (41, 40, Conditioning::Well),
            (40, 40, Conditioning::Ill),
            (41, 40, Conditioning::Deficient),
            (60, 12, Conditioning::Deficient),
            (200, 3, Conditioning::Ill),
            (129, 128, Conditioning::Well),
        ];
        let mut worst_well = 0.0_f64;
        let mut worst_ill = 0.0_f64;
        let mut worst_gradient = 0.0_f64;
        let mut seed = 0x00A1_1CE5_u64;
        for &(rows, columns, conditioning) in &shapes {
            for &fit_intercept in &[false, true] {
                for &weighted in &[false, true] {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let (values, targets) = corpus_design(rows, columns, conditioning, seed);
                    let data = DenseMatrix::new(values.clone(), rows, columns).unwrap();
                    let weight_values = (0..rows)
                        .map(|row| 0.25 + (row % 7) as f32 * 0.5)
                        .collect::<Vec<f32>>();
                    let weights =
                        weighted.then(|| SampleWeights::new(weight_values.clone()).unwrap());

                    let direct = fit_dense_by(
                        &data.as_view(),
                        &targets,
                        weights.as_ref(),
                        fit_intercept,
                        0.0,
                        false,
                    )
                    .unwrap();
                    let reduced = fit_dense_by(
                        &data.as_view(),
                        &targets,
                        weights.as_ref(),
                        fit_intercept,
                        0.0,
                        true,
                    )
                    .unwrap();

                    assert_eq!(
                        direct.rank, reduced.rank,
                        "the two paths disagree on the rank of a {rows}x{columns} \
                         {conditioning:?} design (intercept {fit_intercept}, weighted \
                         {weighted}): {} against {}",
                        direct.rank, reduced.rank
                    );
                    if matches!(conditioning, Conditioning::Deficient) {
                        assert_eq!(
                            reduced.rank,
                            columns - 1,
                            "a duplicated column leaves rank {} at {rows}x{columns}",
                            columns - 1
                        );
                    }

                    let scale = direct
                        .coefficients
                        .iter()
                        .fold(direct.intercept.abs(), |m, &v| m.max(v.abs()))
                        .max(f64::MIN_POSITIVE);
                    let mut deviation = (direct.intercept - reduced.intercept).abs() / scale;
                    for (&a, &b) in direct.coefficients.iter().zip(&reduced.coefficients) {
                        deviation = deviation.max((a - b).abs() / scale);
                    }
                    match conditioning {
                        Conditioning::Ill => worst_ill = worst_ill.max(deviation),
                        _ => worst_well = worst_well.max(deviation),
                    }

                    for fit in [&direct, &reduced] {
                        worst_gradient = worst_gradient.max(scaled_normal_equation_gradient(
                            &values,
                            &targets,
                            rows,
                            columns,
                            &fit.coefficients,
                            if fit_intercept { fit.intercept } else { 0.0 },
                            weighted.then_some(weight_values.as_slice()),
                        ));
                    }
                }
            }
        }
        assert!(
            worst_well < 1.0e-12,
            "the reduced and direct paths disagree by {worst_well:e} relative on a \
             well-conditioned or exactly rank-deficient design, which is far past the \
             last-bit spread two backward-stable solves are entitled to"
        );
        assert!(
            worst_ill < 1.0e-6,
            "the reduced and direct paths disagree by {worst_ill:e} relative on a \
             design conditioned at about 1e8, past the `cond * eps` spread"
        );
        assert!(
            worst_gradient < 1.0e-12,
            "some fit is not a least-squares solution: worst scaled normal-equation \
             gradient {worst_gradient:e}"
        );
    }

    /// The numerical floor under the rank cutoff scales with the **design's**
    /// rows, however the solve reaches the singular values.
    ///
    /// This is the assertion the predecessor of this reduction found survived
    /// its first mutant. Handing [`solve_min_norm`] the reduced `R`'s own
    /// dimensions instead of the design's shrinks the floor by the aspect
    /// ratio, and no ordinary design notices: an exactly rank-deficient design
    /// puts its vanishing singular value *below both* floors, and a healthy one
    /// puts every singular value above both. Only a design with a singular value
    /// in the band between them separates the two rules, and no such design
    /// occurs by accident — so this one is constructed.
    ///
    /// 2000 rows, two columns, the second exactly twice the first except in one
    /// row where it is one `f32` ulp larger. That row's magnitude is `2^-16`
    /// rather than `1`, which dilutes the perturbation without changing the
    /// design's scale, and lands `σ₂/σ₁` at about `1.6e-14`: about 37 times the
    /// floor `R`'s dimensions would give (`eps * 2`) and about 27 times *below*
    /// the floor the design's give (`eps * 2000`). The design is rank 1, and a
    /// path that reports 2 has taken the wrong floor.
    #[test]
    fn the_rank_floor_scales_with_the_designs_rows_and_not_the_reductions() {
        pin_sequential();
        let rows = 2000_usize;
        let small = (2.0_f64).powi(-16) as f32;
        let mut values = vec![0.0_f32; rows * 2];
        for row in 0..rows {
            let base = if row == rows / 2 { small } else { 1.0_f32 };
            values[row * 2] = base;
            values[row * 2 + 1] = 2.0 * base;
        }
        let perturbed = (rows / 2) * 2 + 1;
        values[perturbed] = f32::from_bits(values[perturbed].to_bits() + 1);
        let data = DenseMatrix::new(values.clone(), rows, 2).unwrap();
        let targets = (0..rows)
            .map(|row| if row % 3 == 0 { 1.0_f32 } else { -0.5 })
            .collect::<Vec<f32>>();

        // The design really does sit in the band, so the assertion below is not
        // vacuous. Both bounds are checked, in units of `σ₁`.
        let preprocessed = preprocess(&data.as_view(), &targets, None, false);
        let (upper, _) = reduce_by_qr(&preprocessed).unwrap();
        let svd = upper.as_ref().thin_svd().unwrap();
        let singular = svd.S();
        let singular = singular.column_vector();
        let largest = singular.iter().copied().fold(0.0_f64, f64::max);
        let ratio = singular[1] / largest;
        assert!(
            ratio > f64::EPSILON * 2.0 * 4.0,
            "σ₂/σ₁ = {ratio:e} does not clear the floor `R`'s own dimensions would \
             give, so the two rules are not being separated"
        );
        assert!(
            ratio < f64::EPSILON * rows as f64 / 4.0,
            "σ₂/σ₁ = {ratio:e} is not below the floor the design's dimensions give, \
             so the two rules are not being separated"
        );

        for reduce in [false, true] {
            let fit = fit_dense_by(&data.as_view(), &targets, None, false, 0.0, reduce).unwrap();
            assert_eq!(
                fit.rank,
                1,
                "the {} path reports rank {} on a design whose second singular value \
                 is below the floor `eps * rows * σ₁`, so it took the reduction's \
                 dimensions rather than the design's",
                if reduce { "reduced" } else { "direct" },
                fit.rank
            );
        }
    }

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
