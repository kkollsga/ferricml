//! Every iterative solver refuses an exhausted budget rather than returning
//! its last iterate.
//!
//! The rule is stated once, in the changelog entry for
//! `ModelError::SolverDidNotConverge`, as a claim about *every* iterative
//! solver: reported "when an iterative solver exhausts `max_iter` … instead of
//! returning the last iterate as though it were a fitted model." It was a claim
//! about the crate with no mechanism behind it, and it was false twice — first
//! for `PlattCalibrator`, then for `LogisticRegression` on both of its target
//! shapes, each found by reading the loop rather than by anything failing.
//!
//! This file is the mechanism. It enumerates every fitting entry point the
//! public API gives a caller a `max_iter` and a `tol` for, starves each of them
//! on a problem one step cannot solve, and requires the refusal. A new
//! estimator with an iteration budget is one row here, and a solver that
//! regresses to returning its last iterate fails on the row it already has.
//!
//! # What this file cannot decide
//!
//! Only that a refusal happens — never that the *acceptance* is right. The two
//! Newton solvers here accept an exhausted budget when the Newton decrement
//! certifies the iterate is at the minimum, because refusing on plain
//! exhaustion would convert a large region of correct fits into spurious
//! errors; a rule good enough to satisfy this file would be satisfied by
//! exactly that worse defect. Which quantity certifies an exhausted budget is
//! settled by measurement, and pinned by the region tests that live beside each
//! solver. This file only holds the floor those tests stand on.

use ferricml::api::ModelError;
use ferricml::calibration::{PlattCalibrator, PlattParams};
use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LogisticRegression, LogisticRegressionParams,
    LogisticSolver,
};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseError, PairwiseLinearRanker, PairwiseLinearRankerParams,
    PairwiseObservation,
};

/// A budget no solver can meet, and a tolerance none can meet in one step.
const STARVED_ITERATIONS: usize = 1;
const UNREACHABLE_TOLERANCE: f32 = 1.0e-12;

/// A design with genuine structure in both columns, so a single step of any of
/// these solvers lands nowhere near the minimum.
fn design() -> DenseMatrix {
    let mut values = Vec::with_capacity(80);
    for row in 0..40_usize {
        let step = row as f32 * 0.25 - 5.0;
        values.push(step);
        values.push(step.mul_add(step, -1.0) * 0.1);
    }
    DenseMatrix::new(values, 40, 2).expect("fixture shape")
}

fn binary_targets() -> BinaryTargets {
    BinaryTargets::new((0..40).map(|row| u8::from(row >= 20)).collect()).expect("both classes")
}

fn class_targets() -> ClassTargets {
    ClassTargets::new((0..40).map(|row| (row % 3) as u8).collect()).expect("three classes")
}

fn regression_targets() -> RegressionTargets {
    RegressionTargets::new(
        (0..40)
            .map(|row| (row as f32).mul_add(0.7, -3.0).sin() * 4.0)
            .collect(),
    )
    .expect("finite targets")
}

fn weights() -> SampleWeights {
    SampleWeights::new(
        (0..40)
            .map(|row| if row % 3 == 0 { 3.0 } else { 0.5 })
            .collect(),
    )
    .expect("positive weights")
}

/// One row of the contract: starve the solver, then fit it properly.
struct Row {
    name: &'static str,
    /// Runs the entry point under a one-iteration budget.
    starved: fn() -> Result<(), ModelError>,
    /// Runs the same entry point under its default budget.
    generous: fn() -> Result<(), ModelError>,
}

fn logistic(solver: LogisticSolver) -> LogisticRegressionParams {
    LogisticRegressionParams::default().with_solver(solver)
}

fn starve(params: LogisticRegressionParams) -> LogisticRegressionParams {
    params
        .with_max_iter(STARVED_ITERATIONS)
        .with_tol(UNREACHABLE_TOLERANCE)
}

fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "LogisticRegression::fit under Newton",
            starved: || {
                LogisticRegression::fit(
                    &design().as_view(),
                    &binary_targets(),
                    starve(logistic(LogisticSolver::Newton)),
                )
                .map(drop)
            },
            generous: || {
                LogisticRegression::fit(
                    &design().as_view(),
                    &binary_targets(),
                    logistic(LogisticSolver::Newton),
                )
                .map(drop)
            },
        },
        Row {
            name: "LogisticRegression::fit under L-BFGS",
            starved: || {
                LogisticRegression::fit(
                    &design().as_view(),
                    &binary_targets(),
                    starve(logistic(LogisticSolver::Lbfgs)),
                )
                .map(drop)
            },
            generous: || {
                LogisticRegression::fit(
                    &design().as_view(),
                    &binary_targets(),
                    logistic(LogisticSolver::Lbfgs),
                )
                .map(drop)
            },
        },
        Row {
            name: "LogisticRegression::fit_weighted under Newton",
            starved: || {
                LogisticRegression::fit_weighted(
                    &design().as_view(),
                    &binary_targets(),
                    &weights(),
                    starve(logistic(LogisticSolver::Newton)),
                )
                .map(drop)
            },
            generous: || {
                LogisticRegression::fit_weighted(
                    &design().as_view(),
                    &binary_targets(),
                    &weights(),
                    logistic(LogisticSolver::Newton),
                )
                .map(drop)
            },
        },
        Row {
            name: "LogisticRegression::fit_multiclass under Newton",
            starved: || {
                LogisticRegression::fit_multiclass(
                    &design().as_view(),
                    &class_targets(),
                    starve(logistic(LogisticSolver::Newton)),
                )
                .map(drop)
            },
            generous: || {
                LogisticRegression::fit_multiclass(
                    &design().as_view(),
                    &class_targets(),
                    logistic(LogisticSolver::Newton),
                )
                .map(drop)
            },
        },
        Row {
            name: "LogisticRegression::fit_multiclass under L-BFGS",
            starved: || {
                LogisticRegression::fit_multiclass(
                    &design().as_view(),
                    &class_targets(),
                    starve(logistic(LogisticSolver::Lbfgs)),
                )
                .map(drop)
            },
            generous: || {
                LogisticRegression::fit_multiclass(
                    &design().as_view(),
                    &class_targets(),
                    logistic(LogisticSolver::Lbfgs),
                )
                .map(drop)
            },
        },
        Row {
            name: "LogisticRegression::fit_multiclass_weighted under Newton",
            starved: || {
                LogisticRegression::fit_multiclass_weighted(
                    &design().as_view(),
                    &class_targets(),
                    &weights(),
                    starve(logistic(LogisticSolver::Newton)),
                )
                .map(drop)
            },
            generous: || {
                LogisticRegression::fit_multiclass_weighted(
                    &design().as_view(),
                    &class_targets(),
                    &weights(),
                    logistic(LogisticSolver::Newton),
                )
                .map(drop)
            },
        },
        Row {
            name: "Lasso::fit",
            starved: || {
                Lasso::fit(
                    &design().as_view(),
                    &regression_targets(),
                    LassoParams::default()
                        .with_alpha(0.01)
                        .with_max_iter(STARVED_ITERATIONS)
                        .with_tol(UNREACHABLE_TOLERANCE),
                )
                .map(drop)
            },
            generous: || {
                Lasso::fit(
                    &design().as_view(),
                    &regression_targets(),
                    LassoParams::default().with_alpha(0.01),
                )
                .map(drop)
            },
        },
        Row {
            name: "ElasticNet::fit",
            starved: || {
                ElasticNet::fit(
                    &design().as_view(),
                    &regression_targets(),
                    ElasticNetParams::default()
                        .with_alpha(0.01)
                        .with_l1_ratio(0.5)
                        .with_max_iter(STARVED_ITERATIONS)
                        .with_tol(UNREACHABLE_TOLERANCE),
                )
                .map(drop)
            },
            generous: || {
                ElasticNet::fit(
                    &design().as_view(),
                    &regression_targets(),
                    ElasticNetParams::default()
                        .with_alpha(0.01)
                        .with_l1_ratio(0.5),
                )
                .map(drop)
            },
        },
        Row {
            name: "PlattCalibrator::fit",
            starved: || {
                let scores: Vec<f32> = (0..40).map(|row| row as f32 * 0.25 - 5.0).collect();
                PlattCalibrator::fit(
                    &scores,
                    &binary_targets(),
                    PlattParams::default()
                        .with_max_iter(STARVED_ITERATIONS)
                        .with_tol(UNREACHABLE_TOLERANCE),
                )
                .map(drop)
            },
            generous: || {
                let scores: Vec<f32> = (0..40).map(|row| row as f32 * 0.25 - 5.0).collect();
                PlattCalibrator::fit(&scores, &binary_targets(), PlattParams::default()).map(drop)
            },
        },
        Row {
            name: "PairwiseLinearRanker::fit",
            starved: || {
                fit_ranker(
                    PairwiseLinearRankerParams::default()
                        .with_max_iter(STARVED_ITERATIONS)
                        .with_tol(UNREACHABLE_TOLERANCE),
                )
            },
            generous: || fit_ranker(PairwiseLinearRankerParams::default()),
        },
    ]
}

/// The ranker reaches a logistic fit through its own error type, so its row
/// unwraps that wrapper rather than skipping the estimator.
fn fit_ranker(params: PairwiseLinearRankerParams) -> Result<(), ModelError> {
    let items = design();
    let mut observations = Vec::new();
    for left in 0..12_usize {
        for right in (left + 1)..12 {
            observations.push(
                PairwiseObservation::new(
                    PairIndex::new(left, right).expect("distinct"),
                    if (left + right) % 3 == 0 {
                        PairOutcome::RightPreferred
                    } else {
                        PairOutcome::LeftPreferred
                    },
                    1.0,
                )
                .expect("unit weight"),
            );
        }
    }
    match PairwiseLinearRanker::fit(&items.as_view(), &observations, params) {
        Ok(_) => Ok(()),
        Err(PairwiseError::Model(error)) => Err(error),
        Err(other) => panic!("the ranker refused for an unrelated reason: {other:?}"),
    }
}

/// The contract, over every solver the public API budgets.
#[test]
fn no_iterative_solver_returns_an_unconverged_iterate() {
    let rows = rows();
    for row in &rows {
        assert_eq!(
            (row.starved)(),
            Err(ModelError::SolverDidNotConverge {
                iterations: STARVED_ITERATIONS
            }),
            "{} returned a model from a one-iteration budget",
            row.name
        );
    }

    // And the same call succeeds when it is given a budget, so each refusal
    // above is about the budget rather than about the problem. Without this the
    // whole file would pass on an estimator that had simply stopped working.
    for row in &rows {
        assert_eq!(
            (row.generous)(),
            Ok(()),
            "{} could not fit its own problem at the default budget",
            row.name
        );
    }
}

/// The enumeration is the contract, so its size is asserted.
///
/// A solver dropped from the list would otherwise weaken this file silently,
/// which is the exact failure mode — an unenforced general claim — that the
/// file exists to end.
#[test]
fn every_budgeted_entry_point_in_the_public_api_is_enumerated() {
    // Six logistic entry points (two solvers over the binary and multinomial
    // shapes, plus the two weighted binary and multinomial Newton paths), the
    // two coordinate-descent regressors, the Platt calibrator, and the pairwise
    // ranker that reaches a logistic fit through its own error type.
    assert_eq!(rows().len(), 10);
    let mut names: Vec<&str> = rows().iter().map(|row| row.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), 10, "two rows share a name");
}
