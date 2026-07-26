//! What `ModelError::SolverDidNotConverge` *says* has to be true wherever it is
//! constructed.
//!
//! [`solver_convergence_contract`](../solver_convergence_contract/index.html)
//! holds the floor under the refusal itself: every budgeted solver reports
//! rather than returning its last iterate. This file holds the floor under the
//! *sentence*. One variant serves several causes, so a message that names one
//! of them is a false statement everywhere the other applies — and a false
//! statement in an error message is not cosmetic, because it is the only thing
//! a caller has to decide what to change.
//!
//! # The defect this file was written against
//!
//! The message read "solver reached max_iter after {iterations} iterations
//! without converging". On a 12x2 logistic fit under L-BFGS with `C = 0.1`,
//! `tol = 1e-12` and `max_iter = 500`, it rendered as *"solver reached max_iter
//! after 9 iterations"*. The budget was never touched: the line search stopped
//! because a `tol` below the objective's numerical resolution is not something
//! a value-comparing search can certify. A caller reading that sentence raises
//! `max_iter`, which is the one parameter that cannot help.
//!
//! The two Newton paths carry the same shape — their backtracking search can
//! find no descending step and break before the budget — so this was never one
//! solver's wording problem.
//!
//! # Why the check is a pair of blocklists
//!
//! One message covers both causes, so the honest wording names neither, and
//! the mechanical form of "names neither" is that neither cause's vocabulary
//! appears. Both lists are applied to *every* refusal, including the ones where
//! one of them would have been true: a single message cannot be true at one
//! site by asserting something false at another. A deliberate rewording has to
//! come back here and say which word it wants and why it is true everywhere.

use ferricml::api::ModelError;
use ferricml::calibration::{PlattCalibrator, PlattParams};
use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets};
use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LogisticRegression, LogisticRegressionParams,
    LogisticSolver,
};

/// Vocabulary that claims the iteration budget ran out.
///
/// False on every refusal that stopped before the budget, which is the defect
/// this file exists to keep fixed.
const BUDGET_CLAIMS: [&str; 6] = [
    "max_iter",
    "budget",
    "exhaust",
    "ran out",
    "limit",
    "iteration cap",
];

/// Vocabulary that claims the solver stalled on numerical resolution.
///
/// The mirror image: true of a collapsed bracket and false of a fit that simply
/// ran out of iterations. A rewording that fixed one direction by breaking the
/// other would be the same defect with the sign flipped.
const STALL_CLAIMS: [&str; 6] = [
    "line search",
    "line-search",
    "bracket",
    "resolution",
    "no further",
    "stall",
];

/// One refusal, with the budget it was given and what it rendered as.
struct Refusal {
    name: &'static str,
    max_iter: usize,
    iterations: usize,
    rendered: String,
}

/// Runs a fitting entry point that must refuse, and records the rendering.
fn refuse(
    name: &'static str,
    max_iter: usize,
    fit: impl FnOnce() -> Result<(), ModelError>,
) -> Refusal {
    let error = fit().expect_err(name);
    let ModelError::SolverDidNotConverge { iterations } = error else {
        panic!("{name} refused for an unrelated reason: {error:?}");
    };
    assert!(
        iterations <= max_iter,
        "{name} reported {iterations} iterations against a budget of {max_iter}, \
         which breaks the bound the variant's documentation rests on"
    );
    Refusal {
        name,
        max_iter,
        iterations,
        rendered: error.to_string(),
    }
}

/// A 12x2 binary problem whose L-BFGS fit reaches its optimum in single-digit
/// iterations, so an unreachable `tol` collapses the line search a long way
/// below any ordinary budget.
fn small_binary() -> (DenseMatrix, BinaryTargets) {
    let values = vec![
        0.0, 0.0, 0.5, 0.2, 0.2, 0.6, 1.0, 0.3, 2.0, 0.1, 1.8, 0.5, 2.2, 0.9, 0.3, 2.0, 0.8, 2.4,
        1.2, 2.2, 1.0, 3.0, 0.1, 1.0,
    ];
    (
        DenseMatrix::new(values, 12, 2).expect("fixture shape"),
        BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]).expect("both classes"),
    )
}

fn small_classes() -> ClassTargets {
    ClassTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0]).expect("three classes")
}

fn small_regression() -> RegressionTargets {
    RegressionTargets::new(vec![
        0.1, 0.6, 0.9, 1.4, 2.1, 2.3, 3.0, 2.4, 3.1, 3.4, 4.0, 1.1,
    ])
    .expect("finite targets")
}

/// The budget genuinely too small to converge in, used for the other arm.
const STARVED: usize = 1;
/// A budget nothing here needs, used to make the stall arm unambiguous.
const GENEROUS: usize = 500;

fn every_refusal() -> Vec<Refusal> {
    let (data, targets) = small_binary();
    vec![
        // The stall arm: a budget of 500 that the solver never approaches.
        refuse(
            "L-BFGS binary, tol below the objective's resolution",
            GENEROUS,
            || {
                LogisticRegression::fit(
                    &data.as_view(),
                    &targets,
                    LogisticRegressionParams::default()
                        .with_solver(LogisticSolver::Lbfgs)
                        .with_c(0.1)
                        .with_max_iter(GENEROUS)
                        .with_tol(1.0e-12),
                )
                .map(drop)
            },
        ),
        // The budget arm, once per construction site in `src/`.
        refuse("L-BFGS binary, starved budget", STARVED, || {
            LogisticRegression::fit(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default()
                    .with_solver(LogisticSolver::Lbfgs)
                    .with_c(1.0e6)
                    .with_max_iter(STARVED)
                    .with_tol(1.0e-12),
            )
            .map(drop)
        }),
        refuse("Newton binary, starved budget", STARVED, || {
            LogisticRegression::fit(
                &data.as_view(),
                &targets,
                LogisticRegressionParams::default()
                    .with_max_iter(STARVED)
                    .with_tol(1.0e-12),
            )
            .map(drop)
        }),
        refuse("Newton multinomial, starved budget", STARVED, || {
            LogisticRegression::fit_multiclass(
                &data.as_view(),
                &small_classes(),
                LogisticRegressionParams::default()
                    .with_max_iter(STARVED)
                    .with_tol(1.0e-12),
            )
            .map(drop)
        }),
        refuse("Lasso coordinate descent, starved budget", STARVED, || {
            Lasso::fit(
                &data.as_view(),
                &small_regression(),
                LassoParams::default()
                    .with_alpha(0.01)
                    .with_max_iter(STARVED)
                    .with_tol(1.0e-12),
            )
            .map(drop)
        }),
        refuse(
            "ElasticNet coordinate descent, starved budget",
            STARVED,
            || {
                ElasticNet::fit(
                    &data.as_view(),
                    &small_regression(),
                    ElasticNetParams::default()
                        .with_alpha(0.01)
                        .with_l1_ratio(0.5)
                        .with_max_iter(STARVED)
                        .with_tol(1.0e-12),
                )
                .map(drop)
            },
        ),
        refuse("Platt calibration, starved budget", STARVED, || {
            let scores: Vec<f32> = (0..12).map(|row| row as f32 * 0.5 - 3.0).collect();
            PlattCalibrator::fit(
                &scores,
                &targets,
                PlattParams::default()
                    .with_max_iter(STARVED)
                    .with_tol(1.0e-12),
            )
            .map(drop)
        }),
    ]
}

/// A refusal that stopped well short of its budget exists, and its message does
/// not blame the budget.
///
/// The two halves are one test on purpose. The blocklist alone would pass on a
/// message that happens to avoid those words while the reproduction quietly
/// stopped reproducing; the iteration bound alone proves the input still hits
/// the stall arm but says nothing about what the caller is told.
#[test]
fn a_refusal_short_of_the_budget_does_not_blame_the_budget() {
    let refusals = every_refusal();
    let stall = &refusals[0];
    assert!(
        stall.iterations * 10 < stall.max_iter,
        "the stall reproduction used {} of {} iterations, which is no longer far \
         enough below the budget to prove the budget was not the constraint",
        stall.iterations,
        stall.max_iter
    );
    println!(
        "stall arm: {} reported {} iterations against a budget of {} and rendered as {:?}",
        stall.name, stall.iterations, stall.max_iter, stall.rendered
    );

    let lowered = stall.rendered.to_ascii_lowercase();
    for claim in BUDGET_CLAIMS {
        assert!(
            !lowered.contains(claim),
            "a refusal that used {} of {} iterations rendered as {:?}, which claims \
             the budget with {claim:?}",
            stall.iterations,
            stall.max_iter,
            stall.rendered
        );
    }
}

/// No refusal message names either cause, at any construction site.
#[test]
fn no_refusal_message_names_a_cause_it_cannot_know() {
    for refusal in every_refusal() {
        let lowered = refusal.rendered.to_ascii_lowercase();
        for claim in BUDGET_CLAIMS.into_iter().chain(STALL_CLAIMS) {
            assert!(
                !lowered.contains(claim),
                "{} rendered as {:?}, which names a cause one variant cannot name \
                 for every site: {claim:?}",
                refusal.name,
                refusal.rendered
            );
        }
        // Naming no cause is not licence to say nothing: the one fact the
        // variant carries has to survive into the sentence, because comparing
        // it with `max_iter` is what tells the two causes apart.
        assert!(
            refusal.rendered.contains(&refusal.iterations.to_string()),
            "{} rendered as {:?} without its iteration count {}",
            refusal.name,
            refusal.rendered,
            refusal.iterations
        );
    }
}
