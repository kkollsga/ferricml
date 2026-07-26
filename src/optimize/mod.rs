//! Crate-private bounded iterative optimization.
//!
//! An estimator used to own its update rule outright. That is correct for one
//! model and does not survive the second: the binary logistic fit's exact
//! Newton step needs a `parameters x parameters` system, and a multinomial fit
//! stacks that to `(classes * parameters)^2`, so the solver that is right for a
//! handful of features refuses to exist at all for a few thousand. This module
//! owns the *matrix-free* alternative — limited-memory BFGS with a strong-Wolfe
//! line search — so a new objective reaches a second-order-quality solver
//! without shipping a solver of its own.
//!
//! It also owns the step-length rule the *exact* Newton paths need. An exact
//! step is globally convergent on nothing: it minimizes a local quadratic
//! model, and where that model is untrustworthy the step overshoots and keeps
//! overshooting. [`armijo_backtracking`] is the standard remedy, kept here
//! rather than inside an estimator so the binary and multinomial logistic paths
//! — and any later exact-Newton consumer — share one rule instead of three
//! copies. Its own documentation states why those consumers do not reach for the
//! strong-Wolfe search next to it.
//!
//! # Boundaries
//!
//! - Private. Optimizers are not public vocabulary; keeping them internal lets
//!   the representation change without API churn.
//! - This module depends on [`crate::numeric`] and, for the objectives it is
//!   given, on [`crate::loss`]. It names no estimator family, so it cannot
//!   acquire knowledge of the models it minimizes;
//!   `scripts/check_source_layout.py` enforces that mechanically.
//! - The caller owns the objective *and* its storage. [`Problem`] takes
//!   `&mut self` precisely so a linear model can keep its design matrix and its
//!   score buffer inside the problem, which is what makes a whole solve
//!   allocation-free after [`LbfgsWorkspace::new`].
//! - Everything is bounded before the loop starts: history length, iteration
//!   count, line-search steps, and zoom steps are all fixed by
//!   [`LbfgsOptions`]. Running out of any of them is a typed error, never a
//!   silently truncated answer that looks like a fitted model.
//!
//! # Determinism
//!
//! Every reduction in this module goes through [`crate::numeric::sum_in_order`]
//! in a fixed index order, per rule 2 of the accumulation policy. An iterative
//! solver is exactly the kind of code that invites reassociating a dot product
//! for speed; nothing here does, because the fitted coefficients are the fixed
//! point of these reductions and reordering one would move them.
//!
//! The line search is bisection-based rather than interpolation-based inside
//! its zoom phase. That is a determinism decision as much as a robustness one:
//! bisection's next trial depends only on the current bracket, so the iterate
//! sequence is a function of the inputs alone.

mod damping;
mod lbfgs;
mod line_search;

pub(crate) use damping::armijo_backtracking;
pub(crate) use lbfgs::{
    DEFAULT_MEMORY, LbfgsOptions, LbfgsWorkspace, OptimizeError, Problem, minimize,
};
