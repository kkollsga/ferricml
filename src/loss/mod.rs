//! Crate-private objective contract shared by FerricML's solvers.
//!
//! An estimator used to fuse three separate decisions into one body of code:
//! *what* is being minimized, *how* a raw score becomes the quantity the target
//! is compared with, and *which* update rule walks downhill. This module owns
//! the first two. A solver names an [`Objective`] as a generic parameter and
//! reads its declared properties, so adding a loss no longer costs one
//! estimator-sized change per estimator.
//!
//! # Boundaries
//!
//! - The contract is private. Losses are not public vocabulary, and keeping
//!   them internal lets their representation change without API churn.
//! - Dispatch is entirely at compile time. Nothing here is object-safe by
//!   accident and nothing takes `&self`, so no hot path can acquire a per-row
//!   virtual call by mistake.
//! - This module depends on [`crate::numeric`] and on nothing else in the
//!   crate. An objective that named a concrete estimator would invert the
//!   dependency the contract exists to create;
//!   `scripts/check_source_layout.py` enforces that mechanically.
//! - Reductions over rows stay with the caller. Fitted artifacts depend on
//!   accumulation order, and only the solver knows the order its determinism
//!   guarantee is written against.

mod binary_log_loss;
mod boosting;
mod linear;
mod link;
mod objective;
mod squared_error;

pub(crate) use binary_log_loss::BinaryLogLoss;
pub(crate) use boosting::{
    constant_hessian_total, negative_gradient_sum, newton_leaf_value, newton_split_score,
};
pub(crate) use linear::{accumulate_newton_row, raw_score};
pub(crate) use objective::Objective;
pub(crate) use squared_error::SquaredError;
