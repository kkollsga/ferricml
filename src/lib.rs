//! Lean, pure-Rust classical machine learning.
//!
//! FerricML focuses first on linear models and random forests, with
//! scikit-style estimator semantics and fast, allocation-conscious inference.

pub mod api;
pub mod artifact;
pub mod data;
pub mod ensemble;
pub mod linear_model;
pub mod pipeline;
pub mod preprocessing;
pub mod ranking;

// The private runtime is intentionally introduced one green commit before its
// public estimator orchestration.
#[allow(dead_code)]
mod boosting;
mod forest;
