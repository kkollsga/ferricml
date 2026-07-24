//! Lean, pure-Rust classical machine learning.
//!
//! FerricML focuses first on linear models and random forests, with stable
//! estimator semantics and fast, allocation-conscious inference.

pub mod api;
pub mod artifact;
pub mod data;
pub mod ensemble;
pub mod linear_model;
pub mod metrics;
pub mod pipeline;
pub mod preprocessing;
pub mod ranking;

mod boosting;
mod forest;
