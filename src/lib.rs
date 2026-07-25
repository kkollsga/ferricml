//! Lean, pure-Rust classical machine learning.
//!
//! FerricML focuses first on linear models and random forests, with stable
//! estimator semantics and fast, allocation-conscious inference.

pub mod api;
pub mod artifact;
pub mod calibration;
pub mod data;
pub mod dummy;
pub mod ensemble;
pub mod inspection;
pub mod linear_model;
mod loss;
pub mod metrics;
pub mod model_selection;
mod numeric;
mod optimize;
pub mod pipeline;
pub mod preprocessing;
pub mod ranking;
pub mod tree;
