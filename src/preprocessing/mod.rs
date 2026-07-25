//! Fitted dense preprocessing transformers.

mod max_abs_scaler;
mod min_max_scaler;
mod robust_scaler;
mod scaling;
mod standard_scaler;

pub use max_abs_scaler::{MaxAbsScaler, MaxAbsScalerParams};
pub use min_max_scaler::{MinMaxScaler, MinMaxScalerParams};
pub use robust_scaler::{RobustScaler, RobustScalerParams};
pub use standard_scaler::{StandardScaler, StandardScalerParams};
