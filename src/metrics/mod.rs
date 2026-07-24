//! Deterministic classification and regression metrics.

use std::error::Error;
use std::fmt;

mod averaging;
mod classification;
mod confusion;
mod regression;

pub use averaging::{Average, Averaging, ZeroDivision};
pub use classification::{
    BinaryConfusionMatrix, accuracy_score, binary_confusion_matrix, brier_score, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
};
pub use confusion::{ClassCounts, ConfusionMatrix};
pub use regression::{mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error};

/// Errors produced while evaluating predictions.
///
/// Inputs are numbered in call order: expected values are input zero and
/// predicted values, probabilities, or scores are input one.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetricError {
    /// Expected and predicted inputs have different lengths.
    LengthMismatch {
        /// Expected input length.
        expected: usize,
        /// Actual input length.
        actual: usize,
    },
    /// A metric received no observations.
    Empty,
    /// A binary metric received a label other than zero or one.
    InvalidBinaryTarget {
        /// Zero for expected labels, one for predicted labels.
        input: usize,
        /// Position of the invalid label.
        index: usize,
        /// Invalid label value.
        value: u8,
    },
    /// A numeric input contained NaN or infinity.
    NonFiniteValue {
        /// Zero for expected values, one for predicted values or scores.
        input: usize,
        /// Position of the invalid value.
        index: usize,
    },
    /// A probability was outside the closed interval from zero to one.
    InvalidProbability {
        /// Position of the invalid probability.
        index: usize,
    },
    /// A binary-only average was requested for a wider label set.
    NotBinary {
        /// Number of distinct observed labels.
        labels: usize,
    },
    /// An F-score beta was not finite and strictly positive.
    InvalidBeta,
    /// The metric denominator or required class distribution is undefined.
    Undefined,
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(f, "expected {expected} metric values, got {actual}")
            }
            Self::Empty => f.write_str("metric requires at least one observation"),
            Self::InvalidBinaryTarget {
                input,
                index,
                value,
            } => write!(
                f,
                "binary label in input {input} at index {index} is {value}, expected 0 or 1"
            ),
            Self::NonFiniteValue { input, index } => {
                write!(
                    f,
                    "metric value in input {input} at index {index} is not finite"
                )
            }
            Self::InvalidProbability { index } => {
                write!(f, "probability at index {index} must be in 0..=1")
            }
            Self::NotBinary { labels } => write!(
                f,
                "binary averaging requires labels 0 and 1, got {labels} distinct labels"
            ),
            Self::InvalidBeta => f.write_str("F-score beta must be finite and strictly positive"),
            Self::Undefined => f.write_str("metric is undefined for these observations"),
        }
    }
}

impl Error for MetricError {}

fn validate_lengths(expected: usize, actual: usize) -> Result<(), MetricError> {
    if expected != actual {
        return Err(MetricError::LengthMismatch { expected, actual });
    }
    if expected == 0 {
        return Err(MetricError::Empty);
    }
    Ok(())
}

fn validate_binary(values: &[u8], input: usize) -> Result<(), MetricError> {
    if let Some((index, &value)) = values.iter().enumerate().find(|(_, value)| **value > 1) {
        return Err(MetricError::InvalidBinaryTarget {
            input,
            index,
            value,
        });
    }
    Ok(())
}

fn validate_finite(values: &[f32], input: usize) -> Result<(), MetricError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetricError::NonFiniteValue { input, index });
    }
    Ok(())
}

fn validate_probabilities(values: &[f32]) -> Result<(), MetricError> {
    validate_finite(values, 1)?;
    if let Some(index) = values.iter().position(|value| !(0.0..=1.0).contains(value)) {
        return Err(MetricError::InvalidProbability { index });
    }
    Ok(())
}
