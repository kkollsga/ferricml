//! Private histogram-boosting runtime boundaries.

pub(crate) mod binning;
pub(crate) mod grower;
pub(crate) mod predictor;

use std::error::Error;
use std::fmt;

pub(crate) const MAX_BINS: usize = 255;
pub(crate) const MAX_TREE_NODES: usize = 131_071;
pub(crate) const MAX_TREE_LEAVES: usize = 65_536;
pub(crate) const MAX_TREE_DEPTH: usize = 256;
pub(crate) const MAX_TREES: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BoostingError {
    InvalidMaxBins,
    InvalidMaxLeafNodes,
    InvalidMaxDepth,
    InvalidMinSamplesLeaf,
    InvalidL2Regularization,
    ResidualLength { rows: usize, residuals: usize },
    NonFiniteResidual { index: usize },
    FeatureDimension { expected: usize, actual: usize },
    TooManyFeatures,
    TreeTooLarge,
    InvalidTree,
}

impl fmt::Display for BoostingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaxBins => write!(f, "max_bins must be in 2..={MAX_BINS}"),
            Self::InvalidMaxLeafNodes => {
                write!(f, "max_leaf_nodes must be in 2..={MAX_TREE_LEAVES}")
            }
            Self::InvalidMaxDepth => f.write_str("max_depth must be at least one when set"),
            Self::InvalidMinSamplesLeaf => f.write_str("min_samples_leaf must be at least one"),
            Self::InvalidL2Regularization => {
                f.write_str("l2_regularization must be finite and non-negative")
            }
            Self::ResidualLength { rows, residuals } => write!(
                f,
                "residual length {residuals} does not match row count {rows}"
            ),
            Self::NonFiniteResidual { index } => {
                write!(f, "residual at index {index} is not finite")
            }
            Self::FeatureDimension { expected, actual } => {
                write!(f, "expected {expected} features, got {actual}")
            }
            Self::TooManyFeatures => f.write_str("feature count exceeds boosting tree format"),
            Self::TreeTooLarge => f.write_str("boosting tree exceeds compact format"),
            Self::InvalidTree => f.write_str("boosting tree topology or value is invalid"),
        }
    }
}

impl Error for BoostingError {}
