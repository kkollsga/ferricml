//! Pairwise linear ranking and tie-aware rank metrics.

mod metrics;
mod pairwise;

pub use metrics::{
    RankingMetricError, decisive_directional_accuracy, kendall_tau_b, spearman_correlation,
    three_way_accuracy,
};
pub use pairwise::{
    PairDataError, PairIndex, PairOutcome, PairwiseError, PairwiseLinearRanker,
    PairwiseLinearRankerParams, PairwiseObservation,
};
