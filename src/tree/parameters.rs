//! Public decision-tree parameter types.

/// How many randomly selected features are considered at each split.
///
/// The candidate set is drawn uniformly **without replacement**, and every
/// drawn column consumes the quota whether or not it turns out to admit a
/// split. See the recorded divergences in the reference contract: this is a
/// smaller claim than the reference's, which skips a column that is constant
/// within the node and keeps drawing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxFeatures {
    /// Consider every feature.
    All,
    /// Consider `floor(sqrt(n_features))`, with a minimum of one.
    Sqrt,
    /// Consider exactly this many features.
    Count(usize),
}
