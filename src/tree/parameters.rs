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

use crate::artifact::ArtifactError;

const MAX_FEATURES_ALL: u32 = 1;
const MAX_FEATURES_SQRT: u32 = 2;
const MAX_FEATURES_COUNT: u32 = 3;

/// The on-disk tag and count for a feature-selection policy.
///
/// Defined beside the type rather than inside one estimator, so every
/// tree-shaped artifact spells the same policy the same way and a model cannot
/// acquire a second valid encoding by being written by a different estimator.
pub(crate) fn encode_max_features(value: MaxFeatures) -> Result<(u32, u32), ArtifactError> {
    Ok(match value {
        MaxFeatures::All => (MAX_FEATURES_ALL, 0),
        MaxFeatures::Sqrt => (MAX_FEATURES_SQRT, 0),
        MaxFeatures::Count(count) => (
            MAX_FEATURES_COUNT,
            u32::try_from(count).map_err(|_| ArtifactError::InvalidPayload)?,
        ),
    })
}

pub(crate) fn decode_max_features(tag: u32, count: u32, n_features: usize) -> Option<MaxFeatures> {
    match tag {
        MAX_FEATURES_ALL if count == 0 => Some(MaxFeatures::All),
        MAX_FEATURES_SQRT if count == 0 => Some(MaxFeatures::Sqrt),
        MAX_FEATURES_COUNT if count != 0 && count as usize <= n_features => {
            Some(MaxFeatures::Count(count as usize))
        }
        _ => None,
    }
}

macro_rules! tree_params {
    ($name:ident, $estimator:literal) => {
        #[doc = concat!("Parameters for [`", $estimator, "`](super::", $estimator, ").")]
        ///
        /// A single tree has no ensemble to average over, so there is no member
        /// count, no bootstrap resampling, and no thread count here. What
        /// remains is exactly what growing one tree depends on.
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name {
            max_depth: Option<usize>,
            min_samples_split: usize,
            min_samples_leaf: usize,
            max_features: MaxFeatures,
            random_state: u64,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    max_depth: None,
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    max_features: MaxFeatures::All,
                    random_state: 0,
                }
            }
        }

        impl $name {
            /// Sets the maximum tree depth. `None` grows until another limit.
            #[must_use]
            pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
                self.max_depth = max_depth;
                self
            }

            /// Sets the minimum samples required to split a node.
            #[must_use]
            pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
                self.min_samples_split = min_samples_split;
                self
            }

            /// Sets the minimum samples required in each leaf.
            #[must_use]
            pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
                self.min_samples_leaf = min_samples_leaf;
                self
            }

            /// Sets how many features are considered at each split.
            #[must_use]
            pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
                self.max_features = max_features;
                self
            }

            /// Sets the deterministic training seed.
            ///
            /// A tree still draws its candidate columns from this seed even at
            /// [`MaxFeatures::All`], because the draw order is also the
            /// cross-column tie-break.
            #[must_use]
            pub fn with_random_state(mut self, random_state: u64) -> Self {
                self.random_state = random_state;
                self
            }

            /// Returns the maximum tree depth.
            pub const fn max_depth(&self) -> Option<usize> {
                self.max_depth
            }

            /// Returns the minimum samples required to split a node.
            pub const fn min_samples_split(&self) -> usize {
                self.min_samples_split
            }

            /// Returns the minimum samples required in each leaf.
            pub const fn min_samples_leaf(&self) -> usize {
                self.min_samples_leaf
            }

            /// Returns the feature-selection policy.
            pub const fn max_features(&self) -> MaxFeatures {
                self.max_features
            }

            /// Returns the deterministic training seed.
            pub const fn random_state(&self) -> u64 {
                self.random_state
            }
        }
    };
}

tree_params!(DecisionTreeClassifierParams, "DecisionTreeClassifier");
tree_params!(DecisionTreeRegressorParams, "DecisionTreeRegressor");
