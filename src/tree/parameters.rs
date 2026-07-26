//! Public decision-tree parameter types.

/// How many randomly selected features are considered at each split.
///
/// This is the one type every tree-shaped estimator shares, and
/// `ferricml::tree::MaxFeatures` is its only public path — a forest's
/// `with_max_features` takes *this* type. It lives here because it is a
/// property of growing a tree, which is what both a standalone tree and one
/// member of a forest do; the ensembles consume it, which is the direction the
/// `tree-below-estimators` layout rule enforces. It was briefly reachable as
/// `ensemble::MaxFeatures` as well, and rustdoc picked that alias as the
/// canonical path, so a standalone tree's own parameter rendered as an ensemble
/// type. One type keeps one path.
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

/// How a node chooses the split it takes among its candidate columns.
///
/// This is a parameter rather than two estimator types because the reference
/// treats it as one too: its singular randomized tree classes are its standard
/// tree classes with this setting changed, bit-identically so in every matched
/// pair the specification room compared. FerricML claims the same relation, and
/// spends one typed parameter on it rather than two more public types and two
/// more permanent artifact names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Splitter {
    /// Evaluate every boundary between adjacent distinct values in each
    /// candidate column and keep the best-scoring split.
    #[default]
    Best,
    /// Draw one threshold uniformly inside each candidate column's own range
    /// within the node, and keep the best-scoring of those draws.
    ///
    /// This is what makes an *extremely randomized* tree: the column set is
    /// still sampled, but the threshold is no longer optimized within a column,
    /// so individual trees decorrelate and fit far faster.
    Random,
}

use crate::artifact::ArtifactError;

const SPLITTER_BEST: u32 = 1;
const SPLITTER_RANDOM: u32 = 2;

/// The on-disk tag for a split-selection policy.
///
/// Beside the type for the same reason the feature policy's tag is: one
/// spelling per policy, so a model cannot acquire a second valid encoding by
/// being written through a different estimator.
pub(crate) const fn encode_splitter(value: Splitter) -> u32 {
    match value {
        Splitter::Best => SPLITTER_BEST,
        Splitter::Random => SPLITTER_RANDOM,
    }
}

pub(crate) const fn decode_splitter(tag: u32) -> Option<Splitter> {
    match tag {
        SPLITTER_BEST => Some(Splitter::Best),
        SPLITTER_RANDOM => Some(Splitter::Random),
        _ => None,
    }
}

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
            splitter: Splitter,
            random_state: u64,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    max_depth: None,
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    max_features: MaxFeatures::All,
                    splitter: Splitter::Best,
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

            /// Sets how a node picks its threshold among the candidate
            /// columns.
            ///
            /// [`Splitter::Random`] draws one threshold per candidate column
            /// instead of optimizing within it, which is what makes the tree
            /// *extremely randomized*. The candidate columns themselves are
            /// drawn the same way under both settings.
            #[must_use]
            pub fn with_splitter(mut self, splitter: Splitter) -> Self {
                self.splitter = splitter;
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

            /// Returns the split-selection policy.
            pub const fn splitter(&self) -> Splitter {
                self.splitter
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
