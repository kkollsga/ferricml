//! Parameters every bagged tree ensemble shares.
//!
//! One macro generates the parameter type for each such estimator, so two
//! ensembles cannot drift into spelling the same knob two ways — in their
//! builders, in their accessors, or on disk. What differs between them is
//! stated as macro arguments: the default feature policy, and the split search
//! their trees use, which is fixed by the type rather than exposed as a knob an
//! ensemble has no meaning for.

// Imported, not re-exported. `MaxFeatures` is defined beside the grower that
// consumes it, in `crate::tree`, and `ferricml::tree::MaxFeatures` is its one
// public path: the ensembles are consumers of the type rather than its owner,
// which is the same direction the `tree-below-estimators` layout rule enforces.
use crate::tree::MaxFeatures;

use crate::artifact::ArtifactError;

/// Requested training parallelism for an estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NJobs {
    /// Use one worker. This deterministic, low-overhead mode is the default.
    #[default]
    Serial,
    /// Use all hardware threads visible to the process.
    All,
    /// Use at most the given number of worker threads.
    Count(usize),
}

impl NJobs {
    pub(crate) fn resolved(self) -> usize {
        match self {
            Self::Serial => 1,
            Self::All => std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            Self::Count(count) => count,
        }
    }
}

const N_JOBS_SERIAL: u32 = 1;
const N_JOBS_ALL: u32 = 2;
const N_JOBS_COUNT: u32 = 3;

pub(crate) fn encode_n_jobs(value: NJobs) -> Result<(u32, u32), ArtifactError> {
    Ok(match value {
        NJobs::Serial => (N_JOBS_SERIAL, 0),
        NJobs::All => (N_JOBS_ALL, 0),
        NJobs::Count(count) => (
            N_JOBS_COUNT,
            u32::try_from(count).map_err(|_| ArtifactError::InvalidPayload)?,
        ),
    })
}

pub(crate) fn decode_n_jobs(tag: u32, count: u32) -> Option<NJobs> {
    match tag {
        N_JOBS_SERIAL if count == 0 => Some(NJobs::Serial),
        N_JOBS_ALL if count == 0 => Some(NJobs::All),
        N_JOBS_COUNT if count != 0 => Some(NJobs::Count(count as usize)),
        _ => None,
    }
}

/// The parameter values one fitted ensemble carries on disk.
///
/// A plain record rather than a generic bound: the codec needs the values, not
/// the type they came from, and keeping it concrete is what lets one encoder
/// serve every ensemble kind without a trait object or a monomorphised copy of
/// the whole metadata block per estimator.
pub(crate) struct ForestFields {
    pub(crate) n_features_in: usize,
    pub(crate) n_estimators: usize,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) max_features: MaxFeatures,
    pub(crate) bootstrap: bool,
    pub(crate) random_state: u64,
    pub(crate) n_jobs: NJobs,
}

macro_rules! forest_params {
    ($name:ident, $default_features:expr, $splitter:expr) => {
        #[doc = concat!("Parameters for [`", stringify!($name), "`].")]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name {
            n_estimators: usize,
            max_depth: Option<usize>,
            min_samples_split: usize,
            min_samples_leaf: usize,
            max_features: $crate::tree::MaxFeatures,
            bootstrap: bool,
            random_state: u64,
            n_jobs: $crate::ensemble::NJobs,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    n_estimators: 100,
                    max_depth: None,
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    max_features: $default_features,
                    bootstrap: $crate::ensemble::forest::parameters::default_bootstrap($splitter),
                    random_state: 0,
                    n_jobs: $crate::ensemble::NJobs::Serial,
                }
            }
        }

        impl $name {
            /// Sets the number of fitted trees.
            #[must_use]
            pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
                self.n_estimators = n_estimators;
                self
            }

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
            pub fn with_max_features(mut self, max_features: $crate::tree::MaxFeatures) -> Self {
                self.max_features = max_features;
                self
            }

            /// Enables or disables bootstrap sampling.
            #[must_use]
            pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
                self.bootstrap = bootstrap;
                self
            }

            /// Sets the deterministic training seed.
            #[must_use]
            pub fn with_random_state(mut self, random_state: u64) -> Self {
                self.random_state = random_state;
                self
            }

            /// Sets training parallelism.
            #[must_use]
            pub fn with_n_jobs(mut self, n_jobs: $crate::ensemble::NJobs) -> Self {
                self.n_jobs = n_jobs;
                self
            }

            /// Returns the number of fitted trees requested.
            pub const fn n_estimators(&self) -> usize {
                self.n_estimators
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
            pub const fn max_features(&self) -> $crate::tree::MaxFeatures {
                self.max_features
            }

            /// Returns whether bootstrap sampling is enabled.
            pub const fn bootstrap(&self) -> bool {
                self.bootstrap
            }

            /// Returns the deterministic training seed.
            pub const fn random_state(&self) -> u64 {
                self.random_state
            }

            /// Returns requested training parallelism.
            pub const fn n_jobs(&self) -> $crate::ensemble::NJobs {
                self.n_jobs
            }

            /// The fields the artifact carries, in one record the codec reads.
            pub(crate) fn artifact_fields(
                &self,
                n_features_in: usize,
            ) -> $crate::ensemble::forest::parameters::ForestFields {
                $crate::ensemble::forest::parameters::ForestFields {
                    n_features_in,
                    n_estimators: self.n_estimators,
                    max_depth: self.max_depth,
                    min_samples_split: self.min_samples_split,
                    min_samples_leaf: self.min_samples_leaf,
                    max_features: self.max_features,
                    bootstrap: self.bootstrap,
                    random_state: self.random_state,
                    n_jobs: self.n_jobs,
                }
            }

            /// Rebuilds the parameters a decoded artifact recorded.
            pub(crate) fn from_artifact_fields(
                fields: &$crate::ensemble::forest::parameters::ForestFields,
            ) -> Self {
                Self::default()
                    .with_n_estimators(fields.n_estimators)
                    .with_max_depth(fields.max_depth)
                    .with_min_samples_split(fields.min_samples_split)
                    .with_min_samples_leaf(fields.min_samples_leaf)
                    .with_max_features(fields.max_features)
                    .with_bootstrap(fields.bootstrap)
                    .with_random_state(fields.random_state)
                    .with_n_jobs(fields.n_jobs)
            }
        }

        impl From<&$name> for $crate::ensemble::forest::training::ForestConfig {
            fn from(params: &$name) -> Self {
                Self {
                    n_estimators: params.n_estimators(),
                    bootstrap: params.bootstrap(),
                    random_state: params.random_state(),
                    n_jobs: params.n_jobs().resolved(),
                    grower: $crate::tree::GrowerConfig {
                        max_depth: params.max_depth(),
                        min_samples_split: params.min_samples_split(),
                        min_samples_leaf: params.min_samples_leaf(),
                        max_features: params.max_features(),
                        splitter: $splitter,
                    },
                }
            }
        }
    };
}

/// A random forest resamples by default; a randomized ensemble does not.
///
/// Tying the default to the split search is what keeps the pair together: the
/// randomized ensemble gets its decorrelation from the thresholds, so drawing
/// a bootstrap sample on top of that only removes training rows.
pub(crate) const fn default_bootstrap(splitter: crate::tree::Splitter) -> bool {
    matches!(splitter, crate::tree::Splitter::Best)
}

pub(crate) use forest_params;
