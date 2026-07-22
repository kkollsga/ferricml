//! Random-forest public parameters and fitted models.

pub use crate::forest::{RandomForestClassifier, RandomForestRegressor};

/// How many randomly selected features are considered at each split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxFeatures {
    /// Consider every feature.
    All,
    /// Consider `floor(sqrt(n_features))`, with a minimum of one.
    Sqrt,
    /// Consider exactly this many features.
    Count(usize),
}

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

macro_rules! forest_params {
    ($name:ident, $default_features:expr) => {
        #[doc = concat!("Parameters for [`", stringify!($name), "`].")]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name {
            n_estimators: usize,
            max_depth: Option<usize>,
            min_samples_split: usize,
            min_samples_leaf: usize,
            max_features: MaxFeatures,
            bootstrap: bool,
            random_state: u64,
            n_jobs: NJobs,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    n_estimators: 100,
                    max_depth: None,
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    max_features: $default_features,
                    bootstrap: true,
                    random_state: 0,
                    n_jobs: NJobs::Serial,
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
            pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
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
            pub fn with_n_jobs(mut self, n_jobs: NJobs) -> Self {
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
            pub const fn max_features(&self) -> MaxFeatures {
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
            pub const fn n_jobs(&self) -> NJobs {
                self.n_jobs
            }
        }
    };
}

forest_params!(RandomForestClassifierParams, MaxFeatures::Sqrt);
forest_params!(RandomForestRegressorParams, MaxFeatures::All);
