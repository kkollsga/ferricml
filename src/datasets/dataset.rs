use crate::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};

/// A generated dataset's targets, in whichever validated vocabulary its task
/// uses.
///
/// This enum is owned by `datasets` rather than by [`crate::data`], and that is
/// a boundary rather than an accident: `data` holds the validated containers an
/// estimator accepts, and none of them needs to know that a *different* target
/// shape exists. A generator does, because one generator produces all three.
/// Adding this enum to `data` would put a sum type in front of every estimator
/// signature that takes exactly one arm of it.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Target {
    /// Two-class labels, every value `0` or `1`.
    Binary(BinaryTargets),
    /// General class labels, carrying their observed class set.
    Class(ClassTargets),
    /// Finite real-valued targets.
    Regression(RegressionTargets),
}

/// What a generated dataset's task is actually right about.
///
/// This is the thing no generator FerricML had before this module could
/// provide, and the reason a generator is worth owning rather than scripting.
/// Without it a comparison can measure only *where* two implementations
/// disagree; with it, it can measure which one is closer to correct.
///
/// The variant list grows with the task families. It is `#[non_exhaustive]`
/// because a family that arrives with a new kind of ground truth must not be a
/// breaking change for a caller that only ever matches the arms it asked for.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Truth {
    /// No task was drawn, so the design matrix is the whole dataset.
    ///
    /// This is what a bare source produces. It is a variant rather than an
    /// empty coefficient vector or a `None`, because "this data has no correct
    /// answer" and "this data's correct answer is all zeros" are different
    /// statements and a consumer must not be able to confuse them.
    DesignOnly,
    /// A task was drawn, but nothing was recorded about what it is right about.
    ///
    /// This is what the absorbed frozen lanes produce. Their targets are
    /// deterministic functions of the design, so a correct answer exists in
    /// principle — but the lanes were written to compare two implementations
    /// against *each other*, so neither the coefficients behind a classification
    /// score nor a noise-free regression target was ever kept, and inventing one
    /// now would be a claim this module cannot support.
    ///
    /// It is distinct from [`Truth::DesignOnly`], which says no task exists at
    /// all, and it is what a consumer checks before trusting an accuracy number
    /// to mean distance from correct rather than distance from another library.
    Unrecorded,
    /// The target's conditional mean is a known function of a known linear
    /// predictor `Xβ + b`.
    ///
    /// Reported by the linear and generalized linear regression families. The
    /// coefficients are the answer a correctly specified fit should recover, and
    /// [`Truth::LinearPredictor::conditional_mean`] is `E[y | x]` — the target
    /// with its noise removed, which is what a mean-squared error should be
    /// measured against when the question is how much of the *recoverable*
    /// signal a model found.
    LinearPredictor {
        /// The true coefficient of each column, exactly zero on the
        /// uninformative ones.
        coefficients: Vec<f32>,
        /// The true intercept of the linear predictor.
        intercept: f32,
        /// `E[y | x]` at each row, after the family's link.
        conditional_mean: Vec<f32>,
    },
    /// The target's conditional mean is known, and no linear predictor produces
    /// it.
    ///
    /// Reported by the nonlinear regression family. The absence of coefficients
    /// is the statement: a consumer must not read "no coefficients recorded" as
    /// "the coefficients are zero", and the two are different variants here for
    /// exactly that reason.
    ConditionalMean {
        /// `E[y | x]` at each row.
        values: Vec<f32>,
    },
    /// Labels were drawn from a Bayes probability that is a logistic function of
    /// a known linear score.
    ///
    /// This is what makes a probabilistic classifier measurable against the
    /// right answer rather than against another implementation: the recorded
    /// probabilities are `P(y = 1 | x)` for the labels as generated, including
    /// any label noise the contamination applied, so a perfectly calibrated
    /// model matches them exactly.
    LinearBayes {
        /// The true coefficient of each column of the score.
        coefficients: Vec<f32>,
        /// The intercept solved for, so the mean probability is the requested
        /// prevalence.
        intercept: f32,
        /// `P(y = 1 | x)` at each row.
        probabilities: Vec<f32>,
    },
    /// Labels were drawn from a known Bayes probability that no linear score
    /// produces.
    ///
    /// Reported by the nonlinear binary family. As with
    /// [`Truth::ConditionalMean`], the missing coefficients are a statement
    /// rather than an omission.
    Bayes {
        /// `P(y = 1 | x)` at each row.
        probabilities: Vec<f32>,
    },
    /// A design built to a requested condition number and rank, with a known
    /// linear predictor behind its target.
    ///
    /// The rank is exact rather than numerical: the columns past it are exact
    /// copies of the leading ones, so a solver's reported rank can be compared
    /// with `assert_eq!`.
    ///
    /// **On a rank-deficient design the recorded coefficients are not the answer
    /// a solver should return.** The least-squares problem then has an affine
    /// set of minimizers, all of which fit the data equally well, and FerricML
    /// returns the minimum-norm point of that set. The drawn coefficients are
    /// one other point in it. Recovering them is only meaningful when
    /// [`Truth::ConditionedDesign::rank`] equals the design's column count.
    ConditionedDesign {
        /// The coefficients the target was drawn from.
        coefficients: Vec<f32>,
        /// The intercept the target was drawn with.
        intercept: f32,
        /// `E[y | x]` at each row.
        conditional_mean: Vec<f32>,
        /// The design's exact algebraic rank.
        rank: usize,
    },
}

impl Truth {
    /// The true coefficients, when the family knows them.
    ///
    /// `None` covers three different situations on purpose — no task, an
    /// absorbed lane that recorded nothing, and a family whose conditional mean
    /// no linear predictor produces — because a caller asking this question
    /// wants one answer for all three: there is no coefficient vector to compare
    /// against. A caller that needs to tell them apart matches the variant.
    ///
    /// ```
    /// use ferricml::datasets::{Recipe, Task};
    ///
    /// let recipe = Recipe::seeded(64, 4, 7)?.with_task(Task::LinearRegression {
    ///     informative: 2,
    ///     coefficient_scale: 1.0,
    ///     intercept: 0.5,
    ///     noise_scale: 0.0,
    /// })?;
    /// let dataset = recipe.generate();
    /// let coefficients = dataset.truth().coefficients().expect("a linear family");
    ///
    /// // The uninformative columns are exactly zero, not merely small: a model
    /// // that declines to use them is exactly right, and that is checkable.
    /// assert_eq!(coefficients[2], 0.0);
    /// assert_eq!(coefficients[3], 0.0);
    /// # Ok::<(), ferricml::datasets::DatasetError>(())
    /// ```
    pub fn coefficients(&self) -> Option<&[f32]> {
        match self {
            Self::LinearPredictor { coefficients, .. }
            | Self::LinearBayes { coefficients, .. }
            | Self::ConditionedDesign { coefficients, .. } => Some(coefficients),
            _ => None,
        }
    }

    /// The true intercept, when the family knows one.
    pub fn intercept(&self) -> Option<f32> {
        match *self {
            Self::LinearPredictor { intercept, .. }
            | Self::LinearBayes { intercept, .. }
            | Self::ConditionedDesign { intercept, .. } => Some(intercept),
            _ => None,
        }
    }

    /// `E[y | x]` at each row, when the family draws a continuous target.
    pub fn conditional_mean(&self) -> Option<&[f32]> {
        match self {
            Self::LinearPredictor {
                conditional_mean, ..
            }
            | Self::ConditionedDesign {
                conditional_mean, ..
            } => Some(conditional_mean),
            Self::ConditionalMean { values } => Some(values),
            _ => None,
        }
    }

    /// `P(y = 1 | x)` at each row, when the family draws labels.
    pub fn probabilities(&self) -> Option<&[f32]> {
        match self {
            Self::LinearBayes { probabilities, .. } | Self::Bayes { probabilities } => {
                Some(probabilities)
            }
            _ => None,
        }
    }

    /// The design's exact algebraic rank, when the family fixed it.
    pub fn rank(&self) -> Option<usize> {
        match *self {
            Self::ConditionedDesign { rank, .. } => Some(rank),
            _ => None,
        }
    }
}

/// A generated dataset: the design matrix, whatever task was drawn over it, and
/// the ground truth behind that task.
///
/// Everything here is a function of the [`Recipe`](super::Recipe) that produced
/// it, including [`Dataset::spec_digest`], which is carried so a dataset that
/// has travelled — through a file, a cache, or another process — can still say
/// which recipe it came from.
///
/// ```
/// use ferricml::datasets::{Recipe, Truth};
///
/// let recipe = Recipe::seeded(32, 5, 3)?;
/// let dataset = recipe.generate();
///
/// assert_eq!(dataset.features().rows(), 32);
/// assert_eq!(dataset.features().columns(), 5);
/// assert_eq!(dataset.spec_digest(), recipe.spec_digest());
///
/// // A recipe with no task family has nothing to be right about, and says so.
/// assert!(dataset.target().is_none());
/// assert_eq!(dataset.truth(), &Truth::DesignOnly);
/// # Ok::<(), ferricml::datasets::DatasetError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    features: DenseMatrix,
    target: Option<Target>,
    weights: Option<SampleWeights>,
    truth: Truth,
    groups: Option<Vec<u64>>,
    spec_digest: [u8; 32],
}

impl Dataset {
    /// Assembles a dataset from parts a task family has already produced.
    ///
    /// Crate-private because every field has to agree with every other — the
    /// target, the weights and the groups all have one entry per design row —
    /// and the only code positioned to know that is the family that generated
    /// them together. A caller reaches a `Dataset` by generating one.
    pub(super) fn from_parts(
        features: DenseMatrix,
        target: Option<Target>,
        weights: Option<SampleWeights>,
        truth: Truth,
        groups: Option<Vec<u64>>,
        spec_digest: [u8; 32],
    ) -> Self {
        debug_assert!(
            target_len(target.as_ref()).is_none_or(|len| len == features.rows()),
            "target length disagrees with the design's row count"
        );
        debug_assert!(
            weights
                .as_ref()
                .is_none_or(|w| w.as_slice().len() == features.rows()),
            "weight count disagrees with the design's row count"
        );
        debug_assert!(
            groups.as_ref().is_none_or(|g| g.len() == features.rows()),
            "group count disagrees with the design's row count"
        );
        Self {
            features,
            target,
            weights,
            truth,
            groups,
            spec_digest,
        }
    }

    /// Returns the design matrix.
    #[inline]
    pub const fn features(&self) -> &DenseMatrix {
        &self.features
    }

    /// Returns the targets, or `None` when no task family was drawn.
    #[inline]
    pub const fn target(&self) -> Option<&Target> {
        self.target.as_ref()
    }

    /// Returns the per-row sample weights, or `None` when the recipe asked for
    /// none.
    ///
    /// A dataset without weights is not a dataset whose weights are all one:
    /// the two fit differently under any estimator that normalizes by total
    /// weight, so the absence is reported rather than filled in.
    #[inline]
    pub const fn weights(&self) -> Option<&SampleWeights> {
        self.weights.as_ref()
    }

    /// Returns the ground truth behind this dataset's task.
    #[inline]
    pub const fn truth(&self) -> &Truth {
        &self.truth
    }

    /// Returns the per-row group identifiers, or `None` when the recipe asked
    /// for none.
    ///
    /// `u64` rather than a generic label type because that is what FerricML's
    /// grouped splitters take, so a generated dataset feeds
    /// `GroupKFold::split` without an adapter.
    #[inline]
    pub fn groups(&self) -> Option<&[u64]> {
        self.groups.as_deref()
    }

    /// Returns the digest of the recipe that produced this dataset.
    #[inline]
    pub const fn spec_digest(&self) -> [u8; 32] {
        self.spec_digest
    }

    /// Consumes the dataset and returns its design matrix.
    #[inline]
    pub fn into_features(self) -> DenseMatrix {
        self.features
    }
}

/// The row count a target vector claims, whichever vocabulary it uses.
fn target_len(target: Option<&Target>) -> Option<usize> {
    target.map(|target| match target {
        Target::Binary(targets) => targets.len(),
        Target::Class(targets) => targets.len(),
        Target::Regression(targets) => targets.len(),
    })
}
