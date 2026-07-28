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
