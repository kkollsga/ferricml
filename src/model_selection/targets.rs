//! The target vocabulary classification model selection folds over.

use crate::data::{BinaryTargets, ClassTargets, SelectionError};

/// A validated classification target vector that model selection can fold.
///
/// Cross-validation and search need exactly two things from a target vector:
/// the labels to score against, and a fold-sized copy of it that still carries
/// the container's own guarantees. Both [`BinaryTargets`] and [`ClassTargets`]
/// already provide those, so this trait names the requirement rather than
/// adding one, and one cross-validation entry point serves both.
///
/// # Why this is a type parameter and not a second entry point
///
/// Label arity is a property of the **metric**, not of the loop.
/// [`ClassificationScore::output_kind`](super::ClassificationScore::output_kind)
/// already says whether a score reads labels, a positive-probability column, or
/// a whole probability matrix, and the matrix form works for any observed class
/// set. So the scoring layer was already arity-agnostic while cross-validation
/// was split by arity — a split that would have doubled every time the
/// capability question was answered as well. Making the target vocabulary a
/// parameter keeps the two axes composing instead of multiplying: the arity
/// lives in `T`, the "does this model give probabilities?" question lives in the
/// [`ScorableClassifier`](super::ScorableClassifier) view, and a third shape of
/// either adds an implementation rather than an entry point.
///
/// # Sealed
///
/// FerricML implements this for its own validated target containers and nothing
/// else. An implementation outside the crate could not uphold what the fold
/// loop relies on — that `select` preserves the container's construction-time
/// guarantees — and a target shape that model selection should serve is a new
/// container in `data`, which arrives with its implementation.
pub trait ClassificationTargets: sealed::Sealed + Sized {
    /// The validated labels, one per row, in row order.
    fn as_slice(&self) -> &[u8];

    /// Copies the selected targets, preserving this type's own guarantees.
    ///
    /// This is the container's inherent `select`, named here so the fold loop
    /// can hand a training fold straight back to a fitting closure that takes
    /// the same type.
    fn select(&self, indices: &[usize]) -> Result<Self, SelectionError>;
}

impl ClassificationTargets for BinaryTargets {
    fn as_slice(&self) -> &[u8] {
        Self::as_slice(self)
    }

    fn select(&self, indices: &[usize]) -> Result<Self, SelectionError> {
        Self::select(self, indices)
    }
}

impl ClassificationTargets for ClassTargets {
    fn as_slice(&self) -> &[u8] {
        Self::as_slice(self)
    }

    fn select(&self, indices: &[usize]) -> Result<Self, SelectionError> {
        Self::select(self, indices)
    }
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for crate::data::BinaryTargets {}
    impl Sealed for crate::data::ClassTargets {}
}
