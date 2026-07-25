//! Decision trees, and the private grower every tree-shaped estimator shares.
//!
//! The packed inference layout, the logical-tree bridge, and the split search
//! live here rather than inside one estimator family, because a standalone
//! tree and one member of a forest are the *same* tree. Sharing the grower is
//! what makes that a fact about the code instead of a claim about two
//! implementations that happen to agree.
//!
//! The direction of the dependency is the boundary that matters, and it is
//! mechanically enforced by the `tree-below-estimators` layout rule: the
//! ensemble families consume this module, never the reverse. Naming one of
//! them here is exactly what that rule refuses. The runtime representations
//! (`PackedTree`, `ClassTree`, `BuildNode`) stay crate-private, so no caller
//! can come to depend on a forest's — or a tree's — node layout.

mod grower;
mod packed;
mod parameters;

pub(crate) use grower::{
    Classification, GrowerConfig, Objective, Regression, grow_class_tree, grow_tree,
};
pub(crate) use packed::{ClassTree, FEATURE_MASK, PackedTree};

#[cfg(test)]
pub(crate) use packed::{LEFT_IS_LEAF, PackedNode, RIGHT_IS_LEAF};

pub use parameters::MaxFeatures;
