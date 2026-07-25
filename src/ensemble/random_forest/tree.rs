use crate::api::ModelError;
use crate::artifact::{ArtifactError, LogicalTreeNode};

pub(super) const LEAF_FEATURE: u32 = u32::MAX;
pub(super) const NO_CHILD: u32 = u32::MAX;
pub(super) const LEFT_IS_LEAF: u32 = 1 << 31;
pub(super) const RIGHT_IS_LEAF: u32 = 1 << 30;
pub(super) const FEATURE_MASK: u32 = RIGHT_IS_LEAF - 1;

/// Temporary uniform node used while building a tree.
///
/// A leaf has `feature == u32::MAX`, no children, and stores its prediction in
/// `payload`. A branch stores its threshold in the same field and sends values
/// `<= threshold` left and other values right.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(super) struct BuildNode {
    pub(super) feature: u32,
    pub(super) left: u32,
    pub(super) right: u32,
    /// Split threshold for branches, prediction for leaves.
    pub(super) payload: f32,
}

impl BuildNode {
    #[inline]
    pub(super) fn is_leaf(&self) -> bool {
        self.feature == LEAF_FEATURE
    }

    #[inline]
    pub(super) fn threshold(&self) -> f32 {
        debug_assert!(!self.is_leaf());
        self.payload
    }

    #[inline]
    pub(super) fn value(&self) -> f32 {
        debug_assert!(self.is_leaf());
        self.payload
    }

    pub(super) fn leaf(value: f32) -> Self {
        Self {
            feature: LEAF_FEATURE,
            left: NO_CHILD,
            right: NO_CHILD,
            payload: value,
        }
    }
}

/// One fixed-width inference branch. Two flag bits distinguish child branch
/// indices from inline leaf `f32` bits. This avoids fetching a second node or
/// array entry merely to discover a leaf.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(super) struct PackedNode {
    pub(super) left: u32,
    pub(super) right: u32,
    pub(super) threshold: f32,
    pub(super) feature_and_flags: u32,
}

/// A compact decision tree optimized for inference.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PackedTree {
    pub(super) nodes: Vec<PackedNode>,
    pub(super) root_leaf: Option<f32>,
}

/// Packs validated build nodes into the inference layout.
///
/// `encode_leaf` turns a leaf's build-node index into the `u32` its parent's
/// child slot carries. A scalar tree stores the leaf value's bits there; a
/// class tree stores the leaf's ordinal, because `classes` probabilities do not
/// fit in a child slot. Returning `None` means the root is itself a leaf, which
/// has no parent slot to be stored in.
///
/// Packing runs once per fitted tree and is deliberately shared: the encoding
/// of a leaf varies between tree flavours, but the pre-order token assignment
/// that keeps a parent adjacent to its left descendants must not.
fn pack_topology(
    build: &[BuildNode],
    n_features: usize,
    mut encode_leaf: impl FnMut(usize) -> u32,
) -> Result<Option<Vec<PackedNode>>, ModelError> {
    validate_build_topology(build, n_features)?;
    if build[0].is_leaf() {
        return Ok(None);
    }
    let mut branch_indices = vec![NO_CHILD; build.len()];
    let mut nodes = Vec::new();
    // Assign tokens in the builder's pre-order so a parent and its left
    // descendants remain adjacent during inference.
    for (index, node) in build.iter().enumerate() {
        if !node.is_leaf() {
            let packed_index = u32::try_from(nodes.len()).map_err(|_| ModelError::TreeTooLarge)?;
            nodes.push(PackedNode {
                left: 0,
                right: 0,
                threshold: node.threshold(),
                feature_and_flags: node.feature,
            });
            branch_indices[index] = packed_index;
        }
    }
    for (index, node) in build.iter().enumerate().filter(|(_, node)| !node.is_leaf()) {
        let packed = &mut nodes[branch_indices[index] as usize];
        if build[node.left as usize].is_leaf() {
            packed.feature_and_flags |= LEFT_IS_LEAF;
            packed.left = encode_leaf(node.left as usize);
        } else {
            packed.left = branch_indices[node.left as usize];
        }
        if build[node.right as usize].is_leaf() {
            packed.feature_and_flags |= RIGHT_IS_LEAF;
            packed.right = encode_leaf(node.right as usize);
        } else {
            packed.right = branch_indices[node.right as usize];
        }
    }
    Ok(Some(nodes))
}

impl PackedTree {
    pub(super) fn from_build_nodes(
        build: Vec<BuildNode>,
        n_features: usize,
    ) -> Result<Self, ModelError> {
        let Some(nodes) =
            pack_topology(&build, n_features, |index| build[index].value().to_bits())?
        else {
            return Ok(Self {
                nodes: Vec::new(),
                root_leaf: Some(build[0].value()),
            });
        };
        let tree = Self {
            nodes,
            root_leaf: None,
        };
        debug_assert!(tree.has_valid_packed_topology(n_features));
        Ok(tree)
    }

    #[inline(always)]
    pub(super) fn predict(&self, row: &[f32]) -> f32 {
        if let Some(value) = self.root_leaf {
            return value;
        }
        let mut index = 0usize;
        loop {
            // SAFETY: `from_build_nodes` validates every branch token against
            // the immutable packed buffer before the model becomes observable.
            // Each non-leaf child therefore remains a valid node index.
            let node = unsafe { self.nodes.get_unchecked(index) };
            // SAFETY: construction validates every encoded feature against the
            // fitted width, and public prediction validates `row` to that width
            // before entering tree traversal.
            let value =
                unsafe { *row.get_unchecked((node.feature_and_flags & FEATURE_MASK) as usize) };
            if value <= node.threshold {
                if node.feature_and_flags & LEFT_IS_LEAF != 0 {
                    return f32::from_bits(node.left);
                }
                index = node.left as usize;
            } else {
                if node.feature_and_flags & RIGHT_IS_LEAF != 0 {
                    return f32::from_bits(node.right);
                }
                index = node.right as usize;
            }
        }
    }

    fn has_valid_packed_topology(&self, n_features: usize) -> bool {
        if self.root_leaf.is_some() {
            return self.nodes.is_empty() && self.root_leaf.is_some_and(f32::is_finite);
        }
        !self.nodes.is_empty()
            && self.nodes.iter().all(|node| {
                ((node.feature_and_flags & FEATURE_MASK) as usize) < n_features
                    && node.threshold.is_finite()
                    && child_is_valid(
                        node.left,
                        node.feature_and_flags & LEFT_IS_LEAF,
                        self.nodes.len(),
                    )
                    && child_is_valid(
                        node.right,
                        node.feature_and_flags & RIGHT_IS_LEAF,
                        self.nodes.len(),
                    )
            })
    }
}

/// Conversion between the private inference layout and the stable logical
/// tree records used by artifacts.
///
/// The packed layout is deliberately not a serialization format: it stores
/// leaves inline in their parent's child slots as flagged `f32` bits, and it
/// keeps only branch nodes in `nodes`. Encoding therefore *synthesizes* the
/// leaf records, and decoding never trusts the logical bytes — it rebuilds
/// [`BuildNode`]s and re-runs the same topology validator that fitting uses.
impl PackedTree {
    /// Number of records [`PackedTree::to_logical_nodes`] will produce.
    ///
    /// Every branch has exactly two children, so a tree with `b` stored
    /// branches expands to `b` branch records and `b + 1` synthesized leaves.
    pub(super) fn logical_node_count(&self) -> usize {
        if self.root_leaf.is_some() {
            1
        } else {
            self.nodes.len() * 2 + 1
        }
    }

    /// Largest absolute leaf value, including inline and root leaves.
    pub(super) fn max_abs_leaf(&self) -> f32 {
        if let Some(value) = self.root_leaf {
            return value.abs();
        }
        self.nodes.iter().fold(0.0_f32, |largest, node| {
            let mut largest = largest;
            if node.feature_and_flags & LEFT_IS_LEAF != 0 {
                largest = largest.max(f32::from_bits(node.left).abs());
            }
            if node.feature_and_flags & RIGHT_IS_LEAF != 0 {
                largest = largest.max(f32::from_bits(node.right).abs());
            }
            largest
        })
    }

    /// Expands the packed tree into pre-order logical records.
    ///
    /// The result satisfies the logical-tree contract: node `0` is the root,
    /// a branch's left child is always the next record, and a tree with `L`
    /// leaves has exactly `2L - 1` records. A `root_leaf` tree becomes the
    /// single-node logical tree.
    pub(super) fn to_logical_nodes(&self) -> Vec<LogicalTreeNode> {
        if let Some(value) = self.root_leaf {
            return vec![LogicalTreeNode::Leaf { value }];
        }
        // Every branch contributes one record and every branch has exactly two
        // children, so a tree with `b` branches has `b + 1` leaves.
        let mut nodes = Vec::with_capacity(self.nodes.len() * 2 + 1);
        let mut pending = vec![Emit::Branch(0)];
        while let Some(task) = pending.pop() {
            match task {
                Emit::Leaf(value) => nodes.push(LogicalTreeNode::Leaf { value }),
                Emit::AttachRight(parent) => {
                    let right = u32::try_from(nodes.len()).expect("bounded node count");
                    match &mut nodes[parent] {
                        LogicalTreeNode::Branch { right: slot, .. } => *slot = right,
                        LogicalTreeNode::Leaf { .. } => unreachable!("branch record"),
                    }
                }
                Emit::Branch(packed_index) => {
                    let node = self.nodes[packed_index];
                    let index = nodes.len();
                    let left = u32::try_from(index + 1).expect("bounded node count");
                    nodes.push(LogicalTreeNode::Branch {
                        feature: node.feature_and_flags & FEATURE_MASK,
                        threshold: node.threshold,
                        left,
                        // Patched once the left subtree has been emitted.
                        right: 0,
                    });
                    // Popped in reverse: left subtree, then the right index
                    // patch, then the right subtree.
                    pending.push(child(node.right, node.feature_and_flags & RIGHT_IS_LEAF));
                    pending.push(Emit::AttachRight(index));
                    pending.push(child(node.left, node.feature_and_flags & LEFT_IS_LEAF));
                }
            }
        }
        nodes
    }

    /// Rebuilds a packed tree from validated logical records.
    ///
    /// Decoded records are turned back into [`BuildNode`]s and handed to
    /// [`PackedTree::from_build_nodes`], so an artifact goes through exactly
    /// the topology, feature-width, and finiteness checks that a freshly
    /// fitted tree does.
    pub(super) fn from_logical_nodes(
        nodes: &[LogicalTreeNode],
        n_features: usize,
    ) -> Result<Self, ArtifactError> {
        let build = nodes
            .iter()
            .map(|&node| match node {
                LogicalTreeNode::Leaf { value } => BuildNode::leaf(value),
                LogicalTreeNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                } => BuildNode {
                    feature,
                    left,
                    right,
                    payload: threshold,
                },
            })
            .collect();
        Self::from_build_nodes(build, n_features).map_err(|_| ArtifactError::InvalidPayload)
    }
}

/// A decision tree whose leaves store one probability per class.
///
/// The packed layout stores a scalar leaf inline in its parent's child slot,
/// which `classes` probabilities cannot fit in. The child slot therefore holds
/// the leaf's **ordinal** — a plain index into [`ClassTree::probabilities`],
/// not punned float bits — assigned in the same pre-order the branches are.
/// Traversal is otherwise identical, and inference returns a borrowed row of
/// probabilities so a forest can average without allocating.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ClassTree {
    nodes: Vec<PackedNode>,
    /// Row-major, one row of `classes` values per leaf ordinal. A tree whose
    /// root is a leaf has no branches and exactly one row.
    probabilities: Vec<f32>,
    classes: usize,
}

impl ClassTree {
    /// Builds a class tree from validated build nodes and their retained
    /// per-node class weights.
    ///
    /// `weights[i]` is the total weight reaching build node `i` and
    /// `class_weights[i * classes ..]` its per-class split, so a leaf's stored
    /// probability is one division of `f64` accumulations narrowed once.
    pub(super) fn from_build_nodes(
        build: Vec<BuildNode>,
        class_weights: &[f64],
        weights: &[u64],
        classes: usize,
        n_features: usize,
    ) -> Result<Self, ModelError> {
        debug_assert_eq!(class_weights.len(), build.len() * classes);
        debug_assert_eq!(weights.len(), build.len());
        let mut probabilities = Vec::new();
        let mut push_leaf = |index: usize| {
            let total = weights[index] as f64;
            for class in 0..classes {
                probabilities.push((class_weights[index * classes + class] / total) as f32);
            }
        };
        let mut ordinal = 0_u32;
        let nodes = pack_topology(&build, n_features, |index| {
            push_leaf(index);
            let assigned = ordinal;
            ordinal += 1;
            assigned
        })?;
        let Some(nodes) = nodes else {
            push_leaf(0);
            return Ok(Self {
                nodes: Vec::new(),
                probabilities,
                classes,
            });
        };
        let tree = Self {
            nodes,
            probabilities,
            classes,
        };
        if !tree.has_valid_class_topology(n_features) {
            return Err(ModelError::TreeTooLarge);
        }
        Ok(tree)
    }

    /// Number of probability columns each leaf stores.
    #[inline]
    pub(super) fn classes(&self) -> usize {
        self.classes
    }

    /// The leaf probabilities one row reaches.
    #[inline(always)]
    pub(super) fn probabilities(&self, row: &[f32]) -> &[f32] {
        let ordinal = if self.nodes.is_empty() {
            0
        } else {
            self.leaf_ordinal(row) as usize
        };
        &self.probabilities[ordinal * self.classes..(ordinal + 1) * self.classes]
    }

    #[inline(always)]
    fn leaf_ordinal(&self, row: &[f32]) -> u32 {
        let mut index = 0usize;
        loop {
            // SAFETY: `from_build_nodes` validates every branch token and every
            // encoded feature against the immutable packed buffer before the
            // model becomes observable, and public prediction validates `row`
            // to the fitted width before entering traversal.
            let node = unsafe { self.nodes.get_unchecked(index) };
            // SAFETY: as above; the feature index is checked at construction.
            let value =
                unsafe { *row.get_unchecked((node.feature_and_flags & FEATURE_MASK) as usize) };
            if value <= node.threshold {
                if node.feature_and_flags & LEFT_IS_LEAF != 0 {
                    return node.left;
                }
                index = node.left as usize;
            } else {
                if node.feature_and_flags & RIGHT_IS_LEAF != 0 {
                    return node.right;
                }
                index = node.right as usize;
            }
        }
    }

    /// Every branch token is in range, every leaf ordinal addresses a stored
    /// probability row, and every stored probability is a finite `0..=1`.
    fn has_valid_class_topology(&self, n_features: usize) -> bool {
        let leaves = self.nodes.len() + 1;
        if self.probabilities.len() != leaves * self.classes || self.classes == 0 {
            return false;
        }
        if !self
            .probabilities
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        {
            return false;
        }
        self.nodes.iter().all(|node| {
            ((node.feature_and_flags & FEATURE_MASK) as usize) < n_features
                && node.threshold.is_finite()
                && class_child_is_valid(
                    node.left,
                    node.feature_and_flags & LEFT_IS_LEAF,
                    self.nodes.len(),
                    leaves,
                )
                && class_child_is_valid(
                    node.right,
                    node.feature_and_flags & RIGHT_IS_LEAF,
                    self.nodes.len(),
                    leaves,
                )
        })
    }
}

fn class_child_is_valid(value: u32, leaf_flag: u32, node_count: usize, leaves: usize) -> bool {
    if leaf_flag != 0 {
        (value as usize) < leaves
    } else {
        (value as usize) < node_count
    }
}

enum Emit {
    Branch(usize),
    Leaf(f32),
    AttachRight(usize),
}

fn child(value: u32, leaf_flag: u32) -> Emit {
    if leaf_flag != 0 {
        Emit::Leaf(f32::from_bits(value))
    } else {
        Emit::Branch(value as usize)
    }
}

fn child_is_valid(value: u32, leaf_flag: u32, node_count: usize) -> bool {
    if leaf_flag != 0 {
        f32::from_bits(value).is_finite()
    } else {
        (value as usize) < node_count
    }
}

fn validate_build_topology(build: &[BuildNode], n_features: usize) -> Result<(), ModelError> {
    if build.is_empty() || n_features == 0 {
        return Err(ModelError::TreeTooLarge);
    }
    let mut seen = vec![false; build.len()];
    let mut stack = vec![0_usize];
    while let Some(index) = stack.pop() {
        if index >= build.len() || seen[index] {
            return Err(ModelError::TreeTooLarge);
        }
        seen[index] = true;
        let node = build[index];
        if !node.payload.is_finite() {
            return Err(ModelError::TreeTooLarge);
        }
        if node.is_leaf() {
            if node.left != NO_CHILD || node.right != NO_CHILD {
                return Err(ModelError::TreeTooLarge);
            }
            continue;
        }
        let left = node.left as usize;
        let right = node.right as usize;
        if node.feature as usize >= n_features
            || left != index + 1
            || right <= left
            || right >= build.len()
        {
            return Err(ModelError::TreeTooLarge);
        }
        stack.push(right);
        stack.push(left);
    }
    if seen.iter().any(|&visited| !visited) {
        return Err(ModelError::TreeTooLarge);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_builder_topologies_before_packing() {
        assert_eq!(
            PackedTree::from_build_nodes(Vec::new(), 1),
            Err(ModelError::TreeTooLarge)
        );
        assert_eq!(
            PackedTree::from_build_nodes(vec![BuildNode::leaf(0.0)], 0),
            Err(ModelError::TreeTooLarge)
        );

        let invalid_feature = vec![
            BuildNode {
                feature: 1,
                left: 1,
                right: 2,
                payload: 0.5,
            },
            BuildNode::leaf(0.0),
            BuildNode::leaf(1.0),
        ];
        assert_eq!(
            PackedTree::from_build_nodes(invalid_feature, 1),
            Err(ModelError::TreeTooLarge)
        );

        let cycle = vec![
            BuildNode {
                feature: 0,
                left: 0,
                right: 2,
                payload: 0.5,
            },
            BuildNode::leaf(0.0),
            BuildNode::leaf(1.0),
        ];
        assert_eq!(
            PackedTree::from_build_nodes(cycle, 1),
            Err(ModelError::TreeTooLarge)
        );
    }

    fn branch(feature: u32, threshold: f32, left: u32, right: u32) -> BuildNode {
        BuildNode {
            feature,
            left,
            right,
            payload: threshold,
        }
    }

    /// Pre-order build nodes for
    /// `f0 <= 1 ? (f1 <= 2 ? -1 : 3) : (f0 <= 5 ? 7 : (f1 <= 0 ? 11 : 13))`.
    fn mixed_depth_build() -> Vec<BuildNode> {
        vec![
            branch(0, 1.0, 1, 4),
            branch(1, 2.0, 2, 3),
            BuildNode::leaf(-1.0),
            BuildNode::leaf(3.0),
            branch(0, 5.0, 5, 6),
            BuildNode::leaf(7.0),
            branch(1, 0.0, 7, 8),
            BuildNode::leaf(11.0),
            BuildNode::leaf(13.0),
        ]
    }

    #[test]
    fn logical_records_synthesize_inline_leaves_and_round_trip_to_the_same_tree() {
        let build = mixed_depth_build();
        let tree = PackedTree::from_build_nodes(build.clone(), 2).unwrap();
        // Four branches are stored; the five leaves live inline in flag bits.
        assert_eq!(tree.nodes.len(), 4);

        let logical = tree.to_logical_nodes();
        assert_eq!(logical.len(), build.len());
        assert_eq!(
            logical
                .iter()
                .filter(|node| matches!(node, LogicalTreeNode::Leaf { .. }))
                .count(),
            5
        );
        for (index, (&logical_node, build_node)) in logical.iter().zip(&build).enumerate() {
            let expected = if build_node.is_leaf() {
                LogicalTreeNode::Leaf {
                    value: build_node.value(),
                }
            } else {
                LogicalTreeNode::Branch {
                    feature: build_node.feature,
                    threshold: build_node.threshold(),
                    left: build_node.left,
                    right: build_node.right,
                }
            };
            assert_eq!(logical_node, expected, "record {index}");
        }

        let restored = PackedTree::from_logical_nodes(&logical, 2).unwrap();
        assert_eq!(restored, tree);
        assert_eq!(restored.to_logical_nodes(), logical);
        for row in [[0.0, 0.0], [0.0, 9.0], [3.0, 0.0], [9.0, -1.0], [9.0, 9.0]] {
            assert_eq!(restored.predict(&row), tree.predict(&row));
        }
    }

    #[test]
    fn a_root_leaf_tree_maps_to_the_one_node_logical_tree() {
        let tree = PackedTree::from_build_nodes(vec![BuildNode::leaf(2.5)], 3).unwrap();
        assert!(tree.nodes.is_empty());
        let logical = tree.to_logical_nodes();
        assert_eq!(logical, vec![LogicalTreeNode::Leaf { value: 2.5 }]);
        let restored = PackedTree::from_logical_nodes(&logical, 3).unwrap();
        assert_eq!(restored.root_leaf, Some(2.5));
        assert_eq!(restored, tree);
    }

    #[test]
    fn logical_records_are_revalidated_instead_of_trusted() {
        let valid = PackedTree::from_build_nodes(mixed_depth_build(), 2)
            .unwrap()
            .to_logical_nodes();
        assert!(PackedTree::from_logical_nodes(&valid, 2).is_ok());

        let cases: [(&str, Vec<LogicalTreeNode>, usize); 5] = [
            ("empty", Vec::new(), 2),
            (
                "feature beyond the fitted width",
                vec![
                    LogicalTreeNode::Branch {
                        feature: 2,
                        threshold: 1.0,
                        left: 1,
                        right: 2,
                    },
                    LogicalTreeNode::Leaf { value: 0.0 },
                    LogicalTreeNode::Leaf { value: 1.0 },
                ],
                2,
            ),
            (
                "leaf sentinel smuggled in as a branch feature",
                vec![
                    LogicalTreeNode::Branch {
                        feature: LEAF_FEATURE,
                        threshold: 1.0,
                        left: 1,
                        right: 2,
                    },
                    LogicalTreeNode::Leaf { value: 0.0 },
                    LogicalTreeNode::Leaf { value: 1.0 },
                ],
                2,
            ),
            (
                "non-finite leaf value",
                vec![
                    LogicalTreeNode::Branch {
                        feature: 0,
                        threshold: 1.0,
                        left: 1,
                        right: 2,
                    },
                    LogicalTreeNode::Leaf {
                        value: f32::INFINITY,
                    },
                    LogicalTreeNode::Leaf { value: 1.0 },
                ],
                2,
            ),
            (
                "unreachable trailing record",
                vec![
                    LogicalTreeNode::Branch {
                        feature: 0,
                        threshold: 1.0,
                        left: 1,
                        right: 2,
                    },
                    LogicalTreeNode::Leaf { value: 0.0 },
                    LogicalTreeNode::Leaf { value: 1.0 },
                    LogicalTreeNode::Leaf { value: 2.0 },
                ],
                2,
            ),
        ];
        for (name, nodes, n_features) in cases {
            assert_eq!(
                PackedTree::from_logical_nodes(&nodes, n_features),
                Err(ArtifactError::InvalidPayload),
                "{name} was accepted"
            );
        }
        assert_eq!(
            PackedTree::from_logical_nodes(&valid, 0),
            Err(ArtifactError::InvalidPayload)
        );
    }

    #[test]
    fn deep_left_spines_convert_without_recursing() {
        let depth = 4_096;
        let mut build = Vec::with_capacity(depth * 2 + 1);
        for index in 0..depth {
            build.push(branch(
                0,
                0.5,
                (index + 1) as u32,
                (depth * 2 - index) as u32,
            ));
        }
        build.push(BuildNode::leaf(7.0));
        build.extend((0..depth).map(|_| BuildNode::leaf(-1.0)));
        let tree = PackedTree::from_build_nodes(build, 1).unwrap();
        let logical = tree.to_logical_nodes();
        assert_eq!(logical.len(), depth * 2 + 1);
        assert_eq!(PackedTree::from_logical_nodes(&logical, 1).unwrap(), tree);
    }
}
