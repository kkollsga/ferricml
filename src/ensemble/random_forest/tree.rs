use crate::api::ModelError;

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

impl PackedTree {
    pub(super) fn from_build_nodes(
        build: Vec<BuildNode>,
        n_features: usize,
    ) -> Result<Self, ModelError> {
        validate_build_topology(&build, n_features)?;
        if build[0].is_leaf() {
            return Ok(Self {
                nodes: Vec::new(),
                root_leaf: Some(build[0].value()),
            });
        }
        let mut branch_indices = vec![NO_CHILD; build.len()];
        let mut nodes = Vec::new();
        // Assign tokens in the builder's pre-order so a parent and its left
        // descendants remain adjacent during inference.
        for (index, node) in build.iter().enumerate() {
            if !node.is_leaf() {
                let packed_index =
                    u32::try_from(nodes.len()).map_err(|_| ModelError::TreeTooLarge)?;
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
            let left = build[node.left as usize];
            let right = build[node.right as usize];
            let packed = &mut nodes[branch_indices[index] as usize];
            if left.is_leaf() {
                packed.feature_and_flags |= LEFT_IS_LEAF;
                packed.left = left.value().to_bits();
            } else {
                packed.left = branch_indices[node.left as usize];
            }
            if right.is_leaf() {
                packed.feature_and_flags |= RIGHT_IS_LEAF;
                packed.right = right.value().to_bits();
            } else {
                packed.right = branch_indices[node.right as usize];
            }
        }
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
}
