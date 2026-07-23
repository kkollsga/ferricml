//! Immutable compact prediction trees.

use crate::data::MatrixView;

use super::{BoostingError, MAX_TREE_DEPTH, MAX_TREE_NODES};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CompactNode {
    Branch {
        feature: u32,
        threshold: f32,
        left: u32,
        right: u32,
    },
    Leaf {
        value: f32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactTree {
    nodes: Vec<CompactNode>,
}

impl CompactTree {
    pub(crate) fn from_nodes(
        nodes: Vec<CompactNode>,
        n_features: usize,
    ) -> Result<Self, BoostingError> {
        if nodes.is_empty() || n_features == 0 {
            return Err(BoostingError::InvalidTree);
        }
        if nodes.len() > MAX_TREE_NODES {
            return Err(BoostingError::TreeTooLarge);
        }
        let mut seen = vec![false; nodes.len()];
        let mut stack = vec![(0_usize, 0_usize)];
        while let Some((index, depth)) = stack.pop() {
            if index >= nodes.len() || seen[index] || depth > MAX_TREE_DEPTH {
                return Err(BoostingError::InvalidTree);
            }
            seen[index] = true;
            match nodes[index] {
                CompactNode::Leaf { value } if value.is_finite() => {}
                CompactNode::Leaf { .. } => return Err(BoostingError::InvalidTree),
                CompactNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    let left = left as usize;
                    let right = right as usize;
                    if feature as usize >= n_features
                        || !threshold.is_finite()
                        || left != index + 1
                        || right <= left
                        || right >= nodes.len()
                    {
                        return Err(BoostingError::InvalidTree);
                    }
                    stack.push((right, depth + 1));
                    stack.push((left, depth + 1));
                }
            }
        }
        if seen.iter().any(|&seen| !seen) {
            return Err(BoostingError::InvalidTree);
        }
        Ok(Self { nodes })
    }

    pub(crate) fn predict_one(&self, row: &[f32]) -> f32 {
        let mut index = 0_usize;
        loop {
            match self.nodes[index] {
                CompactNode::Leaf { value } => return value,
                CompactNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    index = if row[feature as usize] <= threshold {
                        left as usize
                    } else {
                        right as usize
                    };
                }
            }
        }
    }

    pub(crate) fn add_predictions(&self, data: &MatrixView<'_>, scale: f32, output: &mut [f32]) {
        debug_assert_eq!(data.rows(), output.len());
        for (row, slot) in data.iter_rows().zip(output) {
            *slot += scale * self.predict_one(row);
        }
    }

    #[allow(dead_code)] // Used by stable logical-tree encoding in the next phase.
    pub(crate) fn nodes(&self) -> &[CompactNode] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    #[test]
    fn compact_tree_traverses_raw_thresholds_and_adds_predictions() {
        let tree = CompactTree::from_nodes(
            vec![
                CompactNode::Branch {
                    feature: 0,
                    threshold: 1.5,
                    left: 1,
                    right: 2,
                },
                CompactNode::Leaf { value: -2.0 },
                CompactNode::Leaf { value: 3.0 },
            ],
            1,
        )
        .unwrap();
        assert_eq!(tree.predict_one(&[1.0]), -2.0);
        assert_eq!(tree.predict_one(&[2.0]), 3.0);
        let data = DenseMatrix::new(vec![0.0, 2.0], 2, 1).unwrap();
        let mut output = [1.0, 1.0];
        tree.add_predictions(&data.as_view(), 0.5, &mut output);
        assert_eq!(output, [0.0, 2.5]);
    }

    #[test]
    fn compact_layout_and_pathological_depth_are_bounded() {
        assert!(std::mem::size_of::<CompactNode>() <= 24);
        let depth = MAX_TREE_DEPTH;
        let mut nodes = Vec::with_capacity(depth * 2 + 1);
        for index in 0..depth {
            nodes.push(CompactNode::Branch {
                feature: 0,
                threshold: 0.5,
                left: (index + 1) as u32,
                right: (depth * 2 - index) as u32,
            });
        }
        nodes.push(CompactNode::Leaf { value: 7.0 });
        nodes.extend((0..depth).map(|_| CompactNode::Leaf { value: -1.0 }));
        let tree = CompactTree::from_nodes(nodes, 1).unwrap();
        assert_eq!(tree.predict_one(&[0.0]), 7.0);

        let invalid = CompactTree::from_nodes(
            vec![
                CompactNode::Branch {
                    feature: 0,
                    threshold: 0.0,
                    left: 1,
                    right: 1,
                },
                CompactNode::Leaf { value: 0.0 },
            ],
            1,
        );
        assert_eq!(invalid, Err(BoostingError::InvalidTree));
    }
}
