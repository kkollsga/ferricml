//! Immutable compact prediction trees.

use crate::artifact::{ArtifactError, LogicalTreeNode};
use crate::data::MatrixView;

use super::error::{BoostingError, MAX_TREE_DEPTH, MAX_TREE_NODES};

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
    pub(crate) fn from_logical_nodes(
        nodes: Vec<LogicalTreeNode>,
        n_features: usize,
    ) -> Result<Self, ArtifactError> {
        let nodes = nodes
            .into_iter()
            .map(|node| match node {
                LogicalTreeNode::Leaf { value } => CompactNode::Leaf { value },
                LogicalTreeNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                } => CompactNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                },
            })
            .collect();
        Self::from_nodes(nodes, n_features).map_err(|_| ArtifactError::InvalidPayload)
    }

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

    pub(crate) fn nodes(&self) -> &[CompactNode] {
        &self.nodes
    }

    pub(crate) fn to_logical_nodes(&self) -> Vec<LogicalTreeNode> {
        self.nodes
            .iter()
            .copied()
            .map(|node| match node {
                CompactNode::Leaf { value } => LogicalTreeNode::Leaf { value },
                CompactNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                } => LogicalTreeNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                },
            })
            .collect()
    }

    pub(crate) fn max_abs_leaf(&self) -> f32 {
        self.nodes
            .iter()
            .filter_map(|node| match node {
                CompactNode::Leaf { value } => Some(value.abs()),
                CompactNode::Branch { .. } => None,
            })
            .fold(0.0, f32::max)
    }
}

/// Whether every prediction this ensemble can produce stays inside `f32`.
///
/// The bound is the baseline plus the shrunk worst-case leaf of every tree, so
/// it is an upper bound on `|prediction|` rather than a sample of it, and it is
/// checked once after fitting and again after decoding. A model that cannot
/// answer finitely for *some* input is refused rather than left to report an
/// infinity at prediction time.
pub(crate) fn prediction_bound_is_finite(
    baseline: f32,
    learning_rate: f32,
    trees: &[CompactTree],
) -> bool {
    let mut bound = f64::from(baseline.abs());
    for tree in trees {
        bound += f64::from(learning_rate.abs()) * f64::from(tree.max_abs_leaf());
        if !bound.is_finite() || bound > f64::from(f32::MAX) {
            return false;
        }
    }
    true
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
