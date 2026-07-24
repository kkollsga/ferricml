//! Stable logical tree records independent of estimator runtime layouts.

use super::{ArtifactCursor, ArtifactError, ArtifactPayloadWriter};

const LEAF_TAG: u32 = 0;
const BRANCH_TAG: u32 = 1;
const TREE_HEADER_BYTES: usize = 3 * 4;
const NODE_RECORD_BYTES: usize = 5 * 4;
const MAX_TREE_NODES: usize = 131_071;
const MAX_TREE_LEAVES: usize = 65_536;
const MAX_TREE_DEPTH: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LogicalTreeNode {
    Leaf {
        value: f32,
    },
    Branch {
        feature: u32,
        threshold: f32,
        left: u32,
        right: u32,
    },
}

pub(crate) fn encode_logical_tree(nodes: &[LogicalTreeNode]) -> Result<Vec<u8>, ArtifactError> {
    let (leaf_count, max_depth) = topology_stats(nodes)?;
    let node_count = u32::try_from(nodes.len()).map_err(|_| ArtifactError::InvalidPayload)?;
    let leaf_count = u32::try_from(leaf_count).map_err(|_| ArtifactError::InvalidPayload)?;
    let max_depth = u32::try_from(max_depth).map_err(|_| ArtifactError::InvalidPayload)?;
    let mut payload =
        ArtifactPayloadWriter::with_capacity(TREE_HEADER_BYTES + nodes.len() * NODE_RECORD_BYTES);
    payload.u32(node_count);
    payload.u32(leaf_count);
    payload.u32(max_depth);
    for &node in nodes {
        match node {
            LogicalTreeNode::Leaf { value } => {
                payload.u32(LEAF_TAG);
                payload.f32(value);
                payload.u32(0);
                payload.u32(0);
                payload.u32(0);
            }
            LogicalTreeNode::Branch {
                feature,
                threshold,
                left,
                right,
            } => {
                payload.u32(BRANCH_TAG);
                payload.u32(feature);
                payload.f32(threshold);
                payload.u32(left);
                payload.u32(right);
            }
        }
    }
    Ok(payload.finish())
}

pub(crate) fn decode_logical_tree(
    mut payload: ArtifactCursor<'_>,
) -> Result<Vec<LogicalTreeNode>, ArtifactError> {
    let node_count = payload.u32()? as usize;
    let declared_leaves = payload.u32()? as usize;
    let declared_depth = payload.u32()? as usize;
    if node_count == 0
        || node_count > MAX_TREE_NODES
        || declared_leaves == 0
        || declared_leaves > MAX_TREE_LEAVES
        || node_count != declared_leaves.saturating_mul(2).saturating_sub(1)
        || declared_depth > MAX_TREE_DEPTH
    {
        return Err(ArtifactError::InvalidPayload);
    }
    let expected_bytes = node_count
        .checked_mul(NODE_RECORD_BYTES)
        .ok_or(ArtifactError::InvalidPayload)?;
    if payload.remaining().len() != expected_bytes {
        return Err(if payload.remaining().len() < expected_bytes {
            ArtifactError::Truncated
        } else {
            ArtifactError::TrailingBytes
        });
    }

    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        match payload.u32()? {
            LEAF_TAG => {
                let value = payload.f32()?;
                if !value.is_finite()
                    || payload.u32()? != 0
                    || payload.u32()? != 0
                    || payload.u32()? != 0
                {
                    return Err(ArtifactError::InvalidPayload);
                }
                nodes.push(LogicalTreeNode::Leaf { value });
            }
            BRANCH_TAG => {
                let feature = payload.u32()?;
                let threshold = payload.f32()?;
                let left = payload.u32()?;
                let right = payload.u32()?;
                if !threshold.is_finite() {
                    return Err(ArtifactError::InvalidPayload);
                }
                nodes.push(LogicalTreeNode::Branch {
                    feature,
                    threshold,
                    left,
                    right,
                });
            }
            _ => return Err(ArtifactError::InvalidPayload),
        }
    }
    let (actual_leaves, actual_depth) = topology_stats(&nodes)?;
    if actual_leaves != declared_leaves || actual_depth != declared_depth {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok(nodes)
}

fn topology_stats(nodes: &[LogicalTreeNode]) -> Result<(usize, usize), ArtifactError> {
    if nodes.is_empty() || nodes.len() > MAX_TREE_NODES {
        return Err(ArtifactError::InvalidPayload);
    }
    let mut seen = vec![false; nodes.len()];
    let mut stack = vec![(0_usize, 0_usize)];
    let mut leaves = 0_usize;
    let mut max_depth = 0_usize;
    while let Some((index, depth)) = stack.pop() {
        if index >= nodes.len() || seen[index] || depth > MAX_TREE_DEPTH {
            return Err(ArtifactError::InvalidPayload);
        }
        seen[index] = true;
        max_depth = max_depth.max(depth);
        match nodes[index] {
            LogicalTreeNode::Leaf { value } => {
                if !value.is_finite() {
                    return Err(ArtifactError::InvalidPayload);
                }
                leaves += 1;
            }
            LogicalTreeNode::Branch {
                threshold,
                left,
                right,
                ..
            } => {
                let left = left as usize;
                let right = right as usize;
                if !threshold.is_finite()
                    || left != index + 1
                    || right <= left
                    || right >= nodes.len()
                {
                    return Err(ArtifactError::InvalidPayload);
                }
                stack.push((right, depth + 1));
                stack.push((left, depth + 1));
            }
        }
    }
    if seen.iter().any(|&seen| !seen)
        || leaves == 0
        || leaves > MAX_TREE_LEAVES
        || nodes.len() != leaves.saturating_mul(2).saturating_sub(1)
    {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok((leaves, max_depth))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_tree() -> Vec<LogicalTreeNode> {
        vec![
            LogicalTreeNode::Branch {
                feature: 0,
                threshold: 1.5,
                left: 1,
                right: 2,
            },
            LogicalTreeNode::Leaf { value: -1.0 },
            LogicalTreeNode::Leaf { value: 2.0 },
        ]
    }

    fn decode(bytes: &[u8]) -> Result<Vec<LogicalTreeNode>, ArtifactError> {
        decode_logical_tree(ArtifactCursor::new(bytes))
    }

    #[test]
    fn logical_tree_round_trip_is_deterministic() {
        let tree = valid_tree();
        let bytes = encode_logical_tree(&tree).unwrap();
        assert_eq!(bytes, encode_logical_tree(&tree).unwrap());
        assert_eq!(decode(&bytes).unwrap(), tree);
        assert_eq!(bytes.len(), TREE_HEADER_BYTES + 3 * NODE_RECORD_BYTES);
    }

    #[test]
    fn rejects_counts_before_allocating_records() {
        let mut payload = ArtifactPayloadWriter::with_capacity(TREE_HEADER_BYTES);
        payload.u32((MAX_TREE_NODES + 1) as u32);
        payload.u32(1);
        payload.u32(0);
        assert_eq!(
            decode(&payload.finish()),
            Err(ArtifactError::InvalidPayload)
        );
    }

    #[test]
    fn rejects_feature_children_reachability_cycles_and_nonfinite_values() {
        let bytes = encode_logical_tree(&valid_tree()).unwrap();
        let mut feature = bytes.clone();
        feature[16..20].copy_from_slice(&1_u32.to_le_bytes());
        assert!(decode(&feature).is_ok());

        let mut duplicate_child = bytes.clone();
        duplicate_child[28..32].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(decode(&duplicate_child), Err(ArtifactError::InvalidPayload));

        let mut cycle = bytes.clone();
        cycle[24..28].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(decode(&cycle), Err(ArtifactError::InvalidPayload));

        let mut leaf = bytes;
        leaf[36..40].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert_eq!(decode(&leaf), Err(ArtifactError::InvalidPayload));
    }

    #[test]
    fn rejects_declared_shape_depth_truncation_and_trailing_bytes() {
        let bytes = encode_logical_tree(&valid_tree()).unwrap();
        let mut leaves = bytes.clone();
        leaves[4..8].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(decode(&leaves), Err(ArtifactError::InvalidPayload));
        let mut depth = bytes.clone();
        depth[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(decode(&depth), Err(ArtifactError::InvalidPayload));
        assert_eq!(
            decode(&bytes[..bytes.len() - 1]),
            Err(ArtifactError::Truncated)
        );
        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(decode(&trailing), Err(ArtifactError::TrailingBytes));
    }
}
