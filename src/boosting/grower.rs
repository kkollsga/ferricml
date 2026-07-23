//! Training-only histogram split search and mutable tree growth.

use super::binning::{BinnedMatrix, Binner};
use super::predictor::{CompactNode, CompactTree};
use super::{BoostingError, MAX_TREE_LEAVES, MAX_TREE_NODES};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GrowConfig {
    pub(crate) max_leaf_nodes: usize,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_leaf: usize,
    pub(crate) l2_regularization: f32,
}

impl GrowConfig {
    fn validate(self) -> Result<(), BoostingError> {
        if !(2..=MAX_TREE_LEAVES).contains(&self.max_leaf_nodes) {
            return Err(BoostingError::InvalidMaxLeafNodes);
        }
        if self.max_depth == Some(0) {
            return Err(BoostingError::InvalidMaxDepth);
        }
        if self.min_samples_leaf == 0 {
            return Err(BoostingError::InvalidMinSamplesLeaf);
        }
        if !self.l2_regularization.is_finite() || self.l2_regularization < 0.0 {
            return Err(BoostingError::InvalidL2Regularization);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct GrowingNode {
    samples: Vec<usize>,
    depth: usize,
    value: f32,
    split: Option<GrownSplit>,
}

#[derive(Clone, Copy, Debug)]
struct GrownSplit {
    feature: usize,
    threshold_bin: u8,
    left: usize,
    right: usize,
}

#[derive(Clone, Copy, Debug)]
struct SplitCandidate {
    feature: usize,
    threshold_bin: u8,
    gain: f64,
}

pub(crate) fn grow_tree(
    binned: &BinnedMatrix,
    binner: &Binner,
    residuals: &[f32],
    config: GrowConfig,
) -> Result<CompactTree, BoostingError> {
    config.validate()?;
    if residuals.len() != binned.rows() {
        return Err(BoostingError::ResidualLength {
            rows: binned.rows(),
            residuals: residuals.len(),
        });
    }
    if let Some(index) = residuals.iter().position(|value| !value.is_finite()) {
        return Err(BoostingError::NonFiniteResidual { index });
    }
    if binned.columns() != binner.n_features_in() {
        return Err(BoostingError::FeatureDimension {
            expected: binner.n_features_in(),
            actual: binned.columns(),
        });
    }

    let root_samples = (0..binned.rows()).collect::<Vec<_>>();
    let root_value = leaf_value(&root_samples, residuals, config.l2_regularization);
    let mut nodes = vec![GrowingNode {
        samples: root_samples,
        depth: 0,
        value: root_value,
        split: None,
    }];
    let mut leaf_count = 1_usize;

    while leaf_count < config.max_leaf_nodes {
        let mut selected: Option<(usize, SplitCandidate)> = None;
        for (node_index, node) in nodes.iter().enumerate() {
            if node.split.is_some()
                || config
                    .max_depth
                    .is_some_and(|max_depth| node.depth >= max_depth)
            {
                continue;
            }
            if let Some(candidate) = best_split(binned, residuals, &node.samples, config)
                && selected
                    .as_ref()
                    .is_none_or(|(_, current)| candidate.gain > current.gain)
            {
                selected = Some((node_index, candidate));
            }
        }
        let Some((node_index, candidate)) = selected else {
            break;
        };
        if nodes
            .len()
            .checked_add(2)
            .is_none_or(|count| count > MAX_TREE_NODES)
        {
            return Err(BoostingError::TreeTooLarge);
        }

        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();
        for &sample in &nodes[node_index].samples {
            if binned
                .get(sample, candidate.feature)
                .expect("validated binned sample")
                <= candidate.threshold_bin
            {
                left_samples.push(sample);
            } else {
                right_samples.push(sample);
            }
        }
        debug_assert!(left_samples.len() >= config.min_samples_leaf);
        debug_assert!(right_samples.len() >= config.min_samples_leaf);
        let depth = nodes[node_index].depth + 1;
        let left = nodes.len();
        let right = left + 1;
        let left_value = leaf_value(&left_samples, residuals, config.l2_regularization);
        let right_value = leaf_value(&right_samples, residuals, config.l2_regularization);
        nodes.push(GrowingNode {
            samples: left_samples,
            depth,
            value: left_value,
            split: None,
        });
        nodes.push(GrowingNode {
            samples: right_samples,
            depth,
            value: right_value,
            split: None,
        });
        nodes[node_index].split = Some(GrownSplit {
            feature: candidate.feature,
            threshold_bin: candidate.threshold_bin,
            left,
            right,
        });
        leaf_count += 1;
    }

    compile_tree(&nodes, binner)
}

fn best_split(
    binned: &BinnedMatrix,
    residuals: &[f32],
    samples: &[usize],
    config: GrowConfig,
) -> Option<SplitCandidate> {
    if samples.len() < config.min_samples_leaf.checked_mul(2)? {
        return None;
    }
    let total_sum = samples
        .iter()
        .map(|&sample| f64::from(residuals[sample]))
        .sum::<f64>();
    let parent_score = score(total_sum, samples.len(), config.l2_regularization);
    let mut best = None;
    for feature in 0..binned.columns() {
        let max_bin = samples
            .iter()
            .map(|&sample| {
                binned
                    .get(sample, feature)
                    .expect("validated binned sample")
            })
            .max()
            .unwrap_or(0);
        if max_bin == 0 {
            continue;
        }
        let bin_count = usize::from(max_bin) + 1;
        let mut counts = vec![0_usize; bin_count];
        let mut sums = vec![0.0_f64; bin_count];
        for &sample in samples {
            let bin = usize::from(
                binned
                    .get(sample, feature)
                    .expect("validated binned sample"),
            );
            counts[bin] += 1;
            sums[bin] += f64::from(residuals[sample]);
        }
        let mut left_count = 0_usize;
        let mut left_sum = 0.0_f64;
        for threshold in 0..bin_count - 1 {
            left_count += counts[threshold];
            left_sum += sums[threshold];
            let right_count = samples.len() - left_count;
            if left_count < config.min_samples_leaf || right_count < config.min_samples_leaf {
                continue;
            }
            let right_sum = total_sum - left_sum;
            let gain = score(left_sum, left_count, config.l2_regularization)
                + score(right_sum, right_count, config.l2_regularization)
                - parent_score;
            if gain > 0.0
                && best
                    .as_ref()
                    .is_none_or(|candidate: &SplitCandidate| gain > candidate.gain)
            {
                best = Some(SplitCandidate {
                    feature,
                    threshold_bin: threshold as u8,
                    gain,
                });
            }
        }
    }
    best
}

fn score(sum: f64, count: usize, l2_regularization: f32) -> f64 {
    sum * sum / (count as f64 + f64::from(l2_regularization))
}

fn leaf_value(samples: &[usize], residuals: &[f32], l2_regularization: f32) -> f32 {
    let sum = samples
        .iter()
        .map(|&sample| f64::from(residuals[sample]))
        .sum::<f64>();
    (sum / (samples.len() as f64 + f64::from(l2_regularization))) as f32
}

fn compile_tree(nodes: &[GrowingNode], binner: &Binner) -> Result<CompactTree, BoostingError> {
    let mut order = Vec::with_capacity(nodes.len());
    let mut stack = vec![0_usize];
    while let Some(index) = stack.pop() {
        order.push(index);
        if let Some(split) = nodes[index].split {
            stack.push(split.right);
            stack.push(split.left);
        }
    }
    if order.len() != nodes.len() || order.len() > MAX_TREE_NODES {
        return Err(BoostingError::TreeTooLarge);
    }
    let mut remap = vec![0_usize; nodes.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    let mut compact = Vec::with_capacity(nodes.len());
    for &old in &order {
        let node = &nodes[old];
        if let Some(split) = node.split {
            compact.push(CompactNode::Branch {
                feature: u32::try_from(split.feature)
                    .map_err(|_| BoostingError::TooManyFeatures)?,
                threshold: binner.threshold(split.feature, split.threshold_bin),
                left: u32::try_from(remap[split.left]).map_err(|_| BoostingError::TreeTooLarge)?,
                right: u32::try_from(remap[split.right])
                    .map_err(|_| BoostingError::TreeTooLarge)?,
            });
        } else {
            compact.push(CompactNode::Leaf { value: node.value });
        }
    }
    CompactTree::from_nodes(compact, binner.n_features_in())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn data() -> DenseMatrix {
        DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap()
    }

    fn config() -> GrowConfig {
        GrowConfig {
            max_leaf_nodes: 4,
            max_depth: Some(3),
            min_samples_leaf: 1,
            l2_regularization: 0.0,
        }
    }

    #[test]
    fn grows_deterministic_compact_preorder_tree() {
        let data = data();
        let binner = Binner::fit(&data.as_view(), 8).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let residuals = [-2.0, -2.0, -2.0, -2.0, 3.0, 3.0, 3.0, 3.0];
        let first = grow_tree(&binned, &binner, &residuals, config()).unwrap();
        let second = grow_tree(&binned, &binner, &residuals, config()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.nodes().len(), 3);
        assert_eq!(first.predict_one(&[0.0]), -2.0);
        assert_eq!(first.predict_one(&[7.0]), 3.0);
        assert!(matches!(
            first.nodes()[0],
            CompactNode::Branch {
                feature: 0,
                threshold: 3.5,
                left: 1,
                right: 2
            }
        ));
    }

    #[test]
    fn equal_gain_splits_choose_the_first_feature() {
        let data = DenseMatrix::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 4, 2).unwrap();
        let binner = Binner::fit(&data.as_view(), 4).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let tree = grow_tree(
            &binned,
            &binner,
            &[-1.0, -1.0, 1.0, 1.0],
            GrowConfig {
                max_leaf_nodes: 2,
                ..config()
            },
        )
        .unwrap();
        assert!(matches!(
            tree.nodes()[0],
            CompactNode::Branch { feature: 0, .. }
        ));
    }

    #[test]
    fn depth_leaf_and_regularization_controls_are_enforced() {
        let data = data();
        let binner = Binner::fit(&data.as_view(), 4).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let residuals = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
        let tree = grow_tree(
            &binned,
            &binner,
            &residuals,
            GrowConfig {
                max_leaf_nodes: 8,
                max_depth: Some(1),
                min_samples_leaf: 2,
                l2_regularization: 2.0,
            },
        )
        .unwrap();
        assert_eq!(tree.nodes().len(), 3);
        assert_eq!(tree.predict_one(&[0.0]), -1.666_666_6);
        assert_eq!(tree.predict_one(&[7.0]), 1.666_666_6);
    }

    #[test]
    fn validates_residuals_and_growth_configuration() {
        let data = data();
        let binner = Binner::fit(&data.as_view(), 4).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        assert_eq!(
            grow_tree(&binned, &binner, &[1.0], config()),
            Err(BoostingError::ResidualLength {
                rows: 8,
                residuals: 1
            })
        );
        let mut residuals = [0.0; 8];
        residuals[3] = f32::NAN;
        assert_eq!(
            grow_tree(&binned, &binner, &residuals, config()),
            Err(BoostingError::NonFiniteResidual { index: 3 })
        );
        assert_eq!(
            grow_tree(
                &binned,
                &binner,
                &[0.0; 8],
                GrowConfig {
                    max_leaf_nodes: 1,
                    ..config()
                }
            ),
            Err(BoostingError::InvalidMaxLeafNodes)
        );
    }
}
