//! Training-only histogram split search and mutable tree growth.

use super::binning::{BinnedMatrix, Binner};
use super::predictor::{CompactNode, CompactTree};
use super::{BoostingError, MAX_BINS, MAX_TREE_LEAVES, MAX_TREE_NODES};
use crate::loss::{
    Objective, constant_hessian_total, negative_gradient_sum, newton_leaf_value, newton_split_score,
};
use crate::numeric::sum_in_order;

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
    candidate: Option<SplitCandidate>,
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

/// Grows one tree against the negative gradients of `O`.
///
/// The tree searches histograms; the objective supplies every derivative. For
/// squared error the negative gradients are the familiar residuals, which is
/// why this function's error variants keep that name.
///
/// `O` must declare a constant hessian: the split search carries one histogram
/// of gradient sums plus a weight total, and a per-sample hessian cannot be
/// recovered from either. That is a compile-time requirement, not a runtime
/// check.
///
/// A sample weight scales that row's gradient and its share of every weight
/// total, so the minimum leaf size counts weight rather than rows. That is what
/// makes an integer weight the same tree as repeating the row. The bin grid is
/// **not** weighted: it is fitted from the observed feature values, which
/// repeating a row does not change either.
pub(crate) fn grow_tree<O: Objective>(
    binned: &BinnedMatrix,
    binner: &Binner,
    negative_gradients: &[f32],
    sample_weights: Option<&[f32]>,
    config: GrowConfig,
) -> Result<CompactTree, BoostingError> {
    config.validate()?;
    if negative_gradients.len() != binned.rows() {
        return Err(BoostingError::ResidualLength {
            rows: binned.rows(),
            residuals: negative_gradients.len(),
        });
    }
    if let Some(index) = negative_gradients
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(BoostingError::NonFiniteResidual { index });
    }
    if binned.columns() != binner.n_features_in() {
        return Err(BoostingError::FeatureDimension {
            expected: binner.n_features_in(),
            actual: binned.columns(),
        });
    }

    if let Some(weights) = sample_weights
        && weights.len() != binned.rows()
    {
        return Err(BoostingError::ResidualLength {
            rows: binned.rows(),
            residuals: weights.len(),
        });
    }

    let root_samples = (0..binned.rows()).collect::<Vec<_>>();
    let root_weight = weight_total(&root_samples, sample_weights);
    let root_value = leaf_value::<O>(
        &root_samples,
        negative_gradients,
        sample_weights,
        root_weight,
        config.l2_regularization,
    );
    let mut workspace = SplitWorkspace::new();
    let root_candidate = best_split::<O>(
        binned,
        negative_gradients,
        sample_weights,
        &root_samples,
        root_weight,
        config,
        &mut workspace,
    );
    let mut nodes = vec![GrowingNode {
        samples: root_samples,
        depth: 0,
        value: root_value,
        split: None,
        candidate: root_candidate,
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
            if let Some(candidate) = node.candidate
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
        let left_weight = weight_total(&left_samples, sample_weights);
        let right_weight = weight_total(&right_samples, sample_weights);
        debug_assert!(left_weight >= config.min_samples_leaf as f64);
        debug_assert!(right_weight >= config.min_samples_leaf as f64);
        let depth = nodes[node_index].depth + 1;
        let left = nodes.len();
        let right = left + 1;
        let left_value = leaf_value::<O>(
            &left_samples,
            negative_gradients,
            sample_weights,
            left_weight,
            config.l2_regularization,
        );
        let right_value = leaf_value::<O>(
            &right_samples,
            negative_gradients,
            sample_weights,
            right_weight,
            config.l2_regularization,
        );
        let left_candidate = if config.max_depth.is_none_or(|max_depth| depth < max_depth) {
            best_split::<O>(
                binned,
                negative_gradients,
                sample_weights,
                &left_samples,
                left_weight,
                config,
                &mut workspace,
            )
        } else {
            None
        };
        let right_candidate = if config.max_depth.is_none_or(|max_depth| depth < max_depth) {
            best_split::<O>(
                binned,
                negative_gradients,
                sample_weights,
                &right_samples,
                right_weight,
                config,
                &mut workspace,
            )
        } else {
            None
        };
        nodes.push(GrowingNode {
            samples: left_samples,
            depth,
            value: left_value,
            split: None,
            candidate: left_candidate,
        });
        nodes.push(GrowingNode {
            samples: right_samples,
            depth,
            value: right_value,
            split: None,
            candidate: right_candidate,
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

fn best_split<O: Objective>(
    binned: &BinnedMatrix,
    negative_gradients: &[f32],
    sample_weights: Option<&[f32]>,
    samples: &[usize],
    total_weight: f64,
    config: GrowConfig,
    workspace: &mut SplitWorkspace,
) -> Option<SplitCandidate> {
    if total_weight < config.min_samples_leaf.checked_mul(2)? as f64 {
        return None;
    }
    let total_sum = negative_gradient_sum(samples, negative_gradients, sample_weights);
    let parent_score = score::<O>(total_sum, total_weight, config.l2_regularization);
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
        let (bin_weights, sums) = workspace.reset(bin_count);
        match sample_weights {
            None => {
                for &sample in samples {
                    let bin = usize::from(
                        binned
                            .get(sample, feature)
                            .expect("validated binned sample"),
                    );
                    bin_weights[bin] += 1.0;
                    sums[bin] += f64::from(negative_gradients[sample]);
                }
            }
            Some(weights) => {
                for &sample in samples {
                    let bin = usize::from(
                        binned
                            .get(sample, feature)
                            .expect("validated binned sample"),
                    );
                    let weight = f64::from(weights[sample]);
                    bin_weights[bin] += weight;
                    sums[bin] += weight * f64::from(negative_gradients[sample]);
                }
            }
        }
        let mut left_weight = 0.0_f64;
        let mut left_sum = 0.0_f64;
        let minimum = config.min_samples_leaf as f64;
        for threshold in 0..bin_count - 1 {
            left_weight += bin_weights[threshold];
            left_sum += sums[threshold];
            let right_weight = total_weight - left_weight;
            if left_weight < minimum || right_weight < minimum {
                continue;
            }
            let right_sum = total_sum - left_sum;
            let gain = score::<O>(left_sum, left_weight, config.l2_regularization)
                + score::<O>(right_sum, right_weight, config.l2_regularization)
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

struct SplitWorkspace {
    /// Total sample weight per bin; a plain count when unweighted.
    bin_weights: Vec<f64>,
    sums: Vec<f64>,
}

impl SplitWorkspace {
    fn new() -> Self {
        Self {
            bin_weights: vec![0.0; MAX_BINS],
            sums: vec![0.0; MAX_BINS],
        }
    }

    fn reset(&mut self, bin_count: usize) -> (&mut [f64], &mut [f64]) {
        debug_assert!(bin_count <= MAX_BINS);
        let bin_weights = &mut self.bin_weights[..bin_count];
        let sums = &mut self.sums[..bin_count];
        bin_weights.fill(0.0);
        sums.fill(0.0);
        (bin_weights, sums)
    }
}

/// Total sample weight of one node.
///
/// Unweighted this is the row count, produced without a pass over the samples,
/// so an unweighted fit computes exactly the quantity it always did.
fn weight_total(samples: &[usize], sample_weights: Option<&[f32]>) -> f64 {
    match sample_weights {
        None => samples.len() as f64,
        Some(weights) => sum_in_order(samples.iter().map(|&sample| f64::from(weights[sample]))),
    }
}

fn score<O: Objective>(sum: f64, weight: f64, l2_regularization: f32) -> f64 {
    newton_split_score(sum, constant_hessian_total::<O>(weight), l2_regularization)
}

fn leaf_value<O: Objective>(
    samples: &[usize],
    negative_gradients: &[f32],
    sample_weights: Option<&[f32]>,
    weight: f64,
    l2_regularization: f32,
) -> f32 {
    newton_leaf_value(
        negative_gradient_sum(samples, negative_gradients, sample_weights),
        constant_hessian_total::<O>(weight),
        l2_regularization,
    )
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
    use crate::loss::SquaredError;

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
        let first =
            grow_tree::<SquaredError>(&binned, &binner, &residuals, None, config()).unwrap();
        let second =
            grow_tree::<SquaredError>(&binned, &binner, &residuals, None, config()).unwrap();
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
        let tree = grow_tree::<SquaredError>(
            &binned,
            &binner,
            &[-1.0, -1.0, 1.0, 1.0],
            None,
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
        let tree = grow_tree::<SquaredError>(
            &binned,
            &binner,
            &residuals,
            None,
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
            grow_tree::<SquaredError>(&binned, &binner, &[1.0], None, config()),
            Err(BoostingError::ResidualLength {
                rows: 8,
                residuals: 1
            })
        );
        let mut residuals = [0.0; 8];
        residuals[3] = f32::NAN;
        assert_eq!(
            grow_tree::<SquaredError>(&binned, &binner, &residuals, None, config()),
            Err(BoostingError::NonFiniteResidual { index: 3 })
        );
        assert_eq!(
            grow_tree::<SquaredError>(
                &binned,
                &binner,
                &[0.0; 8],
                None,
                GrowConfig {
                    max_leaf_nodes: 1,
                    ..config()
                }
            ),
            Err(BoostingError::InvalidMaxLeafNodes)
        );
    }
}
