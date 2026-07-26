//! Training-only histogram split search and mutable tree growth.

use super::binning::{BinnedMatrix, Binner};
use super::error::{BoostingError, MAX_BINS, MAX_TREE_LEAVES, MAX_TREE_NODES};
use super::predictor::{CompactNode, CompactTree};
use crate::loss::{
    BoostingObjective, hessian_sum, negative_gradient_sum, newton_leaf_value, newton_split_score,
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

/// The per-sample statistics one boosting iteration searches over.
///
/// The three slices are always read together and always index by the same row,
/// so they travel as one value: a signature that took them separately made it
/// possible to hand a split search one node's gradients beside another's
/// weights.
#[derive(Clone, Copy)]
pub(crate) struct SampleStatistics<'a> {
    /// One negative gradient per row, in row order.
    pub(crate) negative_gradients: &'a [f32],
    /// One hessian per row, or empty where the objective's hessian is constant
    /// and the node's weight total already determines it.
    pub(crate) hessians: &'a [f32],
    /// Optional per-row sample weights.
    pub(crate) sample_weights: Option<&'a [f32]>,
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
/// `hessians` carries one per-sample second derivative and is required exactly
/// when `O` declares a varying hessian; a constant-hessian objective passes an
/// empty slice, because its Newton denominator is recoverable from the node's
/// weight total and accumulating a second histogram would be strictly wasted
/// work. Which arm runs is decided at compile time from the objective's
/// `CONSTANT_HESSIAN` declaration, so neither objective pays for the other's
/// statistics and the constant-hessian scan keeps exactly the two accumulators
/// it has always had.
///
/// A sample weight scales that row's gradient, its hessian, and its share of
/// every weight total, so the minimum leaf size counts weight rather than rows.
/// That is what makes an integer weight the same tree as repeating the row. The
/// bin grid is **not** weighted: it is fitted from the observed feature values,
/// which repeating a row does not change either.
pub(crate) fn grow_tree<O: BoostingObjective>(
    binned: &BinnedMatrix,
    binner: &Binner,
    statistics: SampleStatistics<'_>,
    config: GrowConfig,
) -> Result<CompactTree, BoostingError> {
    let SampleStatistics {
        negative_gradients,
        hessians,
        sample_weights,
    } = statistics;
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
    if !O::CONSTANT_HESSIAN {
        if hessians.len() != binned.rows() {
            return Err(BoostingError::ResidualLength {
                rows: binned.rows(),
                residuals: hessians.len(),
            });
        }
        if let Some(index) = hessians.iter().position(|value| !value.is_finite()) {
            return Err(BoostingError::NonFiniteResidual { index });
        }
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
    let root_totals = node_totals::<O>(&root_samples, statistics);
    let root_value = leaf_value::<O>(
        &root_samples,
        statistics,
        root_totals,
        config.l2_regularization,
    );
    let mut workspace = SplitWorkspace::new();
    let root_candidate = best_split::<O>(
        binned,
        statistics,
        &root_samples,
        root_totals,
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
        let left_totals = node_totals::<O>(&left_samples, statistics);
        let right_totals = node_totals::<O>(&right_samples, statistics);
        debug_assert!(left_totals.weight >= config.min_samples_leaf as f64);
        debug_assert!(right_totals.weight >= config.min_samples_leaf as f64);
        let depth = nodes[node_index].depth + 1;
        let left = nodes.len();
        let right = left + 1;
        let left_value = leaf_value::<O>(
            &left_samples,
            statistics,
            left_totals,
            config.l2_regularization,
        );
        let right_value = leaf_value::<O>(
            &right_samples,
            statistics,
            right_totals,
            config.l2_regularization,
        );
        let left_candidate = if config.max_depth.is_none_or(|max_depth| depth < max_depth) {
            best_split::<O>(
                binned,
                statistics,
                &left_samples,
                left_totals,
                config,
                &mut workspace,
            )
        } else {
            None
        };
        let right_candidate = if config.max_depth.is_none_or(|max_depth| depth < max_depth) {
            best_split::<O>(
                binned,
                statistics,
                &right_samples,
                right_totals,
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

fn best_split<O: BoostingObjective>(
    binned: &BinnedMatrix,
    statistics: SampleStatistics<'_>,
    samples: &[usize],
    totals: NodeTotals,
    config: GrowConfig,
    workspace: &mut SplitWorkspace,
) -> Option<SplitCandidate> {
    let SampleStatistics {
        negative_gradients,
        hessians,
        sample_weights,
    } = statistics;
    let total_weight = totals.weight;
    if total_weight < config.min_samples_leaf.checked_mul(2)? as f64 {
        return None;
    }
    let total_sum = negative_gradient_sum(samples, negative_gradients, sample_weights);
    let parent_score = score::<O>(total_sum, totals, config.l2_regularization);
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
        let (bin_weights, sums, bin_hessians) = workspace.reset(bin_count, O::CONSTANT_HESSIAN);
        if O::CONSTANT_HESSIAN {
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
        } else {
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
                        bin_hessians[bin] += f64::from(hessians[sample]);
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
                        bin_hessians[bin] += weight * f64::from(hessians[sample]);
                    }
                }
            }
        }
        let mut left_weight = 0.0_f64;
        let mut left_sum = 0.0_f64;
        let mut left_hessian = 0.0_f64;
        let minimum = config.min_samples_leaf as f64;
        for threshold in 0..bin_count - 1 {
            left_weight += bin_weights[threshold];
            left_sum += sums[threshold];
            if !O::CONSTANT_HESSIAN {
                left_hessian += bin_hessians[threshold];
            }
            let right_weight = total_weight - left_weight;
            if left_weight < minimum || right_weight < minimum {
                continue;
            }
            let right_sum = total_sum - left_sum;
            let left_totals = NodeTotals {
                weight: left_weight,
                hessian_sum: left_hessian,
            };
            let right_totals = NodeTotals {
                weight: right_weight,
                hessian_sum: totals.hessian_sum - left_hessian,
            };
            let gain = score::<O>(left_sum, left_totals, config.l2_regularization)
                + score::<O>(right_sum, right_totals, config.l2_regularization)
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
    /// Total hessian per bin, used only by an objective whose hessian varies.
    ///
    /// The allocation is unconditional and one scan's worth; leaving it out for
    /// squared error would trade a fixed 2 KiB for a branch on every reset.
    /// What squared error must not pay is the *per-sample* accumulation, and
    /// that is what the compile-time arm above removes.
    bin_hessians: Vec<f64>,
}

impl SplitWorkspace {
    fn new() -> Self {
        Self {
            bin_weights: vec![0.0; MAX_BINS],
            sums: vec![0.0; MAX_BINS],
            bin_hessians: vec![0.0; MAX_BINS],
        }
    }

    /// Clears and lends the per-bin accumulators for one feature scan.
    ///
    /// `varying_hessian` is the caller's compile-time constant, so a
    /// constant-hessian grower clears exactly the two buffers it fills and
    /// receives an empty third slice it cannot index by accident.
    fn reset(
        &mut self,
        bin_count: usize,
        constant_hessian: bool,
    ) -> (&mut [f64], &mut [f64], &mut [f64]) {
        debug_assert!(bin_count <= MAX_BINS);
        let bin_weights = &mut self.bin_weights[..bin_count];
        let sums = &mut self.sums[..bin_count];
        bin_weights.fill(0.0);
        sums.fill(0.0);
        let bin_hessians = if constant_hessian {
            &mut self.bin_hessians[..0]
        } else {
            let bin_hessians = &mut self.bin_hessians[..bin_count];
            bin_hessians.fill(0.0);
            bin_hessians
        };
        (bin_weights, sums, bin_hessians)
    }
}

/// The weight and curvature totals of one node.
///
/// `weight` bounds the minimum leaf size and `hessian_sum` is the Newton
/// denominator's numerator-free half. They are carried together because a split
/// derives both from the same prefix scan, and because keeping them in one value
/// makes it impossible to pass a left node's weight beside a parent's curvature.
#[derive(Clone, Copy, Debug)]
struct NodeTotals {
    weight: f64,
    /// Weighted sum of per-sample hessians; unused where the hessian is
    /// constant, and left at zero there rather than computed and discarded.
    hessian_sum: f64,
}

/// Totals of one node, computed over its samples.
///
/// The weight arm is unchanged: unweighted it is the row count, produced without
/// a pass over the samples, so an unweighted fit computes exactly the quantity it
/// always did. The curvature arm runs only where the hessian varies.
fn node_totals<O: BoostingObjective>(
    samples: &[usize],
    statistics: SampleStatistics<'_>,
) -> NodeTotals {
    NodeTotals {
        weight: match statistics.sample_weights {
            None => samples.len() as f64,
            Some(weights) => sum_in_order(samples.iter().map(|&sample| f64::from(weights[sample]))),
        },
        hessian_sum: if O::CONSTANT_HESSIAN {
            0.0
        } else {
            hessian_sum(samples, statistics.hessians, statistics.sample_weights)
        },
    }
}

fn score<O: BoostingObjective>(sum: f64, totals: NodeTotals, l2_regularization: f32) -> f64 {
    newton_split_score(
        sum,
        O::node_hessian_total(totals.weight, totals.hessian_sum),
        l2_regularization,
    )
}

fn leaf_value<O: BoostingObjective>(
    samples: &[usize],
    statistics: SampleStatistics<'_>,
    totals: NodeTotals,
    l2_regularization: f32,
) -> f32 {
    newton_leaf_value(
        negative_gradient_sum(
            samples,
            statistics.negative_gradients,
            statistics.sample_weights,
        ),
        O::node_hessian_total(totals.weight, totals.hessian_sum),
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
    use crate::loss::{BinaryLogLoss, Objective, SquaredError};

    /// Statistics for a constant-hessian objective: no curvature slice.
    fn scalar(negative_gradients: &[f32]) -> SampleStatistics<'_> {
        SampleStatistics {
            negative_gradients,
            hessians: &[],
            sample_weights: None,
        }
    }

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
        let binner = Binner::fit(&data.as_view(), 8, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let residuals = [-2.0, -2.0, -2.0, -2.0, 3.0, 3.0, 3.0, 3.0];
        let first =
            grow_tree::<SquaredError>(&binned, &binner, scalar(&residuals), config()).unwrap();
        let second =
            grow_tree::<SquaredError>(&binned, &binner, scalar(&residuals), config()).unwrap();
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
        let binner = Binner::fit(&data.as_view(), 4, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let tree = grow_tree::<SquaredError>(
            &binned,
            &binner,
            scalar(&[-1.0, -1.0, 1.0, 1.0]),
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
        let binner = Binner::fit(&data.as_view(), 4, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let residuals = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
        let tree = grow_tree::<SquaredError>(
            &binned,
            &binner,
            scalar(&residuals),
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

    /// The varying-hessian arm divides by the accumulated curvature, not by the
    /// row count, and the two disagree once the raw scores stop being equal.
    #[test]
    fn a_varying_hessian_leaf_divides_by_its_own_curvature() {
        let data = data();
        let binner = Binner::fit(&data.as_view(), 8, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        // Deliberately unequal raw scores, so `p(1 - p)` differs per row.
        let raws = [-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0, 0.5, -0.5];
        let targets = [0.0_f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let gradients = raws
            .iter()
            .zip(targets)
            .map(|(&raw, target)| BinaryLogLoss::negative_gradient(raw, target) as f32)
            .collect::<Vec<_>>();
        let hessians = raws
            .iter()
            .map(|&raw| BinaryLogLoss::hessian(raw, 0.0) as f32)
            .collect::<Vec<_>>();
        let tree = grow_tree::<BinaryLogLoss>(
            &binned,
            &binner,
            SampleStatistics {
                negative_gradients: &gradients,
                hessians: &hessians,
                sample_weights: None,
            },
            GrowConfig {
                max_leaf_nodes: 2,
                max_depth: Some(1),
                min_samples_leaf: 4,
                l2_regularization: 0.0,
            },
        )
        .unwrap();
        assert_eq!(tree.nodes().len(), 3);

        let leaf = |range: std::ops::Range<usize>| {
            let samples = range.collect::<Vec<_>>();
            newton_leaf_value(
                negative_gradient_sum(&samples, &gradients, None),
                hessian_sum(&samples, &hessians, None),
                0.0,
            )
        };
        assert_eq!(tree.predict_one(&[0.0]), leaf(0..4));
        assert_eq!(tree.predict_one(&[7.0]), leaf(4..8));
        // A count denominator would give a different leaf: the curvature of
        // these four rows is not four times any one of them.
        let counted = newton_leaf_value(
            negative_gradient_sum(&[0, 1, 2, 3], &gradients, None),
            4.0,
            0.0,
        );
        assert_ne!(tree.predict_one(&[0.0]), counted);
    }

    /// Weighting is the same identity for the curvature half as for the
    /// gradient half, so an integer weight is still a repeated row.
    #[test]
    fn a_varying_hessian_integer_weight_is_the_same_tree_as_repeating_the_row() {
        let raws = [-1.5_f64, 0.25, 1.0, 2.5];
        let targets = [0.0_f64, 1.0, 0.0, 1.0];
        let statistic = |index: usize| {
            (
                BinaryLogLoss::negative_gradient(raws[index], targets[index]) as f32,
                BinaryLogLoss::hessian(raws[index], 0.0) as f32,
            )
        };
        let config = GrowConfig {
            max_leaf_nodes: 2,
            max_depth: Some(1),
            min_samples_leaf: 1,
            l2_regularization: 0.5,
        };

        let features = [0.0_f32, 1.0, 2.0, 3.0];
        let data = DenseMatrix::new(features.to_vec(), 4, 1).unwrap();
        let binner = Binner::fit(&data.as_view(), 4, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        let gradients = (0..4).map(|row| statistic(row).0).collect::<Vec<_>>();
        let hessians = (0..4).map(|row| statistic(row).1).collect::<Vec<_>>();
        let weighted = grow_tree::<BinaryLogLoss>(
            &binned,
            &binner,
            SampleStatistics {
                negative_gradients: &gradients,
                hessians: &hessians,
                sample_weights: Some(&[1.0, 3.0, 1.0, 1.0]),
            },
            config,
        )
        .unwrap();

        let repeated_rows = [0_usize, 1, 1, 1, 2, 3];
        let repeated_features = repeated_rows
            .iter()
            .map(|&row| features[row])
            .collect::<Vec<_>>();
        let repeated_data = DenseMatrix::new(repeated_features, repeated_rows.len(), 1).unwrap();
        let repeated_binner = Binner::fit(&repeated_data.as_view(), 4, None).unwrap();
        let repeated_binned = repeated_binner.transform(&repeated_data.as_view()).unwrap();
        let repeated_gradients = repeated_rows
            .iter()
            .map(|&row| statistic(row).0)
            .collect::<Vec<_>>();
        let repeated_hessians = repeated_rows
            .iter()
            .map(|&row| statistic(row).1)
            .collect::<Vec<_>>();
        let repeated = grow_tree::<BinaryLogLoss>(
            &repeated_binned,
            &repeated_binner,
            SampleStatistics {
                negative_gradients: &repeated_gradients,
                hessians: &repeated_hessians,
                sample_weights: None,
            },
            config,
        )
        .unwrap();
        assert_eq!(weighted, repeated);

        // Unit weights reproduce the unweighted tree bit for bit.
        let unit = SampleStatistics {
            negative_gradients: &gradients,
            hessians: &hessians,
            sample_weights: Some(&[1.0; 4]),
        };
        let plain = SampleStatistics {
            sample_weights: None,
            ..unit
        };
        assert_eq!(
            grow_tree::<BinaryLogLoss>(&binned, &binner, unit, config).unwrap(),
            grow_tree::<BinaryLogLoss>(&binned, &binner, plain, config).unwrap()
        );
    }

    #[test]
    fn a_varying_hessian_grower_validates_its_curvature_slice() {
        let data = data();
        let binner = Binner::fit(&data.as_view(), 8, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        assert_eq!(
            grow_tree::<BinaryLogLoss>(
                &binned,
                &binner,
                SampleStatistics {
                    negative_gradients: &[0.1; 8],
                    hessians: &[0.25; 3],
                    sample_weights: None,
                },
                config()
            ),
            Err(BoostingError::ResidualLength {
                rows: 8,
                residuals: 3
            })
        );
        let mut hessians = [0.25_f32; 8];
        hessians[5] = f32::INFINITY;
        assert_eq!(
            grow_tree::<BinaryLogLoss>(
                &binned,
                &binner,
                SampleStatistics {
                    negative_gradients: &[0.1; 8],
                    hessians: &hessians,
                    sample_weights: None,
                },
                config()
            ),
            Err(BoostingError::NonFiniteResidual { index: 5 })
        );
    }

    #[test]
    fn validates_residuals_and_growth_configuration() {
        let data = data();
        let binner = Binner::fit(&data.as_view(), 4, None).unwrap();
        let binned = binner.transform(&data.as_view()).unwrap();
        assert_eq!(
            grow_tree::<SquaredError>(&binned, &binner, scalar(&[1.0]), config()),
            Err(BoostingError::ResidualLength {
                rows: 8,
                residuals: 1
            })
        );
        let mut residuals = [0.0; 8];
        residuals[3] = f32::NAN;
        assert_eq!(
            grow_tree::<SquaredError>(&binned, &binner, scalar(&residuals), config()),
            Err(BoostingError::NonFiniteResidual { index: 3 })
        );
        assert_eq!(
            grow_tree::<SquaredError>(
                &binned,
                &binner,
                scalar(&[0.0; 8]),
                GrowConfig {
                    max_leaf_nodes: 1,
                    ..config()
                }
            ),
            Err(BoostingError::InvalidMaxLeafNodes)
        );
    }
}
