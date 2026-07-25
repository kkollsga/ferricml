//! The growth controls every boosted estimator in this family shares.
//!
//! Both boosted estimators expose the same seven parameters with the same
//! meaning, and only the objective differs. Their public parameter types stay
//! separate — a classifier's builder must not accept a regressor's — but the
//! validation, the bound arithmetic, and the translation into a grow
//! configuration are one implementation, borrowed through [`BoostingControls`].
//!
//! Sharing the *checks* rather than the *types* is deliberate. A single shared
//! parameter struct would put a regressor's `Debug` output and its private
//! layout at the mercy of a classifier's needs, while two copies of these rules
//! would let the two estimators drift into accepting different parameters for
//! the same documented name.

use super::error::{
    BoostingError, MAX_BINS, MAX_TOTAL_NODES, MAX_TREE_DEPTH, MAX_TREE_LEAVES, MAX_TREES,
};
use super::grower::GrowConfig;
use crate::api::ModelError;

/// A borrowed view of the seven controls shared by both boosted estimators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct BoostingControls {
    pub(super) learning_rate: f32,
    pub(super) max_iter: usize,
    pub(super) max_leaf_nodes: usize,
    pub(super) max_depth: Option<usize>,
    pub(super) min_samples_leaf: usize,
    pub(super) l2_regularization: f32,
    pub(super) max_bins: usize,
}

impl BoostingControls {
    /// The per-tree growth configuration these controls describe.
    pub(super) const fn grow_config(self) -> GrowConfig {
        GrowConfig {
            max_leaf_nodes: self.max_leaf_nodes,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
        }
    }
}

/// Rejects any parameter outside its documented range.
///
/// Every check runs before a bin grid is fitted or a tree allocated, so an
/// invalid configuration costs no training work.
pub(super) fn validate_controls(controls: BoostingControls) -> Result<(), ModelError> {
    if !controls.learning_rate.is_finite() || controls.learning_rate <= 0.0 {
        return Err(ModelError::InvalidLearningRate);
    }
    if !(1..=MAX_TREES).contains(&controls.max_iter) {
        return Err(ModelError::InvalidBoostingIterationCount);
    }
    if !(2..=MAX_TREE_LEAVES).contains(&controls.max_leaf_nodes) {
        return Err(ModelError::InvalidMaxLeafNodes);
    }
    if controls
        .max_depth
        .is_some_and(|max_depth| !(1..=MAX_TREE_DEPTH).contains(&max_depth))
    {
        return Err(ModelError::InvalidBoostingMaxDepth);
    }
    if controls.min_samples_leaf == 0 {
        return Err(ModelError::InvalidMinSamplesLeaf);
    }
    if !controls.l2_regularization.is_finite() || controls.l2_regularization < 0.0 {
        return Err(ModelError::InvalidL2Regularization);
    }
    if !(2..=MAX_BINS).contains(&controls.max_bins) {
        return Err(ModelError::InvalidMaxBins);
    }
    Ok(())
}

/// Refuses a configuration whose largest possible ensemble exceeds the format.
///
/// The bound is computed from the parameters alone, so it is answered before
/// fitting rather than discovered part-way through it.
pub(super) fn validate_control_bounds(controls: BoostingControls) -> Result<(), ModelError> {
    let maximum_nodes = controls
        .max_leaf_nodes
        .checked_mul(2)
        .and_then(|nodes| nodes.checked_sub(1))
        .and_then(|nodes| nodes.checked_mul(controls.max_iter))
        .ok_or(ModelError::BoostingModelTooLarge)?;
    if maximum_nodes > MAX_TOTAL_NODES {
        return Err(ModelError::BoostingModelTooLarge);
    }
    Ok(())
}

pub(super) fn map_boosting_error(error: BoostingError) -> ModelError {
    match error {
        BoostingError::InvalidMaxBins => ModelError::InvalidMaxBins,
        BoostingError::InvalidMaxLeafNodes => ModelError::InvalidMaxLeafNodes,
        BoostingError::InvalidMaxDepth => ModelError::InvalidBoostingMaxDepth,
        BoostingError::InvalidMinSamplesLeaf => ModelError::InvalidMinSamplesLeaf,
        BoostingError::InvalidL2Regularization => ModelError::InvalidL2Regularization,
        BoostingError::FeatureDimension { expected, actual } => {
            ModelError::FeatureDimension { expected, actual }
        }
        BoostingError::TooManyFeatures => ModelError::TooManyFeatures,
        BoostingError::TreeTooLarge | BoostingError::InvalidTree => ModelError::TreeTooLarge,
        BoostingError::ResidualLength { .. } | BoostingError::NonFiniteResidual { .. } => {
            ModelError::NumericalOverflow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controls() -> BoostingControls {
        BoostingControls {
            learning_rate: 0.1,
            max_iter: 100,
            max_leaf_nodes: 31,
            max_depth: None,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            max_bins: 255,
        }
    }

    #[test]
    fn every_control_is_rejected_at_its_own_boundary() {
        let cases: [(BoostingControls, ModelError); 9] = [
            (
                BoostingControls {
                    learning_rate: 0.0,
                    ..controls()
                },
                ModelError::InvalidLearningRate,
            ),
            (
                BoostingControls {
                    learning_rate: f32::NAN,
                    ..controls()
                },
                ModelError::InvalidLearningRate,
            ),
            (
                BoostingControls {
                    max_iter: 0,
                    ..controls()
                },
                ModelError::InvalidBoostingIterationCount,
            ),
            (
                BoostingControls {
                    max_leaf_nodes: 1,
                    ..controls()
                },
                ModelError::InvalidMaxLeafNodes,
            ),
            (
                BoostingControls {
                    max_depth: Some(0),
                    ..controls()
                },
                ModelError::InvalidBoostingMaxDepth,
            ),
            (
                BoostingControls {
                    min_samples_leaf: 0,
                    ..controls()
                },
                ModelError::InvalidMinSamplesLeaf,
            ),
            (
                BoostingControls {
                    l2_regularization: -1.0,
                    ..controls()
                },
                ModelError::InvalidL2Regularization,
            ),
            (
                BoostingControls {
                    max_bins: 1,
                    ..controls()
                },
                ModelError::InvalidMaxBins,
            ),
            (
                BoostingControls {
                    max_bins: MAX_BINS + 1,
                    ..controls()
                },
                ModelError::InvalidMaxBins,
            ),
        ];
        for (controls, expected) in cases {
            assert_eq!(validate_controls(controls), Err(expected));
        }
        assert_eq!(validate_controls(controls()), Ok(()));
    }

    #[test]
    fn the_aggregate_node_bound_is_answered_from_the_parameters_alone() {
        assert_eq!(validate_control_bounds(controls()), Ok(()));
        assert_eq!(
            validate_control_bounds(BoostingControls {
                max_iter: MAX_TREES,
                max_leaf_nodes: MAX_TREE_LEAVES,
                ..controls()
            }),
            Err(ModelError::BoostingModelTooLarge)
        );
    }

    #[test]
    fn the_grow_configuration_carries_exactly_the_per_tree_controls() {
        let controls = BoostingControls {
            max_leaf_nodes: 7,
            max_depth: Some(3),
            min_samples_leaf: 4,
            l2_regularization: 1.5,
            ..controls()
        };
        assert_eq!(
            controls.grow_config(),
            GrowConfig {
                max_leaf_nodes: 7,
                max_depth: Some(3),
                min_samples_leaf: 4,
                l2_regularization: 1.5,
            }
        );
    }
}
