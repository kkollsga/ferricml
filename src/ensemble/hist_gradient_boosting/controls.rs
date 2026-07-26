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

/// Translates a private boosting failure into its public counterpart.
///
/// The mapping is injective, and the unit tests assert that rather than the
/// individual pairs: a lossy arm is invisible from the code that writes it, and
/// two such arms each cost a distinguishable failure. What the mapping does not
/// carry is the residual index, because no public FerricML error names a row or
/// an observation — the private error keeps it for a debugger, and the public
/// one reports the condition.
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
        BoostingError::TreeTooLarge => ModelError::TreeTooLarge,
        BoostingError::InvalidTree => ModelError::InvalidTreeStructure,
        BoostingError::ResidualLength { rows, residuals } => ModelError::OutputLength {
            expected: rows,
            actual: residuals,
        },
        BoostingError::NonFiniteResidual { .. } => ModelError::NumericalOverflow,
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

    /// One representative of every private boosting failure.
    ///
    /// The `match` below exists only to be exhaustive: adding a variant to
    /// [`BoostingError`] without adding it here stops compiling, so the list
    /// cannot silently fall behind the enum it claims to enumerate.
    fn every_boosting_error() -> Vec<BoostingError> {
        let all = vec![
            BoostingError::InvalidMaxBins,
            BoostingError::InvalidMaxLeafNodes,
            BoostingError::InvalidMaxDepth,
            BoostingError::InvalidMinSamplesLeaf,
            BoostingError::InvalidL2Regularization,
            BoostingError::ResidualLength {
                rows: 8,
                residuals: 5,
            },
            BoostingError::NonFiniteResidual { index: 3 },
            BoostingError::FeatureDimension {
                expected: 4,
                actual: 7,
            },
            BoostingError::TooManyFeatures,
            BoostingError::TreeTooLarge,
            BoostingError::InvalidTree,
        ];
        for error in &all {
            match error {
                BoostingError::InvalidMaxBins
                | BoostingError::InvalidMaxLeafNodes
                | BoostingError::InvalidMaxDepth
                | BoostingError::InvalidMinSamplesLeaf
                | BoostingError::InvalidL2Regularization
                | BoostingError::ResidualLength { .. }
                | BoostingError::NonFiniteResidual { .. }
                | BoostingError::FeatureDimension { .. }
                | BoostingError::TooManyFeatures
                | BoostingError::TreeTooLarge
                | BoostingError::InvalidTree => {}
            }
        }
        all
    }

    /// No two private failures may arrive as one public error.
    ///
    /// A lossy conversion is not visible from the mapping's own code, which is
    /// how `TreeTooLarge`/`InvalidTree` and `ResidualLength`/`NonFiniteResidual`
    /// each spent a release arriving as a single variant. Injectivity is the
    /// property that would have caught it, so it is asserted rather than the
    /// eleven individual pairs.
    #[test]
    fn distinct_boosting_failures_stay_distinct_through_the_public_mapping() {
        let all = every_boosting_error();
        let mut mapped: Vec<(BoostingError, ModelError)> = Vec::new();
        for error in all {
            let public = map_boosting_error(error.clone());
            if let Some((collision, _)) = mapped
                .iter()
                .find(|(_, existing)| existing == &public)
                .cloned()
            {
                panic!("{collision:?} and {error:?} both map to {public:?}");
            }
            mapped.push((error, public));
        }
        assert_eq!(mapped.len(), 11);
    }

    /// The two failures that were collapsed must keep the identity they had.
    ///
    /// Injectivity alone would be satisfied by mapping them to any two
    /// unrelated variants, so the meaning of each is pinned separately. Neither
    /// carries the row index the private error holds: no public FerricML error
    /// names an observation, and inventing that convention here was ruled out.
    #[test]
    fn the_uncollapsed_failures_report_what_they_are() {
        assert_eq!(
            map_boosting_error(BoostingError::ResidualLength {
                rows: 8,
                residuals: 5
            }),
            ModelError::OutputLength {
                expected: 8,
                actual: 5
            },
        );
        assert_eq!(
            map_boosting_error(BoostingError::NonFiniteResidual { index: 3 }),
            ModelError::NumericalOverflow,
        );
        assert_eq!(
            map_boosting_error(BoostingError::TreeTooLarge),
            ModelError::TreeTooLarge,
        );
        assert_eq!(
            map_boosting_error(BoostingError::InvalidTree),
            ModelError::InvalidTreeStructure,
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
