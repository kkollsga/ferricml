use super::super::forest::{forest_classifier, forest_regressor};
use super::parameters::{RandomForestClassifierParams, RandomForestRegressorParams};
use crate::artifact::{
    RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND, RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
};

forest_classifier!(
    RandomForestClassifier,
    RandomForestClassifierParams,
    RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND,
    "A random-forest classifier.",
);

forest_regressor!(
    RandomForestRegressor,
    RandomForestRegressorParams,
    RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
    crate::tree::Regression,
    "A random-forest regressor.  Predictions are averages of tree leaf means.",
);

/// Internal bytes used only for deterministic implementation tests.
#[cfg(test)]
impl RandomForestRegressor {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        packed_model_bytes(self.core.n_features_in, &self.core.trees, b"FRFR")
    }
}

/// The same, for a binary classifier fit.
#[cfg(test)]
impl RandomForestClassifier {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        packed_model_bytes(self.core.n_features_in, self.binary_trees(), b"FRFC")
    }
}

#[cfg(test)]
fn packed_model_bytes(
    n_features: usize,
    trees: &[crate::tree::PackedTree],
    magic: &[u8; 4],
) -> Vec<u8> {
    let node_count: usize = trees.iter().map(|tree| tree.nodes.len()).sum();
    let mut bytes = Vec::with_capacity(24 + trees.len() * 13 + node_count * 16);
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(n_features as u64).to_le_bytes());
    bytes.extend_from_slice(&(trees.len() as u64).to_le_bytes());
    for tree in trees {
        bytes.push(u8::from(tree.root_leaf.is_some()));
        bytes.extend_from_slice(&tree.root_leaf.unwrap_or_default().to_bits().to_le_bytes());
        bytes.extend_from_slice(&(tree.nodes.len() as u64).to_le_bytes());
        for node in &tree.nodes {
            bytes.extend_from_slice(&node.feature_and_flags.to_le_bytes());
            bytes.extend_from_slice(&node.left.to_le_bytes());
            bytes.extend_from_slice(&node.right.to_le_bytes());
            bytes.extend_from_slice(&node.threshold.to_bits().to_le_bytes());
        }
    }
    bytes
}
