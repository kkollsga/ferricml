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
    "",
    "```",
    "use ferricml::api::Classifier;",
    "use ferricml::data::{BinaryTargets, DenseMatrix};",
    "use ferricml::ensemble::{RandomForestClassifier, RandomForestClassifierParams};",
    "",
    "// Bootstrapping is on by default, so each tree sees a resample of the",
    "// rows. A handful of rows is not enough for that to average out.",
    "let values: Vec<f32> = (0..40).map(|index| index as f32).collect();",
    "let labels: Vec<u8> = (0..40).map(|index| u8::from(index >= 20)).collect();",
    "let data = DenseMatrix::new(values, 40, 1)?;",
    "let labels = BinaryTargets::new(labels)?;",
    "",
    "let model = RandomForestClassifier::fit(",
    "    &data.as_view(),",
    "    &labels,",
    "    RandomForestClassifierParams::default()",
    "        .with_n_estimators(16)",
    "        .with_random_state(7),",
    ")?;",
    "",
    "assert_eq!(model.predict(&data.as_view())?, labels.as_slice().to_vec());",
    "",
    "// A seeded fit is reproducible: identical parameters give identical",
    "// predictions, which is the contract every fitted artifact rests on.",
    "let again = RandomForestClassifier::fit(",
    "    &data.as_view(),",
    "    &labels,",
    "    RandomForestClassifierParams::default()",
    "        .with_n_estimators(16)",
    "        .with_random_state(7),",
    ")?;",
    "assert_eq!(model.predict_proba(&data.as_view())?, again.predict_proba(&data.as_view())?);",
    "# Ok::<(), Box<dyn std::error::Error>>(())",
    "```",
);

forest_regressor!(
    RandomForestRegressor,
    RandomForestRegressorParams,
    RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
    crate::tree::Regression,
    "A random-forest regressor.  Predictions are averages of tree leaf means.",
    "",
    "```",
    "use ferricml::data::{DenseMatrix, RegressionTargets};",
    "use ferricml::ensemble::{NJobs, RandomForestRegressor, RandomForestRegressorParams};",
    "",
    "let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;",
    "let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0])?;",
    "",
    "let params = RandomForestRegressorParams::default()",
    "    .with_n_estimators(8)",
    "    .with_random_state(11);",
    "let serial = RandomForestRegressor::fit(&data.as_view(), &targets, params.clone())?;",
    "",
    "// Worker count does not change the fit. Tree `i`'s seed comes from `i`",
    "// alone and finished trees are sorted back into index order, so a",
    "// parallel fit is the same model as a serial one rather than merely a",
    "// similar one.",
    "let parallel = RandomForestRegressor::fit(",
    "    &data.as_view(),",
    "    &targets,",
    "    params.with_n_jobs(NJobs::Count(4)),",
    ")?;",
    "assert_eq!(",
    "    serial.predict(&data.as_view())?,",
    "    parallel.predict(&data.as_view())?,",
    ");",
    "# Ok::<(), Box<dyn std::error::Error>>(())",
    "```",
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
