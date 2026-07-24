use super::parameters::{
    MaxFeatures, NJobs, RandomForestClassifierParams, RandomForestRegressorParams,
};
use super::training::{Classification, ForestConfig, Regression, train_forest};
use super::tree::{FEATURE_MASK, PackedTree};
use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, HasParams, ModelError, Regressor,
    validate_prediction,
};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND, SchemaRole,
    decode_component, decode_logical_tree, decode_v2_envelope, encode_component,
    encode_logical_tree, encode_v2_envelope,
};
use crate::data::{BinaryTargets, MatrixView, RegressionTargets};

const REGRESSOR_PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;
const REGRESSOR_OBJECTIVE_VERSION: u32 = 1;
const METADATA_BYTES: usize = 13 * 4 + 8;

/// Ceilings applied identically when encoding and decoding, so an artifact
/// that this crate produced always decodes and a hostile one allocates
/// nothing unbounded.
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const MAX_ARTIFACT_TREES: usize = 4_096;
const MAX_ARTIFACT_TOTAL_NODES: usize = 1_048_576;

// A decoded feature index must never reach the packed layout's flag bits.
const _: () = assert!(MAX_ARTIFACT_FEATURES < FEATURE_MASK as usize);

const MAX_FEATURES_ALL: u32 = 1;
const MAX_FEATURES_SQRT: u32 = 2;
const MAX_FEATURES_COUNT: u32 = 3;
const N_JOBS_SERIAL: u32 = 1;
const N_JOBS_ALL: u32 = 2;
const N_JOBS_COUNT: u32 = 3;

/// A random-forest binary classifier.
///
/// Class labels are sorted, and probability columns follow that order. Models
/// fitted on a single class expose one probability column containing `1.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestClassifier {
    pub(super) n_features_in: usize,
    pub(super) params: RandomForestClassifierParams,
    pub(super) classes: Vec<u8>,
    pub(super) trees: Vec<PackedTree>,
}

/// A random-forest regressor.  Predictions are averages of tree leaf means.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestRegressor {
    pub(super) n_features_in: usize,
    pub(super) params: RandomForestRegressorParams,
    pub(super) trees: Vec<PackedTree>,
}

impl RandomForestClassifier {
    /// Returns the feature width required by this model.
    #[inline]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact parameters used to fit this model.
    #[inline]
    pub fn get_params(&self) -> &RandomForestClassifierParams {
        &self.params
    }

    /// Returns sorted class labels observed during fitting.
    #[inline]
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.as_slice().len(), &config)?;
        for (index, &value) in targets.as_slice().iter().enumerate() {
            if value > 1 {
                return Err(ModelError::InvalidBinaryTarget { index, value });
            }
        }
        let saw_zero = targets.as_slice().contains(&0);
        let saw_one = targets.as_slice().contains(&1);
        let classes = match (saw_zero, saw_one) {
            (true, true) => vec![0, 1],
            (true, false) => vec![0],
            (false, true) => vec![1],
            (false, false) => unreachable!("non-empty validated binary targets"),
        };
        let trees = train_forest(data, targets.as_slice(), &config, Classification)?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            trees,
        })
    }

    /// Predicts the class label for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(self.classes[0]);
        }
        let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
        // `classes` is sorted as [0, 1]. An exact tie selects its first class.
        Ok(u8::from(positive > 0.5))
    }

    /// Predicts one label per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        if self.classes.len() == 1 {
            check_prediction_data(data, data.rows(), data.rows(), self.n_features_in)?;
            return Ok(vec![self.classes[0]; data.rows()]);
        }
        // The allocating API may use a temporary score buffer. Processing one
        // tree across the batch keeps its nodes hot and is materially faster
        // for the locked 32+ row workloads. The `_into` label API remains the
        // strictly allocation-free option.
        let mut scores = vec![0.0; data.rows()];
        self.accumulate_positive_into(data, &mut scores)?;
        Ok(scores
            .into_iter()
            .map(|positive| u8::from(positive > 0.5))
            .collect())
    }

    /// Predicts one label per row without allocating.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(self.classes[0]);
            return Ok(());
        }
        for (row, slot) in data.iter_rows().zip(output) {
            let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
            *slot = u8::from(positive > 0.5);
        }
        Ok(())
    }

    /// Predicts probabilities for one sample in [`Self::classes`] order.
    pub fn predict_proba_one(&self, row: &[f32]) -> Result<Vec<f32>, ModelError> {
        let mut output = vec![0.0; self.classes.len()];
        self.predict_proba_one_into(row, &mut output)?;
        Ok(output)
    }

    /// Predicts probabilities for one sample into caller-owned storage.
    pub fn predict_proba_one_into(
        &self,
        row: &[f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_row(row, self.n_features_in)?;
        check_output_len(output.len(), self.classes.len())?;
        if self.classes.len() == 1 {
            output[0] = 1.0;
        } else {
            let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
            output[0] = 1.0 - positive;
            output[1] = positive;
        }
        Ok(())
    }

    /// Predicts row-major probabilities, allocating `rows * classes().len()`
    /// values.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities without allocating.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let expected = probability_output_len(data.rows(), self.classes.len())?;
        check_prediction_data(data, output.len(), expected, self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(1.0);
            return Ok(());
        }
        output.fill(0.0);
        for tree in &self.trees {
            for (row, probabilities) in data.iter_rows().zip(output.chunks_exact_mut(2)) {
                probabilities[1] += tree.predict(row);
            }
        }
        for probabilities in output.chunks_exact_mut(2) {
            let positive = (probabilities[1] / self.trees.len() as f32).clamp(0.0, 1.0);
            probabilities[0] = 1.0 - positive;
            probabilities[1] = positive;
        }
        Ok(())
    }

    /// Returns the requested fitted-class probability for one sample.
    pub fn predict_class_proba_one(&self, row: &[f32], class: u8) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        let class_index = self.class_index(class)?;
        if self.classes.len() == 1 {
            return Ok(1.0);
        }
        let positive = mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0);
        Ok(if class_index == 0 {
            1.0 - positive
        } else {
            positive
        })
    }

    /// Predicts one fitted-class probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_class_proba(self, data, class)
    }

    /// Predicts one fitted-class probability column without allocating.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let class_index = self.class_index(class)?;
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(1.0);
            return Ok(());
        }
        self.accumulate_positive_into(data, output)?;
        if class_index == 0 {
            for slot in output {
                *slot = 1.0 - *slot;
            }
        }
        Ok(())
    }

    fn accumulate_positive_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        output.fill(0.0);
        for tree in &self.trees {
            for (row, slot) in data.iter_rows().zip(output.iter_mut()) {
                *slot += tree.predict(row);
            }
        }
        for slot in output {
            *slot = (*slot / self.trees.len() as f32).clamp(0.0, 1.0);
        }
        Ok(())
    }

    #[inline]
    fn class_index(&self, class: u8) -> Result<usize, ModelError> {
        self.classes
            .binary_search(&class)
            .map_err(|_| ModelError::UnknownClass { class })
    }

    /// Returns the positive-class probability for one sample.
    ///
    /// This explicit method preserves the Phase A probability behavior. The
    /// label and two-column probability methods land in Phase B.
    pub fn predict_positive_proba(&self, row: &[f32]) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(f32::from(self.classes[0]));
        }
        Ok(mean_tree_prediction(&self.trees, row).clamp(0.0, 1.0))
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub fn predict_positive_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(f32::from(self.classes[0]));
            return Ok(());
        }
        for (index, slot) in output.iter_mut().enumerate() {
            *slot =
                mean_tree_prediction(&self.trees, data.row(index).expect("validated row index"))
                    .clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Internal bytes used only for deterministic implementation tests.
    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        packed_model_bytes(self.n_features_in, &self.trees, b"FRFC")
    }
}

impl Estimator for RandomForestClassifier {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Classifier for RandomForestClassifier {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        RandomForestClassifier::predict_into(self, data, output)
    }

    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        RandomForestClassifier::predict_proba_into(self, data, output)
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        RandomForestClassifier::predict_class_proba_into(self, data, class, output)
    }
}

impl HasParams for RandomForestClassifier {
    type Params = RandomForestClassifierParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

/// Declares nothing: bootstrap resampling has no weighted entry point yet, and
/// the classifier has no artifact kind until leaf probability semantics are
/// frozen.
impl HasCapabilities for RandomForestClassifier {}

impl RandomForestRegressor {
    /// Returns the feature width required by this model.
    #[inline]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact parameters used to fit this model.
    #[inline]
    pub fn get_params(&self) -> &RandomForestRegressorParams {
        &self.params
    }

    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: RandomForestRegressorParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.as_slice().len(), &config)?;
        for (index, value) in targets.as_slice().iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::NonFiniteTarget { index });
            }
        }
        let trees = train_forest(data, targets.as_slice(), &config, Regression)?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            trees,
        })
    }

    /// Predicts one regression value for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        validate_prediction(mean_tree_prediction(&self.trees, row), 0)
    }

    /// Predicts one value per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = validate_prediction(
                mean_tree_prediction(&self.trees, data.row(index).expect("validated row index")),
                index,
            )?;
        }
        Ok(())
    }

    /// Encodes the fitted parameters and canonical logical trees.
    ///
    /// The private packed inference layout is never serialized. Each tree is
    /// expanded into stable logical records first, so the compact runtime
    /// representation stays free to change.
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let n_estimators =
            u32::try_from(self.params.n_estimators()).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_depth = self
            .params
            .max_depth()
            .map(u32::try_from)
            .transpose()
            .map_err(|_| ArtifactError::InvalidPayload)?
            .unwrap_or(0);
        let min_samples_split = u32::try_from(self.params.min_samples_split())
            .map_err(|_| ArtifactError::InvalidPayload)?;
        let min_samples_leaf = u32::try_from(self.params.min_samples_leaf())
            .map_err(|_| ArtifactError::InvalidPayload)?;
        let (max_features_tag, max_features_count) =
            encode_max_features(self.params.max_features())?;
        let (n_jobs_tag, n_jobs_count) = encode_n_jobs(self.params.n_jobs())?;
        let tree_count =
            u32::try_from(self.trees.len()).map_err(|_| ArtifactError::InvalidPayload)?;
        let total_nodes = self.trees.iter().try_fold(0_usize, |total, tree| {
            total
                .checked_add(tree.logical_node_count())
                .ok_or(ArtifactError::InvalidPayload)
        })?;
        if self.trees.len() > MAX_ARTIFACT_TREES
            || total_nodes > MAX_ARTIFACT_TOTAL_NODES
            || self.n_features_in > MAX_ARTIFACT_FEATURES
            || !prediction_bound_is_finite(&self.trees)
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let total_nodes = u32::try_from(total_nodes).map_err(|_| ArtifactError::InvalidPayload)?;

        let mut metadata = ArtifactPayloadWriter::with_capacity(METADATA_BYTES);
        metadata.u32(REGRESSOR_OBJECTIVE_VERSION);
        metadata.u32(n_features);
        metadata.u32(n_estimators);
        metadata.u32(max_depth);
        metadata.u32(min_samples_split);
        metadata.u32(min_samples_leaf);
        metadata.u32(max_features_tag);
        metadata.u32(max_features_count);
        metadata.u32(u32::from(self.params.bootstrap()));
        metadata.u64(self.params.random_state());
        metadata.u32(n_jobs_tag);
        metadata.u32(n_jobs_count);
        metadata.u32(tree_count);
        metadata.u32(total_nodes);
        let mut payload = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        for tree in &self.trees {
            payload.extend_from_slice(&encode_component(
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
                &encode_logical_tree(&tree.to_logical_nodes())?,
            )?);
        }
        encode_v2_envelope(
            RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
            REGRESSOR_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes and revalidates logical trees before building runtime state.
    ///
    /// Counts and parameters are checked before any tree is read, and each
    /// decoded tree is rebuilt through the same topology validator that
    /// fitting uses, so the encoded bytes are never trusted.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
            REGRESSOR_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let objective_version = metadata.u32()?;
        let n_features_in = metadata.u32()? as usize;
        let n_estimators = metadata.u32()? as usize;
        let encoded_depth = metadata.u32()? as usize;
        let min_samples_split = metadata.u32()? as usize;
        let min_samples_leaf = metadata.u32()? as usize;
        let max_features_tag = metadata.u32()?;
        let max_features_count = metadata.u32()?;
        let bootstrap = metadata.u32()?;
        let random_state = metadata.u64()?;
        let n_jobs_tag = metadata.u32()?;
        let n_jobs_count = metadata.u32()?;
        let tree_count = metadata.u32()? as usize;
        let declared_total_nodes = metadata.u32()? as usize;
        if !metadata.is_empty()
            || objective_version != REGRESSOR_OBJECTIVE_VERSION
            || n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || n_estimators == 0
            || n_estimators != tree_count
            || tree_count > MAX_ARTIFACT_TREES
            || encoded_depth > MAX_ARTIFACT_TOTAL_NODES
            || min_samples_split < 2
            || min_samples_leaf == 0
            || bootstrap > 1
            || declared_total_nodes < tree_count
            || declared_total_nodes > MAX_ARTIFACT_TOTAL_NODES
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let (Some(max_features), Some(n_jobs)) = (
            decode_max_features(max_features_tag, max_features_count, n_features_in),
            decode_n_jobs(n_jobs_tag, n_jobs_count),
        ) else {
            return Err(ArtifactError::InvalidPayload);
        };
        let params = RandomForestRegressorParams::default()
            .with_n_estimators(n_estimators)
            .with_max_depth((encoded_depth != 0).then_some(encoded_depth))
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_max_features(max_features)
            .with_bootstrap(bootstrap == 1)
            .with_random_state(random_state)
            .with_n_jobs(n_jobs);

        let mut trees = Vec::with_capacity(tree_count);
        let mut actual_total_nodes = 0_usize;
        for _ in 0..tree_count {
            let logical = decode_logical_tree(decode_component(
                &mut envelope,
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
            )?)?;
            actual_total_nodes = actual_total_nodes
                .checked_add(logical.len())
                .ok_or(ArtifactError::InvalidPayload)?;
            if actual_total_nodes > declared_total_nodes {
                return Err(ArtifactError::InvalidPayload);
            }
            trees.push(PackedTree::from_logical_nodes(&logical, n_features_in)?);
        }
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        if actual_total_nodes != declared_total_nodes || !prediction_bound_is_finite(&trees) {
            return Err(ArtifactError::InvalidPayload);
        }
        Ok(Self {
            n_features_in,
            params,
            trees,
        })
    }

    /// Internal bytes used only for deterministic implementation tests.
    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        packed_model_bytes(self.n_features_in, &self.trees, b"FRFR")
    }
}

fn encode_max_features(value: MaxFeatures) -> Result<(u32, u32), ArtifactError> {
    Ok(match value {
        MaxFeatures::All => (MAX_FEATURES_ALL, 0),
        MaxFeatures::Sqrt => (MAX_FEATURES_SQRT, 0),
        MaxFeatures::Count(count) => (
            MAX_FEATURES_COUNT,
            u32::try_from(count).map_err(|_| ArtifactError::InvalidPayload)?,
        ),
    })
}

fn decode_max_features(tag: u32, count: u32, n_features: usize) -> Option<MaxFeatures> {
    match tag {
        MAX_FEATURES_ALL if count == 0 => Some(MaxFeatures::All),
        MAX_FEATURES_SQRT if count == 0 => Some(MaxFeatures::Sqrt),
        MAX_FEATURES_COUNT if count != 0 && count as usize <= n_features => {
            Some(MaxFeatures::Count(count as usize))
        }
        _ => None,
    }
}

fn encode_n_jobs(value: NJobs) -> Result<(u32, u32), ArtifactError> {
    Ok(match value {
        NJobs::Serial => (N_JOBS_SERIAL, 0),
        NJobs::All => (N_JOBS_ALL, 0),
        NJobs::Count(count) => (
            N_JOBS_COUNT,
            u32::try_from(count).map_err(|_| ArtifactError::InvalidPayload)?,
        ),
    })
}

fn decode_n_jobs(tag: u32, count: u32) -> Option<NJobs> {
    match tag {
        N_JOBS_SERIAL if count == 0 => Some(NJobs::Serial),
        N_JOBS_ALL if count == 0 => Some(NJobs::All),
        N_JOBS_COUNT if count != 0 => Some(NJobs::Count(count as usize)),
        _ => None,
    }
}

/// Whether averaging every tree can stay inside `f32`.
///
/// Prediction sums leaf values before dividing by the tree count, so the
/// bound is the sum of per-tree leaf magnitudes rather than their mean.
fn prediction_bound_is_finite(trees: &[PackedTree]) -> bool {
    let mut bound = 0.0_f64;
    for tree in trees {
        bound += f64::from(tree.max_abs_leaf());
        if !bound.is_finite() || bound > f64::from(f32::MAX) {
            return false;
        }
    }
    true
}

impl Estimator for RandomForestRegressor {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Regressor for RandomForestRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        RandomForestRegressor::predict_into(self, data, output)
    }
}

impl HasParams for RandomForestRegressor {
    type Params = RandomForestRegressorParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl HasCapabilities for RandomForestRegressor {
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_artifact(true);
}

fn mean_tree_prediction(trees: &[PackedTree], row: &[f32]) -> f32 {
    let sum: f32 = trees.iter().map(|tree| tree.predict(row)).sum();
    sum / trees.len() as f32
}

fn check_row(row: &[f32], expected: usize) -> Result<(), ModelError> {
    if row.len() != expected {
        return Err(ModelError::FeatureDimension {
            expected,
            actual: row.len(),
        });
    }
    if let Some(column) = row.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteFeature { row: 0, column });
    }
    Ok(())
}

fn check_prediction_data(
    data: &MatrixView<'_>,
    output_len: usize,
    expected_output_len: usize,
    expected_features: usize,
) -> Result<(), ModelError> {
    if data.columns() != expected_features {
        return Err(ModelError::FeatureDimension {
            expected: expected_features,
            actual: data.columns(),
        });
    }
    check_output_len(output_len, expected_output_len)
}

fn check_output_len(actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::OutputLength { expected, actual });
    }
    Ok(())
}

fn probability_output_len(rows: usize, classes: usize) -> Result<usize, ModelError> {
    rows.checked_mul(classes)
        .ok_or(ModelError::OutputShapeOverflow {
            rows,
            columns: classes,
        })
}

fn validate_common(
    data: &MatrixView<'_>,
    target_len: usize,
    config: &ForestConfig,
) -> Result<(), ModelError> {
    if data.rows() == 0 || data.columns() == 0 {
        return Err(ModelError::EmptyData);
    }
    if target_len == 0 {
        return Err(ModelError::EmptyTargets);
    }
    if target_len != data.rows() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: target_len,
        });
    }
    if config.n_estimators == 0 {
        return Err(ModelError::InvalidEstimatorCount);
    }
    if config.max_depth == Some(0) {
        return Err(ModelError::InvalidMaxDepth);
    }
    if config.min_samples_split < 2 {
        return Err(ModelError::InvalidMinSamplesSplit);
    }
    if config.min_samples_leaf == 0 {
        return Err(ModelError::InvalidMinSamplesLeaf);
    }
    if config.n_jobs == 0 {
        return Err(ModelError::InvalidJobCount);
    }
    if data.rows() > u32::MAX as usize {
        return Err(ModelError::TooManyRows);
    }
    if data.columns() > FEATURE_MASK as usize {
        return Err(ModelError::TooManyFeatures);
    }
    if let MaxFeatures::Count(requested) = config.max_features
        && (requested == 0 || requested > data.columns())
    {
        return Err(ModelError::InvalidMaxFeatures {
            requested,
            available: data.columns(),
        });
    }
    for row in 0..data.rows() {
        for (column, value) in data
            .row(row)
            .expect("validated row index")
            .iter()
            .enumerate()
        {
            if !value.is_finite() {
                return Err(ModelError::NonFiniteFeature { row, column });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn packed_model_bytes(n_features: usize, trees: &[PackedTree], magic: &[u8; 4]) -> Vec<u8> {
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
