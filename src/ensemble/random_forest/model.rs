use super::parameters::{
    MaxFeatures, NJobs, RandomForestClassifierParams, RandomForestRegressorParams,
};
use super::training::{ForestConfig, train_class_forest, train_forest};
use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, HasParams, ModelError,
    ProbabilisticClassifier, Regressor, validate_prediction,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, LogicalTreeNode, MIN_ENCODED_TREE_BYTES,
    RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND, RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND, SchemaRole,
    decode_component, decode_logical_tree, decode_v2_envelope, encode_component,
    encode_logical_tree, encode_v2_envelope,
};
use crate::data::{BinaryTargets, ClassTargets, MatrixView, RegressionTargets, SampleWeights};
use crate::tree::{
    ClassTree, Classification, FEATURE_MASK, PackedTree, Regression, decode_max_features,
    encode_max_features,
};

const REGRESSOR_PAYLOAD_VERSION: u16 = 1;
const CLASSIFIER_PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
/// Per-tree leaf distributions, in pre-order leaf rank. Written only by the
/// multiclass flavour, immediately after that tree's topology component.
const LEAF_PROBABILITY_COMPONENT_KIND: u16 = 3;
const COMPONENT_VERSION: u16 = 1;
const REGRESSOR_OBJECTIVE_VERSION: u32 = 1;
const CLASSIFIER_OBJECTIVE_VERSION: u32 = 1;
const METADATA_BYTES: usize = 13 * 4 + 8;
/// The classifier metadata's fixed words, before its class list: the regressor
/// fields plus a forest-flavour tag and a class count.
const CLASSIFIER_METADATA_BYTES: usize = 15 * 4 + 8;

/// Which leaf arithmetic the encoded forest uses. The two are different models,
/// so the tag is read before any tree is, and neither flavour's trees are ever
/// handed to the other's builder.
const FOREST_BINARY: u32 = 1;
const FOREST_MULTICLASS: u32 = 2;

/// Ceilings applied identically when encoding and decoding, so an artifact
/// that this crate produced always decodes and a hostile one allocates
/// nothing unbounded.
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const MAX_ARTIFACT_TREES: usize = 4_096;
const MAX_ARTIFACT_TOTAL_NODES: usize = 1_048_576;

// A decoded feature index must never reach the packed layout's flag bits.
const _: () = assert!(MAX_ARTIFACT_FEATURES < FEATURE_MASK as usize);

/// Every class label is a `u8`, so no fit can observe more classes than this.
/// Scalar and single-column prediction paths keep one averaged probability row
/// on the stack at this width rather than allocating inside an `_into` method.
const MAX_CLASSES: usize = 256;

const N_JOBS_SERIAL: u32 = 1;
const N_JOBS_ALL: u32 = 2;
const N_JOBS_COUNT: u32 = 3;

/// The fitted trees behind a [`RandomForestClassifier`].
///
/// The two flavours are kept apart rather than unified because their leaf
/// arithmetic is genuinely different: a binary leaf stores one probability and
/// the ensemble averages that scalar, while a multiclass leaf stores one
/// probability per class and the ensemble averages the vector. Forcing the
/// binary fit through the vector path would change values it has already
/// frozen, for no gain.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Forest {
    /// Trees whose leaf is the probability of class `1`.
    Binary(Vec<PackedTree>),
    /// Trees whose leaf is one probability per observed class.
    Multiclass(Vec<ClassTree>),
}

/// A random-forest classifier.
///
/// Class labels are sorted, and probability columns follow that order. Models
/// fitted on a single class expose one probability column containing `1.0`.
///
/// [`fit`](Self::fit) takes binary targets and keeps the asymmetric
/// scalar-leaf representation FerricML froze first.
/// [`fit_multiclass`](Self::fit_multiclass) takes any observed class set and
/// fits natively multiclass trees whose ensemble probability is the **mean of
/// the per-tree probability vectors** — soft averaging, not a majority vote of
/// per-tree labels. The two are different models even on the same two-class
/// data. Both persist, under one artifact kind that records which leaf
/// arithmetic it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestClassifier {
    pub(super) n_features_in: usize,
    pub(super) params: RandomForestClassifierParams,
    pub(super) classes: Vec<u8>,
    pub(super) forest: Forest,
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

    /// Fits a binary classifier over `0`/`1` targets.
    ///
    /// This is the asymmetric scalar-leaf fit: each tree stores the probability
    /// of class `1` and the ensemble averages that scalar.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_binary(data, targets, None, params)
    }

    /// Fits a binary classifier with per-row sample weights.
    ///
    /// A weight scales the row's contribution to every impurity and leaf
    /// statistic, and composes with the bootstrap replication count. Weights of
    /// exactly one reproduce [`Self::fit`] bit for bit, and an integer weight is
    /// the same fit as repeating that row that many times.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: &SampleWeights,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_binary(data, targets, Some(sample_weights), params)
    }

    fn fit_binary(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: Option<&SampleWeights>,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.as_slice().len(), sample_weights, &config)?;
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
        let trees = train_forest(
            data,
            targets.as_slice(),
            sample_weights.map(SampleWeights::as_slice),
            &config,
            Classification,
        )?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            forest: Forest::Binary(trees),
        })
    }

    /// Fits a natively multiclass classifier over any observed class set.
    ///
    /// Each tree splits on multiclass Gini impurity and stores one probability
    /// per class at every leaf. The ensemble probability is the **mean of the
    /// per-tree probability vectors**, which is a strictly different rule from
    /// a majority vote over per-tree labels — soft averaging produces values a
    /// vote cannot, and the two disagree on real data.
    ///
    /// A single observed class is accepted: the fit succeeds with one
    /// probability column containing `1.0`, matching the single-class contract
    /// the binary entry point already has. Two observed classes are also
    /// accepted and produce a vector-leaf model, which is a different — and
    /// deliberately not interchangeable — model from [`Self::fit`] on the same
    /// data.
    pub fn fit_multiclass(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_multiclass_internal(data, targets, None, params)
    }

    /// Fits a natively multiclass classifier with per-row sample weights.
    ///
    /// The weight scales the row's contribution to the multiclass Gini
    /// statistics and to every leaf distribution, exactly as it does for the
    /// binary fit.
    pub fn fit_multiclass_weighted(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        sample_weights: &SampleWeights,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_multiclass_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_multiclass_internal(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        sample_weights: Option<&SampleWeights>,
        params: RandomForestClassifierParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.len(), sample_weights, &config)?;
        let classes = targets.classes().to_vec();
        let class_of_row = targets
            .as_slice()
            .iter()
            .map(|&label| {
                targets
                    .class_index(label)
                    .expect("every target label is an observed class")
            })
            .collect::<Vec<_>>();
        let trees = train_class_forest(
            data,
            &class_of_row,
            classes.len(),
            sample_weights.map(SampleWeights::as_slice),
            &config,
        )?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            forest: Forest::Multiclass(trees),
        })
    }

    /// Predicts the class label for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        check_row(row, self.n_features_in)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    return Ok(self.classes[0]);
                }
                let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                // `classes` is sorted as [0, 1]. An exact tie selects its first class.
                Ok(u8::from(positive > 0.5))
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                average_class_probabilities(trees, row, probabilities);
                Ok(self.classes[argmax(probabilities)])
            }
        }
    }

    /// Predicts one label per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        match &self.forest {
            Forest::Binary(_) => {
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
            Forest::Multiclass(_) => <Self as Classifier>::predict(self, data),
        }
    }

    /// Predicts one label per row without allocating.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    output.fill(self.classes[0]);
                    return Ok(());
                }
                for (row, slot) in data.iter_rows().zip(output) {
                    let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                    *slot = u8::from(positive > 0.5);
                }
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                for (row, slot) in data.iter_rows().zip(output) {
                    // Argmax of the averaged probabilities, so a label can never
                    // disagree with the probability row a caller can read.
                    average_class_probabilities(trees, row, probabilities);
                    *slot = self.classes[argmax(probabilities)];
                }
            }
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
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    output[0] = 1.0;
                } else {
                    let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                    output[0] = 1.0 - positive;
                    output[1] = positive;
                }
            }
            Forest::Multiclass(trees) => average_class_probabilities(trees, row, output),
        }
        Ok(())
    }

    /// Predicts row-major probabilities, allocating `rows * classes().len()`
    /// values.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities without allocating.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let columns = self.classes.len();
        let expected = probability_output_len(data.rows(), columns)?;
        check_prediction_data(data, output.len(), expected, self.n_features_in)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if columns == 1 {
                    output.fill(1.0);
                    return Ok(());
                }
                output.fill(0.0);
                for tree in trees {
                    for (row, probabilities) in data.iter_rows().zip(output.chunks_exact_mut(2)) {
                        probabilities[1] += tree.predict(row);
                    }
                }
                for probabilities in output.chunks_exact_mut(2) {
                    let positive = (probabilities[1] / trees.len() as f32).clamp(0.0, 1.0);
                    probabilities[0] = 1.0 - positive;
                    probabilities[1] = positive;
                }
            }
            Forest::Multiclass(trees) => {
                // One tree across the whole batch, then the next, so each
                // tree's nodes and leaf block stay hot.
                output.fill(0.0);
                for tree in trees {
                    for (row, probabilities) in
                        data.iter_rows().zip(output.chunks_exact_mut(columns))
                    {
                        for (slot, &value) in probabilities.iter_mut().zip(tree.probabilities(row))
                        {
                            *slot += value;
                        }
                    }
                }
                let divisor = trees.len() as f32;
                for slot in output {
                    *slot = (*slot / divisor).clamp(0.0, 1.0);
                }
            }
        }
        Ok(())
    }

    /// Returns the requested fitted-class probability for one sample.
    pub fn predict_class_proba_one(&self, row: &[f32], class: u8) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        let class_index = self.class_index(class)?;
        match &self.forest {
            Forest::Binary(trees) => {
                if self.classes.len() == 1 {
                    return Ok(1.0);
                }
                let positive = mean_tree_prediction(trees, row).clamp(0.0, 1.0);
                Ok(if class_index == 0 {
                    1.0 - positive
                } else {
                    positive
                })
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                average_class_probabilities(trees, row, probabilities);
                Ok(probabilities[class_index])
            }
        }
    }

    /// Predicts one fitted-class probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_class_proba(self, data, class)
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
        match &self.forest {
            Forest::Binary(_) => {
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
            }
            Forest::Multiclass(trees) => {
                let mut storage = [0.0_f32; MAX_CLASSES];
                let probabilities = &mut storage[..self.classes.len()];
                for (row, slot) in data.iter_rows().zip(output) {
                    average_class_probabilities(trees, row, probabilities);
                    *slot = probabilities[class_index];
                }
            }
        }
        Ok(())
    }

    fn accumulate_positive_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let Forest::Binary(trees) = &self.forest else {
            unreachable!("the scalar accumulation serves the binary fit only");
        };
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        output.fill(0.0);
        for tree in trees {
            for (row, slot) in data.iter_rows().zip(output.iter_mut()) {
                *slot += tree.predict(row);
            }
        }
        for slot in output {
            *slot = (*slot / trees.len() as f32).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Encodes the fitted parameters, class list, and canonical logical trees.
    ///
    /// The two fits are different models with different leaf arithmetic, so the
    /// payload records which one it holds and the reader refuses to build the
    /// other. A binary fit reuses the scalar logical-tree records unchanged —
    /// the same codec the regressor and the boosted trees use. A multiclass fit
    /// writes the same topology records with a reserved zero where a scalar leaf
    /// carries its value, followed by that tree's leaf distributions in
    /// pre-order leaf rank. Storing rank rather than the runtime leaf ordinal is
    /// what keeps the encoding unique: the ordinals could be permuted together
    /// with the block to name one model twice.
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let (flavour, tree_count, total_nodes) = match &self.forest {
            Forest::Binary(trees) => (
                FOREST_BINARY,
                trees.len(),
                trees.iter().try_fold(0_usize, |total, tree| {
                    total
                        .checked_add(tree.logical_node_count())
                        .ok_or(ArtifactError::InvalidPayload)
                })?,
            ),
            Forest::Multiclass(trees) => (
                FOREST_MULTICLASS,
                trees.len(),
                trees.iter().try_fold(0_usize, |total, tree| {
                    total
                        .checked_add(tree.logical_node_count())
                        .ok_or(ArtifactError::InvalidPayload)
                })?,
            ),
        };
        if tree_count > MAX_ARTIFACT_TREES
            || total_nodes > MAX_ARTIFACT_TOTAL_NODES
            || self.n_features_in > MAX_ARTIFACT_FEATURES
            || self.classes.is_empty()
            || self.classes.len() > MAX_CLASSES
        {
            return Err(ArtifactError::InvalidPayload);
        }

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
        let tree_count = u32::try_from(tree_count).map_err(|_| ArtifactError::InvalidPayload)?;
        let total_nodes = u32::try_from(total_nodes).map_err(|_| ArtifactError::InvalidPayload)?;
        let class_count =
            u32::try_from(self.classes.len()).map_err(|_| ArtifactError::InvalidPayload)?;

        let mut metadata = ArtifactPayloadWriter::with_capacity(
            CLASSIFIER_METADATA_BYTES + self.classes.len() * 4,
        );
        metadata.u32(CLASSIFIER_OBJECTIVE_VERSION);
        metadata.u32(flavour);
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
        metadata.u32(class_count);
        for &class in &self.classes {
            metadata.u32(u32::from(class));
        }
        let mut payload = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        match &self.forest {
            Forest::Binary(trees) => {
                for tree in trees {
                    payload.extend_from_slice(&encode_component(
                        TREE_COMPONENT_KIND,
                        COMPONENT_VERSION,
                        &encode_logical_tree(&tree.to_logical_nodes())?,
                    )?);
                }
            }
            Forest::Multiclass(trees) => {
                for tree in trees {
                    let (nodes, probabilities) = tree.to_logical_nodes();
                    payload.extend_from_slice(&encode_component(
                        TREE_COMPONENT_KIND,
                        COMPONENT_VERSION,
                        &encode_logical_tree(&nodes)?,
                    )?);
                    let mut block =
                        ArtifactPayloadWriter::with_capacity(8 + probabilities.len() * 4);
                    block.u32(
                        u32::try_from(probabilities.len() / self.classes.len())
                            .map_err(|_| ArtifactError::InvalidPayload)?,
                    );
                    block.u32(class_count);
                    for &value in &probabilities {
                        block.f32(value);
                    }
                    payload.extend_from_slice(&encode_component(
                        LEAF_PROBABILITY_COMPONENT_KIND,
                        COMPONENT_VERSION,
                        &block.finish(),
                    )?);
                }
            }
        }
        encode_v2_envelope(
            RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND,
            CLASSIFIER_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes and revalidates a classifier before building runtime state.
    ///
    /// Counts, parameters, and the class list are checked before any tree is
    /// read, every decoded tree re-enters the same topology validator fitting
    /// uses, and every decoded probability re-enters the same class-topology
    /// invariant a fitted tree satisfies.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND,
            CLASSIFIER_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let objective_version = metadata.u32()?;
        let flavour = metadata.u32()?;
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
        let class_count = metadata.u32()? as usize;
        if objective_version != CLASSIFIER_OBJECTIVE_VERSION
            || (flavour != FOREST_BINARY && flavour != FOREST_MULTICLASS)
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
            || class_count == 0
            || class_count > MAX_CLASSES
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut classes: Vec<u8> = Vec::with_capacity(metadata.bounded_capacity(class_count, 4));
        for _ in 0..class_count {
            let label = u8::try_from(metadata.u32()?).map_err(|_| ArtifactError::InvalidPayload)?;
            if classes.last().is_some_and(|&previous| previous >= label) {
                return Err(ArtifactError::InvalidPayload);
            }
            classes.push(label);
        }
        // A binary fit is asymmetric: its scalar leaf is the probability of
        // class `1`, and prediction reads the label straight out of that
        // comparison. Only `[0]`, `[1]`, and `[0, 1]` mean anything there.
        if flavour == FOREST_BINARY && classes.iter().any(|&label| label > 1) {
            return Err(ArtifactError::InvalidPayload);
        }
        if !metadata.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        let (Some(max_features), Some(n_jobs)) = (
            decode_max_features(max_features_tag, max_features_count, n_features_in),
            decode_n_jobs(n_jobs_tag, n_jobs_count),
        ) else {
            return Err(ArtifactError::InvalidPayload);
        };
        let params = RandomForestClassifierParams::default()
            .with_n_estimators(n_estimators)
            .with_max_depth((encoded_depth != 0).then_some(encoded_depth))
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_max_features(max_features)
            .with_bootstrap(bootstrap == 1)
            .with_random_state(random_state)
            .with_n_jobs(n_jobs);

        let mut actual_total_nodes = 0_usize;
        let forest = if flavour == FOREST_BINARY {
            let mut trees =
                Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
            for _ in 0..tree_count {
                let logical = decode_logical_tree(decode_component(
                    &mut envelope,
                    TREE_COMPONENT_KIND,
                    COMPONENT_VERSION,
                )?)?;
                actual_total_nodes =
                    accumulate_nodes(actual_total_nodes, logical.len(), declared_total_nodes)?;
                // A fitted binary leaf is a probability, so nothing else is a
                // model this crate could have produced.
                if logical.iter().any(|node| {
                    matches!(node, LogicalTreeNode::Leaf { value } if !(0.0..=1.0).contains(value))
                }) {
                    return Err(ArtifactError::InvalidPayload);
                }
                trees.push(PackedTree::from_logical_nodes(&logical, n_features_in)?);
            }
            Forest::Binary(trees)
        } else {
            let mut trees =
                Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
            for _ in 0..tree_count {
                let logical = decode_logical_tree(decode_component(
                    &mut envelope,
                    TREE_COMPONENT_KIND,
                    COMPONENT_VERSION,
                )?)?;
                actual_total_nodes =
                    accumulate_nodes(actual_total_nodes, logical.len(), declared_total_nodes)?;
                let probabilities = decode_leaf_probabilities(
                    decode_component(
                        &mut envelope,
                        LEAF_PROBABILITY_COMPONENT_KIND,
                        COMPONENT_VERSION,
                    )?,
                    class_count,
                )?;
                trees.push(ClassTree::from_logical_nodes(
                    &logical,
                    &probabilities,
                    class_count,
                    n_features_in,
                )?);
            }
            Forest::Multiclass(trees)
        };
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        if actual_total_nodes != declared_total_nodes {
            return Err(ArtifactError::InvalidPayload);
        }
        Ok(Self {
            n_features_in,
            params,
            classes,
            forest,
        })
    }

    /// The scalar trees of a binary fit, for in-crate structural tests.
    #[cfg(test)]
    pub(super) fn binary_trees(&self) -> &[PackedTree] {
        match &self.forest {
            Forest::Binary(trees) => trees,
            Forest::Multiclass(_) => unreachable!("binary fixture"),
        }
    }

    #[inline]
    fn class_index(&self, class: u8) -> Result<usize, ModelError> {
        self.classes
            .binary_search(&class)
            .map_err(|_| ModelError::UnknownClass { class })
    }

    /// Returns the positive-class probability for one sample.
    ///
    /// Defined only for a binary fit. A multiclass fit has no positive class
    /// and reports [`ModelError::MulticlassOutput`] instead of returning one
    /// column of a vector that has no distinguished member.
    pub fn predict_positive_proba(&self, row: &[f32]) -> Result<f32, ModelError> {
        let trees = self.require_binary_forest()?;
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(f32::from(self.classes[0]));
        }
        Ok(mean_tree_prediction(trees, row).clamp(0.0, 1.0))
    }

    /// Predict every row without allocating.  `output.len()` must equal the
    /// number of input rows.
    pub fn predict_positive_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let trees = self.require_binary_forest()?;
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(f32::from(self.classes[0]));
            return Ok(());
        }
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = mean_tree_prediction(trees, data.row(index).expect("validated row index"))
                .clamp(0.0, 1.0);
        }
        Ok(())
    }

    fn require_binary_forest(&self) -> Result<&[PackedTree], ModelError> {
        match &self.forest {
            Forest::Binary(trees) => Ok(trees),
            Forest::Multiclass(_) => Err(ModelError::MulticlassOutput {
                columns: self.classes.len(),
            }),
        }
    }

    /// Internal bytes used only for deterministic implementation tests.
    #[cfg(test)]
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        match &self.forest {
            Forest::Binary(trees) => packed_model_bytes(self.n_features_in, trees, b"FRFC"),
            Forest::Multiclass(_) => unreachable!("scalar packing is a binary-fit fixture"),
        }
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
}

impl ProbabilisticClassifier for RandomForestClassifier {
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

/// Declares weighted fitting, multiclass fitting, and persistence. The artifact
/// covers *both* leaf representations, so the declaration holds for every fit
/// this type offers rather than for one of its two entry points.
impl HasCapabilities for RandomForestClassifier {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true)
        .with_multiclass(true)
        .with_probability(true);
}

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
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits with per-row sample weights.
    ///
    /// A weight scales the row's contribution to the variance and leaf mean of
    /// every node it reaches, and composes with the bootstrap replication
    /// count. Weights of exactly one reproduce [`Self::fit`] bit for bit, and an
    /// integer weight is the same fit as repeating that row that many times.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: RandomForestRegressorParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: RandomForestRegressorParams,
    ) -> Result<Self, ModelError> {
        let config = ForestConfig::from(&params);
        validate_common(data, targets.as_slice().len(), sample_weights, &config)?;
        for (index, value) in targets.as_slice().iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::NonFiniteTarget { index });
            }
        }
        let trees = train_forest(
            data,
            targets.as_slice(),
            sample_weights.map(SampleWeights::as_slice),
            &config,
            Regression,
        )?;
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

        let mut trees =
            Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
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

/// Adds one decoded tree's records to the running total, refusing to pass the
/// count the metadata declared before the next tree is even read.
fn accumulate_nodes(total: usize, added: usize, declared: usize) -> Result<usize, ArtifactError> {
    let total = total
        .checked_add(added)
        .ok_or(ArtifactError::InvalidPayload)?;
    if total > declared {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok(total)
}

/// Reads one tree's leaf distributions, in pre-order leaf rank.
///
/// The declared leaf count is checked against the class count and against the
/// bytes actually present before anything is reserved, and every value must be
/// the finite `0..=1` a fitted leaf distribution holds.
fn decode_leaf_probabilities(
    mut cursor: ArtifactCursor<'_>,
    class_count: usize,
) -> Result<Vec<f32>, ArtifactError> {
    let leaves = cursor.u32()? as usize;
    let declared_classes = cursor.u32()? as usize;
    let expected = leaves.checked_mul(class_count);
    if leaves == 0 || declared_classes != class_count || expected.is_none() {
        return Err(ArtifactError::InvalidPayload);
    }
    let expected = expected.expect("checked above");
    let mut probabilities = Vec::with_capacity(cursor.bounded_capacity(expected, 4));
    for _ in 0..expected {
        let value = cursor.f32()?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ArtifactError::InvalidPayload);
        }
        probabilities.push(value);
    }
    if !cursor.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(probabilities)
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
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
}

fn mean_tree_prediction(trees: &[PackedTree], row: &[f32]) -> f32 {
    let sum: f32 = trees.iter().map(|tree| tree.predict(row)).sum();
    sum / trees.len() as f32
}

/// Soft averaging: the mean of the per-tree probability vectors.
///
/// This is not a vote. Every tree contributes its whole leaf distribution, so
/// the ensemble can produce values no count of `trees.len()` labels could —
/// which is exactly what distinguishes the two rules observably.
///
/// The accumulation is in the storage width under rule 3 of the accumulation
/// policy: the term count is the fitted tree count and every value is bounded
/// by one.
fn average_class_probabilities(trees: &[ClassTree], row: &[f32], output: &mut [f32]) {
    debug_assert!(
        trees
            .first()
            .is_none_or(|tree| tree.classes() == output.len())
    );
    output.fill(0.0);
    for tree in trees {
        for (slot, &value) in output.iter_mut().zip(tree.probabilities(row)) {
            *slot += value;
        }
    }
    let divisor = trees.len() as f32;
    for slot in output {
        *slot = (*slot / divisor).clamp(0.0, 1.0);
    }
}

/// Index of the largest value, with an exact tie going to the lowest index.
///
/// Class labels are sorted, so this is the smallest tied *label* — which is not
/// the same as the first class: with classes `[5, 9, 20]`, a tie between the
/// last two selects `9`.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
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
    sample_weights: Option<&SampleWeights>,
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
    if let Some(sample_weights) = sample_weights
        && data.rows() != sample_weights.len()
    {
        return Err(ModelError::SampleWeightLength {
            rows: data.rows(),
            weights: sample_weights.len(),
        });
    }
    if config.n_estimators == 0 {
        return Err(ModelError::InvalidEstimatorCount);
    }
    if config.grower.max_depth == Some(0) {
        return Err(ModelError::InvalidMaxDepth);
    }
    if config.grower.min_samples_split < 2 {
        return Err(ModelError::InvalidMinSamplesSplit);
    }
    if config.grower.min_samples_leaf == 0 {
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
    if let MaxFeatures::Count(requested) = config.grower.max_features
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
