use super::grower::{
    Classification, GrowerConfig, grow_class_tree, grow_tree, unbootstrapped_sample,
};
use super::packed::{ClassTree, PackedTree};
use super::parameters::{DecisionTreeClassifierParams, encode_max_features, encode_splitter};
use super::validation::{
    MAX_ARTIFACT_FEATURES, MAX_ARTIFACT_TOTAL_NODES, check_output_len, check_prediction_data,
    check_row, read_common_metadata, tree_seed, validate_fit, write_common_metadata,
};
use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, HasParams, ModelError,
    ProbabilisticClassifier,
};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, DECISION_TREE_CLASSIFIER_ARTIFACT_KIND,
    LogicalTreeNode, ModelArtifact, SchemaRole, decode_component, decode_logical_tree,
    decode_v2_envelope, encode_component, encode_logical_tree, encode_v2_envelope,
};
use crate::data::{BinaryTargets, ClassTargets, MatrixView, SampleWeights};
use crate::numeric::OwnedRng;

const PAYLOAD_VERSION: u16 = 1;
const OBJECTIVE_VERSION: u32 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
/// The leaf distributions, in pre-order leaf rank. Written only by the
/// multiclass flavour, immediately after its topology component.
const LEAF_PROBABILITY_COMPONENT_KIND: u16 = 3;
const COMPONENT_VERSION: u16 = 1;
const METADATA_BYTES: usize = 11 * 4 + 8;

/// Which leaf arithmetic the encoded tree uses. The two are different models,
/// so the tag is read before the tree is, and neither flavour's records are
/// ever handed to the other's builder.
const TREE_BINARY: u32 = 1;
const TREE_MULTICLASS: u32 = 2;

/// Every class label is a `u8`, so no fit can observe more classes than this.
const MAX_CLASSES: usize = 256;

/// The fitted tree behind a [`DecisionTreeClassifier`].
///
/// The two flavours are kept apart rather than unified for the same reason the
/// forest keeps them apart: their leaf arithmetic genuinely differs. A binary
/// leaf stores the probability of class `1` as a scalar; a multiclass leaf
/// stores one probability per class. Mirroring the forest here is not
/// conservatism — it is what makes a standalone tree bit-identical to a
/// one-tree forest through *both* of the forest's fitting entry points rather
/// than only one, since the two entry points run different builders.
#[derive(Clone, Debug, PartialEq)]
enum Tree {
    Binary(PackedTree),
    Multiclass(ClassTree),
}

/// A single classification tree.
///
/// Class labels are sorted and probability columns follow that order. A model
/// fitted on a single class exposes one probability column containing `1.0`.
///
/// [`fit`](Self::fit) takes binary targets and stores one probability per leaf;
/// [`fit_multiclass`](Self::fit_multiclass) takes any observed class set and
/// stores a full distribution per leaf. As with the forest, the two are
/// different models even on the same two-class data, and both persist under one
/// artifact kind that records which leaf arithmetic it holds.
///
/// ```
/// use ferricml::api::Classifier;
/// use ferricml::data::{BinaryTargets, DenseMatrix};
/// use ferricml::tree::{DecisionTreeClassifier, DecisionTreeClassifierParams};
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
/// let labels = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;
///
/// let model = DecisionTreeClassifier::fit(
///     &data.as_view(),
///     &labels,
///     DecisionTreeClassifierParams::default(),
/// )?;
///
/// assert_eq!(model.classes(), &[0, 1]);
/// assert_eq!(model.predict(&data.as_view())?, vec![0, 0, 0, 1, 1, 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A multiclass fit stores a full distribution per leaf and takes any observed
/// class set:
///
/// ```
/// use ferricml::api::Classifier;
/// use ferricml::data::{ClassTargets, DenseMatrix};
/// use ferricml::tree::{DecisionTreeClassifier, DecisionTreeClassifierParams};
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
/// let labels = ClassTargets::new(vec![3, 3, 7, 7, 10, 10])?;
///
/// let model = DecisionTreeClassifier::fit_multiclass(
///     &data.as_view(),
///     &labels,
///     DecisionTreeClassifierParams::default(),
/// )?;
///
/// assert_eq!(model.classes(), &[3, 7, 10]);
/// assert_eq!(model.predict(&data.as_view())?, vec![3, 3, 7, 7, 10, 10]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionTreeClassifier {
    n_features_in: usize,
    params: DecisionTreeClassifierParams,
    classes: Vec<u8>,
    tree: Tree,
}

impl DecisionTreeClassifier {
    /// Returns the feature width required by this model.
    #[inline]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact parameters used to fit this model.
    #[inline]
    pub fn get_params(&self) -> &DecisionTreeClassifierParams {
        &self.params
    }

    /// Returns sorted class labels observed during fitting.
    #[inline]
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    /// Fits a binary classifier over `0`/`1` targets.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: DecisionTreeClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_binary(data, targets, None, params)
    }

    /// Fits a binary classifier with per-row sample weights.
    ///
    /// A weight scales the row's contribution to every impurity and leaf
    /// statistic. Weights of exactly one reproduce [`Self::fit`] bit for bit,
    /// and an integer weight is the same fit as repeating that row that many
    /// times.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: &SampleWeights,
        params: DecisionTreeClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_binary(data, targets, Some(sample_weights), params)
    }

    fn fit_binary(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: Option<&SampleWeights>,
        params: DecisionTreeClassifierParams,
    ) -> Result<Self, ModelError> {
        let config = grower_config(&params);
        validate_fit(
            data,
            targets.as_slice().len(),
            sample_weights.map(SampleWeights::len),
            &config,
        )?;
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
        let (weights, rows) =
            unbootstrapped_sample(data.rows(), sample_weights.map(SampleWeights::as_slice));
        let mut rng = OwnedRng::new(tree_seed(params.random_state()));
        let tree = grow_tree(
            data,
            targets.as_slice(),
            &weights,
            rows,
            &config,
            Classification,
            &mut rng,
        )?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            tree: Tree::Binary(tree),
        })
    }

    /// Fits a natively multiclass classifier over any observed class set.
    ///
    /// Each node splits on multiclass Gini impurity and every leaf stores one
    /// probability per class. A single observed class is accepted and yields
    /// one probability column containing `1.0`.
    pub fn fit_multiclass(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        params: DecisionTreeClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_multiclass_internal(data, targets, None, params)
    }

    /// Fits a natively multiclass classifier with per-row sample weights.
    pub fn fit_multiclass_weighted(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        sample_weights: &SampleWeights,
        params: DecisionTreeClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_multiclass_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_multiclass_internal(
        data: &MatrixView<'_>,
        targets: &ClassTargets,
        sample_weights: Option<&SampleWeights>,
        params: DecisionTreeClassifierParams,
    ) -> Result<Self, ModelError> {
        let config = grower_config(&params);
        validate_fit(
            data,
            targets.len(),
            sample_weights.map(SampleWeights::len),
            &config,
        )?;
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
        let (weights, rows) =
            unbootstrapped_sample(data.rows(), sample_weights.map(SampleWeights::as_slice));
        let mut rng = OwnedRng::new(tree_seed(params.random_state()));
        let tree = grow_class_tree(
            data,
            &class_of_row,
            classes.len(),
            &weights,
            rows,
            &config,
            &mut rng,
        )?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            tree: Tree::Multiclass(tree),
        })
    }

    /// The packed binary tree, for the in-crate grower-equivalence proof.
    #[cfg(test)]
    pub(crate) fn packed_binary(&self) -> &PackedTree {
        match &self.tree {
            Tree::Binary(tree) => tree,
            Tree::Multiclass(_) => unreachable!("binary fixture"),
        }
    }

    /// The packed multiclass tree, for the same proof through the other
    /// fitting entry point.
    #[cfg(test)]
    pub(crate) fn packed_multiclass(&self) -> &ClassTree {
        match &self.tree {
            Tree::Multiclass(tree) => tree,
            Tree::Binary(_) => unreachable!("multiclass fixture"),
        }
    }

    /// Predicts the class label for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        check_row(row, self.n_features_in)?;
        Ok(match &self.tree {
            Tree::Binary(tree) => {
                if self.classes.len() == 1 {
                    self.classes[0]
                } else {
                    // `classes` is sorted as [0, 1]. An exact tie selects its
                    // first class, matching the forest.
                    u8::from(positive_probability(tree, row) > 0.5)
                }
            }
            Tree::Multiclass(tree) => self.classes[argmax(tree.probabilities(row))],
        })
    }

    /// Predicts one label per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Predicts one label per row without allocating.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        match &self.tree {
            Tree::Binary(tree) => {
                if self.classes.len() == 1 {
                    output.fill(self.classes[0]);
                    return Ok(());
                }
                for (row, slot) in data.iter_rows().zip(output) {
                    *slot = u8::from(positive_probability(tree, row) > 0.5);
                }
            }
            Tree::Multiclass(tree) => {
                for (row, slot) in data.iter_rows().zip(output) {
                    // Argmax of the same probabilities a caller can read, so a
                    // label can never disagree with its probability row.
                    *slot = self.classes[argmax(tree.probabilities(row))];
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
        self.write_probabilities(row, output);
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
        let expected = data
            .rows()
            .checked_mul(columns)
            .ok_or(ModelError::OutputShapeOverflow {
                rows: data.rows(),
                columns,
            })?;
        check_prediction_data(data, output.len(), expected, self.n_features_in)?;
        for (row, slot) in data.iter_rows().zip(output.chunks_exact_mut(columns)) {
            self.write_probabilities(row, slot);
        }
        Ok(())
    }

    /// Returns the requested fitted-class probability for one sample.
    pub fn predict_class_proba_one(&self, row: &[f32], class: u8) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        let class_index = self.class_index(class)?;
        let mut storage = [0.0_f32; MAX_CLASSES];
        let probabilities = &mut storage[..self.classes.len()];
        self.write_probabilities(row, probabilities);
        Ok(probabilities[class_index])
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
        let mut storage = [0.0_f32; MAX_CLASSES];
        let probabilities = &mut storage[..self.classes.len()];
        for (row, slot) in data.iter_rows().zip(output) {
            self.write_probabilities(row, probabilities);
            *slot = probabilities[class_index];
        }
        Ok(())
    }

    /// Returns the positive-class probability for one sample.
    ///
    /// Defined only for a binary fit. A multiclass fit has no positive class
    /// and reports [`ModelError::MulticlassOutput`] rather than returning one
    /// column of a vector with no distinguished member.
    pub fn predict_positive_proba_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        let tree = self.require_binary_tree()?;
        check_row(row, self.n_features_in)?;
        if self.classes.len() == 1 {
            return Ok(f32::from(self.classes[0]));
        }
        Ok(positive_probability(tree, row))
    }

    /// Predicts the positive-class probability for every row, allocating the
    /// output.
    pub fn predict_positive_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        // Before the buffer, not inside `_into` after it, so an unusable
        // request costs no allocation.
        self.require_binary_tree()?;
        check_prediction_data(data, data.rows(), data.rows(), self.n_features_in)?;
        let mut output = vec![0.0; data.rows()];
        self.predict_positive_proba_into(data, &mut output)?;
        Ok(output)
    }

    /// Predicts the positive-class probability for every row without
    /// allocating. `output.len()` must equal the number of input rows.
    pub fn predict_positive_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let tree = self.require_binary_tree()?;
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        if self.classes.len() == 1 {
            output.fill(f32::from(self.classes[0]));
            return Ok(());
        }
        for (row, slot) in data.iter_rows().zip(output) {
            *slot = positive_probability(tree, row);
        }
        Ok(())
    }

    /// Writes one row's probabilities in `classes` order.
    ///
    /// The single place the two leaf representations are turned into the one
    /// shape every probability entry point returns, so a scalar leaf and a
    /// distribution leaf cannot disagree about what a caller sees.
    fn write_probabilities(&self, row: &[f32], output: &mut [f32]) {
        match &self.tree {
            Tree::Binary(tree) => {
                if self.classes.len() == 1 {
                    output[0] = 1.0;
                } else {
                    let positive = positive_probability(tree, row);
                    output[0] = 1.0 - positive;
                    output[1] = positive;
                }
            }
            Tree::Multiclass(tree) => output.copy_from_slice(tree.probabilities(row)),
        }
    }

    #[inline]
    fn class_index(&self, class: u8) -> Result<usize, ModelError> {
        self.classes
            .binary_search(&class)
            .map_err(|_| ModelError::UnknownClass { class })
    }

    fn require_binary_tree(&self) -> Result<&PackedTree, ModelError> {
        match &self.tree {
            Tree::Binary(tree) => Ok(tree),
            Tree::Multiclass(_) => Err(ModelError::MulticlassOutput {
                columns: self.classes.len(),
            }),
        }
    }
}

impl ModelArtifact for DecisionTreeClassifier {
    const ARTIFACT_KIND: u16 = DECISION_TREE_CLASSIFIER_ARTIFACT_KIND;

    /// Encodes the fitted parameters, class list, and canonical logical tree.
    ///
    /// A binary fit reuses the scalar logical-tree records unchanged — the same
    /// codec the forest and the boosted trees use. A multiclass fit writes the
    /// same topology records with a reserved zero where a scalar leaf carries
    /// its value, followed by the leaf distributions in pre-order leaf rank.
    /// Storing rank rather than the runtime leaf ordinal is what keeps the
    /// encoding unique: the ordinals could be permuted together with the block
    /// to name one model twice.
    fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let (flavour, node_count) = match &self.tree {
            Tree::Binary(tree) => (TREE_BINARY, tree.logical_node_count()),
            Tree::Multiclass(tree) => (TREE_MULTICLASS, tree.logical_node_count()),
        };
        if self.n_features_in > MAX_ARTIFACT_FEATURES
            || node_count > MAX_ARTIFACT_TOTAL_NODES
            || self.classes.is_empty()
            || self.classes.len() > MAX_CLASSES
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let class_count =
            u32::try_from(self.classes.len()).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut metadata =
            ArtifactPayloadWriter::with_capacity(METADATA_BYTES + self.classes.len() * 4);
        write_common_metadata(
            &mut metadata,
            OBJECTIVE_VERSION,
            self.n_features_in,
            self.params.max_depth(),
            self.params.min_samples_split(),
            self.params.min_samples_leaf(),
            encode_max_features(self.params.max_features())?,
            encode_splitter(self.params.splitter()),
            self.params.random_state(),
            node_count,
        )?;
        metadata.u32(flavour);
        metadata.u32(class_count);
        for &class in &self.classes {
            metadata.u32(u32::from(class));
        }
        let mut payload = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        match &self.tree {
            Tree::Binary(tree) => {
                payload.extend_from_slice(&encode_component(
                    TREE_COMPONENT_KIND,
                    COMPONENT_VERSION,
                    &encode_logical_tree(&tree.to_logical_nodes())?,
                )?);
            }
            Tree::Multiclass(tree) => {
                let (nodes, probabilities) = tree.to_logical_nodes();
                payload.extend_from_slice(&encode_component(
                    TREE_COMPONENT_KIND,
                    COMPONENT_VERSION,
                    &encode_logical_tree(&nodes)?,
                )?);
                let mut block = ArtifactPayloadWriter::with_capacity(8 + probabilities.len() * 4);
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
        encode_v2_envelope(
            Self::ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes and revalidates a classifier before building runtime state.
    ///
    /// Parameters and the class list are checked before the tree is read, the
    /// decoded records re-enter the same topology validator fitting uses, and
    /// decoded probabilities re-enter the same class-topology invariant a
    /// fitted tree satisfies.
    fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            Self::ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let common = read_common_metadata(&mut metadata, OBJECTIVE_VERSION)?;
        let flavour = metadata.u32()?;
        let class_count = metadata.u32()? as usize;
        if (flavour != TREE_BINARY && flavour != TREE_MULTICLASS)
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
        if flavour == TREE_BINARY && classes.iter().any(|&label| label > 1) {
            return Err(ArtifactError::InvalidPayload);
        }
        if !metadata.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }

        let logical = decode_logical_tree(decode_component(
            &mut envelope,
            TREE_COMPONENT_KIND,
            COMPONENT_VERSION,
        )?)?;
        if logical.len() != common.node_count {
            return Err(ArtifactError::InvalidPayload);
        }
        let tree = if flavour == TREE_BINARY {
            // A fitted binary leaf is a probability, so nothing else is a model
            // this crate could have produced.
            if logical.iter().any(|node| {
                matches!(node, LogicalTreeNode::Leaf { value } if !(0.0..=1.0).contains(value))
            }) {
                return Err(ArtifactError::InvalidPayload);
            }
            Tree::Binary(PackedTree::from_logical_nodes(
                &logical,
                common.n_features_in,
            )?)
        } else {
            let probabilities = decode_leaf_probabilities(
                decode_component(
                    &mut envelope,
                    LEAF_PROBABILITY_COMPONENT_KIND,
                    COMPONENT_VERSION,
                )?,
                class_count,
            )?;
            Tree::Multiclass(ClassTree::from_logical_nodes(
                &logical,
                &probabilities,
                class_count,
                common.n_features_in,
            )?)
        };
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(Self {
            n_features_in: common.n_features_in,
            params: DecisionTreeClassifierParams::default()
                .with_max_depth(common.max_depth)
                .with_min_samples_split(common.min_samples_split)
                .with_min_samples_leaf(common.min_samples_leaf)
                .with_max_features(common.max_features)
                .with_splitter(common.splitter)
                .with_random_state(common.random_state),
            classes,
            tree,
        })
    }
}

/// Reads the leaf distributions, in pre-order leaf rank.
///
/// The declared leaf count is checked against the class count and the bytes
/// actually present before anything is reserved, and every value must be the
/// finite `0..=1` a fitted leaf distribution holds.
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

/// One tree's stored probability of class `1`.
///
/// Clamped for the same reason the forest clamps its average: the value is read
/// straight out of a decoded artifact on a restored model, and a probability
/// entry point must not be the place where an out-of-range byte first shows up.
#[inline]
fn positive_probability(tree: &PackedTree, row: &[f32]) -> f32 {
    tree.predict(row).clamp(0.0, 1.0)
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

fn grower_config(params: &DecisionTreeClassifierParams) -> GrowerConfig {
    GrowerConfig {
        max_depth: params.max_depth(),
        min_samples_split: params.min_samples_split(),
        min_samples_leaf: params.min_samples_leaf(),
        max_features: params.max_features(),
        splitter: params.splitter(),
    }
}

impl Estimator for DecisionTreeClassifier {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Classifier for DecisionTreeClassifier {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        DecisionTreeClassifier::predict_into(self, data, output)
    }
}

impl ProbabilisticClassifier for DecisionTreeClassifier {
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        DecisionTreeClassifier::predict_proba_into(self, data, output)
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        DecisionTreeClassifier::predict_class_proba_into(self, data, class, output)
    }
}

impl HasParams for DecisionTreeClassifier {
    type Params = DecisionTreeClassifierParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

/// Declares weighted fitting, multiclass fitting, persistence, and genuine
/// probabilities. A tree's leaf *is* a distribution over the training rows that
/// reached it, so the probability declaration is earned rather than squashed
/// out of a score.
impl HasCapabilities for DecisionTreeClassifier {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true)
        .with_multiclass(true)
        .with_probability(true);
}
