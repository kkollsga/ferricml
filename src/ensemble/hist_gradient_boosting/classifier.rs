//! Deterministic dense histogram gradient-boosted binary classification.

use super::binning::Binner;
use super::controls::{
    BoostingControls, map_boosting_error, validate_control_bounds, validate_controls,
};
use super::error::{MAX_TOTAL_NODES, MAX_TREES};
use super::grower::{SampleStatistics, grow_tree};
use super::predictor::{CompactTree, prediction_bound_is_finite};
use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, HasParams, ModelError,
    ProbabilisticClassifier, validate_prediction, validate_scalar_row,
};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, HIST_GRADIENT_BOOSTING_CLASSIFIER_ARTIFACT_KIND,
    MIN_ENCODED_TREE_BYTES, ModelArtifact, SchemaRole, decode_component, decode_logical_tree,
    decode_v2_envelope, encode_component, encode_logical_tree, encode_v2_envelope,
};
use crate::data::{BinaryTargets, MatrixView, SampleWeights};
use crate::loss::{BinaryLogLoss, BoostingObjective};
use crate::numeric::{sigmoid_f32, sum_in_order};

const ARTIFACT_PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;

/// The objective this estimator's artifacts are fitted against.
///
/// Read from the objective type, so the field can never name a loss the model
/// was not fitted with, and distinct from the regressor's by construction.
const OBJECTIVE_VERSION: u32 = <BinaryLogLoss as BoostingObjective>::ARTIFACT_OBJECTIVE_TAG;

/// The class labels a binary fit reports, in sorted order.
///
/// A boosted classifier requires both labels to be present, so its class list is
/// not fitted data — it is `[0, 1]` for every model of this type. That is why
/// the artifact does not store it: the payload version already says so.
const BINARY_CLASSES: [u8; 2] = [0, 1];

/// Parameters for [`HistGradientBoostingClassifier`].
///
/// Every control has the same name, default, and meaning as the regressor's.
/// They are a separate type rather than a shared one so a classifier's builder
/// cannot be handed a regressor's configuration, while the validation behind
/// them is one implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct HistGradientBoostingClassifierParams {
    learning_rate: f32,
    max_iter: usize,
    max_leaf_nodes: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2_regularization: f32,
    max_bins: usize,
}

impl Default for HistGradientBoostingClassifierParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iter: 100,
            max_leaf_nodes: 31,
            max_depth: None,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            max_bins: 255,
        }
    }
}

impl HistGradientBoostingClassifierParams {
    /// Sets the shrinkage applied to each fitted tree.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the number of boosting iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the maximum number of leaves in each tree.
    #[must_use]
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: usize) -> Self {
        self.max_leaf_nodes = max_leaf_nodes;
        self
    }

    /// Sets the maximum depth of each tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum training weight in each leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Sets the non-negative L2 term in each leaf denominator.
    #[must_use]
    pub fn with_l2_regularization(mut self, l2_regularization: f32) -> Self {
        self.l2_regularization = l2_regularization;
        self
    }

    /// Sets the maximum number of histogram bins per feature.
    #[must_use]
    pub fn with_max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = max_bins;
        self
    }

    /// Returns the per-tree shrinkage.
    pub const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the requested number of boosting iterations.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the per-tree leaf-count limit.
    pub const fn max_leaf_nodes(&self) -> usize {
        self.max_leaf_nodes
    }

    /// Returns the per-tree depth limit.
    pub const fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Returns the minimum training weight in each leaf.
    pub const fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Returns the L2 leaf-denominator term.
    pub const fn l2_regularization(&self) -> f32 {
        self.l2_regularization
    }

    /// Returns the maximum histogram bin count.
    pub const fn max_bins(&self) -> usize {
        self.max_bins
    }

    pub(super) const fn controls(&self) -> BoostingControls {
        BoostingControls {
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            max_leaf_nodes: self.max_leaf_nodes,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            l2_regularization: self.l2_regularization,
            max_bins: self.max_bins,
        }
    }
}

/// Serial binary log-loss histogram gradient-boosted classifier.
///
/// The model is an additive ensemble in **raw score** space: a fitted baseline
/// plus the shrunk prediction of every tree. The logistic sigmoid of that score
/// is the probability of class `1`, which is what
/// [`predict_proba`](ProbabilisticClassifier::predict_proba) reports in its second column.
///
/// Each boosting iteration fits one tree to the negative gradient of the binary
/// log loss, `y - p`, and divides every leaf by the summed curvature `p(1 - p)`
/// of its rows rather than by their count. That distinction is the whole
/// difference from the squared-error regressor: with a constant hessian the two
/// denominators coincide, and with a varying one they do not.
///
/// Only two-class fitting exists. Multiclass boosting is a separate model with a
/// separate objective, not a widening of this one, and is deliberately absent
/// rather than approximated — see [`Self::fit`].
///
/// `min_samples_leaf` defaults to `20`, so a dataset smaller than twice that
/// admits no split at all and the fit returns the baseline alone. The example
/// below uses 64 rows for that reason.
///
/// ```
/// use ferricml::api::{Classifier, ProbabilisticClassifier};
/// use ferricml::data::{BinaryTargets, DenseMatrix};
/// use ferricml::ensemble::{
///     HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
/// };
///
/// let values: Vec<f32> = (0..64).map(|index| index as f32).collect();
/// let labels: Vec<u8> = (0..64).map(|index| u8::from(index >= 32)).collect();
/// let data = DenseMatrix::new(values, 64, 1)?;
/// let labels = BinaryTargets::new(labels)?;
///
/// let model = HistGradientBoostingClassifier::fit(
///     &data.as_view(),
///     &labels,
///     HistGradientBoostingClassifierParams::default(),
/// )?;
///
/// // Both labels must be present to fit, so the class list is always [0, 1].
/// assert_eq!(model.classes(), &[0, 1]);
/// assert_eq!(model.predict(&data.as_view())?, labels.as_slice().to_vec());
///
/// // The second probability column is the sigmoid of the raw additive score.
/// let probabilities = model.predict_proba(&data.as_view())?;
/// assert!(probabilities[1] < 0.5); // first row, truly class 0
/// assert!(probabilities[127] > 0.5); // last row, truly class 1
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct HistGradientBoostingClassifier {
    n_features_in: usize,
    params: HistGradientBoostingClassifierParams,
    baseline: f32,
    trees: Vec<CompactTree>,
}

impl HistGradientBoostingClassifier {
    /// Fits a deterministic dense binary log-loss boosted ensemble.
    ///
    /// Both class labels must be present. A single observed class is reported as
    /// [`ModelError::RequiresTwoClasses`], the same refusal
    /// [`LogisticRegression`](crate::linear_model::LogisticRegression) gives:
    /// the log-loss baseline is the log-odds of the positive rate, which is not
    /// a finite number when that rate is `0` or `1`, so a one-class fit has no
    /// model to report rather than a degenerate one. The reference implementation
    /// instead fits and returns two probability columns beside a one-element
    /// class list, a shape inconsistency FerricML does not reproduce.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: HistGradientBoostingClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits with per-row sample weights.
    ///
    /// A weight scales the row's gradient, its curvature, and its share of every
    /// node's weight total, so the baseline is the log-odds of the weighted
    /// positive rate and the minimum leaf size counts weight rather than rows.
    /// Weights of exactly one reproduce [`Self::fit`] bit for bit, and an integer
    /// weight is the same fit as repeating that row that many times.
    ///
    /// A class whose rows all carry weight zero is not present in the training
    /// sample, so it is refused exactly as an unobserved class is.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: &SampleWeights,
        params: HistGradientBoostingClassifierParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        sample_weights: Option<&SampleWeights>,
        params: HistGradientBoostingClassifierParams,
    ) -> Result<Self, ModelError> {
        if data.rows() != targets.len() {
            return Err(ModelError::TargetLength {
                rows: data.rows(),
                targets: targets.len(),
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
        validate_controls(params.controls())?;
        validate_control_bounds(params.controls())?;

        let weights = sample_weights.map(SampleWeights::as_slice);
        let baseline = fitted_baseline(targets.as_slice(), weights)?;
        let binner = Binner::fit(data, params.max_bins).map_err(map_boosting_error)?;
        let binned = binner.transform(data).map_err(map_boosting_error)?;

        let mut raw_scores = vec![baseline; data.rows()];
        let mut negative_gradients = vec![0.0_f32; data.rows()];
        let mut curvatures = vec![0.0_f32; data.rows()];
        let mut trees = Vec::with_capacity(params.max_iter);
        let config = params.controls().grow_config();
        for _ in 0..params.max_iter {
            update_statistics(
                targets.as_slice(),
                &raw_scores,
                &mut negative_gradients,
                &mut curvatures,
            )?;
            let tree = grow_tree::<BinaryLogLoss>(
                &binned,
                &binner,
                SampleStatistics {
                    negative_gradients: &negative_gradients,
                    hessians: &curvatures,
                    sample_weights: weights,
                },
                config,
            )
            .map_err(map_boosting_error)?;
            tree.add_predictions(data, params.learning_rate, &mut raw_scores);
            if raw_scores.iter().any(|score| !score.is_finite()) {
                return Err(ModelError::NumericalOverflow);
            }
            trees.push(tree);
        }
        if !prediction_bound_is_finite(baseline, params.learning_rate, &trees) {
            return Err(ModelError::NumericalOverflow);
        }
        Ok(Self {
            n_features_in: data.columns(),
            params,
            baseline,
            trees,
        })
    }

    /// Returns the fitted baseline raw score.
    ///
    /// This is the log-odds of the training positive rate, which is the constant
    /// raw score minimizing the log loss before any tree is added.
    pub const fn baseline(&self) -> f32 {
        self.baseline
    }

    /// Returns the number of fitted boosting iterations.
    pub fn n_iter(&self) -> usize {
        self.trees.len()
    }

    /// Returns the fitted input width.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact fitted parameters.
    pub const fn get_params(&self) -> &HistGradientBoostingClassifierParams {
        &self.params
    }

    /// Returns the sorted class labels, always `[0, 1]`.
    pub const fn classes(&self) -> &[u8] {
        &BINARY_CLASSES
    }

    /// The additive raw score of one row: the quantity trees sum into.
    fn raw_score(&self, row: &[f32]) -> f32 {
        self.trees.iter().fold(self.baseline, |score, tree| {
            score + self.params.learning_rate * tree.predict_one(row)
        })
    }

    /// Returns the raw, unsquashed decision score for one row.
    ///
    /// Positive scores favour class `1`. The sigmoid of this value is
    /// [`Self::predict_positive_proba_one`].
    pub fn decision_function_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        validate_scalar_row(row, self.n_features_in)?;
        validate_prediction(self.raw_score(row), 0)
    }

    /// Returns one raw decision score per row, allocating the output.
    pub fn decision_function(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        // Before the buffer, not inside `_into` after it. `check_batch` repeats
        // the same check for callers that reach the `_into` form directly.
        self.check_batch(data, data.rows(), data.rows())?;
        let mut output = vec![0.0; data.rows()];
        self.decision_function_into(data, &mut output)?;
        Ok(output)
    }

    /// Writes one raw decision score per row into caller-owned storage.
    pub fn decision_function_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.check_batch(data, output.len(), data.rows())?;
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            *slot = validate_prediction(self.raw_score(row), row_index)?;
        }
        Ok(())
    }

    /// Predicts the probability of class `1` for one row.
    pub fn predict_positive_proba_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        Ok(sigmoid_f32(self.decision_function_one(row)?))
    }

    /// Predicts the probability of class `1` per row, allocating the output.
    pub fn predict_positive_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        // Before the buffer, not inside `_into` after it. `check_batch` repeats
        // the same check for callers that reach the `_into` form directly.
        self.check_batch(data, data.rows(), data.rows())?;
        let mut output = vec![0.0; data.rows()];
        self.predict_positive_proba_into(data, &mut output)?;
        Ok(output)
    }

    /// Writes the probability of class `1` per row into caller-owned storage.
    pub fn predict_positive_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.decision_function_into(data, output)?;
        for slot in output {
            *slot = sigmoid_f32(*slot);
        }
        Ok(())
    }

    /// Predicts one label.
    ///
    /// An exactly balanced probability resolves to class `0`, the lower of the
    /// two labels, matching every other FerricML binary classifier.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        Ok(u8::from(self.predict_positive_proba_one(row)? > 0.5))
    }

    /// Predicts one label per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Predicts one label per row into caller-owned storage.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        <Self as Classifier>::predict_into(self, data, output)
    }

    /// Predicts row-major probabilities with one column per class.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities into caller-owned storage.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba_into(self, data, output)
    }

    /// Predicts one requested class probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_class_proba(self, data, class)
    }

    /// Predicts one requested class probability column without allocating.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as ProbabilisticClassifier>::predict_class_proba_into(self, data, class, output)
    }

    /// Rejects a batch whose feature width is not the fitted one.
    ///
    /// Split out of [`Self::check_batch`] so an entry point that also resolves
    /// a requested class can validate the shape of the input before the
    /// content of the request, which is the crate's uniform precedence.
    fn check_batch_width(&self, data: &MatrixView<'_>) -> Result<(), ModelError> {
        if data.columns() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: data.columns(),
            });
        }
        Ok(())
    }

    /// Validates a batch request's feature width and output length together.
    fn check_batch(
        &self,
        data: &MatrixView<'_>,
        output_len: usize,
        expected: usize,
    ) -> Result<(), ModelError> {
        self.check_batch_width(data)?;
        if output_len != expected {
            return Err(ModelError::OutputLength {
                expected,
                actual: output_len,
            });
        }
        Ok(())
    }
}

impl ModelArtifact for HistGradientBoostingClassifier {
    const ARTIFACT_KIND: u16 = HIST_GRADIENT_BOOSTING_CLASSIFIER_ARTIFACT_KIND;

    /// Encodes the fitted baseline, parameters, and canonical logical trees.
    ///
    /// The payload is shaped exactly like the regressor's — the same twelve
    /// metadata words followed by one length-delimited logical tree per
    /// iteration — because the two models differ in what their leaves *mean*,
    /// not in what a tree record has to say. What separates them is the estimator
    /// kind, checked before a byte is hashed, and the objective word, which names
    /// binary log loss rather than squared error. Either one alone would refuse a
    /// crossed artifact; both are written because they answer different questions
    /// and the second is what a future second loss under this kind would select
    /// on.
    ///
    /// The class list is not stored. Both labels are required to fit, so `[0, 1]`
    /// is a property of the payload version rather than of the fitted data, and
    /// storing it would add a field whose only valid value decode would have to
    /// re-derive anyway. A multiclass boosted model is a different payload
    /// version, and that version is where an observed class list belongs.
    fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let n_features =
            u32::try_from(self.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_leaf_nodes =
            u32::try_from(self.params.max_leaf_nodes).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_depth = self
            .params
            .max_depth
            .map(u32::try_from)
            .transpose()
            .map_err(|_| ArtifactError::InvalidPayload)?
            .unwrap_or(0);
        let min_samples_leaf = u32::try_from(self.params.min_samples_leaf)
            .map_err(|_| ArtifactError::InvalidPayload)?;
        let max_bins =
            u32::try_from(self.params.max_bins).map_err(|_| ArtifactError::InvalidPayload)?;
        let tree_count =
            u32::try_from(self.trees.len()).map_err(|_| ArtifactError::InvalidPayload)?;
        let total_nodes = self.trees.iter().try_fold(0_usize, |total, tree| {
            total
                .checked_add(tree.nodes().len())
                .ok_or(ArtifactError::InvalidPayload)
        })?;
        let total_nodes = u32::try_from(total_nodes).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut metadata = ArtifactPayloadWriter::with_capacity(12 * 4);
        metadata.u32(OBJECTIVE_VERSION);
        metadata.u32(n_features);
        metadata.f32(self.params.learning_rate);
        metadata.u32(max_iter);
        metadata.u32(max_leaf_nodes);
        metadata.u32(max_depth);
        metadata.u32(min_samples_leaf);
        metadata.f32(self.params.l2_regularization);
        metadata.u32(max_bins);
        metadata.f32(self.baseline);
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
            Self::ARTIFACT_KIND,
            ARTIFACT_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes and revalidates every parameter and tree before building state.
    ///
    /// Nothing decoded is trusted. Every parameter re-enters the same validators
    /// fitting uses, every tree re-enters the same topology validator, every
    /// declared count is checked against the bytes actually present before a
    /// reservation is made, and the aggregate node total must match what the
    /// trees really contain. A model whose raw score could not stay finite is
    /// refused rather than left to report an infinity at prediction time.
    fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            Self::ARTIFACT_KIND,
            ARTIFACT_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let objective_version = metadata.u32()?;
        let n_features_in = metadata.u32()? as usize;
        let learning_rate = metadata.f32()?;
        let max_iter = metadata.u32()? as usize;
        let max_leaf_nodes = metadata.u32()? as usize;
        let encoded_depth = metadata.u32()? as usize;
        let min_samples_leaf = metadata.u32()? as usize;
        let l2_regularization = metadata.f32()?;
        let max_bins = metadata.u32()? as usize;
        let baseline = metadata.f32()?;
        let tree_count = metadata.u32()? as usize;
        let declared_total_nodes = metadata.u32()? as usize;
        let params = HistGradientBoostingClassifierParams {
            learning_rate,
            max_iter,
            max_leaf_nodes,
            max_depth: (encoded_depth != 0).then_some(encoded_depth),
            min_samples_leaf,
            l2_regularization,
            max_bins,
        };
        if !metadata.is_empty()
            || objective_version != OBJECTIVE_VERSION
            || n_features_in == 0
            || n_features_in > 1_000_000
            || !baseline.is_finite()
            || tree_count == 0
            || tree_count > MAX_TREES
            || tree_count != max_iter
            || declared_total_nodes == 0
            || declared_total_nodes > MAX_TOTAL_NODES
            || validate_controls(params.controls()).is_err()
            || validate_control_bounds(params.controls()).is_err()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut trees =
            Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
        let mut actual_total_nodes = 0_usize;
        for _ in 0..tree_count {
            let logical_nodes = decode_logical_tree(decode_component(
                &mut envelope,
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
            )?)?;
            let tree = CompactTree::from_logical_nodes(logical_nodes, n_features_in)?;
            actual_total_nodes = actual_total_nodes
                .checked_add(tree.nodes().len())
                .ok_or(ArtifactError::InvalidPayload)?;
            if actual_total_nodes > declared_total_nodes || actual_total_nodes > MAX_TOTAL_NODES {
                return Err(ArtifactError::InvalidPayload);
            }
            trees.push(tree);
        }
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        if actual_total_nodes != declared_total_nodes {
            return Err(ArtifactError::InvalidPayload);
        }
        if !prediction_bound_is_finite(baseline, params.learning_rate, &trees) {
            return Err(ArtifactError::InvalidPayload);
        }
        Ok(Self {
            n_features_in,
            params,
            baseline,
            trees,
        })
    }
}

impl Estimator for HistGradientBoostingClassifier {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

/// Weighted fitting, persistence, probabilities, and a raw decision score are
/// all offered; multiclass fitting is not, because a multiclass boosted model
/// needs the multinomial objective and one tree per class per iteration, which
/// is a different model rather than a wider fit of this one.
impl HasCapabilities for HistGradientBoostingClassifier {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true)
        .with_decision_function(true)
        .with_probability(true);
}

impl HasParams for HistGradientBoostingClassifier {
    type Params = HistGradientBoostingClassifierParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Classifier for HistGradientBoostingClassifier {
    fn classes(&self) -> &[u8] {
        &BINARY_CLASSES
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        self.check_batch(data, output.len(), data.rows())?;
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            let score = validate_prediction(self.raw_score(row), row_index)?;
            *slot = u8::from(sigmoid_f32(score) > 0.5);
        }
        Ok(())
    }
}

impl ProbabilisticClassifier for HistGradientBoostingClassifier {
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let expected = data.rows().checked_mul(BINARY_CLASSES.len()).ok_or(
            ModelError::OutputShapeOverflow {
                rows: data.rows(),
                columns: BINARY_CLASSES.len(),
            },
        )?;
        self.check_batch(data, output.len(), expected)?;
        for (row_index, (row, probabilities)) in data
            .iter_rows()
            .zip(output.chunks_exact_mut(BINARY_CLASSES.len()))
            .enumerate()
        {
            let score = validate_prediction(self.raw_score(row), row_index)?;
            let positive = sigmoid_f32(score);
            probabilities[0] = 1.0 - positive;
            probabilities[1] = positive;
        }
        Ok(())
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        // Width before class: the shape of the input is validated before the
        // content of the request, so this primitive reports the same error its
        // allocating partner does.
        self.check_batch_width(data)?;
        let column = BINARY_CLASSES
            .binary_search(&class)
            .map_err(|_| ModelError::UnknownClass { class })?;
        self.check_batch(data, output.len(), data.rows())?;
        for (row_index, (row, slot)) in data.iter_rows().zip(output).enumerate() {
            let score = validate_prediction(self.raw_score(row), row_index)?;
            let positive = sigmoid_f32(score);
            *slot = if column == 1 {
                positive
            } else {
                1.0 - positive
            };
        }
        Ok(())
    }
}

/// The constant raw score that minimizes the log loss over the training rows.
///
/// Minimizing `sum w_i * (softplus(raw) - y_i * raw)` over one constant `raw`
/// puts the fitted probability at the weighted positive rate, so the baseline is
/// that rate's log-odds. Both accumulations run in ascending row order under the
/// crate's accumulation policy, so the baseline is reproducible bit for bit.
///
/// The refusal below is what keeps that value finite: a rate of exactly `0` or
/// `1` has no finite log-odds, and a class carrying no weight is a class that is
/// not in the training sample.
fn fitted_baseline(targets: &[u8], sample_weights: Option<&[f32]>) -> Result<f32, ModelError> {
    let (positive, total) = match sample_weights {
        None => (
            sum_in_order(targets.iter().map(|&target| f64::from(target))),
            targets.len() as f64,
        ),
        Some(weights) => (
            sum_in_order(
                targets
                    .iter()
                    .zip(weights)
                    .map(|(&target, &weight)| f64::from(weight) * f64::from(target)),
            ),
            sum_in_order(weights.iter().map(|&weight| f64::from(weight))),
        ),
    };
    if positive <= 0.0 || positive >= total {
        return Err(ModelError::RequiresTwoClasses);
    }
    let baseline = (positive / (total - positive)).ln() as f32;
    if !baseline.is_finite() {
        return Err(ModelError::NumericalOverflow);
    }
    Ok(baseline)
}

/// Refreshes the per-row negative gradient and curvature at the current scores.
///
/// Both come from one evaluation of the objective's inverse link per row, which
/// is the only transcendental on the fitting path. Storing them as `f32` matches
/// the width every other fitted quantity in this family uses; the arithmetic
/// itself is `f64`.
fn update_statistics(
    targets: &[u8],
    raw_scores: &[f32],
    negative_gradients: &mut [f32],
    curvatures: &mut [f32],
) -> Result<(), ModelError> {
    debug_assert_eq!(targets.len(), raw_scores.len());
    for (index, (&target, &score)) in targets.iter().zip(raw_scores).enumerate() {
        let (gradient, curvature) =
            BinaryLogLoss::negative_gradient_and_curvature(f64::from(score), f64::from(target));
        let gradient = gradient as f32;
        let curvature = curvature as f32;
        if !gradient.is_finite() || !curvature.is_finite() {
            return Err(ModelError::NonFinitePrediction { row: index });
        }
        negative_gradients[index] = gradient;
        curvatures[index] = curvature;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND;
    use crate::data::DenseMatrix;
    use crate::ensemble::HistGradientBoostingRegressor;
    use crate::ensemble::hist_gradient_boosting::error::{MAX_TREE_LEAVES, MAX_TREES};
    use sha2::{Digest, Sha256};

    fn resign_artifact(bytes: &mut [u8]) {
        let checksum_offset = bytes.len() - 32;
        let checksum = Sha256::digest(&bytes[..checksum_offset]);
        bytes[checksum_offset..].copy_from_slice(&checksum);
    }

    /// Eight rows of distinct integer features, separable at 3.5.
    fn separable() -> (DenseMatrix, BinaryTargets) {
        (
            DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap(),
            BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).unwrap(),
        )
    }

    fn one_tree_params() -> HistGradientBoostingClassifierParams {
        HistGradientBoostingClassifierParams::default()
            .with_learning_rate(1.0)
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1)
            .with_max_bins(8)
    }

    fn weight_params() -> HistGradientBoostingClassifierParams {
        HistGradientBoostingClassifierParams::default()
            .with_max_iter(4)
            .with_max_leaf_nodes(4)
            .with_min_samples_leaf(1)
            .with_max_bins(8)
    }

    /// One Newton step on a balanced separable split, computed by hand.
    ///
    /// The baseline is the log-odds of a positive rate of one half, which is
    /// exactly zero, so every row starts at `p = 0.5`. Each side then has
    /// gradient `+-4 * 0.5 = +-2` and curvature `4 * 0.25 = 1`, giving leaves of
    /// `-+2` unregularized and `-+2 / 3` at `l2 = 2`. Both values agree with the
    /// reference's `decision_function` to float32 rounding.
    #[test]
    fn one_boosting_update_matches_the_hand_computed_newton_step() {
        let (data, targets) = separable();
        for (l2, expected) in [(0.0_f32, 2.0_f32), (2.0, 2.0 / 3.0)] {
            let model = HistGradientBoostingClassifier::fit(
                &data.as_view(),
                &targets,
                one_tree_params().with_l2_regularization(l2),
            )
            .unwrap();
            assert_eq!(model.baseline(), 0.0);
            assert_eq!(model.n_iter(), 1);
            let scores = model.decision_function(&data.as_view()).unwrap();
            for score in &scores[..4] {
                assert!((score + expected).abs() <= 1.0e-6, "left leaf {score}");
            }
            for score in &scores[4..] {
                assert!((score - expected).abs() <= 1.0e-6, "right leaf {score}");
            }
            assert_eq!(
                model.predict(&data.as_view()).unwrap(),
                targets.as_slice(),
                "l2 = {l2}"
            );
        }
    }

    /// The leaf denominator is the summed curvature, which is what makes this
    /// model different from a squared-error fit of the same labels.
    ///
    /// After one unshrunk step the rows are no longer at `p = 0.5`, so the second
    /// step's curvature is not its row count and a count denominator would give a
    /// visibly different score.
    #[test]
    fn the_second_step_divides_by_curvature_rather_than_by_row_count() {
        let (data, targets) = separable();
        let model = HistGradientBoostingClassifier::fit(
            &data.as_view(),
            &targets,
            one_tree_params().with_max_iter(2),
        )
        .unwrap();
        let score = model.decision_function(&data.as_view()).unwrap()[7];

        // Hand-rolled second step: p = sigmoid(2) on the positive side.
        let probability = 1.0_f64 / (1.0 + (-2.0_f64).exp());
        let gradient = 4.0 * (1.0 - probability);
        let curvature = 4.0 * probability * (1.0 - probability);
        let by_curvature = 2.0 + (gradient / curvature) as f32;
        let by_row_count = 2.0 + (gradient / 4.0) as f32;
        assert!(
            (score - by_curvature).abs() <= 1.0e-5,
            "second step {score} vs curvature-denominated {by_curvature}"
        );
        assert!(
            (score - by_row_count).abs() > 1.0,
            "a row-count denominator would have been distinguishable"
        );
    }

    #[test]
    fn probabilities_are_bounded_sum_to_one_and_agree_with_the_label() {
        let data = DenseMatrix::new(
            (0..40).map(|value| (value % 7) as f32 - 3.0).collect(),
            20,
            2,
        )
        .unwrap();
        let targets =
            BinaryTargets::new((0..20).map(|row| u8::from(row % 3 == 0)).collect()).unwrap();
        let model = HistGradientBoostingClassifier::fit(
            &data.as_view(),
            &targets,
            HistGradientBoostingClassifierParams::default()
                .with_max_iter(12)
                .with_max_leaf_nodes(4)
                .with_min_samples_leaf(1)
                .with_max_bins(8),
        )
        .unwrap();
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        let labels = model.predict(&data.as_view()).unwrap();
        let scores = model.decision_function(&data.as_view()).unwrap();
        assert_eq!(model.classes(), &[0, 1]);
        assert_eq!(probabilities.len(), 2 * data.rows());
        for (row, (pair, (&label, &score))) in probabilities
            .chunks_exact(2)
            .zip(labels.iter().zip(&scores))
            .enumerate()
        {
            assert!((0.0..=1.0).contains(&pair[0]), "row {row}: {pair:?}");
            assert!((0.0..=1.0).contains(&pair[1]), "row {row}: {pair:?}");
            assert!((pair[0] + pair[1] - 1.0).abs() <= 1.0e-6, "row {row}");
            // The label is the argmax of the row a caller can read, and the
            // score's sign is the same decision.
            assert_eq!(label, u8::from(pair[1] > pair[0]), "row {row}");
            assert_eq!(label, u8::from(score > 0.0), "row {row}");
        }
    }

    #[test]
    fn scalar_batch_column_and_into_predictions_all_agree() {
        let (data, targets) = separable();
        let model = HistGradientBoostingClassifier::fit(
            &data.as_view(),
            &targets,
            one_tree_params().with_max_iter(3).with_max_leaf_nodes(3),
        )
        .unwrap();
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        let mut proba_into = vec![0.0; probabilities.len()];
        model
            .predict_proba_into(&data.as_view(), &mut proba_into)
            .unwrap();
        assert_eq!(probabilities, proba_into);

        let labels = model.predict(&data.as_view()).unwrap();
        let mut labels_into = vec![9_u8; labels.len()];
        model
            .predict_into(&data.as_view(), &mut labels_into)
            .unwrap();
        assert_eq!(labels, labels_into);

        let scores = model.decision_function(&data.as_view()).unwrap();
        let mut scores_into = vec![0.0; scores.len()];
        model
            .decision_function_into(&data.as_view(), &mut scores_into)
            .unwrap();
        assert_eq!(scores, scores_into);

        for class in [0_u8, 1] {
            let column = model.predict_class_proba(&data.as_view(), class).unwrap();
            for (row, &value) in column.iter().enumerate() {
                assert_eq!(value, probabilities[row * 2 + usize::from(class)]);
            }
        }

        for (row_index, row) in data.iter_rows().enumerate() {
            assert_eq!(model.predict_one(row).unwrap(), labels[row_index]);
            assert_eq!(
                model.predict_positive_proba_one(row).unwrap(),
                probabilities[row_index * 2 + 1]
            );
            assert_eq!(model.decision_function_one(row).unwrap(), scores[row_index]);
        }
    }

    #[test]
    fn refitting_the_same_inputs_reproduces_the_same_model() {
        let (data, targets) = separable();
        let first = HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
            .unwrap();
        let second =
            HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
                .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn unit_weights_reproduce_the_unweighted_fit_bit_for_bit() {
        let (data, targets) = separable();
        let ones = SampleWeights::new(vec![1.0; data.rows()]).unwrap();
        let plain = HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
            .unwrap();
        let weighted = HistGradientBoostingClassifier::fit_weighted(
            &data.as_view(),
            &targets,
            &ones,
            weight_params(),
        )
        .unwrap();
        assert_eq!(plain, weighted);
        assert_eq!(plain.baseline().to_bits(), weighted.baseline().to_bits());
    }

    /// An integer weight is the same fit as repeating the row that many times.
    ///
    /// The bin grid survives the comparison because it is fitted from the
    /// distinct observed values, which a repeated row does not change.
    #[test]
    fn an_integer_weight_is_the_same_fit_as_repeating_the_row() {
        let (data, targets) = separable();
        let repeat = 5;
        let times = 3;
        let mut weights = vec![1.0_f32; data.rows()];
        weights[repeat] = times as f32;
        let weighted = HistGradientBoostingClassifier::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(weights).unwrap(),
            weight_params(),
        )
        .unwrap();

        let mut rows = Vec::new();
        let mut repeated_labels = Vec::new();
        for (row, &label) in targets.as_slice().iter().enumerate() {
            let copies = if row == repeat { times } else { 1 };
            for _ in 0..copies {
                rows.extend_from_slice(data.as_view().row(row).unwrap());
                repeated_labels.push(label);
            }
        }
        let repeated_data = DenseMatrix::new(rows, repeated_labels.len(), 1).unwrap();
        let repeated = HistGradientBoostingClassifier::fit(
            &repeated_data.as_view(),
            &BinaryTargets::new(repeated_labels).unwrap(),
            weight_params(),
        )
        .unwrap();

        assert_eq!(weighted.baseline().to_bits(), repeated.baseline().to_bits());
        assert_eq!(
            weighted.decision_function(&data.as_view()).unwrap(),
            repeated.decision_function(&data.as_view()).unwrap()
        );
    }

    #[test]
    fn weighting_is_not_inert_and_a_length_mismatch_is_refused() {
        let (data, targets) = separable();
        assert_eq!(
            HistGradientBoostingClassifier::fit_weighted(
                &data.as_view(),
                &targets,
                &SampleWeights::new(vec![1.0; data.rows() - 1]).unwrap(),
                weight_params(),
            )
            .unwrap_err(),
            ModelError::SampleWeightLength {
                rows: data.rows(),
                weights: data.rows() - 1,
            }
        );

        let mut skewed = vec![1.0_f32; data.rows()];
        skewed[0] = 9.0;
        let weighted = HistGradientBoostingClassifier::fit_weighted(
            &data.as_view(),
            &targets,
            &SampleWeights::new(skewed).unwrap(),
            weight_params(),
        )
        .unwrap();
        let plain = HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
            .unwrap();
        assert_ne!(plain.baseline(), weighted.baseline());
    }

    /// The baseline is the log-odds of the (weighted) positive rate.
    #[test]
    fn the_baseline_is_the_log_odds_of_the_positive_rate() {
        let data = DenseMatrix::new(vec![1.0; 8], 8, 1).unwrap();
        for labels in [
            vec![0_u8, 0, 0, 0, 0, 1, 1, 1],
            vec![0, 1, 1, 1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        ] {
            let positives = labels.iter().filter(|&&label| label == 1).count() as f64;
            let rate = positives / labels.len() as f64;
            let model = HistGradientBoostingClassifier::fit(
                &data.as_view(),
                &BinaryTargets::new(labels).unwrap(),
                one_tree_params(),
            )
            .unwrap();
            assert_eq!(model.baseline(), (rate / (1.0 - rate)).ln() as f32);
            // A constant feature admits no split, so every leaf is the root's
            // Newton step, and that step is zero at the optimal baseline.
            let probabilities = model.predict_proba(&data.as_view()).unwrap();
            assert!((f64::from(probabilities[1]) - rate).abs() <= 1.0e-6);
        }
    }

    /// One observed class has no finite log-odds, so it is refused rather than
    /// answered with a degenerate model.
    #[test]
    fn a_single_observed_class_is_refused() {
        let (data, _) = separable();
        for labels in [vec![0_u8; 8], vec![1_u8; 8]] {
            assert_eq!(
                HistGradientBoostingClassifier::fit(
                    &data.as_view(),
                    &BinaryTargets::new(labels).unwrap(),
                    one_tree_params(),
                ),
                Err(ModelError::RequiresTwoClasses)
            );
        }

        // A class present only on zero-weight rows is not in the sample either.
        let mut weights = vec![1.0_f32; 8];
        for weight in &mut weights[4..] {
            *weight = 0.0;
        }
        assert_eq!(
            HistGradientBoostingClassifier::fit_weighted(
                &data.as_view(),
                &BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).unwrap(),
                &SampleWeights::new(weights).unwrap(),
                one_tree_params(),
            ),
            Err(ModelError::RequiresTwoClasses)
        );
    }

    #[test]
    fn validates_every_parameter_shape_and_output_before_writes() {
        let (data, targets) = separable();
        let cases = [
            (
                HistGradientBoostingClassifierParams::default().with_learning_rate(0.0),
                ModelError::InvalidLearningRate,
            ),
            (
                HistGradientBoostingClassifierParams::default().with_max_iter(0),
                ModelError::InvalidBoostingIterationCount,
            ),
            (
                HistGradientBoostingClassifierParams::default().with_max_leaf_nodes(1),
                ModelError::InvalidMaxLeafNodes,
            ),
            (
                HistGradientBoostingClassifierParams::default().with_max_depth(Some(0)),
                ModelError::InvalidBoostingMaxDepth,
            ),
            (
                HistGradientBoostingClassifierParams::default().with_min_samples_leaf(0),
                ModelError::InvalidMinSamplesLeaf,
            ),
            (
                HistGradientBoostingClassifierParams::default().with_l2_regularization(-1.0),
                ModelError::InvalidL2Regularization,
            ),
            (
                HistGradientBoostingClassifierParams::default().with_max_bins(1),
                ModelError::InvalidMaxBins,
            ),
        ];
        for (params, expected) in cases {
            assert_eq!(
                HistGradientBoostingClassifier::fit(&data.as_view(), &targets, params),
                Err(expected)
            );
        }
        assert_eq!(
            HistGradientBoostingClassifier::fit(
                &data.as_view(),
                &BinaryTargets::new(vec![0, 1, 0]).unwrap(),
                one_tree_params(),
            ),
            Err(ModelError::TargetLength {
                rows: 8,
                targets: 3
            })
        );
        assert_eq!(
            HistGradientBoostingClassifier::fit(
                &data.as_view(),
                &targets,
                HistGradientBoostingClassifierParams::default()
                    .with_max_iter(MAX_TREES)
                    .with_max_leaf_nodes(MAX_TREE_LEAVES),
            ),
            Err(ModelError::BoostingModelTooLarge)
        );

        let model =
            HistGradientBoostingClassifier::fit(&data.as_view(), &targets, one_tree_params())
                .unwrap();
        let mut labels = [7_u8; 7];
        assert_eq!(
            model.predict_into(&data.as_view(), &mut labels),
            Err(ModelError::OutputLength {
                expected: 8,
                actual: 7
            })
        );
        assert_eq!(labels, [7_u8; 7]);
        let mut probabilities = [-1.0_f32; 15];
        assert_eq!(
            model.predict_proba_into(&data.as_view(), &mut probabilities),
            Err(ModelError::OutputLength {
                expected: 16,
                actual: 15
            })
        );
        assert_eq!(probabilities, [-1.0_f32; 15]);
        assert_eq!(
            model.predict_class_proba(&data.as_view(), 2).unwrap_err(),
            ModelError::UnknownClass { class: 2 }
        );

        let wide = DenseMatrix::new(vec![0.0; 16], 8, 2).unwrap();
        assert_eq!(
            model.predict(&wide.as_view()).unwrap_err(),
            ModelError::FeatureDimension {
                expected: 1,
                actual: 2
            }
        );
        assert_eq!(
            model.predict_one(&[1.0, 2.0]).unwrap_err(),
            ModelError::FeatureDimension {
                expected: 1,
                actual: 2
            }
        );
    }

    /// A node whose rows the model already separates has a curvature that
    /// underflows to zero; the objective's floor keeps the leaf defined.
    #[test]
    fn a_fully_saturated_fit_stays_finite() {
        let (data, targets) = separable();
        let model = HistGradientBoostingClassifier::fit(
            &data.as_view(),
            &targets,
            one_tree_params().with_max_iter(64),
        )
        .unwrap();
        let scores = model.decision_function(&data.as_view()).unwrap();
        assert!(scores.iter().all(|score| score.is_finite()));
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        assert!(
            probabilities
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        assert_eq!(model.predict(&data.as_view()).unwrap(), targets.as_slice());
    }

    #[test]
    fn artifact_round_trip_is_deterministic_schema_bound_and_kind_isolated() {
        let (data, targets) = separable();
        let model = HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
            .unwrap();
        let schema = [23; 32];
        let left = model.to_artifact(schema).unwrap();
        assert_eq!(left, model.to_artifact(schema).unwrap());

        let decoded = HistGradientBoostingClassifier::from_artifact(&left, schema).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(decoded.classes(), model.classes());
        assert_eq!(decoded.get_params(), model.get_params());
        assert_eq!(decoded.baseline().to_bits(), model.baseline().to_bits());
        assert_eq!(
            decoded.predict_proba(&data.as_view()).unwrap(),
            model.predict_proba(&data.as_view()).unwrap()
        );
        assert_eq!(
            decoded.decision_function(&data.as_view()).unwrap(),
            model.decision_function(&data.as_view()).unwrap()
        );
        // Canonicity: the decoder re-encodes exactly the bytes it accepted.
        assert_eq!(decoded.to_artifact(schema).unwrap(), left);

        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&left, [24; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );

        let mut corrupted = left.clone();
        corrupted[140] ^= 1;
        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }

    /// A regressor's artifact must never decode as a classifier's, and the two
    /// discriminators must both refuse it: the kind on its own, and — were the
    /// kind ever to coincide — the objective word.
    #[test]
    fn a_regressor_artifact_is_refused_by_kind_and_by_objective() {
        let (data, targets) = separable();
        let schema = [29; 32];
        let classifier =
            HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
                .unwrap();
        let values = crate::data::RegressionTargets::new(
            targets
                .as_slice()
                .iter()
                .map(|&label| f32::from(label))
                .collect(),
        )
        .unwrap();
        let regressor = HistGradientBoostingRegressor::fit(
            &data.as_view(),
            &values,
            crate::ensemble::HistGradientBoostingRegressorParams::default()
                .with_max_iter(4)
                .with_max_leaf_nodes(4)
                .with_min_samples_leaf(1)
                .with_max_bins(8),
        )
        .unwrap();

        let classifier_bytes = classifier.to_artifact(schema).unwrap();
        let regressor_bytes = regressor.to_artifact(schema).unwrap();
        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&regressor_bytes, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind {
                found: HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND,
            }
        );
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&classifier_bytes, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind {
                found: HIST_GRADIENT_BOOSTING_CLASSIFIER_ARTIFACT_KIND,
            }
        );

        // Relabel a classifier artifact with the regressor's kind. The kind check
        // now passes and the objective word is the field that refuses it.
        let mut relabelled = classifier_bytes.clone();
        relabelled[10..12]
            .copy_from_slice(&HIST_GRADIENT_BOOSTING_REGRESSOR_ARTIFACT_KIND.to_le_bytes());
        resign_artifact(&mut relabelled);
        assert_eq!(
            HistGradientBoostingRegressor::from_artifact(&relabelled, schema).unwrap_err(),
            ArtifactError::InvalidPayload,
            "the objective word must refuse a crossed payload the kind admitted"
        );
    }

    #[test]
    fn artifact_rejects_invalid_metadata_tree_records_and_framing() {
        let (data, targets) = separable();
        let model = HistGradientBoostingClassifier::fit(&data.as_view(), &targets, weight_params())
            .unwrap();
        let schema = [31; 32];
        let bytes = model.to_artifact(schema).unwrap();

        // Metadata words, in the order they are written.
        for (name, word, value) in [
            ("objective", 0_usize, 1_u32),
            ("objective zero", 0, 0),
            ("feature width", 1, 0),
            ("tree count disagreeing with max_iter", 10, 3),
            ("total nodes", 11, 4),
            ("max leaf nodes below two", 4, 1),
            ("max bins below two", 8, 1),
        ] {
            let mut corrupted = bytes.clone();
            let offset = 68 + word * 4;
            corrupted[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            resign_artifact(&mut corrupted);
            assert_eq!(
                HistGradientBoostingClassifier::from_artifact(&corrupted, schema).unwrap_err(),
                ArtifactError::InvalidPayload,
                "{name} was accepted"
            );
        }

        // A non-finite baseline is refused rather than stored.
        let mut baseline = bytes.clone();
        baseline[68 + 9 * 4..68 + 10 * 4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        resign_artifact(&mut baseline);
        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&baseline, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut component_kind = bytes.clone();
        component_kind[116..118].copy_from_slice(&3_u16.to_le_bytes());
        resign_artifact(&mut component_kind);
        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&component_kind, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut trailing = bytes.clone();
        trailing.insert(bytes.len() - 32, 0);
        resign_artifact(&mut trailing);
        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&trailing, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&bytes[..120], schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
        assert_eq!(
            HistGradientBoostingClassifier::from_artifact(&[], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }

    #[test]
    fn a_constant_feature_column_fits_deterministically() {
        let data = DenseMatrix::new(vec![1.0; 12], 6, 2).unwrap();
        let targets = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1]).unwrap();
        let params = HistGradientBoostingClassifierParams::default()
            .with_max_iter(4)
            .with_min_samples_leaf(1);
        let first =
            HistGradientBoostingClassifier::fit(&data.as_view(), &targets, params.clone()).unwrap();
        let second =
            HistGradientBoostingClassifier::fit(&data.as_view(), &targets, params).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.baseline(), 0.0);
        assert_eq!(first.predict(&data.as_view()).unwrap(), vec![0; 6]);
    }
}
