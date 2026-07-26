use super::grower::{GrowerConfig, Regression, grow_tree, unbootstrapped_sample};
use super::packed::PackedTree;
use super::parameters::{DecisionTreeRegressorParams, encode_max_features, encode_splitter};
use super::validation::{
    MAX_ARTIFACT_FEATURES, MAX_ARTIFACT_TOTAL_NODES, check_prediction_data, check_row,
    read_common_metadata, tree_seed, validate_fit, write_common_metadata,
};
use crate::api::{
    Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Regressor, validate_prediction,
};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, DECISION_TREE_REGRESSOR_ARTIFACT_KIND, ModelArtifact,
    SchemaRole, decode_component, decode_logical_tree, decode_v2_envelope, encode_component,
    encode_logical_tree, encode_v2_envelope,
};
use crate::data::{MatrixView, RegressionTargets, SampleWeights};
use crate::numeric::OwnedRng;

const PAYLOAD_VERSION: u16 = 1;
const OBJECTIVE_VERSION: u32 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;
const METADATA_BYTES: usize = 9 * 4 + 8;

/// A single regression tree. Predictions are leaf means.
///
/// This is the same tree a random forest grows, not a reimplementation of one:
/// both call one grower under one configuration type. A tree fitted here is
/// bit-identical to the single member of a one-tree, no-bootstrap, all-columns
/// forest at the same seed, and that is asserted rather than assumed.
///
/// The forest is deliberately not named as a rustdoc link here: `tree` sits
/// below every estimator family that consumes it, and the layout ratchet reads
/// a link to one as the dependency it forbids.
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::tree::{DecisionTreeRegressor, DecisionTreeRegressorParams};
///
/// // A step: everything below 2.5 is worth 0, everything above is worth 10.
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
/// let targets = RegressionTargets::new(vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0])?;
///
/// let model = DecisionTreeRegressor::fit(
///     &data.as_view(),
///     &targets,
///     DecisionTreeRegressorParams::default(),
/// )?;
///
/// // A tree can represent a step exactly, which is what a linear fit cannot.
/// assert_eq!(model.predict(&data.as_view())?, vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Depth bounds the fit. A depth-one tree is one split, so it can produce at
/// most two distinct predictions however much structure the data has:
///
/// ```
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::tree::{DecisionTreeRegressor, DecisionTreeRegressorParams};
///
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
/// let targets = RegressionTargets::new(vec![0.0, 1.0, 2.0, 3.0])?;
///
/// let stump = DecisionTreeRegressor::fit(
///     &data.as_view(),
///     &targets,
///     DecisionTreeRegressorParams::default().with_max_depth(Some(1)),
/// )?;
///
/// let mut distinct = stump.predict(&data.as_view())?;
/// distinct.dedup();
/// assert_eq!(distinct.len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionTreeRegressor {
    n_features_in: usize,
    params: DecisionTreeRegressorParams,
    tree: PackedTree,
}

impl DecisionTreeRegressor {
    /// Returns the feature width required by this model.
    #[inline]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact parameters used to fit this model.
    #[inline]
    pub fn get_params(&self) -> &DecisionTreeRegressorParams {
        &self.params
    }

    /// Fits one regression tree.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        params: DecisionTreeRegressorParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, None, params)
    }

    /// Fits with per-row sample weights.
    ///
    /// A weight scales the row's contribution to the variance and leaf mean of
    /// every node it reaches. Weights of exactly one reproduce [`Self::fit`]
    /// bit for bit, and an integer weight is the same fit as repeating that row
    /// that many times — which is why the node-size bounds count summed weight
    /// rather than rows.
    pub fn fit_weighted(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: &SampleWeights,
        params: DecisionTreeRegressorParams,
    ) -> Result<Self, ModelError> {
        Self::fit_internal(data, targets, Some(sample_weights), params)
    }

    fn fit_internal(
        data: &MatrixView<'_>,
        targets: &RegressionTargets,
        sample_weights: Option<&SampleWeights>,
        params: DecisionTreeRegressorParams,
    ) -> Result<Self, ModelError> {
        let config = grower_config(&params);
        validate_fit(
            data,
            targets.as_slice().len(),
            sample_weights.map(SampleWeights::len),
            &config,
        )?;
        for (index, value) in targets.as_slice().iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::NonFiniteTarget { index });
            }
        }
        let sample_weights = sample_weights.map(SampleWeights::as_slice);
        let (weights, rows) = unbootstrapped_sample(data.rows(), sample_weights);
        let mut rng = OwnedRng::new(tree_seed(params.random_state()));
        let tree = grow_tree(
            data,
            targets.as_slice(),
            &weights,
            rows,
            &config,
            Regression,
            &mut rng,
        )?;
        Ok(Self {
            n_features_in: data.columns(),
            params,
            tree,
        })
    }

    /// The packed tree, for the in-crate grower-equivalence proof.
    ///
    /// Never public: the whole point of keeping the layout private is that a
    /// caller cannot come to depend on it. The equivalence test is in-crate
    /// precisely so the claim can be bit-exact without exporting the bytes it
    /// compares.
    #[cfg(test)]
    pub(crate) fn packed(&self) -> &PackedTree {
        &self.tree
    }

    /// Predicts one regression value for one sample.
    pub fn predict_one(&self, row: &[f32]) -> Result<f32, ModelError> {
        check_row(row, self.n_features_in)?;
        validate_prediction(self.tree.predict(row), 0)
    }

    /// Predicts one value per row, allocating the output vector.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predict every row without allocating. `output.len()` must equal the
    /// number of input rows.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        check_prediction_data(data, output.len(), data.rows(), self.n_features_in)?;
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = validate_prediction(
                self.tree
                    .predict(data.row(index).expect("validated row index")),
                index,
            )?;
        }
        Ok(())
    }
}

impl ModelArtifact for DecisionTreeRegressor {
    const ARTIFACT_KIND: u16 = DECISION_TREE_REGRESSOR_ARTIFACT_KIND;

    /// Encodes the fitted parameters and the canonical logical tree.
    ///
    /// The private packed inference layout is never serialized; the tree is
    /// expanded into stable logical records first, so the compact runtime
    /// representation stays free to change.
    fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let node_count = self.tree.logical_node_count();
        if self.n_features_in > MAX_ARTIFACT_FEATURES
            || node_count > MAX_ARTIFACT_TOTAL_NODES
            || !f64::from(self.tree.max_abs_leaf()).is_finite()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let mut metadata = ArtifactPayloadWriter::with_capacity(METADATA_BYTES);
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
        let mut payload = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        payload.extend_from_slice(&encode_component(
            TREE_COMPONENT_KIND,
            COMPONENT_VERSION,
            &encode_logical_tree(&self.tree.to_logical_nodes())?,
        )?);
        encode_v2_envelope(
            Self::ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes and revalidates a tree before building runtime state.
    ///
    /// Parameters and the declared node count are checked before the tree is
    /// read, and the decoded records re-enter the same topology validator that
    /// fitting uses, so the encoded bytes are never trusted.
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
        if !metadata.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        let logical = decode_logical_tree(decode_component(
            &mut envelope,
            TREE_COMPONENT_KIND,
            COMPONENT_VERSION,
        )?)?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        if logical.len() != common.node_count {
            return Err(ArtifactError::InvalidPayload);
        }
        let tree = PackedTree::from_logical_nodes(&logical, common.n_features_in)?;
        if !f64::from(tree.max_abs_leaf()).is_finite() {
            return Err(ArtifactError::InvalidPayload);
        }
        Ok(Self {
            n_features_in: common.n_features_in,
            params: DecisionTreeRegressorParams::default()
                .with_max_depth(common.max_depth)
                .with_min_samples_split(common.min_samples_split)
                .with_min_samples_leaf(common.min_samples_leaf)
                .with_max_features(common.max_features)
                .with_splitter(common.splitter)
                .with_random_state(common.random_state),
            tree,
        })
    }
}

fn grower_config(params: &DecisionTreeRegressorParams) -> GrowerConfig {
    GrowerConfig {
        max_depth: params.max_depth(),
        min_samples_split: params.min_samples_split(),
        min_samples_leaf: params.min_samples_leaf(),
        max_features: params.max_features(),
        splitter: params.splitter(),
    }
}

impl Estimator for DecisionTreeRegressor {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl Regressor for DecisionTreeRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        DecisionTreeRegressor::predict_into(self, data, output)
    }
}

impl HasParams for DecisionTreeRegressor {
    type Params = DecisionTreeRegressorParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

/// Weighted fitting and persistence.
///
/// A sample weight is a fractional row count here, so it multiplies every
/// impurity, split-bound test and leaf mean — the same rule the forests use,
/// which is what makes an integer weight identical to repeating the row. The
/// classifier sibling declares `multiclass` and `probability` on top of these
/// two; both name a class set, and a regression leaf holds a mean.
impl HasCapabilities for DecisionTreeRegressor {
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_sample_weights(true)
        .with_artifact(true);
}
