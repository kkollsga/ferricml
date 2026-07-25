use crate::artifact::{
    ANY_CLASSIFIER_ARTIFACT_KIND, ArtifactError, ArtifactPayloadWriter, SchemaRole,
    decode_component, decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::MatrixView;
use crate::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams, RandomForestClassifier,
    RandomForestClassifierParams,
};
use crate::linear_model::{LogisticRegression, LogisticRegressionParams};

use super::super::{Capabilities, Classifier, Estimator, HasCapabilities, ModelError};

const ANY_CLASSIFIER_PAYLOAD_VERSION: u16 = 1;
const DISPATCH_COMPONENT_KIND: u16 = 1;
const MODEL_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;
const DISPATCH_VERSION: u32 = 1;
const DISPATCH_METADATA_BYTES: usize = 2 * 4;

const VARIANT_RANDOM_FOREST: u32 = 1;
const VARIANT_LOGISTIC_REGRESSION: u32 = 2;
const VARIANT_HIST_GRADIENT_BOOSTING: u32 = 3;

/// Parameters retained by a fitted [`AnyClassifier`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyClassifierParams<'a> {
    /// Random-forest classifier parameters.
    RandomForest(&'a RandomForestClassifierParams),
    /// Logistic-regression classifier parameters.
    LogisticRegression(&'a LogisticRegressionParams),
    /// Histogram gradient-boosted classifier parameters.
    HistGradientBoosting(&'a HistGradientBoostingClassifierParams),
}

/// An owned fitted classifier selected at runtime.
///
/// Matching happens once for each batch call; traversal remains statically
/// dispatched inside the concrete model.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyClassifier {
    /// A fitted random-forest classifier.
    RandomForest(RandomForestClassifier),
    /// A fitted logistic-regression classifier.
    LogisticRegression(LogisticRegression),
    /// A fitted histogram gradient-boosted classifier.
    HistGradientBoosting(HistGradientBoostingClassifier),
}

impl AnyClassifier {
    /// Returns the feature width required by this model.
    pub fn n_features_in(&self) -> usize {
        <Self as Estimator>::n_features_in(self)
    }

    /// Returns sorted class labels observed during fitting.
    pub fn classes(&self) -> &[u8] {
        <Self as Classifier>::classes(self)
    }

    /// Returns the capabilities of the estimator type this value holds.
    ///
    /// [`HasCapabilities::CAPABILITIES`] is the intersection over every
    /// variant, which is what batch dispatch can rely on without inspecting
    /// the value. This reports the selected variant instead, which is what a
    /// caller needs before deciding whether this particular fitted model can
    /// be refitted with weights or persisted.
    pub fn capabilities(&self) -> Capabilities {
        match self {
            Self::RandomForest(_) => RandomForestClassifier::CAPABILITIES,
            Self::LogisticRegression(_) => LogisticRegression::CAPABILITIES,
            Self::HistGradientBoosting(_) => HistGradientBoostingClassifier::CAPABILITIES,
        }
    }

    /// Returns the concrete fitted parameters without erasing their type.
    pub fn get_params(&self) -> AnyClassifierParams<'_> {
        match self {
            Self::RandomForest(model) => AnyClassifierParams::RandomForest(model.get_params()),
            Self::LogisticRegression(model) => {
                AnyClassifierParams::LogisticRegression(model.get_params())
            }
            Self::HistGradientBoosting(model) => {
                AnyClassifierParams::HistGradientBoosting(model.get_params())
            }
        }
    }

    /// Predicts one label per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Predicts one label per row without allocating.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        <Self as Classifier>::predict_into(self, data, output)
    }

    /// Predicts row-major probabilities, allocating the output.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities without allocating.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Classifier>::predict_proba_into(self, data, output)
    }

    /// Predicts one fitted-class probability column without allocating.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Classifier>::predict_class_proba_into(self, data, class, output)
    }

    /// Predicts one fitted-class probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as Classifier>::predict_class_proba(self, data, class)
    }

    /// Encodes the selected runtime variant and its complete model artifact.
    ///
    /// The dispatch envelope records only which variant was fitted; the model
    /// itself is the estimator's own schema-bound artifact, nested whole and
    /// length-delimited. Adding a variant therefore never changes an existing
    /// estimator's payload, and a variant that carries more than one payload
    /// schema of its own — as logistic regression does — keeps choosing between
    /// them itself.
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let (variant, model) = match self {
            Self::RandomForest(model) => (VARIANT_RANDOM_FOREST, model.to_artifact(schema)?),
            Self::LogisticRegression(model) => {
                (VARIANT_LOGISTIC_REGRESSION, model.to_artifact(schema)?)
            }
            Self::HistGradientBoosting(model) => {
                (VARIANT_HIST_GRADIENT_BOOSTING, model.to_artifact(schema)?)
            }
        };
        let mut metadata = ArtifactPayloadWriter::with_capacity(DISPATCH_METADATA_BYTES);
        metadata.u32(DISPATCH_VERSION);
        metadata.u32(variant);
        let mut payload = encode_component(
            DISPATCH_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        payload.extend_from_slice(&encode_component(
            MODEL_COMPONENT_KIND,
            COMPONENT_VERSION,
            &model,
        )?);
        encode_v2_envelope(
            ANY_CLASSIFIER_ARTIFACT_KIND,
            ANY_CLASSIFIER_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Restores the fitted runtime variant recorded by [`Self::to_artifact`].
    ///
    /// The nested model is decoded by its own estimator, so it is checksummed,
    /// schema-bound, and validated exactly as it would be on its own. A variant
    /// tag that disagrees with the nested payload is rejected by that
    /// estimator's kind check rather than silently reinterpreted.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            ANY_CLASSIFIER_ARTIFACT_KIND,
            ANY_CLASSIFIER_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, DISPATCH_COMPONENT_KIND, COMPONENT_VERSION)?;
        let dispatch_version = metadata.u32()?;
        let variant = metadata.u32()?;
        if !metadata.is_empty() || dispatch_version != DISPATCH_VERSION {
            return Err(ArtifactError::InvalidPayload);
        }
        let model =
            decode_component(&mut envelope, MODEL_COMPONENT_KIND, COMPONENT_VERSION)?.remaining();
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(match variant {
            VARIANT_RANDOM_FOREST => {
                Self::RandomForest(RandomForestClassifier::from_artifact(model, schema)?)
            }
            VARIANT_LOGISTIC_REGRESSION => {
                Self::LogisticRegression(LogisticRegression::from_artifact(model, schema)?)
            }
            VARIANT_HIST_GRADIENT_BOOSTING => Self::HistGradientBoosting(
                HistGradientBoostingClassifier::from_artifact(model, schema)?,
            ),
            _ => return Err(ArtifactError::InvalidPayload),
        })
    }
}

impl From<RandomForestClassifier> for AnyClassifier {
    fn from(model: RandomForestClassifier) -> Self {
        Self::RandomForest(model)
    }
}

impl From<LogisticRegression> for AnyClassifier {
    fn from(model: LogisticRegression) -> Self {
        Self::LogisticRegression(model)
    }
}

impl From<HistGradientBoostingClassifier> for AnyClassifier {
    fn from(model: HistGradientBoostingClassifier) -> Self {
        Self::HistGradientBoosting(model)
    }
}

impl Estimator for AnyClassifier {
    fn n_features_in(&self) -> usize {
        match self {
            Self::RandomForest(model) => model.n_features_in(),
            Self::LogisticRegression(model) => model.n_features_in(),
            Self::HistGradientBoosting(model) => model.n_features_in(),
        }
    }
}

/// Declares only what holds for every variant, so a caller that has not
/// inspected the runtime variant is never promised more than it gets.
///
/// Weighted and multiclass fitting are both declared away structurally rather
/// than composed: the enum owns fitted models and no fitting entry point, so it
/// could accept neither weights nor a class set even though every variant can.
/// An intersection would have declared multiclass fitting the enum does not
/// offer. It still *holds* and serves a multiclass model — `classes()` and
/// `predict_proba` are already shaped by the fitted model — which is a property
/// of the value, not a capability of this type.
/// Persistence, by contrast, is composed rather than declared away: every
/// variant persists every fit it offers, and this enum has an artifact entry
/// point, so the intersection is the truth.
impl HasCapabilities for AnyClassifier {
    const CAPABILITIES: Capabilities = RandomForestClassifier::CAPABILITIES
        .intersection(LogisticRegression::CAPABILITIES)
        .intersection(HistGradientBoostingClassifier::CAPABILITIES)
        .with_sample_weights(false)
        .with_multiclass(false);
}

impl Classifier for AnyClassifier {
    fn classes(&self) -> &[u8] {
        match self {
            Self::RandomForest(model) => model.classes(),
            Self::LogisticRegression(model) => model.classes(),
            Self::HistGradientBoosting(model) => model.classes(),
        }
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_into(data, output),
            Self::LogisticRegression(model) => model.predict_into(data, output),
            Self::HistGradientBoosting(model) => model.predict_into(data, output),
        }
    }

    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_proba_into(data, output),
            Self::LogisticRegression(model) => model.predict_proba_into(data, output),
            Self::HistGradientBoosting(model) => model.predict_proba_into(data, output),
        }
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_class_proba_into(data, class, output),
            Self::LogisticRegression(model) => model.predict_class_proba_into(data, class, output),
            Self::HistGradientBoosting(model) => {
                model.predict_class_proba_into(data, class, output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::ArtifactError;
    use crate::data::{BinaryTargets, ClassTargets, DenseMatrix};
    use crate::ensemble::HistGradientBoostingClassifierParams;
    use crate::ensemble::MaxFeatures;
    use sha2::{Digest, Sha256};

    const PAYLOAD_START: usize = 24 + 36;
    const DISPATCH_START: usize = PAYLOAD_START + 8;

    fn resign(bytes: &mut [u8]) {
        let checksum_offset = bytes.len() - 32;
        let checksum = Sha256::digest(&bytes[..checksum_offset]);
        bytes[checksum_offset..].copy_from_slice(&checksum);
    }

    fn fixture() -> (DenseMatrix, BinaryTargets, ClassTargets) {
        (
            DenseMatrix::new(vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 8.0], 4, 2).unwrap(),
            BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
            ClassTargets::new(vec![3, 7, 10, 7]).unwrap(),
        )
    }

    fn forest_params() -> RandomForestClassifierParams {
        RandomForestClassifierParams::default()
            .with_n_estimators(3)
            .with_max_features(MaxFeatures::All)
            .with_random_state(3)
    }

    /// Every runtime variant, including both fits of each variant that offers
    /// two. The dispatch envelope must restore the variant *and* the payload
    /// schema the variant chose for itself.
    fn variants(
        data: &DenseMatrix,
        binary: &BinaryTargets,
        classes: &ClassTargets,
    ) -> Vec<AnyClassifier> {
        vec![
            RandomForestClassifier::fit(&data.as_view(), binary, forest_params())
                .unwrap()
                .into(),
            RandomForestClassifier::fit_multiclass(&data.as_view(), classes, forest_params())
                .unwrap()
                .into(),
            LogisticRegression::fit(&data.as_view(), binary, LogisticRegressionParams::default())
                .unwrap()
                .into(),
            LogisticRegression::fit_multiclass(
                &data.as_view(),
                classes,
                LogisticRegressionParams::default(),
            )
            .unwrap()
            .into(),
            HistGradientBoostingClassifier::fit(
                &data.as_view(),
                binary,
                HistGradientBoostingClassifierParams::default()
                    .with_max_iter(3)
                    .with_max_leaf_nodes(2)
                    .with_min_samples_leaf(1)
                    .with_max_bins(4),
            )
            .unwrap()
            .into(),
        ]
    }

    #[test]
    fn dispatch_artifact_round_trips_every_runtime_variant() {
        let (data, binary, classes) = fixture();
        let schema = [13; 32];
        for model in variants(&data, &binary, &classes) {
            let bytes = model.to_artifact(schema).unwrap();
            assert_eq!(bytes, model.to_artifact(schema).unwrap());
            let decoded = AnyClassifier::from_artifact(&bytes, schema).unwrap();
            assert_eq!(decoded, model);
            assert_eq!(
                std::mem::discriminant(&decoded),
                std::mem::discriminant(&model)
            );
            assert_eq!(decoded.n_features_in(), model.n_features_in());
            assert_eq!(decoded.classes(), model.classes());
            assert_eq!(decoded.get_params(), model.get_params());
            assert_eq!(decoded.capabilities(), model.capabilities());
            assert_eq!(
                decoded.predict(&data.as_view()).unwrap(),
                model.predict(&data.as_view()).unwrap()
            );
            assert_eq!(
                decoded.predict_proba(&data.as_view()).unwrap(),
                model.predict_proba(&data.as_view()).unwrap()
            );
            assert_eq!(decoded.to_artifact(schema).unwrap(), bytes);
        }
    }

    #[test]
    fn dispatch_artifact_is_schema_bound_and_kind_isolated() {
        let (data, binary, _) = fixture();
        let schema = [21; 32];
        let forest =
            RandomForestClassifier::fit(&data.as_view(), &binary, forest_params()).unwrap();
        let erased: AnyClassifier = forest.clone().into();
        let bytes = erased.to_artifact(schema).unwrap();

        assert_eq!(
            AnyClassifier::from_artifact(&bytes, [22; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        assert_eq!(
            RandomForestClassifier::from_artifact(&bytes, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind {
                found: ANY_CLASSIFIER_ARTIFACT_KIND,
            }
        );
        assert_eq!(
            AnyClassifier::from_artifact(&forest.to_artifact(schema).unwrap(), schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 11 }
        );

        let mut corrupted = bytes;
        corrupted[DISPATCH_START] ^= 1;
        assert_eq!(
            AnyClassifier::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }

    #[test]
    fn dispatch_artifact_rejects_unknown_versions_variants_and_framing() {
        let (data, binary, _) = fixture();
        let schema = [27; 32];
        let erased: AnyClassifier =
            RandomForestClassifier::fit(&data.as_view(), &binary, forest_params())
                .unwrap()
                .into();
        let bytes = erased.to_artifact(schema).unwrap();

        for (name, offset, value) in [
            ("dispatch version", 0, 2_u32),
            ("unknown variant", 4, 0),
            ("variant beyond the known set", 4, 5),
        ] {
            let mut corrupted = bytes.clone();
            corrupted[DISPATCH_START + offset..DISPATCH_START + offset + 4]
                .copy_from_slice(&value.to_le_bytes());
            resign(&mut corrupted);
            assert_eq!(
                AnyClassifier::from_artifact(&corrupted, schema).unwrap_err(),
                ArtifactError::InvalidPayload,
                "{name} was accepted"
            );
        }

        // A variant tag that disagrees with the nested payload is caught by
        // the nested estimator's own kind check, not silently reinterpreted.
        let mut mislabelled = bytes.clone();
        mislabelled[DISPATCH_START + 4..DISPATCH_START + 8]
            .copy_from_slice(&VARIANT_LOGISTIC_REGRESSION.to_le_bytes());
        resign(&mut mislabelled);
        assert_eq!(
            AnyClassifier::from_artifact(&mislabelled, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 11 }
        );

        let mut component_kind = bytes.clone();
        component_kind[PAYLOAD_START..PAYLOAD_START + 2].copy_from_slice(&9_u16.to_le_bytes());
        resign(&mut component_kind);
        assert_eq!(
            AnyClassifier::from_artifact(&component_kind, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut trailing = bytes.clone();
        trailing.insert(bytes.len() - 32, 0);
        resign(&mut trailing);
        assert_eq!(
            AnyClassifier::from_artifact(&trailing, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        assert_eq!(
            AnyClassifier::from_artifact(&bytes[..bytes.len() - 1], schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
        assert_eq!(
            AnyClassifier::from_artifact(&[], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }
}
