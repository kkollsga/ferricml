use crate::artifact::{
    ANY_REGRESSOR_ARTIFACT_KIND, ArtifactError, ArtifactPayloadWriter, SchemaRole,
    decode_component, decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::MatrixView;
use crate::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use crate::linear_model::{LinearRegression, LinearRegressionParams, Ridge, RidgeParams};

use super::super::{Capabilities, Estimator, HasCapabilities, ModelError, Regressor};

const ANY_REGRESSOR_PAYLOAD_VERSION: u16 = 1;
const DISPATCH_COMPONENT_KIND: u16 = 1;
const MODEL_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;
const DISPATCH_VERSION: u32 = 1;
const DISPATCH_METADATA_BYTES: usize = 2 * 4;

const VARIANT_RANDOM_FOREST: u32 = 1;
const VARIANT_LINEAR_REGRESSION: u32 = 2;
const VARIANT_RIDGE: u32 = 3;
const VARIANT_HIST_GRADIENT_BOOSTING: u32 = 4;

/// Parameters retained by a fitted [`AnyRegressor`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyRegressorParams<'a> {
    /// Random-forest regressor parameters.
    RandomForest(&'a RandomForestRegressorParams),
    /// Ordinary least-squares regressor parameters.
    LinearRegression(&'a LinearRegressionParams),
    /// Ridge-regression parameters.
    Ridge(&'a RidgeParams),
    /// Histogram gradient-boosting parameters.
    HistGradientBoosting(&'a HistGradientBoostingRegressorParams),
}

/// An owned fitted regressor selected at runtime.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnyRegressor {
    /// A fitted random-forest regressor.
    RandomForest(RandomForestRegressor),
    /// A fitted ordinary least-squares regressor.
    LinearRegression(LinearRegression),
    /// A fitted ridge regressor.
    Ridge(Ridge),
    /// A fitted histogram gradient-boosted regressor.
    HistGradientBoosting(HistGradientBoostingRegressor),
}

impl AnyRegressor {
    /// Returns the feature width required by this model.
    pub fn n_features_in(&self) -> usize {
        <Self as Estimator>::n_features_in(self)
    }

    /// Returns the capabilities of the estimator type this value holds.
    ///
    /// [`HasCapabilities::CAPABILITIES`] is the intersection over every
    /// variant, which is what batch dispatch can rely on without inspecting
    /// the value. This reports the selected variant instead, which is what a
    /// caller needs before deciding whether this particular fitted model can
    /// be refitted with weights.
    pub fn capabilities(&self) -> Capabilities {
        match self {
            Self::RandomForest(_) => RandomForestRegressor::CAPABILITIES,
            Self::LinearRegression(_) => LinearRegression::CAPABILITIES,
            Self::Ridge(_) => Ridge::CAPABILITIES,
            Self::HistGradientBoosting(_) => HistGradientBoostingRegressor::CAPABILITIES,
        }
    }

    /// Returns the concrete fitted parameters without erasing their type.
    pub fn get_params(&self) -> AnyRegressorParams<'_> {
        match self {
            Self::RandomForest(model) => AnyRegressorParams::RandomForest(model.get_params()),
            Self::LinearRegression(model) => {
                AnyRegressorParams::LinearRegression(model.get_params())
            }
            Self::Ridge(model) => AnyRegressorParams::Ridge(model.get_params()),
            Self::HistGradientBoosting(model) => {
                AnyRegressorParams::HistGradientBoosting(model.get_params())
            }
        }
    }

    /// Predicts one value per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as Regressor>::predict(self, data)
    }

    /// Predicts one value per row without allocating.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as Regressor>::predict_into(self, data, output)
    }

    /// Encodes the selected runtime variant and its complete model artifact.
    ///
    /// The dispatch envelope records only which variant was fitted; the model
    /// itself is the estimator's own schema-bound artifact, nested whole and
    /// length-delimited. Adding a variant therefore never changes an existing
    /// estimator's payload.
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        let (variant, model) = match self {
            Self::RandomForest(model) => (VARIANT_RANDOM_FOREST, model.to_artifact(schema)?),
            Self::LinearRegression(model) => {
                (VARIANT_LINEAR_REGRESSION, model.to_artifact(schema)?)
            }
            Self::Ridge(model) => (VARIANT_RIDGE, model.to_artifact(schema)?),
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
            ANY_REGRESSOR_ARTIFACT_KIND,
            ANY_REGRESSOR_PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Restores the fitted runtime variant recorded by [`Self::to_artifact`].
    ///
    /// The nested model is decoded by its own estimator, so it is checksummed,
    /// schema-bound, and validated exactly as it would be on its own. A
    /// variant tag that disagrees with the nested payload is rejected by that
    /// estimator's kind check rather than silently reinterpreted.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            ANY_REGRESSOR_ARTIFACT_KIND,
            ANY_REGRESSOR_PAYLOAD_VERSION,
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
                Self::RandomForest(RandomForestRegressor::from_artifact(model, schema)?)
            }
            VARIANT_LINEAR_REGRESSION => {
                Self::LinearRegression(LinearRegression::from_artifact(model, schema)?)
            }
            VARIANT_RIDGE => Self::Ridge(Ridge::from_artifact(model, schema)?),
            VARIANT_HIST_GRADIENT_BOOSTING => Self::HistGradientBoosting(
                HistGradientBoostingRegressor::from_artifact(model, schema)?,
            ),
            _ => return Err(ArtifactError::InvalidPayload),
        })
    }
}

impl From<RandomForestRegressor> for AnyRegressor {
    fn from(model: RandomForestRegressor) -> Self {
        Self::RandomForest(model)
    }
}

impl From<LinearRegression> for AnyRegressor {
    fn from(model: LinearRegression) -> Self {
        Self::LinearRegression(model)
    }
}

impl From<Ridge> for AnyRegressor {
    fn from(model: Ridge) -> Self {
        Self::Ridge(model)
    }
}

impl From<HistGradientBoostingRegressor> for AnyRegressor {
    fn from(model: HistGradientBoostingRegressor) -> Self {
        Self::HistGradientBoosting(model)
    }
}

impl Estimator for AnyRegressor {
    fn n_features_in(&self) -> usize {
        match self {
            Self::RandomForest(model) => model.n_features_in(),
            Self::LinearRegression(model) => model.n_features_in(),
            Self::Ridge(model) => model.n_features_in(),
            Self::HistGradientBoosting(model) => model.n_features_in(),
        }
    }
}

/// Declares only what holds for every variant, so a caller that has not
/// inspected the runtime variant is never promised more than it gets. Every
/// variant persists, so the dispatch enum persists too.
///
/// Weighted fitting is declared away structurally rather than composed: the
/// enum owns fitted models and no fitting entry point, so it could not accept
/// weights even if every variant did.
impl HasCapabilities for AnyRegressor {
    const CAPABILITIES: Capabilities = RandomForestRegressor::CAPABILITIES
        .intersection(LinearRegression::CAPABILITIES)
        .intersection(Ridge::CAPABILITIES)
        .intersection(HistGradientBoostingRegressor::CAPABILITIES)
        .with_sample_weights(false);
}

impl Regressor for AnyRegressor {
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError> {
        match self {
            Self::RandomForest(model) => model.predict_into(data, output),
            Self::LinearRegression(model) => model.predict_into(data, output),
            Self::Ridge(model) => model.predict_into(data, output),
            Self::HistGradientBoosting(model) => model.predict_into(data, output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DenseMatrix, RegressionTargets};
    use crate::ensemble::MaxFeatures;
    use sha2::{Digest, Sha256};

    const PAYLOAD_START: usize = 24 + 36;
    const DISPATCH_START: usize = PAYLOAD_START + 8;

    fn resign(bytes: &mut [u8]) {
        let checksum_offset = bytes.len() - 32;
        let checksum = Sha256::digest(&bytes[..checksum_offset]);
        bytes[checksum_offset..].copy_from_slice(&checksum);
    }

    fn fixture() -> (DenseMatrix, RegressionTargets) {
        (
            DenseMatrix::new(vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 8.0], 4, 2).unwrap(),
            RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap(),
        )
    }

    fn variants(data: &DenseMatrix, targets: &RegressionTargets) -> Vec<AnyRegressor> {
        vec![
            RandomForestRegressor::fit(
                &data.as_view(),
                targets,
                RandomForestRegressorParams::default()
                    .with_n_estimators(3)
                    .with_max_features(MaxFeatures::All)
                    .with_random_state(3),
            )
            .unwrap()
            .into(),
            LinearRegression::fit(&data.as_view(), targets, LinearRegressionParams::default())
                .unwrap()
                .into(),
            Ridge::fit(&data.as_view(), targets, RidgeParams::default())
                .unwrap()
                .into(),
            HistGradientBoostingRegressor::fit(
                &data.as_view(),
                targets,
                HistGradientBoostingRegressorParams::default()
                    .with_max_iter(2)
                    .with_max_leaf_nodes(2)
                    .with_min_samples_leaf(1),
            )
            .unwrap()
            .into(),
        ]
    }

    #[test]
    fn dispatch_artifact_round_trips_every_runtime_variant() {
        let (data, targets) = fixture();
        let schema = [13; 32];
        for model in variants(&data, &targets) {
            let bytes = model.to_artifact(schema).unwrap();
            assert_eq!(bytes, model.to_artifact(schema).unwrap());
            let decoded = AnyRegressor::from_artifact(&bytes, schema).unwrap();
            assert_eq!(decoded, model);
            assert_eq!(
                std::mem::discriminant(&decoded),
                std::mem::discriminant(&model)
            );
            assert_eq!(decoded.n_features_in(), model.n_features_in());
            assert_eq!(decoded.get_params(), model.get_params());
            assert_eq!(
                decoded.predict(&data.as_view()).unwrap(),
                model.predict(&data.as_view()).unwrap()
            );
            assert_eq!(decoded.to_artifact(schema).unwrap(), bytes);
        }
    }

    #[test]
    fn dispatch_artifact_is_schema_bound_and_kind_isolated() {
        let (data, targets) = fixture();
        let schema = [21; 32];
        let ridge = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
        let erased: AnyRegressor = ridge.clone().into();
        let bytes = erased.to_artifact(schema).unwrap();

        assert_eq!(
            AnyRegressor::from_artifact(&bytes, [22; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        assert_eq!(
            Ridge::from_artifact(&bytes, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind {
                found: ANY_REGRESSOR_ARTIFACT_KIND,
            }
        );
        assert_eq!(
            AnyRegressor::from_artifact(&ridge.to_artifact(schema).unwrap(), schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 3 }
        );

        let mut corrupted = bytes;
        corrupted[DISPATCH_START] ^= 1;
        assert_eq!(
            AnyRegressor::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }

    #[test]
    fn dispatch_artifact_rejects_unknown_versions_variants_and_framing() {
        let (data, targets) = fixture();
        let schema = [27; 32];
        let erased: AnyRegressor = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())
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
                AnyRegressor::from_artifact(&corrupted, schema).unwrap_err(),
                ArtifactError::InvalidPayload,
                "{name} was accepted"
            );
        }

        // A variant tag that disagrees with the nested payload is caught by
        // the nested estimator's own kind check, not silently reinterpreted.
        let mut mislabelled = bytes.clone();
        mislabelled[DISPATCH_START + 4..DISPATCH_START + 8]
            .copy_from_slice(&VARIANT_LINEAR_REGRESSION.to_le_bytes());
        resign(&mut mislabelled);
        assert_eq!(
            AnyRegressor::from_artifact(&mislabelled, schema).unwrap_err(),
            ArtifactError::UnsupportedModelKind { found: 3 }
        );

        let mut component_kind = bytes.clone();
        component_kind[PAYLOAD_START..PAYLOAD_START + 2].copy_from_slice(&9_u16.to_le_bytes());
        resign(&mut component_kind);
        assert_eq!(
            AnyRegressor::from_artifact(&component_kind, schema).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut trailing = bytes.clone();
        trailing.insert(bytes.len() - 32, 0);
        resign(&mut trailing);
        assert_eq!(
            AnyRegressor::from_artifact(&trailing, schema).unwrap_err(),
            ArtifactError::TrailingBytes
        );

        assert_eq!(
            AnyRegressor::from_artifact(&bytes[..bytes.len() - 1], schema).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
        assert_eq!(
            AnyRegressor::from_artifact(&[], schema).unwrap_err(),
            ArtifactError::Truncated
        );
    }
}
