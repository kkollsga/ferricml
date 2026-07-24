//! Generic fitted preprocessing pipelines.
//!
//! [`Pipeline`] composes one fitted transformer with one fitted estimator.
//! [`StagedPipeline`] composes two or more transform stages, and can fit the
//! whole composition in one pass. Every part stays a concrete type, so
//! transformation and the callback passed to `with_transformed` are statically
//! dispatched and no stage is reached through a trait object.

mod artifact;
mod stack;
mod staged;

pub use artifact::{ModelArtifact, PersistedStack, StageArtifact};
pub use stack::TransformerStack;
pub use staged::StagedPipeline;

use crate::api::Classifier;
use crate::api::{
    Capabilities, Estimator, HasCapabilities, ModelError, Transformer, validate_transformed_shape,
};
use crate::artifact::{
    ArtifactError, STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND,
    STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND, STANDARD_SCALER_RIDGE_PIPELINE_ARTIFACT_KIND,
    SchemaRole, decode_component, decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::{DenseMatrix, MatrixView};
use crate::linear_model::{LinearRegression, LogisticRegression, Ridge};
use crate::preprocessing::StandardScaler;

const PIPELINE_PAYLOAD_VERSION: u16 = 1;
const TRANSFORMER_COMPONENT_KIND: u16 = 1;
const ESTIMATOR_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;

/// One fitted transformer followed by one fitted estimator.
///
/// Construction validates the feature-width handoff. Allocation-sensitive
/// callers reuse a workspace and call [`Pipeline::with_transformed`]; callers
/// that prefer convenience can use [`Pipeline::transform`].
#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline<T, E> {
    transformer: T,
    estimator: E,
}

impl<T, E> Pipeline<T, E>
where
    T: Transformer,
    E: Estimator,
{
    /// Composes fitted parts after validating their feature-width handoff.
    pub fn new(transformer: T, estimator: E) -> Result<Self, ModelError> {
        let transformed = transformer.n_features_out();
        let expected = estimator.n_features_in();
        if transformed != expected {
            return Err(ModelError::FeatureDimension {
                expected,
                actual: transformed,
            });
        }
        Ok(Self {
            transformer,
            estimator,
        })
    }

    /// Returns the fitted transformer.
    pub const fn transformer(&self) -> &T {
        &self.transformer
    }

    /// Returns the fitted final estimator.
    pub const fn estimator(&self) -> &E {
        &self.estimator
    }

    /// Consumes the pipeline and returns its fitted parts.
    pub fn into_parts(self) -> (T, E) {
        (self.transformer, self.estimator)
    }

    /// Number of `f32` values required for a transformed batch workspace.
    pub fn workspace_len(&self, rows: usize) -> Result<usize, ModelError> {
        rows.checked_mul(self.transformer.n_features_out())
            .ok_or(ModelError::OutputShapeOverflow {
                rows,
                columns: self.transformer.n_features_out(),
            })
    }

    /// Transforms into caller-owned workspace and returns its validated view.
    pub fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        workspace: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        let transformed = self.transformer.transform_into(data, workspace)?;
        validate_transformed_shape(data.rows(), self.transformer.n_features_out(), &transformed)?;
        Ok(transformed)
    }

    /// Transforms into a newly allocated dense matrix.
    pub fn transform(&self, data: &MatrixView<'_>) -> Result<DenseMatrix, ModelError> {
        self.transformer.transform(data)
    }

    /// Runs an operation on a transformed batch without allocating or erasing
    /// either fitted type.
    ///
    /// This is the extension point for future classifier/regressor convenience
    /// methods: the callback can call an estimator's `_into` method while the
    /// caller reuses `workspace` across batches.
    pub fn with_transformed<R>(
        &self,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        operation: impl FnOnce(&E, &MatrixView<'_>) -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        let transformed = self.transform_into(data, workspace)?;
        operation(&self.estimator, &transformed)
    }
}

impl<T, E> Estimator for Pipeline<T, E>
where
    T: Transformer,
    E: Estimator,
{
    fn n_features_in(&self) -> usize {
        self.transformer.n_features_in()
    }
}

fn encode_pipeline_artifact(
    kind: u16,
    input_schema: [u8; 32],
    transformed_schema: [u8; 32],
    transformer_artifact: &[u8],
    estimator_artifact: &[u8],
) -> Result<Vec<u8>, ArtifactError> {
    let transformer = encode_component(
        TRANSFORMER_COMPONENT_KIND,
        COMPONENT_VERSION,
        transformer_artifact,
    )?;
    let estimator = encode_component(
        ESTIMATOR_COMPONENT_KIND,
        COMPONENT_VERSION,
        estimator_artifact,
    )?;
    let mut payload = Vec::with_capacity(transformer.len() + estimator.len());
    payload.extend_from_slice(&transformer);
    payload.extend_from_slice(&estimator);
    encode_v2_envelope(
        kind,
        PIPELINE_PAYLOAD_VERSION,
        &[
            (SchemaRole::Input, input_schema),
            (SchemaRole::Transformed, transformed_schema),
        ],
        &payload,
    )
}

fn decode_pipeline_components(
    bytes: &[u8],
    kind: u16,
    input_schema: [u8; 32],
    transformed_schema: [u8; 32],
) -> Result<(&[u8], &[u8]), ArtifactError> {
    let mut envelope = decode_v2_envelope(
        bytes,
        kind,
        PIPELINE_PAYLOAD_VERSION,
        &[
            (SchemaRole::Input, input_schema),
            (SchemaRole::Transformed, transformed_schema),
        ],
    )?;
    let transformer =
        decode_component(&mut envelope, TRANSFORMER_COMPONENT_KIND, COMPONENT_VERSION)?;
    let estimator = decode_component(&mut envelope, ESTIMATOR_COMPONENT_KIND, COMPONENT_VERSION)?;
    if !envelope.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok((transformer.remaining(), estimator.remaining()))
}

/// A fitted composition persists exactly when both of its parts do.
///
/// Weighted fitting is declared away structurally: [`Pipeline`] composes parts
/// that are already fitted, so accepting weights is a property of fitting each
/// part, not of the composition. Only compositions with a concrete artifact
/// declare capabilities at all; asking an arbitrary `Pipeline<T, E>` is a
/// compile error rather than a wrong answer.
impl HasCapabilities for Pipeline<StandardScaler, LogisticRegression> {
    const CAPABILITIES: Capabilities = StandardScaler::CAPABILITIES
        .intersection(LogisticRegression::CAPABILITIES)
        .with_sample_weights(false);
}

impl Pipeline<StandardScaler, LogisticRegression> {
    /// Writes labels using caller-owned transform and output buffers.
    pub fn predict_into(
        &self,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError> {
        self.with_transformed(data, workspace, |model, transformed| {
            model.predict_into(transformed, output)
        })
    }

    /// Writes probabilities using caller-owned transform and output buffers.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.with_transformed(data, workspace, |model, transformed| {
            model.predict_proba_into(transformed, output)
        })
    }

    /// Writes one requested probability column without allocating.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.with_transformed(data, workspace, |model, transformed| {
            Classifier::predict_class_proba_into(model, transformed, class, output)
        })
    }

    /// Writes raw decision scores without allocating.
    pub fn decision_function_into(
        &self,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.with_transformed(data, workspace, |model, transformed| {
            model.decision_function_into(transformed, output)
        })
    }

    /// Encodes this concrete fitted pipeline and both schema identities.
    pub fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        encode_pipeline_artifact(
            STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            &self
                .transformer
                .to_artifact(input_schema, transformed_schema)?,
            &self.estimator.to_artifact(transformed_schema)?,
        )
    }

    /// Decodes this concrete pipeline and validates the fitted width handoff.
    pub fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let (transformer, estimator) = decode_pipeline_components(
            bytes,
            STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND,
            input_schema,
            transformed_schema,
        )?;
        let transformer =
            StandardScaler::from_artifact(transformer, input_schema, transformed_schema)?;
        let estimator = LogisticRegression::from_artifact(estimator, transformed_schema)?;
        Self::new(transformer, estimator).map_err(|_| ArtifactError::InvalidPayload)
    }
}

macro_rules! impl_scaler_regression_pipeline {
    ($estimator:ty, $kind:expr) => {
        impl HasCapabilities for Pipeline<StandardScaler, $estimator> {
            const CAPABILITIES: Capabilities = StandardScaler::CAPABILITIES
                .intersection(<$estimator>::CAPABILITIES)
                .with_sample_weights(false);
        }

        impl Pipeline<StandardScaler, $estimator> {
            /// Writes predictions using caller-owned transform and output buffers.
            pub fn predict_into(
                &self,
                data: &MatrixView<'_>,
                workspace: &mut [f32],
                output: &mut [f32],
            ) -> Result<(), ModelError> {
                self.with_transformed(data, workspace, |model, transformed| {
                    model.predict_into(transformed, output)
                })
            }

            /// Encodes this concrete fitted pipeline and both schema identities.
            pub fn to_artifact(
                &self,
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Vec<u8>, ArtifactError> {
                encode_pipeline_artifact(
                    $kind,
                    input_schema,
                    transformed_schema,
                    &self
                        .transformer
                        .to_artifact(input_schema, transformed_schema)?,
                    &self.estimator.to_artifact(transformed_schema)?,
                )
            }

            /// Decodes this concrete pipeline and validates the fitted width handoff.
            pub fn from_artifact(
                bytes: &[u8],
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Self, ArtifactError> {
                let (transformer, estimator) =
                    decode_pipeline_components(bytes, $kind, input_schema, transformed_schema)?;
                let transformer =
                    StandardScaler::from_artifact(transformer, input_schema, transformed_schema)?;
                let estimator = <$estimator>::from_artifact(estimator, transformed_schema)?;
                Self::new(transformer, estimator).map_err(|_| ArtifactError::InvalidPayload)
            }
        }
    };
}

impl_scaler_regression_pipeline!(
    LinearRegression,
    STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND
);
impl_scaler_regression_pipeline!(Ridge, STANDARD_SCALER_RIDGE_PIPELINE_ARTIFACT_KIND);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinaryTargets, RegressionTargets};
    use crate::linear_model::{LinearRegressionParams, LogisticRegressionParams, RidgeParams};
    use crate::preprocessing::StandardScalerParams;
    use sha2::{Digest, Sha256};

    fn data() -> DenseMatrix {
        DenseMatrix::new(vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 8.0], 4, 2).unwrap()
    }

    fn scaler() -> StandardScaler {
        StandardScaler::fit(&data().as_view(), StandardScalerParams::default()).unwrap()
    }

    #[test]
    fn logistic_pipeline_round_trips_and_reuses_workspace() {
        let transformed = scaler().transform(&data().as_view()).unwrap();
        let model = LogisticRegression::fit(
            &transformed.as_view(),
            &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let pipeline = Pipeline::new(scaler(), model).unwrap();
        let mut workspace = vec![0.0; pipeline.workspace_len(4).unwrap()];
        let mut expected = [0; 4];
        pipeline
            .predict_into(&data().as_view(), &mut workspace, &mut expected)
            .unwrap();
        let bytes = pipeline.to_artifact([1; 32], [2; 32]).unwrap();
        let decoded =
            Pipeline::<StandardScaler, LogisticRegression>::from_artifact(&bytes, [1; 32], [2; 32])
                .unwrap();
        let mut actual = [0; 4];
        decoded
            .predict_into(&data().as_view(), &mut workspace, &mut actual)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(bytes, pipeline.to_artifact([1; 32], [2; 32]).unwrap());
    }

    #[test]
    fn regression_pipeline_artifacts_keep_estimator_identity() {
        let transformed = scaler().transform(&data().as_view()).unwrap();
        let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
        let linear = Pipeline::new(
            scaler(),
            LinearRegression::fit(
                &transformed.as_view(),
                &targets,
                LinearRegressionParams::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let bytes = linear.to_artifact([3; 32], [4; 32]).unwrap();
        assert_eq!(
            Pipeline::<StandardScaler, Ridge>::from_artifact(&bytes, [3; 32], [4; 32]).unwrap_err(),
            ArtifactError::UnsupportedModelKind {
                found: STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND
            }
        );

        let ridge = Pipeline::new(
            scaler(),
            Ridge::fit(&transformed.as_view(), &targets, RidgeParams::default()).unwrap(),
        )
        .unwrap();
        let encoded = ridge.to_artifact([3; 32], [4; 32]).unwrap();
        let decoded =
            Pipeline::<StandardScaler, Ridge>::from_artifact(&encoded, [3; 32], [4; 32]).unwrap();
        let mut workspace = vec![0.0; decoded.workspace_len(4).unwrap()];
        let mut output = [0.0; 4];
        decoded
            .predict_into(&data().as_view(), &mut workspace, &mut output)
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pipeline_artifact_rejects_corruption_bad_components_and_width_handoff() {
        let transformed = scaler().transform(&data().as_view()).unwrap();
        let model = LogisticRegression::fit(
            &transformed.as_view(),
            &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let pipeline = Pipeline::new(scaler(), model).unwrap();
        let bytes = pipeline.to_artifact([1; 32], [2; 32]).unwrap();

        let mut corrupted = bytes.clone();
        corrupted[110] ^= 1;
        assert_eq!(
            Pipeline::<StandardScaler, LogisticRegression>::from_artifact(
                &corrupted, [1; 32], [2; 32]
            )
            .unwrap_err(),
            ArtifactError::ChecksumMismatch
        );

        let mut bad_component = bytes.clone();
        bad_component[96..98].copy_from_slice(&ESTIMATOR_COMPONENT_KIND.to_le_bytes());
        let checksum_start = bad_component.len() - 32;
        let checksum = Sha256::digest(&bad_component[..checksum_start]);
        bad_component[checksum_start..].copy_from_slice(&checksum);
        assert_eq!(
            Pipeline::<StandardScaler, LogisticRegression>::from_artifact(
                &bad_component,
                [1; 32],
                [2; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut bad_length = bytes.clone();
        let declared = u32::from_le_bytes(bad_length[100..104].try_into().unwrap());
        bad_length[100..104].copy_from_slice(&(declared + 1).to_le_bytes());
        let checksum_start = bad_length.len() - 32;
        let checksum = Sha256::digest(&bad_length[..checksum_start]);
        bad_length[checksum_start..].copy_from_slice(&checksum);
        assert_eq!(
            Pipeline::<StandardScaler, LogisticRegression>::from_artifact(
                &bad_length,
                [1; 32],
                [2; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let one_column = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
        let narrow_model = LogisticRegression::fit(
            &one_column.as_view(),
            &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
            LogisticRegressionParams::default(),
        )
        .unwrap();
        let mismatched = encode_pipeline_artifact(
            STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND,
            [1; 32],
            [2; 32],
            &scaler().to_artifact([1; 32], [2; 32]).unwrap(),
            &narrow_model.to_artifact([2; 32]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            Pipeline::<StandardScaler, LogisticRegression>::from_artifact(
                &mismatched,
                [1; 32],
                [2; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }
}
