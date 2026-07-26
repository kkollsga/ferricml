//! Generic fitted preprocessing pipelines.
//!
//! [`Pipeline`] composes one fitted transformer with one fitted estimator, and
//! can fit both in one pass. [`StagedPipeline`] composes a
//! [`TransformerStack`] of one to [`MAX_STAGES`] transform stages with an
//! estimator. Every part stays a concrete type, so transformation and the
//! callback passed to `with_transformed` are statically dispatched and no stage
//! is reached through a trait object.

mod artifact;
mod stack;
mod staged;

pub use artifact::PersistedStack;
pub use stack::{MAX_STAGES, TransformerStack};
pub use staged::StagedPipeline;

use crate::api::ProbabilisticClassifier;
use crate::api::{
    Capabilities, Estimator, HasCapabilities, ModelError, Transformer, validate_transformed_shape,
};
use crate::artifact::{
    ArtifactError, ModelArtifact, STANDARD_SCALER_LINEAR_PIPELINE_ARTIFACT_KIND,
    STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND, STANDARD_SCALER_RIDGE_PIPELINE_ARTIFACT_KIND,
    SchemaRole, StageArtifact, decode_component, decode_v2_envelope, encode_component,
    encode_v2_envelope,
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
///
/// A pipeline is one fitted object, so the transformation travels with the
/// model. That is what stops the classic mistake of scaling the training data
/// and then predicting on unscaled rows: there is no separate step left to
/// forget.
///
/// ```
/// use ferricml::api::Estimator;
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::linear_model::{Ridge, RidgeParams};
/// use ferricml::pipeline::Pipeline;
/// use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
///
/// // Two columns on wildly different scales.
/// let data = DenseMatrix::new(
///     vec![1.0, 1000.0, 2.0, 3000.0, 3.0, 2000.0, 4.0, 5000.0],
///     4,
///     2,
/// )?;
/// let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;
///
/// // Fit the scaler, then fit the model on what the scaler produced.
/// let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default())?;
/// let scaled = scaler.transform(&data.as_view())?;
/// let model = Ridge::fit(&scaled.as_view(), &targets, RidgeParams::default())?;
///
/// // One object from here on. Construction checks the width handoff.
/// let pipeline = Pipeline::new(scaler, model)?;
/// assert_eq!(pipeline.n_features_in(), 2);
///
/// // Inference takes raw rows and scales them on the way through, into a
/// // workspace the caller owns, so a batch allocates nothing.
/// let rows = 4;
/// let mut workspace = vec![0.0_f32; pipeline.workspace_len(rows)?];
/// let mut predictions = vec![0.0_f32; rows];
/// pipeline.predict_into(&data.as_view(), &mut workspace, &mut predictions)?;
///
/// assert!(predictions.iter().all(|value| value.is_finite()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
    /// Fits the transformer and the estimator in one pass, in that order.
    ///
    /// The transformer closure sees the caller's data and the estimator closure
    /// sees what the transformer produced, so the estimator cannot be fitted on
    /// untransformed rows by mistake — the one handoff error that yields a
    /// silently wrong model instead of a width mismatch.
    ///
    /// The single-transformer case gets this for the same reason every longer
    /// composition has it: one-call fitting is a property of a composition, not
    /// of a length.
    ///
    /// ```
    /// use ferricml::data::{DenseMatrix, RegressionTargets};
    /// use ferricml::linear_model::{Ridge, RidgeParams};
    /// use ferricml::pipeline::Pipeline;
    /// use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
    ///
    /// let data = DenseMatrix::new(vec![1.0, 1000.0, 2.0, 3000.0, 3.0, 2000.0, 4.0, 5000.0], 4, 2)?;
    /// let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;
    ///
    /// let pipeline = Pipeline::fit(
    ///     &data.as_view(),
    ///     |view| StandardScaler::fit(view, StandardScalerParams::default()),
    ///     |view| Ridge::fit(view, &targets, RidgeParams::default()),
    /// )?;
    ///
    /// let mut workspace = vec![0.0_f32; pipeline.workspace_len(4)?];
    /// let mut predictions = vec![0.0_f32; 4];
    /// pipeline.predict_into(&data.as_view(), &mut workspace, &mut predictions)?;
    /// assert!(predictions.iter().all(|value| value.is_finite()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fit(
        data: &MatrixView<'_>,
        fit_transformer: impl FnOnce(&MatrixView<'_>) -> Result<T, ModelError>,
        fit_estimator: impl FnOnce(&MatrixView<'_>) -> Result<E, ModelError>,
    ) -> Result<Self, ModelError> {
        let transformer = fit_transformer(data)?;
        let transformed = transformer.transform(data)?;
        let estimator = fit_estimator(&transformed.as_view())?;
        Self::new(transformer, estimator)
    }

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
///
/// A decision function is **not** an intersection. Persisting is a property
/// every part must have, but a raw decision score is a property of the *final
/// estimator* alone — a transformer never has one, so intersecting would make
/// the field structurally unable to be true for any pipeline, while this
/// composition really does expose
/// [`decision_function_into`](Pipeline::decision_function_into). It is
/// therefore taken from the estimator, exactly where the method comes from.
impl HasCapabilities for Pipeline<StandardScaler, LogisticRegression> {
    const CAPABILITIES: Capabilities = StandardScaler::CAPABILITIES
        .intersection(LogisticRegression::CAPABILITIES)
        .with_sample_weights(false)
        .with_decision_function(LogisticRegression::CAPABILITIES.decision_function())
        .with_probability(LogisticRegression::CAPABILITIES.probability());
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
            ProbabilisticClassifier::predict_class_proba_into(model, transformed, class, output)
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
}

impl StageArtifact for Pipeline<StandardScaler, LogisticRegression> {
    const ARTIFACT_KIND: u16 = STANDARD_SCALER_LOGISTIC_PIPELINE_ARTIFACT_KIND;

    /// Encodes this concrete fitted pipeline and both schema identities.
    fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        encode_pipeline_artifact(
            Self::ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            &self
                .transformer
                .to_artifact(input_schema, transformed_schema)?,
            &self.estimator.to_artifact(transformed_schema)?,
        )
    }

    /// Decodes this concrete pipeline and validates the fitted width handoff.
    fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let (transformer, estimator) = decode_pipeline_components(
            bytes,
            Self::ARTIFACT_KIND,
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
        /// A fitted composition persists exactly when both of its parts do.
        ///
        /// Weighted fitting is declared away structurally rather than
        /// intersected, for the same reason it is on the logistic
        /// composition: this type composes parts that are *already* fitted, so
        /// accepting weights is a property of fitting each part rather than of
        /// the composition. The remaining fields need no special case here —
        /// neither a scaler nor a regressor declares probabilities, a class
        /// set, or a decision function, so the intersection is already the
        /// truth for all three.
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
        }

        impl StageArtifact for Pipeline<StandardScaler, $estimator> {
            const ARTIFACT_KIND: u16 = $kind;

            /// Encodes this concrete fitted pipeline and both schema identities.
            fn to_artifact(
                &self,
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Vec<u8>, ArtifactError> {
                encode_pipeline_artifact(
                    Self::ARTIFACT_KIND,
                    input_schema,
                    transformed_schema,
                    &self
                        .transformer
                        .to_artifact(input_schema, transformed_schema)?,
                    &self.estimator.to_artifact(transformed_schema)?,
                )
            }

            /// Decodes this concrete pipeline and validates the fitted width handoff.
            fn from_artifact(
                bytes: &[u8],
                input_schema: [u8; 32],
                transformed_schema: [u8; 32],
            ) -> Result<Self, ArtifactError> {
                let (transformer, estimator) = decode_pipeline_components(
                    bytes,
                    Self::ARTIFACT_KIND,
                    input_schema,
                    transformed_schema,
                )?;
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

    /// One-call fitting is a property of a composition, not of a length, so the
    /// single-transformer type has it too — and it means the same thing: the
    /// estimator is fitted on what the transformer produced.
    #[test]
    fn a_one_call_fit_equals_fitting_both_parts_by_hand() {
        let raw = data();
        let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
        let scaler = scaler();
        let transformed = scaler.transform(&raw.as_view()).unwrap();
        let expected = Pipeline::new(
            scaler,
            Ridge::fit(&transformed.as_view(), &targets, RidgeParams::default()).unwrap(),
        )
        .unwrap();

        let fitted = Pipeline::fit(
            &raw.as_view(),
            |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
            |batch| Ridge::fit(batch, &targets, RidgeParams::default()),
        )
        .unwrap();
        assert_eq!(fitted, expected);

        // The estimator really saw the transformed batch: fitting it on the raw
        // one gives a different model.
        let unscaled = Ridge::fit(&raw.as_view(), &targets, RidgeParams::default()).unwrap();
        assert_ne!(fitted.estimator(), &unscaled);
    }

    /// A failing stage stops the composition before the estimator is fitted.
    #[test]
    fn a_failing_transformer_fit_never_reaches_the_estimator() {
        let raw = data();
        let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
        let estimator_fits = std::cell::Cell::new(0_u32);
        let outcome: Result<Pipeline<StandardScaler, Ridge>, ModelError> = Pipeline::fit(
            &raw.as_view(),
            |_| Err(ModelError::EmptyData),
            |batch| {
                estimator_fits.set(estimator_fits.get() + 1);
                Ridge::fit(batch, &targets, RidgeParams::default())
            },
        );
        assert_eq!(outcome.unwrap_err(), ModelError::EmptyData);
        assert_eq!(estimator_fits.get(), 0);
    }

    /// The documented ceiling is the ceiling the impls actually reach.
    ///
    /// This is a compile-time claim: the function body is never run, and the
    /// bound is what fails if the longest generated arity ever falls short of
    /// the constant that advertises it.
    #[test]
    fn the_advertised_stage_ceiling_is_the_one_the_impls_reach() {
        const fn accepts_the_longest_stack<S: TransformerStack>() {}
        type Longest = (
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
            StandardScaler,
        );
        accepts_the_longest_stack::<Longest>();
        accepts_the_longest_stack::<(StandardScaler,)>();
        assert_eq!(MAX_STAGES, 12);
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
