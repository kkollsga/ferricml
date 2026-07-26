//! Trainable multi-stage typed pipelines.
//!
//! [`StagedPipeline`] composes one or more fitted transform stages with one
//! fitted estimator. Every part stays a concrete type, so the whole
//! composition is monomorphized: there is no per-row dynamic dispatch, no
//! parameter erasure, and no string registry of stages.

use crate::api::{Capabilities, Estimator, HasCapabilities, ModelError, Transformer};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, MODEL_ARTIFACT_VERSION, ModelArtifact,
    STAGED_PIPELINE_ARTIFACT_KIND, SchemaRole, StageArtifact, artifact_version, decode_component,
    decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::{DenseMatrix, MatrixView};

use super::{PersistedStack, TransformerStack};

const PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const STAGE_COMPONENT_KIND: u16 = 2;
const ESTIMATOR_COMPONENT_KIND: u16 = 3;
const COMPONENT_VERSION: u16 = 1;

/// One or more fitted transform stages followed by one fitted estimator.
///
/// A composition is built either from already-fitted parts with
/// [`StagedPipeline::new`], which validates every feature-width handoff before
/// the composition exists, or in one training pass with
/// [`StagedPipeline::fit`], which fits each stage on the previous stage's
/// output and only then fits the estimator. Composition and persistence work at
/// every length [`TransformerStack`] covers; one-call fitting is bounded at two
/// stages for the reason recorded on [`StagedPipeline::fit`].
///
/// Inference is allocation-free: [`StagedPipeline::workspace_len`] reports one
/// buffer size, the caller allocates it once, and every batch reuses it
/// through [`StagedPipeline::with_transformed`].
///
/// ```
/// use ferricml::api::Estimator;
/// use ferricml::data::{DenseMatrix, RegressionTargets};
/// use ferricml::linear_model::{Ridge, RidgeParams};
/// use ferricml::pipeline::StagedPipeline;
/// use ferricml::preprocessing::{
///     MinMaxScaler, MinMaxScalerParams, StandardScaler, StandardScalerParams,
/// };
///
/// let data = DenseMatrix::new(
///     vec![1.0, 1000.0, 2.0, 3000.0, 3.0, 2000.0, 4.0, 5000.0],
///     4,
///     2,
/// )?;
/// let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;
///
/// // One training pass: each stage is fitted on the previous stage output,
/// // and the estimator on what the last stage produced.
/// let pipeline = StagedPipeline::fit(
///     &data.as_view(),
///     |view| StandardScaler::fit(view, StandardScalerParams::default()),
///     |view| MinMaxScaler::fit(view, MinMaxScalerParams::default()),
///     |view| Ridge::fit(view, &targets, RidgeParams::default()),
/// )?;
///
/// assert_eq!(pipeline.n_features_in(), 2);
///
/// // One workspace is split into a disjoint segment per stage, so multi-stage
/// // inference allocates nothing at all.
/// let mut workspace = vec![0.0_f32; pipeline.workspace_len(4)?];
/// let predictions = pipeline.with_transformed(
///     &data.as_view(),
///     &mut workspace,
///     |model, view| model.predict(view),
/// )?;
///
/// assert_eq!(predictions.len(), 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct StagedPipeline<S, E> {
    stages: S,
    estimator: E,
}

impl<S, E> StagedPipeline<S, E>
where
    S: TransformerStack,
    E: Estimator,
{
    /// Composes fitted parts after validating every feature-width handoff.
    ///
    /// Each stage-to-stage handoff is checked left to right, then the last
    /// stage's output width against the estimator's fitted input width. A
    /// mismatch anywhere is reported here, so a composition that could not
    /// predict never exists.
    pub fn new(stages: S, estimator: E) -> Result<Self, ModelError> {
        stages.validate_handoff()?;
        let transformed = stages.n_features_out();
        let expected = estimator.n_features_in();
        if transformed != expected {
            return Err(ModelError::FeatureDimension {
                expected,
                actual: transformed,
            });
        }
        Ok(Self { stages, estimator })
    }

    /// Returns the fitted transform stages.
    pub const fn stages(&self) -> &S {
        &self.stages
    }

    /// Returns the fitted final estimator.
    pub const fn estimator(&self) -> &E {
        &self.estimator
    }

    /// Consumes the pipeline and returns its fitted parts.
    pub fn into_parts(self) -> (S, E) {
        (self.stages, self.estimator)
    }

    /// Number of `f32` values required for a transformed batch workspace.
    ///
    /// Every stage writes into a disjoint segment of this one buffer, so a
    /// caller allocates once and reuses it for every batch.
    pub fn workspace_len(&self, rows: usize) -> Result<usize, ModelError> {
        self.stages.workspace_len(rows)
    }

    /// Runs every stage into caller-owned workspace and returns the final view.
    pub fn transform_into<'workspace>(
        &self,
        data: &MatrixView<'_>,
        workspace: &'workspace mut [f32],
    ) -> Result<MatrixView<'workspace>, ModelError> {
        self.stages.transform_into(data, workspace)
    }

    /// Runs every stage into a newly allocated dense matrix.
    pub fn transform(&self, data: &MatrixView<'_>) -> Result<DenseMatrix, ModelError> {
        // Before the workspace, not inside the first stage after it. The stack
        // repeats the check for callers that own their own workspace.
        let expected = self.stages.n_features_in();
        if data.columns() != expected {
            return Err(ModelError::FeatureDimension {
                expected,
                actual: data.columns(),
            });
        }
        let mut workspace = vec![0.0; self.workspace_len(data.rows())?];
        let transformed = self.stages.transform_into(data, &mut workspace)?;
        Ok(transformed.to_dense())
    }

    /// Runs an operation on a fully transformed batch without allocating or
    /// erasing any fitted type.
    ///
    /// This is the allocation-free inference path: the callback receives the
    /// concrete fitted estimator and the transformed batch, so it can call the
    /// estimator's own `_into` method while the caller reuses `workspace`
    /// across batches. It is deliberately the only prediction entry point —
    /// it works for every estimator category, including ones FerricML has not
    /// added yet, and it keeps the estimator's own vocabulary rather than
    /// restating it once per category.
    pub fn with_transformed<R>(
        &self,
        data: &MatrixView<'_>,
        workspace: &mut [f32],
        operation: impl FnOnce(&E, &MatrixView<'_>) -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        let transformed = self.stages.transform_into(data, workspace)?;
        operation(&self.estimator, &transformed)
    }
}

impl<A, B, E> StagedPipeline<(A, B), E>
where
    A: Transformer + HasCapabilities,
    B: Transformer + HasCapabilities,
    E: Estimator,
{
    /// Fits two stages and an estimator in one pass, in that fixed order.
    ///
    /// Each closure receives exactly the batch its part is fitted on: the
    /// first stage sees `data`, the second sees the first stage's output, and
    /// the estimator sees the fully transformed batch. Parameters stay in the
    /// closures, so no parameter type is erased and per-stage sample weights
    /// propagate exactly as the caller writes them.
    ///
    /// A stage that cannot consume the previous stage's output fails here,
    /// before the estimator closure is called at all.
    ///
    /// # Why one-call fitting stops at two stages
    ///
    /// [`StagedPipeline::new`] composes any length the stack trait covers, and
    /// so does persistence; only this one entry point is bounded at two. Both
    /// ways of lifting the bound were built and measured, and both cost more
    /// than the bound does.
    ///
    /// *A second inherent `fit` for a longer composition* is `E0034` at every
    /// existing call site — not only at the new one, and including sites whose
    /// binding is fully annotated — because an inherent associated function is
    /// resolved before `Self` is inferred. Measured: ten call sites in this
    /// repository stop compiling.
    ///
    /// *One `fit` taking a tuple of stage closures* moves the arity out of
    /// method resolution and does give a single entry point at every length.
    /// It compiles, and it makes every stage closure at every call site require
    /// either a `&MatrixView<'_>` parameter annotation or a coercion wrapper:
    /// a closure written inside a tuple is not checked against a function-trait
    /// obligation, so its signature is inferred with early-bound regions and
    /// then rejected as "not general enough". Measured: all fourteen call sites
    /// in this repository, including every two-stage one. Making the common
    /// case carry a permanent workaround for a compiler limitation, in order to
    /// reach lengths reached today by fitting each stage and calling
    /// [`StagedPipeline::new`], is the worse trade.
    ///
    /// Longer compositions are therefore built stage by stage and composed with
    /// [`StagedPipeline::new`], which revalidates every width handoff. If
    /// closure signature inference through tuples improves, the general form
    /// becomes free and this bound should go.
    pub fn fit(
        data: &MatrixView<'_>,
        fit_first: impl FnOnce(&MatrixView<'_>) -> Result<A, ModelError>,
        fit_second: impl FnOnce(&MatrixView<'_>) -> Result<B, ModelError>,
        fit_estimator: impl FnOnce(&MatrixView<'_>) -> Result<E, ModelError>,
    ) -> Result<Self, ModelError> {
        let first = fit_first(data)?;
        let intermediate = first.transform(data)?;
        let second = fit_second(&intermediate.as_view())?;
        let transformed = second.transform(&intermediate.as_view())?;
        let estimator = fit_estimator(&transformed.as_view())?;
        Self::new((first, second), estimator)
    }
}

impl<S, E> Estimator for StagedPipeline<S, E>
where
    S: TransformerStack,
    E: Estimator,
{
    fn n_features_in(&self) -> usize {
        self.stages.n_features_in()
    }
}

/// A composition persists exactly when every one of its parts does.
///
/// That sentence is a statement about the declaration's *value*, so it is
/// computed rather than gated. An earlier version instead bounded this impl on
/// the persistence traits, which conflated two different things: "this
/// composition declares no artifact" and "this composition cannot declare
/// anything at all". A composition holding a stateless stage, or ending in a
/// baseline estimator, is perfectly usable — it simply does not persist — yet
/// under the bound it had no capability vocabulary, and the conformance
/// battery requires a declaring model to check one. Whole families of working
/// compositions were therefore unreachable by the battery for the sole reason
/// that they were honest about not persisting.
///
/// Taking the declaration from the parts is the same shape that fixed
/// `decision_function` on [`Pipeline`](super::Pipeline): ask where the property
/// actually comes from. Persistence genuinely is an intersection — every part
/// must have it — so it is computed as one, from each part's own declaration.
///
/// Weighted fitting is declared away structurally. `StagedPipeline` has no
/// `fit_weighted` of its own — weights reach the parts through the fitting
/// closures a caller writes — so accepting weights is a property of fitting
/// each part, never of the composition.
impl<S, E> HasCapabilities for StagedPipeline<S, E>
where
    S: TransformerStack,
    E: Estimator + HasCapabilities,
{
    const CAPABILITIES: Capabilities =
        Capabilities::NONE.with_artifact(S::STAGES_PERSIST && E::CAPABILITIES.artifact());
}

impl<S, E> StageArtifact for StagedPipeline<S, E>
where
    S: TransformerStack + PersistedStack,
    E: Estimator + ModelArtifact,
{
    const ARTIFACT_KIND: u16 = STAGED_PIPELINE_ARTIFACT_KIND;

    /// Encodes the whole composition and both schema identities.
    ///
    /// The payload records which concrete stage types the composition holds,
    /// in order, plus its estimator type, so a composition never decodes as a
    /// different one.
    fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        let stages = self
            .stages
            .encode_stages(input_schema, transformed_schema)?;
        if stages.len() != S::STAGE_TAGS.len() {
            return Err(ArtifactError::InvalidPayload);
        }
        let count = u32::try_from(stages.len()).map_err(|_| ArtifactError::InvalidPayload)?;

        let mut metadata = ArtifactPayloadWriter::with_capacity(8 + stages.len() * 4);
        metadata.u32(count);
        metadata.u32(u32::from(E::MODEL_TAG));
        for &tag in S::STAGE_TAGS {
            metadata.u32(u32::from(tag));
        }

        let mut payload = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        for stage in &stages {
            payload.extend_from_slice(&encode_component(
                STAGE_COMPONENT_KIND,
                COMPONENT_VERSION,
                stage,
            )?);
        }
        payload.extend_from_slice(&encode_component(
            ESTIMATOR_COMPONENT_KIND,
            COMPONENT_VERSION,
            &self.estimator.to_artifact(transformed_schema)?,
        )?);

        encode_v2_envelope(
            Self::ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[
                (SchemaRole::Input, input_schema),
                (SchemaRole::Transformed, transformed_schema),
            ],
            &payload,
        )
    }

    /// Decodes the whole composition and revalidates every width handoff.
    ///
    /// Bytes are never trusted: the recorded stage count, stage tags, and
    /// estimator tag must all match the composition being decoded into, every
    /// part revalidates its own payload, and the reconstructed composition
    /// goes back through [`StagedPipeline::new`].
    fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let version = artifact_version(bytes)?;
        if version != MODEL_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedVersion { found: version });
        }
        let mut envelope = decode_v2_envelope(
            bytes,
            Self::ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[
                (SchemaRole::Input, input_schema),
                (SchemaRole::Transformed, transformed_schema),
            ],
        )?;

        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let count = metadata.u32()? as usize;
        let estimator_tag = metadata.u32()?;
        if count != S::STAGE_TAGS.len() || estimator_tag != u32::from(E::MODEL_TAG) {
            return Err(ArtifactError::InvalidPayload);
        }
        for &expected in S::STAGE_TAGS {
            if metadata.u32()? != u32::from(expected) {
                return Err(ArtifactError::InvalidPayload);
            }
        }
        if !metadata.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }

        let mut components = Vec::with_capacity(count);
        for _ in 0..count {
            let stage = decode_component(&mut envelope, STAGE_COMPONENT_KIND, COMPONENT_VERSION)?;
            components.push(stage.remaining());
        }
        let estimator =
            decode_component(&mut envelope, ESTIMATOR_COMPONENT_KIND, COMPONENT_VERSION)?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }

        let stages = S::decode_stages(&components, input_schema, transformed_schema)?;
        let estimator = E::from_artifact(estimator.remaining(), transformed_schema)?;
        Self::new(stages, estimator).map_err(|_| ArtifactError::InvalidPayload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::data::RegressionTargets;
    use crate::linear_model::{LinearRegression, Ridge, RidgeParams};
    use crate::preprocessing::{
        MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams, StandardScaler,
        StandardScalerParams,
    };
    use std::cell::Cell;

    fn data() -> DenseMatrix {
        DenseMatrix::new(
            vec![
                0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 7.0, 5.0, 6.0, 6.0, 5.0, 7.0, 4.0,
            ],
            8,
            2,
        )
        .unwrap()
    }

    fn targets() -> RegressionTargets {
        RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]).unwrap()
    }

    fn fitted() -> StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> {
        StagedPipeline::fit(
            &data().as_view(),
            |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
            |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
            |batch| Ridge::fit(batch, &targets(), RidgeParams::default()),
        )
        .unwrap()
    }

    #[test]
    fn a_staged_fit_equals_manually_fitted_stages() {
        let raw = data();
        let first = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let intermediate = first.transform(&raw.as_view()).unwrap();
        let second =
            StandardScaler::fit(&intermediate.as_view(), StandardScalerParams::default()).unwrap();
        let transformed = second.transform(&intermediate.as_view()).unwrap();
        let estimator =
            Ridge::fit(&transformed.as_view(), &targets(), RidgeParams::default()).unwrap();
        let expected = estimator.predict(&transformed.as_view()).unwrap();

        let pipeline = fitted();
        assert_eq!(pipeline.stages().0, first);
        assert_eq!(pipeline.stages().1, second);
        assert_eq!(pipeline.estimator(), &estimator);

        let mut workspace = vec![0.0; pipeline.workspace_len(raw.rows()).unwrap()];
        let mut actual = vec![0.0; raw.rows()];
        pipeline
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut actual)
            })
            .unwrap();
        assert_eq!(actual, expected);
    }

    /// A one-tuple is a real stack, not a degenerate case that happens to work.
    ///
    /// The reference has no special case at one step, and neither does this: a
    /// one-stage composition reports the same widths, splits the same single
    /// workspace, persists under the same envelope, and declares the same
    /// capability as a longer one.
    #[test]
    fn a_single_stage_stack_behaves_exactly_like_a_longer_one() {
        let raw = data();
        let scaler = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let transformed = scaler.transform(&raw.as_view()).unwrap();
        let estimator =
            Ridge::fit(&transformed.as_view(), &targets(), RidgeParams::default()).unwrap();
        let pipeline = StagedPipeline::new((scaler,), estimator).unwrap();

        assert_eq!(Estimator::n_features_in(&pipeline), 2);
        assert_eq!(pipeline.workspace_len(raw.rows()).unwrap(), 8 * 2);
        assert_eq!(
            pipeline.transform(&raw.as_view()).unwrap().as_slice(),
            transformed.as_slice()
        );
        assert!(
            <StagedPipeline<(MinMaxScaler,), Ridge> as HasCapabilities>::CAPABILITIES.artifact()
        );

        let bytes = pipeline.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            StagedPipeline::<(MinMaxScaler,), Ridge>::from_artifact(&bytes, [1; 32], [2; 32])
                .unwrap(),
            pipeline
        );
        // A two-stage composition records a different stage count, so a shorter
        // stack cannot be read as a longer one or the reverse.
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                &bytes, [1; 32], [2; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }

    /// Past the two and three the crate used to stop at, with the handoff bound
    /// and the workspace split both exercised beyond the old ceiling.
    #[test]
    fn a_stack_longer_than_three_chains_persists_and_rejects_a_bad_handoff() {
        let raw = data();
        let first = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let after_first = first.transform(&raw.as_view()).unwrap();
        let second =
            StandardScaler::fit(&after_first.as_view(), StandardScalerParams::default()).unwrap();
        let after_second = second.transform(&after_first.as_view()).unwrap();
        let third = MaxAbsScaler::fit(&after_second.as_view(), MaxAbsScalerParams).unwrap();
        let after_third = third.transform(&after_second.as_view()).unwrap();
        let fourth =
            MinMaxScaler::fit(&after_third.as_view(), MinMaxScalerParams::default()).unwrap();
        let after_fourth = fourth.transform(&after_third.as_view()).unwrap();
        let fifth =
            StandardScaler::fit(&after_fourth.as_view(), StandardScalerParams::default()).unwrap();
        let transformed = fifth.transform(&after_fourth.as_view()).unwrap();
        let estimator =
            Ridge::fit(&transformed.as_view(), &targets(), RidgeParams::default()).unwrap();

        let stages = (
            first.clone(),
            second.clone(),
            third.clone(),
            fourth.clone(),
            fifth,
        );
        let pipeline = StagedPipeline::new(stages, estimator.clone()).unwrap();
        assert_eq!(pipeline.workspace_len(raw.rows()).unwrap(), 8 * 2 * 5);
        assert_eq!(
            pipeline.transform(&raw.as_view()).unwrap().as_slice(),
            transformed.as_slice()
        );

        let mut workspace = vec![0.0; pipeline.workspace_len(raw.rows()).unwrap()];
        let mut predictions = vec![0.0; raw.rows()];
        pipeline
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut predictions)
            })
            .unwrap();
        assert_eq!(
            predictions,
            pipeline
                .estimator()
                .predict(&transformed.as_view())
                .unwrap()
        );

        let bytes = pipeline.to_artifact([3; 32], [4; 32]).unwrap();
        assert_eq!(
            StagedPipeline::<
                (
                    MinMaxScaler,
                    StandardScaler,
                    MaxAbsScaler,
                    MinMaxScaler,
                    StandardScaler,
                ),
                Ridge,
            >::from_artifact(&bytes, [3; 32], [4; 32])
            .unwrap(),
            pipeline
        );

        // The handoff check reaches every position, not just the first pair.
        let narrow = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        let wrong_width = MaxAbsScaler::fit(&narrow.as_view(), MaxAbsScalerParams).unwrap();
        assert_eq!(
            StagedPipeline::new((first, second, wrong_width, fourth), estimator).unwrap_err(),
            ModelError::FeatureDimension {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn three_stages_transform_in_order_through_one_workspace() {
        let raw = data();
        let first = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let after_first = first.transform(&raw.as_view()).unwrap();
        let second =
            StandardScaler::fit(&after_first.as_view(), StandardScalerParams::default()).unwrap();
        let after_second = second.transform(&after_first.as_view()).unwrap();
        let third = MaxAbsScaler::fit(&after_second.as_view(), MaxAbsScalerParams).unwrap();
        let expected = third.transform(&after_second.as_view()).unwrap();
        let estimator =
            Ridge::fit(&expected.as_view(), &targets(), RidgeParams::default()).unwrap();

        let pipeline = StagedPipeline::new((first, second, third), estimator).unwrap();
        assert_eq!(pipeline.workspace_len(raw.rows()).unwrap(), 8 * 2 * 3);
        assert_eq!(
            pipeline.transform(&raw.as_view()).unwrap().as_slice(),
            expected.as_slice()
        );

        let mut workspace = vec![0.0; pipeline.workspace_len(raw.rows()).unwrap()];
        let mut predictions = vec![0.0; raw.rows()];
        pipeline
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut predictions)
            })
            .unwrap();
        assert_eq!(
            predictions,
            pipeline.estimator().predict(&expected.as_view()).unwrap()
        );
    }

    #[test]
    fn inference_reuses_one_workspace_across_batches() {
        let pipeline = fitted();
        let raw = data();
        let mut workspace = vec![0.0; pipeline.workspace_len(raw.rows()).unwrap()];
        let mut first = vec![0.0; raw.rows()];
        let mut second = vec![0.0; raw.rows()];
        pipeline
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut first)
            })
            .unwrap();
        pipeline
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut second)
            })
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(workspace.len(), pipeline.workspace_len(raw.rows()).unwrap());
    }

    #[test]
    fn transform_matches_the_allocation_free_path() {
        let pipeline = fitted();
        let raw = data();
        let mut workspace = vec![0.0; pipeline.workspace_len(raw.rows()).unwrap()];
        let allocating = pipeline.transform(&raw.as_view()).unwrap();
        let into = pipeline
            .transform_into(&raw.as_view(), &mut workspace)
            .unwrap();
        assert_eq!(into.rows(), raw.rows());
        assert_eq!(into.columns(), 2);
        assert_eq!(into.as_slice(), allocating.as_slice());
    }

    #[test]
    fn a_mismatched_handoff_is_rejected_before_the_composition_exists() {
        let narrow = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        let wide = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        let first = MinMaxScaler::fit(&narrow.as_view(), MinMaxScalerParams::default()).unwrap();
        let second = StandardScaler::fit(&wide.as_view(), StandardScalerParams::default()).unwrap();
        let estimator = Ridge::fit(
            &wide.as_view(),
            &RegressionTargets::new(vec![0.0, 1.0, 2.0]).unwrap(),
            RidgeParams::default(),
        )
        .unwrap();
        assert_eq!(
            StagedPipeline::new((first, second), estimator).unwrap_err(),
            ModelError::FeatureDimension {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn a_mismatched_estimator_width_is_rejected_before_the_composition_exists() {
        let raw = data();
        let narrow = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        let first = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let second = StandardScaler::fit(&raw.as_view(), StandardScalerParams::default()).unwrap();
        let estimator = Ridge::fit(
            &narrow.as_view(),
            &RegressionTargets::new(vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
            RidgeParams::default(),
        )
        .unwrap();
        assert_eq!(
            StagedPipeline::new((first, second), estimator).unwrap_err(),
            ModelError::FeatureDimension {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn a_stage_that_cannot_consume_its_input_fails_before_any_estimator_work() {
        let raw = data();
        let narrow = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        let estimator_fits = Cell::new(0_u32);
        let outcome: Result<StagedPipeline<(MinMaxScaler, StandardScaler), Ridge>, ModelError> =
            StagedPipeline::fit(
                &raw.as_view(),
                |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
                // Fitted on a one-column batch instead of the previous stage's
                // two-column output.
                |_| StandardScaler::fit(&narrow.as_view(), StandardScalerParams::default()),
                |batch| {
                    estimator_fits.set(estimator_fits.get() + 1);
                    Ridge::fit(batch, &targets(), RidgeParams::default())
                },
            );
        assert_eq!(
            outcome.unwrap_err(),
            ModelError::FeatureDimension {
                expected: 1,
                actual: 2
            }
        );
        assert_eq!(estimator_fits.get(), 0);
    }

    #[test]
    fn a_wrong_width_batch_is_rejected_before_the_workspace_is_touched() {
        let pipeline = fitted();
        let raw = data();
        let wrong = DenseMatrix::new(vec![1.0; 8 * 3], 8, 3).unwrap();
        let mut workspace = vec![91.0; pipeline.workspace_len(raw.rows()).unwrap()];
        assert_eq!(
            pipeline
                .transform_into(&wrong.as_view(), &mut workspace)
                .unwrap_err(),
            ModelError::FeatureDimension {
                expected: 2,
                actual: 3
            }
        );
        assert!(workspace.iter().all(|&value| value == 91.0));
    }

    #[test]
    fn a_short_workspace_is_rejected_before_any_stage_writes() {
        let pipeline = fitted();
        let raw = data();
        let expected = pipeline.workspace_len(raw.rows()).unwrap();
        let mut workspace = vec![91.0; expected - 1];
        assert_eq!(
            pipeline
                .transform_into(&raw.as_view(), &mut workspace)
                .unwrap_err(),
            ModelError::OutputLength {
                expected,
                actual: expected - 1
            }
        );
        assert!(workspace.iter().all(|&value| value == 91.0));
    }

    #[test]
    fn weighted_stage_fitting_propagates_in_order_and_is_deterministic() {
        use crate::data::SampleWeights;

        let raw = data();
        let weights = SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap();
        let build = || {
            StagedPipeline::fit(
                &raw.as_view(),
                |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
                |batch| {
                    StandardScaler::fit_weighted(batch, &weights, StandardScalerParams::default())
                },
                |batch| Ridge::fit_weighted(batch, &targets(), &weights, RidgeParams::default()),
            )
            .unwrap()
        };
        let first: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> = build();
        let second = build();
        assert_eq!(first, second);

        // The weighted second stage really did see the weights: an unweighted
        // fit of the same composition differs.
        let unweighted: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> =
            StagedPipeline::fit(
                &raw.as_view(),
                |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
                |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
                |batch| Ridge::fit(batch, &targets(), RidgeParams::default()),
            )
            .unwrap();
        assert_ne!(first, unweighted);
    }

    #[test]
    fn unit_weights_reproduce_the_unweighted_composition() {
        use crate::data::SampleWeights;

        let raw = data();
        let weights = SampleWeights::new(vec![1.0; raw.rows()]).unwrap();
        let weighted: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> = StagedPipeline::fit(
            &raw.as_view(),
            |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
            |batch| StandardScaler::fit_weighted(batch, &weights, StandardScalerParams::default()),
            |batch| Ridge::fit(batch, &targets(), RidgeParams::default()),
        )
        .unwrap();
        assert_eq!(weighted, fitted());
    }

    #[test]
    fn metadata_reports_the_first_stage_width() {
        let pipeline = fitted();
        assert_eq!(Estimator::n_features_in(&pipeline), 2);
        let (stages, estimator) = pipeline.into_parts();
        assert_eq!(stages.n_features_out(), 2);
        assert_eq!(estimator.n_features_in(), 2);
    }

    #[test]
    fn a_multi_stage_artifact_round_trips_deterministically_and_predicts_identically() {
        let pipeline = fitted();
        let raw = data();
        let bytes = pipeline.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(bytes, pipeline.to_artifact([1; 32], [2; 32]).unwrap());

        let decoded = StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
            &bytes, [1; 32], [2; 32],
        )
        .unwrap();
        assert_eq!(decoded, pipeline);

        let mut workspace = vec![0.0; pipeline.workspace_len(raw.rows()).unwrap()];
        let mut expected = vec![0.0; raw.rows()];
        let mut actual = vec![0.0; raw.rows()];
        pipeline
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut expected)
            })
            .unwrap();
        decoded
            .with_transformed(&raw.as_view(), &mut workspace, |model, batch| {
                model.predict_into(batch, &mut actual)
            })
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn a_three_stage_artifact_round_trips() {
        let raw = data();
        let first = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let after_first = first.transform(&raw.as_view()).unwrap();
        let second =
            StandardScaler::fit(&after_first.as_view(), StandardScalerParams::default()).unwrap();
        let after_second = second.transform(&after_first.as_view()).unwrap();
        let third = MaxAbsScaler::fit(&after_second.as_view(), MaxAbsScalerParams).unwrap();
        let transformed = third.transform(&after_second.as_view()).unwrap();
        let estimator =
            Ridge::fit(&transformed.as_view(), &targets(), RidgeParams::default()).unwrap();
        let pipeline = StagedPipeline::new((first, second, third), estimator).unwrap();

        let bytes = pipeline.to_artifact([5; 32], [6; 32]).unwrap();
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler, MaxAbsScaler), Ridge>::from_artifact(
                &bytes, [5; 32], [6; 32]
            )
            .unwrap(),
            pipeline
        );

        // A two-stage composition records a different stage count.
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                &bytes, [5; 32], [6; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }

    #[test]
    fn a_composition_never_decodes_as_a_different_one() {
        let raw = data();
        let bytes = fitted().to_artifact([1; 32], [2; 32]).unwrap();

        // Same stage types, opposite order: the recorded tag sequence differs.
        assert_eq!(
            StagedPipeline::<(StandardScaler, MinMaxScaler), Ridge>::from_artifact(
                &bytes, [1; 32], [2; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );

        // Same stages, different estimator type.
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), LinearRegression>::from_artifact(
                &bytes, [1; 32], [2; 32]
            )
            .unwrap_err(),
            ArtifactError::InvalidPayload
        );

        // A single-stage pipeline artifact is a different envelope kind.
        let scaler = MinMaxScaler::fit(&raw.as_view(), MinMaxScalerParams::default()).unwrap();
        let scaler_bytes = scaler.to_artifact([1; 32], [2; 32]).unwrap();
        assert!(matches!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                &scaler_bytes,
                [1; 32],
                [2; 32]
            )
            .unwrap_err(),
            ArtifactError::UnsupportedModelKind { .. }
        ));
    }

    #[test]
    fn a_multi_stage_artifact_is_schema_bound_and_checksummed() {
        let bytes = fitted().to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                &bytes, [9; 32], [2; 32]
            )
            .unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                &bytes, [1; 32], [9; 32]
            )
            .unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );

        let mut corrupted = bytes.clone();
        let last = corrupted.len() - 40;
        corrupted[last] ^= 1;
        assert_eq!(
            StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                &corrupted, [1; 32], [2; 32]
            )
            .unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }

    #[test]
    fn a_composition_declares_persistence_but_never_weighted_fitting() {
        assert!(
            <StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> as HasCapabilities>::CAPABILITIES
                .artifact()
        );
        assert!(
            !<StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> as HasCapabilities>::CAPABILITIES
                .sample_weights()
        );
        // Both parts accept weights when fitted on their own; the composition
        // that only holds them fitted does not.
        assert!(StandardScaler::CAPABILITIES.sample_weights());
        assert!(Ridge::CAPABILITIES.sample_weights());
    }
}
