//! Statically dispatched stacks of fitted transform stages.
//!
//! A stack is a tuple of fitted [`Transformer`]s applied left to right. Every
//! stage is a concrete type, so the whole chain is monomorphized and no stage
//! is reached through a trait object. Intermediate results live in one
//! caller-owned workspace, split per stage, which is what keeps a multi-stage
//! transform free of hidden allocation.

use crate::api::{Estimator, HasCapabilities, ModelError, Transformer, validate_transformed_shape};
use crate::data::MatrixView;

/// One or more fitted transform stages applied in a fixed order.
///
/// FerricML implements this for tuples of two and three fitted transformers.
/// A single stage is [`Pipeline`](super::Pipeline), which needs no workspace
/// split at all.
///
/// The workspace contract is the reason this is a trait rather than a concrete
/// chain type: a stack reports exactly how much `f32` storage it needs for a
/// batch, and every stage writes into a disjoint segment of that one buffer.
pub trait TransformerStack {
    /// Whether every stage in this stack persists.
    ///
    /// Derived from the stages' own capability declarations rather than
    /// declared here, so a stack cannot claim a persistence its parts do not
    /// have and cannot lose one they do. This is what lets a composition
    /// *compute* its artifact capability instead of being gated on one.
    const STAGES_PERSIST: bool;

    /// Input width the first stage was fitted on.
    fn n_features_in(&self) -> usize;

    /// Output width the last stage produces.
    fn n_features_out(&self) -> usize;

    /// Validates every stage-to-stage feature-width handoff, left to right.
    fn validate_handoff(&self) -> Result<(), ModelError>;

    /// Number of `f32` values needed to transform a batch of `rows` rows.
    fn workspace_len(&self, rows: usize) -> Result<usize, ModelError>;

    /// Transforms a batch through every stage into caller-owned workspace.
    ///
    /// The returned view covers the last stage's segment of `workspace`.
    /// Widths, the handoff, and the workspace length are all validated before
    /// any stage writes.
    fn transform_into<'workspace>(
        &self,
        data: &MatrixView<'_>,
        workspace: &'workspace mut [f32],
    ) -> Result<MatrixView<'workspace>, ModelError>;
}

/// Rejects a handoff where one stage's output width is not the next stage's
/// fitted input width.
fn check_handoff(produced: usize, expected: usize) -> Result<(), ModelError> {
    if produced == expected {
        Ok(())
    } else {
        Err(ModelError::FeatureDimension {
            expected,
            actual: produced,
        })
    }
}

/// Values one stage writes for a batch of `rows` rows.
fn stage_len(rows: usize, columns: usize) -> Result<usize, ModelError> {
    rows.checked_mul(columns)
        .ok_or(ModelError::OutputShapeOverflow { rows, columns })
}

/// Adds one stage's segment to a running workspace requirement.
fn extend_workspace(total: usize, rows: usize, columns: usize) -> Result<usize, ModelError> {
    total
        .checked_add(stage_len(rows, columns)?)
        .ok_or(ModelError::OutputShapeOverflow { rows, columns })
}

/// Validates a whole transform request before any stage writes.
fn validate_request<S: TransformerStack + ?Sized>(
    stack: &S,
    data: &MatrixView<'_>,
    workspace: &[f32],
) -> Result<(), ModelError> {
    stack.validate_handoff()?;
    let expected_width = stack.n_features_in();
    if data.columns() != expected_width {
        return Err(ModelError::FeatureDimension {
            expected: expected_width,
            actual: data.columns(),
        });
    }
    let expected = stack.workspace_len(data.rows())?;
    if workspace.len() != expected {
        return Err(ModelError::OutputLength {
            expected,
            actual: workspace.len(),
        });
    }
    Ok(())
}

/// Runs one stage into its workspace segment and re-checks what it returned.
fn run_stage<'segment, T: Transformer>(
    stage: &T,
    data: &MatrixView<'_>,
    segment: &'segment mut [f32],
) -> Result<MatrixView<'segment>, ModelError> {
    let transformed = stage.transform_into(data, segment)?;
    validate_transformed_shape(data.rows(), stage.n_features_out(), &transformed)?;
    Ok(transformed)
}

impl<A, B> TransformerStack for (A, B)
where
    A: Transformer + HasCapabilities,
    B: Transformer + HasCapabilities,
{
    const STAGES_PERSIST: bool = A::CAPABILITIES.artifact() && B::CAPABILITIES.artifact();

    fn n_features_in(&self) -> usize {
        Estimator::n_features_in(&self.0)
    }

    fn n_features_out(&self) -> usize {
        self.1.n_features_out()
    }

    fn validate_handoff(&self) -> Result<(), ModelError> {
        check_handoff(self.0.n_features_out(), Estimator::n_features_in(&self.1))
    }

    fn workspace_len(&self, rows: usize) -> Result<usize, ModelError> {
        let total = extend_workspace(0, rows, self.0.n_features_out())?;
        extend_workspace(total, rows, self.1.n_features_out())
    }

    fn transform_into<'workspace>(
        &self,
        data: &MatrixView<'_>,
        workspace: &'workspace mut [f32],
    ) -> Result<MatrixView<'workspace>, ModelError> {
        validate_request(self, data, workspace)?;
        let (first, rest) =
            workspace.split_at_mut(stage_len(data.rows(), self.0.n_features_out())?);
        let intermediate = run_stage(&self.0, data, first)?;
        run_stage(&self.1, &intermediate, rest)
    }
}

impl<A, B, C> TransformerStack for (A, B, C)
where
    A: Transformer + HasCapabilities,
    B: Transformer + HasCapabilities,
    C: Transformer + HasCapabilities,
{
    const STAGES_PERSIST: bool =
        A::CAPABILITIES.artifact() && B::CAPABILITIES.artifact() && C::CAPABILITIES.artifact();

    fn n_features_in(&self) -> usize {
        Estimator::n_features_in(&self.0)
    }

    fn n_features_out(&self) -> usize {
        self.2.n_features_out()
    }

    fn validate_handoff(&self) -> Result<(), ModelError> {
        check_handoff(self.0.n_features_out(), Estimator::n_features_in(&self.1))?;
        check_handoff(self.1.n_features_out(), Estimator::n_features_in(&self.2))
    }

    fn workspace_len(&self, rows: usize) -> Result<usize, ModelError> {
        let total = extend_workspace(0, rows, self.0.n_features_out())?;
        let total = extend_workspace(total, rows, self.1.n_features_out())?;
        extend_workspace(total, rows, self.2.n_features_out())
    }

    fn transform_into<'workspace>(
        &self,
        data: &MatrixView<'_>,
        workspace: &'workspace mut [f32],
    ) -> Result<MatrixView<'workspace>, ModelError> {
        validate_request(self, data, workspace)?;
        let (first, rest) =
            workspace.split_at_mut(stage_len(data.rows(), self.0.n_features_out())?);
        let (second, rest) = rest.split_at_mut(stage_len(data.rows(), self.1.n_features_out())?);
        let intermediate = run_stage(&self.0, data, first)?;
        let intermediate = run_stage(&self.1, &intermediate, second)?;
        run_stage(&self.2, &intermediate, rest)
    }
}
