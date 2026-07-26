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
/// FerricML implements this for every flat tuple of fitted transformers from
/// one stage to [`MAX_STAGES`]. A one-tuple is a real stack rather than a
/// special case, so nothing about a stack's vocabulary changes with its length;
/// [`Pipeline`](super::Pipeline) remains the two-part shorthand for the common
/// single-transformer composition and needs no workspace split at all.
///
/// # Why flat tuples and a ceiling
///
/// A right-nested pair — `(A, (B, (C,)))` — would give unbounded arity from two
/// impls, and it is rejected. It conflicts (`E0119`) with a flat impl, so it
/// cannot be added beside one; it would rewrite the type parameters of every
/// composition FerricML has shipped; and it replaces the flat tuple vocabulary
/// every signature, artifact tag sequence and test already reads with a shape
/// whose nesting carries no meaning. A fixed ceiling over flat tuples is what
/// the standard library does for the same reason.
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

/// Implements the stack contract for one flat tuple arity.
///
/// The bodies are written as running folds rather than per-arity code, so every
/// arity is the same four operations and none of them can drift from another:
/// widths chain left to right, the workspace requirement accumulates one stage
/// at a time, and each stage runs into the segment split off the front of what
/// its predecessors left.
macro_rules! impl_transformer_stack {
    ($head:ident $head_index:tt $(, $tail:ident $tail_index:tt)*) => {
        impl<$head, $($tail,)*> TransformerStack for ($head, $($tail,)*)
        where
            $head: Transformer + HasCapabilities,
            $($tail: Transformer + HasCapabilities,)*
        {
            const STAGES_PERSIST: bool =
                $head::CAPABILITIES.artifact() $(&& $tail::CAPABILITIES.artifact())*;

            fn n_features_in(&self) -> usize {
                Estimator::n_features_in(&self.$head_index)
            }

            fn n_features_out(&self) -> usize {
                let widths = [
                    self.$head_index.n_features_out(),
                    $(self.$tail_index.n_features_out(),)*
                ];
                // The array is fixed-size per arity, so this is the last stage's
                // width rather than a search, and it never indexes out of range.
                widths[widths.len() - 1]
            }

            fn validate_handoff(&self) -> Result<(), ModelError> {
                let produced = self.$head_index.n_features_out();
                $(
                    check_handoff(produced, Estimator::n_features_in(&self.$tail_index))?;
                    let produced = self.$tail_index.n_features_out();
                )*
                let _ = produced;
                Ok(())
            }

            fn workspace_len(&self, rows: usize) -> Result<usize, ModelError> {
                let total = extend_workspace(0, rows, self.$head_index.n_features_out())?;
                $(let total = extend_workspace(total, rows, self.$tail_index.n_features_out())?;)*
                Ok(total)
            }

            fn transform_into<'workspace>(
                &self,
                data: &MatrixView<'_>,
                workspace: &'workspace mut [f32],
            ) -> Result<MatrixView<'workspace>, ModelError> {
                validate_request(self, data, workspace)?;
                let rows = data.rows();
                let (segment, rest) =
                    workspace.split_at_mut(stage_len(rows, self.$head_index.n_features_out())?);
                let produced = run_stage(&self.$head_index, data, segment)?;
                $(
                    let (segment, rest) =
                        rest.split_at_mut(stage_len(rows, self.$tail_index.n_features_out())?);
                    let produced = run_stage(&self.$tail_index, &produced, segment)?;
                )*
                let _ = rest;
                Ok(produced)
            }
        }
    };
}

/// The longest transform chain a [`TransformerStack`] tuple carries.
///
/// A ceiling exists because the impls are generated per flat tuple arity rather
/// than recursively; see [`TransformerStack`] for why that trade is taken. It
/// matches the standard library's own tuple-trait ceiling.
pub const MAX_STAGES: usize = 12;

impl_transformer_stack!(A 0);
impl_transformer_stack!(A 0, B 1);
impl_transformer_stack!(A 0, B 1, C 2);
impl_transformer_stack!(A 0, B 1, C 2, D 3);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
impl_transformer_stack!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
