//! Generic fitted preprocessing pipelines.
//!
//! The initial seam composes one fitted transformer with one fitted estimator.
//! Both types remain concrete, so transformation and the callback passed to
//! [`Pipeline::with_transformed`] are statically dispatched. Training-time
//! composition and multi-step pipelines will be added with the first concrete
//! preprocessing estimator rather than guessed in advance.

use crate::api::{Estimator, ModelError, Transformer, validate_transformed_shape};
use crate::data::{DenseMatrix, MatrixView};

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
