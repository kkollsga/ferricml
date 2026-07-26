/// Metadata shared by every fitted estimator.
///
/// The trait is object-safe so model selection can dispatch once per batch.
/// Concrete inference methods remain inherent until their shape semantics are
/// shared by all implementations.
pub trait Estimator {
    /// Number of input features observed during fitting.
    fn n_features_in(&self) -> usize;
}

use crate::data::{DenseMatrix, MatrixView};

use super::ModelError;

/// A fitted classification estimator.
///
/// Batch methods are the shared, object-safe contract. Every classifier
/// predicts a label; **not** every classifier produces a probability. The
/// probability half lives on [`ProbabilisticClassifier`], because a
/// margin-based estimator — ridge classification, discriminant analysis, a
/// discrete boosting variant — has a natural output that is a score rather
/// than a distribution, and a required probability method would have forced it
/// either to squash a number it never earned or to fail at run time on a
/// method the type system promised. Both are worse than an honest split.
///
/// A caller that needs probabilities asks for them in its bounds, and gets a
/// compile error rather than a surprise. The one place that is impossible is a
/// runtime dispatch value, where the concrete type is erased by construction;
/// see [`AnyClassifier::as_probabilistic`](crate::api::AnyClassifier::as_probabilistic).
pub trait Classifier: Estimator {
    /// Sorted class labels observed during fitting.
    fn classes(&self) -> &[u8];

    /// Predicts one label per input row into caller-owned storage.
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError>;

    /// Predicts labels, allocating one output value per input row.
    fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        check_batch_width(self.n_features_in(), data)?;
        let mut output = vec![0; data.rows()];
        self.predict_into(data, &mut output)?;
        Ok(output)
    }
}

/// A fitted classifier that also produces a probability per class.
///
/// Implementations must order probability columns exactly as
/// [`Classifier::classes`] and must validate caller-provided output lengths
/// before writing to them.
///
/// This is a separate trait rather than defaulted methods on [`Classifier`]
/// deliberately. A default body would invert the failure it is meant to
/// prevent: a classifier that *has* probabilities but forgets to override one
/// would silently report that it has none, and every generic caller would
/// still compile against a promise that only breaks at run time. The trait
/// stays dyn-compatible, and trait upcasting means a
/// `&dyn ProbabilisticClassifier` is still accepted wherever a
/// `&dyn Classifier` is wanted.
pub trait ProbabilisticClassifier: Classifier {
    /// Predicts row-major probabilities with `classes().len()` columns.
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError>;

    /// Predicts row-major probabilities, allocating the output matrix.
    fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        check_batch_width(self.n_features_in(), data)?;
        let output_len = data.rows().checked_mul(self.classes().len()).ok_or(
            ModelError::OutputShapeOverflow {
                rows: data.rows(),
                columns: self.classes().len(),
            },
        )?;
        let mut output = vec![0.0; output_len];
        self.predict_proba_into(data, &mut output)?;
        Ok(output)
    }

    /// Predicts one requested probability column without materializing the
    /// full probability matrix.
    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError>;

    /// Predicts one requested probability column, allocating the output.
    fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        check_batch_width(self.n_features_in(), data)?;
        let mut output = vec![0.0; data.rows()];
        self.predict_class_proba_into(data, class, &mut output)?;
        Ok(output)
    }
}

/// A fitted regression estimator.
pub trait Regressor: Estimator {
    /// Predicts one value per input row into caller-owned storage.
    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [f32]) -> Result<(), ModelError>;

    /// Predicts values, allocating one output value per input row.
    fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        check_batch_width(self.n_features_in(), data)?;
        let mut output = vec![0.0; data.rows()];
        self.predict_into(data, &mut output)?;
        Ok(output)
    }
}

/// A fitted feature transformer.
///
/// Implementations write into caller-owned, row-major storage and return a
/// validated view over exactly the values they wrote. The returned matrix must
/// preserve the input row count and use [`Transformer::n_features_out`]
/// columns. This makes transformation reusable in allocation-free generic
/// pipelines while retaining an allocating convenience method.
pub trait Transformer: Estimator {
    /// Number of output features produced for each input row.
    fn n_features_out(&self) -> usize;

    /// Transforms a batch into caller-owned storage.
    ///
    /// `output.len()` must equal
    /// `data.rows() * self.n_features_out()`. Implementations validate the
    /// returned values when constructing the [`MatrixView`].
    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError>;

    /// Transforms a batch, allocating one dense output matrix.
    fn transform(&self, data: &MatrixView<'_>) -> Result<DenseMatrix, ModelError> {
        check_batch_width(self.n_features_in(), data)?;
        let columns = self.n_features_out();
        let output_len =
            data.rows()
                .checked_mul(columns)
                .ok_or(ModelError::OutputShapeOverflow {
                    rows: data.rows(),
                    columns,
                })?;
        let mut output = vec![0.0; output_len];
        let transformed = self.transform_into(data, &mut output)?;
        validate_transformed_shape(data.rows(), columns, &transformed)?;
        Ok(DenseMatrix::from_validated_parts(
            output,
            data.rows(),
            columns,
        ))
    }
}

/// Rejects a batch whose width differs from the estimator's fitted width.
///
/// Every allocating default above runs this *before* sizing its output buffer,
/// so an invalid shape fails without allocating; each `_into` primitive repeats
/// the check for callers that use it directly, and reports the same error from
/// the same two numbers, so hoisting it changes ordering rather than behaviour.
fn check_batch_width(expected: usize, data: &MatrixView<'_>) -> Result<(), ModelError> {
    if data.columns() != expected {
        return Err(ModelError::FeatureDimension {
            expected,
            actual: data.columns(),
        });
    }
    Ok(())
}

pub(crate) fn validate_transformed_shape(
    expected_rows: usize,
    expected_columns: usize,
    transformed: &MatrixView<'_>,
) -> Result<(), ModelError> {
    if transformed.rows() != expected_rows {
        return Err(ModelError::OutputLength {
            expected: expected_rows,
            actual: transformed.rows(),
        });
    }
    if transformed.columns() != expected_columns {
        return Err(ModelError::FeatureDimension {
            expected: expected_columns,
            actual: transformed.columns(),
        });
    }
    Ok(())
}

/// Access to the exact parameters retained by a fitted estimator.
///
/// This generic trait complements the object-safe estimator categories. It is
/// intended for static pipelines and reproducibility-sensitive code.
pub trait HasParams {
    /// Concrete parameter type used to fit this estimator.
    type Params;

    /// Returns the exact fitted parameter values.
    fn get_params(&self) -> &Self::Params;
}
