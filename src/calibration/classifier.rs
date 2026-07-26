//! A fitted classifier wrapped in a fitted calibration map.

use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, ModelError, ProbabilisticClassifier,
};
use crate::data::{BinaryTargets, MatrixView};

use super::{
    Calibrator, IsotonicRegression, IsotonicRegressionParams, PlattCalibrator, PlattParams,
};

/// A fitted binary classifier whose probabilities are recalibrated.
///
/// The wrapper owns an already-fitted model and an already-fitted
/// [`Calibrator`], and is itself an ordinary [`Classifier`]: it composes with
/// the scorer, cross-validation, and permutation-importance paths without any
/// of them learning that calibration exists.
///
/// # What is calibrated
///
/// The score handed to the calibrator is the wrapped model's **positive-class
/// probability**, taken through
/// [`predict_class_proba_into`](ProbabilisticClassifier::predict_class_proba_into). That is
/// the one score the [`ProbabilisticClassifier`] contract requires, which is
/// what makes this wrapper work for a model FerricML does not ship. Platt's
/// original formulation calibrates a raw decision function instead;
/// `decision_function` is an inherent method of individual estimators rather
/// than part of the object-safe classifier contract, so a wrapper generic over
/// `C: ProbabilisticClassifier` cannot reach one. A monotone remap of the
/// probability is well defined for every classifier that produces one, and is
/// what makes calibrating a forest — the case that motivates the whole feature
/// — possible at all.
///
/// # Held-out calibration is the caller's explicit choice
///
/// The calibration rows are a parameter. Nothing here reuses the wrapped
/// model's training rows implicitly, because a calibrator fitted on the rows
/// its model already memorised measures that memory rather than the model's
/// probabilities: a model that separates its training data perfectly yields a
/// calibration sample with no overlap, and the fitted map is a step, not a
/// correction.
///
/// # Labels follow the calibrated probabilities
///
/// [`predict`](Classifier::predict) is the argmax of this model's *own*
/// probabilities, not a pass-through of the wrapped model's labels. A row whose
/// calibrated probability crosses `0.5` does change label, which is the point
/// of correcting an overconfident model. A classifier whose labels disagreed
/// with its own probabilities would be a silent wrong answer.
///
/// # Ranking is preserved only by a strictly increasing calibrator
///
/// Calibration is monotone, and monotone is weaker than ranking-preserving. The
/// [`Calibrator`] contract states the three cases; the one that matters here is
/// that a Platt fit whose [`slope`](PlattCalibrator::slope) is negative is a
/// strictly *decreasing* map, and reverses every pairwise comparison — a model
/// with ROC AUC `auc` becomes one with `1.0 - auc`. That is not a fitting
/// failure: it is the maximum-likelihood answer for a calibration sample whose
/// positive rows carry a lower mean score than its negative rows, which a small
/// held-out fold can easily be. Isotonic on such a fold pools instead, which
/// loses ordering rather than inverting it, and in the extreme is constant.
///
/// The fitted map is reachable through [`calibrator`](Self::calibrator), so a
/// caller that depends on ranking checks the sign it depends on:
///
/// ```
/// # use ferricml::api::ProbabilisticClassifier;
/// # use ferricml::calibration::{CalibratedClassifier, PlattParams};
/// # use ferricml::data::{BinaryTargets, DenseMatrix};
/// # use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};
/// # let values: Vec<f32> = (0..20).map(|index| index as f32 - 10.0).collect();
/// # let labels: Vec<u8> = (0..20).map(|index| u8::from(index >= 10)).collect();
/// # let data = DenseMatrix::new(values, 20, 1)?;
/// # let labels = BinaryTargets::new(labels)?;
/// # let model = LogisticRegression::fit(&data.as_view(), &labels, LogisticRegressionParams::default())?;
/// # let holdout = DenseMatrix::new(vec![-7.5_f32, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5], 8, 1)?;
/// # let holdout_labels = BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1])?;
/// let calibrated = CalibratedClassifier::fit_platt(
///     model,
///     &holdout.as_view(),
///     &holdout_labels,
///     PlattParams::default(),
/// )?;
/// assert!(
///     calibrated.calibrator().slope() > 0.0,
///     "this calibration fold inverts the model's ranking",
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ```
/// use ferricml::api::{Classifier, ProbabilisticClassifier};
/// use ferricml::calibration::{CalibratedClassifier, PlattParams};
/// use ferricml::data::{BinaryTargets, DenseMatrix};
/// use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};
/// use ferricml::metrics::roc_auc_score;
///
/// let values: Vec<f32> = (0..20).map(|index| index as f32 - 10.0).collect();
/// let labels: Vec<u8> = (0..20).map(|index| u8::from(index >= 10)).collect();
/// let data = DenseMatrix::new(values, 20, 1)?;
/// let labels = BinaryTargets::new(labels)?;
///
/// let model = LogisticRegression::fit(
///     &data.as_view(),
///     &labels,
///     LogisticRegressionParams::default(),
/// )?;
/// let before = model.predict_proba(&data.as_view())?;
///
/// // Calibration rows are always supplied explicitly, never taken from the
/// // wrapped model's own training rows implicitly. These are held out of the
/// // fit above, which is the workflow this module is written for.
/// let holdout = DenseMatrix::new(
///     vec![-7.5_f32, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5],
///     8,
///     1,
/// )?;
/// let holdout_labels = BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1])?;
/// let calibrated = CalibratedClassifier::fit_platt(
///     model,
///     &holdout.as_view(),
///     &holdout_labels,
///     PlattParams::default(),
/// )?;
/// let after = calibrated.predict_proba(&data.as_view())?;
///
/// // The composition is an ordinary Classifier, so it reaches scoring,
/// // cross-validation and permutation importance unchanged.
/// assert_eq!(calibrated.classes(), &[0, 1]);
///
/// // This fold fitted a positive slope, so the map is strictly increasing and
/// // cannot reorder two rows: every threshold-sweeping score is unchanged.
/// // The condition is asserted rather than assumed, because it is the
/// // condition — a fold whose positive rows score below its negative rows
/// // fits a negative slope, and that map takes ROC AUC to `1.0 - auc`.
/// assert!(calibrated.calibrator().slope() > 0.0);
///
/// // Scored against labels the model does not reproduce exactly, so the two
/// // AUCs are strictly between 0.5 and 1 and could disagree if they were free
/// // to. Two of the twenty labels are flipped.
/// let noisy = BinaryTargets::new(
///     (0..20).map(|index| u8::from((index >= 10) != (index == 8 || index == 11))).collect(),
/// )?;
/// let positive_before: Vec<f32> = before.chunks(2).map(|row| row[1]).collect();
/// let positive_after: Vec<f32> = after.chunks(2).map(|row| row[1]).collect();
/// let raw = roc_auc_score(noisy.as_slice(), &positive_before)?;
/// assert!(raw > 0.5 && raw < 1.0);
/// assert_eq!(raw, roc_auc_score(noisy.as_slice(), &positive_after)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedClassifier<C, K> {
    inner: C,
    calibrator: K,
}

impl<C: ProbabilisticClassifier, K: Calibrator> CalibratedClassifier<C, K> {
    /// Composes an already-fitted model with an already-fitted calibrator.
    ///
    /// The wrapped model must be binary over classes `[0, 1]`: the calibration
    /// map is univariate, so a wider class set is
    /// [`ModelError::MulticlassOutput`] rather than a silently calibrated
    /// column.
    pub fn new(inner: C, calibrator: K) -> Result<Self, ModelError> {
        validate_binary_inner(&inner)?;
        Ok(Self { inner, calibrator })
    }

    /// Returns the wrapped fitted model.
    pub const fn inner(&self) -> &C {
        &self.inner
    }

    /// Returns the fitted calibration map.
    pub const fn calibrator(&self) -> &K {
        &self.calibrator
    }

    /// Consumes the wrapper and returns its two fitted parts.
    pub fn into_parts(self) -> (C, K) {
        (self.inner, self.calibrator)
    }

    /// Predicts labels into caller-owned label and score storage.
    ///
    /// This is the allocation-free form of [`Classifier::predict_into`], which
    /// has nowhere in its signature to put the intermediate scores and so
    /// allocates them. `scores` must hold one value per row and is left holding
    /// this model's calibrated positive-class probabilities.
    pub fn predict_into_with(
        &self,
        data: &MatrixView<'_>,
        scores: &mut [f32],
        output: &mut [u8],
    ) -> Result<(), ModelError> {
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        self.positive_probabilities_into(data, scores)?;
        for (label, &probability) in output.iter_mut().zip(scores.iter()) {
            // The same comparison `predict_proba`'s argmax would make, on the
            // same two values, so labels cannot disagree with probabilities.
            *label = u8::from(probability > 1.0 - probability);
        }
        Ok(())
    }

    /// Rejects a batch whose width differs from the wrapped model's.
    ///
    /// The wrapper's own entry points run this before allocating any scratch or
    /// output storage. The wrapped model repeats the check when it is reached,
    /// and reports the same error from the same two numbers, so this changes
    /// when a mismatch is noticed rather than what is reported.
    fn check_batch_width(&self, data: &MatrixView<'_>) -> Result<(), ModelError> {
        let expected = self.inner.n_features_in();
        if data.columns() != expected {
            return Err(ModelError::FeatureDimension {
                expected,
                actual: data.columns(),
            });
        }
        Ok(())
    }

    /// Writes the calibrated positive-class probability of each row.
    fn positive_probabilities_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.inner.predict_class_proba_into(data, 1, output)?;
        self.calibrator.calibrate_in_place(output);
        Ok(())
    }
}

impl<C: ProbabilisticClassifier> CalibratedClassifier<C, IsotonicRegression> {
    /// Fits an isotonic calibration map on caller-supplied held-out rows.
    pub fn fit_isotonic(
        inner: C,
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: IsotonicRegressionParams,
    ) -> Result<Self, ModelError> {
        let scores = calibration_scores(&inner, data, targets)?;
        let calibrator = IsotonicRegression::fit_calibration(&scores, targets, params)?;
        Ok(Self { inner, calibrator })
    }
}

impl<C: ProbabilisticClassifier> CalibratedClassifier<C, PlattCalibrator> {
    /// Fits a Platt calibration map on caller-supplied held-out rows.
    pub fn fit_platt(
        inner: C,
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: PlattParams,
    ) -> Result<Self, ModelError> {
        let scores = calibration_scores(&inner, data, targets)?;
        let calibrator = PlattCalibrator::fit(&scores, targets, params)?;
        Ok(Self { inner, calibrator })
    }

    /// Writes one raw calibrated decision score per row.
    ///
    /// This is the score whose logistic squashing is
    /// [`predict_class_proba`](ProbabilisticClassifier::predict_class_proba) for class `1`.
    /// It exists only on a Platt-calibrated model, which is exactly what
    /// [`Capabilities::decision_function`] declares.
    pub fn decision_function_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.inner.predict_class_proba_into(data, 1, output)?;
        for slot in output.iter_mut() {
            *slot = self.calibrator.decision_score(*slot);
        }
        Ok(())
    }

    /// Returns one raw calibrated decision score per row.
    pub fn decision_function(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        self.check_batch_width(data)?;
        let mut output = vec![0.0; data.rows()];
        self.decision_function_into(data, &mut output)?;
        Ok(output)
    }
}

/// Validates the calibration sample and returns the wrapped model's scores.
///
/// Every shape check happens before the score buffer is allocated, so a
/// rejected call does no work and writes nothing.
fn calibration_scores<C: ProbabilisticClassifier>(
    inner: &C,
    data: &MatrixView<'_>,
    targets: &BinaryTargets,
) -> Result<Vec<f32>, ModelError> {
    validate_binary_inner(inner)?;
    if data.rows() == 0 || data.columns() == 0 {
        return Err(ModelError::EmptyData);
    }
    if data.columns() != inner.n_features_in() {
        return Err(ModelError::FeatureDimension {
            expected: inner.n_features_in(),
            actual: data.columns(),
        });
    }
    if data.rows() != targets.len() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: targets.len(),
        });
    }
    let mut scores = vec![0.0; data.rows()];
    inner.predict_class_proba_into(data, 1, &mut scores)?;
    Ok(scores)
}

/// Rejects a wrapped model whose class set the univariate map cannot address.
fn validate_binary_inner<C: Classifier>(inner: &C) -> Result<(), ModelError> {
    match inner.classes() {
        [0, 1] => Ok(()),
        classes if classes.len() < 2 => Err(ModelError::RequiresTwoClasses),
        classes if classes.len() > 2 => Err(ModelError::MulticlassOutput {
            columns: classes.len(),
        }),
        // Two classes, but the positive one is not label `1`, so there is no
        // positive-class probability column to calibrate.
        _ => Err(ModelError::UnknownClass { class: 1 }),
    }
}

impl<C: ProbabilisticClassifier, K: Calibrator> Estimator for CalibratedClassifier<C, K> {
    fn n_features_in(&self) -> usize {
        self.inner.n_features_in()
    }
}

impl<C: ProbabilisticClassifier> HasCapabilities for CalibratedClassifier<C, IsotonicRegression> {
    /// Calibrated probabilities, and deliberately not the wrapped model's
    /// other declarations.
    ///
    /// Producing a probability is the whole point of the wrapper, so
    /// `probability` is declared here whatever the wrapped model declares.
    /// Everything else is declared away structurally rather than intersected,
    /// because an intersection would have promised an entry point that does not
    /// exist on the wrapper at all: the composition owns already-fitted parts,
    /// so it has no weighted fitting entry point whatever the wrapped model can
    /// do; it has no artifact kind; and its own `fit` takes binary targets, so
    /// it offers no multiclass entry point either. `decision_function` is the
    /// one field that varies between the two calibrators, and it is absent here
    /// — see the Platt composition below.
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_probability(true);
}

impl<C: ProbabilisticClassifier> HasCapabilities for CalibratedClassifier<C, PlattCalibrator> {
    /// A raw decision score, which the parametric calibrator genuinely has.
    ///
    /// `slope * score + intercept` is a real-valued score whose sigmoid is the
    /// calibrated probability. The isotonic composition has no such score — its
    /// map is a piecewise-linear step, not a squashed line — which is what
    /// makes this declaration vary rather than being a constant dressed up as
    /// a capability.
    const CAPABILITIES: Capabilities = Capabilities::NONE
        .with_decision_function(true)
        .with_probability(true);
}

impl<C: ProbabilisticClassifier, K: Calibrator> Classifier for CalibratedClassifier<C, K> {
    fn classes(&self) -> &[u8] {
        // Construction rejected anything other than `[0, 1]`, so this is the
        // wrapped model's own class list rather than a second copy that could
        // drift from it.
        self.inner.classes()
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        if output.len() != data.rows() {
            return Err(ModelError::OutputLength {
                expected: data.rows(),
                actual: output.len(),
            });
        }
        // This is an `_into` method, so the scratch buffer below is the only
        // allocation it makes at all. A batch it will refuse must not pay for
        // it.
        self.check_batch_width(data)?;
        let mut scores = vec![0.0; data.rows()];
        self.predict_into_with(data, &mut scores, output)
    }
}

impl<C: ProbabilisticClassifier, K: Calibrator> ProbabilisticClassifier
    for CalibratedClassifier<C, K>
{
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        let rows = data.rows();
        let expected = rows
            .checked_mul(2)
            .ok_or(ModelError::OutputShapeOverflow { rows, columns: 2 })?;
        if output.len() != expected {
            return Err(ModelError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        // Park the calibrated positive probabilities in the upper half, then
        // expand forwards into both columns. Row `i` writes slots `2i` and
        // `2i + 1` and every still-unread source sits at `rows + j` for
        // `j > i`, which is strictly above `2i + 1` while `i < rows`. So the
        // whole matrix is produced without a second buffer.
        self.positive_probabilities_into(data, &mut output[rows..])?;
        for index in 0..rows {
            let positive = output[rows + index];
            output[2 * index] = 1.0 - positive;
            output[2 * index + 1] = positive;
        }
        Ok(())
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        // Width before class: the shape of the input is validated before the
        // content of the request. The wrapped model repeats the check when it
        // is reached, so this changes when a mismatch is noticed rather than
        // what is reported — but it is what makes this primitive report the
        // same error its allocating partner does.
        self.check_batch_width(data)?;
        if class > 1 {
            return Err(ModelError::UnknownClass { class });
        }
        self.positive_probabilities_into(data, output)?;
        if class == 0 {
            for slot in output.iter_mut() {
                *slot = 1.0 - *slot;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::dummy::{DummyClassifier, DummyClassifierParams};

    /// A classifier whose positive probability is a fixed function of column 0.
    ///
    /// Wrapping a stub rather than a shipped estimator is the point: the
    /// wrapper is generic over the public contract, so this file proves it
    /// against the contract and nothing else.
    #[derive(Clone, Debug, PartialEq)]
    struct Overconfident {
        classes: Vec<u8>,
    }

    impl Overconfident {
        fn binary() -> Self {
            Self {
                classes: vec![0, 1],
            }
        }

        fn probability(row: &[f32]) -> f32 {
            (row[0] * 4.0 - 2.0).clamp(0.0, 1.0)
        }
    }

    impl Estimator for Overconfident {
        fn n_features_in(&self) -> usize {
            1
        }
    }

    impl Classifier for Overconfident {
        fn classes(&self) -> &[u8] {
            &self.classes
        }

        fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
            self.validate(data, output.len())?;
            for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
                *slot = u8::from(Self::probability(row) > 0.5);
            }
            Ok(())
        }
    }

    impl ProbabilisticClassifier for Overconfident {
        fn predict_proba_into(
            &self,
            data: &MatrixView<'_>,
            output: &mut [f32],
        ) -> Result<(), ModelError> {
            self.validate(data, output.len() / 2)?;
            if output.len() != data.rows() * 2 {
                return Err(ModelError::OutputLength {
                    expected: data.rows() * 2,
                    actual: output.len(),
                });
            }
            for (slots, row) in output.chunks_exact_mut(2).zip(data.iter_rows()) {
                let positive = Self::probability(row);
                slots[0] = 1.0 - positive;
                slots[1] = positive;
            }
            Ok(())
        }

        fn predict_class_proba_into(
            &self,
            data: &MatrixView<'_>,
            class: u8,
            output: &mut [f32],
        ) -> Result<(), ModelError> {
            self.validate(data, output.len())?;
            if !self.classes.contains(&class) {
                return Err(ModelError::UnknownClass { class });
            }
            for (slot, row) in output.iter_mut().zip(data.iter_rows()) {
                let positive = Self::probability(row);
                *slot = if class == 1 { positive } else { 1.0 - positive };
            }
            Ok(())
        }
    }

    impl Overconfident {
        fn validate(&self, data: &MatrixView<'_>, rows: usize) -> Result<(), ModelError> {
            if data.columns() != self.n_features_in() {
                return Err(ModelError::FeatureDimension {
                    expected: self.n_features_in(),
                    actual: data.columns(),
                });
            }
            if rows != data.rows() {
                return Err(ModelError::OutputLength {
                    expected: data.rows(),
                    actual: rows,
                });
            }
            Ok(())
        }
    }

    fn sample() -> (DenseMatrix, BinaryTargets) {
        let values: Vec<f32> = (0..24).map(|step| step as f32 / 23.0).collect();
        let labels: Vec<u8> = (0..24).map(|step| u8::from(step % 3 != 0)).collect();
        (
            DenseMatrix::new(values, 24, 1).unwrap(),
            BinaryTargets::new(labels).unwrap(),
        )
    }

    #[test]
    fn calibrated_probabilities_form_valid_rows_and_agree_with_their_columns() {
        let (data, labels) = sample();
        let model = CalibratedClassifier::fit_isotonic(
            Overconfident::binary(),
            &data.as_view(),
            &labels,
            IsotonicRegressionParams,
        )
        .unwrap();
        assert_eq!(model.classes(), &[0, 1]);
        assert_eq!(model.n_features_in(), 1);

        let matrix = model.predict_proba(&data.as_view()).unwrap();
        let positive = model.predict_class_proba(&data.as_view(), 1).unwrap();
        let negative = model.predict_class_proba(&data.as_view(), 0).unwrap();
        let predicted = model.predict(&data.as_view()).unwrap();
        assert_eq!(matrix.len(), data.rows() * 2);
        for (index, row) in matrix.chunks_exact(2).enumerate() {
            assert_eq!(row[1], positive[index]);
            assert_eq!(row[0], negative[index]);
            assert!((row[0] + row[1] - 1.0).abs() <= 1.0e-6, "row {index}");
            assert!((0.0..=1.0).contains(&row[1]));
            let expected = u8::from(row[1] > row[0]);
            assert_eq!(predicted[index], expected, "row {index}");
        }
    }

    #[test]
    fn the_allocation_free_paths_agree_with_the_allocating_ones() {
        let (data, labels) = sample();
        let model = CalibratedClassifier::fit_platt(
            Overconfident::binary(),
            &data.as_view(),
            &labels,
            PlattParams::default(),
        )
        .unwrap();
        let view = data.as_view();

        let mut matrix = vec![f32::MAX; data.rows() * 2];
        model.predict_proba_into(&view, &mut matrix).unwrap();
        assert_eq!(matrix, model.predict_proba(&view).unwrap());

        let mut labels_into = vec![u8::MAX; data.rows()];
        let mut scores = vec![f32::MAX; data.rows()];
        model
            .predict_into_with(&view, &mut scores, &mut labels_into)
            .unwrap();
        assert_eq!(labels_into, model.predict(&view).unwrap());
        assert_eq!(scores, model.predict_class_proba(&view, 1).unwrap());

        let mut decisions = vec![f32::MAX; data.rows()];
        model.decision_function_into(&view, &mut decisions).unwrap();
        assert_eq!(decisions, model.decision_function(&view).unwrap());
        for (&decision, &probability) in decisions.iter().zip(&scores) {
            assert_eq!(crate::numeric::sigmoid_f32(decision), probability);
        }
    }

    #[test]
    fn every_prediction_path_validates_before_writing() {
        let (data, labels) = sample();
        let model = CalibratedClassifier::fit_isotonic(
            Overconfident::binary(),
            &data.as_view(),
            &labels,
            IsotonicRegressionParams,
        )
        .unwrap();
        let wide = DenseMatrix::new(vec![0.5; data.rows() * 2], data.rows(), 2).unwrap();

        let mut sentinel = vec![f32::MAX; data.rows() * 2];
        assert_eq!(
            model.predict_proba_into(&wide.as_view(), &mut sentinel),
            Err(ModelError::FeatureDimension {
                expected: 1,
                actual: 2,
            })
        );
        assert!(sentinel.iter().all(|value| *value == f32::MAX));
        assert_eq!(
            model.predict_proba_into(&data.as_view(), &mut sentinel[..3]),
            Err(ModelError::OutputLength {
                expected: 48,
                actual: 3,
            })
        );
        assert!(sentinel.iter().all(|value| *value == f32::MAX));

        let mut labels_sentinel = vec![u8::MAX; data.rows()];
        assert_eq!(
            model.predict_into(&data.as_view(), &mut labels_sentinel[..2]),
            Err(ModelError::OutputLength {
                expected: 24,
                actual: 2,
            })
        );
        assert!(labels_sentinel.iter().all(|value| *value == u8::MAX));
        assert_eq!(
            model.predict_class_proba_into(&data.as_view(), 2, &mut sentinel[..data.rows()]),
            Err(ModelError::UnknownClass { class: 2 })
        );
        assert!(sentinel.iter().all(|value| *value == f32::MAX));
    }

    #[test]
    fn a_wrapped_model_the_map_cannot_address_is_rejected_at_construction() {
        let (data, labels) = sample();
        let single = DummyClassifier::fit(
            &data.as_view(),
            &BinaryTargets::new(vec![1; data.rows()]).unwrap(),
            DummyClassifierParams,
        )
        .unwrap();
        assert_eq!(single.classes(), &[1]);
        assert_eq!(
            CalibratedClassifier::fit_isotonic(
                single,
                &data.as_view(),
                &labels,
                IsotonicRegressionParams
            )
            .unwrap_err(),
            ModelError::RequiresTwoClasses
        );

        let wide = Overconfident {
            classes: vec![0, 1, 2],
        };
        assert_eq!(
            CalibratedClassifier::fit_isotonic(
                wide,
                &data.as_view(),
                &labels,
                IsotonicRegressionParams
            )
            .unwrap_err(),
            ModelError::MulticlassOutput { columns: 3 }
        );

        let relabelled = Overconfident {
            classes: vec![3, 7],
        };
        assert_eq!(
            CalibratedClassifier::fit_isotonic(
                relabelled,
                &data.as_view(),
                &labels,
                IsotonicRegressionParams
            )
            .unwrap_err(),
            ModelError::UnknownClass { class: 1 }
        );
    }

    #[test]
    fn fitting_validates_the_calibration_sample_before_predicting_on_it() {
        let (data, labels) = sample();
        let wide = DenseMatrix::new(vec![0.5; data.rows() * 2], data.rows(), 2).unwrap();
        assert_eq!(
            CalibratedClassifier::fit_isotonic(
                Overconfident::binary(),
                &wide.as_view(),
                &labels,
                IsotonicRegressionParams
            )
            .unwrap_err(),
            ModelError::FeatureDimension {
                expected: 1,
                actual: 2,
            }
        );
        assert_eq!(
            CalibratedClassifier::fit_isotonic(
                Overconfident::binary(),
                &data.as_view(),
                &BinaryTargets::new(vec![0, 1]).unwrap(),
                IsotonicRegressionParams,
            )
            .unwrap_err(),
            ModelError::TargetLength {
                rows: 24,
                targets: 2,
            }
        );
        assert_eq!(
            CalibratedClassifier::fit_isotonic(
                Overconfident::binary(),
                &data.as_view(),
                &BinaryTargets::new(vec![0; data.rows()]).unwrap(),
                IsotonicRegressionParams,
            )
            .unwrap_err(),
            ModelError::RequiresTwoClasses
        );
    }

    #[test]
    fn refitting_the_same_calibration_sample_reproduces_the_same_model() {
        let (data, labels) = sample();
        for _ in 0..2 {
            assert_eq!(
                CalibratedClassifier::fit_isotonic(
                    Overconfident::binary(),
                    &data.as_view(),
                    &labels,
                    IsotonicRegressionParams,
                )
                .unwrap(),
                CalibratedClassifier::fit_isotonic(
                    Overconfident::binary(),
                    &data.as_view(),
                    &labels,
                    IsotonicRegressionParams,
                )
                .unwrap()
            );
            assert_eq!(
                CalibratedClassifier::fit_platt(
                    Overconfident::binary(),
                    &data.as_view(),
                    &labels,
                    PlattParams::default()
                )
                .unwrap(),
                CalibratedClassifier::fit_platt(
                    Overconfident::binary(),
                    &data.as_view(),
                    &labels,
                    PlattParams::default()
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn the_composition_declares_only_what_it_really_offers() {
        assert_eq!(
            <CalibratedClassifier<Overconfident, IsotonicRegression> as HasCapabilities>::CAPABILITIES,
            Capabilities::NONE.with_probability(true)
        );
        assert_eq!(
            <CalibratedClassifier<Overconfident, PlattCalibrator> as HasCapabilities>::CAPABILITIES,
            Capabilities::NONE
                .with_decision_function(true)
                .with_probability(true)
        );
    }
}
