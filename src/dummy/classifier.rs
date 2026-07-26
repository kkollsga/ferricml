use crate::api::{
    Capabilities, Classifier, Estimator, HasCapabilities, HasParams, ModelError,
    ProbabilisticClassifier, validate_scalar_row,
};
use crate::data::{BinaryTargets, MatrixView};

/// Parameters for [`DummyClassifier`].
///
/// The majority-class baseline has nothing to tune. This type exists so the
/// baseline is fitted exactly like every other FerricML estimator, and so a
/// future strategy choice can be added without changing the `fit` signature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DummyClassifierParams;

/// A classifier that ignores its features and predicts the majority class.
///
/// Probabilities are the observed class frequencies, so they are identical for
/// every row and sum to one. This is the quality floor a real classifier has to
/// beat: matching it means the features contributed nothing.
///
/// ```
/// use ferricml::api::{Classifier, ProbabilisticClassifier};
/// use ferricml::data::{BinaryTargets, DenseMatrix};
/// use ferricml::dummy::{DummyClassifier, DummyClassifierParams};
///
/// // Three of class 0, one of class 1.
/// let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
/// let labels = BinaryTargets::new(vec![0, 0, 0, 1])?;
///
/// let baseline = DummyClassifier::fit(
///     &data.as_view(),
///     &labels,
///     DummyClassifierParams::default(),
/// )?;
///
/// // The majority class, for every row, whatever the features say.
/// assert_eq!(baseline.predict(&data.as_view())?, vec![0, 0, 0, 0]);
///
/// // Probabilities are the observed frequencies, identical on every row.
/// let probabilities = baseline.predict_proba(&data.as_view())?;
/// assert_eq!(&probabilities[0..2], &[0.75, 0.25]);
/// assert_eq!(&probabilities[6..8], &[0.75, 0.25]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DummyClassifier {
    n_features_in: usize,
    params: DummyClassifierParams,
    classes: Vec<u8>,
    priors: Vec<f32>,
    majority: u8,
}

impl DummyClassifier {
    /// Fits the class frequencies observed in the training targets.
    ///
    /// Ties are resolved towards the smaller class label, matching the tie
    /// rule every other FerricML classifier uses.
    pub fn fit(
        data: &MatrixView<'_>,
        targets: &BinaryTargets,
        params: DummyClassifierParams,
    ) -> Result<Self, ModelError> {
        if data.rows() == 0 || data.columns() == 0 {
            return Err(ModelError::EmptyData);
        }
        if targets.len() != data.rows() {
            return Err(ModelError::TargetLength {
                rows: data.rows(),
                targets: targets.len(),
            });
        }

        // `BinaryTargets` admits only `0` and `1`, so the label is the counter
        // index and there is no third case to reject first.
        let mut counts = [0_usize; 2];
        for &value in targets.as_slice() {
            counts[usize::from(value)] += 1;
        }

        let classes: Vec<u8> = (0..2_u8)
            .filter(|&label| counts[usize::from(label)] > 0)
            .collect();
        let total = targets.len() as f64;
        let priors: Vec<f32> = classes
            .iter()
            .map(|&label| (counts[usize::from(label)] as f64 / total) as f32)
            .collect();
        let majority = Self::majority_label(&classes, &counts);

        Ok(Self {
            n_features_in: data.columns(),
            params,
            classes,
            priors,
            majority,
        })
    }

    /// Scans ascending and keeps only a strictly larger **count**, so an exact
    /// tie resolves towards the smaller label.
    ///
    /// The comparison is on the counts rather than on [`Self::class_priors`]
    /// because a prior is an `f32` view of a ratio of `usize` counts, and
    /// narrowing is not order-preserving: two counts that differ by one collapse
    /// onto one `f32` as soon as their relative gap falls below half an ulp.
    /// The first such pair is `(16_777_216, 16_777_217)`, where both ratios
    /// round to exactly `0.5` — the strict `>` then fails and the *minority*
    /// class is returned. Counts are exact, so the comparison on them is exact.
    ///
    /// One consequence is deliberate: at such a total the predicted label is no
    /// longer the first maximum of the reported probabilities, because the
    /// reported probabilities are equal where the counts are not. The
    /// prediction follows the counts; [`Self::class_priors`] says so.
    fn majority_label(classes: &[u8], counts: &[usize; 2]) -> u8 {
        let mut best = 0;
        for index in 1..classes.len() {
            if counts[usize::from(classes[index])] > counts[usize::from(classes[best])] {
                best = index;
            }
        }
        classes[best]
    }

    /// Returns the feature width required by this model.
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns sorted class labels observed during fitting.
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    /// Returns the exact fitted parameters.
    pub const fn get_params(&self) -> &DummyClassifierParams {
        &self.params
    }

    /// Returns the observed class frequencies, ordered like [`Self::classes`].
    ///
    /// These are `f32` narrowings of exact `usize` ratios, so two frequencies
    /// can be equal here while the counts behind them are not — from a training
    /// set of `33_554_433` rows onwards, a one-row majority is invisible at this
    /// precision. [`Self::predict`] compares the counts, so it still reports the
    /// true majority; only the reported frequencies lose the distinction.
    pub fn class_priors(&self) -> &[f32] {
        &self.priors
    }

    /// Predicts the majority class for one validated row.
    pub fn predict_one(&self, row: &[f32]) -> Result<u8, ModelError> {
        validate_scalar_row(row, self.n_features_in)?;
        Ok(self.majority)
    }

    /// Predicts one label per row, allocating the output.
    pub fn predict(&self, data: &MatrixView<'_>) -> Result<Vec<u8>, ModelError> {
        <Self as Classifier>::predict(self, data)
    }

    /// Predicts one label per row into caller-owned storage.
    pub fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        <Self as Classifier>::predict_into(self, data, output)
    }

    /// Predicts row-major probabilities, allocating the output.
    pub fn predict_proba(&self, data: &MatrixView<'_>) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba(self, data)
    }

    /// Predicts row-major probabilities into caller-owned storage.
    pub fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as ProbabilisticClassifier>::predict_proba_into(self, data, output)
    }

    /// Predicts one requested probability column, allocating the output.
    pub fn predict_class_proba(
        &self,
        data: &MatrixView<'_>,
        class: u8,
    ) -> Result<Vec<f32>, ModelError> {
        <Self as ProbabilisticClassifier>::predict_class_proba(self, data, class)
    }

    /// Predicts one requested probability column into caller-owned storage.
    pub fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        <Self as ProbabilisticClassifier>::predict_class_proba_into(self, data, class, output)
    }

    fn validate_batch(&self, data: &MatrixView<'_>) -> Result<(), ModelError> {
        if data.columns() != self.n_features_in {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in,
                actual: data.columns(),
            });
        }
        Ok(())
    }

    fn validate_output(expected: usize, actual: usize) -> Result<(), ModelError> {
        if expected != actual {
            return Err(ModelError::OutputLength { expected, actual });
        }
        Ok(())
    }
}

impl Estimator for DummyClassifier {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasParams for DummyClassifier {
    type Params = DummyClassifierParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

/// Declares probabilities and nothing else: a baseline is refitted rather than
/// persisted, and has no weighted entry point. The probabilities are the fitted
/// class frequencies, which is a real distribution rather than a squashed
/// score, so the declaration is earned rather than a formality.
impl HasCapabilities for DummyClassifier {
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_probability(true);
}

impl Classifier for DummyClassifier {
    fn classes(&self) -> &[u8] {
        &self.classes
    }

    fn predict_into(&self, data: &MatrixView<'_>, output: &mut [u8]) -> Result<(), ModelError> {
        self.validate_batch(data)?;
        Self::validate_output(data.rows(), output.len())?;
        output.fill(self.majority);
        Ok(())
    }
}

impl ProbabilisticClassifier for DummyClassifier {
    fn predict_proba_into(
        &self,
        data: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.validate_batch(data)?;
        let expected =
            data.rows()
                .checked_mul(self.classes.len())
                .ok_or(ModelError::OutputShapeOverflow {
                    rows: data.rows(),
                    columns: self.classes.len(),
                })?;
        Self::validate_output(expected, output.len())?;
        for row in output.chunks_exact_mut(self.classes.len()) {
            row.copy_from_slice(&self.priors);
        }
        Ok(())
    }

    fn predict_class_proba_into(
        &self,
        data: &MatrixView<'_>,
        class: u8,
        output: &mut [f32],
    ) -> Result<(), ModelError> {
        self.validate_batch(data)?;
        let column = self
            .classes
            .iter()
            .position(|&label| label == class)
            .ok_or(ModelError::UnknownClass { class })?;
        Self::validate_output(data.rows(), output.len())?;
        output.fill(self.priors[column]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn data() -> DenseMatrix {
        DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3, 2).unwrap()
    }

    #[test]
    fn predicts_the_majority_class_and_its_observed_frequencies() {
        let model = DummyClassifier::fit(
            &data().as_view(),
            &BinaryTargets::new(vec![0, 1, 1]).unwrap(),
            DummyClassifierParams,
        )
        .unwrap();

        assert_eq!(model.classes(), &[0, 1]);
        assert_eq!(model.class_priors(), &[1.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(model.predict(&data().as_view()).unwrap(), vec![1; 3]);
        assert_eq!(
            model.predict_proba(&data().as_view()).unwrap(),
            vec![
                1.0 / 3.0,
                2.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0
            ]
        );
        assert_eq!(model.predict_one(&[9.0, 9.0]).unwrap(), 1);
    }

    #[test]
    fn an_exact_tie_selects_the_smaller_class() {
        let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 2, 2).unwrap();
        let model = DummyClassifier::fit(
            &data.as_view(),
            &BinaryTargets::new(vec![0, 1]).unwrap(),
            DummyClassifierParams,
        )
        .unwrap();

        assert_eq!(model.class_priors(), &[0.5, 0.5]);
        assert_eq!(model.predict(&data.as_view()).unwrap(), vec![0; 2]);
    }

    /// The smallest training set on which the majority is invisible in `f32`.
    ///
    /// `16_777_216 / 33_554_433` and `16_777_217 / 33_554_433` both round to
    /// exactly `0.5` — bit pattern `0x3f00_0000` for each — so a comparison on
    /// the reported frequencies sees a tie, resolves it towards the smaller
    /// label, and predicts the class that occurs *less* often. Searched rather
    /// than guessed: at every smaller power-of-two total, down to `1_048_577`,
    /// the same one-row majority is still strictly larger as an `f32`, so this
    /// pair is the first collapse and not merely one of them.
    ///
    /// The row count is what the defect costs to state; the test is cheap
    /// anyway, because the features are one constant column a baseline ignores.
    #[test]
    fn the_majority_is_predicted_where_the_two_frequencies_round_to_one_f32() {
        let minority = 16_777_216_usize;
        let majority = minority + 1;
        let rows = minority + majority;

        let features = vec![0.0_f32; rows];
        let view = MatrixView::new(&features, rows, 1).unwrap();
        let mut labels = vec![0_u8; rows];
        labels[minority..].fill(1);
        let model = DummyClassifier::fit(
            &view,
            &BinaryTargets::new(labels).unwrap(),
            DummyClassifierParams,
        )
        .unwrap();

        // The premise: the two frequencies really are one `f32`, so a
        // comparison on them cannot see the majority. Asserted so that a future
        // change to a wider prior turns this test into a statement about a
        // condition that no longer holds rather than into a silent pass.
        assert_eq!(model.class_priors(), &[0.5, 0.5]);
        assert_eq!(
            model.class_priors()[0].to_bits(),
            model.class_priors()[1].to_bits()
        );

        // Class `1` occurs once more than class `0`, so it is the majority.
        assert_eq!(model.predict_one(&[0.0]).unwrap(), 1);
        assert_eq!(
            model
                .predict(&MatrixView::new(&features[..3], 3, 1).unwrap())
                .unwrap(),
            vec![1; 3]
        );
    }

    #[test]
    fn a_single_observed_class_uses_one_probability_column() {
        let model = DummyClassifier::fit(
            &data().as_view(),
            &BinaryTargets::new(vec![1, 1, 1]).unwrap(),
            DummyClassifierParams,
        )
        .unwrap();

        assert_eq!(model.classes(), &[1]);
        assert_eq!(
            model.predict_proba(&data().as_view()).unwrap(),
            vec![1.0; 3]
        );
        assert_eq!(
            model.predict_class_proba(&data().as_view(), 0).unwrap_err(),
            ModelError::UnknownClass { class: 0 }
        );
    }

    #[test]
    fn fitting_rejects_mismatched_targets_before_any_work() {
        assert_eq!(
            DummyClassifier::fit(
                &data().as_view(),
                &BinaryTargets::new(vec![0, 1]).unwrap(),
                DummyClassifierParams,
            )
            .unwrap_err(),
            ModelError::TargetLength {
                rows: 3,
                targets: 2
            }
        );
    }
}
