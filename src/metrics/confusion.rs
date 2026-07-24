//! The label-set confusion matrix every averaged classification score reads.

use super::{Average, Averaging, MetricError, ZeroDivision, validate_lengths};

/// One class's contribution to a [`ConfusionMatrix`].
///
/// These four counts are what every per-class classification score is built
/// from, so a metric that needs a new combination of them needs no new access
/// to the matrix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassCounts {
    true_positives: u64,
    false_positives: u64,
    false_negatives: u64,
}

impl ClassCounts {
    /// Rows of this class predicted as this class.
    pub const fn true_positives(self) -> u64 {
        self.true_positives
    }

    /// Rows of another class predicted as this class.
    pub const fn false_positives(self) -> u64 {
        self.false_positives
    }

    /// Rows of this class predicted as another class.
    pub const fn false_negatives(self) -> u64 {
        self.false_negatives
    }

    /// True rows carrying this class.
    pub const fn support(self) -> u64 {
        self.true_positives + self.false_negatives
    }

    /// Rows predicted as this class.
    pub const fn predicted(self) -> u64 {
        self.true_positives + self.false_positives
    }
}

/// Counts for a classification result over any observed label set.
///
/// Rows are expected labels and columns are predicted labels, both ordered by
/// [`ConfusionMatrix::labels`] — the sorted union of the labels observed in
/// either input. Every averaged score in FerricML is derived from this one
/// validated pass over the data, so binary and multiclass evaluation share a
/// single definition of what is being counted.
///
/// ```
/// use ferricml::metrics::{Average, ConfusionMatrix};
///
/// let matrix = ConfusionMatrix::new(&[0, 1, 2, 2], &[0, 1, 2, 1])?;
/// assert_eq!(matrix.labels(), &[0, 1, 2]);
/// // Single-label micro-averaging pools every class, so it equals accuracy.
/// assert_eq!(matrix.precision(Average::Micro), Ok(matrix.accuracy()));
/// # Ok::<(), ferricml::metrics::MetricError>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfusionMatrix {
    labels: Vec<u8>,
    counts: Vec<u64>,
    total: u64,
}

impl ConfusionMatrix {
    /// Counts one classification result over the observed label set.
    pub fn new(expected: &[u8], predicted: &[u8]) -> Result<Self, MetricError> {
        validate_lengths(expected.len(), predicted.len())?;

        let mut observed = [false; 256];
        for &label in expected.iter().chain(predicted) {
            observed[label as usize] = true;
        }
        let labels = (0..=u8::MAX)
            .filter(|&label| observed[label as usize])
            .collect::<Vec<_>>();
        let mut position = [0_usize; 256];
        for (index, &label) in labels.iter().enumerate() {
            position[label as usize] = index;
        }

        let mut counts = vec![0_u64; labels.len() * labels.len()];
        for (&expected, &predicted) in expected.iter().zip(predicted) {
            let row = position[expected as usize];
            let column = position[predicted as usize];
            counts[row * labels.len() + column] += 1;
        }
        Ok(Self {
            labels,
            counts,
            total: expected.len() as u64,
        })
    }

    /// Sorted labels observed in either input, in row and column order.
    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    /// Number of observed labels.
    pub fn n_labels(&self) -> usize {
        self.labels.len()
    }

    /// Row-major counts with expected labels as rows and predicted as columns.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Total counted observations.
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// The four counts for one label position, or `None` when out of range.
    pub fn class_counts(&self, index: usize) -> Option<ClassCounts> {
        if index >= self.labels.len() {
            return None;
        }
        Some(self.counts_at(index))
    }

    /// Fraction of rows whose predicted label equals the expected label.
    ///
    /// The matrix cannot be empty, so accuracy is always defined.
    pub fn accuracy(&self) -> f64 {
        let correct = (0..self.labels.len())
            .map(|index| self.counts[index * self.labels.len() + index])
            .sum::<u64>();
        correct as f64 / self.total as f64
    }

    /// Positive predictive value, combined the requested way.
    pub fn precision(&self, averaging: impl Into<Averaging>) -> Result<f64, MetricError> {
        self.average(averaging.into(), Score::Precision)
    }

    /// True-positive rate, combined the requested way.
    pub fn recall(&self, averaging: impl Into<Averaging>) -> Result<f64, MetricError> {
        self.average(averaging.into(), Score::Recall)
    }

    /// Harmonic mean of precision and recall, combined the requested way.
    pub fn f1(&self, averaging: impl Into<Averaging>) -> Result<f64, MetricError> {
        self.fbeta(1.0, averaging)
    }

    /// Weighted harmonic mean of precision and recall, combined the requested
    /// way.
    ///
    /// `beta` weighs recall over precision: `beta` above one favors recall,
    /// below one favors precision. It must be finite and strictly positive.
    pub fn fbeta(&self, beta: f64, averaging: impl Into<Averaging>) -> Result<f64, MetricError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(MetricError::InvalidBeta);
        }
        self.average(
            averaging.into(),
            Score::FBeta {
                squared: beta * beta,
            },
        )
    }

    fn counts_at(&self, index: usize) -> ClassCounts {
        let labels = self.labels.len();
        let true_positives = self.counts[index * labels + index];
        let row = self.counts[index * labels..(index + 1) * labels]
            .iter()
            .sum::<u64>();
        let column = (0..labels)
            .map(|other| self.counts[other * labels + index])
            .sum::<u64>();
        ClassCounts {
            true_positives,
            false_positives: column - true_positives,
            false_negatives: row - true_positives,
        }
    }

    fn average(&self, averaging: Averaging, score: Score) -> Result<f64, MetricError> {
        match averaging.average() {
            Average::Binary => {
                if self.labels.last().is_some_and(|&label| label > 1) {
                    return Err(MetricError::NotBinary {
                        labels: self.labels.len(),
                    });
                }
                let positive = self.labels.iter().position(|&label| label == 1);
                let counts = positive.map_or(ClassCounts::default(), |index| self.counts_at(index));
                score.of(counts).ok_or(MetricError::Undefined)
            }
            Average::Micro => {
                let pooled =
                    (0..self.labels.len()).fold(ClassCounts::default(), |mut sum, index| {
                        let counts = self.counts_at(index);
                        sum.true_positives += counts.true_positives;
                        sum.false_positives += counts.false_positives;
                        sum.false_negatives += counts.false_negatives;
                        sum
                    });
                score.of(pooled).ok_or(MetricError::Undefined)
            }
            Average::Macro => self.combine(score, averaging.zero_division(), |_| Some(1.0)),
            Average::Weighted => self.combine(score, averaging.zero_division(), |counts| {
                // A class with no true rows carries no weight, so it never
                // reaches the zero-division policy at all.
                (counts.support() > 0).then(|| counts.support() as f64)
            }),
        }
    }

    /// Combines per-class scores under one weighting and one empty-denominator
    /// policy, in ascending label order so the sum is deterministic.
    fn combine(
        &self,
        score: Score,
        zero_division: ZeroDivision,
        weight: impl Fn(ClassCounts) -> Option<f64>,
    ) -> Result<f64, MetricError> {
        let mut total = 0.0_f64;
        let mut divisor = 0.0_f64;
        for index in 0..self.labels.len() {
            let counts = self.counts_at(index);
            let Some(weight) = weight(counts) else {
                continue;
            };
            let value = match score.of(counts) {
                Some(value) => value,
                None => match zero_division {
                    ZeroDivision::Error => return Err(MetricError::Undefined),
                    ZeroDivision::Zero => 0.0,
                    ZeroDivision::Skip => continue,
                },
            };
            total += weight * value;
            divisor += weight;
        }
        if divisor == 0.0 {
            return Err(MetricError::Undefined);
        }
        Ok(total / divisor)
    }
}

/// A per-class score expressed as one ratio over [`ClassCounts`].
#[derive(Clone, Copy, Debug)]
enum Score {
    Precision,
    Recall,
    FBeta { squared: f64 },
}

impl Score {
    /// Returns the score, or `None` when its denominator is empty.
    fn of(self, counts: ClassCounts) -> Option<f64> {
        let (numerator, denominator) = match self {
            Self::Precision => (counts.true_positives() as f64, counts.predicted() as f64),
            Self::Recall => (counts.true_positives() as f64, counts.support() as f64),
            Self::FBeta { squared } => {
                let scale = 1.0 + squared;
                (
                    scale * counts.true_positives() as f64,
                    scale * counts.true_positives() as f64
                        + squared * counts.false_negatives() as f64
                        + counts.false_positives() as f64,
                )
            }
        };
        (denominator > 0.0).then(|| numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{f1_score, precision_score, recall_score};

    const THREE_CLASS_EXPECTED: [u8; 8] = [0, 1, 2, 2, 1, 0, 1, 2];
    const THREE_CLASS_PREDICTED: [u8; 8] = [0, 2, 2, 2, 1, 0, 0, 1];

    fn three_class() -> ConfusionMatrix {
        ConfusionMatrix::new(&THREE_CLASS_EXPECTED, &THREE_CLASS_PREDICTED).unwrap()
    }

    fn assert_near(actual: Result<f64, MetricError>, expected: f64) {
        let actual = actual.expect("score is defined");
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn the_matrix_counts_the_observed_label_union_in_sorted_order() {
        let matrix = three_class();
        assert_eq!(matrix.labels(), &[0, 1, 2]);
        assert_eq!(matrix.n_labels(), 3);
        assert_eq!(matrix.counts(), &[2, 0, 0, 1, 1, 1, 0, 1, 2]);
        assert_eq!(matrix.total(), 8);
        assert_eq!(matrix.counts().iter().sum::<u64>(), matrix.total());

        let class_one = matrix.class_counts(1).unwrap();
        assert_eq!(class_one.true_positives(), 1);
        assert_eq!(class_one.false_positives(), 1);
        assert_eq!(class_one.false_negatives(), 2);
        assert_eq!(class_one.support(), 3);
        assert_eq!(class_one.predicted(), 2);
        assert_eq!(matrix.class_counts(3), None);

        // A label absent from both inputs is absent from the matrix.
        let sparse = ConfusionMatrix::new(&[0, 7], &[7, 7]).unwrap();
        assert_eq!(sparse.labels(), &[0, 7]);
        assert_eq!(sparse.counts(), &[0, 1, 0, 1]);
    }

    #[test]
    fn micro_averaging_equals_accuracy_for_single_label_predictions() {
        for (expected, predicted) in [
            (&THREE_CLASS_EXPECTED[..], &THREE_CLASS_PREDICTED[..]),
            (&[0, 1, 2][..], &[0, 1, 1][..]),
            (&[0, 0, 1][..], &[0, 2, 1][..]),
            (&[3, 3, 3][..], &[3, 3, 3][..]),
            (&[0, 1][..], &[1, 0][..]),
        ] {
            let matrix = ConfusionMatrix::new(expected, predicted).unwrap();
            let accuracy = matrix.accuracy();
            assert_eq!(matrix.precision(Average::Micro), Ok(accuracy));
            assert_eq!(matrix.recall(Average::Micro), Ok(accuracy));
            assert_eq!(matrix.f1(Average::Micro), Ok(accuracy));
            assert_eq!(matrix.fbeta(2.0, Average::Micro), Ok(accuracy));
            assert_eq!(matrix.fbeta(0.5, Average::Micro), Ok(accuracy));
        }
    }

    #[test]
    fn binary_averaging_reproduces_the_binary_metrics_exactly() {
        for (expected, predicted) in [
            (&[0, 0, 1, 1][..], &[0, 1, 1, 1][..]),
            (&[0, 0, 0, 1, 1, 1][..], &[0, 1, 0, 1, 0, 1][..]),
            (&[1, 1][..], &[0, 1][..]),
            (&[0, 0][..], &[0, 1][..]),
            (&[0, 0][..], &[0, 0][..]),
        ] {
            let matrix = ConfusionMatrix::new(expected, predicted).unwrap();
            assert_eq!(
                matrix.precision(Average::Binary),
                precision_score(expected, predicted)
            );
            assert_eq!(
                matrix.recall(Average::Binary),
                recall_score(expected, predicted)
            );
            assert_eq!(matrix.f1(Average::Binary), f1_score(expected, predicted));
            assert_eq!(matrix.accuracy(), matrix.precision(Average::Micro).unwrap());
        }
    }

    #[test]
    fn a_wider_label_set_rejects_binary_averaging_instead_of_scoring_one_class() {
        let matrix = three_class();
        assert_eq!(
            matrix.precision(Average::Binary),
            Err(MetricError::NotBinary { labels: 3 })
        );
        assert_eq!(
            matrix.f1(Average::Binary),
            Err(MetricError::NotBinary { labels: 3 })
        );
        // An all-negative result is still a binary label set.
        let negative = ConfusionMatrix::new(&[0, 0], &[0, 0]).unwrap();
        assert_eq!(
            negative.recall(Average::Binary),
            Err(MetricError::Undefined)
        );
    }

    #[test]
    fn macro_and_weighted_averages_match_the_reference_conventions() {
        let matrix = three_class();
        assert_near(matrix.precision(Average::Macro), 0.611_111_111_111_111);
        assert_near(matrix.recall(Average::Macro), 0.666_666_666_666_666_6);
        assert_near(matrix.f1(Average::Macro), 0.622_222_222_222_222_2);
        assert_near(matrix.fbeta(2.0, Average::Macro), 0.644_300_144_300_144_3);
        assert_near(matrix.precision(Average::Weighted), 0.604_166_666_666_666_6);
        assert_near(matrix.recall(Average::Weighted), 0.625);
        assert_near(matrix.f1(Average::Weighted), 0.6);
        assert_near(
            matrix.fbeta(2.0, Average::Weighted),
            0.611_201_298_701_298_7,
        );

        // A class present only in the predictions carries no averaging weight.
        let absent_truth = ConfusionMatrix::new(&[0, 0, 1], &[0, 2, 1]).unwrap();
        assert_near(absent_truth.precision(Average::Weighted), 1.0);
        assert_near(absent_truth.f1(Average::Weighted), 0.777_777_777_777_777_7);
        assert_near(
            absent_truth.precision(Average::Macro),
            0.666_666_666_666_666_6,
        );
        assert_near(absent_truth.f1(Average::Macro), 0.555_555_555_555_555_5);
    }

    #[test]
    fn an_empty_denominator_is_an_error_until_the_caller_says_otherwise() {
        // Class 2 is never predicted, so it has no precision.
        let matrix = ConfusionMatrix::new(&[0, 1, 2], &[0, 1, 1]).unwrap();
        assert_eq!(
            matrix.precision(Average::Macro),
            Err(MetricError::Undefined)
        );
        assert_eq!(
            matrix.precision(Average::Weighted),
            Err(MetricError::Undefined)
        );
        for average in [Average::Macro, Average::Weighted] {
            assert_near(
                matrix.precision(Averaging::new(average).with_zero_division(ZeroDivision::Skip)),
                0.75,
            );
            assert_near(
                matrix.precision(Averaging::new(average).with_zero_division(ZeroDivision::Zero)),
                0.5,
            );
        }
        // Recall and F-score are defined for every observed label here.
        assert_near(matrix.recall(Average::Macro), 0.666_666_666_666_666_6);
        assert_near(matrix.f1(Average::Macro), 0.555_555_555_555_555_5);

        // Skipping every class leaves nothing to average.
        let nothing_predicted = ConfusionMatrix::new(&[0, 0], &[0, 0]).unwrap();
        assert_eq!(
            nothing_predicted
                .recall(Averaging::new(Average::Macro).with_zero_division(ZeroDivision::Skip)),
            Ok(1.0)
        );
        let absent = ConfusionMatrix::new(&[0, 0], &[1, 1]).unwrap();
        assert_eq!(
            absent.recall(Averaging::new(Average::Macro).with_zero_division(ZeroDivision::Skip)),
            Ok(0.0)
        );
    }

    #[test]
    fn beta_selects_between_precision_and_recall_and_rejects_invalid_values() {
        // Recall 1.0 beats precision 2/3, so a recall-heavy beta scores higher.
        let matrix = ConfusionMatrix::new(&[0, 0, 1, 1], &[0, 1, 1, 1]).unwrap();
        let precision_heavy = matrix.fbeta(0.5, Average::Binary).unwrap();
        let balanced = matrix.f1(Average::Binary).unwrap();
        let recall_heavy = matrix.fbeta(2.0, Average::Binary).unwrap();
        assert!(precision_heavy < balanced && balanced < recall_heavy);
        assert_near(Ok(recall_heavy), 0.909_090_909_090_909_1);
        assert_near(Ok(precision_heavy), 0.714_285_714_285_714_2);

        for beta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                matrix.fbeta(beta, Average::Binary),
                Err(MetricError::InvalidBeta)
            );
        }
    }

    #[test]
    fn a_perfect_result_scores_one_under_every_average() {
        let matrix = ConfusionMatrix::new(&[0, 1, 2], &[0, 1, 2]).unwrap();
        for average in [Average::Micro, Average::Macro, Average::Weighted] {
            assert_eq!(matrix.precision(average), Ok(1.0));
            assert_eq!(matrix.recall(average), Ok(1.0));
            assert_eq!(matrix.f1(average), Ok(1.0));
        }
        assert_eq!(matrix.accuracy(), 1.0);
    }

    #[test]
    fn matrix_construction_validates_before_counting() {
        assert_eq!(
            ConfusionMatrix::new(&[0], &[]),
            Err(MetricError::LengthMismatch {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(ConfusionMatrix::new(&[], &[]), Err(MetricError::Empty));
    }
}
