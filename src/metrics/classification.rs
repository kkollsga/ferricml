use super::{MetricError, validate_binary, validate_lengths, validate_probabilities};

const LOG_LOSS_EPSILON: f64 = 1.0e-15;

/// Counts for a binary classification result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BinaryConfusionMatrix {
    true_negatives: u64,
    false_positives: u64,
    false_negatives: u64,
    true_positives: u64,
}

impl BinaryConfusionMatrix {
    /// Correctly predicted zero labels.
    pub const fn true_negatives(self) -> u64 {
        self.true_negatives
    }

    /// Zero labels predicted as one.
    pub const fn false_positives(self) -> u64 {
        self.false_positives
    }

    /// One labels predicted as zero.
    pub const fn false_negatives(self) -> u64 {
        self.false_negatives
    }

    /// Correctly predicted one labels.
    pub const fn true_positives(self) -> u64 {
        self.true_positives
    }

    /// Total observations represented by this matrix.
    pub const fn total(self) -> u64 {
        self.true_negatives + self.false_positives + self.false_negatives + self.true_positives
    }
}

/// Fraction of labels predicted exactly.
///
/// Unlike the other classification metrics, accuracy accepts arbitrary `u8`
/// labels so it can evaluate non-binary label sets.
pub fn accuracy_score(expected: &[u8], predicted: &[u8]) -> Result<f64, MetricError> {
    validate_lengths(expected.len(), predicted.len())?;
    let correct = expected
        .iter()
        .zip(predicted)
        .filter(|(expected, predicted)| expected == predicted)
        .count();
    Ok(correct as f64 / expected.len() as f64)
}

/// Computes binary true-negative, false-positive, false-negative, and
/// true-positive counts.
pub fn binary_confusion_matrix(
    expected: &[u8],
    predicted: &[u8],
) -> Result<BinaryConfusionMatrix, MetricError> {
    validate_binary_pair(expected, predicted)?;
    let mut matrix = BinaryConfusionMatrix::default();
    for (&expected, &predicted) in expected.iter().zip(predicted) {
        match (expected, predicted) {
            (0, 0) => matrix.true_negatives += 1,
            (0, 1) => matrix.false_positives += 1,
            (1, 0) => matrix.false_negatives += 1,
            (1, 1) => matrix.true_positives += 1,
            _ => unreachable!("binary labels were validated"),
        }
    }
    Ok(matrix)
}

/// Positive predictive value for binary labels.
pub fn precision_score(expected: &[u8], predicted: &[u8]) -> Result<f64, MetricError> {
    let matrix = binary_confusion_matrix(expected, predicted)?;
    ratio(
        matrix.true_positives,
        matrix.true_positives + matrix.false_positives,
    )
}

/// True-positive rate for binary labels.
pub fn recall_score(expected: &[u8], predicted: &[u8]) -> Result<f64, MetricError> {
    let matrix = binary_confusion_matrix(expected, predicted)?;
    ratio(
        matrix.true_positives,
        matrix.true_positives + matrix.false_negatives,
    )
}

/// Harmonic mean of binary precision and recall.
pub fn f1_score(expected: &[u8], predicted: &[u8]) -> Result<f64, MetricError> {
    let matrix = binary_confusion_matrix(expected, predicted)?;
    ratio(
        2 * matrix.true_positives,
        2 * matrix.true_positives + matrix.false_positives + matrix.false_negatives,
    )
}

/// Mean squared error of positive-class probabilities.
pub fn brier_score(expected: &[u8], positive_probabilities: &[f32]) -> Result<f64, MetricError> {
    validate_binary_probabilities(expected, positive_probabilities)?;
    let sum = expected
        .iter()
        .zip(positive_probabilities)
        .map(|(&expected, &probability)| {
            let error = f64::from(probability) - f64::from(expected);
            error * error
        })
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

/// Mean binary logarithmic loss for positive-class probabilities.
///
/// Valid endpoint probabilities are clipped to `1e-15..=1-1e-15` so an exact
/// but wrong zero or one produces a finite, deterministic loss.
pub fn log_loss(expected: &[u8], positive_probabilities: &[f32]) -> Result<f64, MetricError> {
    validate_binary_probabilities(expected, positive_probabilities)?;
    let sum = expected
        .iter()
        .zip(positive_probabilities)
        .map(|(&expected, &probability)| {
            let probability =
                f64::from(probability).clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON);
            if expected == 1 {
                -probability.ln()
            } else {
                -(1.0 - probability).ln()
            }
        })
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

/// Mean cross-entropy of a multiclass probability matrix.
///
/// `probabilities` is row-major with one column per entry of `classes`, exactly
/// as a fitted classifier's `predict_proba` produces it, and `classes` must be
/// the sorted, deduplicated label set that names those columns. Only the column
/// of each row's true class contributes, and endpoint probabilities are clipped
/// to `1e-15..=1-1e-15` so an exact but wrong zero gives a finite,
/// deterministic loss.
///
/// # Rows are read as given
///
/// FerricML does not renormalize probability rows anywhere, and this metric is
/// no exception: it scores the numbers the model produced rather than a
/// rescaled version of them. A row that sums to one only to `f32` rounding —
/// which is the documented contract — therefore contributes the loss its own
/// values imply.
///
/// At two classes this agrees with [`log_loss`] on the same predictions, which
/// is what keeps binary and multiclass evaluation one definition rather than
/// two.
pub fn multiclass_log_loss(
    expected: &[u8],
    classes: &[u8],
    probabilities: &[f32],
) -> Result<f64, MetricError> {
    let columns = validate_probability_matrix(expected, classes, probabilities)?;
    let sum = expected
        .iter()
        .zip(probabilities.chunks_exact(columns))
        .map(|(&label, row)| {
            let column = classes
                .binary_search(&label)
                .expect("labels were validated against the class list");
            let probability =
                f64::from(row[column]).clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON);
            -probability.ln()
        })
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

/// Mean squared error of a whole multiclass probability row.
///
/// Each row contributes `Σₖ (pₖ - [y = k])²` over *every* class, which is the
/// standard multiclass Brier score. Inputs are shaped exactly as for
/// [`multiclass_log_loss`].
///
/// # Not interchangeable with [`brier_score`] at two classes
///
/// The binary [`brier_score`] squares the error of the positive column alone;
/// this squares both columns, and the two columns carry equal and opposite
/// errors. So on the same two-class predictions **this returns exactly twice
/// [`brier_score`]**. Both conventions are standard and neither is wrong; they
/// are named apart and this relationship is stated because silently differing
/// by a factor of two is the trap worth removing.
///
/// Rows are read as given, without renormalization, for the same reason as in
/// [`multiclass_log_loss`].
pub fn multiclass_brier_score(
    expected: &[u8],
    classes: &[u8],
    probabilities: &[f32],
) -> Result<f64, MetricError> {
    let columns = validate_probability_matrix(expected, classes, probabilities)?;
    let sum = expected
        .iter()
        .zip(probabilities.chunks_exact(columns))
        .map(|(&label, row)| {
            let observed = classes
                .binary_search(&label)
                .expect("labels were validated against the class list");
            row.iter()
                .enumerate()
                .map(|(column, &probability)| {
                    let error = f64::from(probability) - f64::from(column == observed);
                    error * error
                })
                .sum::<f64>()
        })
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

/// Validates a probability matrix against its labels and returns its width.
///
/// The order is deliberate and frozen: the class list is checked before
/// anything is indexed by it, then the matrix shape, then the values, then the
/// labels against the columns that exist.
fn validate_probability_matrix(
    expected: &[u8],
    classes: &[u8],
    probabilities: &[f32],
) -> Result<usize, MetricError> {
    if classes.is_empty() || classes.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MetricError::InvalidClassSet);
    }
    let columns = classes.len();
    let cells = expected
        .len()
        .checked_mul(columns)
        .ok_or(MetricError::InvalidClassSet)?;
    validate_lengths(cells, probabilities.len())?;
    validate_probabilities(probabilities)?;
    if let Some((index, &value)) = expected
        .iter()
        .enumerate()
        .find(|(_, value)| classes.binary_search(value).is_err())
    {
        return Err(MetricError::UnknownClass { index, value });
    }
    Ok(columns)
}

/// Area under the binary receiver-operating-characteristic curve.
///
/// Scores may be any finite values. Equal scores receive average ranks. The
/// result is undefined unless both binary classes are present.
pub fn roc_auc_score(expected: &[u8], scores: &[f32]) -> Result<f64, MetricError> {
    validate_lengths(expected.len(), scores.len())?;
    validate_binary(expected, 0)?;
    super::validate_finite(scores, 1)?;

    let positives = expected.iter().filter(|&&value| value == 1).count();
    let negatives = expected.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(MetricError::Undefined);
    }

    let order = super::ranking::ascending_score_order(scores);
    let mut positive_rank_sum = 0.0_f64;
    for (start, end) in super::ranking::tie_groups(scores, &order) {
        let average_rank = (start + 1 + end) as f64 / 2.0;
        let tied_positives = order[start..end]
            .iter()
            .filter(|&&index| expected[index] == 1)
            .count();
        positive_rank_sum += average_rank * tied_positives as f64;
    }

    let positives = positives as f64;
    let negatives = negatives as f64;
    Ok((positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))
}

fn validate_binary_pair(expected: &[u8], predicted: &[u8]) -> Result<(), MetricError> {
    validate_lengths(expected.len(), predicted.len())?;
    validate_binary(expected, 0)?;
    validate_binary(predicted, 1)
}

fn validate_binary_probabilities(
    expected: &[u8],
    probabilities: &[f32],
) -> Result<(), MetricError> {
    validate_lengths(expected.len(), probabilities.len())?;
    validate_binary(expected, 0)?;
    validate_probabilities(probabilities)
}

fn ratio(numerator: u64, denominator: u64) -> Result<f64, MetricError> {
    if denominator == 0 {
        return Err(MetricError::Undefined);
    }
    Ok(numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusion_and_derived_metrics_match_hand_calculation() {
        let expected = [0, 0, 0, 1, 1, 1];
        let predicted = [0, 1, 0, 1, 0, 1];
        let matrix = binary_confusion_matrix(&expected, &predicted).unwrap();
        assert_eq!(matrix.true_negatives(), 2);
        assert_eq!(matrix.false_positives(), 1);
        assert_eq!(matrix.false_negatives(), 1);
        assert_eq!(matrix.true_positives(), 2);
        assert_eq!(matrix.total(), expected.len() as u64);
        assert_eq!(accuracy_score(&expected, &predicted), Ok(2.0 / 3.0));
        assert_eq!(precision_score(&expected, &predicted), Ok(2.0 / 3.0));
        assert_eq!(recall_score(&expected, &predicted), Ok(2.0 / 3.0));
        assert_eq!(f1_score(&expected, &predicted), Ok(2.0 / 3.0));
    }

    #[test]
    fn accuracy_accepts_general_labels_but_binary_metrics_reject_them() {
        assert_eq!(accuracy_score(&[2, 3, 4], &[2, 0, 4]), Ok(2.0 / 3.0));
        assert_eq!(
            binary_confusion_matrix(&[0, 2], &[0, 1]),
            Err(MetricError::InvalidBinaryTarget {
                input: 0,
                index: 1,
                value: 2,
            })
        );
        assert_eq!(
            binary_confusion_matrix(&[0, 1], &[0, 2]),
            Err(MetricError::InvalidBinaryTarget {
                input: 1,
                index: 1,
                value: 2,
            })
        );
    }

    #[test]
    fn undefined_binary_denominators_are_explicit() {
        assert_eq!(
            precision_score(&[0, 0], &[0, 0]),
            Err(MetricError::Undefined)
        );
        assert_eq!(recall_score(&[0, 0], &[0, 1]), Err(MetricError::Undefined));
        assert_eq!(f1_score(&[0, 0], &[0, 0]), Err(MetricError::Undefined));
    }

    #[test]
    fn probability_metrics_validate_and_handle_endpoints() {
        let expected = [0, 1, 1, 0];
        let probabilities = [0.1, 0.9, 0.8, 0.2];
        assert!(
            (brier_score(&expected, &probabilities).unwrap() - 0.025_000_000_372_529_21).abs()
                < 1.0e-15
        );
        assert!(
            (log_loss(&expected, &probabilities).unwrap() - 0.164_252_037_728_709_9).abs()
                < 1.0e-12
        );
        assert_eq!(brier_score(&[0, 1], &[0.0, 1.0]), Ok(0.0));
        assert!(log_loss(&[1], &[0.0]).unwrap().is_finite());
        assert_eq!(
            brier_score(&[0], &[1.1]),
            Err(MetricError::InvalidProbability { index: 0 })
        );
        assert_eq!(
            log_loss(&[0], &[f32::NAN]),
            Err(MetricError::NonFiniteValue { input: 1, index: 0 })
        );
    }

    #[test]
    fn auc_handles_perfect_reversed_constant_and_tied_scores() {
        let expected = [0, 0, 1, 1];
        assert_eq!(roc_auc_score(&expected, &[0.0, 0.2, 0.8, 1.0]), Ok(1.0));
        assert_eq!(roc_auc_score(&expected, &[1.0, 0.8, 0.2, 0.0]), Ok(0.0));
        assert_eq!(roc_auc_score(&expected, &[0.5; 4]), Ok(0.5));
        assert_eq!(roc_auc_score(&[0, 1, 1], &[0.0, 0.5, 0.5]), Ok(1.0));
        assert_eq!(
            roc_auc_score(&[0, 0], &[0.0, 1.0]),
            Err(MetricError::Undefined)
        );
    }

    #[test]
    fn auc_matches_pairwise_oracle_on_small_exhaustive_inputs() {
        for len in 2..=7 {
            let label_variants = 1_usize << len;
            let score_variants = 3_usize.pow(len as u32);
            for label_bits in 1..label_variants - 1 {
                let expected = (0..len)
                    .map(|index| ((label_bits >> index) & 1) as u8)
                    .collect::<Vec<_>>();
                for mut encoded in 0..score_variants {
                    let mut scores = Vec::with_capacity(len);
                    for _ in 0..len {
                        scores.push((encoded % 3) as f32);
                        encoded /= 3;
                    }
                    let actual = roc_auc_score(&expected, &scores).unwrap();
                    let expected_auc = pairwise_auc(&expected, &scores);
                    assert!((actual - expected_auc).abs() < 1.0e-15);
                }
            }
        }
    }

    fn pairwise_auc(expected: &[u8], scores: &[f32]) -> f64 {
        let mut credit = 0.0;
        let mut pairs = 0_u64;
        for (positive, &positive_label) in expected.iter().enumerate() {
            if positive_label != 1 {
                continue;
            }
            for (negative, &negative_label) in expected.iter().enumerate() {
                if negative_label != 0 {
                    continue;
                }
                pairs += 1;
                credit += if scores[positive] > scores[negative] {
                    1.0
                } else if scores[positive] == scores[negative] {
                    0.5
                } else {
                    0.0
                };
            }
        }
        credit / pairs as f64
    }

    #[test]
    fn multiclass_probability_metrics_match_hand_calculation() {
        // Three rows, non-contiguous labels, columns ordered by `classes`.
        let expected = [7_u8, 3, 10];
        let classes = [3_u8, 7, 10];
        let probabilities = [
            0.2, 0.7, 0.1, // true class 7 -> column 1
            0.6, 0.3, 0.1, // true class 3 -> column 0
            0.1, 0.2, 0.7, // true class 10 -> column 2
        ];
        let loss = multiclass_log_loss(&expected, &classes, &probabilities).unwrap();
        let hand =
            -(f64::from(0.7_f32).ln() + f64::from(0.6_f32).ln() + f64::from(0.7_f32).ln()) / 3.0;
        assert!((loss - hand).abs() < 1.0e-12, "{loss} != {hand}");

        let brier = multiclass_brier_score(&expected, &classes, &probabilities).unwrap();
        let mut hand = 0.0_f64;
        for (row, &label) in expected.iter().enumerate() {
            let observed = classes.iter().position(|&class| class == label).unwrap();
            for column in 0..3 {
                let error =
                    f64::from(probabilities[row * 3 + column]) - f64::from(column == observed);
                hand += error * error;
            }
        }
        assert!(
            (brier - hand / 3.0).abs() < 1.0e-12,
            "{brier} != {}",
            hand / 3.0
        );

        // A perfect prediction scores zero on both.
        let perfect = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            multiclass_brier_score(&[3, 7, 10], &classes, &perfect),
            Ok(0.0)
        );
        // Clipping keeps an exact `1.0` just below one, so a perfect
        // cross-entropy is the clip's residual rather than a hard zero.
        assert!(multiclass_log_loss(&[3, 7, 10], &classes, &perfect).unwrap() < 1.0e-14);
        // A confidently wrong endpoint stays finite rather than diverging.
        assert!(
            multiclass_log_loss(&[10], &classes, &[1.0, 0.0, 0.0])
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn the_multiclass_metrics_agree_with_their_binary_counterparts_where_they_overlap() {
        let expected = [0_u8, 1, 1, 0, 1];
        let positive = [0.1_f32, 0.9, 0.8, 0.35, 0.55];
        let classes = [0_u8, 1];
        let matrix = positive
            .iter()
            .flat_map(|&value| [1.0 - value, value])
            .collect::<Vec<f32>>();

        // Cross-entropy reads only the true class's column, so the two
        // definitions coincide. They agree to `f32` precision rather than
        // exactly, because the binary form takes the complement in `f64` while
        // a probability matrix already stores the complement as an `f32`.
        let binary = log_loss(&expected, &positive).unwrap();
        let multiclass = multiclass_log_loss(&expected, &classes, &matrix).unwrap();
        assert!(
            (binary - multiclass).abs() < 1.0e-7,
            "{binary} != {multiclass}"
        );

        // Brier does not: summing both columns doubles the value, which is the
        // documented relationship rather than a discrepancy.
        let binary = brier_score(&expected, &positive).unwrap();
        let multiclass = multiclass_brier_score(&expected, &classes, &matrix).unwrap();
        assert!(
            (multiclass - 2.0 * binary).abs() < 1.0e-7,
            "{multiclass} != 2 * {binary}"
        );
    }

    #[test]
    fn multiclass_rows_are_scored_as_given_rather_than_renormalized() {
        // A row that does not sum to one is exactly what the no-renormalization
        // contract produces, so the metric must not quietly rescale it.
        let classes = [0_u8, 1, 2];
        let short = [0.25_f32, 0.25, 0.25];
        let scored = multiclass_log_loss(&[0], &classes, &short).unwrap();
        assert!((scored - -f64::from(0.25_f32).ln()).abs() < 1.0e-12);
        assert!(
            (scored - -f64::from(1.0_f32 / 3.0).ln()).abs() > 1.0e-3,
            "the row was renormalized"
        );

        let brier = multiclass_brier_score(&[0], &classes, &short).unwrap();
        assert!(
            (brier - (0.5625 + 0.0625 + 0.0625)).abs() < 1.0e-6,
            "{brier}"
        );

        // A single class is one column of ones, which both metrics accept.
        assert_eq!(multiclass_brier_score(&[9], &[9], &[1.0]), Ok(0.0));
        assert!(multiclass_log_loss(&[9], &[9], &[1.0]).unwrap() < 1.0e-14);
    }

    #[test]
    fn multiclass_metrics_validate_the_class_set_shape_and_labels_in_order() {
        let probabilities = [0.5_f32, 0.5];
        for classes in [&[][..], &[1, 1][..], &[2, 1][..]] {
            assert_eq!(
                multiclass_log_loss(&[1], classes, &probabilities),
                Err(MetricError::InvalidClassSet),
                "{classes:?}"
            );
        }
        assert_eq!(
            multiclass_log_loss(&[1], &[0, 1], &[0.5]),
            Err(MetricError::LengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            multiclass_log_loss(&[], &[0, 1], &[]),
            Err(MetricError::Empty)
        );
        assert_eq!(
            multiclass_brier_score(&[1], &[0, 1], &[0.5, 1.5]),
            Err(MetricError::InvalidProbability { index: 1 })
        );
        assert_eq!(
            multiclass_brier_score(&[1], &[0, 1], &[0.5, f32::NAN]),
            Err(MetricError::NonFiniteValue { input: 1, index: 1 })
        );
        // Shape and values are checked before the labels are matched to columns.
        assert_eq!(
            multiclass_log_loss(&[5], &[0, 1], &probabilities),
            Err(MetricError::UnknownClass { index: 0, value: 5 })
        );
    }

    #[test]
    fn validation_order_is_length_then_empty_then_content() {
        assert_eq!(
            accuracy_score(&[1], &[]),
            Err(MetricError::LengthMismatch {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(accuracy_score(&[], &[]), Err(MetricError::Empty));
        assert_eq!(
            brier_score(&[2], &[]),
            Err(MetricError::LengthMismatch {
                expected: 1,
                actual: 0,
            })
        );
    }
}
