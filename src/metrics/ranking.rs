//! Threshold-swept curves and the score ordering every rank metric shares.
//!
//! Ranked classification metrics all answer the same question in different
//! words: as the decision threshold sweeps from above every score down to the
//! lowest one, how do the counts change? Ordering the scores once, with equal
//! scores kept together as one tie group, is what makes ROC AUC, the two
//! curves, and average precision agree with each other by construction instead
//! of by coincidence.

use super::{MetricError, validate_binary, validate_finite, validate_lengths};

/// Indices of `scores` in ascending score order, ties broken by ascending
/// index.
///
/// `total_cmp` gives a total order over every `f32`, and the index tie-break
/// makes the permutation unique, so the result is deterministic for any input.
pub(super) fn ascending_score_order(scores: &[f32]) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        scores[left]
            .total_cmp(&scores[right])
            .then_with(|| left.cmp(&right))
    });
    order
}

/// Half-open bounds of each run of equal scores within `order`, ascending.
///
/// Equal scores cannot be separated by a threshold, so every metric here must
/// treat one run as one indivisible step.
pub(super) fn tie_groups(scores: &[f32], order: &[usize]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && scores[order[end]] == scores[order[start]] {
            end += 1;
        }
        groups.push((start, end));
        start = end;
    }
    groups
}

/// Counts of both classes, or the reason the ranked metric is undefined.
fn class_counts(expected: &[u8], scores: &[f32]) -> Result<(f64, f64), MetricError> {
    validate_lengths(expected.len(), scores.len())?;
    validate_binary(expected, 0)?;
    validate_finite(scores, 1)?;
    let positives = expected.iter().filter(|&&value| value == 1).count();
    let negatives = expected.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(MetricError::Undefined);
    }
    Ok((positives as f64, negatives as f64))
}

/// A receiver-operating-characteristic curve.
///
/// The three slices are parallel and ordered by decreasing threshold, so
/// index `0` is the operating point that predicts nothing positive. That point
/// has no score above it, so its threshold is reported as `f32::INFINITY`;
/// every later threshold is one of the observed scores, each appearing once
/// however many rows share it. Both rate slices are non-decreasing and run
/// from `0.0` to `1.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct RocCurve {
    thresholds: Vec<f32>,
    false_positive_rates: Vec<f64>,
    true_positive_rates: Vec<f64>,
}

impl RocCurve {
    /// Decision thresholds, decreasing, starting above every score.
    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Fraction of negative rows scored at or above each threshold.
    pub fn false_positive_rates(&self) -> &[f64] {
        &self.false_positive_rates
    }

    /// Fraction of positive rows scored at or above each threshold.
    pub fn true_positive_rates(&self) -> &[f64] {
        &self.true_positive_rates
    }

    /// Number of operating points, including the leading empty prediction.
    pub fn len(&self) -> usize {
        self.thresholds.len()
    }

    /// Returns whether the curve holds no operating points.
    ///
    /// A successfully computed curve always holds at least two, so this exists
    /// only to keep the collection-style interface explicit.
    pub fn is_empty(&self) -> bool {
        self.thresholds.is_empty()
    }
}

/// A precision-recall curve.
///
/// The three slices are parallel and ordered by decreasing threshold. Unlike
/// [`RocCurve`] there is no leading point: predicting nothing positive leaves
/// precision undefined rather than at a conventional value, so the curve
/// starts at the highest observed score. Recall is non-decreasing; precision is
/// not monotone in general.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionRecallCurve {
    thresholds: Vec<f32>,
    precisions: Vec<f64>,
    recalls: Vec<f64>,
}

impl PrecisionRecallCurve {
    /// Decision thresholds, decreasing, each an observed score.
    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Fraction of rows predicted positive that are positive, per threshold.
    pub fn precisions(&self) -> &[f64] {
        &self.precisions
    }

    /// Fraction of positive rows predicted positive, per threshold.
    pub fn recalls(&self) -> &[f64] {
        &self.recalls
    }

    /// Number of operating points.
    pub fn len(&self) -> usize {
        self.thresholds.len()
    }

    /// Returns whether the curve holds no operating points.
    ///
    /// A successfully computed curve always holds at least one, so this exists
    /// only to keep the collection-style interface explicit.
    pub fn is_empty(&self) -> bool {
        self.thresholds.is_empty()
    }
}

/// Sweeps the decision threshold and reports true and false positive rates.
///
/// Scores may be any finite values. The result is undefined unless both binary
/// classes are present, because an absent class leaves one rate without a
/// denominator.
pub fn roc_curve(expected: &[u8], scores: &[f32]) -> Result<RocCurve, MetricError> {
    let (positives, negatives) = class_counts(expected, scores)?;
    let order = ascending_score_order(scores);
    let groups = tie_groups(scores, &order);

    let mut thresholds = Vec::with_capacity(groups.len() + 1);
    let mut false_positive_rates = Vec::with_capacity(groups.len() + 1);
    let mut true_positive_rates = Vec::with_capacity(groups.len() + 1);
    thresholds.push(f32::INFINITY);
    false_positive_rates.push(0.0);
    true_positive_rates.push(0.0);

    let mut true_positives = 0.0_f64;
    let mut false_positives = 0.0_f64;
    for &(start, end) in groups.iter().rev() {
        for &index in &order[start..end] {
            if expected[index] == 1 {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }
        }
        thresholds.push(scores[order[start]]);
        false_positive_rates.push(false_positives / negatives);
        true_positive_rates.push(true_positives / positives);
    }
    Ok(RocCurve {
        thresholds,
        false_positive_rates,
        true_positive_rates,
    })
}

/// Sweeps the decision threshold and reports precision and recall.
///
/// The result is undefined unless at least one positive row exists, because
/// recall would otherwise have no denominator.
pub fn precision_recall_curve(
    expected: &[u8],
    scores: &[f32],
) -> Result<PrecisionRecallCurve, MetricError> {
    validate_lengths(expected.len(), scores.len())?;
    validate_binary(expected, 0)?;
    validate_finite(scores, 1)?;
    let positives = expected.iter().filter(|&&value| value == 1).count() as f64;
    if positives == 0.0 {
        return Err(MetricError::Undefined);
    }

    let order = ascending_score_order(scores);
    let groups = tie_groups(scores, &order);
    let mut thresholds = Vec::with_capacity(groups.len());
    let mut precisions = Vec::with_capacity(groups.len());
    let mut recalls = Vec::with_capacity(groups.len());

    let mut true_positives = 0.0_f64;
    let mut predicted = 0.0_f64;
    for &(start, end) in groups.iter().rev() {
        for &index in &order[start..end] {
            if expected[index] == 1 {
                true_positives += 1.0;
            }
            predicted += 1.0;
        }
        thresholds.push(scores[order[start]]);
        precisions.push(true_positives / predicted);
        recalls.push(true_positives / positives);
    }
    Ok(PrecisionRecallCurve {
        thresholds,
        precisions,
        recalls,
    })
}

/// Area under the precision-recall curve as a step function.
///
/// Each threshold contributes its precision weighted by the recall it gained,
/// with no interpolation between operating points, so the value is unaffected
/// by how the curve is drawn.
pub fn average_precision_score(expected: &[u8], scores: &[f32]) -> Result<f64, MetricError> {
    let curve = precision_recall_curve(expected, scores)?;
    let mut area = 0.0_f64;
    let mut previous_recall = 0.0_f64;
    for (&precision, &recall) in curve.precisions.iter().zip(&curve.recalls) {
        area += (recall - previous_recall) * precision;
        previous_recall = recall;
    }
    Ok(area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::roc_auc_score;

    fn assert_near(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "point {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn the_roc_curve_matches_the_reference_operating_points() {
        let curve = roc_curve(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8]).unwrap();
        assert_eq!(curve.thresholds(), &[f32::INFINITY, 0.8, 0.4, 0.35, 0.1]);
        assert_near(curve.false_positive_rates(), &[0.0, 0.0, 0.5, 0.5, 1.0]);
        assert_near(curve.true_positive_rates(), &[0.0, 0.5, 0.5, 1.0, 1.0]);
        assert_eq!(curve.len(), 5);
        assert!(!curve.is_empty());

        // One tie group is one indivisible step.
        let tied = roc_curve(&[0, 1, 1, 0, 1], &[0.5, 0.5, 0.5, 0.2, 0.9]).unwrap();
        assert_eq!(tied.thresholds(), &[f32::INFINITY, 0.9, 0.5, 0.2]);
        assert_near(tied.false_positive_rates(), &[0.0, 0.0, 0.5, 1.0]);
        assert_near(
            tied.true_positive_rates(),
            &[0.0, 0.333_333_333_333_333_3, 1.0, 1.0],
        );
    }

    #[test]
    fn the_roc_curve_is_monotone_and_ends_at_the_unit_corner() {
        for scores in [
            vec![0.9_f32, 0.1, 0.4, 0.4, 0.7, 0.2],
            vec![0.5; 6],
            vec![-3.0, 2.0, 2.0, 0.0, 1.0, -1.0],
        ] {
            let expected = [0_u8, 1, 0, 1, 1, 0];
            let curve = roc_curve(&expected, &scores).unwrap();
            assert_eq!(curve.false_positive_rates()[0], 0.0);
            assert_eq!(curve.true_positive_rates()[0], 0.0);
            assert_eq!(*curve.false_positive_rates().last().unwrap(), 1.0);
            assert_eq!(*curve.true_positive_rates().last().unwrap(), 1.0);
            assert!(
                curve.thresholds().windows(2).all(|pair| pair[0] > pair[1]),
                "{:?}",
                curve.thresholds()
            );
            for rates in [curve.false_positive_rates(), curve.true_positive_rates()] {
                assert!(rates.windows(2).all(|pair| pair[0] <= pair[1]), "{rates:?}");
            }
        }
    }

    #[test]
    fn trapezoidal_area_under_the_roc_curve_equals_the_auc_score() {
        for (expected, scores) in [
            (vec![0_u8, 0, 1, 1], vec![0.1_f32, 0.4, 0.35, 0.8]),
            (vec![0, 1, 1, 0, 1], vec![0.5, 0.5, 0.5, 0.2, 0.9]),
            (vec![1, 0, 1, 0, 0, 1], vec![0.9, 0.9, 0.2, 0.2, 0.5, 0.5]),
            (vec![0, 1], vec![0.3, 0.3]),
        ] {
            let curve = roc_curve(&expected, &scores).unwrap();
            let area = curve
                .false_positive_rates()
                .windows(2)
                .zip(curve.true_positive_rates().windows(2))
                .map(|(false_rate, true_rate)| {
                    (false_rate[1] - false_rate[0]) * (true_rate[0] + true_rate[1]) / 2.0
                })
                .sum::<f64>();
            let score = roc_auc_score(&expected, &scores).unwrap();
            assert!(
                (area - score).abs() < 1.0e-12,
                "curve area {area} disagrees with AUC {score}"
            );
        }
    }

    #[test]
    fn the_precision_recall_curve_matches_the_reference_operating_points() {
        let curve = precision_recall_curve(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8]).unwrap();
        assert_eq!(curve.thresholds(), &[0.8, 0.4, 0.35, 0.1]);
        assert_near(
            curve.precisions(),
            &[1.0, 0.5, 0.666_666_666_666_666_6, 0.5],
        );
        assert_near(curve.recalls(), &[0.5, 0.5, 1.0, 1.0]);

        let tied = precision_recall_curve(&[0, 1, 1, 0, 1], &[0.5, 0.5, 0.5, 0.2, 0.9]).unwrap();
        assert_eq!(tied.thresholds(), &[0.9, 0.5, 0.2]);
        assert_near(tied.precisions(), &[1.0, 0.75, 0.6]);
        assert_near(tied.recalls(), &[0.333_333_333_333_333_3, 1.0, 1.0]);
        assert!(tied.recalls().windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(*tied.recalls().last().unwrap(), 1.0);
        assert_eq!(tied.len(), 3);
        assert!(!tied.is_empty());
    }

    #[test]
    fn average_precision_is_the_recall_weighted_step_area() {
        assert_eq!(
            average_precision_score(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8]),
            Ok(0.833_333_333_333_333_3)
        );
        assert_eq!(
            average_precision_score(&[0, 1, 1, 0, 1], &[0.5, 0.5, 0.5, 0.2, 0.9]),
            Ok(0.833_333_333_333_333_3)
        );
        assert_eq!(average_precision_score(&[0, 1], &[0.3, 0.3]), Ok(0.5));
        assert_eq!(
            average_precision_score(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9]),
            Ok(1.0)
        );
        // A constant score reports the positive rate, whatever the labels' order.
        assert_eq!(average_precision_score(&[1, 0, 0, 0], &[0.5; 4]), Ok(0.25));
    }

    #[test]
    fn ranked_metrics_reject_absent_classes_and_invalid_inputs() {
        assert_eq!(
            roc_curve(&[1, 1], &[0.1, 0.9]).unwrap_err(),
            MetricError::Undefined
        );
        assert_eq!(
            roc_curve(&[0, 0], &[0.1, 0.9]).unwrap_err(),
            MetricError::Undefined
        );
        // Precision and recall only need positives to exist.
        assert!(precision_recall_curve(&[1, 1], &[0.1, 0.9]).is_ok());
        assert_eq!(
            precision_recall_curve(&[0, 0], &[0.1, 0.9]).unwrap_err(),
            MetricError::Undefined
        );
        assert_eq!(
            average_precision_score(&[0], &[]),
            Err(MetricError::LengthMismatch {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(average_precision_score(&[], &[]), Err(MetricError::Empty));
        assert_eq!(
            roc_curve(&[0, 2], &[0.1, 0.9]),
            Err(MetricError::InvalidBinaryTarget {
                input: 0,
                index: 1,
                value: 2,
            })
        );
        assert_eq!(
            precision_recall_curve(&[0, 1], &[0.1, f32::NAN]),
            Err(MetricError::NonFiniteValue { input: 1, index: 1 })
        );
    }

    #[test]
    fn the_shared_order_is_deterministic_under_ties_and_signed_zero() {
        let scores = [1.0_f32, 0.0, -0.0, 1.0, f32::MIN, 0.0];
        let order = ascending_score_order(&scores);
        assert_eq!(order, ascending_score_order(&scores));
        assert!(
            order
                .windows(2)
                .all(|pair| scores[pair[0]].total_cmp(&scores[pair[1]]).is_le())
        );
        assert_eq!(order, vec![4, 2, 1, 5, 0, 3]);
        // `total_cmp` orders -0.0 below 0.0, but no threshold can separate two
        // numerically equal scores, so a tie group is numeric equality.
        assert_eq!(tie_groups(&scores, &order), vec![(0, 1), (1, 4), (4, 6)]);
    }
}
