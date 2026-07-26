//! Tie-aware ranking metrics with explicit undefined results.

use std::error::Error;
use std::fmt;

use super::PairOutcome;

/// Errors produced by ranking metrics.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RankingMetricError {
    /// Expected and predicted inputs have different lengths.
    LengthMismatch {
        /// Length of the expected-score input.
        expected: usize,
        /// Length of the predicted-score input.
        actual: usize,
    },
    /// A metric received no usable observations.
    Empty,
    /// A score is NaN or infinite.
    NonFiniteScore {
        /// Which input held the offending score: `0` expected, `1` predicted.
        input: usize,
        /// Zero-based position of that score within its input.
        index: usize,
    },
    /// The metric denominator is zero.
    Undefined,
}

impl fmt::Display for RankingMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(f, "expected {expected} ranking values, got {actual}")
            }
            Self::Empty => f.write_str("ranking metric requires at least one usable observation"),
            Self::NonFiniteScore { input, index } => {
                write!(
                    f,
                    "ranking score in input {input} at index {index} is not finite"
                )
            }
            Self::Undefined => f.write_str("ranking metric is undefined for constant ordering"),
        }
    }
}

impl Error for RankingMetricError {}

/// Accuracy over expected decisive outcomes; expected ties are excluded.
pub fn decisive_directional_accuracy(
    expected: &[PairOutcome],
    predicted: &[PairOutcome],
) -> Result<f64, RankingMetricError> {
    validate_lengths(expected.len(), predicted.len())?;
    let mut correct = 0_usize;
    let mut decisive = 0_usize;
    for (&expected, &predicted) in expected.iter().zip(predicted) {
        if expected != PairOutcome::Tie {
            decisive += 1;
            correct += usize::from(expected == predicted);
        }
    }
    if decisive == 0 {
        return Err(RankingMetricError::Empty);
    }
    Ok(correct as f64 / decisive as f64)
}

/// Exact three-way accuracy including ties.
pub fn three_way_accuracy(
    expected: &[PairOutcome],
    predicted: &[PairOutcome],
) -> Result<f64, RankingMetricError> {
    validate_lengths(expected.len(), predicted.len())?;
    if expected.is_empty() {
        return Err(RankingMetricError::Empty);
    }
    let correct = expected
        .iter()
        .zip(predicted)
        .filter(|(left, right)| left == right)
        .count();
    Ok(correct as f64 / expected.len() as f64)
}

/// Spearman rank correlation with exact-tie average ranks.
pub fn spearman_correlation(
    expected: &[f64],
    predicted: &[f64],
) -> Result<f64, RankingMetricError> {
    validate_scores(expected, predicted)?;
    if expected.is_empty() {
        return Err(RankingMetricError::Empty);
    }
    if expected.len() < 2 {
        return Err(RankingMetricError::Undefined);
    }
    let expected_ranks = average_ranks(expected);
    let predicted_ranks = average_ranks(predicted);
    pearson(&expected_ranks, &predicted_ranks)
}

/// Kendall's tau-b, including ties in either score vector.
pub fn kendall_tau_b(expected: &[f64], predicted: &[f64]) -> Result<f64, RankingMetricError> {
    validate_scores(expected, predicted)?;
    if expected.is_empty() {
        return Err(RankingMetricError::Empty);
    }
    if expected.len() < 2 {
        return Err(RankingMetricError::Undefined);
    }
    let mut concordant = 0_u64;
    let mut discordant = 0_u64;
    let mut expected_only_ties = 0_u64;
    let mut predicted_only_ties = 0_u64;
    for left in 0..expected.len() {
        for right in left + 1..expected.len() {
            let expected_order = if expected[left] == expected[right] {
                std::cmp::Ordering::Equal
            } else {
                expected[left].total_cmp(&expected[right])
            };
            let predicted_order = if predicted[left] == predicted[right] {
                std::cmp::Ordering::Equal
            } else {
                predicted[left].total_cmp(&predicted[right])
            };
            match (expected_order.is_eq(), predicted_order.is_eq()) {
                (true, true) => {}
                (true, false) => expected_only_ties += 1,
                (false, true) => predicted_only_ties += 1,
                (false, false) if expected_order == predicted_order => concordant += 1,
                (false, false) => discordant += 1,
            }
        }
    }
    let ordered = concordant + discordant;
    let left = ordered + predicted_only_ties;
    let right = ordered + expected_only_ties;
    let denominator = ((left as f64) * (right as f64)).sqrt();
    if denominator == 0.0 {
        return Err(RankingMetricError::Undefined);
    }
    Ok((concordant as f64 - discordant as f64) / denominator)
}

fn validate_lengths(expected: usize, actual: usize) -> Result<(), RankingMetricError> {
    if expected != actual {
        return Err(RankingMetricError::LengthMismatch { expected, actual });
    }
    Ok(())
}

fn validate_scores(expected: &[f64], predicted: &[f64]) -> Result<(), RankingMetricError> {
    validate_lengths(expected.len(), predicted.len())?;
    for (input, values) in [expected, predicted].into_iter().enumerate() {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(RankingMetricError::NonFiniteScore { input, index });
        }
    }
    Ok(())
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average = ((start + 1 + end) as f64) / 2.0;
        for &index in &order[start..end] {
            ranks[index] = average;
        }
        start = end;
    }
    ranks
}

fn pearson(left: &[f64], right: &[f64]) -> Result<f64, RankingMetricError> {
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator == 0.0 {
        return Err(RankingMetricError::Undefined);
    }
    Ok(covariance / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_accuracies_handle_ties_and_empty_decisive_sets() {
        let expected = [
            PairOutcome::LeftPreferred,
            PairOutcome::Tie,
            PairOutcome::RightPreferred,
        ];
        let predicted = [
            PairOutcome::LeftPreferred,
            PairOutcome::LeftPreferred,
            PairOutcome::Tie,
        ];
        assert_eq!(
            decisive_directional_accuracy(&expected, &predicted),
            Ok(0.5)
        );
        assert_eq!(three_way_accuracy(&expected, &predicted), Ok(1.0 / 3.0));
        assert_eq!(
            decisive_directional_accuracy(&[PairOutcome::Tie], &[PairOutcome::Tie]),
            Err(RankingMetricError::Empty)
        );
        assert_eq!(three_way_accuracy(&[], &[]), Err(RankingMetricError::Empty));
    }

    #[test]
    fn spearman_uses_average_ranks_and_rejects_constant_inputs() {
        assert_eq!(
            spearman_correlation(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]),
            Ok(1.0)
        );
        assert_eq!(
            spearman_correlation(&[1.0, 2.0, 3.0], &[30.0, 20.0, 10.0]),
            Ok(-1.0)
        );
        let tied = spearman_correlation(&[1.0, 1.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((tied - 0.866_025_403_784_438_7).abs() < 1.0e-15);
        assert_eq!(
            spearman_correlation(&[1.0, 1.0], &[2.0, 3.0]),
            Err(RankingMetricError::Undefined)
        );
        assert_eq!(
            spearman_correlation(&[], &[]),
            Err(RankingMetricError::Empty)
        );
    }

    #[test]
    fn kendall_tau_b_covers_concordance_discordance_and_ties() {
        assert_eq!(kendall_tau_b(&[1.0, 2.0, 3.0], &[2.0, 4.0, 8.0]), Ok(1.0));
        assert_eq!(kendall_tau_b(&[1.0, 2.0, 3.0], &[8.0, 4.0, 2.0]), Ok(-1.0));
        let tied = kendall_tau_b(&[1.0, 1.0, 2.0], &[1.0, 2.0, 2.0]).unwrap();
        assert!((tied - 0.5).abs() < 1.0e-15);
        assert_eq!(
            kendall_tau_b(&[1.0, 1.0], &[2.0, 2.0]),
            Err(RankingMetricError::Undefined)
        );
    }

    #[test]
    fn metrics_validate_lengths_and_finiteness() {
        assert_eq!(
            three_way_accuracy(&[PairOutcome::Tie], &[]),
            Err(RankingMetricError::LengthMismatch {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            spearman_correlation(&[1.0, f64::NAN], &[1.0, 2.0]),
            Err(RankingMetricError::NonFiniteScore { input: 0, index: 1 })
        );
    }
}
