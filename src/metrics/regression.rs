use super::{MetricError, validate_finite, validate_lengths};

/// Mean absolute prediction error.
pub fn mean_absolute_error(expected: &[f32], predicted: &[f32]) -> Result<f64, MetricError> {
    validate_regression_inputs(expected, predicted)?;
    let sum = expected
        .iter()
        .zip(predicted)
        .map(|(&expected, &predicted)| (f64::from(predicted) - f64::from(expected)).abs())
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

/// Mean squared prediction error.
pub fn mean_squared_error(expected: &[f32], predicted: &[f32]) -> Result<f64, MetricError> {
    validate_regression_inputs(expected, predicted)?;
    let sum = expected
        .iter()
        .zip(predicted)
        .map(|(&expected, &predicted)| {
            let error = f64::from(predicted) - f64::from(expected);
            error * error
        })
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

/// Square root of mean squared prediction error.
pub fn root_mean_squared_error(expected: &[f32], predicted: &[f32]) -> Result<f64, MetricError> {
    Ok(mean_squared_error(expected, predicted)?.sqrt())
}

/// Coefficient of determination.
///
/// The score is undefined when every expected target is identical.
pub fn r2_score(expected: &[f32], predicted: &[f32]) -> Result<f64, MetricError> {
    validate_regression_inputs(expected, predicted)?;
    let mean = mean(expected);
    let mut residual_sum = 0.0_f64;
    let mut total_sum = 0.0_f64;
    for (&expected, &predicted) in expected.iter().zip(predicted) {
        let residual = f64::from(predicted) - f64::from(expected);
        let centered = f64::from(expected) - mean;
        residual_sum += residual * residual;
        total_sum += centered * centered;
    }
    if total_sum == 0.0 {
        return Err(MetricError::Undefined);
    }
    Ok(1.0 - residual_sum / total_sum)
}

/// Median of the absolute prediction errors.
///
/// The median is the mean of the two central errors for an even number of
/// observations, which keeps the value continuous as observations are added.
/// Unlike the mean error it is unmoved by a small number of large errors.
pub fn median_absolute_error(expected: &[f32], predicted: &[f32]) -> Result<f64, MetricError> {
    validate_regression_inputs(expected, predicted)?;
    let mut errors = expected
        .iter()
        .zip(predicted)
        .map(|(&expected, &predicted)| (f64::from(predicted) - f64::from(expected)).abs())
        .collect::<Vec<_>>();
    errors.sort_unstable_by(f64::total_cmp);
    let middle = errors.len() / 2;
    Ok(if errors.len() % 2 == 0 {
        (errors[middle - 1] + errors[middle]) / 2.0
    } else {
        errors[middle]
    })
}

/// Fraction of the target's variance the predictions account for.
///
/// This differs from [`r2_score`] by ignoring a constant offset: a prediction
/// that is systematically shifted but perfectly shaped explains all the
/// variance while scoring below one on R². The score is undefined when every
/// expected target is identical.
pub fn explained_variance_score(expected: &[f32], predicted: &[f32]) -> Result<f64, MetricError> {
    validate_regression_inputs(expected, predicted)?;
    let expected_mean = mean(expected);
    let residual_mean = expected
        .iter()
        .zip(predicted)
        .map(|(&expected, &predicted)| f64::from(expected) - f64::from(predicted))
        .sum::<f64>()
        / expected.len() as f64;
    let mut residual_sum = 0.0_f64;
    let mut total_sum = 0.0_f64;
    for (&expected, &predicted) in expected.iter().zip(predicted) {
        let residual = f64::from(expected) - f64::from(predicted) - residual_mean;
        let centered = f64::from(expected) - expected_mean;
        residual_sum += residual * residual;
        total_sum += centered * centered;
    }
    if total_sum == 0.0 {
        return Err(MetricError::Undefined);
    }
    Ok(1.0 - residual_sum / total_sum)
}

/// Mean absolute error relative to each expected value.
///
/// The result is a fraction, not a percentage. Every expected value is a
/// denominator, so a single expected zero makes the whole metric undefined
/// rather than being clamped to an arbitrary floor that would silently
/// dominate the mean.
pub fn mean_absolute_percentage_error(
    expected: &[f32],
    predicted: &[f32],
) -> Result<f64, MetricError> {
    validate_regression_inputs(expected, predicted)?;
    // Signed zeros compare equal, so this rejects a negative zero too.
    if expected.contains(&0.0) {
        return Err(MetricError::Undefined);
    }
    let sum = expected
        .iter()
        .zip(predicted)
        .map(|(&expected, &predicted)| {
            (f64::from(predicted) - f64::from(expected)).abs() / f64::from(expected).abs()
        })
        .sum::<f64>();
    Ok(sum / expected.len() as f64)
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64
}

fn validate_regression_inputs(expected: &[f32], predicted: &[f32]) -> Result<(), MetricError> {
    validate_lengths(expected.len(), predicted.len())?;
    validate_finite(expected, 0)?;
    validate_finite(predicted, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_metrics_match_hand_calculation() {
        let expected = [1.0, 2.0, 3.0, 4.0];
        let predicted = [1.0, 3.0, 2.0, 5.0];
        assert_eq!(mean_absolute_error(&expected, &predicted), Ok(0.75));
        assert_eq!(mean_squared_error(&expected, &predicted), Ok(0.75));
        assert_eq!(
            root_mean_squared_error(&expected, &predicted),
            Ok(0.75_f64.sqrt())
        );
        assert_eq!(r2_score(&expected, &predicted), Ok(0.4));
    }

    #[test]
    fn metric_identities_and_translation_invariance_hold() {
        let expected = [0.0, 2.0, 4.0, 8.0];
        let predicted = [1.0, 1.0, 5.0, 7.0];
        let mse = mean_squared_error(&expected, &predicted).unwrap();
        let rmse = root_mean_squared_error(&expected, &predicted).unwrap();
        assert!((rmse * rmse - mse).abs() < 1.0e-15);

        let shifted_expected = expected.map(|value| value + 100.0);
        let shifted_predicted = predicted.map(|value| value + 100.0);
        assert_eq!(
            r2_score(&expected, &predicted),
            r2_score(&shifted_expected, &shifted_predicted)
        );
    }

    #[test]
    fn arithmetic_promotes_before_subtraction_and_squaring() {
        let expected = [-f32::MAX, f32::MAX];
        let predicted = [f32::MAX, -f32::MAX];
        assert!(
            mean_absolute_error(&expected, &predicted)
                .unwrap()
                .is_finite()
        );
        assert!(
            mean_squared_error(&expected, &predicted)
                .unwrap()
                .is_finite()
        );
        assert!(
            root_mean_squared_error(&expected, &predicted)
                .unwrap()
                .is_finite()
        );
        assert!(r2_score(&expected, &predicted).unwrap().is_finite());
    }

    #[test]
    fn constant_targets_make_r2_undefined() {
        assert_eq!(
            r2_score(&[2.0, 2.0], &[2.0, 2.0]),
            Err(MetricError::Undefined)
        );
        assert_eq!(
            r2_score(&[2.0, 2.0], &[1.0, 3.0]),
            Err(MetricError::Undefined)
        );
    }

    #[test]
    fn the_added_regression_metrics_match_the_reference_conventions() {
        let expected = [3.0, -0.5, 2.0, 7.0];
        let predicted = [2.5, 0.0, 2.0, 8.0];
        assert_eq!(median_absolute_error(&expected, &predicted), Ok(0.5));
        assert!(
            (explained_variance_score(&expected, &predicted).unwrap() - 0.957_173_447_537_473_2)
                .abs()
                < 1.0e-12
        );
        assert!(
            (mean_absolute_percentage_error(&expected, &predicted).unwrap()
                - 0.327_380_952_380_952_4)
                .abs()
                < 1.0e-12
        );
        // An even count takes the mean of the two central errors.
        assert_eq!(
            median_absolute_error(&[0.0, 0.0, 0.0, 0.0], &[1.0, 2.0, 3.0, 10.0]),
            Ok(2.5)
        );
        assert_eq!(
            median_absolute_error(&[0.0, 0.0, 0.0], &[1.0, 2.0, 30.0]),
            Ok(2.0)
        );
    }

    #[test]
    fn explained_variance_ignores_the_constant_offset_that_r2_penalizes() {
        let expected = [1.0, 2.0, 3.0];
        let shifted = [2.0, 3.0, 4.0];
        assert_eq!(explained_variance_score(&expected, &shifted), Ok(1.0));
        assert!(r2_score(&expected, &shifted).unwrap() < 1.0);
        // Without an offset the two agree.
        assert_eq!(
            explained_variance_score(&expected, &expected),
            r2_score(&expected, &expected)
        );
    }

    #[test]
    fn undefined_added_metrics_are_explicit() {
        // Constant targets have no variance to explain.
        assert_eq!(
            explained_variance_score(&[2.0, 2.0], &[1.0, 3.0]),
            Err(MetricError::Undefined)
        );
        // A zero expected value is a zero denominator, not a clamped one.
        for expected in [[0.0_f32, 1.0], [1.0, 0.0], [-0.0, 1.0]] {
            assert_eq!(
                mean_absolute_percentage_error(&expected, &[1.0, 1.0]),
                Err(MetricError::Undefined)
            );
        }
        assert_eq!(
            mean_absolute_percentage_error(&[], &[]),
            Err(MetricError::Empty)
        );
        assert_eq!(
            median_absolute_error(&[0.0], &[f32::NAN]),
            Err(MetricError::NonFiniteValue { input: 1, index: 0 })
        );
    }

    #[test]
    fn regression_validation_is_ordered_and_rejects_non_finite_values() {
        assert_eq!(
            mean_squared_error(&[f32::NAN], &[]),
            Err(MetricError::LengthMismatch {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(mean_squared_error(&[], &[]), Err(MetricError::Empty));
        assert_eq!(
            mean_squared_error(&[f32::NAN], &[0.0]),
            Err(MetricError::NonFiniteValue { input: 0, index: 0 })
        );
        assert_eq!(
            mean_squared_error(&[0.0], &[f32::INFINITY]),
            Err(MetricError::NonFiniteValue { input: 1, index: 0 })
        );
    }
}
