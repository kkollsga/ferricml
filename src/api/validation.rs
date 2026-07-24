use super::ModelError;

pub(crate) fn validate_scalar_row(row: &[f32], expected: usize) -> Result<(), ModelError> {
    if row.len() != expected {
        return Err(ModelError::FeatureDimension {
            expected,
            actual: row.len(),
        });
    }
    if let Some(column) = row.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteFeature { row: 0, column });
    }
    Ok(())
}

pub(crate) fn validate_prediction(value: f32, row: usize) -> Result<f32, ModelError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ModelError::NonFinitePrediction { row })
    }
}
