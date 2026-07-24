use super::DataError;

/// An owned, non-empty vector of finite, non-negative sample weights.
///
/// At least one weight is positive. Estimators check that the weight count
/// matches the training row count at their public fit boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleWeights {
    values: Vec<f32>,
    total: f64,
}

impl SampleWeights {
    /// Validates and stores sample weights.
    pub fn new(values: Vec<f32>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptySampleWeights);
        }
        let mut total = 0.0_f64;
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DataError::NonFiniteSampleWeight { index });
            }
            if value < 0.0 {
                return Err(DataError::NegativeSampleWeight { index });
            }
            total += f64::from(value);
        }
        if total <= 0.0 {
            return Err(DataError::ZeroTotalSampleWeight);
        }
        Ok(Self { values, total })
    }

    /// Returns the number of sample weights.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the vector is empty.
    ///
    /// Valid instances are never empty; this method is provided for standard
    /// collection-style interfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the validated sample weights.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Returns a sample weight, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Returns the finite positive sum accumulated in input order.
    #[inline]
    pub const fn total(&self) -> f64 {
        self.total
    }

    /// Consumes the weights and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}
