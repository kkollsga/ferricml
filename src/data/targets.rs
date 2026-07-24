use super::selection::validate_selection;
use super::{DataError, SelectionError};

fn validate_finite(values: &[f32]) -> Result<(), DataError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(DataError::NonFiniteValue { index });
    }
    Ok(())
}

/// An owned, non-empty vector of binary classification targets.
///
/// Every target is guaranteed to be either `0` or `1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryTargets {
    values: Vec<u8>,
}

impl BinaryTargets {
    /// Validates and stores binary targets.
    pub fn new(values: Vec<u8>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptyTargets);
        }
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, value)| **value > 1) {
            return Err(DataError::InvalidBinaryTarget { index, value });
        }
        Ok(Self { values })
    }

    /// Returns the number of targets.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this vector is empty.
    ///
    /// Valid instances are never empty; this method is provided for standard
    /// collection-style interfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the validated target values.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.values
    }

    /// Returns a target, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u8> {
        self.values.get(index).copied()
    }

    /// Consumes the targets and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<u8> {
        self.values
    }

    /// Copies selected targets in the requested order.
    pub fn select(&self, indices: &[usize]) -> Result<Self, SelectionError> {
        validate_selection(indices, self.values.len())?;
        Ok(Self {
            values: indices.iter().map(|&index| self.values[index]).collect(),
        })
    }
}

/// An owned, non-empty vector of finite regression targets.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionTargets {
    values: Vec<f32>,
}

impl RegressionTargets {
    /// Validates and stores regression targets.
    pub fn new(values: Vec<f32>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptyTargets);
        }
        validate_finite(&values)?;
        Ok(Self { values })
    }

    /// Returns the number of targets.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this vector is empty.
    ///
    /// Valid instances are never empty; this method is provided for standard
    /// collection-style interfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the finite target values.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Returns a target, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Consumes the targets and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Copies selected targets in the requested order.
    pub fn select(&self, indices: &[usize]) -> Result<Self, SelectionError> {
        validate_selection(indices, self.values.len())?;
        Ok(Self {
            values: indices.iter().map(|&index| self.values[index]).collect(),
        })
    }
}
