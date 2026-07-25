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

/// An owned, non-empty vector of general classification targets.
///
/// Every `u8` is a valid class label, so construction only rejects an empty
/// vector. What this type adds over a bare `Vec<u8>` is the *observed class
/// set*: the sorted, deduplicated labels actually present, computed once at
/// construction and carried alongside the values.
///
/// That set is the column order of every probability matrix produced from these
/// targets. It is deliberately not assumed contiguous and not assumed
/// zero-based — labels `{7, 3, 10}` give classes `[3, 7, 10]`, and column `j`
/// is the probability of `classes()[j]`. There is no notion of a "missing"
/// class between two observed labels, so a caller never has to renumber its
/// labels to fit a dense range.
///
/// ```
/// use ferricml::data::ClassTargets;
///
/// let targets = ClassTargets::new(vec![7, 3, 10, 3])?;
/// assert_eq!(targets.classes(), &[3, 7, 10]);
/// assert_eq!(targets.n_classes(), 3);
/// assert_eq!(targets.class_index(10), Some(2));
/// assert_eq!(targets.class_index(0), None);
/// # Ok::<(), ferricml::data::DataError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassTargets {
    values: Vec<u8>,
    classes: Vec<u8>,
}

impl ClassTargets {
    /// Validates targets and records their sorted observed class set.
    pub fn new(values: Vec<u8>) -> Result<Self, DataError> {
        if values.is_empty() {
            return Err(DataError::EmptyTargets);
        }
        let classes = observed_classes(&values);
        Ok(Self { values, classes })
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

    /// Returns the target values.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.values
    }

    /// Returns a target, or `None` if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u8> {
        self.values.get(index).copied()
    }

    /// Returns the sorted, deduplicated labels observed in these targets.
    ///
    /// This is the probability column order for any classifier fitted on them.
    #[inline]
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    /// Returns the number of distinct observed labels.
    ///
    /// Always at least one, because valid targets are never empty.
    #[inline]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Returns the column index of `label`, or `None` when it was not observed.
    #[inline]
    pub fn class_index(&self, label: u8) -> Option<usize> {
        self.classes.binary_search(&label).ok()
    }

    /// Consumes the targets and returns their allocation.
    #[inline]
    pub fn into_values(self) -> Vec<u8> {
        self.values
    }

    /// Copies selected targets in the requested order.
    ///
    /// The observed class set is recomputed from the selection, so a fold that
    /// happens to contain fewer labels than the whole vector reports exactly
    /// the labels it contains.
    pub fn select(&self, indices: &[usize]) -> Result<Self, SelectionError> {
        validate_selection(indices, self.values.len())?;
        let values = indices
            .iter()
            .map(|&index| self.values[index])
            .collect::<Vec<_>>();
        let classes = observed_classes(&values);
        Ok(Self { values, classes })
    }
}

impl From<BinaryTargets> for ClassTargets {
    /// Widens validated binary targets without revalidating them.
    ///
    /// The observed class set of `[0, 0]` is `[0]`, not `[0, 1]`: it records
    /// what was seen, never what was permitted.
    fn from(targets: BinaryTargets) -> Self {
        let values = targets.into_values();
        let classes = observed_classes(&values);
        Self { values, classes }
    }
}

/// Sorted, deduplicated labels, in one pass plus a fixed 256-entry scan.
///
/// The fixed scan is what makes the result independent of the input order, and
/// therefore of anything a caller happens to do upstream.
fn observed_classes(values: &[u8]) -> Vec<u8> {
    let mut observed = [false; 256];
    for &value in values {
        observed[value as usize] = true;
    }
    (0..=u8::MAX)
        .filter(|&label| observed[label as usize])
        .collect()
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
