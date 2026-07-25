use super::DataError;

/// An owned, non-empty vector of finite, non-negative sample weights.
///
/// At least one weight is positive. Estimators check that the weight count
/// matches the training row count at their public fit boundary.
///
/// # Class weighting is expressed here
///
/// FerricML has exactly one weighting concept. There is no `class_weight`
/// parameter on any estimator: a per-class weight is a function of the label,
/// which is to say a per-row weight, and giving it a second spelling would mean
/// two weighting systems that every estimator, every capability declaration,
/// and every artifact would have to agree about. Composing a class weight into
/// a row weight is three lines the caller writes once:
///
/// ```
/// use ferricml::data::{ClassTargets, SampleWeights};
///
/// let targets = ClassTargets::new(vec![0, 0, 0, 1]).unwrap();
/// // Inverse frequency, scaled so the total weight is the row count — the
/// // balanced rule, written where the caller can see and change it.
/// let classes = targets.classes().len() as f32;
/// let mut counts = vec![0_usize; classes as usize];
/// for &label in targets.as_slice() {
///     counts[targets.class_index(label).unwrap()] += 1;
/// }
/// let rows = targets.as_slice().len() as f32;
/// let weights = SampleWeights::new(
///     targets
///         .as_slice()
///         .iter()
///         .map(|&label| {
///             rows / (classes * counts[targets.class_index(label).unwrap()] as f32)
///         })
///         .collect(),
/// )
/// .unwrap();
///
/// assert_eq!(weights.as_slice(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0]);
/// assert!((weights.total() - 4.0).abs() < 1e-6);
/// ```
///
/// A caller wanting a different balancing rule writes a different closure
/// instead of asking for another parameter value.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ClassTargets;

    /// The balanced class-weight rule composed into row weights.
    ///
    /// This is the recipe the type documentation states, executed: it is what
    /// makes a `class_weight` parameter unnecessary rather than merely
    /// unimplemented.
    fn balanced(targets: &ClassTargets) -> SampleWeights {
        let classes = targets.classes().len();
        let mut counts = vec![0_usize; classes];
        for &label in targets.as_slice() {
            counts[targets.class_index(label).expect("observed label")] += 1;
        }
        let rows = targets.as_slice().len() as f32;
        SampleWeights::new(
            targets
                .as_slice()
                .iter()
                .map(|&label| {
                    let count = counts[targets.class_index(label).expect("observed label")];
                    rows / (classes as f32 * count as f32)
                })
                .collect(),
        )
        .expect("positive weights")
    }

    #[test]
    fn balanced_class_weights_equalize_class_totals_without_a_class_weight_parameter() {
        let targets = ClassTargets::new(vec![3, 3, 3, 3, 3, 3, 7, 7, 10]).unwrap();
        let weights = balanced(&targets);

        // Every class ends with the same total weight, which is the whole
        // content of "balanced".
        let mut per_class = vec![0.0_f64; targets.classes().len()];
        for (&label, &weight) in targets.as_slice().iter().zip(weights.as_slice()) {
            per_class[targets.class_index(label).unwrap()] += f64::from(weight);
        }
        let expected = per_class[0];
        for total in &per_class {
            assert!((total - expected).abs() < 1e-5, "{per_class:?}");
        }
        // And the weights still sum to the row count, so the rule rescales the
        // classes without inflating the training set.
        assert!((weights.total() - targets.as_slice().len() as f64).abs() < 1e-5);
    }

    #[test]
    fn an_already_balanced_problem_gets_unit_weights() {
        let targets = ClassTargets::new(vec![0, 0, 1, 1]).unwrap();
        assert_eq!(balanced(&targets).as_slice(), [1.0, 1.0, 1.0, 1.0]);
    }
}
