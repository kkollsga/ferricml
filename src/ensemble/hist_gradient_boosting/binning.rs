//! Deterministic per-feature threshold fitting and bin assignment.

use crate::data::MatrixView;

use super::error::{BoostingError, MAX_BINS};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Binner {
    thresholds: Vec<Vec<f32>>,
}

impl Binner {
    /// Fits the grid over the rows that are in the training sample.
    ///
    /// # Why the weights are read here at all
    ///
    /// The grid itself is **not** weighted: a row's weight does not move a
    /// threshold, which is exactly why weighting a row and repeating it agree.
    /// What the weights decide is *membership*. FerricML's semantic across the
    /// whole crate is that a row of zero weight is not in the training sample —
    /// `tree::grower` drops such rows from its row list and the forest
    /// resamples only positively weighted rows — and a row that is not in the
    /// sample cannot contribute a distinct feature value for the grid to place
    /// a bin edge on.
    ///
    /// Reading every row instead was a real divergence, not a cosmetic one: the
    /// statistics already ignore a zero-weight row, so it influenced the fit
    /// through the grid and through nothing else. Measured on nine rows over
    /// one column with the row holding the unique value `4.0` zero-weighted,
    /// the zero-weighted fit and the deleted-row fit disagreed by up to `1.35`
    /// on a target range of `0..20`, at every `max_bins` from `3` to `16`.
    ///
    /// [`SampleWeights`](crate::data::SampleWeights) guarantees at least one
    /// positive weight, so no column is left with nothing to bin.
    pub(crate) fn fit(
        data: &MatrixView<'_>,
        max_bins: usize,
        sample_weights: Option<&[f32]>,
    ) -> Result<Self, BoostingError> {
        if !(2..=MAX_BINS).contains(&max_bins) {
            return Err(BoostingError::InvalidMaxBins);
        }
        if data.columns() > u32::MAX as usize {
            return Err(BoostingError::TooManyFeatures);
        }
        if let Some(weights) = sample_weights
            && weights.len() != data.rows()
        {
            return Err(BoostingError::ResidualLength {
                rows: data.rows(),
                residuals: weights.len(),
            });
        }
        let mut thresholds = Vec::with_capacity(data.columns());
        for column in 0..data.columns() {
            let mut unique = data
                .iter_rows()
                .enumerate()
                .filter(|(row, _)| sample_weights.is_none_or(|weights| weights[*row] > 0.0))
                .map(|(_, row)| row[column])
                .collect::<Vec<_>>();
            unique.sort_by(f32::total_cmp);
            unique.dedup_by(|left, right| *left == *right);
            let mut feature_thresholds = Vec::with_capacity(unique.len().min(max_bins) - 1);
            if unique.len() <= max_bins {
                for values in unique.windows(2) {
                    feature_thresholds.push(midpoint(values[0], values[1]));
                }
            } else {
                for bin in 1..max_bins {
                    let split = bin * unique.len() / max_bins;
                    let threshold = midpoint(unique[split - 1], unique[split]);
                    if feature_thresholds.last().copied() != Some(threshold) {
                        feature_thresholds.push(threshold);
                    }
                }
            }
            thresholds.push(feature_thresholds);
        }
        Ok(Self { thresholds })
    }

    pub(crate) fn transform(&self, data: &MatrixView<'_>) -> Result<BinnedMatrix, BoostingError> {
        if data.columns() != self.thresholds.len() {
            return Err(BoostingError::FeatureDimension {
                expected: self.thresholds.len(),
                actual: data.columns(),
            });
        }
        let mut bins = Vec::with_capacity(data.as_slice().len());
        for row in data.iter_rows() {
            for (column, &value) in row.iter().enumerate() {
                let bin = self.thresholds[column].partition_point(|&threshold| value > threshold);
                bins.push(u8::try_from(bin).expect("max_bins bounds every bin"));
            }
        }
        Ok(BinnedMatrix {
            bins,
            rows: data.rows(),
            columns: data.columns(),
        })
    }

    pub(crate) fn n_features_in(&self) -> usize {
        self.thresholds.len()
    }

    #[allow(dead_code)]
    pub(crate) fn thresholds(&self) -> &[Vec<f32>] {
        &self.thresholds
    }

    pub(crate) fn threshold(&self, feature: usize, bin: u8) -> f32 {
        self.thresholds[feature][usize::from(bin)]
    }
}

fn midpoint(left: f32, right: f32) -> f32 {
    let midpoint = ((f64::from(left) + f64::from(right)) * 0.5) as f32;
    if midpoint > left && midpoint < right {
        midpoint
    } else {
        left
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BinnedMatrix {
    bins: Vec<u8>,
    rows: usize,
    columns: usize,
}

impl BinnedMatrix {
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    #[allow(dead_code)]
    pub(crate) fn row(&self, index: usize) -> Option<&[u8]> {
        let start = index.checked_mul(self.columns)?;
        self.bins.get(start..start + self.columns)
    }

    pub(crate) fn get(&self, row: usize, column: usize) -> Option<u8> {
        if column >= self.columns {
            return None;
        }
        self.bins
            .get(row.checked_mul(self.columns)? + column)
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    #[test]
    fn exact_unique_values_produce_midpoint_thresholds_and_bins() {
        let data = DenseMatrix::new(vec![0.0, 5.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0], 4, 2).unwrap();
        let binner = Binner::fit(&data.as_view(), 8, None).unwrap();
        assert_eq!(binner.thresholds(), &[vec![0.5, 1.5, 2.5], vec![]]);
        let binned = binner.transform(&data.as_view()).unwrap();
        assert_eq!(binned.row(0), Some(&[0, 0][..]));
        assert_eq!(binned.row(1), Some(&[1, 0][..]));
        assert_eq!(binned.row(3), Some(&[3, 0][..]));
    }

    #[test]
    fn quantile_thresholds_are_bounded_and_deterministic() {
        let data = DenseMatrix::new((0..20).map(|value| value as f32).collect(), 20, 1).unwrap();
        let first = Binner::fit(&data.as_view(), 4, None).unwrap();
        let second = Binner::fit(&data.as_view(), 4, None).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.thresholds(), &[vec![4.5, 9.5, 14.5]]);
        let binned = first.transform(&data.as_view()).unwrap();
        assert_eq!(binned.get(0, 0), Some(0));
        assert_eq!(binned.get(5, 0), Some(1));
        assert_eq!(binned.get(19, 0), Some(3));
    }

    #[test]
    fn validates_bin_count_and_feature_handoff() {
        let data = DenseMatrix::new(vec![0.0, 1.0], 2, 1).unwrap();
        assert_eq!(
            Binner::fit(&data.as_view(), 1, None),
            Err(BoostingError::InvalidMaxBins)
        );
        let binner = Binner::fit(&data.as_view(), 2, None).unwrap();
        let wider = DenseMatrix::new(vec![0.0, 1.0], 1, 2).unwrap();
        assert_eq!(
            binner.transform(&wider.as_view()),
            Err(BoostingError::FeatureDimension {
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn extreme_and_adjacent_midpoints_preserve_observed_order() {
        let adjacent = f32::from_bits(1.0_f32.to_bits() + 1);
        let data = DenseMatrix::new(vec![-f32::MAX, 1.0, f32::MAX, adjacent], 2, 2).unwrap();
        let binner = Binner::fit(&data.as_view(), 2, None).unwrap();
        assert_eq!(binner.thresholds()[0], vec![0.0]);
        assert_eq!(binner.thresholds()[1], vec![1.0]);
        let binned = binner.transform(&data.as_view()).unwrap();
        assert_eq!(binned.row(0), Some(&[0, 0][..]));
        assert_eq!(binned.row(1), Some(&[1, 1][..]));
    }
}
