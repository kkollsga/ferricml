//! Order-respecting splitters.

use super::{Split, SplitError, validate_sample_count};

/// Forward-chaining splitter for ordered observations.
///
/// Rows are assumed to be in time order, index `0` oldest. Each split trains on
/// a prefix and tests on the window immediately after it, so no fold is ever
/// fitted on a row that comes after the rows it is evaluated on. Later folds
/// train on strictly more history than earlier ones.
///
/// Test windows all hold `samples / (n_splits + 1)` rows and are aligned to the
/// end of the dataset, so any remainder lengthens the first training window
/// rather than making the windows uneven. Every fold except the last therefore
/// leaves the rows after its test window out of both partitions — a deliberate
/// [`Split::partial`], because using them either way would leak the future.
///
/// ```
/// use ferricml::model_selection::TimeSeriesSplit;
///
/// let folds = TimeSeriesSplit::new(2).split(10)?.collect::<Vec<_>>();
/// assert_eq!(folds[0].train_indices(), &[0, 1, 2, 3]);
/// assert_eq!(folds[0].test_indices(), &[4, 5, 6]);
/// assert_eq!(folds[1].train_indices(), &[0, 1, 2, 3, 4, 5, 6]);
/// assert_eq!(folds[1].test_indices(), &[7, 8, 9]);
/// # Ok::<(), ferricml::model_selection::SplitError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    gap: usize,
}

impl TimeSeriesSplit {
    /// Creates a forward-chaining configuration with no gap.
    pub const fn new(n_splits: usize) -> Self {
        Self { n_splits, gap: 0 }
    }

    /// Drops this many rows between each training window and its test window.
    ///
    /// A gap is how a caller states that a target observed at time `t` is not
    /// knowable until `t + gap`, so training right up to the test window would
    /// leak.
    #[must_use]
    pub const fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Returns the number of splits.
    pub const fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Returns the gap between each training window and its test window.
    pub const fn gap(&self) -> usize {
        self.gap
    }

    /// Validates the sample count and returns an iterator over ordered splits.
    pub fn split(&self, samples: usize) -> Result<TimeSeriesSplitIter, SplitError> {
        validate_sample_count(samples)?;
        let invalid = || SplitError::InvalidTimeSeriesWindow {
            splits: self.n_splits,
            gap: self.gap,
            samples,
        };
        if self.n_splits == 0 {
            return Err(invalid());
        }
        let test_size = samples / (self.n_splits + 1);
        if test_size == 0 {
            return Err(invalid());
        }
        let first_test_start = samples - self.n_splits * test_size;
        if first_test_start <= self.gap {
            return Err(invalid());
        }
        Ok(TimeSeriesSplitIter {
            samples,
            test_size,
            first_test_start,
            gap: self.gap,
            n_splits: self.n_splits,
            next_split: 0,
        })
    }
}

/// Iterator over forward-chaining splits.
#[derive(Clone, Debug)]
pub struct TimeSeriesSplitIter {
    samples: usize,
    test_size: usize,
    first_test_start: usize,
    gap: usize,
    n_splits: usize,
    next_split: usize,
}

impl Iterator for TimeSeriesSplitIter {
    type Item = Split;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_split == self.n_splits {
            return None;
        }
        let test_start = self.first_test_start + self.next_split * self.test_size;
        self.next_split += 1;
        let train_end = test_start - self.gap;
        Some(
            Split::partial(
                self.samples,
                (0..train_end).collect(),
                (test_start..test_start + self.test_size).collect(),
            )
            .expect("forward-chaining windows are non-empty, ordered, and in bounds"),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_splits - self.next_split;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TimeSeriesSplitIter {}

/// Splitter that holds out one sample at a time.
///
/// This is K-fold with one fold per sample: the most training data any
/// resampling scheme can give each fit, at the cost of one fit per row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Creates the configuration.
    pub const fn new() -> Self {
        Self
    }

    /// Validates the sample count and returns one split per sample, in order.
    pub fn split(&self, samples: usize) -> Result<LeaveOneOutIter, SplitError> {
        validate_sample_count(samples)?;
        Ok(LeaveOneOutIter {
            samples,
            next_index: 0,
        })
    }
}

/// Iterator over leave-one-out splits.
#[derive(Clone, Debug)]
pub struct LeaveOneOutIter {
    samples: usize,
    next_index: usize,
}

impl Iterator for LeaveOneOutIter {
    type Item = Split;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index == self.samples {
            return None;
        }
        let index = self.next_index;
        self.next_index += 1;
        Some(Split::from_test_indices(self.samples, vec![index]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.samples - self.next_index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LeaveOneOutIter {}

#[cfg(test)]
mod tests {
    use super::*;

    fn windows(samples: usize, splitter: TimeSeriesSplit) -> Vec<(Vec<usize>, Vec<usize>)> {
        splitter
            .split(samples)
            .unwrap()
            .map(|split| {
                (
                    split.train_indices().to_vec(),
                    split.test_indices().to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn forward_chaining_windows_match_the_reference_layout() {
        assert_eq!(
            windows(10, TimeSeriesSplit::new(2)),
            vec![
                (vec![0, 1, 2, 3], vec![4, 5, 6]),
                (vec![0, 1, 2, 3, 4, 5, 6], vec![7, 8, 9]),
            ]
        );
        assert_eq!(
            windows(10, TimeSeriesSplit::new(3)),
            vec![
                (vec![0, 1, 2, 3], vec![4, 5]),
                (vec![0, 1, 2, 3, 4, 5], vec![6, 7]),
                (vec![0, 1, 2, 3, 4, 5, 6, 7], vec![8, 9]),
            ]
        );
        assert_eq!(
            windows(12, TimeSeriesSplit::new(3)),
            vec![
                (vec![0, 1, 2], vec![3, 4, 5]),
                (vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8]),
                (vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![9, 10, 11]),
            ]
        );
        assert_eq!(
            windows(7, TimeSeriesSplit::new(3)),
            vec![
                (vec![0, 1, 2, 3], vec![4]),
                (vec![0, 1, 2, 3, 4], vec![5]),
                (vec![0, 1, 2, 3, 4, 5], vec![6]),
            ]
        );
    }

    #[test]
    fn a_gap_removes_the_rows_immediately_before_each_test_window() {
        assert_eq!(
            windows(12, TimeSeriesSplit::new(3).with_gap(1)),
            vec![
                (vec![0, 1], vec![3, 4, 5]),
                (vec![0, 1, 2, 3, 4], vec![6, 7, 8]),
                (vec![0, 1, 2, 3, 4, 5, 6, 7], vec![9, 10, 11]),
            ]
        );
    }

    #[test]
    fn every_training_row_strictly_precedes_every_test_row() {
        for samples in 2..=40 {
            for n_splits in 1..samples {
                for gap in 0..3 {
                    let splitter = TimeSeriesSplit::new(n_splits).with_gap(gap);
                    let Ok(iterator) = splitter.split(samples) else {
                        continue;
                    };
                    let mut previous_train = 0;
                    let mut folds = 0;
                    for split in iterator {
                        let last_train = *split.train_indices().last().unwrap();
                        let first_test = split.test_indices()[0];
                        assert!(
                            last_train < first_test,
                            "samples={samples} splits={n_splits} gap={gap}: \
                             train ends at {last_train}, test starts at {first_test}"
                        );
                        assert_eq!(first_test - last_train, gap + 1);
                        assert_eq!(split.train_indices(), (0..=last_train).collect::<Vec<_>>());
                        assert!(
                            split
                                .test_indices()
                                .windows(2)
                                .all(|pair| pair[1] == pair[0] + 1)
                        );
                        assert!(last_train >= previous_train, "history must not shrink");
                        assert!(*split.test_indices().last().unwrap() < samples);
                        assert_eq!(split.sample_count(), samples);
                        previous_train = last_train;
                        folds += 1;
                    }
                    assert_eq!(folds, n_splits);
                }
            }
        }
    }

    #[test]
    fn test_windows_are_equal_sized_contiguous_and_end_at_the_last_row() {
        for samples in 2..=40 {
            for n_splits in 1..samples {
                let Ok(splits) = TimeSeriesSplit::new(n_splits).split(samples) else {
                    continue;
                };
                let splits = splits.collect::<Vec<_>>();
                let size = splits[0].test_indices().len();
                assert_eq!(size, samples / (n_splits + 1));
                let mut expected_start = splits[0].test_indices()[0];
                for split in &splits {
                    assert_eq!(split.test_indices().len(), size);
                    assert_eq!(split.test_indices()[0], expected_start);
                    expected_start += size;
                }
                assert_eq!(
                    *splits.last().unwrap().test_indices().last().unwrap(),
                    samples - 1
                );
            }
        }
    }

    #[test]
    fn impossible_windows_are_reported_rather_than_silently_shrunk() {
        assert_eq!(
            TimeSeriesSplit::new(1).split(1).unwrap_err(),
            SplitError::NotEnoughSamples { samples: 1 }
        );
        assert_eq!(
            TimeSeriesSplit::new(0).split(8).unwrap_err(),
            SplitError::InvalidTimeSeriesWindow {
                splits: 0,
                gap: 0,
                samples: 8,
            }
        );
        // Four splits need five windows, and four rows cannot fill them.
        assert_eq!(
            TimeSeriesSplit::new(4).split(4).unwrap_err(),
            SplitError::InvalidTimeSeriesWindow {
                splits: 4,
                gap: 0,
                samples: 4,
            }
        );
        // The gap would consume the whole first training window.
        assert_eq!(TimeSeriesSplit::new(2).split(6).unwrap().count(), 2);
        assert_eq!(
            TimeSeriesSplit::new(2).with_gap(2).split(6).unwrap_err(),
            SplitError::InvalidTimeSeriesWindow {
                splits: 2,
                gap: 2,
                samples: 6,
            }
        );
    }

    #[test]
    fn leave_one_out_holds_out_each_sample_exactly_once() {
        for samples in 2..=32 {
            let splits = LeaveOneOut::new().split(samples).unwrap();
            assert_eq!(splits.len(), samples);
            let mut held_out = vec![0_usize; samples];
            for (position, split) in splits.enumerate() {
                assert_eq!(split.test_indices(), &[position]);
                assert_eq!(split.train_indices().len(), samples - 1);
                assert_eq!(split.sample_count(), samples);
                assert_eq!(split.covered_samples(), samples);
                assert!(
                    split
                        .train_indices()
                        .windows(2)
                        .all(|pair| pair[0] < pair[1])
                );
                assert!(!split.train_indices().contains(&position));
                for &index in split.test_indices() {
                    held_out[index] += 1;
                }
            }
            assert_eq!(held_out, vec![1; samples]);
        }
        assert_eq!(
            LeaveOneOut::new().split(1).unwrap_err(),
            SplitError::NotEnoughSamples { samples: 1 }
        );
        assert_eq!(LeaveOneOut::new(), LeaveOneOut);
    }
}
