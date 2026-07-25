//! Repeated resampling over an existing splitter.

use super::{KFold, KFoldIter, Split, SplitError, repeat_seed, validate_fold_count};

/// K-fold splitter run several times with different shuffles.
///
/// One K-fold run estimates quality from a single partition of the data, and
/// that partition is itself a source of variance. Repeating it with a different
/// shuffle each time and reporting every fold lets a caller separate the
/// model's variance from the partition's.
///
/// Each repeat's shuffle seed is derived from the configured seed and the
/// repeat index, so the whole sequence is reproducible from one number and no
/// two repeats produce the same partition. Splits arrive repeat by repeat, in
/// fold order within each repeat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatedKFold {
    n_splits: usize,
    n_repeats: usize,
    random_state: u64,
}

impl RepeatedKFold {
    /// Creates a repeated K-fold configuration.
    pub const fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            random_state: 0,
        }
    }

    /// Sets the seed every repeat's shuffle is derived from.
    #[must_use]
    pub const fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// Returns the number of folds per repeat.
    pub const fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Returns the number of repeats.
    pub const fn n_repeats(&self) -> usize {
        self.n_repeats
    }

    /// Returns the seed every repeat's shuffle is derived from.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    /// Validates the sample count and returns an iterator over every split of
    /// every repeat.
    pub fn split(&self, samples: usize) -> Result<RepeatedKFoldIter, SplitError> {
        validate_fold_count(samples, self.n_splits)?;
        if self.n_repeats == 0 {
            return Err(SplitError::InvalidRepeatCount { repeats: 0 });
        }
        let mut iterator = RepeatedKFoldIter {
            samples,
            n_splits: self.n_splits,
            n_repeats: self.n_repeats,
            random_state: self.random_state,
            next_repeat: 0,
            folds: None,
        };
        iterator.start_repeat()?;
        Ok(iterator)
    }
}

/// Iterator over every fold of every repeat.
#[derive(Clone, Debug)]
pub struct RepeatedKFoldIter {
    samples: usize,
    n_splits: usize,
    n_repeats: usize,
    random_state: u64,
    next_repeat: usize,
    folds: Option<KFoldIter>,
}

impl RepeatedKFoldIter {
    fn start_repeat(&mut self) -> Result<(), SplitError> {
        let seed = repeat_seed(self.random_state, self.next_repeat);
        self.folds = Some(
            KFold::new(self.n_splits)
                .with_shuffle(true)
                .with_random_state(seed)
                .split(self.samples)?,
        );
        self.next_repeat += 1;
        Ok(())
    }
}

impl Iterator for RepeatedKFoldIter {
    type Item = Split;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(split) = self.folds.as_mut()?.next() {
                return Some(split);
            }
            if self.next_repeat == self.n_repeats {
                self.folds = None;
                return None;
            }
            self.start_repeat()
                .expect("the sample and fold counts were validated once");
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_repeats = self.n_repeats - self.next_repeat;
        let remaining = self.folds.as_ref().map_or(0, ExactSizeIterator::len)
            + remaining_repeats * self.n_splits;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RepeatedKFoldIter {}

#[cfg(test)]
mod tests {
    use super::*;

    fn folds(splitter: RepeatedKFold, samples: usize) -> Vec<Vec<usize>> {
        splitter
            .split(samples)
            .unwrap()
            .map(|split| split.test_indices().to_vec())
            .collect()
    }

    #[test]
    fn every_repeat_covers_every_sample_exactly_once() {
        for samples in 2..=24 {
            for n_splits in 2..=samples.min(5) {
                for n_repeats in 1..=3 {
                    let splits = RepeatedKFold::new(n_splits, n_repeats)
                        .with_random_state(11)
                        .split(samples)
                        .unwrap();
                    assert_eq!(splits.len(), n_splits * n_repeats);
                    let mut held_out = vec![0_usize; samples];
                    let mut seen = 0;
                    for split in splits {
                        assert_eq!(split.sample_count(), samples);
                        assert_eq!(split.covered_samples(), samples);
                        for &index in split.test_indices() {
                            held_out[index] += 1;
                        }
                        seen += 1;
                    }
                    assert_eq!(seen, n_splits * n_repeats);
                    assert_eq!(held_out, vec![n_repeats; samples]);
                }
            }
        }
    }

    #[test]
    fn repeats_partition_differently_but_reproducibly() {
        let splitter = RepeatedKFold::new(3, 3).with_random_state(7);
        let first = folds(splitter, 30);
        assert_eq!(first, folds(splitter, 30));
        assert_eq!(first.len(), 9);
        assert_ne!(first[0..3], first[3..6]);
        assert_ne!(first[3..6], first[6..9]);
        assert_ne!(first[0..3], first[6..9]);

        // A different seed gives a different sequence.
        assert_ne!(
            first,
            folds(RepeatedKFold::new(3, 3).with_random_state(8), 30)
        );
        // Consecutive seeds do not shift the same partition by one repeat.
        let shifted = folds(RepeatedKFold::new(3, 3).with_random_state(8), 30);
        assert_ne!(first[3..6], shifted[0..3]);
    }

    #[test]
    fn one_repeat_is_ordinary_shuffled_k_fold() {
        let seed = repeat_seed(4, 0);
        let repeated = folds(RepeatedKFold::new(4, 1).with_random_state(4), 20);
        let plain = KFold::new(4)
            .with_shuffle(true)
            .with_random_state(seed)
            .split(20)
            .unwrap()
            .map(|split| split.test_indices().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(repeated, plain);
    }

    #[test]
    fn invalid_shapes_are_reported_before_any_partitioning() {
        assert_eq!(
            RepeatedKFold::new(2, 0).split(8).unwrap_err(),
            SplitError::InvalidRepeatCount { repeats: 0 }
        );
        assert_eq!(
            RepeatedKFold::new(9, 2).split(8).unwrap_err(),
            SplitError::InvalidFoldCount {
                folds: 9,
                samples: 8,
            }
        );
        assert_eq!(
            RepeatedKFold::new(2, 2).split(1).unwrap_err(),
            SplitError::NotEnoughSamples { samples: 1 }
        );
    }

    #[test]
    fn the_remaining_count_stays_exact_while_iterating() {
        let mut splits = RepeatedKFold::new(3, 2)
            .with_random_state(1)
            .split(9)
            .unwrap();
        for remaining in (0..6).rev() {
            assert!(splits.next().is_some());
            assert_eq!(splits.len(), remaining);
        }
        assert!(splits.next().is_none());
        assert_eq!(splits.len(), 0);
    }
}
