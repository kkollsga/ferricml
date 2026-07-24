//! Deterministic validated dataset partitions.

use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;

/// Identifies one side of a train/test split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitPartition {
    /// Indices used for fitting.
    Train,
    /// Indices held out for evaluation.
    Test,
}

/// Errors produced while constructing deterministic dataset splits.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplitError {
    /// At least two samples are required for a non-empty train/test split.
    NotEnoughSamples {
        /// Available sample count.
        samples: usize,
    },
    /// A requested test count would leave an empty train or test partition.
    InvalidTestCount {
        /// Requested test samples.
        test: usize,
        /// Available sample count.
        samples: usize,
    },
    /// A test fraction was not finite and strictly between zero and one.
    InvalidTestFraction,
    /// A fold count was outside `2..=samples`.
    InvalidFoldCount {
        /// Requested fold count.
        folds: usize,
        /// Available sample count.
        samples: usize,
    },
    /// A custom split contained an empty partition.
    EmptyPartition {
        /// Empty side of the split.
        partition: SplitPartition,
    },
    /// A custom split index was outside the dataset.
    IndexOutOfBounds {
        /// Side containing the invalid index.
        partition: SplitPartition,
        /// Position within that side's index list.
        position: usize,
        /// Invalid dataset index.
        index: usize,
        /// Available sample count.
        samples: usize,
    },
    /// A custom split repeated an index within one partition.
    DuplicateIndex {
        /// Side containing the repeated index.
        partition: SplitPartition,
        /// Repeated dataset index.
        index: usize,
    },
    /// A custom split placed one index in both partitions.
    OverlappingIndex {
        /// Overlapping dataset index.
        index: usize,
    },
    /// A custom split did not cover the dataset exactly once.
    IncompleteCoverage {
        /// Required number of covered samples.
        expected: usize,
        /// Number of supplied indices.
        actual: usize,
    },
    /// A class did not contain enough rows for every requested partition.
    InsufficientClassMembers {
        /// Class label.
        label: u8,
        /// Rows carrying this label.
        count: usize,
        /// Required partition count.
        partitions: usize,
    },
    /// A stratified holdout side could not contain every observed class.
    PartitionTooSmallForClasses {
        /// Side that is too small.
        partition: SplitPartition,
        /// Rows assigned to that side.
        rows: usize,
        /// Number of observed classes.
        classes: usize,
    },
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEnoughSamples { samples } => {
                write!(f, "split requires at least two samples, got {samples}")
            }
            Self::InvalidTestCount { test, samples } => write!(
                f,
                "test count {test} must leave non-empty partitions for {samples} samples"
            ),
            Self::InvalidTestFraction => {
                f.write_str("test fraction must be finite and strictly between zero and one")
            }
            Self::InvalidFoldCount { folds, samples } => {
                write!(f, "fold count {folds} must be in 2..={samples}")
            }
            Self::EmptyPartition { partition } => {
                write!(f, "{partition:?} partition must not be empty")
            }
            Self::IndexOutOfBounds {
                partition,
                position,
                index,
                samples,
            } => write!(
                f,
                "{partition:?} index {index} at position {position} is outside 0..{samples}"
            ),
            Self::DuplicateIndex { partition, index } => {
                write!(f, "{partition:?} partition repeats index {index}")
            }
            Self::OverlappingIndex { index } => {
                write!(f, "train and test partitions overlap at index {index}")
            }
            Self::IncompleteCoverage { expected, actual } => write!(
                f,
                "split must cover {expected} samples exactly once, got {actual} indices"
            ),
            Self::InsufficientClassMembers {
                label,
                count,
                partitions,
            } => write!(
                f,
                "class {label} has {count} rows but needs at least {partitions}"
            ),
            Self::PartitionTooSmallForClasses {
                partition,
                rows,
                classes,
            } => write!(
                f,
                "{partition:?} partition has {rows} rows for {classes} observed classes"
            ),
        }
    }
}

impl Error for SplitError {}

/// Owned indices for one complete train/test partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
}

impl Split {
    /// Validates that the two index lists are non-empty, disjoint, unique,
    /// in-bounds, and cover every sample exactly once.
    pub fn new(
        samples: usize,
        train_indices: Vec<usize>,
        test_indices: Vec<usize>,
    ) -> Result<Self, SplitError> {
        validate_sample_count(samples)?;
        if train_indices.is_empty() {
            return Err(SplitError::EmptyPartition {
                partition: SplitPartition::Train,
            });
        }
        if test_indices.is_empty() {
            return Err(SplitError::EmptyPartition {
                partition: SplitPartition::Test,
            });
        }
        validate_bounds(samples, SplitPartition::Train, &train_indices)?;
        validate_bounds(samples, SplitPartition::Test, &test_indices)?;
        let actual = train_indices.len().saturating_add(test_indices.len());
        if actual != samples {
            return Err(SplitError::IncompleteCoverage {
                expected: samples,
                actual,
            });
        }

        let mut membership = vec![0_u8; samples];
        for &index in &train_indices {
            if membership[index] != 0 {
                return Err(SplitError::DuplicateIndex {
                    partition: SplitPartition::Train,
                    index,
                });
            }
            membership[index] = 1;
        }
        for &index in &test_indices {
            match membership[index] {
                0 => membership[index] = 2,
                1 => return Err(SplitError::OverlappingIndex { index }),
                _ => {
                    return Err(SplitError::DuplicateIndex {
                        partition: SplitPartition::Test,
                        index,
                    });
                }
            }
        }
        if membership.contains(&0) {
            return Err(SplitError::IncompleteCoverage {
                expected: samples,
                actual,
            });
        }
        Ok(Self {
            train_indices,
            test_indices,
        })
    }

    /// Dataset indices used for fitting.
    pub fn train_indices(&self) -> &[usize] {
        &self.train_indices
    }

    /// Dataset indices held out for evaluation.
    pub fn test_indices(&self) -> &[usize] {
        &self.test_indices
    }

    /// Total number of samples covered by this split.
    pub fn sample_count(&self) -> usize {
        self.train_indices.len() + self.test_indices.len()
    }

    fn from_test_indices(samples: usize, mut test_indices: Vec<usize>) -> Self {
        test_indices.sort_unstable();
        let mut train_indices = Vec::with_capacity(samples - test_indices.len());
        let mut test_position = 0;
        for index in 0..samples {
            if test_indices.get(test_position) == Some(&index) {
                test_position += 1;
            } else {
                train_indices.push(index);
            }
        }
        Self {
            train_indices,
            test_indices,
        }
    }
}

/// Requested holdout test size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TestSize {
    /// Exact number of held-out rows.
    Count(usize),
    /// Fraction of rows, rounded upward.
    Fraction(f64),
}

/// Parameters for deterministic train/test splitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoldoutParams {
    test_size: TestSize,
    shuffle: bool,
    random_state: u64,
}

impl Default for HoldoutParams {
    fn default() -> Self {
        Self {
            test_size: TestSize::Fraction(0.25),
            shuffle: true,
            random_state: 0,
        }
    }
}

impl HoldoutParams {
    /// Sets the exact or fractional test size.
    #[must_use]
    pub fn with_test_size(mut self, test_size: TestSize) -> Self {
        self.test_size = test_size;
        self
    }

    /// Enables or disables deterministic shuffling before membership is chosen.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the deterministic shuffle seed.
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// Returns the requested test size.
    pub const fn test_size(&self) -> TestSize {
        self.test_size
    }

    /// Returns whether membership is shuffled.
    pub const fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Returns the deterministic shuffle seed.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }
}

/// Splits sample indices into one train and test partition.
pub fn train_test_split(samples: usize, params: HoldoutParams) -> Result<Split, SplitError> {
    let test_count = resolve_test_count(samples, params.test_size)?;
    let train_count = samples - test_count;
    if !params.shuffle {
        return Ok(Split {
            train_indices: (0..train_count).collect(),
            test_indices: (train_count..samples).collect(),
        });
    }

    let mut order = (0..samples).collect::<Vec<_>>();
    let mut test_membership = vec![0_u8; samples];
    let mut rng = StableRng::new(params.random_state);
    for index in (train_count..samples).rev() {
        let other = rng.index(index + 1);
        order.swap(index, other);
        test_membership[order[index]] = 1;
    }
    drop(order);
    Ok(split_from_test_membership(&test_membership, test_count))
}

/// Splits indices while preserving every observed label in both partitions.
pub fn stratified_train_test_split(
    labels: &[u8],
    params: HoldoutParams,
) -> Result<Split, SplitError> {
    let samples = labels.len();
    let test_count = resolve_test_count(samples, params.test_size)?;
    let counts = class_counts(labels);
    let classes = counts.iter().filter(|&&count| count > 0).count();
    let train_count = samples - test_count;
    if test_count < classes {
        return Err(SplitError::PartitionTooSmallForClasses {
            partition: SplitPartition::Test,
            rows: test_count,
            classes,
        });
    }
    if train_count < classes {
        return Err(SplitError::PartitionTooSmallForClasses {
            partition: SplitPartition::Train,
            rows: train_count,
            classes,
        });
    }
    for (label, &count) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
        if count < 2 {
            return Err(SplitError::InsufficientClassMembers {
                label: label as u8,
                count,
                partitions: 2,
            });
        }
    }

    let mut remaining_quotas = stratified_test_quotas(&counts, test_count);
    let mut remaining_total = test_count;
    let mut test_membership = vec![0_u8; samples];
    if params.shuffle {
        let mut order = (0..samples).collect::<Vec<_>>();
        let mut rng = StableRng::new(params.random_state);
        for position in (1..samples).rev() {
            let other = rng.index(position + 1);
            order.swap(position, other);
            let index = order[position];
            let label = labels[index] as usize;
            if remaining_quotas[label] > 0 {
                test_membership[index] = 1;
                remaining_quotas[label] -= 1;
                remaining_total -= 1;
                if remaining_total == 0 {
                    break;
                }
            }
        }
    } else {
        for index in (0..samples).rev() {
            let label = labels[index] as usize;
            if remaining_quotas[label] > 0 {
                test_membership[index] = 1;
                remaining_quotas[label] -= 1;
                remaining_total -= 1;
                if remaining_total == 0 {
                    break;
                }
            }
        }
    }
    debug_assert_eq!(remaining_total, 0);
    Ok(split_from_test_membership(&test_membership, test_count))
}

/// Deterministic K-fold splitter configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: u64,
}

impl KFold {
    /// Creates an unshuffled K-fold configuration.
    pub const fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: 0,
        }
    }

    /// Enables or disables deterministic shuffling before folds are assigned.
    #[must_use]
    pub const fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the deterministic shuffle seed.
    #[must_use]
    pub const fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// Returns the number of folds.
    pub const fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Returns whether membership is shuffled.
    pub const fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Returns the deterministic shuffle seed.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    /// Validates the sample count and returns an iterator over complete splits.
    pub fn split(&self, samples: usize) -> Result<KFoldIter, SplitError> {
        validate_fold_count(samples, self.n_splits)?;
        let mut order = (0..samples).collect::<Vec<_>>();
        if self.shuffle {
            stable_shuffle(&mut order, self.random_state);
        }
        Ok(KFoldIter {
            order,
            n_splits: self.n_splits,
            next_fold: 0,
        })
    }
}

/// Iterator over deterministic K-fold partitions.
#[derive(Clone, Debug)]
pub struct KFoldIter {
    order: Vec<usize>,
    n_splits: usize,
    next_fold: usize,
}

impl Iterator for KFoldIter {
    type Item = Split;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_fold == self.n_splits {
            return None;
        }
        let samples = self.order.len();
        let (start, end) = fold_bounds(samples, self.n_splits, self.next_fold);
        self.next_fold += 1;
        Some(Split::from_test_indices(
            samples,
            self.order[start..end].to_vec(),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_splits - self.next_fold;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for KFoldIter {}

/// Deterministic label-stratified K-fold splitter configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: u64,
}

impl StratifiedKFold {
    /// Creates an unshuffled stratified K-fold configuration.
    pub const fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: 0,
        }
    }

    /// Enables or disables deterministic shuffling within each class.
    #[must_use]
    pub const fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the deterministic shuffle seed.
    #[must_use]
    pub const fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// Returns the number of folds.
    pub const fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Returns whether class membership is shuffled.
    pub const fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Returns the deterministic shuffle seed.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    /// Validates labels and returns an iterator over stratified splits.
    pub fn split(&self, labels: &[u8]) -> Result<StratifiedKFoldIter, SplitError> {
        let samples = labels.len();
        validate_fold_count(samples, self.n_splits)?;
        let counts = class_counts(labels);
        for (label, &count) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
            if count < self.n_splits {
                return Err(SplitError::InsufficientClassMembers {
                    label: label as u8,
                    count,
                    partitions: self.n_splits,
                });
            }
        }

        let mut buckets = (0..=u8::MAX)
            .map(|_| Vec::<usize>::new())
            .collect::<Vec<_>>();
        for (index, &label) in labels.iter().enumerate() {
            buckets[label as usize].push(index);
        }
        if self.shuffle {
            let mut rng = StableRng::new(self.random_state);
            for bucket in &mut buckets {
                shuffle_with_rng(bucket, &mut rng);
            }
        }

        let mut assignments = vec![0_usize; samples];
        let mut offset = 0;
        for bucket in buckets {
            for (position, index) in bucket.iter().copied().enumerate() {
                assignments[index] = (offset + position) % self.n_splits;
            }
            offset = (offset + bucket.len()) % self.n_splits;
        }
        Ok(StratifiedKFoldIter {
            assignments,
            n_splits: self.n_splits,
            next_fold: 0,
        })
    }
}

/// Iterator over deterministic stratified K-fold partitions.
#[derive(Clone, Debug)]
pub struct StratifiedKFoldIter {
    assignments: Vec<usize>,
    n_splits: usize,
    next_fold: usize,
}

impl Iterator for StratifiedKFoldIter {
    type Item = Split;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_fold == self.n_splits {
            return None;
        }
        let fold = self.next_fold;
        self.next_fold += 1;
        let test_indices = self
            .assignments
            .iter()
            .enumerate()
            .filter_map(|(index, &assigned)| (assigned == fold).then_some(index))
            .collect();
        Some(Split::from_test_indices(
            self.assignments.len(),
            test_indices,
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_splits - self.next_fold;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for StratifiedKFoldIter {}

fn validate_sample_count(samples: usize) -> Result<(), SplitError> {
    if samples < 2 {
        return Err(SplitError::NotEnoughSamples { samples });
    }
    Ok(())
}

fn validate_bounds(
    samples: usize,
    partition: SplitPartition,
    indices: &[usize],
) -> Result<(), SplitError> {
    if let Some((position, &index)) = indices
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= samples)
    {
        return Err(SplitError::IndexOutOfBounds {
            partition,
            position,
            index,
            samples,
        });
    }
    Ok(())
}

fn resolve_test_count(samples: usize, test_size: TestSize) -> Result<usize, SplitError> {
    validate_sample_count(samples)?;
    let test = match test_size {
        TestSize::Count(count) => count,
        TestSize::Fraction(fraction) => {
            if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
                return Err(SplitError::InvalidTestFraction);
            }
            (samples as f64 * fraction).ceil() as usize
        }
    };
    if test == 0 || test >= samples {
        return Err(SplitError::InvalidTestCount { test, samples });
    }
    Ok(test)
}

fn validate_fold_count(samples: usize, folds: usize) -> Result<(), SplitError> {
    validate_sample_count(samples)?;
    if folds < 2 || folds > samples {
        return Err(SplitError::InvalidFoldCount { folds, samples });
    }
    Ok(())
}

fn fold_bounds(samples: usize, folds: usize, fold: usize) -> (usize, usize) {
    let base = samples / folds;
    let larger = samples % folds;
    let start = fold * base + fold.min(larger);
    let size = base + usize::from(fold < larger);
    (start, start + size)
}

fn class_counts(labels: &[u8]) -> [usize; 256] {
    let mut counts = [0_usize; 256];
    for &label in labels {
        counts[label as usize] += 1;
    }
    counts
}

fn split_from_test_membership(test_membership: &[u8], test_count: usize) -> Split {
    let mut train_indices = Vec::with_capacity(test_membership.len() - test_count);
    let mut test_indices = Vec::with_capacity(test_count);
    for (index, &is_test) in test_membership.iter().enumerate() {
        if is_test != 0 {
            test_indices.push(index);
        } else {
            train_indices.push(index);
        }
    }
    Split {
        train_indices,
        test_indices,
    }
}

fn stratified_test_quotas(counts: &[usize; 256], test_count: usize) -> [usize; 256] {
    let mut quotas = counts.map(|count| usize::from(count > 0));
    let mut remaining = test_count - quotas.iter().sum::<usize>();
    let mut candidates = counts
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(label, count)| {
            (count > 2).then_some(QuotaCandidate {
                label,
                quota: 1,
                count,
            })
        })
        .collect::<BinaryHeap<_>>();
    while remaining > 0 {
        let mut candidate = candidates
            .pop()
            .expect("validated train partition leaves class capacity");
        candidate.quota += 1;
        quotas[candidate.label] = candidate.quota;
        remaining -= 1;
        if candidate.quota + 1 < candidate.count {
            candidates.push(candidate);
        }
    }
    quotas
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QuotaCandidate {
    label: usize,
    quota: usize,
    count: usize,
}

impl Ord for QuotaCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_ratio = (self.quota as u128) * (other.count as u128);
        let other_ratio = (other.quota as u128) * (self.count as u128);
        other_ratio
            .cmp(&self_ratio)
            .then_with(|| other.label.cmp(&self.label))
    }
}

impl PartialOrd for QuotaCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn stable_shuffle(values: &mut [usize], seed: u64) {
    shuffle_with_rng(values, &mut StableRng::new(seed));
}

fn shuffle_with_rng(values: &mut [usize], rng: &mut StableRng) {
    for index in (1..values.len()).rev() {
        let other = rng.index(index + 1);
        values.swap(index, other);
    }
}

/// Private SplitMix64 stream with rejection-sampled bounded indices. This is
/// deliberately independent from fitted-model random streams.
struct StableRng {
    state: u64,
}

impl StableRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bound = upper as u64;
        let reject_below = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                return (value % bound) as usize;
            }
        }
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_train_test_split(samples: usize, params: HoldoutParams) -> Result<Split, SplitError> {
        let test_count = resolve_test_count(samples, params.test_size)?;
        let mut order = (0..samples).collect::<Vec<_>>();
        if params.shuffle {
            stable_shuffle(&mut order, params.random_state);
        }
        let train_count = samples - test_count;
        let mut train_indices = order[..train_count].to_vec();
        let mut test_indices = order[train_count..].to_vec();
        train_indices.sort_unstable();
        test_indices.sort_unstable();
        Ok(Split {
            train_indices,
            test_indices,
        })
    }

    fn legacy_stratified_train_test_split(
        labels: &[u8],
        params: HoldoutParams,
    ) -> Result<Split, SplitError> {
        let samples = labels.len();
        let test_count = resolve_test_count(samples, params.test_size)?;
        let counts = class_counts(labels);
        let classes = counts.iter().filter(|&&count| count > 0).count();
        let train_count = samples - test_count;
        if test_count < classes {
            return Err(SplitError::PartitionTooSmallForClasses {
                partition: SplitPartition::Test,
                rows: test_count,
                classes,
            });
        }
        if train_count < classes {
            return Err(SplitError::PartitionTooSmallForClasses {
                partition: SplitPartition::Train,
                rows: train_count,
                classes,
            });
        }
        for (label, &count) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
            if count < 2 {
                return Err(SplitError::InsufficientClassMembers {
                    label: label as u8,
                    count,
                    partitions: 2,
                });
            }
        }

        let mut order = (0..samples).collect::<Vec<_>>();
        if params.shuffle {
            stable_shuffle(&mut order, params.random_state);
        }
        let mut buckets = (0..=u8::MAX).map(|_| Vec::new()).collect::<Vec<_>>();
        for index in order {
            buckets[labels[index] as usize].push(index);
        }
        let quotas = legacy_stratified_test_quotas(&counts, test_count);
        let mut train_indices = Vec::with_capacity(train_count);
        let mut test_indices = Vec::with_capacity(test_count);
        for (label, bucket) in buckets.into_iter().enumerate() {
            let test_start = bucket.len().saturating_sub(quotas[label]);
            train_indices.extend_from_slice(&bucket[..test_start]);
            test_indices.extend_from_slice(&bucket[test_start..]);
        }
        train_indices.sort_unstable();
        test_indices.sort_unstable();
        Ok(Split {
            train_indices,
            test_indices,
        })
    }

    fn legacy_stratified_test_quotas(counts: &[usize; 256], test_count: usize) -> [usize; 256] {
        let mut quotas = counts.map(|count| usize::from(count > 0));
        let mut remaining = test_count - quotas.iter().sum::<usize>();
        while remaining > 0 {
            let label = (0..quotas.len())
                .filter(|&label| quotas[label] + 1 < counts[label])
                .min_by(|&left, &right| {
                    let left_ratio = (quotas[left] as u128) * (counts[right] as u128);
                    let right_ratio = (quotas[right] as u128) * (counts[left] as u128);
                    left_ratio.cmp(&right_ratio).then_with(|| left.cmp(&right))
                })
                .expect("validated train partition leaves class capacity");
            quotas[label] += 1;
            remaining -= 1;
        }
        quotas
    }

    #[test]
    fn custom_split_validates_complete_disjoint_coverage() {
        assert_eq!(
            Split::new(4, vec![0, 1, 2], vec![3])
                .unwrap()
                .train_indices(),
            &[0, 1, 2]
        );
        assert_eq!(
            Split::new(4, vec![0, 1], vec![1, 3]),
            Err(SplitError::OverlappingIndex { index: 1 })
        );
        assert_eq!(
            Split::new(4, vec![0, 0], vec![2, 3]),
            Err(SplitError::DuplicateIndex {
                partition: SplitPartition::Train,
                index: 0,
            })
        );
        assert_eq!(
            Split::new(4, vec![0, 1], vec![2]),
            Err(SplitError::IncompleteCoverage {
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(
            Split::new(4, vec![0, 4], vec![2, 3]),
            Err(SplitError::IndexOutOfBounds {
                partition: SplitPartition::Train,
                position: 1,
                index: 4,
                samples: 4,
            })
        );
    }

    #[test]
    fn holdout_rounding_order_and_seed_are_frozen() {
        let unshuffled = train_test_split(
            10,
            HoldoutParams::default()
                .with_test_size(TestSize::Fraction(0.21))
                .with_shuffle(false),
        )
        .unwrap();
        assert_eq!(unshuffled.train_indices(), &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(unshuffled.test_indices(), &[7, 8, 9]);

        let shuffled = train_test_split(
            10,
            HoldoutParams::default()
                .with_test_size(TestSize::Count(3))
                .with_random_state(42),
        )
        .unwrap();
        assert_eq!(shuffled.train_indices(), &[0, 4, 5, 6, 7, 8, 9]);
        assert_eq!(shuffled.test_indices(), &[1, 2, 3]);
    }

    #[test]
    fn holdout_membership_matches_legacy_for_all_small_partitions() {
        for samples in 2..=128 {
            for test_count in 1..samples {
                for shuffle in [false, true] {
                    for seed in [0, 1, 42, u64::MAX] {
                        let params = HoldoutParams::default()
                            .with_test_size(TestSize::Count(test_count))
                            .with_shuffle(shuffle)
                            .with_random_state(seed);
                        assert_eq!(
                            train_test_split(samples, params),
                            legacy_train_test_split(samples, params),
                            "samples={samples}, test={test_count}, shuffle={shuffle}, seed={seed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn holdout_rejects_invalid_sizes_before_membership_allocation() {
        assert_eq!(
            train_test_split(1, HoldoutParams::default()),
            Err(SplitError::NotEnoughSamples { samples: 1 })
        );
        for fraction in [f64::NAN, f64::INFINITY, -0.1, 0.0, 1.0, 1.1] {
            assert_eq!(
                train_test_split(
                    4,
                    HoldoutParams::default().with_test_size(TestSize::Fraction(fraction))
                ),
                Err(SplitError::InvalidTestFraction)
            );
        }
        assert_eq!(
            train_test_split(
                4,
                HoldoutParams::default().with_test_size(TestSize::Count(4))
            ),
            Err(SplitError::InvalidTestCount {
                test: 4,
                samples: 4,
            })
        );
    }

    #[test]
    fn kfold_exhaustively_covers_each_sample_once() {
        for samples in 2..=64 {
            for folds in 2..=samples {
                for seed in [0, 1, 42] {
                    let splits = KFold::new(folds)
                        .with_shuffle(true)
                        .with_random_state(seed)
                        .split(samples)
                        .unwrap()
                        .collect::<Vec<_>>();
                    assert_eq!(splits.len(), folds);
                    let mut test_counts = vec![0_usize; samples];
                    let sizes = splits
                        .iter()
                        .map(|split| {
                            assert_complete_split(samples, split);
                            for &index in split.test_indices() {
                                test_counts[index] += 1;
                            }
                            split.test_indices().len()
                        })
                        .collect::<Vec<_>>();
                    assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
                    assert_eq!(test_counts, vec![1; samples]);
                }
            }
        }
    }

    #[test]
    fn unshuffled_kfold_has_exact_contiguous_membership() {
        let tests = KFold::new(3)
            .split(8)
            .unwrap()
            .map(|split| split.test_indices().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(tests, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7]]);
    }

    #[test]
    fn stratified_folds_balance_every_class_and_global_size() {
        for folds in 2..=8 {
            for classes in 1..=4_u8 {
                let labels = (0..classes)
                    .flat_map(|label| std::iter::repeat_n(label, folds + label as usize + 2))
                    .collect::<Vec<_>>();
                for seed in [0, 7, 99] {
                    let splits = StratifiedKFold::new(folds)
                        .with_shuffle(true)
                        .with_random_state(seed)
                        .split(&labels)
                        .unwrap()
                        .collect::<Vec<_>>();
                    let global_sizes = splits
                        .iter()
                        .map(|split| split.test_indices().len())
                        .collect::<Vec<_>>();
                    assert!(
                        global_sizes.iter().max().unwrap() - global_sizes.iter().min().unwrap()
                            <= 1
                    );
                    for label in 0..classes {
                        let counts = splits
                            .iter()
                            .map(|split| {
                                split
                                    .test_indices()
                                    .iter()
                                    .filter(|&&index| labels[index] == label)
                                    .count()
                            })
                            .collect::<Vec<_>>();
                        assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
                    }
                }
            }
        }
    }

    #[test]
    fn stratified_holdout_preserves_classes_and_exact_size() {
        let labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let split = stratified_train_test_split(
            &labels,
            HoldoutParams::default()
                .with_test_size(TestSize::Count(4))
                .with_random_state(17),
        )
        .unwrap();
        assert_eq!(split.test_indices().len(), 4);
        for label in [0, 1] {
            assert!(
                split
                    .test_indices()
                    .iter()
                    .any(|&index| labels[index] == label)
            );
            assert!(
                split
                    .train_indices()
                    .iter()
                    .any(|&index| labels[index] == label)
            );
        }
    }

    #[test]
    fn stratified_holdout_membership_matches_legacy_across_class_shapes() {
        let mut cases = vec![
            (0_u8..4)
                .flat_map(|label| std::iter::repeat_n(label, 8))
                .collect::<Vec<_>>(),
            [2_usize, 3, 7, 11]
                .into_iter()
                .enumerate()
                .flat_map(|(label, count)| std::iter::repeat_n(label as u8, count))
                .collect(),
            [(0_u8, 5_usize), (17, 9), (255, 4)]
                .into_iter()
                .flat_map(|(label, count)| std::iter::repeat_n(label, count))
                .collect(),
        ];
        cases.push(
            (0_u8..=254)
                .flat_map(|label| std::iter::repeat_n(label, 2))
                .collect(),
        );
        cases.push(
            (0_u8..=u8::MAX)
                .flat_map(|label| std::iter::repeat_n(label, 2))
                .collect(),
        );

        for labels in cases {
            let classes = class_counts(&labels)
                .into_iter()
                .filter(|&count| count > 0)
                .count();
            for test_count in classes..=labels.len() - classes {
                for shuffle in [false, true] {
                    for seed in [0, 1, 42, u64::MAX] {
                        let params = HoldoutParams::default()
                            .with_test_size(TestSize::Count(test_count))
                            .with_shuffle(shuffle)
                            .with_random_state(seed);
                        assert_eq!(
                            stratified_train_test_split(&labels, params),
                            legacy_stratified_train_test_split(&labels, params),
                            "rows={}, classes={classes}, test={test_count}, shuffle={shuffle}, seed={seed}",
                            labels.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stratification_rejects_impossible_class_partitions() {
        assert_eq!(
            StratifiedKFold::new(3).split(&[0, 0, 1, 1, 1]).unwrap_err(),
            SplitError::InsufficientClassMembers {
                label: 0,
                count: 2,
                partitions: 3,
            }
        );
        assert_eq!(
            stratified_train_test_split(
                &[0, 0, 1, 1],
                HoldoutParams::default().with_test_size(TestSize::Count(1))
            ),
            Err(SplitError::PartitionTooSmallForClasses {
                partition: SplitPartition::Test,
                rows: 1,
                classes: 2,
            })
        );
    }

    fn assert_complete_split(samples: usize, split: &Split) {
        assert!(
            split
                .train_indices()
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(
            split
                .test_indices()
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        let mut membership = vec![0_u8; samples];
        for &index in split.train_indices() {
            membership[index] += 1;
        }
        for &index in split.test_indices() {
            membership[index] += 1;
        }
        assert_eq!(membership, vec![1; samples]);
    }
}
