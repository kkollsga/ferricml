//! Repeated random holdouts that keep every group whole.

use super::grouped::group_sizes;
use super::{Split, SplitError, StableRng, repeat_seed, shuffle_with_rng, validate_sample_count};

/// Requested holdout size measured in **whole groups**.
///
/// This is deliberately a separate type from [`TestSize`](super::TestSize),
/// which counts rows everywhere else in FerricML. A grouped holdout cannot
/// honour a row count exactly: rows only move a whole group at a time, so any
/// row target becomes an approximation with an unstated rounding rule. Counting
/// groups keeps the size exact and keeps the guarantee sayable — *this many
/// whole groups are held out* — at the cost of a test partition whose row count
/// depends on how large those groups happen to be.
///
/// One type carrying both meanings would be a footgun that reads correct at
/// every call site, so the counting unit is in the type name instead.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum TestGroupSize {
    /// Exact number of whole groups held out.
    Count(usize),
    /// Fraction of the **distinct groups**, rounded upward.
    ///
    /// The fraction is applied to the number of groups, never to the number of
    /// rows, so `Fraction(0.5)` over four groups holds out two groups whether
    /// they carry two rows or two hundred.
    Fraction(f64),
}

/// Repeated random group-respecting holdouts.
///
/// Each split draws whole groups at random for the test partition, so no group
/// is ever on both sides of a split and a score cannot be inflated by having
/// seen the same entity during fitting. Splits are **independent draws, not a
/// partition**: unlike [`GroupKFold`](super::GroupKFold), two splits may hold
/// out the same group, and no promise is made that every row is eventually
/// tested. That independence is the point — it is what lets a caller ask for
/// any number of holdouts at any size.
///
/// Size is measured in groups, through [`TestGroupSize`]. Every split therefore
/// holds out exactly the requested number of whole groups, and the number of
/// held-out *rows* varies between splits whenever group sizes differ.
///
/// Assignment is deterministic: the distinct group identifiers are taken in
/// ascending order, shuffled with a seed derived from the configured one and
/// the split index, and the first entries become that split's test groups.
/// Identical parameters therefore reproduce identical membership. Because the
/// shuffle runs over sorted identifiers, renaming groups can change which of
/// them is drawn, unlike `GroupKFold`, whose assignment depends only on group
/// sizes.
///
/// ```
/// use ferricml::model_selection::{GroupShuffleSplit, TestGroupSize};
///
/// // Six rows in three groups, carrying one, two, and three rows.
/// let groups = [0, 1, 1, 2, 2, 2];
/// let split = GroupShuffleSplit::new(1)
///     .with_test_size(TestGroupSize::Count(1))
///     .with_random_state(7)
///     .split(&groups)?
///     .next()
///     .expect("one split was requested");
///
/// // Exactly one whole group is held out, however many rows it carries.
/// let held = split.test_indices();
/// assert!(held == [0] || held == [1, 2] || held == [3, 4, 5]);
/// # Ok::<(), ferricml::model_selection::SplitError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupShuffleSplit {
    n_splits: usize,
    test_size: TestGroupSize,
    random_state: u64,
}

impl GroupShuffleSplit {
    /// Creates a configuration drawing `n_splits` independent holdouts.
    pub const fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: TestGroupSize::Fraction(0.25),
            random_state: 0,
        }
    }

    /// Sets how many whole groups each split holds out.
    #[must_use]
    pub const fn with_test_size(mut self, test_size: TestGroupSize) -> Self {
        self.test_size = test_size;
        self
    }

    /// Sets the seed every split's group draw is derived from.
    #[must_use]
    pub const fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// Returns the number of independent holdouts.
    pub const fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Returns the requested holdout size in groups.
    pub const fn test_size(&self) -> TestGroupSize {
        self.test_size
    }

    /// Returns the seed every split's group draw is derived from.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    /// Validates the group labels and returns an iterator over complete splits.
    ///
    /// One entry per row, naming the group that row belongs to. Group
    /// identifiers carry no meaning beyond equality and their sort order.
    pub fn split(&self, groups: &[u64]) -> Result<GroupShuffleSplitIter, SplitError> {
        validate_sample_count(groups.len())?;
        if self.n_splits == 0 {
            return Err(SplitError::InvalidRepeatCount { repeats: 0 });
        }

        let sizes = group_sizes(groups);
        let distinct = sizes.len();
        if distinct < 2 {
            return Err(SplitError::InsufficientGroups {
                groups: distinct,
                partitions: 2,
            });
        }
        let test_groups = resolve_test_group_count(distinct, self.test_size)?;

        let group_of_row = groups
            .iter()
            .map(|group| {
                sizes
                    .binary_search_by_key(group, |&(group, _)| group)
                    .expect("every row's group was counted")
            })
            .collect();
        Ok(GroupShuffleSplitIter {
            group_of_row,
            distinct,
            test_groups,
            n_splits: self.n_splits,
            random_state: self.random_state,
            next_split: 0,
        })
    }
}

/// Iterator over independent group-respecting holdouts.
#[derive(Clone, Debug)]
pub struct GroupShuffleSplitIter {
    group_of_row: Vec<usize>,
    distinct: usize,
    test_groups: usize,
    n_splits: usize,
    random_state: u64,
    next_split: usize,
}

impl Iterator for GroupShuffleSplitIter {
    type Item = Split;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_split == self.n_splits {
            return None;
        }
        let mut order = (0..self.distinct).collect::<Vec<_>>();
        let mut rng = StableRng::new(repeat_seed(self.random_state, self.next_split));
        shuffle_with_rng(&mut order, &mut rng);
        self.next_split += 1;

        let mut is_test_group = vec![false; self.distinct];
        for &group in &order[..self.test_groups] {
            is_test_group[group] = true;
        }
        let test_indices = self
            .group_of_row
            .iter()
            .enumerate()
            .filter_map(|(index, &group)| is_test_group[group].then_some(index))
            .collect();
        Some(Split::from_test_indices(
            self.group_of_row.len(),
            test_indices,
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_splits - self.next_split;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for GroupShuffleSplitIter {}

/// Resolves a requested holdout size against the number of distinct groups.
fn resolve_test_group_count(groups: usize, test_size: TestGroupSize) -> Result<usize, SplitError> {
    let test_groups = match test_size {
        TestGroupSize::Count(count) => count,
        TestGroupSize::Fraction(fraction) => {
            if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
                return Err(SplitError::InvalidTestFraction);
            }
            (groups as f64 * fraction).ceil() as usize
        }
    };
    if test_groups == 0 || test_groups >= groups {
        return Err(SplitError::InvalidTestGroupCount {
            test_groups,
            groups,
        });
    }
    Ok(test_groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DenseMatrix, RegressionTargets};
    use crate::linear_model::{Ridge, RidgeParams};
    use crate::model_selection::{RegressionScorer, cross_validate_regressor};
    use std::cell::Cell;
    use std::rc::Rc;

    const CASES: [&[u64]; 5] = [
        &[0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4],
        &[7, 7, 7, 1, 1, 9, 9, 9, 9, 3],
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[u64::MAX, 0, u64::MAX, 5, 5, 0],
        &[2, 2, 2, 2, 2, 3],
    ];

    fn distinct(groups: &[u64]) -> usize {
        group_sizes(groups).len()
    }

    #[test]
    fn no_group_appears_on_both_sides_of_any_split() {
        for groups in CASES {
            let available = distinct(groups);
            for test_groups in 1..available {
                for seed in [0, 1, 42, u64::MAX] {
                    let splits = GroupShuffleSplit::new(4)
                        .with_test_size(TestGroupSize::Count(test_groups))
                        .with_random_state(seed)
                        .split(groups)
                        .unwrap();
                    let mut seen = 0;
                    for split in splits {
                        for &group in groups {
                            let in_train = split
                                .train_indices()
                                .iter()
                                .any(|&index| groups[index] == group);
                            let in_test = split
                                .test_indices()
                                .iter()
                                .any(|&index| groups[index] == group);
                            assert!(
                                !(in_train && in_test),
                                "group {group} leaked across a split of {groups:?} \
                                 (test_groups={test_groups}, seed={seed})"
                            );
                            assert!(in_train || in_test, "group {group} vanished");
                        }
                        // Each individual split still names every row exactly once.
                        assert_eq!(split.sample_count(), groups.len());
                        assert_eq!(split.covered_samples(), groups.len());
                        seen += 1;
                    }
                    assert_eq!(seen, 4);
                }
            }
        }
    }

    #[test]
    fn every_split_holds_out_exactly_the_requested_number_of_groups() {
        for groups in CASES {
            let available = distinct(groups);
            for test_groups in 1..available {
                for split in GroupShuffleSplit::new(3)
                    .with_test_size(TestGroupSize::Count(test_groups))
                    .with_random_state(11)
                    .split(groups)
                    .unwrap()
                {
                    let held = split
                        .test_indices()
                        .iter()
                        .map(|&index| groups[index])
                        .collect::<Vec<_>>();
                    let mut distinct_held = held.clone();
                    distinct_held.sort_unstable();
                    distinct_held.dedup();
                    assert_eq!(distinct_held.len(), test_groups, "{groups:?}");
                }
            }
        }
    }

    /// The headline semantic: the size is in groups, and the row count follows
    /// from whichever groups were drawn rather than being targeted itself.
    #[test]
    fn the_test_size_counts_groups_and_not_rows() {
        // Four groups carrying 1, 1, 1, and 9 rows: half the groups are never
        // half the rows.
        let groups = [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3];
        let mut observed_row_counts = Vec::new();
        for seed in 0..16 {
            let split = GroupShuffleSplit::new(1)
                .with_test_size(TestGroupSize::Fraction(0.5))
                .with_random_state(seed)
                .split(&groups)
                .unwrap()
                .next()
                .unwrap();
            let held = split
                .test_indices()
                .iter()
                .map(|&index| groups[index])
                .collect::<Vec<_>>();
            let mut distinct_held = held.clone();
            distinct_held.sort_unstable();
            distinct_held.dedup();
            assert_eq!(distinct_held.len(), 2, "half of four groups is two groups");
            observed_row_counts.push(split.test_indices().len());
        }
        // Never half the rows, and not even a fixed number of them.
        assert!(observed_row_counts.iter().all(|&rows| rows != 6));
        observed_row_counts.sort_unstable();
        observed_row_counts.dedup();
        assert!(
            observed_row_counts.len() > 1,
            "row counts must follow the drawn groups, got {observed_row_counts:?}"
        );
    }

    #[test]
    fn a_fraction_of_groups_rounds_upward() {
        let groups = [0, 1, 2, 3, 4];
        for (fraction, expected) in [(0.2, 1), (0.21, 2), (0.5, 3), (0.8, 4)] {
            let split = GroupShuffleSplit::new(1)
                .with_test_size(TestGroupSize::Fraction(fraction))
                .split(&groups)
                .unwrap()
                .next()
                .unwrap();
            assert_eq!(split.test_indices().len(), expected, "fraction {fraction}");
        }
    }

    #[test]
    fn splits_are_independent_draws_rather_than_a_partition() {
        let groups = [0, 0, 1, 1, 2, 2, 3, 3];
        let holdouts = GroupShuffleSplit::new(8)
            .with_test_size(TestGroupSize::Count(1))
            .with_random_state(3)
            .split(&groups)
            .unwrap()
            .map(|split| split.test_indices().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(holdouts.len(), 8);
        // Independent draws repeat; a partition never would.
        let mut sorted = holdouts.clone();
        sorted.sort();
        sorted.dedup();
        assert!(
            sorted.len() < holdouts.len(),
            "eight independent draws from four groups must repeat: {holdouts:?}"
        );
    }

    #[test]
    fn identical_parameters_reproduce_identical_membership() {
        let groups = CASES[0];
        let run = |seed| {
            GroupShuffleSplit::new(5)
                .with_test_size(TestGroupSize::Count(2))
                .with_random_state(seed)
                .split(groups)
                .unwrap()
                .map(|split| split.test_indices().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(19), run(19));
        assert_ne!(run(19), run(20));
        // Splits within one run differ from each other.
        let splits = run(19);
        assert!(splits.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn invalid_shapes_are_reported_before_any_assignment() {
        assert_eq!(
            GroupShuffleSplit::new(2).split(&[0]).unwrap_err(),
            SplitError::NotEnoughSamples { samples: 1 }
        );
        assert_eq!(
            GroupShuffleSplit::new(0).split(&[0, 1]).unwrap_err(),
            SplitError::InvalidRepeatCount { repeats: 0 }
        );
        assert_eq!(
            GroupShuffleSplit::new(1).split(&[4, 4, 4]).unwrap_err(),
            SplitError::InsufficientGroups {
                groups: 1,
                partitions: 2,
            }
        );
        for fraction in [f64::NAN, f64::INFINITY, -0.1, 0.0, 1.0, 1.1] {
            assert_eq!(
                GroupShuffleSplit::new(1)
                    .with_test_size(TestGroupSize::Fraction(fraction))
                    .split(&[0, 1, 2, 3])
                    .unwrap_err(),
                SplitError::InvalidTestFraction,
                "fraction {fraction}"
            );
        }
        for count in [0, 3, 9] {
            assert_eq!(
                GroupShuffleSplit::new(1)
                    .with_test_size(TestGroupSize::Count(count))
                    .split(&[0, 0, 1, 2, 2])
                    .unwrap_err(),
                SplitError::InvalidTestGroupCount {
                    test_groups: count,
                    groups: 3,
                },
                "count {count}"
            );
        }
    }

    #[test]
    fn accessors_report_the_configured_values() {
        let splitter = GroupShuffleSplit::new(3)
            .with_test_size(TestGroupSize::Count(2))
            .with_random_state(88);
        assert_eq!(splitter.n_splits(), 3);
        assert_eq!(splitter.test_size(), TestGroupSize::Count(2));
        assert_eq!(splitter.random_state(), 88);
        let mut splits = splitter.split(CASES[0]).unwrap();
        for remaining in (0..3).rev() {
            assert!(splits.next().is_some());
            assert_eq!(splits.len(), remaining);
        }
        assert!(splits.next().is_none());
    }

    /// Non-leakage proven end to end: the fitted model never observes a row
    /// from a group it is scored on.
    #[test]
    fn cross_validation_never_fits_on_a_held_out_group() {
        let groups = [0_u64, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4];
        let data = DenseMatrix::new(
            groups
                .iter()
                .enumerate()
                .flat_map(|(row, &group)| [row as f32, group as f32])
                .collect(),
            groups.len(),
            2,
        )
        .unwrap();
        let targets =
            RegressionTargets::new((0..groups.len()).map(|row| row as f32).collect()).unwrap();
        let splits = GroupShuffleSplit::new(4)
            .with_test_size(TestGroupSize::Count(2))
            .with_random_state(31)
            .split(&groups)
            .unwrap()
            .collect::<Vec<_>>();

        let calls = Rc::new(Cell::new(0_usize));
        let fit_calls = Rc::clone(&calls);
        let observed = splits.clone();
        let result = cross_validate_regressor(
            &data.as_view(),
            &targets,
            splits,
            RegressionScorer::MeanAbsoluteError,
            move |train, train_targets| {
                let fold = fit_calls.get();
                fit_calls.set(fold + 1);
                let held_out_groups = observed[fold]
                    .test_indices()
                    .iter()
                    .map(|&index| groups[index] as f32)
                    .collect::<Vec<_>>();
                assert!(
                    train
                        .iter_rows()
                        .all(|row| !held_out_groups.contains(&row[1])),
                    "fold {fold} trained on a held-out group"
                );
                Ridge::fit(train, train_targets, RidgeParams::default())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 4);
        assert_eq!(result.len(), 4);
    }
}
