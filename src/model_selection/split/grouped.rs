//! Splitters that keep related rows on one side of every split.

use super::{Split, SplitError, validate_fold_count};

/// K-fold splitter that never puts one group on both sides of a split.
///
/// A group identifies rows that share an entity — the same patient, session, or
/// document — whose feature values are not independent. Splitting them across
/// train and test leaks, and the resulting score measures memorization rather
/// than generalization, so this splitter assigns whole groups to folds.
///
/// Assignment is deterministic and needs no seed: groups are taken in
/// decreasing size, ties broken by increasing group identifier, and each is
/// placed in the fold holding the fewest rows so far, ties broken by the lowest
/// fold index. Fold sizes are therefore as even as whole groups allow, which is
/// not exactly even when group sizes are not.
///
/// ```
/// use ferricml::model_selection::GroupKFold;
///
/// let groups = [0, 0, 1, 1, 1, 2];
/// let folds = GroupKFold::new(2).split(&groups)?.collect::<Vec<_>>();
/// // Group 1 owns rows 2..=4 and never straddles a split.
/// assert!(folds.iter().all(|fold| {
///     let held = fold.test_indices();
///     [2, 3, 4].iter().all(|row| held.contains(row))
///         || [2, 3, 4].iter().all(|row| !held.contains(row))
/// }));
/// # Ok::<(), ferricml::model_selection::SplitError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    /// Creates a group-aware K-fold configuration.
    pub const fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Returns the number of folds.
    pub const fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Validates the group labels and returns an iterator over complete splits.
    ///
    /// One entry per row, naming the group that row belongs to. Group
    /// identifiers carry no order or meaning beyond equality.
    pub fn split(&self, groups: &[u64]) -> Result<GroupKFoldIter, SplitError> {
        validate_fold_count(groups.len(), self.n_splits)?;

        let mut sizes = group_sizes(groups);
        let distinct = sizes.len();
        if distinct < self.n_splits {
            return Err(SplitError::InsufficientGroups {
                groups: distinct,
                partitions: self.n_splits,
            });
        }
        // Largest first, so the remaining capacity decisions get finer as the
        // folds fill; equal sizes keep ascending identifier order.
        sizes.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));

        let mut loads = vec![0_usize; self.n_splits];
        let mut fold_of_group = sizes
            .iter()
            .map(|&(group, _)| (group, 0))
            .collect::<Vec<_>>();
        for (position, &(_, size)) in sizes.iter().enumerate() {
            let lightest = loads
                .iter()
                .enumerate()
                .min_by_key(|&(fold, &load)| (load, fold))
                .map(|(fold, _)| fold)
                .expect("a validated fold count is at least two");
            loads[lightest] += size;
            fold_of_group[position].1 = lightest;
        }
        fold_of_group.sort_unstable();

        let assignments = groups
            .iter()
            .map(|group| {
                let position = fold_of_group
                    .binary_search_by_key(group, |&(group, _)| group)
                    .expect("every row's group was counted");
                fold_of_group[position].1
            })
            .collect();
        Ok(GroupKFoldIter {
            assignments,
            n_splits: self.n_splits,
            next_fold: 0,
        })
    }
}

/// Iterator over group-aware K-fold partitions.
#[derive(Clone, Debug)]
pub struct GroupKFoldIter {
    assignments: Vec<usize>,
    n_splits: usize,
    next_fold: usize,
}

impl Iterator for GroupKFoldIter {
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

impl ExactSizeIterator for GroupKFoldIter {}

/// Distinct groups with their row counts, in ascending identifier order.
fn group_sizes(groups: &[u64]) -> Vec<(u64, usize)> {
    let mut sorted = groups.to_vec();
    sorted.sort_unstable();
    let mut sizes: Vec<(u64, usize)> = Vec::new();
    for group in sorted {
        match sizes.last_mut() {
            Some(last) if last.0 == group => last.1 += 1,
            _ => sizes.push((group, 1)),
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUPS: [u64; 12] = [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4];

    fn test_windows(groups: &[u64], n_splits: usize) -> Vec<Vec<usize>> {
        GroupKFold::new(n_splits)
            .split(groups)
            .unwrap()
            .map(|split| split.test_indices().to_vec())
            .collect()
    }

    #[test]
    fn fold_membership_matches_the_reference_assignment() {
        assert_eq!(
            test_windows(&GROUPS, 3),
            vec![vec![8, 9, 10, 11], vec![2, 3, 4, 5], vec![0, 1, 6, 7]]
        );
    }

    #[test]
    fn no_group_appears_on_both_sides_of_any_split() {
        let cases: [&[u64]; 5] = [
            &GROUPS,
            &[7, 7, 7, 1, 1, 9, 9, 9, 9, 3],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[u64::MAX, 0, u64::MAX, 5, 5, 0],
            &[2, 2, 2, 2, 2, 3, 4, 5],
        ];
        for groups in cases {
            for n_splits in 2..=4 {
                let Ok(splits) = GroupKFold::new(n_splits).split(groups) else {
                    continue;
                };
                let mut held_out = vec![0_usize; groups.len()];
                let mut folds = 0;
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
                            "group {group} leaked across a split of {groups:?}"
                        );
                        assert!(in_train || in_test, "group {group} vanished");
                    }
                    assert_eq!(split.sample_count(), groups.len());
                    assert_eq!(split.covered_samples(), groups.len());
                    for &index in split.test_indices() {
                        held_out[index] += 1;
                    }
                    folds += 1;
                }
                assert_eq!(folds, n_splits);
                assert_eq!(held_out, vec![1; groups.len()]);
            }
        }
    }

    #[test]
    fn assignment_is_deterministic_and_independent_of_group_naming() {
        let renamed = GROUPS
            .iter()
            .map(|group| u64::from(u32::MAX) + group * 1_000)
            .collect::<Vec<_>>();
        assert_eq!(test_windows(&GROUPS, 3), test_windows(&renamed, 3));
        assert_eq!(test_windows(&GROUPS, 3), test_windows(&GROUPS, 3));
    }

    #[test]
    fn folds_are_as_even_as_whole_groups_allow() {
        let sizes = test_windows(&GROUPS, 2)
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>();
        assert_eq!(sizes, vec![6, 6]);
        // Groups of 5 and 1 cannot make two folds of 3.
        let lumpy = [0, 0, 0, 0, 0, 1];
        assert_eq!(
            test_windows(&lumpy, 2)
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![5, 1]
        );
    }

    #[test]
    fn too_few_groups_or_rows_are_reported_before_any_assignment() {
        assert_eq!(
            GroupKFold::new(3).split(&[0, 0, 1, 1]).unwrap_err(),
            SplitError::InsufficientGroups {
                groups: 2,
                partitions: 3,
            }
        );
        assert_eq!(
            GroupKFold::new(2).split(&[0]).unwrap_err(),
            SplitError::NotEnoughSamples { samples: 1 }
        );
        assert_eq!(
            GroupKFold::new(1).split(&[0, 1]).unwrap_err(),
            SplitError::InvalidFoldCount {
                folds: 1,
                samples: 2,
            }
        );
    }
}
