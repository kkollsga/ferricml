//! Evidence that the two suites still span the generator.
//!
//! # The one test the rest of this file exists to support
//!
//! `every_family_has_an_accuracy_case` is the load-bearing assertion, and its
//! value is entirely in *when* it fires. The failure it is built for is not a
//! wrong number; it is silence — the crate gains a family, the suites keep
//! passing, and every report they produce quietly describes less of the
//! generator than its title claims.
//!
//! That failure only becomes visible if three separate things line up, so all
//! three are asserted here rather than assumed:
//!
//! 1. **The roster is the declaration order.** `Family::ALL` is data, and
//!    `the_family_roster_is_the_declared_order` compares it against the
//!    exhaustive walk `Family::COUNT` is computed from. A roster that lost a
//!    member, gained a duplicate, or was reordered fails there.
//! 2. **The suites are tables, not maps.** Both `cases` functions are written
//!    out, so "spans every family" is a claim about the table rather than a
//!    tautology about the code. If they were `match` arms over `Family`, the two
//!    closure tests below would be incapable of failing.
//! 3. **Nothing else fires first.** Adding a `Task` variant fails to compile at
//!    `Task::family`; adding a `Family` variant fails to compile at
//!    `Family::next` and then at the length of `Family::ALL`. The closure tests
//!    are what is left once the compiler has run out of things to say — which is
//!    exactly the "wrote the family, forgot the case" state.
//!
//! Verified by planting, on 2026-07-28, and each link checked separately rather
//! than assumed:
//!
//! * An eleventh, planted variant of the task enum stops the crate compiling in
//!   five places, one of them `Task::family`'s match.
//! * Giving it a planted family variant stops the crate compiling at
//!   `Family::next` and at `Family::label`.
//! * Placing that family in the declaration order moves `Family::COUNT` to
//!   eleven, and the roster literal then fails with *expected an array with a
//!   size of 11, found one with a size of 10*.
//! * Only after the roster is repaired does anything run — and then
//!   `every_family_has_an_accuracy_case` fails with `families with no accuracy
//!   case: [Planted]`, alongside `every_family_has_a_performance_grid_row`.
//!
//! Deleting the `Task::Clustered` entry from `AccuracySuite::cases` — the "family
//! exists, case was dropped" half — fails the same test with `families with no
//! accuracy case: [Clustered]`, plus the two count assertions below, which is
//! three independent readings of one mistake. Both plants were removed.
//!
//! # Why so little is generated here
//!
//! The accuracy suite is generated in full — ten cases of `256x8` is nothing.
//! The performance grid is not: ninety cases up to `4096x128` is roughly nine
//! million design values, which belongs on a registered runner under the
//! performance protocol and not in a debug-build test. What is asserted for the
//! grid is that every case *exists and is valid* — which is the property a
//! measurement run depends on — plus generation at the smallest point only.

use super::*;
use std::collections::BTreeSet;

/// Walks the declaration order `Family::COUNT` is computed from.
///
/// Deliberately a second implementation of that walk rather than a call into
/// one: `COUNT` is a const evaluation and this is a runtime one, so they agree
/// only if `Family::next` says the same thing in both contexts.
fn declared_order() -> Vec<Family> {
    let mut order = Vec::new();
    let mut family = Some(Family::LinearRegression);
    while let Some(current) = family {
        order.push(current);
        family = current.next();
    }
    order
}

#[test]
fn the_family_roster_is_the_declared_order() {
    assert_eq!(
        declared_order(),
        Family::ALL.to_vec(),
        "`Family::ALL` must be the declaration order `Family::COUNT` is counted \
         from. A family that reached the enum without reaching the roster leaves \
         every suite spanning less than it claims, and the closure tests below \
         cannot see it."
    );
    assert_eq!(
        Family::ALL.len(),
        Family::COUNT,
        "the roster's length is its declared type, so this can only fail if \
         `COUNT` stopped being that length"
    );
}

#[test]
fn every_family_carries_a_distinct_stable_label() {
    let labels: BTreeSet<&str> = Family::ALL.iter().map(|family| family.label()).collect();
    assert_eq!(
        labels.len(),
        Family::COUNT,
        "two families share a label, so two different problems would be recorded \
         under one identity: {labels:?}"
    );
    for family in Family::ALL {
        let label = family.label();
        assert!(
            !label.is_empty() && label.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
            "`{label}` is not a lower-case hyphenated identifier, and these labels \
             are what recorded results are filed under"
        );
    }
}

#[test]
fn every_family_has_an_accuracy_case() {
    let covered: Vec<Family> = AccuracySuite::cases()
        .iter()
        .map(SuiteCase::family)
        .collect();
    let missing: Vec<Family> = Family::ALL
        .into_iter()
        .filter(|family| !covered.contains(family))
        .collect();

    assert!(
        missing.is_empty(),
        "families with no accuracy case: {missing:?}. `AccuracySuite::cases` is a \
         written-out table rather than a map over `Family::ALL`, so a new family \
         needs a case chosen for it — what problem in that family is small, \
         clean, and has a recoverable answer — rather than a mechanical entry."
    );
}

#[test]
fn every_family_has_a_performance_grid_row() {
    let covered: Vec<Family> = PerformanceGrid::cases()
        .iter()
        .map(SuiteCase::family)
        .collect();
    let missing: Vec<Family> = Family::ALL
        .into_iter()
        .filter(|family| !covered.contains(family))
        .collect();

    assert!(
        missing.is_empty(),
        "families with no performance grid row: {missing:?}. A family absent here \
         is a family whose generation cost is never measured, so the plan's claim \
         that generation is negligible against the fit it feeds would be a claim \
         about the other families only."
    );
}

#[test]
fn the_accuracy_suite_is_one_case_per_family_in_roster_order() {
    let cases = AccuracySuite::cases();
    assert_eq!(
        cases.len(),
        Family::COUNT,
        "the accuracy suite is one case per family; a second case for one family \
         would make a per-family report ambiguous about which case it read"
    );
    let families: Vec<Family> = cases.iter().map(SuiteCase::family).collect();
    assert_eq!(
        families,
        Family::ALL.to_vec(),
        "the suite's order is the roster's, so a reader comparing two runs reads \
         the same family at the same index"
    );
    for case in &cases {
        assert_eq!(
            case.name(),
            case.family().label(),
            "a case's name is its family's label, and nothing else"
        );
    }
}

#[test]
fn every_accuracy_case_generates_the_shape_it_declares() {
    for case in AccuracySuite::cases() {
        let dataset = case.generate();
        let features = dataset.features();
        assert_eq!(
            (features.rows(), features.columns()),
            (AccuracySuite::ROWS, AccuracySuite::COLUMNS),
            "{} generated {}x{}",
            case.name(),
            features.rows(),
            features.columns(),
        );
        assert_eq!(
            dataset.spec_digest(),
            case.recipe().spec_digest(),
            "{} generated data whose digest does not identify its own recipe",
            case.name(),
        );
    }
}

#[test]
fn every_accuracy_case_records_a_truth_its_family_can_be_scored_against() {
    for case in AccuracySuite::cases() {
        let dataset = case.generate();
        let truth = dataset.truth();
        assert_ne!(
            *truth,
            Truth::DesignOnly,
            "{} has no task, so there is nothing to be right about",
            case.name(),
        );
        assert_ne!(
            *truth,
            Truth::Unrecorded,
            "{} reports an unrecorded truth, which is a statement reserved for the \
             absorbed lanes — a family that draws a target knows what it drew",
            case.name(),
        );

        // Every case that draws a target has one, and the clustered case is the
        // reason that is worth asserting rather than assuming: it draws none,
        // and an empty target vector would be a claim that it does.
        let has_target = dataset.target().is_some();
        assert_eq!(
            has_target,
            case.family() != Family::Clustered,
            "{} disagrees with its family about whether it has a target",
            case.name(),
        );
    }
}

/// The families whose accuracy case can only promise per-runner bytes.
///
/// Pinned rather than computed, because the point is to notice a change. A
/// family switching envelopes is a real event — a link replaced by a rational
/// approximation, or a bit-exact family acquiring a transcendental — and it has
/// to move this list and the documentation page together, not silently widen
/// what the suite promises.
const PER_RUNNER_ACCURACY_CASES: [Family; 5] = [
    Family::GlmRegression,
    Family::IllConditioned,
    Family::LinearBinary,
    Family::NonlinearBinary,
    Family::Multiclass,
];

#[test]
fn the_accuracy_suite_declares_which_of_its_members_are_per_runner() {
    let per_runner: Vec<Family> = AccuracySuite::cases()
        .iter()
        .filter(|case| case.portability() == Portability::PerRunner)
        .map(SuiteCase::family)
        .collect();

    assert_eq!(
        per_runner,
        PER_RUNNER_ACCURACY_CASES.to_vec(),
        "the suite's determinism envelope changed. Half of D3's contract is that \
         a family declares its own envelope; the other half is that a suite \
         spanning every family says which of its members carry the weaker one. \
         `docs/dataset-suites.md` names the same five, and both have to move \
         together."
    );
}

#[test]
fn every_bit_exact_accuracy_case_reproduces_itself_byte_for_byte() {
    for case in AccuracySuite::cases() {
        if case.portability() != Portability::BitExact {
            continue;
        }
        let first = case.generate();
        let second = case.recipe().generate();
        assert_eq!(
            first.features().as_slice(),
            second.features().as_slice(),
            "{} is declared bit-exact and did not reproduce its own design",
            case.name(),
        );
        assert_eq!(
            first.target(),
            second.target(),
            "{} is declared bit-exact and did not reproduce its own target",
            case.name(),
        );
    }
}

#[test]
fn the_accuracy_suite_gives_every_family_its_own_identity() {
    let digests: BTreeSet<[u8; 32]> = AccuracySuite::cases()
        .iter()
        .map(|case| case.recipe().spec_digest())
        .collect();
    assert_eq!(
        digests.len(),
        Family::COUNT,
        "two accuracy cases share a spec digest, so a cache keyed on it would \
         serve one family's data for another's"
    );
}

#[test]
fn the_accuracy_suite_recovers_the_coefficients_it_drew() {
    let case = AccuracySuite::cases()
        .into_iter()
        .find(|case| case.family() == Family::LinearRegression)
        .expect("the suite spans every family");
    let dataset = case.generate();
    let truth = dataset.truth();
    let beta = truth.coefficients().expect("a linear family records beta");
    let mean = truth
        .conditional_mean()
        .expect("a linear family records its noise-free target");

    // The noise-free target is the design times the drawn coefficients plus the
    // intercept, and asserting that is what makes "the truth belongs to this
    // data" a check rather than a label. The tolerance is one `f32` rounding per
    // accumulated term over eight columns.
    let intercept = truth.intercept().expect("a linear family records one");
    for (row, values) in dataset.features().iter_rows().enumerate() {
        let expected = (values
            .iter()
            .zip(beta)
            .map(|(&value, &coefficient)| f64::from(value) * f64::from(coefficient))
            .sum::<f64>()
            + f64::from(intercept)) as f32;
        assert!(
            (mean[row] - expected).abs() < 1e-5,
            "row {row}: recorded conditional mean {} against {expected}",
            mean[row],
        );
    }
}

#[test]
fn the_performance_grid_covers_every_shape_for_every_family() {
    let cases = PerformanceGrid::cases();
    assert_eq!(
        cases.len(),
        PerformanceGrid::ROWS.len() * PerformanceGrid::COLUMNS.len() * Family::COUNT,
        "the grid is a full cross product; a missing cell is a shape whose cost \
         nobody measures"
    );

    let mut points: BTreeSet<(usize, usize, &str)> = BTreeSet::new();
    for case in &cases {
        let recipe = case.recipe();
        assert!(
            PerformanceGrid::ROWS.contains(&recipe.rows()),
            "{} sits at {} rows, which is not a grid point",
            case.name(),
            recipe.rows(),
        );
        assert!(
            PerformanceGrid::COLUMNS.contains(&recipe.columns()),
            "{} sits at {} columns, which is not a grid point",
            case.name(),
            recipe.columns(),
        );
        assert!(
            points.insert((recipe.rows(), recipe.columns(), case.name())),
            "{} appears twice at {}x{}",
            case.name(),
            recipe.rows(),
            recipe.columns(),
        );
    }
}

#[test]
fn the_performance_grid_generates_at_its_smallest_point() {
    let smallest = PerformanceGrid::ROWS[0];
    let narrowest = PerformanceGrid::COLUMNS[0];
    let mut generated = 0;
    for case in PerformanceGrid::cases() {
        let recipe = case.recipe();
        if recipe.rows() != smallest || recipe.columns() != narrowest {
            continue;
        }
        let dataset = case.generate();
        assert_eq!(
            (dataset.features().rows(), dataset.features().columns()),
            (smallest, narrowest),
            "{} generated the wrong shape",
            case.name(),
        );
        generated += 1;
    }
    assert_eq!(
        generated,
        Family::COUNT,
        "every family must be generable at the grid's smallest point, or the \
         larger points cannot be trusted either"
    );
}
