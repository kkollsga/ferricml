//! Evidence for the multiclass and structural families.
//!
//! # Where each tolerance comes from
//!
//! Three of the four families are [`Portability::BitExact`], and their evidence
//! is the same as the absorbed lanes': pinned literals compared with
//! `assert_eq!` on `f32`. There is no tolerance to justify, because there is no
//! rounding an implementation is free to choose.
//!
//! The multiclass family is not, and neither is anything measured through a fit.
//! Every tolerance below is derived from the sampling law of the quantity it
//! bounds, and the derivation is written beside the assertion:
//!
//! * A **realized class rate** is a mean of `n` independent Bernoulli
//!   indicators, so four standard deviations is `4 sqrt(p (1 - p) / n)`. It is
//!   conservative here rather than exact: the per-row probabilities differ, and
//!   a Poisson-binomial with a given mean has *less* variance than the binomial
//!   with the same mean. Measured worst over the eight swept cases: `1.57`
//!   deviations.
//! * A **window-fitted coefficient** carries two errors, and one of them
//!   dominates. Ordinary least squares over a uniform design on `[-1, 1)` has
//!   standard error `σ sqrt(3 / w)` per coefficient; more importantly, a window
//!   spanning `Δt` of a drifting problem recovers the *time-average* of the
//!   coefficient over that window plus a fluctuation of order
//!   `0.3 Δt sqrt(p) ‖δ‖ / sqrt(w)`, because the design's second moment and the
//!   row's time are independent but not orthogonal in a finite sample. Both are
//!   worked out where they are used.
//! * A **fitted ranker's agreement with the recorded order** is bounded by what
//!   the assertion is for rather than by what a fit achieves. The pairs are
//!   exactly separable by the recorded coefficients — asserted with no tolerance
//!   at all — so the remaining question is only whether the pairs, the indices
//!   and the item matrix line up. Each way they could fail to (an index off by a
//!   query block, an inverted outcome, pairs drawn across queries) puts the
//!   agreement at chance. The floors below are far above chance and below the
//!   measured values, which is exactly the gap they are meant to sit in.

use super::*;
use crate::data::DenseMatrix;
use crate::linear_model::{LinearRegression, LinearRegressionParams};
use crate::model_selection::{GroupKFold, TimeSeriesSplit};
use crate::ranking::{
    PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, decisive_directional_accuracy,
    spearman_correlation,
};
use std::collections::BTreeMap;

/// Four binomial standard deviations of a rate `p` over `n` draws.
fn binomial_tolerance(rate: f64, draws: usize) -> f64 {
    4.0 * (rate * (1.0 - rate) / draws as f64).sqrt()
}

fn class_labels(dataset: &Dataset) -> &[u8] {
    match dataset.target() {
        Some(Target::Class(targets)) => targets.as_slice(),
        other => panic!("expected class labels, got {other:?}"),
    }
}

fn regression_values(dataset: &Dataset) -> &[f32] {
    match dataset.target() {
        Some(Target::Regression(targets)) => targets.as_slice(),
        other => panic!("expected a regression target, got {other:?}"),
    }
}

/// The realized share of every class.
fn class_shares(labels: &[u8], classes: usize) -> Vec<f64> {
    let mut counts = vec![0_usize; classes];
    for &label in labels {
        counts[label as usize] += 1;
    }
    counts
        .into_iter()
        .map(|count| count as f64 / labels.len() as f64)
        .collect()
}

/// The requested marginal of every class, recomputed independently of the
/// generator.
///
/// Deliberately a second derivation rather than a call into the module under
/// test: an assertion that the realized balance matches the requested one is
/// worth nothing if "requested" is read back out of the same code that produced
/// it.
fn requested_shares(balance: ClassBalance, classes: usize) -> Vec<f64> {
    let weights: Vec<f64> = match balance {
        ClassBalance::Imbalanced { ratio } => (0..classes)
            .map(|class| f64::from(ratio).powf(-(class as f64) / (classes - 1) as f64))
            .collect(),
        _ => vec![1.0; classes],
    };
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / total).collect()
}

/// Fits the crate's own least-squares path to a row window of a dataset.
fn fit_window(dataset: &Dataset, from: usize, to: usize) -> LinearRegression {
    let indices: Vec<usize> = (from..to).collect();
    let design = dataset
        .features()
        .select_rows(&indices)
        .expect("an in-range window");
    let targets = match dataset.target() {
        Some(Target::Regression(targets)) => targets.select(&indices).expect("an in-range window"),
        other => panic!("expected a regression target, got {other:?}"),
    };
    LinearRegression::fit(
        &design.as_view(),
        &targets,
        LinearRegressionParams::default(),
    )
    .expect("a generated window fits")
}

/// The bit-exact structural families emit exactly the values they emitted when
/// they landed.
///
/// `assert_eq!` on `f32` rather than a tolerance, and allowed to be: a cluster
/// centre is a uniform draw, a row time is a division of two exactly
/// representable integers, a drifting coefficient is an interpolation, and a
/// relevance grade is integer arithmetic over a sort. Not one of the three
/// evaluates a transcendental, so a moved literal here is a defect in this crate
/// and never a platform difference — which is what [`Portability::BitExact`]
/// claims, asserted rather than described.
#[test]
fn the_bit_exact_structural_families_emit_their_recorded_values() {
    let clustered = Recipe::seeded(6, 3, 11)
        .unwrap()
        .with_task(Task::Clustered {
            blobs: 3,
            spread: 0.1,
        })
        .unwrap();
    assert_eq!(clustered.portability(), Portability::BitExact);
    let clustered = clustered.generate();
    assert_eq!(
        clustered.truth().cluster_centres().unwrap(),
        [
            0.5341575,
            -0.15752995,
            0.18214095,
            0.021734715,
            -0.80112684,
            -0.6121378,
            -0.372234,
            0.064558506,
            -0.044912457,
        ]
    );
    assert_eq!(
        clustered.features().as_slice(),
        [
            0.4725672,
            -0.11002739,
            0.096132,
            -0.067390524,
            -0.88632333,
            -0.6267451,
            -0.42296356,
            -0.028363302,
            -0.10261122,
            0.46847016,
            -0.11479499,
            0.15346709,
            -0.07578597,
            -0.84842616,
            -0.68372166,
            -0.3942931,
            0.07770334,
            0.011719953,
        ]
    );
    assert_eq!(
        clustered.truth().cluster_assignments().unwrap(),
        [0, 1, 2, 0, 1, 2]
    );
    assert_eq!(clustered.truth().blobs(), Some(3));

    let timed = Recipe::seeded(6, 4, 11)
        .unwrap()
        .with_task(Task::TimeOrdered {
            informative: 2,
            coefficient_scale: 1.0,
            drift: 0.5,
            intercept: 0.25,
            noise_scale: 0.1,
        })
        .unwrap();
    assert_eq!(timed.portability(), Portability::BitExact);
    let timed = timed.generate();
    // The uninformative columns are exactly zero at *both* ends, not merely
    // small: "this column never mattered, and never starts to" is a statement a
    // drift detector has to be checkable against.
    assert_eq!(
        timed.truth().start_coefficients().unwrap(),
        [-0.16027308, 0.18930042, 0.0, 0.0]
    );
    assert_eq!(
        timed.truth().end_coefficients().unwrap(),
        [0.22536999, 0.13616359, 0.0, 0.0]
    );
    assert_eq!(
        timed.truth().times().unwrap(),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    );
    assert_eq!(
        timed.truth().conditional_mean().unwrap(),
        [
            0.43863523, 0.2947369, 0.1430863, 0.10619249, 0.35261735, 0.22191614,
        ]
    );
    assert_eq!(
        regression_values(&timed),
        [
            0.33969685,
            0.28984722,
            0.108372755,
            0.08172676,
            0.25502843,
            0.1702217,
        ]
    );

    let ranked = Recipe::seeded(12, 3, 11)
        .unwrap()
        .with_task(Task::Ranking {
            queries: 3,
            docs_per_query: 4,
            grades: 3,
            informative: 2,
            coefficient_scale: 1.0,
        })
        .unwrap();
    assert_eq!(ranked.portability(), Portability::BitExact);
    let ranked = ranked.generate();
    assert_eq!(
        ranked.truth().coefficients().unwrap(),
        [0.9085052, 0.4818691, 0.0]
    );
    assert_eq!(
        ranked.truth().utilities().unwrap(),
        [
            -0.33065102,
            -1.2202431,
            -0.9086422,
            -0.39084652,
            -1.1139015,
            -0.13706721,
            -0.22317916,
            -0.68870586,
            0.23152404,
            -0.55346364,
            0.7310799,
            -0.3483056,
        ]
    );
    assert_eq!(class_labels(&ranked), [2, 0, 1, 2, 0, 2, 2, 1, 2, 0, 2, 1]);
    assert_eq!(
        ranked.groups().unwrap(),
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    );
    assert_eq!(ranked.truth().grades(), Some(3));
}

/// Group labels partition the rows, and feed `GroupKFold::split` unadapted.
///
/// "Partition" is three claims and all three are asserted: every row carries
/// exactly one label, the labels are exactly `0..groups` with none unused, and
/// the sizes sum to the row count. A generator that produced a label set with a
/// hole in it would still split — `GroupKFold` counts distinct identifiers — and
/// would quietly give a caller asking for twelve groups eleven.
///
/// The unadapted claim is made by *doing* it: `dataset.groups()` is `&[u64]` and
/// goes straight into `GroupKFold::split`, with no conversion between them. What
/// comes back is then checked for the integrity property the splitter exists
/// for, over every pattern, because a grouping the splitter cannot honour is a
/// grouping this module should not produce.
#[test]
fn group_labels_partition_the_rows_and_feed_group_k_fold_unadapted() {
    const ROWS: usize = 600;
    for pattern in [
        GroupPattern::RoundRobin { groups: 7 },
        GroupPattern::Contiguous { groups: 7 },
        GroupPattern::Unbalanced {
            groups: 4,
            ratio: 3.0,
        },
        GroupPattern::Unbalanced {
            groups: 5,
            ratio: 1.0,
        },
    ] {
        let recipe = Recipe::seeded(ROWS, 4, 9)
            .unwrap()
            .with_groups(pattern)
            .unwrap();
        assert_eq!(recipe.group_pattern(), Some(pattern));
        // A grouping is a label, not a design change: the matrix is the one the
        // ungrouped recipe produces, byte for byte.
        assert_eq!(
            recipe.design().as_slice(),
            Recipe::seeded(ROWS, 4, 9).unwrap().design().as_slice()
        );
        let dataset = recipe.generate();
        let groups = dataset.groups().expect("a grouped recipe");
        assert_eq!(groups.len(), ROWS);

        let mut sizes: BTreeMap<u64, usize> = BTreeMap::new();
        for &group in groups {
            *sizes.entry(group).or_default() += 1;
        }
        let expected = match pattern {
            GroupPattern::RoundRobin { groups } | GroupPattern::Contiguous { groups } => groups,
            GroupPattern::Unbalanced { groups, .. } => groups,
        };
        assert_eq!(sizes.len(), expected, "{pattern:?} lost a group");
        assert!(
            sizes.keys().copied().eq(0..expected as u64),
            "{pattern:?} left a hole in the identifiers: {:?}",
            sizes.keys().collect::<Vec<_>>()
        );
        assert_eq!(sizes.values().sum::<usize>(), ROWS);

        // Unadapted: `&[u64]` in, splits out.
        let folds: Vec<_> = GroupKFold::new(3).split(groups).unwrap().collect();
        assert_eq!(folds.len(), 3);
        for fold in &folds {
            let held: Vec<bool> = {
                let mut mask = vec![false; ROWS];
                for &row in fold.test_indices() {
                    mask[row] = true;
                }
                mask
            };
            for group in 0..expected as u64 {
                let rows: Vec<usize> = (0..ROWS).filter(|&row| groups[row] == group).collect();
                let inside = rows.iter().filter(|&&row| held[row]).count();
                assert!(
                    inside == 0 || inside == rows.len(),
                    "{pattern:?} group {group} straddles a split"
                );
            }
        }
    }
}

/// An unbalanced grouping realizes the size ratio it was asked for.
///
/// Exactly, at a row count that divides evenly: the sizes are an apportionment
/// of `rows` across weights that interpolate linearly from `ratio` down to `1`,
/// so with `600` rows over `4` groups the ideal sizes `225, 175, 125, 75` are
/// already integers and the largest is exactly three times the smallest. The
/// assertion is `assert_eq!` on the sizes rather than a tolerance on the ratio,
/// because at this shape there is nothing left to round.
///
/// The general case is asserted separately and weakly on purpose: apportionment
/// of an indivisible row count cannot realize an arbitrary ratio exactly, and
/// claiming it could would be the sort of tolerance-shaped promise this module
/// avoids. What it does promise is that the sizes sum to the row count and that
/// no group is empty, which is what the partition claim needs.
#[test]
fn an_unbalanced_grouping_realizes_the_requested_size_ratio() {
    let dataset = Recipe::seeded(600, 3, 4)
        .unwrap()
        .with_groups(GroupPattern::Unbalanced {
            groups: 4,
            ratio: 3.0,
        })
        .unwrap()
        .generate();
    let groups = dataset.groups().unwrap();
    let sizes: Vec<usize> = (0..4)
        .map(|group| groups.iter().filter(|&&g| g == group).count())
        .collect();
    assert_eq!(sizes, [225, 175, 125, 75]);

    // A ratio of one is the equal grouping, not a degenerate case of the
    // unbalanced one.
    let equal = Recipe::seeded(600, 3, 4)
        .unwrap()
        .with_groups(GroupPattern::Unbalanced {
            groups: 5,
            ratio: 1.0,
        })
        .unwrap()
        .generate();
    let groups = equal.groups().unwrap();
    for group in 0..5 {
        assert_eq!(groups.iter().filter(|&&g| g == group).count(), 120);
    }

    // A row count that does not divide still partitions, at every ratio.
    for (rows, count, ratio) in [(101_usize, 7_usize, 5.0_f32), (37, 37, 2.5), (9, 4, 100.0)] {
        let dataset = Recipe::seeded(rows, 3, 4)
            .unwrap()
            .with_groups(GroupPattern::Unbalanced {
                groups: count,
                ratio,
            })
            .unwrap()
            .generate();
        let groups = dataset.groups().unwrap();
        let sizes: Vec<usize> = (0..count as u64)
            .map(|group| groups.iter().filter(|&&g| g == group).count())
            .collect();
        assert_eq!(sizes.iter().sum::<usize>(), rows, "{rows}/{count}@{ratio}");
        assert!(
            sizes.iter().all(|&size| size > 0),
            "{rows}/{count}@{ratio} left a group empty: {sizes:?}"
        );
        assert!(
            sizes.windows(2).all(|pair| pair[0] >= pair[1]),
            "{rows}/{count}@{ratio} is not monotone: {sizes:?}"
        );
    }
}

/// Time order is monotone, and the drift that was asked for is the drift that
/// arrives.
///
/// **The monotone half is exact.** Row `i`'s time is `i / (rows - 1)`, a
/// division of two integers both exact in `f32` below `2^24`, so the sequence is
/// strictly increasing and `TimeSeriesSplit` — which reads the row order and
/// nothing else — is correct on this data with no adapter at all. That is
/// asserted by calling it.
///
/// **The drift half is a measurement, and its tolerance is derived.** Fit the
/// first quarter of the rows and the last quarter. For a design whose rows are
/// independent of their times, ordinary least squares over a window recovers
/// `β(t̄)` — the coefficient vector at the window's *mean* time — plus two
/// errors:
///
/// * the ordinary noise term, standard error `σ sqrt(3 / w)` with
///   `σ = noise_scale / sqrt(3)`, so `noise_scale / sqrt(w)`. At
///   `noise_scale = 0.01` and `w = 1024` that is `3.1e-4`;
/// * a window-averaging fluctuation. Writing `t_i = t̄ + s_i`, the estimate
///   carries `A⁻¹ C δ` with `A = Σ x xᵀ ≈ w/3 I` and `C = Σ s_i x_i x_iᵀ`, whose
///   entries have zero mean and standard deviation about
///   `0.3 sqrt(Σ s_i²) = 0.086 sqrt(w) Δt` for a window spanning `Δt`. That
///   gives about `0.26 Δt sqrt(p) ‖δ‖_∞ / sqrt(w)`, which at `Δt = 0.25`,
///   `p = 6`, `‖δ‖_∞ ≤ 1` and `w = 1024` is `5.0e-3`.
///
/// The second dominates by a factor of sixteen. The tolerance below is `0.03`,
/// about five times that bound; measured worst deviations are `6.1e-3` at
/// `drift = 1` and `6.0e-4` at `drift = 0`. The signal it is measured against is
/// `(t̄_last - t̄_first) |δ| = 0.75 |δ|`, so the test is not merely detecting
/// drift — it is asserting the *magnitude* to within four percent of the largest
/// coefficient's move.
///
/// The `drift = 0` case is what makes that a two-sided claim: a stationary
/// series must measure as stationary under the same tolerance, or the assertion
/// would pass on any generator that merely moved things around.
#[test]
fn time_order_is_monotone_and_the_drift_is_the_one_that_was_asked_for() {
    const ROWS: usize = 4096;
    const COLUMNS: usize = 6;
    const TOLERANCE: f64 = 0.03;
    let quarter = ROWS / 4;

    for drift in [0.0_f32, 1.0] {
        let dataset = Recipe::seeded(ROWS, COLUMNS, 17)
            .unwrap()
            .with_task(Task::TimeOrdered {
                informative: 4,
                coefficient_scale: 1.0,
                drift,
                intercept: 0.5,
                noise_scale: 0.01,
            })
            .unwrap()
            .generate();

        let times = dataset.truth().times().unwrap();
        assert_eq!(times.len(), ROWS);
        assert!(
            times.windows(2).all(|pair| pair[0] < pair[1]),
            "row order must be time order"
        );
        assert_eq!(times[0], 0.0);
        assert_eq!(times[ROWS - 1], 1.0);

        // Unadapted: the splitter takes a sample count, and the data is correct
        // for it because the rows are already in time order.
        let splits: Vec<_> = TimeSeriesSplit::new(4).split(ROWS).unwrap().collect();
        assert_eq!(splits.len(), 4);
        for split in &splits {
            let last_train = *split.train_indices().last().unwrap();
            let first_test = *split.test_indices().first().unwrap();
            assert!(
                last_train < first_test,
                "a time-series split trained on the future"
            );
        }

        let start = dataset.truth().start_coefficients().unwrap();
        let end = dataset.truth().end_coefficients().unwrap();
        let mean_time = |window: &[f32]| {
            window.iter().map(|&t| f64::from(t)).sum::<f64>() / window.len() as f64
        };
        let span = mean_time(&times[ROWS - quarter..]) - mean_time(&times[..quarter]);

        let first = fit_window(&dataset, 0, quarter);
        let last = fit_window(&dataset, ROWS - quarter, ROWS);
        let mut largest_predicted = 0.0_f64;
        for column in 0..COLUMNS {
            let predicted = span * f64::from(end[column] - start[column]);
            let measured = f64::from(last.coefficients()[column] - first.coefficients()[column]);
            largest_predicted = largest_predicted.max(predicted.abs());
            assert!(
                (predicted - measured).abs() <= TOLERANCE,
                "drift {drift} column {column}: recorded ends predict {predicted}, \
                 two windows measured {measured}"
            );
        }
        if drift == 0.0 {
            assert_eq!(
                largest_predicted, 0.0,
                "a stationary family predicts no drift"
            );
        } else {
            // The tolerance has to be small against the thing being measured, or
            // the test would pass on a generator that drifted by nothing.
            assert!(
                largest_predicted > 10.0 * TOLERANCE,
                "the drift signal {largest_predicted} is not large against the tolerance"
            );
        }
    }
}

/// Ranking pairs feed the `ranking` module's pair types unadapted.
///
/// Three claims, in order of how tight they are.
///
/// **Exact, no tolerance:** every pair lies inside one query, its indices are
/// design rows, its outcome is the comparison of the two grades, and — the one
/// that matters — the recorded utility orders every decisive pair the way the
/// pair says. That last one is what makes this data a *ranking* problem rather
/// than a pile of labelled differences, and it is checked against the truth with
/// `assert!` on a sign, not a threshold.
///
/// **Structural:** `dataset.pairs()` is `&[PairwiseObservation]` and goes
/// straight into `PairwiseLinearRanker::fit`, and `PairIndex` values index the
/// same matrix the fit is given. No conversion appears between them anywhere in
/// this test; that absence is the claim.
///
/// **Measured, with a floor set by the failure it detects:** a fitted ranker
/// agrees with the recorded order. The pairs are exactly separable by the
/// recorded coefficients, so what the floor is guarding against is not a hard
/// problem but a *scrambled* one — pairs drawn across query boundaries, an
/// index off by a block, an inverted outcome — each of which puts agreement at
/// chance. `0.95` sits far above chance and below the measured `0.986` worst
/// case; it is chosen from the failure mode, not from what passed.
#[test]
fn ranking_pairs_feed_the_ranking_module_unadapted() {
    for (queries, docs_per_query, grades) in [(64_usize, 8_usize, 3_usize), (128, 4, 2)] {
        let rows = queries * docs_per_query;
        let dataset = Recipe::seeded(rows, 5, 21)
            .unwrap()
            .with_task(Task::Ranking {
                queries,
                docs_per_query,
                grades,
                informative: 4,
                coefficient_scale: 1.5,
            })
            .unwrap()
            .generate();

        let pairs = dataset.pairs().expect("a ranking family draws pairs");
        let groups = dataset.groups().expect("a ranking family groups by query");
        let labels = class_labels(&dataset);
        let utilities = dataset.truth().utilities().unwrap();
        assert_eq!(
            pairs.len(),
            queries * docs_per_query * (docs_per_query - 1) / 2
        );
        assert_eq!(
            *labels.iter().max().unwrap() as usize,
            grades - 1,
            "the best document in a query carries the top grade"
        );

        for observation in pairs {
            let (left, right) = (observation.pair().left(), observation.pair().right());
            assert!(left < rows && right < rows, "a pair left the design");
            assert_eq!(groups[left], groups[right], "a pair crossed a query");
            let expected = match labels[left].cmp(&labels[right]) {
                std::cmp::Ordering::Greater => PairOutcome::LeftPreferred,
                std::cmp::Ordering::Less => PairOutcome::RightPreferred,
                std::cmp::Ordering::Equal => PairOutcome::Tie,
            };
            assert_eq!(observation.outcome(), expected);
            assert_eq!(observation.weight(), 1.0);
            // The recorded utility orders every decisive pair. This is the claim
            // that makes the pairs separable, and it holds exactly.
            match observation.outcome() {
                PairOutcome::LeftPreferred => assert!(utilities[left] > utilities[right]),
                PairOutcome::RightPreferred => assert!(utilities[left] < utilities[right]),
                PairOutcome::Tie => {}
            }
        }

        let ranker = PairwiseLinearRanker::fit(
            &dataset.features().as_view(),
            pairs,
            PairwiseLinearRankerParams::default(),
        )
        .expect("generated pairs fit the crate's own ranker");
        assert_eq!(ranker.n_features_in(), 5);

        let indices: Vec<_> = pairs.iter().map(|pair| pair.pair()).collect();
        let observed: Vec<PairOutcome> = pairs.iter().map(|pair| pair.outcome()).collect();
        let predicted = ranker
            .compare(&dataset.features().as_view(), &indices)
            .unwrap();
        let decisive = decisive_directional_accuracy(&observed, &predicted).unwrap();
        assert!(
            decisive >= 0.95,
            "{queries}x{docs_per_query}: decisive agreement {decisive}"
        );

        let scores = ranker.score_items(&dataset.features().as_view()).unwrap();
        let rho = spearman_correlation(
            &utilities.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &scores.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(
            rho >= 0.95,
            "{queries}x{docs_per_query}: recovered order correlates {rho}"
        );
    }
}

/// The realized class balance matches the request within four binomial
/// deviations.
///
/// Two separate things are asserted, and conflating them is the mistake this
/// test avoids.
///
/// The **solver's** residual is how far the mean recorded probability of a class
/// sits from the requested marginal. That is the fixed point
/// `fit_class_offsets` solves, it has nothing to do with sampling, and it is
/// held to `1e-6` — six orders of magnitude tighter than anything below.
/// Measured worst over the eight cases: `4.5e-10`.
///
/// The **draw's** deviation is how far the realized label frequency sits from
/// that marginal. That is a Poisson-binomial mean with expectation exactly the
/// requested marginal, and its variance is at most the binomial one with the
/// same mean, so `4 sqrt(p (1 - p) / n)` is a conservative four-sigma bound —
/// exceeded with probability below `6e-5` per case even before the
/// conservatism. Measured worst: `1.57` deviations.
#[test]
fn the_realized_class_balance_matches_the_request_within_four_binomial_deviations() {
    const ROWS: usize = 4096;
    const SOLVER_TOLERANCE: f64 = 1e-6;
    for classes in [3_usize, 5] {
        for geometry in [ClassGeometry::Blob, ClassGeometry::Hierarchical] {
            for balance in [
                ClassBalance::Balanced,
                ClassBalance::Imbalanced { ratio: 3.0 },
            ] {
                let recipe = Recipe::seeded(ROWS, 6, 31)
                    .unwrap()
                    .with_task(Task::Multiclass {
                        classes,
                        balance,
                        geometry,
                        separation: 2.0,
                    })
                    .unwrap();
                assert_eq!(recipe.portability(), Portability::PerRunner);
                let dataset = recipe.generate();

                let probabilities = dataset.truth().class_probabilities().unwrap();
                assert_eq!(dataset.truth().classes(), Some(classes));
                assert_eq!(probabilities.len(), ROWS * classes);
                for row in probabilities.chunks_exact(classes) {
                    let total: f64 = row.iter().map(|&p| f64::from(p)).sum();
                    assert!((total - 1.0).abs() < 1e-5, "a row summed to {total}");
                }

                let requested = requested_shares(balance, classes);
                let realized = class_shares(class_labels(&dataset), classes);
                for class in 0..classes {
                    let modelled: f64 = probabilities
                        .chunks_exact(classes)
                        .map(|row| f64::from(row[class]))
                        .sum::<f64>()
                        / ROWS as f64;
                    assert!(
                        (modelled - requested[class]).abs() <= SOLVER_TOLERANCE,
                        "{classes} {geometry:?} {balance:?} class {class}: the offset solver \
                         left the mean probability at {modelled}, not {}",
                        requested[class]
                    );
                    let tolerance = binomial_tolerance(requested[class], ROWS);
                    assert!(
                        (realized[class] - requested[class]).abs() <= tolerance,
                        "{classes} {geometry:?} {balance:?} class {class}: realized \
                         {} against a requested {} at tolerance {tolerance}",
                        realized[class],
                        requested[class]
                    );
                }

                if let ClassBalance::Imbalanced { ratio } = balance {
                    // The ratio is the knob's own definition, and it is the
                    // *requested* marginals that have to realize it exactly.
                    let ratio = f64::from(ratio);
                    assert!(
                        (requested[0] / requested[classes - 1] - ratio).abs() < 1e-9,
                        "the requested marginals do not realize the ratio"
                    );
                    assert!(
                        realized[0] > realized[classes - 1],
                        "an imbalanced draw put the rare class ahead of the common one"
                    );
                }
            }
        }
    }
}

/// The two geometries are two different confusion structures, not one problem
/// with two names.
///
/// A hierarchy's promise is *nested* confusion: a class is mistaken for its
/// sibling more readily than for a cousin. That is derived rather than hoped
/// for. With four classes the score of class `k` is
/// `sep (s₀(k) z₀ + ½ s₁(k) z₁)` where `s₀` reads the top bit and `s₁` the
/// bottom one. Flipping the bottom bit — the sibling — moves the score by
/// `sep |z₁|`; flipping the top bit — a cousin — moves it by `2 sep |z₀|`. The
/// two projections are identically distributed, so the expected score gap to a
/// cousin is exactly **twice** the gap to a sibling, and the sibling therefore
/// carries the larger share of the probability mass. The assertion is that
/// inequality, per class, with no threshold in it.
///
/// The blob geometry carries no such guarantee — its confusable pairs are
/// whichever centres happen to land near each other — and this test does not
/// require it to fail, only to be a different problem. Measured on the fixture
/// below, blob class `3` does put more mass on a cousin than on its sibling
/// (`0.178` against `0.119`), which is the observation that makes the hierarchy
/// worth having.
#[test]
fn the_two_multiclass_geometries_are_two_different_confusion_structures() {
    const CLASSES: usize = 4;
    let recipe = |geometry| {
        Recipe::seeded(4096, 6, 41)
            .unwrap()
            .with_task(Task::Multiclass {
                classes: CLASSES,
                balance: ClassBalance::Balanced,
                geometry,
                separation: 3.0,
            })
            .unwrap()
            .generate()
    };
    let blob = recipe(ClassGeometry::Blob);
    let tree = recipe(ClassGeometry::Hierarchical);

    assert_ne!(
        class_labels(&blob),
        class_labels(&tree),
        "two geometries at one balance must be two problems"
    );
    assert_ne!(blob.spec_digest(), tree.spec_digest());

    let probabilities = tree.truth().class_probabilities().unwrap();
    let mut sibling = [0.0_f64; CLASSES];
    let mut cousin = [0.0_f64; CLASSES];
    let mut counts = [0_usize; CLASSES];
    for row in probabilities.chunks_exact(CLASSES) {
        let top = (0..CLASSES)
            .max_by(|&left, &right| row[left].total_cmp(&row[right]))
            .unwrap();
        counts[top] += 1;
        for (other, &mass) in row.iter().enumerate() {
            if other == top {
                continue;
            } else if other == top ^ 1 {
                sibling[top] += f64::from(mass);
            } else {
                // Two cousins, averaged, so the comparison is mass-per-class
                // against mass-per-class rather than one against two.
                cousin[top] += f64::from(mass) / 2.0;
            }
        }
    }
    for class in 0..CLASSES {
        assert!(counts[class] > 0, "class {class} was never the mode");
        let sibling = sibling[class] / counts[class] as f64;
        let cousin = cousin[class] / counts[class] as f64;
        assert!(
            sibling > cousin,
            "class {class} confuses a cousin ({cousin}) more than its sibling ({sibling})"
        );
    }
}

/// Multiclass label noise overlays the draw; it does not reseed it.
///
/// This is the defect the phase before this one found and fixed for the binary
/// family, and it must not reappear with a different label vocabulary. Seeding
/// the label stream from anything the contamination touches makes the clean and
/// contaminated datasets two *independent* draws, and a five-percent request
/// then changes about `(K - 1) / K` of the labels — seventy-five percent at four
/// classes — while still looking like a plausible dataset.
///
/// Three assertions, and the middle one is the load-bearing one:
///
/// * the design matrix is **identical**, byte for byte;
/// * the fraction of labels that changed is the requested rate, to within four
///   binomial deviations. A flip always lands on a different class, so the
///   change rate *is* the flip rate; a reseed would put it near `0.75`, which is
///   about eighty standard deviations away;
/// * the recorded probabilities are the clean ones put through the noise
///   channel, `p(1 - e) + (1 - p) e / (K - 1)`, to within `1e-6`. That is `8`
///   `f32` units at magnitude one, against three roundings; measured worst
///   `5.6e-8`. It proves the *clean* probabilities are unmoved — the strongest
///   form of "the rows the noise did not touch are identical", because it holds
///   for every row rather than for the ones that happened not to flip.
#[test]
fn multiclass_label_noise_overlays_rather_than_reseeds() {
    const CLASSES: usize = 4;
    const ROWS: usize = 4096;
    let base = Recipe::seeded(ROWS, 6, 51)
        .unwrap()
        .with_task(Task::Multiclass {
            classes: CLASSES,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Hierarchical,
            separation: 2.5,
        })
        .unwrap();
    let clean = base.generate();

    for rate in [0.05_f32, 0.2] {
        let noisy = base
            .with_contamination(Contamination::none().with_label_noise(rate))
            .unwrap()
            .generate();
        assert_eq!(
            clean.features().as_slice(),
            noisy.features().as_slice(),
            "a label knob moved the design"
        );

        let before = class_labels(&clean);
        let after = class_labels(&noisy);
        let changed = before.iter().zip(after).filter(|(a, b)| a != b).count() as f64 / ROWS as f64;
        let rate = f64::from(rate);
        let tolerance = binomial_tolerance(rate, ROWS);
        assert!(
            (changed - rate).abs() <= tolerance,
            "requested {rate}, changed {changed}, tolerance {tolerance}"
        );
        // The reseeded outcome, named so the assertion above is read as ruling
        // it out rather than as a loose bound.
        let reseeded = (CLASSES - 1) as f64 / CLASSES as f64;
        assert!(changed < reseeded / 2.0);

        let clean_probabilities = clean.truth().class_probabilities().unwrap();
        let noisy_probabilities = noisy.truth().class_probabilities().unwrap();
        for (index, (&before, &after)) in clean_probabilities
            .iter()
            .zip(noisy_probabilities)
            .enumerate()
        {
            let expected = f64::from(before) * (1.0 - rate)
                + (1.0 - f64::from(before)) * rate / (CLASSES - 1) as f64;
            assert!(
                (expected - f64::from(after)).abs() <= 1e-6,
                "entry {index}: the clean probability {before} moved under noise {rate}"
            );
        }
    }
}

/// A clustered family has no target, and its recorded assignment is recoverable
/// from the design.
///
/// The no-target half is the reason
/// [`Dataset::target`](super::Dataset::target) is an `Option` at all, and it is
/// asserted three ways: the target is `None`, the numeric view is `None` too,
/// and the truth is a cluster assignment rather than
/// [`Truth::Unrecorded`] — the family knows the answer, it simply is not a
/// column of numbers.
///
/// The recoverable half is derived, not tolerated. Every row sits within
/// `spread * sqrt(columns)` of its own centre, because each coordinate is
/// displaced by at most `spread`. By the triangle inequality, nearest-centre
/// assignment is exactly the recorded one whenever twice that bound is below the
/// smallest distance between two centres — so the test *checks that condition*
/// on the generated centres and then asserts exact recovery for every row. There
/// is no tolerance: the separation either holds or it does not, and where it
/// holds the recovery is complete.
#[test]
fn a_clustered_family_has_no_target_and_records_a_recoverable_assignment() {
    const ROWS: usize = 512;
    const COLUMNS: usize = 4;
    const BLOBS: usize = 5;
    const SPREAD: f32 = 0.1;

    let recipe = Recipe::seeded(ROWS, COLUMNS, 13)
        .unwrap()
        .with_task(Task::Clustered {
            blobs: BLOBS,
            spread: SPREAD,
        })
        .unwrap();
    let dataset = recipe.generate();

    assert!(dataset.target().is_none());
    assert_eq!(recipe.target_values(), None);
    let mut buffer = vec![1.0_f32];
    recipe.target_values_into(&mut buffer);
    assert!(buffer.is_empty());
    assert_ne!(dataset.truth(), &Truth::Unrecorded);
    assert_ne!(dataset.truth(), &Truth::DesignOnly);

    let centres = dataset.truth().cluster_centres().unwrap();
    let assignments = dataset.truth().cluster_assignments().unwrap();
    assert_eq!(centres.len(), BLOBS * COLUMNS);
    assert_eq!(assignments.len(), ROWS);
    // Dealt in turn, so the clusters are as equal as the row count allows.
    for blob in 0..BLOBS {
        let members = assignments.iter().filter(|&&a| a == blob).count();
        assert!(
            members.abs_diff(ROWS / BLOBS) <= 1,
            "cluster {blob} holds {members}"
        );
    }

    let distance = |left: &[f32], right: &[f32]| -> f64 {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut smallest_gap = f64::INFINITY;
    for left in 0..BLOBS {
        for right in left + 1..BLOBS {
            smallest_gap = smallest_gap.min(distance(
                &centres[left * COLUMNS..(left + 1) * COLUMNS],
                &centres[right * COLUMNS..(right + 1) * COLUMNS],
            ));
        }
    }
    let displacement = f64::from(SPREAD) * (COLUMNS as f64).sqrt();
    assert!(
        2.0 * displacement < smallest_gap,
        "the fixture's clusters are not separable: gap {smallest_gap}, displacement {displacement}"
    );
    for (row, &expected) in dataset.features().iter_rows().zip(assignments) {
        let nearest = (0..BLOBS)
            .min_by(|&left, &right| {
                distance(row, &centres[left * COLUMNS..(left + 1) * COLUMNS]).total_cmp(&distance(
                    row,
                    &centres[right * COLUMNS..(right + 1) * COLUMNS],
                ))
            })
            .unwrap();
        assert_eq!(
            nearest, expected,
            "a row is nearer another cluster's centre"
        );
    }

    // A zero spread collapses each cluster onto its centre exactly, which is the
    // degenerate case a clusterer has no excuse on.
    let tight = Recipe::seeded(ROWS, COLUMNS, 13)
        .unwrap()
        .with_task(Task::Clustered {
            blobs: BLOBS,
            spread: 0.0,
        })
        .unwrap()
        .generate();
    let centres = tight.truth().cluster_centres().unwrap();
    for (index, row) in tight.features().iter_rows().enumerate() {
        let blob = index % BLOBS;
        assert_eq!(row, &centres[blob * COLUMNS..(blob + 1) * COLUMNS]);
    }
}

/// Every structural parameter is refused by name, before anything is generated.
#[test]
fn every_structural_parameter_is_refused_by_name_before_generation() {
    let recipe = Recipe::seeded(32, 6, 1).unwrap();

    for classes in [0_usize, 1] {
        assert_eq!(
            recipe.with_task(Task::Multiclass {
                classes,
                balance: ClassBalance::Balanced,
                geometry: ClassGeometry::Blob,
                separation: 1.0,
            }),
            Err(DatasetError::TooFewClasses { classes })
        );
    }
    assert_eq!(
        recipe.with_task(Task::Multiclass {
            classes: 257,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Blob,
            separation: 1.0,
        }),
        Err(DatasetError::TooManyClasses {
            classes: 257,
            limit: 256
        })
    );
    assert_eq!(
        recipe.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Blob,
            separation: 0.0,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Separation
        })
    );
    assert_eq!(
        recipe.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Imbalanced { ratio: 0.5 },
            geometry: ClassGeometry::Blob,
            separation: 1.0,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::BalanceRatio
        })
    );
    assert_eq!(
        recipe.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Imbalanced { ratio: f32::NAN },
            geometry: ClassGeometry::Blob,
            separation: 1.0,
        }),
        Err(DatasetError::NonFiniteParameter {
            parameter: Parameter::BalanceRatio
        })
    );

    assert_eq!(
        recipe.with_task(Task::Clustered {
            blobs: 0,
            spread: 0.1
        }),
        Err(DatasetError::ZeroBlobs)
    );
    assert_eq!(
        recipe.with_task(Task::Clustered {
            blobs: 33,
            spread: 0.1
        }),
        Err(DatasetError::BlobsExceedRows {
            blobs: 33,
            rows: 32
        })
    );
    assert_eq!(
        recipe.with_task(Task::Clustered {
            blobs: 3,
            spread: -0.1
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Spread
        })
    );

    assert_eq!(
        recipe.with_task(Task::TimeOrdered {
            informative: 2,
            coefficient_scale: 1.0,
            drift: -1.0,
            intercept: 0.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Drift
        })
    );
    assert_eq!(
        recipe.with_task(Task::TimeOrdered {
            informative: 7,
            coefficient_scale: 1.0,
            drift: 0.0,
            intercept: 0.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::InformativeColumnsExceedDesign {
            informative: 7,
            columns: 6
        })
    );

    assert_eq!(
        recipe.with_task(Task::Ranking {
            queries: 32,
            docs_per_query: 1,
            grades: 2,
            informative: 2,
            coefficient_scale: 1.0,
        }),
        Err(DatasetError::TooFewDocumentsPerQuery { docs_per_query: 1 })
    );
    assert_eq!(
        recipe.with_task(Task::Ranking {
            queries: 8,
            docs_per_query: 4,
            grades: 1,
            informative: 2,
            coefficient_scale: 1.0,
        }),
        Err(DatasetError::TooFewGrades { grades: 1 })
    );
    assert_eq!(
        recipe.with_task(Task::Ranking {
            queries: 7,
            docs_per_query: 4,
            grades: 2,
            informative: 2,
            coefficient_scale: 1.0,
        }),
        Err(DatasetError::RankingShapeMismatch {
            rows: 32,
            queries: 7,
            docs_per_query: 4
        })
    );

    assert_eq!(
        recipe.with_groups(GroupPattern::RoundRobin { groups: 0 }),
        Err(DatasetError::ZeroGroups)
    );
    assert_eq!(
        recipe.with_groups(GroupPattern::Contiguous { groups: 33 }),
        Err(DatasetError::GroupsExceedRows {
            groups: 33,
            rows: 32
        })
    );
    assert_eq!(
        recipe.with_groups(GroupPattern::Unbalanced {
            groups: 4,
            ratio: 0.5
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::GroupSizeRatio
        })
    );

    // Every message names itself, and no two of them read the same.
    let messages: Vec<String> = [
        DatasetError::TooFewClasses { classes: 1 },
        DatasetError::TooManyClasses {
            classes: 257,
            limit: 256,
        },
        DatasetError::ZeroBlobs,
        DatasetError::BlobsExceedRows {
            blobs: 33,
            rows: 32,
        },
        DatasetError::RankingShapeMismatch {
            rows: 32,
            queries: 7,
            docs_per_query: 4,
        },
        DatasetError::TooFewDocumentsPerQuery { docs_per_query: 1 },
        DatasetError::TooFewGrades { grades: 1 },
        DatasetError::ZeroGroups,
        DatasetError::GroupsExceedRows {
            groups: 33,
            rows: 32,
        },
        DatasetError::GroupPatternConflictsWithTask,
        DatasetError::ContaminationConflictsWithTask {
            parameter: Parameter::DuplicateRows,
        },
        DatasetError::ParameterOutOfRange {
            parameter: Parameter::BalanceRatio,
        },
        DatasetError::ParameterOutOfRange {
            parameter: Parameter::GroupSizeRatio,
        },
        DatasetError::ParameterOutOfRange {
            parameter: Parameter::Spread,
        },
        DatasetError::ParameterOutOfRange {
            parameter: Parameter::Drift,
        },
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    for (index, message) in messages.iter().enumerate() {
        assert!(
            message.chars().next().is_some_and(char::is_lowercase),
            "error messages read as sentence fragments: {message}"
        );
        assert!(
            !messages[index + 1..].contains(message),
            "two error variants share the message {message:?}"
        );
    }
}

/// A structural combination that would falsify the recorded truth is refused
/// rather than resolved.
///
/// Two of these, and both are the same discipline the contamination checks
/// already follow: a request that cannot be honoured is a build error, never a
/// silently different dataset.
///
/// * **Row duplication over a clustered design.** The recorded assignment is a
///   function of the row *index*, so a duplicated row would carry another
///   cluster's features under its own recorded label — a ground truth that is
///   wrong for exactly the rows a caller added on purpose.
/// * **A group pattern over a ranking task.** The task's group labels are its
///   query identifiers and its pairs are within-query by construction. A pattern
///   winning would leave the pairs and the groups describing two different
///   partitions of one design, and a leakage check run on the second would pass
///   while the first leaked.
#[test]
fn a_structural_combination_that_would_falsify_its_truth_is_refused() {
    let clustered = Recipe::seeded(64, 4, 3)
        .unwrap()
        .with_task(Task::Clustered {
            blobs: 4,
            spread: 0.2,
        })
        .unwrap();
    assert_eq!(
        clustered.with_contamination(Contamination::none().with_duplicate_rows(0.1)),
        Err(DatasetError::ContaminationConflictsWithTask {
            parameter: Parameter::DuplicateRows
        })
    );
    // Order of the builder calls does not change which recipes exist.
    assert_eq!(
        Recipe::seeded(64, 4, 3)
            .unwrap()
            .with_contamination(Contamination::none().with_duplicate_rows(0.1))
            .unwrap()
            .with_task(Task::Clustered {
                blobs: 4,
                spread: 0.2
            }),
        Err(DatasetError::ContaminationConflictsWithTask {
            parameter: Parameter::DuplicateRows
        })
    );
    // An unsupervised family has no target to displace, so an outlier fraction
    // is refused too — the check names the target it needs rather than negating
    // "draws labels", which would have let this through.
    assert_eq!(
        clustered.with_contamination(Contamination::none().with_outlier_fraction(0.05)),
        Err(DatasetError::ContaminationNeedsAdditiveNoise {
            parameter: Parameter::OutlierFraction
        })
    );
    assert_eq!(
        clustered.with_contamination(Contamination::none().with_label_noise(0.05)),
        Err(DatasetError::ContaminationNeedsLabels {
            parameter: Parameter::LabelNoise
        })
    );
    // A design-shaping knob that leaves the rows where they are still composes.
    assert!(
        clustered
            .with_contamination(Contamination::none().with_constant_columns(1))
            .is_ok()
    );

    let ranking = Recipe::seeded(32, 4, 3)
        .unwrap()
        .with_task(Task::Ranking {
            queries: 8,
            docs_per_query: 4,
            grades: 2,
            informative: 2,
            coefficient_scale: 1.0,
        })
        .unwrap();
    assert_eq!(
        ranking.with_groups(GroupPattern::Contiguous { groups: 4 }),
        Err(DatasetError::GroupPatternConflictsWithTask)
    );
    assert_eq!(
        Recipe::seeded(32, 4, 3)
            .unwrap()
            .with_groups(GroupPattern::Contiguous { groups: 4 })
            .unwrap()
            .with_task(Task::Ranking {
                queries: 8,
                docs_per_query: 4,
                grades: 2,
                informative: 2,
                coefficient_scale: 1.0,
            }),
        Err(DatasetError::GroupPatternConflictsWithTask)
    );
    // A ranking family's grades are ranks, so flipping one would contradict the
    // pairs derived from it.
    assert_eq!(
        ranking.with_contamination(Contamination::none().with_label_noise(0.05)),
        Err(DatasetError::ContaminationNeedsLabels {
            parameter: Parameter::LabelNoise
        })
    );
    assert_eq!(
        ranking.with_weights(WeightPattern::ClassBalanced),
        Err(DatasetError::WeightPatternNeedsLabels)
    );
}

/// Class-balancing weights generalize past two classes.
///
/// The pattern's promise is that every *observed* class carries the same total
/// weight. With `c` observed classes that total is `rows / c` each, which is a
/// generalization of the binary case rather than a change to it — at two
/// classes it is still half the row count apiece.
#[test]
fn class_balanced_weights_give_every_class_one_share() {
    const ROWS: usize = 2048;
    const CLASSES: usize = 5;
    let dataset = Recipe::seeded(ROWS, 6, 61)
        .unwrap()
        .with_task(Task::Multiclass {
            classes: CLASSES,
            balance: ClassBalance::Imbalanced { ratio: 4.0 },
            geometry: ClassGeometry::Blob,
            separation: 2.0,
        })
        .unwrap()
        .with_weights(WeightPattern::ClassBalanced)
        .unwrap()
        .generate();

    let labels = class_labels(&dataset);
    let weights = dataset.weights().unwrap().as_slice();
    let share = ROWS as f64 / CLASSES as f64;
    let mut counts = [0_usize; CLASSES];
    for &label in labels {
        counts[label as usize] += 1;
    }
    for (class, &count) in counts.iter().enumerate() {
        assert!(count > 0, "the fixture lost class {class}");
        let total: f64 = labels
            .iter()
            .zip(weights)
            .filter(|&(&label, _)| label as usize == class)
            .map(|(_, &weight)| f64::from(weight))
            .sum();
        // One rounding per row in `f32`, so the total carries at most
        // `rows * eps * share`; `1e-2` is far above that and far below the
        // `share / 4` an unbalanced family would show without the correction.
        assert!(
            (total - share).abs() < 1e-2,
            "class {class} totals {total}, not {share}"
        );
    }
    // The correction is doing something: the rarest class is up-weighted well
    // above the commonest.
    let rarest = counts
        .iter()
        .enumerate()
        .min_by_key(|&(_, &c)| c)
        .unwrap()
        .0;
    let commonest = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .unwrap()
        .0;
    let weight_of =
        |class: usize| weights[labels.iter().position(|&l| l as usize == class).unwrap()];
    assert!(weight_of(rarest) > weight_of(commonest));
}

/// The structural families extend the recipe's identity rather than sharing it.
///
/// Every knob any of them carries has to move the spec digest, or a materialized
/// dataset cached under it would be served for a request it does not answer. The
/// group pattern is included even though it changes no design value: it changes
/// the *dataset*, and the digest is the dataset's identity.
#[test]
fn the_spec_digest_separates_every_structural_knob() {
    let base = Recipe::seeded(64, 6, 71).unwrap();
    let recipes = [
        base,
        base.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Blob,
            separation: 2.0,
        })
        .unwrap(),
        base.with_task(Task::Multiclass {
            classes: 4,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Blob,
            separation: 2.0,
        })
        .unwrap(),
        base.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Imbalanced { ratio: 2.0 },
            geometry: ClassGeometry::Blob,
            separation: 2.0,
        })
        .unwrap(),
        base.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Hierarchical,
            separation: 2.0,
        })
        .unwrap(),
        base.with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Blob,
            separation: 2.5,
        })
        .unwrap(),
        base.with_task(Task::Clustered {
            blobs: 3,
            spread: 0.1,
        })
        .unwrap(),
        base.with_task(Task::Clustered {
            blobs: 4,
            spread: 0.1,
        })
        .unwrap(),
        base.with_task(Task::Clustered {
            blobs: 3,
            spread: 0.2,
        })
        .unwrap(),
        base.with_task(Task::TimeOrdered {
            informative: 2,
            coefficient_scale: 1.0,
            drift: 0.0,
            intercept: 0.0,
            noise_scale: 0.1,
        })
        .unwrap(),
        base.with_task(Task::TimeOrdered {
            informative: 2,
            coefficient_scale: 1.0,
            drift: 0.5,
            intercept: 0.0,
            noise_scale: 0.1,
        })
        .unwrap(),
        base.with_task(Task::Ranking {
            queries: 16,
            docs_per_query: 4,
            grades: 2,
            informative: 2,
            coefficient_scale: 1.0,
        })
        .unwrap(),
        base.with_task(Task::Ranking {
            queries: 16,
            docs_per_query: 4,
            grades: 3,
            informative: 2,
            coefficient_scale: 1.0,
        })
        .unwrap(),
        base.with_task(Task::Ranking {
            queries: 8,
            docs_per_query: 8,
            grades: 2,
            informative: 2,
            coefficient_scale: 1.0,
        })
        .unwrap(),
        base.with_groups(GroupPattern::RoundRobin { groups: 4 })
            .unwrap(),
        base.with_groups(GroupPattern::Contiguous { groups: 4 })
            .unwrap(),
        base.with_groups(GroupPattern::RoundRobin { groups: 5 })
            .unwrap(),
        base.with_groups(GroupPattern::Unbalanced {
            groups: 4,
            ratio: 1.0,
        })
        .unwrap(),
        base.with_groups(GroupPattern::Unbalanced {
            groups: 4,
            ratio: 2.0,
        })
        .unwrap(),
    ];
    let digests: Vec<[u8; 32]> = recipes.iter().map(Recipe::spec_digest).collect();
    for (index, digest) in digests.iter().enumerate() {
        assert!(
            !digests[index + 1..].contains(digest),
            "recipe {index} shares a digest with a later one: {:?}",
            recipes[index]
        );
    }
    // A group pattern of one group is still a grouping, and differs from none.
    assert_ne!(
        base.spec_digest(),
        base.with_groups(GroupPattern::RoundRobin { groups: 1 })
            .unwrap()
            .spec_digest()
    );
}

/// Regenerating a structural recipe reproduces its bytes, including the arrays
/// that are not the design.
#[test]
fn regenerating_a_structural_recipe_reproduces_its_bytes() {
    let recipes = [
        Recipe::seeded(96, 5, 81)
            .unwrap()
            .with_task(Task::Multiclass {
                classes: 4,
                balance: ClassBalance::Imbalanced { ratio: 2.0 },
                geometry: ClassGeometry::Hierarchical,
                separation: 2.0,
            })
            .unwrap()
            .with_weights(WeightPattern::ClassBalanced)
            .unwrap(),
        Recipe::seeded(96, 5, 81)
            .unwrap()
            .with_task(Task::Clustered {
                blobs: 6,
                spread: 0.15,
            })
            .unwrap(),
        Recipe::seeded(96, 5, 81)
            .unwrap()
            .with_task(Task::TimeOrdered {
                informative: 3,
                coefficient_scale: 1.5,
                drift: 0.8,
                intercept: -0.25,
                noise_scale: 0.05,
            })
            .unwrap()
            .with_groups(GroupPattern::Contiguous { groups: 8 })
            .unwrap(),
        Recipe::seeded(96, 5, 81)
            .unwrap()
            .with_task(Task::Ranking {
                queries: 24,
                docs_per_query: 4,
                grades: 3,
                informative: 3,
                coefficient_scale: 1.0,
            })
            .unwrap(),
    ];
    for recipe in recipes {
        let first = recipe.generate();
        let second = recipe.generate();
        assert_eq!(first, second, "{recipe:?} is not a function of its recipe");
        assert_eq!(first.spec_digest(), recipe.spec_digest());
        assert_eq!(
            first.features().as_slice(),
            recipe.design().as_slice(),
            "{recipe:?} generates a different design from the one it reshapes"
        );
    }
}

/// The truth accessors answer only for the families that know the answer.
///
/// The `None`s matter as much as the values. A caller asking a clustered dataset
/// for coefficients must get "there are none" rather than an empty vector, and a
/// drifting family must decline to answer
/// [`Truth::coefficients`](super::Truth::coefficients) rather than pick one of
/// its two.
#[test]
fn structural_truth_accessors_report_only_what_the_family_knows() {
    let multiclass = Recipe::seeded(48, 4, 91)
        .unwrap()
        .with_task(Task::Multiclass {
            classes: 3,
            balance: ClassBalance::Balanced,
            geometry: ClassGeometry::Blob,
            separation: 2.0,
        })
        .unwrap()
        .generate();
    let truth = multiclass.truth();
    assert!(truth.class_probabilities().is_some());
    assert_eq!(truth.classes(), Some(3));
    // The binary scalar accessor stays silent: a one-wide row is not what it
    // means, and answering would let one code path index the other's layout.
    assert!(truth.probabilities().is_none());
    assert!(truth.coefficients().is_none());
    assert!(truth.conditional_mean().is_none());
    assert!(truth.cluster_assignments().is_none());
    assert!(truth.times().is_none());
    assert!(truth.utilities().is_none());

    let clustered = Recipe::seeded(48, 4, 91)
        .unwrap()
        .with_task(Task::Clustered {
            blobs: 4,
            spread: 0.1,
        })
        .unwrap()
        .generate();
    let truth = clustered.truth();
    assert!(truth.cluster_assignments().is_some());
    assert!(truth.cluster_centres().is_some());
    assert_eq!(truth.blobs(), Some(4));
    assert!(truth.coefficients().is_none());
    assert!(truth.intercept().is_none());
    assert!(truth.class_probabilities().is_none());

    let timed = Recipe::seeded(48, 4, 91)
        .unwrap()
        .with_task(Task::TimeOrdered {
            informative: 2,
            coefficient_scale: 1.0,
            drift: 0.5,
            intercept: 0.25,
            noise_scale: 0.05,
        })
        .unwrap()
        .generate();
    let truth = timed.truth();
    assert!(truth.start_coefficients().is_some());
    assert!(truth.end_coefficients().is_some());
    assert_ne!(truth.start_coefficients(), truth.end_coefficients());
    assert_eq!(truth.intercept(), Some(0.25));
    assert!(truth.conditional_mean().is_some());
    assert!(truth.times().is_some());
    // Two vectors and no single one: answering would be a wrong answer, not a
    // partial one.
    assert!(truth.coefficients().is_none());

    let ranked = Recipe::seeded(48, 4, 91)
        .unwrap()
        .with_task(Task::Ranking {
            queries: 12,
            docs_per_query: 4,
            grades: 2,
            informative: 2,
            coefficient_scale: 1.0,
        })
        .unwrap()
        .generate();
    let truth = ranked.truth();
    assert!(truth.coefficients().is_some());
    assert!(truth.utilities().is_some());
    assert_eq!(truth.grades(), Some(2));
    assert!(truth.start_coefficients().is_none());
    assert!(truth.conditional_mean().is_none());

    // Only the ranking family draws pairs; everything else says so.
    assert!(ranked.pairs().is_some());
    for dataset in [&multiclass, &clustered, &timed] {
        assert!(dataset.pairs().is_none());
        assert!(dataset.groups().is_none());
    }
}

/// A design carrying a structural family is still an ordinary matrix.
///
/// The generated design goes to `DenseMatrix::new` on the way out, so this
/// asserts the property that check exists to preserve: every value is finite,
/// including the ones a cluster shift or a column duplication produced.
#[test]
fn every_structural_family_generates_a_finite_design() {
    let recipes = [
        Recipe::seeded(64, 5, 101)
            .unwrap()
            .with_task(Task::Multiclass {
                classes: 4,
                balance: ClassBalance::Imbalanced { ratio: 8.0 },
                geometry: ClassGeometry::Hierarchical,
                separation: 12.0,
            })
            .unwrap(),
        Recipe::seeded(64, 5, 101)
            .unwrap()
            .with_task(Task::Clustered {
                blobs: 8,
                spread: 4.0,
            })
            .unwrap(),
        Recipe::seeded(64, 5, 101)
            .unwrap()
            .with_task(Task::TimeOrdered {
                informative: 5,
                coefficient_scale: 1e3,
                drift: 1e3,
                intercept: 0.0,
                noise_scale: 10.0,
            })
            .unwrap(),
        Recipe::seeded(64, 5, 101)
            .unwrap()
            .with_task(Task::Ranking {
                queries: 32,
                docs_per_query: 2,
                grades: 2,
                informative: 5,
                coefficient_scale: 1e3,
            })
            .unwrap(),
    ];
    for recipe in recipes {
        let dataset = recipe.generate();
        let design: &DenseMatrix = dataset.features();
        assert!(design.as_slice().iter().all(|value| value.is_finite()));
        if let Some(Target::Regression(targets)) = dataset.target() {
            assert!(targets.as_slice().iter().all(|value| value.is_finite()));
        }
    }
}
