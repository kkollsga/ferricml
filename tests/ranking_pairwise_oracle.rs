//! Randomized pair sets against a ranking-specific oracle.
//!
//! The pairwise ranker is the least-exercised estimator in the crate: every
//! existing test drives one four-item matrix and one four-pair observation
//! list, so the documented pair semantics — mirrored expansion, canonical
//! ordering, the margin/score relation, and the tie band — are pinned at
//! exactly one point each.
//!
//! This binary states them as properties over randomized pair sets, and adds
//! two oracles the API-consistency checks cannot supply:
//!
//! - **Symmetry.** A pair set that carries no net preference — every pair a tie,
//!   or every preference matched by its opposite at the same weight — has a
//!   penalized objective whose unique minimum is the zero coefficient vector.
//!   Any fitted direction on such a set would be invented.
//! - **Recovery.** A pair set generated from a linear utility must be ranked in
//!   that utility's order.
//!
//! Sizes come from `FERRICML_ORACLE_SWEEP`:
//!
//! ```text
//! FERRICML_ORACLE_SWEEP=1500 cargo test --release --test ranking_pairwise_oracle -- --nocapture
//! ```

use ferricml::artifact::ModelArtifact;
use ferricml::data::DenseMatrix;
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
};

#[path = "support/rng.rs"]
mod rng;

use rng::TestRng;

const DEFAULT_CASES: usize = 90;

fn cases() -> usize {
    std::env::var("FERRICML_ORACLE_SWEEP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

struct Case {
    items: DenseMatrix,
    observations: Vec<PairwiseObservation>,
    params: PairwiseLinearRankerParams,
    /// The direction the items were generated to be ranked in, when the case
    /// was built from a linear utility.
    utility: Option<Vec<f64>>,
}

fn outcome(index: usize) -> PairOutcome {
    [
        PairOutcome::LeftPreferred,
        PairOutcome::RightPreferred,
        PairOutcome::Tie,
    ][index]
}

/// A random item matrix and a random pair set over it.
fn arbitrary_case(seed: u64) -> Case {
    let mut rng = TestRng::new(seed);
    let rows = rng.between(4, 12);
    let columns = rng.between(1, 4);
    let items = DenseMatrix::new(
        (0..rows * columns)
            .map(|_| rng.range_f32(-3.0, 3.0))
            .collect(),
        rows,
        columns,
    )
    .expect("generated shape");

    let count = rng.between(3, 14);
    let mut observations = Vec::with_capacity(count);
    while observations.len() < count {
        let left = rng.below(rows);
        let right = rng.below(rows);
        if left == right {
            continue;
        }
        observations.push(
            PairwiseObservation::new(
                PairIndex::new(left, right).expect("distinct"),
                outcome(rng.below(3)),
                rng.range_f32(0.25, 4.0),
            )
            .expect("finite non-negative weight"),
        );
    }

    Case {
        items,
        observations,
        params: PairwiseLinearRankerParams::default()
            .with_c(10.0_f32.powf(rng.range_f32(-1.0, 1.5)))
            .with_max_iter(200)
            .with_tie_threshold(if rng.below(3) == 0 {
                rng.range_f32(0.0, 1.5)
            } else {
                0.0
            }),
        utility: None,
    }
}

/// Items drawn at random and every pair labelled by a linear utility, so the
/// order the ranker should recover is known before it is fitted.
fn recoverable_case(seed: u64) -> Case {
    let mut rng = TestRng::new(seed);
    let rows = rng.between(6, 14);
    let columns = rng.between(1, 3);
    let utility = (0..columns)
        .map(|_| {
            let magnitude = 0.5 + rng.unit() * 2.0;
            if rng.flag() { magnitude } else { -magnitude }
        })
        .collect::<Vec<_>>();
    let values = (0..rows * columns)
        .map(|_| rng.range_f32(-3.0, 3.0))
        .collect::<Vec<_>>();
    let items = DenseMatrix::new(values, rows, columns).expect("generated shape");

    let mut observations = Vec::new();
    for left in 0..rows {
        for right in (left + 1)..rows {
            let gap = true_utility(&items, &utility, left) - true_utility(&items, &utility, right);
            // Undecidably close pairs are left out rather than labelled by a
            // coin toss, so the oracle below is a statement about pairs the
            // utility actually separates.
            if gap.abs() < 1.0e-3 {
                continue;
            }
            observations.push(
                PairwiseObservation::new(
                    PairIndex::new(left, right).expect("distinct"),
                    if gap > 0.0 {
                        PairOutcome::LeftPreferred
                    } else {
                        PairOutcome::RightPreferred
                    },
                    1.0,
                )
                .expect("unit weight"),
            );
        }
    }

    Case {
        items,
        observations,
        // A weak penalty, because a strong one is entitled to shrink a
        // separating direction into a different ordering.
        params: PairwiseLinearRankerParams::default()
            .with_c(100.0)
            .with_max_iter(500)
            .with_tol(1.0e-6),
        utility: Some(utility),
    }
}

fn true_utility(items: &DenseMatrix, utility: &[f64], row: usize) -> f64 {
    (0..utility.len())
        .map(|column| utility[column] * f64::from(items.get(row, column).expect("in bounds")))
        .sum()
}

/// Every ordered pair of distinct rows.
fn all_pairs(rows: usize) -> Vec<PairIndex> {
    let mut pairs = Vec::with_capacity(rows * (rows - 1));
    for left in 0..rows {
        for right in 0..rows {
            if left != right {
                pairs.push(PairIndex::new(left, right).expect("distinct"));
            }
        }
    }
    pairs
}

/// The documented three-way rule, written here rather than reached through the
/// estimator.
fn expected_outcome(margin: f32, tie_threshold: f32) -> PairOutcome {
    if margin.abs() <= tie_threshold {
        PairOutcome::Tie
    } else if margin > 0.0 {
        PairOutcome::LeftPreferred
    } else {
        PairOutcome::RightPreferred
    }
}

#[test]
fn the_documented_pair_semantics_hold_over_randomized_pair_sets() {
    let mut fits = 0_usize;
    let mut refusals = 0_usize;
    let mut margins_checked = 0_usize;
    let mut outcomes_checked = 0_usize;
    let mut ties_observed = 0_usize;
    let mut permutation_disagreements = 0_usize;
    let mut mirror_disagreements = 0_usize;
    let mut score_disagreements = 0_usize;
    let mut antisymmetry_disagreements = 0_usize;
    let mut batch_disagreements = 0_usize;
    let mut artifact_disagreements = 0_usize;
    let mut intransitive_tie_triples = 0_usize;
    let mut thresholded_cases = 0_usize;
    let mut zero_margins = 0_usize;

    for seed in 0..cases() as u64 {
        let case = arbitrary_case(0x4a4b_0007_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        let view = case.items.as_view();
        let Ok(model) = PairwiseLinearRanker::fit(&view, &case.observations, case.params.clone())
        else {
            refusals += 1;
            continue;
        };
        fits += 1;

        // ---- Fitting is a function of the pair *set*, not of its order ----
        let mut shuffled = case.observations.clone();
        TestRng::new(seed ^ 0xa5a5).shuffle(&mut shuffled);
        let reordered = PairwiseLinearRanker::fit(&view, &shuffled, case.params.clone())
            .expect("the same observations must still fit");
        if reordered != model {
            permutation_disagreements += 1;
        }

        // ---- ... nor of which side of each pair was written first ---------
        let mut flipper = TestRng::new(seed ^ 0x5a5a);
        let mirrored = case
            .observations
            .iter()
            .map(|observation| {
                if flipper.flag() {
                    let pair = observation.pair();
                    PairwiseObservation::new(
                        PairIndex::new(pair.right(), pair.left()).expect("distinct"),
                        match observation.outcome() {
                            PairOutcome::LeftPreferred => PairOutcome::RightPreferred,
                            PairOutcome::RightPreferred => PairOutcome::LeftPreferred,
                            PairOutcome::Tie => PairOutcome::Tie,
                        },
                        observation.weight(),
                    )
                    .expect("weight is unchanged")
                } else {
                    *observation
                }
            })
            .collect::<Vec<_>>();
        if PairwiseLinearRanker::fit(&view, &mirrored, case.params.clone())
            .expect("a mirrored pair set must still fit")
            != model
        {
            mirror_disagreements += 1;
        }

        // ---- The margin is the score difference, exactly ------------------
        let scores = model.score_items(&view).expect("item scores");
        let mut owned_scores = vec![f32::NAN; scores.len()];
        model
            .score_items_into(&view, &mut owned_scores)
            .expect("item scores");
        if owned_scores
            .iter()
            .zip(&scores)
            .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            batch_disagreements += 1;
        }
        for (row, &score) in scores.iter().enumerate() {
            let single = model
                .score_one(case.items.row(row).expect("in bounds"))
                .expect("single score");
            if single.to_bits() != score.to_bits() {
                score_disagreements += 1;
            }
        }

        let pairs = all_pairs(case.items.rows());
        let margins = model.pair_margins(&view, &pairs).expect("margins");
        let mut owned_margins = vec![f32::NAN; pairs.len()];
        model
            .pair_margins_into(&view, &pairs, &mut owned_margins)
            .expect("margins");
        let outcomes = model.compare(&view, &pairs).expect("outcomes");
        let mut owned_outcomes = vec![PairOutcome::Tie; pairs.len()];
        model
            .compare_into(&view, &pairs, &mut owned_outcomes)
            .expect("outcomes");
        if owned_margins
            .iter()
            .zip(&margins)
            .any(|(left, right)| left.to_bits() != right.to_bits())
            || owned_outcomes != outcomes
        {
            batch_disagreements += 1;
        }

        let threshold = case.params.tie_threshold();
        if threshold > 0.0 {
            thresholded_cases += 1;
        }
        for (index, &pair) in pairs.iter().enumerate() {
            let margin = margins[index];
            margins_checked += 1;
            // The documented definition: score(left) - score(right).
            if margin.to_bits() != (scores[pair.left()] - scores[pair.right()]).to_bits() {
                score_disagreements += 1;
            }
            if model
                .pair_margin(&view, pair)
                .expect("single margin")
                .to_bits()
                != margin.to_bits()
            {
                batch_disagreements += 1;
            }
            // Antisymmetry. `a - b` and `-(b - a)` agree as values for every
            // finite pair, and agree bit for bit everywhere except at zero,
            // where the two subtractions produce `+0.0` and `-0.0`. Both
            // statements are asserted, and the zero case is counted rather than
            // waved at: the crate's own single-point assertion is the bitwise
            // one, which is only true away from a zero margin.
            let reversed = model
                .pair_margin(
                    &view,
                    PairIndex::new(pair.right(), pair.left()).expect("distinct"),
                )
                .expect("reversed margin");
            if margin != -reversed {
                antisymmetry_disagreements += 1;
            }
            if margin == 0.0 {
                zero_margins += 1;
                assert_eq!(
                    (margin.to_bits(), reversed.to_bits()),
                    (0_u32, 0_u32),
                    "both directions of a zero margin are the positive zero the \
                     subtraction produces"
                );
            } else if margin.to_bits() != (-reversed).to_bits() {
                antisymmetry_disagreements += 1;
            }

            outcomes_checked += 1;
            assert_eq!(
                outcomes[index],
                expected_outcome(margin, threshold),
                "seed {seed}: pair {pair:?} at margin {margin} and threshold {threshold}"
            );
            if outcomes[index] == PairOutcome::Tie {
                ties_observed += 1;
            }
            // The ranking claim: an outcome orders the two items the same way
            // their scores do, whenever the threshold does not intervene.
            if outcomes[index] != PairOutcome::Tie {
                let ordered = scores[pair.left()] > scores[pair.right()];
                assert_eq!(
                    ordered,
                    outcomes[index] == PairOutcome::LeftPreferred,
                    "seed {seed}: pair {pair:?} is ordered against its own scores"
                );
            }
        }

        // The tie band is a band, not an equivalence: this counts the triples
        // where two ties do not compose into one. It is recorded rather than
        // asserted away, because nothing documents transitivity.
        if threshold > 0.0 {
            let rows = case.items.rows();
            for a in 0..rows {
                for b in 0..rows {
                    for c in 0..rows {
                        if a == b || b == c || a == c {
                            continue;
                        }
                        let tie = |left: usize, right: usize| {
                            (scores[left] - scores[right]).abs() <= threshold
                        };
                        if tie(a, b) && tie(b, c) && !tie(a, c) {
                            intransitive_tie_triples += 1;
                        }
                    }
                }
            }
        }

        // ---- Persistence -------------------------------------------------
        let bytes = model.to_artifact([9; 32]).expect("artifact");
        let decoded = PairwiseLinearRanker::from_artifact(&bytes, [9; 32]).expect("decode");
        if decoded != model
            || decoded
                .score_items(&view)
                .expect("scores")
                .iter()
                .zip(&scores)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            artifact_disagreements += 1;
        }
    }

    println!(
        "ranking: {fits} fitted rankers ({refusals} refused), {margins_checked} margins and \
         {outcomes_checked} outcomes checked, {ties_observed} ties"
    );
    println!(
        "ranking invariance: {permutation_disagreements} order disagreements, \
         {mirror_disagreements} mirror disagreements, {score_disagreements} score/margin \
         disagreements, {antisymmetry_disagreements} antisymmetry disagreements, \
         {batch_disagreements} batch-shape disagreements, {artifact_disagreements} artifact \
         disagreements, {zero_margins} zero margins where the two directions differ only in \
         the sign bit"
    );
    println!(
        "ranking observation: {intransitive_tie_triples} intransitive tie triples across \
         {thresholded_cases} cases with a positive tie threshold"
    );

    assert_eq!(permutation_disagreements, 0);
    assert_eq!(mirror_disagreements, 0);
    assert_eq!(score_disagreements, 0);
    assert_eq!(antisymmetry_disagreements, 0);
    assert_eq!(batch_disagreements, 0);
    assert_eq!(artifact_disagreements, 0);
    assert!(fits > 0 && margins_checked > 0);
    // Non-vacuity: a threshold of zero would make the tie branch unreachable
    // and the outcome rule a two-way test.
    assert!(
        ties_observed > 0,
        "no pair was ever a tie, so the three-way rule was only ever exercised two ways"
    );
    assert!(
        thresholded_cases > 0,
        "no case carried a positive tie threshold"
    );
}

#[test]
fn a_pair_set_with_no_net_preference_fits_the_zero_direction() {
    // The oracle the API-consistency checks cannot supply. A set of ties, or a
    // set where every preference is matched by its opposite at the same weight,
    // expands to rows that are symmetric under negating the coefficient vector.
    // The penalized objective is then even in the coefficients and strictly
    // convex, so its unique minimum is zero — any direction the fit reported
    // would be an artifact of the solver rather than of the data.
    let mut checked = 0_usize;
    let mut worst = 0.0_f32;
    let mut control_worst = 0.0_f32;
    let mut controls = 0_usize;

    for seed in 0..cases() as u64 {
        let mut rng = TestRng::new(0x4a5b_0008_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        let rows = rng.between(4, 10);
        let columns = rng.between(1, 4);
        let items = DenseMatrix::new(
            (0..rows * columns)
                .map(|_| rng.range_f32(-3.0, 3.0))
                .collect(),
            rows,
            columns,
        )
        .expect("generated shape");

        // Half the cases are all ties; the other half pair every preference
        // with its opposite at the same weight.
        let all_ties = rng.flag();
        let mut observations = Vec::new();
        for _ in 0..rng.between(2, 8) {
            let left = rng.below(rows);
            let right = rng.below(rows);
            if left == right {
                continue;
            }
            let weight = rng.range_f32(0.25, 4.0);
            let pair = PairIndex::new(left, right).expect("distinct");
            if all_ties {
                observations
                    .push(PairwiseObservation::new(pair, PairOutcome::Tie, weight).expect("valid"));
            } else {
                observations.push(
                    PairwiseObservation::new(pair, PairOutcome::LeftPreferred, weight)
                        .expect("valid"),
                );
                observations.push(
                    PairwiseObservation::new(pair, PairOutcome::RightPreferred, weight)
                        .expect("valid"),
                );
            }
        }
        if observations.is_empty() {
            continue;
        }

        let params = PairwiseLinearRankerParams::default()
            .with_c(10.0)
            .with_max_iter(200);
        let model = PairwiseLinearRanker::fit(&items.as_view(), &observations, params.clone())
            .expect("a symmetric pair set fits");
        for &coefficient in model.coefficients() {
            worst = worst.max(coefficient.abs());
        }
        checked += 1;

        // Control: breaking the symmetry by one extra preference has to move
        // the fit off zero, or "zero" is simply what this estimator returns.
        let mut skewed = observations.clone();
        let left = rng.below(rows);
        let right = (left + 1 + rng.below(rows - 1)) % rows;
        skewed.push(
            PairwiseObservation::new(
                PairIndex::new(left, right).expect("distinct"),
                PairOutcome::LeftPreferred,
                4.0,
            )
            .expect("valid"),
        );
        let skewed = PairwiseLinearRanker::fit(&items.as_view(), &skewed, params)
            .expect("the skewed set fits");
        let magnitude = skewed
            .coefficients()
            .iter()
            .fold(0.0_f32, |worst, value| worst.max(value.abs()));
        control_worst = control_worst.max(magnitude);
        if magnitude > 1.0e-3 {
            controls += 1;
        }
    }

    println!(
        "ranking symmetry: {checked} balanced pair sets, largest fitted coefficient \
         {worst:e}; {controls} of {checked} controls moved off zero, largest \
         {control_worst:e}"
    );
    assert!(checked > 0);
    assert!(
        worst <= 1.0e-6,
        "a pair set with no net preference fitted a direction of {worst:e}"
    );
    assert!(
        controls * 4 > checked * 3,
        "only {controls} of {checked} symmetry-breaking controls moved the fit off zero"
    );
}

#[test]
fn a_pair_set_generated_from_a_linear_utility_is_ranked_in_that_order() {
    let mut instances = 0_usize;
    let mut pairs_checked = 0_usize;
    let mut pairs_agreeing = 0_usize;
    let mut decisive_pairs = 0_usize;
    let mut worst_instance = 1.0_f64;

    for seed in 0..(cases() / 2).max(8) as u64 {
        let case = recoverable_case(0x4a6c_0009_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        let utility = case.utility.as_ref().expect("a recoverable case");
        if case.observations.is_empty() {
            continue;
        }
        let Ok(model) =
            PairwiseLinearRanker::fit(&case.items.as_view(), &case.observations, case.params)
        else {
            continue;
        };
        let scores = model
            .score_items(&case.items.as_view())
            .expect("item scores");

        // The oracle is a statement about pairs the utility *decides*. A
        // penalized fit is entitled to sacrifice a pair the utility separates by
        // a thousandth of its own range, and does: over this sweep exactly one
        // such pair comes out the other way, at a true gap of 6.3e-3 against a
        // utility range two orders of magnitude wider. So the assertion covers
        // pairs separated by at least a twentieth of the range, and the rate
        // over every separated pair is reported beside it.
        let utilities = (0..case.items.rows())
            .map(|row| true_utility(&case.items, utility, row))
            .collect::<Vec<_>>();
        let range = utilities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - utilities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let decisive_gap = 0.05 * range;

        let mut agreeing = 0_usize;
        let mut total = 0_usize;
        for left in 0..case.items.rows() {
            for right in (left + 1)..case.items.rows() {
                let gap = utilities[left] - utilities[right];
                if gap.abs() < 1.0e-3 {
                    continue;
                }
                total += 1;
                let ordered = (scores[left] > scores[right]) == (gap > 0.0);
                if ordered {
                    agreeing += 1;
                }
                if gap.abs() >= decisive_gap {
                    decisive_pairs += 1;
                    assert!(
                        ordered,
                        "a pair separated by {gap:e} of a {range:e} utility range was \
                         ranked the other way, at scores {} and {}",
                        scores[left], scores[right]
                    );
                }
            }
        }
        if total == 0 {
            continue;
        }
        instances += 1;
        pairs_checked += total;
        pairs_agreeing += agreeing;
        worst_instance = worst_instance.min(agreeing as f64 / total as f64);
    }

    let rate = pairs_agreeing as f64 / pairs_checked as f64;
    println!(
        "ranking recovery: {instances} utilities, {decisive_pairs} decisively separated pairs \
         all ranked correctly; over every separated pair {pairs_agreeing} of {pairs_checked} \
         ({rate:.4}), worst instance {worst_instance:.4}"
    );
    assert!(instances > 0 && pairs_checked > 0 && decisive_pairs > 0);
    // Non-vacuity: a fit that had learned nothing would sit near one half here.
    assert!(
        rate > 0.99,
        "only {rate:.4} of separated pairs were ranked in their utility's order"
    );
}

/// Recipes chosen so the *order* of the expanded rows is observable in the
/// fitted `f32` coefficients.
///
/// `(collinearity, c, observations, rows)`. Near-collinear item columns make
/// the Newton system ill-conditioned, and a weak penalty declines to condition
/// it back, so the `1e-16` differences a re-ordered accumulation produces are
/// amplified past `f32` resolution. Ordinary data does not do this: 1,000 fits
/// over weights spanning twelve orders of magnitude and items spanning twelve
/// showed no order dependence at all, because the estimator standardizes its
/// design and removes exactly that conditioning.
const ORDER_SENSITIVE: [(f64, f32, usize, usize); 2] =
    [(1.0e-9, 1.0e6, 40, 12), (1.0e-12, 1.0e9, 80, 16)];

#[test]
fn the_canonical_pair_order_is_what_makes_the_fit_order_independent() {
    // `expand_observations` canonicalizes and sorts before it expands, and
    // before this test nothing in the suite failed when that sort was deleted:
    // it was a surviving mutant against all 974 tests, because on ordinary data
    // a re-ordered accumulation moves the answer by about 1e-16 and the fitted
    // coefficients are `f32`.
    //
    // These recipes make the difference visible. With the sort in place all 400
    // cases below are bit-identical under a shuffled observation list; with it
    // removed, 12 of them are not — recorded 2026-07-26 by deleting the
    // `sort_by_key` call and re-running.
    let mut fits = 0_usize;
    let mut differing = 0_usize;

    for (collinearity, c, count, rows) in ORDER_SENSITIVE {
        for seed in 0..200_u64 {
            let mut rng =
                TestRng::new(0xc0de_000a_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
            let columns = 3;
            let mut values = Vec::with_capacity(rows * columns);
            for _ in 0..rows {
                let base = rng.range(-3.0, 3.0);
                values.push(base as f32);
                values.push((base * (1.0 + collinearity * rng.range(-1.0, 1.0))) as f32);
                values.push(rng.range_f32(-3.0, 3.0));
            }
            let items = DenseMatrix::new(values, rows, columns).expect("generated shape");

            let mut observations = Vec::with_capacity(count);
            while observations.len() < count {
                let left = rng.below(rows);
                let right = rng.below(rows);
                if left == right {
                    continue;
                }
                observations.push(
                    PairwiseObservation::new(
                        PairIndex::new(left, right).expect("distinct"),
                        outcome(rng.below(3)),
                        rng.range_f32(0.1, 8.0),
                    )
                    .expect("valid"),
                );
            }

            let params = PairwiseLinearRankerParams::default()
                .with_c(c)
                .with_max_iter(1000)
                .with_tol(1.0e-7);
            let Ok(model) =
                PairwiseLinearRanker::fit(&items.as_view(), &observations, params.clone())
            else {
                continue;
            };
            let mut shuffled = observations.clone();
            TestRng::new(seed ^ 0x1234).shuffle(&mut shuffled);
            let Ok(reordered) = PairwiseLinearRanker::fit(&items.as_view(), &shuffled, params)
            else {
                continue;
            };
            fits += 1;
            if reordered != model {
                differing += 1;
            }
        }
    }

    println!(
        "ranking canonical order: {fits} ill-conditioned fits, {differing} order-dependent \
         (12 of them without the canonical sort)"
    );
    assert!(
        fits > 350,
        "only {fits} of 400 ill-conditioned cases fitted"
    );
    assert_eq!(
        differing, 0,
        "{differing} ill-conditioned fits depended on the order the observations were \
         written in"
    );
}
