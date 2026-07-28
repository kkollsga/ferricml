//! Evidence for the regression and binary-classification families.
//!
//! # Two kinds of evidence, because there are two determinism envelopes
//!
//! A [`Portability::BitExact`] family is pinned by literal values, exactly as
//! the absorbed lanes are: every operation on its path is exact or correctly
//! rounded, so a moved value is a defect and never a platform difference.
//!
//! A [`Portability::PerRunner`] family cannot be, and pinning it anyway would be
//! a promise this crate cannot keep across a libm it does not control. Those are
//! held to *properties* instead — a recovered coefficient, a realized
//! prevalence, a realized label-noise rate, a realized condition number — with
//! every tolerance derived from the sampling law of the quantity it bounds
//! rather than from what happened to pass.
//!
//! # Where the binomial tolerances come from
//!
//! A realized prevalence and a realized label-noise rate are both means of `n`
//! independent Bernoulli draws, so their standard deviation is
//! `sqrt(p (1 - p) / n)` exactly. Every such assertion below allows four of
//! those, which a correct generator exceeds with probability about `6e-5` per
//! case. The sweeps here are deterministic — fixed seeds, no re-roll — and their
//! measured worst cases are `2.24` deviations over 200 prevalence cases and
//! `2.53` over 160 label-noise cases, so the bound is roughly `1.6` times the
//! observed extreme: loose enough that a different libm's logistic cannot reach
//! it, tight enough that a generator whose realized rate was even one percent
//! off in absolute terms would fail every case at once.

use super::*;
use crate::linear_model::{LinearRegression, LinearRegressionParams};
use crate::numeric::OwnedRng;
use faer::Mat;

/// Four binomial standard deviations of a rate `p` over `n` draws.
fn binomial_tolerance(rate: f64, draws: usize) -> f64 {
    4.0 * (rate * (1.0 - rate) / draws as f64).sqrt()
}

fn regression_values(dataset: &Dataset) -> &[f32] {
    match dataset.target() {
        Some(Target::Regression(targets)) => targets.as_slice(),
        other => panic!("expected a regression target, got {other:?}"),
    }
}

fn binary_labels(dataset: &Dataset) -> &[u8] {
    match dataset.target() {
        Some(Target::Binary(targets)) => targets.as_slice(),
        other => panic!("expected binary labels, got {other:?}"),
    }
}

/// Fits the crate's own least-squares path to a generated regression dataset.
fn fit_least_squares(dataset: &Dataset) -> LinearRegression {
    let targets = match dataset.target() {
        Some(Target::Regression(targets)) => targets,
        other => panic!("expected a regression target, got {other:?}"),
    };
    LinearRegression::fit(
        &dataset.features().as_view(),
        targets,
        LinearRegressionParams::default(),
    )
    .expect("a generated design fits")
}

/// The ratio of the largest to the smallest singular value of a design.
///
/// Measured with the same decomposition the crate's own least-squares path
/// uses, so "the condition number the family realized" and "the condition
/// number the solver sees" are the same number rather than two estimates.
fn realized_condition_number(dataset: &Dataset) -> f64 {
    let design = dataset.features();
    let matrix = Mat::from_fn(design.rows(), design.columns(), |row, column| {
        f64::from(design.get(row, column).expect("in-bounds cell"))
    });
    let svd = matrix.thin_svd().expect("a finite design decomposes");
    let singular = svd.S();
    let singular = singular.column_vector();
    let largest = singular.iter().copied().fold(0.0_f64, f64::max);
    let smallest = singular.iter().copied().fold(f64::INFINITY, f64::min);
    largest / smallest
}

/// The mean of a slice, accumulated in `f64`.
fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64
}

/// The bit-exact families emit exactly the values they emitted when they landed.
///
/// These are `assert_eq!` on `f32` rather than a tolerance, and they are allowed
/// to be: [`Task::LinearRegression`] and the interaction and piecewise nonlinear
/// shapes evaluate no transcendental at all, so every value on their path is
/// fixed by IEEE-754 rather than by a libm. A moved literal here is a defect in
/// this crate, never a platform difference — which is exactly the claim
/// [`Portability::BitExact`] makes, asserted rather than described.
///
/// **Re-frozen once, when the task dials left `Recipe::stream_digest`.** That
/// change was the whole point of moving them — a dial had been reseeding the
/// auxiliary streams — so every drawn coefficient and every drawn noise term
/// here moved with it, deliberately and in one commit.
///
/// The re-freeze carries its own evidence that only the *auxiliary* streams
/// moved: the two nonlinear conditional means below are byte-identical to their
/// pre-change literals, because a nonlinear shape is a function of the design
/// alone and the design a `NonlinearRegression` recipe draws never passed
/// through the task's digest. Their *targets* moved, because the additive noise
/// did. A change that had disturbed the design stream would have moved both.
#[test]
fn the_bit_exact_families_emit_their_recorded_first_values() {
    let recipe = Recipe::seeded(6, 4, 11)
        .unwrap()
        .with_task(Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.25,
            noise_scale: 0.1,
        })
        .unwrap();
    assert_eq!(recipe.portability(), Portability::BitExact);
    let dataset = recipe.generate();

    // The uninformative columns are exactly zero: "this column does not matter"
    // is a statement a consumer must be able to check, not merely believe.
    assert_eq!(
        dataset.truth().coefficients().unwrap(),
        [0.31747937, -0.26101267, 0.0, 0.0]
    );
    assert_eq!(dataset.truth().intercept(), Some(0.25));
    assert_eq!(
        dataset.truth().conditional_mean().unwrap(),
        [
            -0.06952426,
            0.017645527,
            0.23827066,
            0.06384924,
            0.14391439,
            0.5539454,
        ]
    );
    assert_eq!(
        regression_values(&dataset),
        [
            -0.14038074,
            0.009581805,
            0.22630431,
            0.024172205,
            0.082978025,
            0.5472678,
        ]
    );

    let heads: [(NonlinearKind, [f32; 6], [f32; 6]); 2] = [
        (
            NonlinearKind::Interaction,
            [
                1.4157436, 2.0203612, 1.654461, 1.0635431, -0.7834641, 0.882303,
            ],
            [
                1.394437, 1.9905711, 1.6079082, 0.98611003, -0.6835806, 0.7957359,
            ],
        ),
        (
            NonlinearKind::Piecewise,
            [
                -2.345626, -2.1884937, 0.8635033, -1.0202256, 1.2385874, -1.6798728,
            ],
            [
                -2.4009402, -2.115744, 0.7740759, -1.0105724, 1.2599751, -1.6995444,
            ],
        ),
    ];
    for (kind, conditional_mean, targets) in heads {
        let recipe = Recipe::seeded(6, 5, 11)
            .unwrap()
            .with_task(Task::NonlinearRegression {
                kind,
                noise_scale: 0.1,
            })
            .unwrap();
        assert_eq!(recipe.portability(), Portability::BitExact, "{kind:?}");
        let dataset = recipe.generate();
        assert_eq!(
            dataset.truth().conditional_mean().unwrap(),
            conditional_mean,
            "{kind:?} conditional mean moved"
        );
        assert_eq!(
            regression_values(&dataset),
            targets,
            "{kind:?} targets moved"
        );
        // A nonlinear shape knows its conditional mean and no coefficients, and
        // the absence is a variant rather than a zero vector.
        assert_eq!(dataset.truth().coefficients(), None);
    }
}

/// The linear family's recorded coefficients are the ones a solver recovers.
///
/// This is the property the whole phase exists for: with a recorded `β`, a
/// comparison can ask which of two implementations is closer to *right* rather
/// than only where they disagree.
///
/// **The two tolerances are derived, not tuned.**
///
/// * At `noise_scale = 0` the target is an exact linear function of the design,
///   and the only error left is representational: the design values carry 24-bit
///   mantissas, the target is their linear combination rounded to `f32` once,
///   and the solve runs in `f64`. The residual is bounded by about
///   `eps_f32 * max|β| * columns`, which is `1.2e-7 * 2 * 8 ≈ 2e-6`; `1e-5` is
///   five times that bound. Measured: `1.8e-9`.
/// * At `noise_scale = 1e-3` the ordinary-least-squares standard error over a
///   uniform design on `[-1, 1)` is `σ sqrt(3 / n)`, because `E[x²] = 1/3`; at
///   `n = 2048` that is `3.8e-5`. `3e-4` is eight standard errors, which a
///   correct fit exceeds with probability below `1e-15`. Measured worst
///   coefficient error: `2.9e-5`.
///
/// Neither number was chosen by running the test. A tolerance that had been
/// would be worthless here, because the quantity it bounds is exactly the
/// quantity under test.
#[test]
fn a_linear_family_recovers_its_recorded_coefficients_at_low_noise() {
    for (noise_scale, tolerance) in [(0.0_f32, 1e-5_f64), (1e-3, 3e-4)] {
        let dataset = Recipe::seeded(2048, 8, 3)
            .unwrap()
            .with_task(Task::LinearRegression {
                informative: 5,
                coefficient_scale: 2.0,
                intercept: -0.75,
                noise_scale,
            })
            .unwrap()
            .generate();
        let fit = fit_least_squares(&dataset);
        let truth = dataset.truth().coefficients().unwrap();
        assert_eq!(fit.coefficients().len(), truth.len());
        for (column, (&recorded, &fitted)) in truth.iter().zip(fit.coefficients()).enumerate() {
            assert!(
                f64::from(recorded - fitted).abs() <= tolerance,
                "column {column} at noise {noise_scale}: recorded {recorded}, fitted {fitted}"
            );
        }
        assert!(
            f64::from(fit.intercept() - dataset.truth().intercept().unwrap()).abs() <= tolerance,
            "intercept at noise {noise_scale}: fitted {}",
            fit.intercept()
        );
        // The recorded conditional mean is the target with its noise removed, so
        // it is what a model's *achievable* error is measured against.
        let noise_free = dataset.truth().conditional_mean().unwrap();
        let residual = regression_values(&dataset)
            .iter()
            .zip(noise_free)
            .map(|(&target, &centre)| f64::from(target - centre).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            residual <= f64::from(noise_scale) + 1e-6,
            "the target departs from its recorded mean by {residual}, past the noise half-width"
        );
    }
}

/// The realized positive rate is the requested one, at every prevalence.
///
/// The intercept is solved for by bisection so the *mean Bayes probability*
/// equals the request exactly; the realized rate is then a binomial draw around
/// it, and the tolerance is four of that binomial's standard deviations. See the
/// module note for why four.
#[test]
fn the_realized_prevalence_matches_the_request_within_four_binomial_deviations() {
    const ROWS: usize = 4096;
    for prevalence in [0.02_f32, 0.05, 0.25, 0.5, 0.8] {
        for seed in [0_u64, 7, 19, 23, 31] {
            let dataset = Recipe::seeded(ROWS, 6, seed)
                .unwrap()
                .with_task(Task::LinearBinary {
                    informative: 4,
                    separation: 3.0,
                    prevalence,
                })
                .unwrap()
                .generate();

            // The mean probability is the requested prevalence to the last bits
            // bisection can reach, and that is a stronger statement than the
            // realized rate: it says the *problem* has the requested prevalence,
            // not merely that this draw did.
            let probabilities = dataset.truth().probabilities().unwrap();
            assert!(
                (mean(probabilities) - f64::from(prevalence)).abs() < 1e-6,
                "the solved intercept left mean P(y=1) at {}",
                mean(probabilities)
            );

            let labels = binary_labels(&dataset);
            let realized =
                labels.iter().filter(|&&label| label == 1).count() as f64 / labels.len() as f64;
            let tolerance = binomial_tolerance(f64::from(prevalence), ROWS);
            assert!(
                (realized - f64::from(prevalence)).abs() <= tolerance,
                "prevalence {prevalence} seed {seed}: realized {realized}, tolerance {tolerance}"
            );
        }
    }
}

/// The realized label-noise rate is the requested one, and the flips are an
/// overlay rather than a redraw.
///
/// The second half is the part that took a defect to find. When the auxiliary
/// streams were seeded from the full spec digest, switching label noise on
/// reseeded the label draw itself: a five-percent request produced a
/// fifty-six-percent difference, because the two datasets were independent draws
/// and the measured quantity was the draw. The streams are seeded from a digest
/// that excludes the contamination for exactly this reason, and this test is
/// what holds that in place — every unflipped row must be *identical*, not
/// merely similarly distributed.
#[test]
fn the_realized_label_noise_rate_matches_the_request_and_flips_the_same_clean_labels() {
    const ROWS: usize = 4096;
    let clean = Recipe::seeded(ROWS, 6, 9)
        .unwrap()
        .with_task(Task::LinearBinary {
            informative: 4,
            separation: 3.0,
            prevalence: 0.4,
        })
        .unwrap();
    let clean_dataset = clean.generate();
    let clean_labels = binary_labels(&clean_dataset).to_vec();
    let clean_probabilities = clean_dataset.truth().probabilities().unwrap().to_vec();

    for rate in [0.01_f32, 0.05, 0.2, 0.5] {
        let dataset = clean
            .with_contamination(Contamination::none().with_label_noise(rate))
            .unwrap()
            .generate();
        let labels = binary_labels(&dataset);
        let flipped = clean_labels
            .iter()
            .zip(labels)
            .filter(|(left, right)| left != right)
            .count();
        let realized = flipped as f64 / ROWS as f64;
        let tolerance = binomial_tolerance(f64::from(rate), ROWS);
        assert!(
            (realized - f64::from(rate)).abs() <= tolerance,
            "label noise {rate}: realized {realized}, tolerance {tolerance}"
        );

        // Every row that was not flipped kept the label the clean recipe drew.
        // A reseed would have changed roughly half of them.
        let agreeing = clean_labels
            .iter()
            .zip(labels)
            .filter(|(left, right)| left == right)
            .count();
        assert_eq!(agreeing + flipped, ROWS);

        // The recorded probabilities are `P(observed label = 1 | x)`, so they
        // carry the noise the caller asked for. A model that matched the
        // *pre-noise* probability would look mis-calibrated by exactly the
        // contamination, which is the wrong answer to report.
        let recorded = dataset.truth().probabilities().unwrap();
        for (row, (&clean_probability, &noisy)) in
            clean_probabilities.iter().zip(recorded).enumerate()
        {
            let expected = f64::from(clean_probability) * (1.0 - f64::from(rate))
                + (1.0 - f64::from(clean_probability)) * f64::from(rate);
            assert!(
                (f64::from(noisy) - expected).abs() < 1e-6,
                "row {row} at rate {rate}: recorded {noisy}, expected {expected}"
            );
        }
    }
}

/// The realized condition number is within a factor of four of the request.
///
/// **Why a factor and not a tolerance.** The family fixes the ratio of *column
/// scales* exactly; the singular-value ratio a solver sees is that times the
/// conditioning of the underlying uniform design, which is a random quantity of
/// order one whose size grows with the aspect ratio. There is no construction
/// that hits a requested singular-value ratio exactly without an orthogonal
/// factorization the generator has no business computing, so the honest contract
/// is a factor.
///
/// **Why four.** Measured across this sweep the realized ratio runs from `1.005`
/// to `1.26` times the request, so four is about three times the observed
/// extreme — headroom for a different libm's `powf` and for a design shape not
/// swept here, while still pinning the knob to within a factor on a scale that
/// spans eight decades. A generator that ignored the request entirely would come
/// back at ratio `1e-8` and fail by seven orders of magnitude.
#[test]
fn the_realized_condition_number_is_within_a_factor_of_the_request() {
    const FACTOR: f64 = 4.0;
    for requested in [1e2_f32, 1e4, 1e6, 1e8] {
        for (rows, columns) in [(64_usize, 8_usize), (2048, 16), (512, 4), (128, 32)] {
            let recipe = Recipe::seeded(rows, columns, 5)
                .unwrap()
                .with_task(Task::IllConditioned {
                    condition_number: requested,
                    rank: columns,
                    coefficient_scale: 1.0,
                    noise_scale: 0.01,
                })
                .unwrap();
            // A real power of ten is on this path, and the recipe says so.
            assert_eq!(recipe.portability(), Portability::PerRunner);

            let realized = realized_condition_number(&recipe.generate());
            let ratio = realized / f64::from(requested);
            assert!(
                (1.0 / FACTOR..=FACTOR).contains(&ratio),
                "requested {requested:e} at {rows}x{columns}: realized {realized:e} (ratio {ratio})"
            );
        }
    }

    // A condition number of one is the identity: the design is exactly what the
    // source drew, asserted by value rather than by its spectrum.
    let plain = Recipe::seeded(64, 8, 5).unwrap();
    let conditioned = plain
        .with_task(Task::IllConditioned {
            condition_number: 1.0,
            rank: 8,
            coefficient_scale: 1.0,
            noise_scale: 0.0,
        })
        .unwrap();
    assert_eq!(
        plain.design().as_slice(),
        conditioned.design().as_slice(),
        "a condition number of one must not touch the design"
    );
}

/// The ill-conditioned family reproduces the corpus the crate's least-squares
/// path is tested against.
///
/// `src/linear_model/least_squares.rs` builds its rank-deficiency corpus in
/// three conditionings — well conditioned, columns scaled geometrically down to
/// `1e-8`, and a last column duplicating the first — and the R-SVD reduction's
/// agreement, rank-floor and minimum-norm tests all rest on them. This family
/// has to be able to serve that corpus, and the claim is made three ways rather
/// than asserted once:
///
/// 1. **The conditioning transform is bit-identical.** The corpus's expression
///    is transcribed verbatim below and applied to the same base values as
///    `condition_columns`, and the two results are compared with `assert_eq!`.
///    At the corpus's `1e8` the family's `log10` is exactly `8`, so its exponent
///    reduces to the corpus's own `-8 * column / (columns - 1)` term for term.
/// 2. **The rank-deficient shape is the same shape.** With `rank = columns - 1`
///    the family's last column is an exact copy of its first, which is the
///    corpus's `Deficient` case verbatim.
/// 3. **The solver draws the same conclusion.** Fitting the crate's own
///    least-squares path to the family's output reports exactly the recorded
///    rank, at every shape and requested rank swept here. That is an `assert_eq!`
///    on an integer, not a tolerance, because the duplicated columns make the
///    algebraic rank exact rather than numerical.
#[test]
fn the_ill_conditioned_family_reproduces_the_least_squares_corpus() {
    // (1) The conditioning transform, against the corpus's own expression.
    for columns in [2_usize, 4, 8, 16, 48] {
        const ROWS: usize = 32;
        let mut base = vec![0.0_f32; ROWS * columns];
        let mut rng = OwnedRng::new(7);
        for value in &mut base {
            *value = (rng.unit_f64() * 2.0 - 1.0) as f32;
        }

        // Transcribed from `least_squares.rs`'s `Conditioning::Ill` arm, in the
        // same width and the same association order. A simplification here would
        // make this test agree with itself rather than with the corpus.
        let mut corpus = base.clone();
        for column in 0..columns {
            let scale = 10.0_f64.powf(-8.0 * column as f64 / (columns - 1).max(1) as f64) as f32;
            for row in 0..ROWS {
                corpus[row * columns + column] *= scale;
            }
        }

        let mut ours = base.clone();
        super::task::condition_columns(ROWS, columns, 1e8, &mut ours);
        assert_eq!(
            ours, corpus,
            "the family's conditioning departs from the corpus's at {columns} columns"
        );

        // And the corpus's `Deficient` arm, which is `rank = columns - 1`.
        let mut corpus = base.clone();
        for row in 0..ROWS {
            corpus[row * columns + columns - 1] = corpus[row * columns];
        }
        let mut ours = base.clone();
        super::task::duplicate_columns(ROWS, columns, columns - 1, &mut ours);
        assert_eq!(
            ours, corpus,
            "the family's rank deficiency departs from the corpus's at {columns} columns"
        );
    }

    // (2) and (3): the shape is exact and the solver reports it.
    //
    // The two conditionings are held to different statements, and the difference
    // is the corpus's whole subject. On a well-conditioned base the algebraic
    // rank is the numerical rank, so the solver's answer is compared with
    // `assert_eq!`. At a condition number of `1e8` in `f32` it is not: the
    // trailing singular values fall under the rank cutoff, and the solver
    // reports a *numerical* rank at or below the algebraic one. That is the
    // behaviour `least_squares.rs` is written to get right, not a defect here,
    // and `Truth` records the algebraic rank precisely so the two can be told
    // apart.
    let mut ill_conditioned_lost_rank = false;
    for (rows, columns) in [(64_usize, 8_usize), (2048, 16), (512, 4)] {
        for condition_number in [1.0_f32, 1e8] {
            for rank in [columns, columns - 1, columns / 2, 1] {
                let dataset = Recipe::seeded(rows, columns, 5)
                    .unwrap()
                    .with_task(Task::IllConditioned {
                        condition_number,
                        rank,
                        coefficient_scale: 1.0,
                        noise_scale: 0.01,
                    })
                    .unwrap()
                    .generate();

                assert_eq!(dataset.truth().rank(), Some(rank));
                let design = dataset.features();
                for column in rank..columns {
                    for row in 0..rows {
                        assert_eq!(
                            design.get(row, column),
                            design.get(row, column - rank),
                            "column {column} is not an exact copy of column {}",
                            column - rank
                        );
                    }
                }

                let fitted = fit_least_squares(&dataset).rank();
                if condition_number == 1.0 {
                    assert_eq!(
                        fitted, rank,
                        "the solver reports rank {fitted} on a well-conditioned {rows}x{columns} \
                         design built to rank {rank}"
                    );
                } else {
                    assert!(
                        (1..=rank).contains(&fitted),
                        "the solver reports numerical rank {fitted} on a {rows}x{columns} design \
                         whose algebraic rank is {rank}"
                    );
                    ill_conditioned_lost_rank |= fitted < rank;
                }
            }
        }
    }
    assert!(
        ill_conditioned_lost_rank,
        "no ill-conditioned design in the sweep lost numerical rank, so the family is not \
         reaching the regime the least-squares corpus exists to exercise"
    );
}

/// The generalized linear families stay inside their own support and recover
/// their recorded rate.
///
/// The tolerance on the mean is derived from the response's own sampling law
/// rather than typed in: a Poisson mean over `n` rows has standard error
/// `sqrt(μ / n)`, and a multiplicative uniform noise of half-width `d` has
/// standard error `μ d / sqrt(3 n)`. Four of either is the bound.
#[test]
fn the_glm_families_stay_inside_their_support_and_recover_their_rate() {
    const ROWS: usize = 2048;

    let counts = Recipe::seeded(ROWS, 6, 21)
        .unwrap()
        .with_task(Task::GlmRegression {
            link: GlmLink::LogCount,
            informative: 3,
            coefficient_scale: 0.5,
            intercept: 1.0,
            dispersion: 1.0,
        })
        .unwrap()
        .generate();
    let values = regression_values(&counts);
    assert!(
        values
            .iter()
            .all(|&value| value >= 0.0 && value.fract() == 0.0),
        "a count response left the non-negative integers"
    );
    let rate = mean(counts.truth().conditional_mean().unwrap());
    let tolerance = 4.0 * (rate / ROWS as f64).sqrt();
    assert!(
        (mean(values) - rate).abs() <= tolerance,
        "count mean {} departs from the recorded rate {rate} by more than {tolerance}",
        mean(values)
    );

    let dispersion = 0.5_f32;
    let positive = Recipe::seeded(ROWS, 6, 21)
        .unwrap()
        .with_task(Task::GlmRegression {
            link: GlmLink::LogPositive,
            informative: 3,
            coefficient_scale: 0.5,
            intercept: 1.0,
            dispersion,
        })
        .unwrap()
        .generate();
    let values = regression_values(&positive);
    assert!(
        values.iter().all(|&value| value > 0.0),
        "a positive response reached zero or below"
    );
    let rate = mean(positive.truth().conditional_mean().unwrap());
    let tolerance = 4.0 * rate * f64::from(dispersion) / (3.0 * ROWS as f64).sqrt();
    assert!(
        (mean(values) - rate).abs() <= tolerance,
        "positive mean {} departs from the recorded rate {rate} by more than {tolerance}",
        mean(values)
    );

    // A dispersion of one or more would let the multiplicative noise reach zero,
    // so the link refuses it rather than emitting a non-positive response.
    assert_eq!(
        Recipe::seeded(16, 4, 1)
            .unwrap()
            .with_task(Task::GlmRegression {
                link: GlmLink::LogPositive,
                informative: 2,
                coefficient_scale: 0.5,
                intercept: 0.0,
                dispersion: 1.0,
            }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Dispersion
        })
    );
    // The same dispersion is admissible on a count response, where it is the
    // ordinary Poisson case rather than the degenerate one. The two links have
    // disjoint admissible ranges, and each refuses exactly what the other takes.
    assert!(
        Recipe::seeded(16, 4, 1)
            .unwrap()
            .with_task(Task::GlmRegression {
                link: GlmLink::LogCount,
                informative: 2,
                coefficient_scale: 0.5,
                intercept: 0.0,
                dispersion: 1.0,
            })
            .is_ok()
    );
    // A Poisson cannot be under-dispersed, so a count response refuses what a
    // positive one requires. That is also what bounds the draw: Knuth's product
    // costs one uniform per unit of rate, and the rate is `μ / dispersion`.
    assert_eq!(
        Recipe::seeded(16, 4, 1)
            .unwrap()
            .with_task(Task::GlmRegression {
                link: GlmLink::LogCount,
                informative: 2,
                coefficient_scale: 0.5,
                intercept: 0.0,
                dispersion: 0.5,
            }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Dispersion
        })
    );
}

/// The four nonlinear boundaries are four different problems, and three of them
/// are problems no linear model can solve.
///
/// Without the first half the four variants could collapse onto one expression
/// and every other assertion about them would still hold. The second half is
/// what makes them worth having: it measures how far the *best* linear rule over
/// the same design falls short of the boundary's own Bayes accuracy.
///
/// # The instrument this replaced was measuring the wrong thing
///
/// It compared each boundary's labels against the labels of one
/// [`Task::LinearBinary`] recipe and required them to disagree on more than a
/// quarter of the rows. That recipe's coefficients are a *random draw*, so the
/// quantity being thresholded was which way an unrelated vector happened to
/// point. Swept over eight seeds at the same shape, the moons disagreement ran
/// `0.2388, 0.6909, 0.5874, 0.5879, 0.4932, 0.4277, 0.6299, 0.5708` — and the
/// assertion's own logic scored the anti-correlated draws at `0.69` as *more*
/// nonlinear than the aligned one at `0.24`, which a sign flip alone produces.
/// The threshold passed on the draw that existed when it was written and failed
/// the moment the draw moved, which is what surfaced it.
///
/// The replacement fits the least-squares linear rule to the boundary's own
/// labels and places its threshold at the labels' own positive rate, so it is
/// invariant to sign and to any other recipe. Measured over seeds `13..=17`:
///
/// | boundary | Bayes − best linear |
/// |---|---|
/// | `Checkerboard` | `0.449 .. 0.470` |
/// | `Xor` | `0.334 .. 0.353` |
/// | `Circles` | `0.228 .. 0.250` |
/// | `Moons` | `-0.009 .. 0.014` |
/// | `LinearBinary` (control) | `-0.015 .. 0.006` |
///
/// # Moons is linearly solvable at this design's width, and says so
///
/// The control is what makes the rest of the table readable: on a genuinely
/// linear problem the gap is zero to within a percent, which is the instrument
/// reading its own null. Moons sits in that null band. Its boundary is
/// `x₂ = 0.6 sin(2 x₁)`, and over `x₁ ∈ [-1, 1)` a sine of that argument is
/// nearly its own tangent line, so the curvature costs the best linear rule
/// under two points of accuracy. Asserting that rather than hiding it is the
/// point: a consumer choosing a boundary to defeat a linear model should choose
/// one of the other three, and a consumer wanting a *mildly* curved boundary
/// now has one that is documented as mild.
#[test]
fn the_nonlinear_binary_boundaries_are_four_different_problems() {
    let kinds = [
        BinaryKind::Xor,
        BinaryKind::Moons,
        BinaryKind::Circles,
        BinaryKind::Checkerboard,
    ];
    let mut datasets = Vec::new();
    for kind in kinds {
        let recipe = Recipe::seeded(2048, 4, 13)
            .unwrap()
            .with_task(Task::NonlinearBinary {
                kind,
                separation: 4.0,
                prevalence: 0.5,
            })
            .unwrap();
        // Every boundary carries the logistic link, so the family is per-runner
        // even where the boundary's own arithmetic is exact. The envelope is a
        // property of the family, not of its parts.
        assert_eq!(recipe.portability(), Portability::PerRunner, "{kind:?}");
        let dataset = recipe.generate();
        // No coefficient vector produces these, and the truth says so with a
        // variant rather than with zeros.
        assert_eq!(dataset.truth().coefficients(), None);
        assert!(dataset.truth().probabilities().is_some());
        datasets.push((kind, dataset));
    }
    for (index, (left_kind, left)) in datasets.iter().enumerate() {
        for (right_kind, right) in &datasets[index + 1..] {
            assert_ne!(
                binary_labels(left),
                binary_labels(right),
                "{left_kind:?} and {right_kind:?} produced the same labels"
            );
        }
    }

    // The instrument's null: a linear problem, where the best linear rule *is*
    // the Bayes rule and the gap must therefore read zero. Without this the
    // three positive readings below would have no scale.
    let linear = Recipe::seeded(2048, 4, 13)
        .unwrap()
        .with_task(Task::LinearBinary {
            informative: 2,
            separation: 4.0,
            prevalence: 0.5,
        })
        .unwrap()
        .generate();
    assert!(
        linear_shortfall(&linear).abs() < NULL_SHORTFALL,
        "the control read {} on a linear problem",
        linear_shortfall(&linear)
    );

    for (kind, dataset) in &datasets {
        let shortfall = linear_shortfall(dataset);
        match kind {
            // A curved, repeating or enclosing boundary: no linear rule comes
            // close, and the smallest of the three reads fifteen times the null
            // band the control and the moons boundary sit inside.
            BinaryKind::Xor | BinaryKind::Circles | BinaryKind::Checkerboard => assert!(
                shortfall > CURVED_SHORTFALL,
                "{kind:?} left the best linear rule only {shortfall} short of Bayes"
            ),
            // Documented above: mild curvature a line absorbs.
            BinaryKind::Moons => assert!(
                shortfall.abs() < NULL_SHORTFALL,
                "the moons boundary read {shortfall}, outside the null band it is \
                 documented to sit in"
            ),
        }
    }
}

/// The largest shortfall a *linearly solvable* problem produced over seeds
/// `13..=17`, rounded up by a factor of three.
///
/// The measured extreme was `0.015` across the linear control and the moons
/// boundary together, so this is loose enough that a different libm's logistic
/// cannot reach it and tight enough that it stays an order of magnitude below
/// the smallest curved reading.
const NULL_SHORTFALL: f64 = 0.05;

/// The smallest shortfall a *curved* boundary produced over the same seeds,
/// rounded down by a third.
///
/// The measured minimum was `0.228`, on `Circles`. Three times the null band and
/// two thirds of the minimum, so the two populations are separated by a factor
/// of ten in either direction rather than by a margin that had to be fitted.
const CURVED_SHORTFALL: f64 = 0.15;

/// How far the best linear rule over a design falls short of the Bayes accuracy
/// the family recorded for it.
///
/// The rule is the least-squares fit to the labels themselves — the linear
/// discriminant direction, up to a scale a threshold absorbs — and its threshold
/// is placed at the labels' own positive rate, so the rule is scored at the same
/// prevalence the labels carry and the reading cannot be moved by an intercept.
/// Both properties matter: the measurement is then a function of the boundary
/// and the design alone, invariant to the sign of any coefficient vector and to
/// every other recipe in the file.
fn linear_shortfall(dataset: &Dataset) -> f64 {
    let labels = binary_labels(dataset);
    let design = dataset.features();
    let targets =
        crate::data::RegressionTargets::new(labels.iter().map(|&label| f32::from(label)).collect())
            .expect("a label is zero or one");
    let fit = LinearRegression::fit(
        &design.as_view(),
        &targets,
        LinearRegressionParams::default(),
    )
    .expect("a generated design fits");
    let scores: Vec<f64> = design
        .iter_rows()
        .map(|row| dot_f64(row, fit.coefficients()) + f64::from(fit.intercept()))
        .collect();

    // The rule calls the highest-scoring `positives` rows positive, so it spends
    // exactly the positive budget the labels do and its accuracy is comparable
    // with a Bayes accuracy that also predicts the majority side of each row.
    let positives = labels.iter().filter(|&&label| label == 1).count();
    let mut ordered = scores.clone();
    ordered.sort_by(f64::total_cmp);
    let threshold = ordered[ordered.len() - positives.max(1)];
    let correct = scores
        .iter()
        .zip(labels)
        .filter(|&(&score, &label)| u8::from(score >= threshold) == label)
        .count();
    let linear = correct as f64 / labels.len() as f64;

    bayes_accuracy(dataset) - linear
}

/// `x · β` in `f64`, for the scoring above.
fn dot_f64(row: &[f32], coefficients: &[f32]) -> f64 {
    row.iter()
        .zip(coefficients)
        .map(|(&value, &coefficient)| f64::from(value) * f64::from(coefficient))
        .sum()
}

/// The accuracy of the Bayes rule the family recorded, which is the mean of
/// `max(p, 1 - p)` over the recorded probabilities.
///
/// This is what makes a difficulty dial measurable without fitting anything: it
/// is a property of the *problem*, so a sweep over it reports what the knob did
/// rather than what a particular estimator did with it.
fn bayes_accuracy(dataset: &Dataset) -> f64 {
    let probabilities = dataset
        .truth()
        .probabilities()
        .expect("a binary family records its Bayes probabilities");
    probabilities
        .iter()
        .map(|&p| f64::from(p).max(1.0 - f64::from(p)))
        .sum::<f64>()
        / probabilities.len() as f64
}

/// Every task parameter is refused by name, before anything is generated.
#[test]
fn every_task_parameter_is_refused_by_name_before_generation() {
    let recipe = Recipe::seeded(32, 6, 1).unwrap();

    assert_eq!(
        recipe.with_task(Task::LinearRegression {
            informative: 0,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::ZeroInformativeColumns)
    );
    assert_eq!(
        recipe.with_task(Task::LinearRegression {
            informative: 7,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::InformativeColumnsExceedDesign {
            informative: 7,
            columns: 6
        })
    );
    assert_eq!(
        recipe.with_task(Task::LinearRegression {
            informative: 2,
            coefficient_scale: 0.0,
            intercept: 0.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::CoefficientScale
        })
    );
    assert_eq!(
        recipe.with_task(Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: f32::NAN,
            noise_scale: 0.0,
        }),
        Err(DatasetError::NonFiniteParameter {
            parameter: Parameter::Intercept
        })
    );
    assert_eq!(
        recipe.with_task(Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: -1.0,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::NoiseScale
        })
    );

    // A nonlinear shape's own column appetite comes through the same error, so a
    // caller reading it learns the same thing whether the count was theirs or
    // the shape's. Friedman reads five columns and cannot be drawn over four.
    assert_eq!(
        Recipe::seeded(32, 4, 1)
            .unwrap()
            .with_task(Task::NonlinearRegression {
                kind: NonlinearKind::Friedman,
                noise_scale: 0.0,
            }),
        Err(DatasetError::InformativeColumnsExceedDesign {
            informative: 5,
            columns: 4
        })
    );

    assert_eq!(
        recipe.with_task(Task::IllConditioned {
            condition_number: 0.5,
            rank: 6,
            coefficient_scale: 1.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::ConditionNumber
        })
    );
    assert_eq!(
        recipe.with_task(Task::IllConditioned {
            condition_number: 10.0,
            rank: 0,
            coefficient_scale: 1.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::ZeroRank)
    );
    assert_eq!(
        recipe.with_task(Task::IllConditioned {
            condition_number: 10.0,
            rank: 9,
            coefficient_scale: 1.0,
            noise_scale: 0.0,
        }),
        Err(DatasetError::RankExceedsDesign {
            rank: 9,
            columns: 6
        })
    );

    for prevalence in [0.0_f32, 1.0, -0.5, f32::NAN] {
        let error = recipe
            .with_task(Task::LinearBinary {
                informative: 2,
                separation: 1.0,
                prevalence,
            })
            .unwrap_err();
        assert!(
            matches!(
                error,
                DatasetError::ParameterOutOfRange {
                    parameter: Parameter::Prevalence
                } | DatasetError::NonFiniteParameter {
                    parameter: Parameter::Prevalence
                }
            ),
            "prevalence {prevalence} was refused as {error}"
        );
    }
    assert_eq!(
        recipe.with_task(Task::NonlinearBinary {
            kind: BinaryKind::Xor,
            separation: -1.0,
            prevalence: 0.5,
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::Separation
        })
    );

    // Every message names itself, and no two of them read the same.
    let messages: Vec<String> = [
        DatasetError::ZeroInformativeColumns,
        DatasetError::InformativeColumnsExceedDesign {
            informative: 7,
            columns: 6,
        },
        DatasetError::ZeroRank,
        DatasetError::RankExceedsDesign {
            rank: 9,
            columns: 6,
        },
        DatasetError::NonFiniteParameter {
            parameter: Parameter::Intercept,
        },
        DatasetError::ParameterOutOfRange {
            parameter: Parameter::Prevalence,
        },
        DatasetError::ConstantColumnsLeaveNoSignal {
            constant_columns: 6,
            columns: 6,
        },
        DatasetError::CollinearPairsExceedDesign {
            pairs: 4,
            available: 6,
        },
        DatasetError::ContaminationNeedsLabels {
            parameter: Parameter::LabelNoise,
        },
        DatasetError::ContaminationNeedsAdditiveNoise {
            parameter: Parameter::HeavyTail,
        },
        DatasetError::WeightPatternNeedsLabels,
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

/// A contamination the current task cannot carry is refused rather than ignored.
///
/// The failure this prevents is silent and expensive: a robustness sweep that
/// set a label-noise rate on a regression task and got clean data back would
/// report the model as robust to a contamination it never received.
#[test]
fn a_contamination_the_task_cannot_carry_is_refused_rather_than_ignored() {
    let regression = Recipe::seeded(64, 6, 2)
        .unwrap()
        .with_task(Task::LinearRegression {
            informative: 3,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.1,
        })
        .unwrap();
    let binary = Recipe::seeded(64, 6, 2)
        .unwrap()
        .with_task(Task::LinearBinary {
            informative: 3,
            separation: 2.0,
            prevalence: 0.3,
        })
        .unwrap();
    let counts = Recipe::seeded(64, 6, 2)
        .unwrap()
        .with_task(Task::GlmRegression {
            link: GlmLink::LogCount,
            informative: 3,
            coefficient_scale: 0.5,
            intercept: 0.5,
            dispersion: 1.0,
        })
        .unwrap();
    let bare = Recipe::seeded(64, 6, 2).unwrap();

    assert_eq!(
        regression.with_contamination(Contamination::none().with_label_noise(0.1)),
        Err(DatasetError::ContaminationNeedsLabels {
            parameter: Parameter::LabelNoise
        })
    );
    assert_eq!(
        bare.with_contamination(Contamination::none().with_label_noise(0.1)),
        Err(DatasetError::ContaminationNeedsLabels {
            parameter: Parameter::LabelNoise
        })
    );
    for parameter in [Parameter::HeavyTail, Parameter::Heteroscedastic] {
        let contamination = if matches!(parameter, Parameter::HeavyTail) {
            Contamination::none().with_heavy_tail(0.1)
        } else {
            Contamination::none().with_heteroscedastic(1.0)
        };
        assert_eq!(
            binary.with_contamination(contamination),
            Err(DatasetError::ContaminationNeedsAdditiveNoise { parameter })
        );
        assert_eq!(
            counts.with_contamination(contamination),
            Err(DatasetError::ContaminationNeedsAdditiveNoise { parameter }),
            "a count response's scatter is its dispersion, not an additive term"
        );
    }
    // An outlier fraction *is* carried by a generalized linear response, which
    // displaces it multiplicatively so a count stays a count.
    assert!(
        counts
            .with_contamination(Contamination::none().with_outlier_fraction(0.05))
            .is_ok()
    );
    assert_eq!(
        binary.with_contamination(Contamination::none().with_outlier_fraction(0.05)),
        Err(DatasetError::ContaminationNeedsAdditiveNoise {
            parameter: Parameter::OutlierFraction
        })
    );

    // Order of the builder calls does not change which recipes exist: setting
    // the contamination first and the task second is refused identically.
    assert_eq!(
        Recipe::seeded(64, 6, 2)
            .unwrap()
            .with_contamination(Contamination::none().with_heavy_tail(0.1))
            .unwrap_err(),
        DatasetError::ContaminationNeedsAdditiveNoise {
            parameter: Parameter::HeavyTail
        }
    );

    assert_eq!(
        regression.with_weights(WeightPattern::ClassBalanced),
        Err(DatasetError::WeightPatternNeedsLabels)
    );
    assert!(binary.with_weights(WeightPattern::ClassBalanced).is_ok());

    // Design-shaping knobs apply to every recipe, task or no task.
    assert!(
        bare.with_contamination(
            Contamination::none()
                .with_duplicate_rows(0.25)
                .with_constant_columns(1)
                .with_collinear_pairs(2)
                .with_feature_scale_spread(2.0)
        )
        .is_ok()
    );
    assert_eq!(
        bare.with_contamination(Contamination::none().with_constant_columns(6)),
        Err(DatasetError::ConstantColumnsLeaveNoSignal {
            constant_columns: 6,
            columns: 6
        })
    );
    assert_eq!(
        bare.with_contamination(
            Contamination::none()
                .with_constant_columns(1)
                .with_collinear_pairs(3)
        ),
        Err(DatasetError::CollinearPairsExceedDesign {
            pairs: 3,
            available: 5
        })
    );
}

/// The design-shaping knobs are realized exactly, because they are index
/// arithmetic rather than draws.
#[test]
fn the_design_shaping_contamination_knobs_are_realized_exactly() {
    const ROWS: usize = 200;
    const COLUMNS: usize = 8;
    let bare = Recipe::seeded(ROWS, COLUMNS, 4).unwrap();
    let plain = bare.design();

    let contaminated = bare
        .with_contamination(
            Contamination::none()
                .with_constant_columns(2)
                .with_collinear_pairs(2)
                .with_duplicate_rows(0.1),
        )
        .unwrap()
        .design();

    // The last two columns are constant.
    for row in 0..ROWS {
        assert_eq!(contaminated.get(row, 6), Some(1.0));
        assert_eq!(contaminated.get(row, 7), Some(1.0));
    }
    // The two columns below the constant tail are exact multiples of the first
    // two, and are *not* copies: a model that only detects duplicated columns
    // has not detected collinearity.
    for row in 0..ROWS {
        assert_eq!(
            contaminated.get(row, 5),
            Some(2.0 * contaminated.get(row, 0).unwrap())
        );
        assert_eq!(
            contaminated.get(row, 4),
            Some(2.0 * contaminated.get(row, 1).unwrap())
        );
        assert_ne!(contaminated.get(row, 5), contaminated.get(row, 0));
    }
    // A tenth of the rows are exact copies of the first tenth, and the rest are
    // untouched by the row duplication.
    let duplicated = 20;
    for row in 0..duplicated {
        assert_eq!(
            contaminated.row(ROWS - duplicated + row),
            contaminated.row(row)
        );
    }
    for row in 0..(ROWS - duplicated) {
        // The columns the other knobs did not touch still carry the source's
        // own values.
        assert_eq!(
            contaminated.get(row, 2),
            plain.get(row, 2),
            "an untouched column moved at row {row}"
        );
    }

    // The per-column scale spread is a geometric schedule across the columns:
    // the last column is `10^-decades` times the first, up to the rounding of a
    // single `f32` multiply.
    let spread = bare
        .with_contamination(Contamination::none().with_feature_scale_spread(3.0))
        .unwrap()
        .design();
    for row in 0..ROWS {
        let first = plain.get(row, 0).unwrap();
        let last = plain.get(row, COLUMNS - 1).unwrap();
        assert_eq!(spread.get(row, 0), Some(first));
        let scaled = spread.get(row, COLUMNS - 1).unwrap();
        if last != 0.0 {
            let ratio = f64::from(scaled) / f64::from(last);
            assert!(
                (ratio - 1e-3).abs() < 1e-3 * 1e-6,
                "the last column was scaled by {ratio} rather than 1e-3"
            );
        }
    }
}

/// The noise-shaping knobs move the target, and move only what they name.
#[test]
fn the_noise_shaping_contamination_knobs_move_only_what_they_name() {
    const ROWS: usize = 4096;
    let base = Recipe::seeded(ROWS, 6, 3)
        .unwrap()
        .with_task(Task::LinearRegression {
            informative: 3,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.1,
        })
        .unwrap();
    let clean = base.target_values().unwrap();
    let clean_extreme = clean
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);

    // A heavy tail leaves most rows exactly where they were and drags a few far
    // past anything the uniform noise reaches. Both halves matter: the first
    // says the knob is a component mixture rather than a rescaling, and the
    // second says the component is genuinely heavy.
    let heavy_rate = 0.1_f32;
    let heavy = base
        .with_contamination(Contamination::none().with_heavy_tail(heavy_rate))
        .unwrap();
    let heavy_values = heavy.target_values().unwrap();
    let moved = clean
        .iter()
        .zip(&heavy_values)
        .filter(|(left, right)| left != right)
        .count();
    let realized = moved as f64 / ROWS as f64;
    assert!(
        (realized - f64::from(heavy_rate)).abs() <= binomial_tolerance(f64::from(heavy_rate), ROWS),
        "heavy-tail rate realized at {realized}"
    );
    let heavy_extreme = heavy_values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        heavy_extreme > 10.0 * clean_extreme,
        "the heavy-tailed component reached only {heavy_extreme} against {clean_extreme}"
    );
    // And the design is untouched: a target contamination is a target
    // contamination.
    assert_eq!(base.design().as_slice(), heavy.design().as_slice());

    // Outliers displace a requested share of the targets.
    let outlier_rate = 0.05_f32;
    let outliers = base
        .with_contamination(Contamination::none().with_outlier_fraction(outlier_rate))
        .unwrap()
        .target_values()
        .unwrap();
    let moved = clean
        .iter()
        .zip(&outliers)
        .filter(|(left, right)| left != right)
        .count();
    let realized = moved as f64 / ROWS as f64;
    assert!(
        (realized - f64::from(outlier_rate)).abs()
            <= binomial_tolerance(f64::from(outlier_rate), ROWS),
        "outlier fraction realized at {realized}"
    );

    // Heteroscedasticity scales the noise by the first feature's magnitude, so
    // it touches every row and widens the spread rather than shifting it.
    let heteroscedastic = base
        .with_contamination(Contamination::none().with_heteroscedastic(1.0))
        .unwrap()
        .target_values()
        .unwrap();
    assert!(
        clean
            .iter()
            .zip(&heteroscedastic)
            .filter(|(left, right)| left != right)
            .count()
            > ROWS * 99 / 100,
        "a heteroscedastic scale left most rows untouched"
    );
}

/// The weight patterns are deterministic functions of the row and the label.
#[test]
fn weight_patterns_are_deterministic_functions_of_the_row_and_the_label() {
    const ROWS: usize = 512;
    let binary = Recipe::seeded(ROWS, 5, 6)
        .unwrap()
        .with_task(Task::LinearBinary {
            informative: 3,
            separation: 2.0,
            prevalence: 0.2,
        })
        .unwrap();

    let uniform = binary
        .with_weights(WeightPattern::Uniform)
        .unwrap()
        .generate();
    assert!(
        uniform
            .weights()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&weight| weight == 1.0)
    );

    let ramp = binary
        .with_weights(WeightPattern::Ramp {
            low: 0.5,
            high: 2.5,
        })
        .unwrap()
        .generate();
    let weights = ramp.weights().unwrap().as_slice();
    assert_eq!(weights[0], 0.5);
    assert_eq!(weights[ROWS - 1], 2.5);
    assert!(
        weights.windows(2).all(|pair| pair[0] <= pair[1]),
        "a ramp must be monotone"
    );

    let alternating = binary
        .with_weights(WeightPattern::Alternating {
            first: 3.0,
            second: 1.0,
        })
        .unwrap()
        .generate();
    let weights = alternating.weights().unwrap().as_slice();
    for (row, &weight) in weights.iter().enumerate() {
        assert_eq!(weight, if row % 2 == 0 { 3.0 } else { 1.0 });
    }

    // Class balancing gives each class the same total weight, which is what
    // turns a controlled prevalence into a controlled *imbalance* experiment.
    let balanced = binary
        .with_weights(WeightPattern::ClassBalanced)
        .unwrap()
        .generate();
    let weights = balanced.weights().unwrap().as_slice();
    let labels = binary_labels(&balanced);
    let positive_total: f64 = labels
        .iter()
        .zip(weights)
        .filter(|&(&label, _)| label == 1)
        .map(|(_, &weight)| f64::from(weight))
        .sum();
    let negative_total: f64 = labels
        .iter()
        .zip(weights)
        .filter(|&(&label, _)| label == 0)
        .map(|(_, &weight)| f64::from(weight))
        .sum();
    assert!(
        (positive_total - negative_total).abs() < 1e-3,
        "class totals were {positive_total} and {negative_total}"
    );
    // The minority class is up-weighted, which is only visible because the
    // prevalence was controlled in the first place.
    let positives = labels.iter().filter(|&&label| label == 1).count();
    assert!(positives * 3 < ROWS, "the fixture is not imbalanced");
    assert!(
        weights[labels.iter().position(|&label| label == 1).unwrap()]
            > weights[labels.iter().position(|&label| label == 0).unwrap()]
    );

    // A weight pattern whose totals cannot be positive is refused at the
    // constructor, before any weights are built.
    assert_eq!(
        binary.with_weights(WeightPattern::Ramp {
            low: 0.0,
            high: 0.0
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::WeightHigh
        })
    );
    assert_eq!(
        binary.with_weights(WeightPattern::Alternating {
            first: -1.0,
            second: 1.0
        }),
        Err(DatasetError::ParameterOutOfRange {
            parameter: Parameter::WeightFirst
        })
    );

    // A dataset without weights is not a dataset whose weights are all one.
    assert!(binary.generate().weights().is_none());
}

/// The declared envelope is the weaker of the task's and the contamination's.
///
/// The interesting case is the one the crate's culture would otherwise get
/// wrong: a boundary whose own arithmetic is exact still reports `PerRunner`,
/// because the logistic link that turns it into a Bayes probability is not. The
/// envelope is a property of the family, never of its parts.
#[test]
fn the_declared_portability_envelope_is_the_weaker_of_the_task_and_the_contamination() {
    assert_eq!(
        Task::LinearRegression {
            informative: 1,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.0,
        }
        .portability(),
        Portability::BitExact
    );
    for kind in [NonlinearKind::Interaction, NonlinearKind::Piecewise] {
        assert_eq!(
            Task::NonlinearRegression {
                kind,
                noise_scale: 0.0
            }
            .portability(),
            Portability::BitExact,
            "{kind:?}"
        );
    }
    for kind in [NonlinearKind::Sinusoid, NonlinearKind::Friedman] {
        assert_eq!(
            Task::NonlinearRegression {
                kind,
                noise_scale: 0.0
            }
            .portability(),
            Portability::PerRunner,
            "{kind:?}"
        );
    }
    for kind in [
        BinaryKind::Xor,
        BinaryKind::Circles,
        BinaryKind::Checkerboard,
    ] {
        assert_eq!(
            kind.boundary_portability(),
            Portability::BitExact,
            "{kind:?}'s own arithmetic is exact"
        );
        assert_eq!(
            Task::NonlinearBinary {
                kind,
                separation: 1.0,
                prevalence: 0.5
            }
            .portability(),
            Portability::PerRunner,
            "{kind:?} still carries the logistic link"
        );
    }
    assert_eq!(
        BinaryKind::Moons.boundary_portability(),
        Portability::PerRunner
    );

    // A bit-exact task with a scale-spread contamination is per-runner, and one
    // with any other contamination is not.
    let recipe = Recipe::seeded(32, 4, 1)
        .unwrap()
        .with_task(Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.1,
        })
        .unwrap();
    assert_eq!(recipe.portability(), Portability::BitExact);
    assert_eq!(
        recipe
            .with_contamination(
                Contamination::none()
                    .with_heavy_tail(0.1)
                    .with_outlier_fraction(0.1)
                    .with_duplicate_rows(0.1)
                    .with_constant_columns(1)
            )
            .unwrap()
            .portability(),
        Portability::BitExact,
        "no knob but the scale spread evaluates a transcendental"
    );
    assert_eq!(
        recipe
            .with_contamination(Contamination::none().with_feature_scale_spread(1.0))
            .unwrap()
            .portability(),
        Portability::PerRunner
    );
}

/// The spec digest separates every field this phase added.
#[test]
fn the_spec_digest_separates_tasks_contaminations_and_weight_patterns() {
    let bare = Recipe::seeded(32, 6, 1).unwrap();
    let tasks = [
        Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.1,
        },
        // One field apart from the first, in each field in turn.
        Task::LinearRegression {
            informative: 3,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.1,
        },
        Task::LinearRegression {
            informative: 2,
            coefficient_scale: 2.0,
            intercept: 0.0,
            noise_scale: 0.1,
        },
        Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.5,
            noise_scale: 0.1,
        },
        Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.0,
            noise_scale: 0.2,
        },
        Task::NonlinearRegression {
            kind: NonlinearKind::Interaction,
            noise_scale: 0.1,
        },
        // A different kind under the same variant: the sub-discriminant is what
        // has to separate these.
        Task::NonlinearRegression {
            kind: NonlinearKind::Piecewise,
            noise_scale: 0.1,
        },
        Task::IllConditioned {
            condition_number: 100.0,
            rank: 6,
            coefficient_scale: 1.0,
            noise_scale: 0.1,
        },
        Task::IllConditioned {
            condition_number: 100.0,
            rank: 5,
            coefficient_scale: 1.0,
            noise_scale: 0.1,
        },
        Task::LinearBinary {
            informative: 2,
            separation: 1.0,
            prevalence: 0.5,
        },
        Task::NonlinearBinary {
            kind: BinaryKind::Xor,
            separation: 1.0,
            prevalence: 0.5,
        },
        Task::NonlinearBinary {
            kind: BinaryKind::Circles,
            separation: 1.0,
            prevalence: 0.5,
        },
    ];
    let mut digests = vec![bare.spec_digest()];
    for task in tasks {
        digests.push(bare.with_task(task).unwrap().spec_digest());
    }

    let regression = bare.with_task(tasks[0]).unwrap();
    for contamination in [
        Contamination::none().with_outlier_fraction(0.1),
        Contamination::none().with_heavy_tail(0.1),
        Contamination::none().with_heteroscedastic(0.1),
        Contamination::none().with_duplicate_rows(0.1),
        Contamination::none().with_constant_columns(1),
        Contamination::none().with_collinear_pairs(1),
        Contamination::none().with_feature_scale_spread(1.0),
    ] {
        digests.push(
            regression
                .with_contamination(contamination)
                .unwrap()
                .spec_digest(),
        );
    }
    for pattern in [
        WeightPattern::Uniform,
        WeightPattern::Ramp {
            low: 1.0,
            high: 2.0,
        },
        // Swapped, which a digest that summed its fields would miss.
        WeightPattern::Ramp {
            low: 2.0,
            high: 1.0,
        },
        WeightPattern::Alternating {
            first: 1.0,
            second: 2.0,
        },
    ] {
        digests.push(regression.with_weights(pattern).unwrap().spec_digest());
    }

    for (index, left) in digests.iter().enumerate() {
        for right in &digests[index + 1..] {
            assert_ne!(left, right, "two recipes at index {index} share a digest");
        }
    }
}

/// A default contamination is inert, so the streams P1 to P3 froze are unmoved.
///
/// The absorbed reference and benchmark fixtures go through the same
/// `design_into` this phase extended, and their pinned literals live in
/// `tests.rs`. This is the direct statement that the extension is a no-op when
/// nothing was asked for: the contaminated path and the untouched path produce
/// the same bytes, value for value, for every source.
#[test]
fn a_default_contamination_leaves_the_generated_design_untouched() {
    for source in [
        Source::Sampled { state: 11 },
        Source::Lattice {
            row_stride: 131,
            column_stride: 17,
            modulus: 1009,
        },
        Source::Xorshift32 { state: 0x9e37_79b9 },
    ] {
        let recipe = Recipe::new(64, 12, source).unwrap();
        let untouched = recipe.design();
        let explicit = recipe
            .with_contamination(Contamination::none())
            .unwrap()
            .design();
        assert_eq!(untouched.as_slice(), explicit.as_slice(), "{source:?}");
        assert_eq!(recipe.contamination(), Contamination::none());
        assert_eq!(recipe.contamination(), Contamination::default());
        assert_eq!(recipe.task(), None);
        assert_eq!(recipe.weight_pattern(), None);
    }
}

/// Regenerating a recipe carrying a task reproduces its bytes exactly.
#[test]
fn regenerating_a_task_recipe_reproduces_its_bytes() {
    let recipes = [
        Recipe::seeded(128, 6, 2)
            .unwrap()
            .with_task(Task::LinearRegression {
                informative: 3,
                coefficient_scale: 1.5,
                intercept: 0.25,
                noise_scale: 0.2,
            })
            .unwrap()
            .with_contamination(
                Contamination::none()
                    .with_heavy_tail(0.1)
                    .with_outlier_fraction(0.05),
            )
            .unwrap(),
        Recipe::seeded(128, 6, 2)
            .unwrap()
            .with_task(Task::LinearBinary {
                informative: 3,
                separation: 2.0,
                prevalence: 0.3,
            })
            .unwrap()
            .with_contamination(Contamination::none().with_label_noise(0.1))
            .unwrap()
            .with_weights(WeightPattern::ClassBalanced)
            .unwrap(),
        Recipe::seeded(128, 6, 2)
            .unwrap()
            .with_task(Task::GlmRegression {
                link: GlmLink::LogCount,
                informative: 2,
                coefficient_scale: 0.4,
                intercept: 0.7,
                dispersion: 2.0,
            })
            .unwrap(),
    ];
    for recipe in recipes {
        assert_eq!(recipe.generate(), recipe.generate());
        assert_eq!(recipe.target_values(), recipe.target_values());
        assert_eq!(recipe.generate().spec_digest(), recipe.spec_digest());
    }
}

/// The truth accessors report exactly what the family knows, and nothing else.
#[test]
fn truth_accessors_report_only_what_the_family_knows() {
    let bare = Recipe::seeded(32, 6, 8).unwrap().generate();
    assert_eq!(bare.truth(), &Truth::DesignOnly);
    assert_eq!(bare.truth().coefficients(), None);
    assert_eq!(bare.truth().conditional_mean(), None);
    assert_eq!(bare.truth().probabilities(), None);
    assert_eq!(bare.truth().intercept(), None);
    assert_eq!(bare.truth().rank(), None);

    // An absorbed lane draws a task and still records nothing, which stays a
    // third statement rather than becoming one of the new variants.
    let lane = ReferenceQuality::new(ReferenceLane::SeparableBinary, 11).train();
    assert_eq!(lane.truth(), &Truth::Unrecorded);
    assert_eq!(lane.truth().coefficients(), None);
    assert_eq!(lane.truth().probabilities(), None);

    let linear = Recipe::seeded(32, 6, 8)
        .unwrap()
        .with_task(Task::LinearRegression {
            informative: 2,
            coefficient_scale: 1.0,
            intercept: 0.5,
            noise_scale: 0.1,
        })
        .unwrap()
        .generate();
    assert_eq!(linear.truth().coefficients().unwrap().len(), 6);
    assert_eq!(linear.truth().conditional_mean().unwrap().len(), 32);
    assert_eq!(linear.truth().intercept(), Some(0.5));
    assert_eq!(linear.truth().probabilities(), None);
    assert_eq!(linear.truth().rank(), None);

    let nonlinear = Recipe::seeded(32, 6, 8)
        .unwrap()
        .with_task(Task::NonlinearRegression {
            kind: NonlinearKind::Sinusoid,
            noise_scale: 0.1,
        })
        .unwrap()
        .generate();
    assert_eq!(nonlinear.truth().coefficients(), None);
    assert_eq!(nonlinear.truth().conditional_mean().unwrap().len(), 32);

    let binary = Recipe::seeded(32, 6, 8)
        .unwrap()
        .with_task(Task::LinearBinary {
            informative: 2,
            separation: 2.0,
            prevalence: 0.5,
        })
        .unwrap()
        .generate();
    assert_eq!(binary.truth().coefficients().unwrap().len(), 6);
    assert_eq!(binary.truth().probabilities().unwrap().len(), 32);
    assert_eq!(binary.truth().conditional_mean(), None);
    assert!(
        binary
            .truth()
            .probabilities()
            .unwrap()
            .iter()
            .all(|&probability| (0.0..=1.0).contains(&probability))
    );

    let conditioned = Recipe::seeded(32, 6, 8)
        .unwrap()
        .with_task(Task::IllConditioned {
            condition_number: 1e4,
            rank: 4,
            coefficient_scale: 1.0,
            noise_scale: 0.0,
        })
        .unwrap()
        .generate();
    assert_eq!(conditioned.truth().rank(), Some(4));
    assert_eq!(conditioned.truth().coefficients().unwrap().len(), 6);
    assert_eq!(conditioned.truth().conditional_mean().unwrap().len(), 32);
}

/// The caller-owned target form writes the values the allocating one returns.
#[test]
fn the_caller_owned_target_form_matches_the_allocating_one_and_reuses_its_buffer() {
    let recipe = Recipe::seeded(256, 5, 12)
        .unwrap()
        .with_task(Task::NonlinearRegression {
            kind: NonlinearKind::Interaction,
            noise_scale: 0.1,
        })
        .unwrap();
    let allocated = recipe.target_values().unwrap();

    let mut buffer = vec![f32::MAX; 4_000];
    recipe.target_values_into(&mut buffer);
    assert_eq!(buffer, allocated);

    // A second fill replaces the contents rather than appending to them.
    let shorter = Recipe::seeded(16, 5, 12)
        .unwrap()
        .with_task(Task::NonlinearRegression {
            kind: NonlinearKind::Interaction,
            noise_scale: 0.1,
        })
        .unwrap();
    shorter.target_values_into(&mut buffer);
    assert_eq!(buffer.len(), 16);
    assert_eq!(buffer, shorter.target_values().unwrap());

    // The allocation survives, which is the whole reason the form exists.
    let capacity = buffer.capacity();
    recipe.target_values_into(&mut buffer);
    assert_eq!(buffer.capacity(), capacity);

    // A classification task reports its labels as `0.0` and `1.0`.
    let binary = Recipe::seeded(64, 5, 12)
        .unwrap()
        .with_task(Task::LinearBinary {
            informative: 2,
            separation: 2.0,
            prevalence: 0.5,
        })
        .unwrap();
    binary.target_values_into(&mut buffer);
    assert!(buffer.iter().all(|&value| value == 0.0 || value == 1.0));
    assert_eq!(
        buffer,
        binary_labels(&binary.generate())
            .iter()
            .map(|&label| f32::from(label))
            .collect::<Vec<_>>()
    );

    // A recipe with no task has no targets, and leaves the buffer empty rather
    // than filling it with zeros a caller could mistake for a target.
    let bare = Recipe::seeded(64, 5, 12).unwrap();
    bare.target_values_into(&mut buffer);
    assert!(buffer.is_empty());
    assert_eq!(bare.target_values(), None);
}

/// A conditioning dial scales a draw it did not change.
///
/// `condition_number` is a dial that reaches the design matrix, which is the one
/// shape of dial that byte identity cannot express. What holds instead is
/// stated here directly: the two recipes share a stream, so the *drawn* design
/// and the drawn coefficients are the same values, and the conditioned design is
/// exactly the crate's own column scaling applied to the unconditioned one.
///
/// The scaling function is the family's own rather than a restatement of it,
/// deliberately. What is under test is that the draw underneath the dial did not
/// move; whether `scale_columns` computes the right scales is
/// `the_realized_condition_number_is_within_a_factor_of_the_request`'s question,
/// and answering it twice in two spellings would only mean the spellings agree.
#[test]
fn a_conditioning_dial_scales_a_fixed_draw() {
    const ROWS: usize = 64;
    const COLUMNS: usize = 6;
    const CONDITION: f32 = 1.0e4;

    let unconditioned = Recipe::seeded(ROWS, COLUMNS, 7)
        .unwrap()
        .with_task(Task::IllConditioned {
            // One leaves the design as the source drew it, so this recipe's
            // design *is* the draw the dial is applied to.
            condition_number: 1.0,
            rank: COLUMNS,
            coefficient_scale: 1.0,
            noise_scale: 0.1,
        })
        .unwrap();
    let conditioned = Recipe::seeded(ROWS, COLUMNS, 7)
        .unwrap()
        .with_task(Task::IllConditioned {
            condition_number: CONDITION,
            rank: COLUMNS,
            coefficient_scale: 1.0,
            noise_scale: 0.1,
        })
        .unwrap();

    assert_eq!(unconditioned.stream_digest(), conditioned.stream_digest());
    let (flat, steep) = (unconditioned.generate(), conditioned.generate());
    assert_eq!(
        flat.truth().coefficients(),
        steep.truth().coefficients(),
        "a conditioning sweep redrew the coefficients"
    );

    let mut expected = flat.features().as_slice().to_vec();
    super::task::scale_columns(
        ROWS,
        COLUMNS,
        f64::from(CONDITION).log10() as f32,
        &mut expected,
    );
    assert_eq!(steep.features().as_slice(), expected.as_slice());

    // The leading column is scaled by exactly one, so byte identity does hold
    // there — the part of the design the dial does not reach is untouched rather
    // than merely close.
    for row in 0..ROWS {
        assert_eq!(
            steep.features().get(row, 0),
            flat.features().get(row, 0),
            "row {row}"
        );
    }
}

/// Bayes accuracy climbs with `separation` and falls with `prevalence`, step by
/// step.
///
/// This is the measurement the partition exists to make possible, and it is the
/// one that surfaced the defect. Bayes accuracy is a property of the *problem* —
/// the mean of `max(p, 1 - p)` over the recorded probabilities — so a ladder over
/// one dial reports what the dial did, with no estimator in the way. Both
/// ladders are strict: a knob whose whole purpose is to order the difficulty of
/// a family must produce an ordering.
///
/// # What it read before the dials left the stream digest
///
/// Every step of the separation ladder redrew the coefficients, so the ladder
/// measured the gap between unrelated draws. At `20000 x 8`, seed `31`, four
/// informative columns, it read:
///
/// | separation | 0.9 | 1.0 | 1.1 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
/// |---|---|---|---|---|---|---|---|---|
/// | before | 0.6198 | 0.5543 | 0.6707 | 0.6409 | 0.7195 | 0.7610 | 0.7280 | 0.7960 |
/// | after | 0.5976 | 0.6076 | 0.6173 | 0.6540 | 0.6939 | 0.7278 | 0.7563 | 0.8007 |
///
/// Three reversals before, none after, and the largest reversal — `0.0655`
/// between `0.9` and `1.0` — was six times the `0.0100` the knob is worth across
/// that interval. The prevalence ladder happened to be monotone before as well,
/// because a marginal rate dominates the accuracy of a rare-class problem, and
/// was confounded all the same: the coefficients behind each rung were a
/// different draw.
#[test]
fn bayes_accuracy_is_monotone_in_the_binary_dials() {
    const ROWS: usize = 20_000;
    const COLUMNS: usize = 8;
    const SEED: u64 = 31;

    let separated = |separation: f32| {
        bayes_accuracy(
            &Recipe::seeded(ROWS, COLUMNS, SEED)
                .unwrap()
                .with_task(Task::LinearBinary {
                    informative: 4,
                    separation,
                    prevalence: 0.5,
                })
                .unwrap()
                .generate(),
        )
    };
    let mut previous = f64::NEG_INFINITY;
    for separation in [0.9_f32, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0, 4.0] {
        let accuracy = separated(separation);
        assert!(
            accuracy > previous,
            "separation {separation} read {accuracy}, below the {previous} before it"
        );
        previous = accuracy;
    }

    // A prevalence away from a half is an easier problem, because predicting the
    // majority class is already right that often. The ladder therefore falls,
    // and at `0.05` it is pinned near `0.95` by the base rate alone.
    let prevalent = |prevalence: f32| {
        bayes_accuracy(
            &Recipe::seeded(ROWS, COLUMNS, SEED)
                .unwrap()
                .with_task(Task::LinearBinary {
                    informative: 4,
                    separation: 2.0,
                    prevalence,
                })
                .unwrap()
                .generate(),
        )
    };
    let mut previous = f64::INFINITY;
    for prevalence in [0.05_f32, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let accuracy = prevalent(prevalence);
        assert!(
            accuracy < previous,
            "prevalence {prevalence} read {accuracy}, above the {previous} before it"
        );
        previous = accuracy;
    }
}
