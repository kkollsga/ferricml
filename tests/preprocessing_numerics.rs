//! Inverse round-trips across the scaler family, over the degenerate regions.
//!
//! Every invertible scaler documents its exactness *conditionally*: exact by
//! construction when both statistics are disabled and on a degenerate column
//! whose divisor was substituted to one, and "exact only when the arithmetic
//! happens to be" everywhere else. Both halves of that are checkable and
//! neither was being checked over anything but hand-written fixtures.
//!
//! This binary checks the stated conditions as *conditions* — a configuration
//! the documentation calls exact must round-trip bit for bit, and the
//! substituted divisor must actually be one where the claim depends on it — and
//! holds the remaining cases to a stated envelope derived from the affine map
//! each scaler publishes rather than from a tolerance chosen to make the test
//! pass.
//!
//! # The envelope
//!
//! Each of the four numeric scalers is one affine map per column,
//! `T = a X + b`, whose inverse is `X = (T - b) / a`. Rounding the forward
//! result to `f32` costs at most half an `f32` ulp, and dividing that error by
//! `a` is what the caller gets back, so
//!
//! ```text
//! |round_trip(X) - X| <= K u (|X| + |T| / |a|) + K d / |a|
//! ```
//!
//! with `u = 2^-24` the `f32` unit roundoff, `d = 2^-150` the largest absolute
//! error of an `f32` subnormal, and `K` a small constant. The sweep records the
//! worst observed `K` rather than only asserting a bound.
//!
//! Sizes come from `FERRICML_ORACLE_SWEEP`:
//!
//! ```text
//! FERRICML_ORACLE_SWEEP=2000 cargo test --release --test preprocessing_numerics -- --nocapture
//! ```

use ferricml::api::ModelError;
use ferricml::data::{DenseMatrix, SampleWeights};
use ferricml::preprocessing::{
    FunctionTransformer, FunctionTransformerParams, MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler,
    MinMaxScalerParams, RobustScaler, RobustScalerParams, StandardScaler, StandardScalerParams,
};

#[path = "support/rng.rs"]
mod rng;

use rng::TestRng;

/// `f32` unit roundoff.
const U: f64 = 1.0 / (1_u64 << 24) as f64;
/// Largest absolute error of an `f32` subnormal: half the smallest subnormal.
const D: f64 = 3.0e-46;
/// The envelope multiple this suite holds the inexact cases to.
///
/// A sweep of 2,000 cases — 2.1 million round-tripped values — reached 0.978,
/// so the derivation above is tight and this is a factor of two of headroom
/// rather than a number chosen to accommodate the observation.
const ENVELOPE_K: f64 = 2.0;

const DEFAULT_CASES: usize = 120;

fn cases() -> usize {
    std::env::var("FERRICML_ORACLE_SWEEP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

/// The shapes a column can take, chosen so the degenerate regions the
/// documentation talks about are actually visited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnKind {
    /// Ordinary spread around zero.
    Ordinary,
    /// Every value identical: zero variance, zero range, zero quantile spread.
    Constant,
    /// Every value exactly zero, which is the max-abs degeneracy.
    AllZero,
    /// A spread thirty orders of magnitude below the values themselves.
    NearZeroSpread,
    /// Values at the top of the `f32` range.
    HugeMagnitude,
    /// Values at the bottom of the normal `f32` range.
    TinyMagnitude,
    /// Two distinct values, which gives a robust scaler a zero interquartile
    /// spread whenever the split falls inside one of them.
    TwoLevels,
    /// A large offset carrying a small spread, the cancellation region.
    OffsetSpread,
}

const KINDS: [ColumnKind; 8] = [
    ColumnKind::Ordinary,
    ColumnKind::Constant,
    ColumnKind::AllZero,
    ColumnKind::NearZeroSpread,
    ColumnKind::HugeMagnitude,
    ColumnKind::TinyMagnitude,
    ColumnKind::TwoLevels,
    ColumnKind::OffsetSpread,
];

fn column_values(kind: ColumnKind, rows: usize, rng: &mut TestRng) -> Vec<f32> {
    match kind {
        ColumnKind::Ordinary => (0..rows).map(|_| rng.range_f32(-3.0, 3.0)).collect(),
        ColumnKind::Constant => {
            let value = rng.range_f32(-100.0, 100.0);
            vec![value; rows]
        }
        ColumnKind::AllZero => vec![0.0; rows],
        ColumnKind::NearZeroSpread => {
            let base = rng.range_f32(-10.0, 10.0);
            (0..rows)
                .map(|_| base + rng.range_f32(-1.0, 1.0) * 1.0e-30)
                .collect()
        }
        ColumnKind::HugeMagnitude => (0..rows)
            .map(|_| rng.range_f32(-1.0, 1.0) * 1.0e30)
            .collect(),
        ColumnKind::TinyMagnitude => (0..rows)
            .map(|_| rng.range_f32(-1.0, 1.0) * 1.0e-30)
            .collect(),
        ColumnKind::TwoLevels => {
            let (low, high) = (rng.range_f32(-5.0, 0.0), rng.range_f32(0.0, 5.0));
            (0..rows)
                .map(|_| if rng.flag() { low } else { high })
                .collect()
        }
        ColumnKind::OffsetSpread => {
            let base = rng.range_f32(1.0e5, 1.0e7);
            (0..rows).map(|_| base + rng.range_f32(-0.5, 0.5)).collect()
        }
    }
}

struct Case {
    data: DenseMatrix,
    kinds: Vec<ColumnKind>,
    weights: SampleWeights,
}

fn case(seed: u64) -> Case {
    let mut rng = TestRng::new(seed);
    let rows = rng.between(4, 40);
    let columns = rng.between(1, 5);
    let kinds = (0..columns)
        .map(|_| KINDS[rng.below(KINDS.len())])
        .collect::<Vec<_>>();
    let per_column = kinds
        .iter()
        .map(|&kind| column_values(kind, rows, &mut rng))
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in &per_column {
            values.push(column[row]);
        }
    }
    let weights = SampleWeights::new((0..rows).map(|_| rng.range_f32(0.25, 4.0)).collect())
        .expect("positive finite weights");
    Case {
        data: DenseMatrix::new(values, rows, columns).expect("generated shape"),
        kinds,
        weights,
    }
}

/// Whether every row of a column carries the same bits.
///
/// This is the condition that actually makes a substituted divisor produce an
/// exact round trip: with a constant column the centre equals every value, so
/// the centred value is exactly zero and adding the centre back recovers the
/// input. A degeneracy criterion that does *not* imply constancy — the robust
/// scaler's zero interquartile spread — does not carry that consequence.
fn column_is_constant(data: &DenseMatrix, column: usize) -> bool {
    let first = data.get(0, column).expect("in bounds").to_bits();
    (1..data.rows()).all(|row| data.get(row, column).expect("in bounds").to_bits() == first)
}

/// Running record for one class of column.
#[derive(Default)]
struct Class {
    values: usize,
    inexact: usize,
    worst_absolute: f64,
    worst_multiple: f64,
}

impl Class {
    /// Records one round-tripped value against its envelope.
    fn record(&mut self, original: f32, transformed: f32, recovered: f32, slope: f64) {
        self.values += 1;
        let error = (f64::from(recovered) - f64::from(original)).abs();
        self.worst_absolute = self.worst_absolute.max(error);
        if recovered.to_bits() != original.to_bits() {
            self.inexact += 1;
        }
        let budget = U * (f64::from(original).abs() + f64::from(transformed).abs() / slope.abs())
            + D / slope.abs();
        if budget > 0.0 {
            self.worst_multiple = self.worst_multiple.max(error / budget);
        } else {
            assert_eq!(
                recovered.to_bits(),
                original.to_bits(),
                "a zero-width envelope demands an exact round trip"
            );
        }
    }

    fn report(&self, name: &str) {
        println!(
            "  {name}: {} values, {} inexact, worst |delta| = {:e}, worst envelope multiple = {:.3}",
            self.values, self.inexact, self.worst_absolute, self.worst_multiple
        );
    }
}

/// A column's round trip together with the affine slope of its forward map.
struct RoundTrip {
    /// `a` in `T = a X + b`, taken from the scaler's published parameters.
    slope: f64,
    /// Whether the documentation calls this column's round trip exact.
    claimed_exact: bool,
}

/// Asserts a class of columns the documentation calls exact, and returns how
/// many values it checked.
fn assert_exact(
    label: &str,
    original: &DenseMatrix,
    recovered: &DenseMatrix,
    column: usize,
) -> usize {
    let columns = original.columns();
    let mut checked = 0;
    for row in 0..original.rows() {
        let want = original.get(row, column).expect("in bounds");
        let got = recovered.get(row, column).expect("in bounds");
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "{label}: column {column} of {columns} is documented as an exact round trip, \
             but {want} came back as {got}"
        );
        checked += 1;
    }
    checked
}

/// Runs one scaler over one case and folds the result into the two classes.
#[allow(clippy::too_many_arguments)]
fn measure(
    label: &str,
    data: &DenseMatrix,
    transformed: &DenseMatrix,
    recovered: &DenseMatrix,
    per_column: &[RoundTrip],
    exact: &mut Class,
    inexact: &mut Class,
    exact_values: &mut usize,
) {
    for (column, trip) in per_column.iter().enumerate() {
        if trip.claimed_exact {
            *exact_values += assert_exact(label, data, recovered, column);
            for row in 0..data.rows() {
                exact.record(
                    data.get(row, column).expect("in bounds"),
                    transformed.get(row, column).expect("in bounds"),
                    recovered.get(row, column).expect("in bounds"),
                    trip.slope,
                );
            }
        } else {
            for row in 0..data.rows() {
                inexact.record(
                    data.get(row, column).expect("in bounds"),
                    transformed.get(row, column).expect("in bounds"),
                    recovered.get(row, column).expect("in bounds"),
                    trip.slope,
                );
            }
        }
    }
}

#[test]
fn every_scaler_inverse_holds_its_exactness_conditions_and_its_envelope() {
    let mut exact = Class::default();
    let mut inexact = Class::default();
    let mut exact_values = 0_usize;
    let mut refusals = 0_usize;
    let mut fits = 0_usize;
    let mut degenerate_columns = 0_usize;
    let mut identity_configurations = 0_usize;
    let mut zero_spread_varying = 0_usize;

    for seed in 0..cases() as u64 {
        let case = case(0x9ca1_0005_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        let view = case.data.as_view();

        // ---- StandardScaler, both toggles, weighted and unweighted --------
        for with_mean in [true, false] {
            for with_std in [true, false] {
                for weighted in [false, true] {
                    let params = StandardScalerParams::default()
                        .with_mean(with_mean)
                        .with_std(with_std);
                    let Ok(scaler) = (if weighted {
                        StandardScaler::fit_weighted(&view, &case.weights, params)
                    } else {
                        StandardScaler::fit(&view, params)
                    }) else {
                        refusals += 1;
                        continue;
                    };
                    fits += 1;
                    let Ok(transformed) = scaler.transform(&view) else {
                        refusals += 1;
                        continue;
                    };
                    let Ok(recovered) = scaler.inverse_transform(&transformed.as_view()) else {
                        refusals += 1;
                        continue;
                    };
                    if !with_mean && !with_std {
                        identity_configurations += 1;
                    }
                    let per_column = (0..case.data.columns())
                        .map(|column| {
                            let divisor = if with_std {
                                scaler.scales()[column]
                            } else {
                                1.0
                            };
                            let degenerate = scaler.variances()[column] == 0.0;
                            if degenerate {
                                if with_std {
                                    assert_eq!(
                                        scaler.scales()[column],
                                        1.0,
                                        "a zero-variance column must keep a divisor of one"
                                    );
                                }
                                // The exactness claim rests on this: a zero
                                // variance forces every value to equal the mean,
                                // so there is nothing for the round trip to lose.
                                assert!(
                                    column_is_constant(&case.data, column),
                                    "a zero-variance column that is not constant would \
                                     break the exactness claim, not merely this test"
                                );
                            }
                            RoundTrip {
                                slope: 1.0 / divisor,
                                // Documented exact: both statistics disabled,
                                // or a column whose divisor was substituted and
                                // which therefore has no mean to subtract off
                                // that is not already its own value.
                                claimed_exact: (!with_mean && !with_std) || degenerate,
                            }
                        })
                        .collect::<Vec<_>>();
                    degenerate_columns += per_column
                        .iter()
                        .filter(|trip| trip.claimed_exact && (with_mean || with_std))
                        .count();
                    measure(
                        "StandardScaler",
                        &case.data,
                        &transformed,
                        &recovered,
                        &per_column,
                        &mut exact,
                        &mut inexact,
                        &mut exact_values,
                    );
                }
            }
        }

        // ---- RobustScaler, both toggles -----------------------------------
        for with_centering in [true, false] {
            for with_scaling in [true, false] {
                let params = RobustScalerParams::default()
                    .with_centering(with_centering)
                    .with_scaling(with_scaling);
                let Ok(scaler) = RobustScaler::fit(&view, params) else {
                    refusals += 1;
                    continue;
                };
                fits += 1;
                let Ok(transformed) = scaler.transform(&view) else {
                    refusals += 1;
                    continue;
                };
                let Ok(recovered) = scaler.inverse_transform(&transformed.as_view()) else {
                    refusals += 1;
                    continue;
                };
                if !with_centering && !with_scaling {
                    identity_configurations += 1;
                }
                let mut degenerate_here = 0_usize;
                let per_column = (0..case.data.columns())
                    .map(|column| {
                        let divisor = if with_scaling {
                            scaler.scales()[column]
                        } else {
                            1.0
                        };
                        let degenerate = scaler.spreads()[column] == 0.0;
                        if degenerate {
                            assert_eq!(
                                scaler.scales()[column],
                                1.0,
                                "a zero-spread column must keep a divisor of one"
                            );
                            if !column_is_constant(&case.data, column) {
                                zero_spread_varying += 1;
                            }
                        }
                        RoundTrip {
                            slope: 1.0 / divisor,
                            // A substituted divisor alone is not enough here.
                            // See `a_zero_spread_robust_column_need_not_be_constant`.
                            claimed_exact: (!with_centering && !with_scaling)
                                || (degenerate && column_is_constant(&case.data, column)),
                        }
                    })
                    .collect::<Vec<_>>();
                degenerate_here += per_column
                    .iter()
                    .filter(|trip| trip.claimed_exact && (with_centering || with_scaling))
                    .count();
                degenerate_columns += degenerate_here;
                measure(
                    "RobustScaler",
                    &case.data,
                    &transformed,
                    &recovered,
                    &per_column,
                    &mut exact,
                    &mut inexact,
                    &mut exact_values,
                );
            }
        }

        // ---- MinMaxScaler, clipping off so the map stays invertible -------
        for range in [(0.0_f64, 1.0_f64), (-1.0, 1.0), (2.0, 7.5)] {
            let params = MinMaxScalerParams::default()
                .with_feature_range(range.0, range.1)
                .with_clip(false);
            let Ok(scaler) = MinMaxScaler::fit(&view, params) else {
                refusals += 1;
                continue;
            };
            fits += 1;
            let Ok(transformed) = scaler.transform(&view) else {
                refusals += 1;
                continue;
            };
            let Ok(recovered) = scaler.inverse_transform(&transformed.as_view()) else {
                refusals += 1;
                continue;
            };
            let per_column = (0..case.data.columns())
                .map(|column| RoundTrip {
                    slope: scaler.scales()[column],
                    // A zero range is only reachable from a constant column, so
                    // the substituted divisor does carry the claim here.
                    claimed_exact: column_is_constant(&case.data, column),
                })
                .collect::<Vec<_>>();
            measure(
                "MinMaxScaler",
                &case.data,
                &transformed,
                &recovered,
                &per_column,
                &mut exact,
                &mut inexact,
                &mut exact_values,
            );
        }

        // ---- MaxAbsScaler --------------------------------------------------
        let Ok(scaler) = MaxAbsScaler::fit(&view, MaxAbsScalerParams) else {
            refusals += 1;
            continue;
        };
        fits += 1;
        let Ok(transformed) = scaler.transform(&view) else {
            refusals += 1;
            continue;
        };
        let Ok(recovered) = scaler.inverse_transform(&transformed.as_view()) else {
            refusals += 1;
            continue;
        };
        let per_column = (0..case.data.columns())
            .map(|column| {
                let all_zero = case.kinds[column] == ColumnKind::AllZero;
                if all_zero {
                    assert_eq!(
                        scaler.scales()[column],
                        1.0,
                        "an all-zero column must keep a divisor of one"
                    );
                }
                RoundTrip {
                    slope: 1.0 / scaler.scales()[column],
                    claimed_exact: all_zero,
                }
            })
            .collect::<Vec<_>>();
        measure(
            "MaxAbsScaler",
            &case.data,
            &transformed,
            &recovered,
            &per_column,
            &mut exact,
            &mut inexact,
            &mut exact_values,
        );
    }

    println!("preprocessing round trips: {fits} fits, {refusals} refused batches");
    exact.report("documented exact");
    inexact.report("documented inexact");
    println!(
        "  {exact_values} values covered by an exactness claim, {degenerate_columns} of them \
         through a substituted divisor, {identity_configurations} identity configurations"
    );
    println!(
        "  {zero_spread_varying} robust columns had a zero interquartile spread while still \
         varying, which is the case the exactness paragraph does not cover"
    );

    assert_eq!(
        exact.inexact, 0,
        "{} values the documentation calls exact did not round-trip exactly",
        exact.inexact
    );
    assert!(
        inexact.worst_multiple <= ENVELOPE_K,
        "the inexact round trip reached {:.3} times its envelope",
        inexact.worst_multiple
    );
    assert!(exact_values > 0 && degenerate_columns > 0 && identity_configurations > 0);
    assert!(
        zero_spread_varying > 0,
        "the sweep never produced a zero-spread column that still varies, so it never \
         reached the region the robust scaler's exactness paragraph gets wrong"
    );
    // Non-vacuity: the inexact class must actually contain inexact values, or
    // the envelope is being checked against nothing.
    assert!(
        inexact.inexact > 0,
        "no value outside the documented-exact configurations was inexact, so the \
         envelope bound never had anything to bound"
    );
    assert!(
        inexact.worst_multiple > 0.0,
        "the worst envelope multiple was zero"
    );
}

fn double(value: f32) -> f32 {
    value * 2.0
}

fn halve(value: f32) -> f32 {
    value * 0.5
}

fn cube(value: f32) -> f32 {
    value * value * value
}

fn cube_root(value: f32) -> f32 {
    value.cbrt()
}

#[test]
fn a_function_transformer_applies_exactly_the_pair_it_was_given() {
    // The one thing this transformer guarantees is that it applies the supplied
    // functions and nothing else — it explicitly does not check that they
    // invert each other. So the claim under test is elementwise identity with
    // the caller's own functions, and the round trip is only exact for a pair
    // that actually is exact.
    let mut exact_round_trips = 0_usize;
    let mut inexact_round_trips = 0_usize;
    let mut applied = 0_usize;

    for seed in 0..cases() as u64 {
        let case = case(0x9cf7_0006_u64.wrapping_add(seed.wrapping_mul(0x9e37_79b9)));
        let view = case.data.as_view();
        for (forward, backward, exact) in [
            (double as fn(f32) -> f32, halve as fn(f32) -> f32, true),
            (cube as fn(f32) -> f32, cube_root as fn(f32) -> f32, false),
        ] {
            let params = FunctionTransformerParams::default()
                .with_func(forward)
                .with_inverse_func(backward);
            let transformer =
                FunctionTransformer::fit(&view, params).expect("a transformer that fits anything");
            let Ok(transformed) = transformer.transform(&view) else {
                continue;
            };
            // Applied exactly: every transformed value is the caller's own
            // function of the input, bit for bit.
            for row in 0..case.data.rows() {
                for column in 0..case.data.columns() {
                    let input = case.data.get(row, column).expect("in bounds");
                    assert_eq!(
                        transformed.get(row, column).expect("in bounds").to_bits(),
                        forward(input).to_bits(),
                        "the forward map was not applied exactly"
                    );
                    applied += 1;
                }
            }
            let Ok(recovered) = transformer.inverse_transform(&transformed.as_view()) else {
                continue;
            };
            for row in 0..case.data.rows() {
                for column in 0..case.data.columns() {
                    let intermediate = transformed.get(row, column).expect("in bounds");
                    assert_eq!(
                        recovered.get(row, column).expect("in bounds").to_bits(),
                        backward(intermediate).to_bits(),
                        "the inverse map was not applied exactly"
                    );
                    let original = case.data.get(row, column).expect("in bounds");
                    let same = recovered.get(row, column).expect("in bounds").to_bits()
                        == original.to_bits();
                    if exact {
                        assert!(
                            same,
                            "doubling and halving must round-trip exactly, {original} \
                             came back as {}",
                            recovered.get(row, column).expect("in bounds")
                        );
                        exact_round_trips += 1;
                    } else if !same {
                        inexact_round_trips += 1;
                    }
                }
            }
        }
    }

    println!(
        "function transformer: {applied} elementwise applications verified, \
         {exact_round_trips} exact round trips through a power-of-two pair, \
         {inexact_round_trips} inexact round trips through cube and cube root"
    );
    assert!(applied > 0 && exact_round_trips > 0);
    // Non-vacuity: an unfaithful pair must be visible as an inexact round trip,
    // or "exact" above is what every pair would produce.
    assert!(
        inexact_round_trips > 0,
        "cube and cube root round-tripped exactly everywhere, so the exactness claim \
         for the power-of-two pair is not distinguishing anything"
    );
}

#[test]
fn clipping_is_the_documented_exception_and_is_not_invertible() {
    // `MinMaxScaler` says clipping is a projection and that inverting a clipped
    // batch recovers the bound. That is a claim about losing information, and it
    // is worth pinning in the same place the exactness claims are.
    let fitted = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).expect("fixture");
    let scaler = MinMaxScaler::fit(
        &fitted.as_view(),
        MinMaxScalerParams::default()
            .with_feature_range(0.0, 1.0)
            .with_clip(true),
    )
    .expect("fit");
    let unseen = DenseMatrix::new(vec![-5.0, 8.0], 2, 1).expect("fixture");
    let transformed = scaler.transform(&unseen.as_view()).expect("transform");
    assert_eq!(transformed.as_slice(), &[0.0, 1.0]);
    let recovered = scaler
        .inverse_transform(&transformed.as_view())
        .expect("inverse");
    assert_eq!(
        recovered.as_slice(),
        &[0.0, 3.0],
        "a clipped value must invert to the bound it was clamped to"
    );

    // Without clipping the same batch survives the round trip inside the
    // envelope, which is what makes the paragraph above a statement about
    // clipping rather than about the scaler.
    let unclipped = MinMaxScaler::fit(
        &fitted.as_view(),
        MinMaxScalerParams::default()
            .with_feature_range(0.0, 1.0)
            .with_clip(false),
    )
    .expect("fit");
    let transformed = unclipped.transform(&unseen.as_view()).expect("transform");
    let recovered = unclipped
        .inverse_transform(&transformed.as_view())
        .expect("inverse");
    for (want, got) in unseen.as_slice().iter().zip(recovered.as_slice()) {
        assert!(
            (want - got).abs() <= 1.0e-5 * want.abs(),
            "unclipped round trip lost {want} as {got}"
        );
    }
}

#[test]
fn a_shape_mismatch_is_refused_by_both_inverse_forms() {
    // The envelope sweep only ever calls these with the right shapes, so the
    // refusal path needs its own statement.
    let data = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("fixture");
    let scaler =
        StandardScaler::fit(&data.as_view(), StandardScalerParams::default()).expect("fit");
    let narrow = DenseMatrix::new(vec![1.0, 2.0], 2, 1).expect("fixture");
    assert_eq!(
        scaler.inverse_transform(&narrow.as_view()).unwrap_err(),
        ModelError::FeatureDimension {
            expected: 2,
            actual: 1
        }
    );
    let mut short = [0.0_f32; 3];
    assert_eq!(
        scaler
            .inverse_transform_into(&data.as_view(), &mut short)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 4,
            actual: 3
        }
    );
}

#[test]
fn a_zero_spread_robust_column_need_not_be_constant() {
    // OPEN DIVERGENCE, found by the sweep above and reported 2026-07-26.
    //
    // `RobustScaler::inverse_transform_into` says the round trip is "exact by
    // construction ... on a degenerate column whose divisor was substituted to
    // one". That sentence is `StandardScaler`'s, and it is sound there because a
    // zero *variance* forces every value to equal the mean. This scaler's
    // degeneracy criterion is a zero *interquartile spread*, which does not:
    // both quartiles can land inside one repeated value while the column still
    // varies at its ends. The substituted divisor is then irrelevant, because
    // the error comes from centring, and the round trip loses a unit in the last
    // place.
    //
    // Nothing here is a floating-point defect — the value is inside the
    // envelope the sweep measures. The claim is what is wrong, and this test
    // pins the counterexample so the correction has something to be checked
    // against.
    let column = vec![
        -4.9017205_f32,
        -4.9017205,
        -4.9017205,
        -4.9017205,
        -4.9017205,
        4.49687,
    ];
    let data = DenseMatrix::new(column.clone(), 6, 1).expect("fixture");
    let scaler = RobustScaler::fit(&data.as_view(), RobustScalerParams::default()).expect("fit");

    // The degeneracy the paragraph names is present ...
    assert_eq!(scaler.spreads(), &[0.0]);
    assert_eq!(
        scaler.scales(),
        &[1.0],
        "the divisor was substituted to one"
    );
    // ... and the column is not constant, which is what the paragraph assumes.
    assert_ne!(column[0], column[5]);

    let transformed = scaler.transform(&data.as_view()).expect("transform");
    let recovered = scaler
        .inverse_transform(&transformed.as_view())
        .expect("inverse");

    // Every row inside the repeated block does round-trip exactly.
    for (row, &value) in column.iter().enumerate().take(5) {
        assert_eq!(
            recovered.get(row, 0).expect("in bounds").to_bits(),
            value.to_bits()
        );
    }
    // The row outside it does not.
    let original = column[5];
    let round_tripped = recovered.get(5, 0).expect("in bounds");
    assert_ne!(
        round_tripped.to_bits(),
        original.to_bits(),
        "the exactness paragraph now says a substituted divisor leaves the centring \
         error rather than removing it; if this row ever round-trips exactly, that \
         paragraph has become too weak and both should be tightened together"
    );
    assert_eq!(
        original.to_bits() - round_tripped.to_bits(),
        1,
        "the loss is one unit in the last place, not a larger defect"
    );
}
