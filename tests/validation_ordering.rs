//! Shape validation must precede allocation and training work.
//!
//! `CLAUDE.md` states the rule as *"keep validation at public boundaries and
//! make invalid shapes fail before allocation or training work begins."* An
//! assertion that a rejected call returns the right error does not test that
//! rule: code which allocates a gigabyte and then errors satisfies it exactly.
//! So every test here measures the *ordering* directly — a rejected call must
//! reach its error having asked the allocator for nothing at all.
//!
//! The meter is a counting global allocator, the same technique
//! `tests/inspection_allocation.rs` and `tests/artifact_hardening.rs` use. It
//! is thread-local rather than process-global (as in `artifact_hardening.rs`)
//! so the tests in this binary can run in parallel without interfering, and it
//! counts allocator *calls* rather than bytes because the claim being made is
//! "nothing was allocated", not "not much was allocated".
//!
//! Every measurement is preceded by an identical *valid* call outside the
//! meter. That warm-up is not decoration: it forces any lazily initialized
//! global on the path to be paid for before counting starts, so a non-zero
//! count can only come from the rejected call itself.

use ferricml::api::{Classifier, ProbabilisticClassifier, Regressor, Transformer};
use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets};
use ferricml::dummy::{
    DummyClassifier, DummyClassifierParams, DummyRegressor, DummyRegressorParams,
};
use ferricml::ensemble::{MaxFeatures, RandomForestClassifier, RandomForestClassifierParams};
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseError, PairwiseLinearRanker, PairwiseLinearRankerParams,
    PairwiseObservation,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// ---------------------------------------------------------------------------
// Allocation meter
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Meter {
    armed: bool,
    calls: usize,
}

impl Meter {
    const IDLE: Self = Self {
        armed: false,
        calls: 0,
    };
}

thread_local! {
    /// Per-thread so two `#[test]` functions in this binary cannot interfere.
    static METER: Cell<Meter> = const { Cell::new(Meter::IDLE) };
}

fn record() {
    let _ = METER.try_with(|cell| {
        let mut meter = cell.get();
        if meter.armed {
            meter.calls += 1;
            cell.set(meter);
        }
    });
}

struct CountingAllocator;

// SAFETY: every method forwards to the system allocator unchanged. The counter
// only observes calls, and it holds no allocation of its own (a const-init
// `Cell<Meter>` needs neither lazy initialization nor a destructor), so
// observing cannot re-enter the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `operation` and reports how many times it called the allocator.
fn allocations(operation: impl FnOnce()) -> usize {
    METER.with(|cell| {
        cell.set(Meter {
            armed: true,
            calls: 0,
        })
    });
    operation();
    METER.with(|cell| {
        let meter = cell.get();
        cell.set(Meter::IDLE);
        meter.calls
    })
}

/// Asserts that `rejected` reaches its error without allocating.
///
/// `accepted` runs first, unmeasured, so lazily initialized state on the same
/// code path cannot be mistaken for the rejected call's own allocation.
#[track_caller]
fn rejects_before_allocating(accepted: impl FnOnce(), rejected: impl FnOnce()) {
    accepted();
    let calls = allocations(rejected);
    assert_eq!(
        calls, 0,
        "a rejected call allocated {calls} time(s) before reporting its error"
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Two columns wide, which is the fitted width of every model below.
fn fitted_width_data() -> DenseMatrix {
    DenseMatrix::new((0..24).map(|value| value as f32).collect(), 12, 2).unwrap()
}

/// Three columns wide, so every model below must refuse it.
fn wrong_width_data() -> DenseMatrix {
    DenseMatrix::new((0..24).map(|value| value as f32).collect(), 8, 3).unwrap()
}

fn dummy_classifier() -> DummyClassifier {
    let targets = BinaryTargets::new((0..12).map(|row| u8::from(row >= 6)).collect()).unwrap();
    DummyClassifier::fit(
        &fitted_width_data().as_view(),
        &targets,
        DummyClassifierParams,
    )
    .unwrap()
}

fn dummy_regressor() -> DummyRegressor {
    let targets = RegressionTargets::new((0..12).map(|row| row as f32).collect()).unwrap();
    DummyRegressor::fit(
        &fitted_width_data().as_view(),
        &targets,
        DummyRegressorParams,
    )
    .unwrap()
}

fn standard_scaler() -> StandardScaler {
    StandardScaler::fit(
        &fitted_width_data().as_view(),
        StandardScalerParams::default(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Audit finding 15 — the allocating trait defaults
// ---------------------------------------------------------------------------
//
// Each default body builds the return buffer and then delegates to the `_into`
// primitive that owns the width check. These call the trait methods explicitly
// rather than the inherent ones, so the default body is what is measured even
// where a concrete type also offers an inherent method of the same name.

#[test]
fn classifier_predict_default_validates_before_allocating() {
    let model = dummy_classifier();
    let fitted = fitted_width_data();
    let wrong = wrong_width_data();
    rejects_before_allocating(
        || {
            Classifier::predict(&model, &fitted.as_view()).unwrap();
        },
        || {
            assert!(Classifier::predict(&model, &wrong.as_view()).is_err());
        },
    );
}

#[test]
fn probabilistic_classifier_predict_proba_default_validates_before_allocating() {
    let model = dummy_classifier();
    let fitted = fitted_width_data();
    let wrong = wrong_width_data();
    rejects_before_allocating(
        || {
            ProbabilisticClassifier::predict_proba(&model, &fitted.as_view()).unwrap();
        },
        || {
            assert!(ProbabilisticClassifier::predict_proba(&model, &wrong.as_view()).is_err());
        },
    );
}

#[test]
fn probabilistic_classifier_predict_class_proba_default_validates_before_allocating() {
    let model = dummy_classifier();
    let fitted = fitted_width_data();
    let wrong = wrong_width_data();
    rejects_before_allocating(
        || {
            ProbabilisticClassifier::predict_class_proba(&model, &fitted.as_view(), 1).unwrap();
        },
        || {
            assert!(
                ProbabilisticClassifier::predict_class_proba(&model, &wrong.as_view(), 1).is_err()
            );
        },
    );
}

#[test]
fn regressor_predict_default_validates_before_allocating() {
    let model = dummy_regressor();
    let fitted = fitted_width_data();
    let wrong = wrong_width_data();
    rejects_before_allocating(
        || {
            Regressor::predict(&model, &fitted.as_view()).unwrap();
        },
        || {
            assert!(Regressor::predict(&model, &wrong.as_view()).is_err());
        },
    );
}

#[test]
fn transformer_transform_default_validates_before_allocating() {
    let model = standard_scaler();
    let fitted = fitted_width_data();
    let wrong = wrong_width_data();
    rejects_before_allocating(
        || {
            Transformer::transform(&model, &fitted.as_view()).unwrap();
        },
        || {
            assert!(Transformer::transform(&model, &wrong.as_view()).is_err());
        },
    );
}

/// The hoisted check must not change *which* error a rejected call reports.
///
/// The ordering tests above would still pass if the defaults had started
/// reporting a different variant, so the identity of the error is pinned
/// separately.
#[test]
fn the_hoisted_width_check_reports_the_same_error_the_into_form_reports() {
    use ferricml::api::ModelError;

    let wrong = wrong_width_data();
    let expected = ModelError::FeatureDimension {
        expected: 2,
        actual: 3,
    };

    let classifier = dummy_classifier();
    assert_eq!(
        Classifier::predict(&classifier, &wrong.as_view()),
        Err(expected.clone())
    );
    assert_eq!(
        ProbabilisticClassifier::predict_proba(&classifier, &wrong.as_view()),
        Err(expected.clone())
    );
    assert_eq!(
        ProbabilisticClassifier::predict_class_proba(&classifier, &wrong.as_view(), 1),
        Err(expected.clone())
    );
    assert_eq!(
        Regressor::predict(&dummy_regressor(), &wrong.as_view()),
        Err(expected.clone())
    );
    assert_eq!(
        Transformer::transform(&standard_scaler(), &wrong.as_view()).err(),
        Some(expected)
    );
}

// ---------------------------------------------------------------------------
// Audit finding E4 — the forest classifier's three prediction branches
// ---------------------------------------------------------------------------
//
// `RandomForestClassifier::predict` (and `ExtraTreesClassifier::predict`, the
// same macro-generated body) branches on the fitted forest shape. The
// single-class branch validated before allocating; the binary and multiclass
// branches allocated first. All three must now agree.

fn forest_params() -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(4)
        .with_max_features(MaxFeatures::All)
        .with_random_state(7)
}

/// A binary fit, which is the only shape that reaches the two scalar branches.
fn binary_fit(labels: Vec<u8>) -> RandomForestClassifier {
    let targets = BinaryTargets::new(labels).unwrap();
    RandomForestClassifier::fit(&fitted_width_data().as_view(), &targets, forest_params()).unwrap()
}

/// One observed label, so `classes().len() == 1` and the constant branch runs.
fn single_class_forest() -> RandomForestClassifier {
    let model = binary_fit(vec![0; 12]);
    assert_eq!(model.classes().len(), 1);
    model
}

/// Both labels observed, so the averaged-score branch runs.
fn binary_forest() -> RandomForestClassifier {
    let model = binary_fit((0..12).map(|row| u8::from(row >= 6)).collect());
    assert_eq!(model.classes().len(), 2);
    model
}

/// Three labels through the multiclass entry point, so the last branch runs.
fn multiclass_forest() -> RandomForestClassifier {
    let targets = ClassTargets::new((0..12).map(|row| (row % 3) as u8).collect()).unwrap();
    let model = RandomForestClassifier::fit_multiclass(
        &fitted_width_data().as_view(),
        &targets,
        forest_params(),
    )
    .unwrap();
    assert_eq!(model.classes().len(), 3);
    model
}

#[test]
fn every_forest_predict_branch_validates_before_allocating() {
    let fitted = fitted_width_data();
    let wrong = wrong_width_data();
    for (branch, model) in [
        ("single-class", single_class_forest()),
        ("binary", binary_forest()),
        ("multiclass", multiclass_forest()),
    ] {
        model.predict(&fitted.as_view()).unwrap();
        let calls = allocations(|| {
            assert!(
                model.predict(&wrong.as_view()).is_err(),
                "{branch} branch accepted a wrong-width batch"
            );
        });
        assert_eq!(
            calls, 0,
            "the {branch} branch allocated {calls} time(s) before reporting its error"
        );
    }
}

/// All three branches must also report the *same* error, which is the part of
/// E4 that "they now agree" actually means.
#[test]
fn every_forest_predict_branch_reports_the_same_width_error() {
    use ferricml::api::ModelError;

    let wrong = wrong_width_data();
    let expected = ModelError::FeatureDimension {
        expected: 2,
        actual: 3,
    };
    for model in [single_class_forest(), binary_forest(), multiclass_forest()] {
        assert_eq!(model.predict(&wrong.as_view()), Err(expected.clone()));
    }
}

// ---------------------------------------------------------------------------
// Audit finding E3 — the pairwise ranker copies and sorts before validating
// ---------------------------------------------------------------------------
//
// `expand_observations` allocated a full canonicalized copy of the batch and
// sorted it, and only then checked each pair index against the item count and
// the total pair weight. A sort is training work by any reading. Because the
// sort runs on the copy, an assertion that the copy never happens also
// establishes that the sort never runs.

/// Large enough that the sort allocates too, not only the copy.
const PAIR_BATCH: usize = 64;

fn ranker_items() -> DenseMatrix {
    DenseMatrix::new((0..32).map(|value| value as f32).collect(), 32, 1).unwrap()
}

/// A batch every one of whose pairs is in bounds and carries positive weight.
fn valid_observations() -> Vec<PairwiseObservation> {
    (0..PAIR_BATCH)
        .map(|index| {
            PairwiseObservation::new(
                PairIndex::new((index % 31) + 1, 0).unwrap(),
                PairOutcome::LeftPreferred,
                1.0,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn the_pairwise_ranker_validates_pair_indices_before_copying_the_batch() {
    let items = ranker_items();
    let accepted = valid_observations();

    // One index past the last item, in the middle of an otherwise valid batch,
    // so nothing about its position makes it cheap to notice.
    let mut rejected = valid_observations();
    rejected[PAIR_BATCH / 2] = PairwiseObservation::new(
        PairIndex::new(items.rows(), 0).unwrap(),
        PairOutcome::Tie,
        1.0,
    )
    .unwrap();

    PairwiseLinearRanker::fit(
        &items.as_view(),
        &accepted,
        PairwiseLinearRankerParams::default(),
    )
    .unwrap();

    let calls = allocations(|| {
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items.as_view(),
                &rejected,
                PairwiseLinearRankerParams::default(),
            )
            .err(),
            Some(PairwiseError::PairIndexOutOfBounds {
                pair: PAIR_BATCH / 2,
                item: 32,
                items: 32,
            })
        );
    });
    assert_eq!(
        calls, 0,
        "an out-of-bounds pair index cost {calls} allocation(s) before being reported"
    );
}

#[test]
fn the_pairwise_ranker_validates_the_total_weight_before_copying_the_batch() {
    let items = ranker_items();
    let accepted = valid_observations();

    // Every pair in bounds, so this reaches the weight check specifically.
    let rejected = (0..PAIR_BATCH)
        .map(|index| {
            PairwiseObservation::new(
                PairIndex::new((index % 31) + 1, 0).unwrap(),
                PairOutcome::LeftPreferred,
                0.0,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    PairwiseLinearRanker::fit(
        &items.as_view(),
        &accepted,
        PairwiseLinearRankerParams::default(),
    )
    .unwrap();

    let calls = allocations(|| {
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items.as_view(),
                &rejected,
                PairwiseLinearRankerParams::default(),
            )
            .err(),
            Some(PairwiseError::ZeroTotalPairWeight)
        );
    });
    assert_eq!(
        calls, 0,
        "a zero-weight batch cost {calls} allocation(s) before being reported"
    );
}
