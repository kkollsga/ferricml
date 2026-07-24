//! Allocation bound for permutation importance.
//!
//! The workspace is allocated once per call, so the cost of extra repeats must
//! be scoring alone. A counting allocator makes that falsifiable: the same run
//! at two very different repeat counts must allocate exactly the same number
//! of times. This file holds one test because the counter is process-global.

use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::ensemble::{MaxFeatures, RandomForestRegressor, RandomForestRegressorParams};
use ferricml::inspection::{PermutationImportanceParams, permutation_importance_regressor_into};
use ferricml::model_selection::RegressionScorer;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

struct CountingAllocator;

// SAFETY: every method forwards to the system allocator unchanged; the counter
// only observes calls.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count(operation: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    operation();
    COUNTING.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn permutation_workspace_is_allocated_once_regardless_of_repeat_count() {
    let rows = 96;
    let columns = 4;
    let mut values = Vec::with_capacity(rows * columns);
    let mut targets = Vec::with_capacity(rows);
    for row in 0..rows {
        let signal = ((row * 37 % 61) as f32 / 30.0) - 1.0;
        values.extend_from_slice(&[signal, (row % 7) as f32, 1.0, -signal]);
        targets.push(5.0 * signal);
    }
    let data = DenseMatrix::new(values, rows, columns).unwrap();
    let targets = RegressionTargets::new(targets).unwrap();
    let model = RandomForestRegressor::fit(
        &data.as_view(),
        &targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(8)
            .with_max_features(MaxFeatures::All)
            .with_random_state(2),
    )
    .unwrap();

    let mut means = vec![0.0; columns];
    let mut std_devs = vec![0.0; columns];
    let mut measure = |n_repeats: usize| {
        count(|| {
            permutation_importance_regressor_into(
                &model,
                &data.as_view(),
                &targets,
                RegressionScorer::MeanSquaredError,
                PermutationImportanceParams::default()
                    .with_n_repeats(n_repeats)
                    .with_random_state(5),
                &mut means,
                &mut std_devs,
            )
            .unwrap();
        })
    };

    let few = measure(2);
    let many = measure(64);
    assert!(few > 0, "the counting allocator observed nothing");
    assert!(few < 32, "workspace allocation grew unexpectedly: {few}");
    assert_eq!(
        few, many,
        "permutation importance allocated more work per repeat: {few} at 2 repeats, {many} at 64"
    );
}
