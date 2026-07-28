//! The exchange reader's allocation oracle.
//!
//! An exchange container is untrusted input: it is a file, it outlives the
//! process that wrote it, and the whole point of the format is that something
//! other than this crate reads and writes beside it. Its array table declares
//! how long each array is, and a decoder that reserved storage from that
//! declaration before reading the bytes behind it would let a tiny file demand
//! an enormous allocation.
//!
//! That is not hypothetical here. Every decoder in `src/artifact/` once did
//! exactly that, and the measurement that found it is the reason this file
//! exists: a **148-byte** artifact reserved **32 MB**, a 216,000-fold
//! amplification, while still returning the correct typed error. No accept or
//! reject test could see it — the refusal was right, and only the cost was
//! wrong — so the fix (`ArtifactCursor::bounded_capacity`) came with an
//! allocation oracle rather than another refusal test. `ExchangeCursor` applies
//! the same rule to this format, and this file is that rule's oracle.
//!
//! # What is measured
//!
//! Peak *live* allocation during one `DatasetExchange::load`, against a budget
//! of [`ALLOC_BASE_BYTES`] plus [`ALLOC_INPUT_FACTOR`] times the two files'
//! combined length. Both numbers are close to what a genuine load measures —
//! `a_genuine_load_stays_inside_the_same_budget` holds them there — because a
//! budget set far above the truth is not an oracle.
//!
//! # Why the meter is a second copy
//!
//! `tests/artifact_hardening.rs` has one too. A `#[global_allocator]` is a
//! property of a whole test binary, so a shared one would either force the
//! allocator on every binary that includes `tests/support`, or couple this
//! file's numbers to a campaign whose budgets are calibrated to artifacts. The
//! duplication is thirty lines and it keeps the two measurements independent.
//!
//! # The oracle is proven live
//!
//! `the_oracle_reports_an_unbounded_reservation` reserves from the same
//! declared length the hostile manifests carry and requires the meter to report
//! it *over* budget. Without that, every assertion below would also pass if the
//! meter had quietly stopped counting.

use ferricml::datasets::{DatasetExchange, ExchangeError, Recipe};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;

// ---------------------------------------------------------------------------
// Peak-allocation meter
// ---------------------------------------------------------------------------

/// Allocation one load may always use, whatever the container's size.
///
/// A load allocates the manifest text, the array file, one owned `String` per
/// array name, the decoded arrays, and the paths it opened. Measured on
/// 2026-07-28: a genuine load of the 1,902-byte probe container peaks at 3,515
/// bytes, so this is a ceiling on the fixed part rather than a licence.
const ALLOC_BASE_BYTES: usize = 4 * 1024;

/// Multiple of the container's byte length a load may allocate beyond that
/// base.
///
/// A load holds the file's bytes and the values decoded from them at the same
/// time, and the decoded values are the same size as the bytes they came from,
/// so **two** is the floor rather than a choice. Three leaves room for the
/// manifest text and the array table beside them, and no room at all for an
/// amplification: the hostile cases below declare four billion `f32` values, so
/// believing any one of them would exceed this budget by six orders of
/// magnitude.
const ALLOC_INPUT_FACTOR: usize = 3;

#[derive(Clone, Copy)]
struct Meter {
    armed: bool,
    live: usize,
    peak: usize,
}

impl Meter {
    const IDLE: Self = Self {
        armed: false,
        live: 0,
        peak: 0,
    };
}

thread_local! {
    /// Per-thread, so two `#[test]` functions in this binary cannot interfere.
    static METER: Cell<Meter> = const { Cell::new(Meter::IDLE) };
}

fn record(delta: isize) {
    let _ = METER.try_with(|cell| {
        let mut meter = cell.get();
        if !meter.armed {
            return;
        }
        meter.live = meter.live.saturating_add_signed(delta);
        meter.peak = meter.peak.max(meter.live);
        cell.set(meter);
    });
}

struct PeakAllocator;

// SAFETY: every method forwards to the system allocator unchanged. The meter
// only observes sizes, and it holds no allocation of its own (a const-init
// `Cell<Meter>` needs neither lazy initialization nor a destructor), so
// observing cannot re-enter the allocator.
unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size() as isize);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size() as isize);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(-(layout.size() as isize));
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size as isize - layout.size() as isize);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

/// Runs `operation` and reports its peak live allocation in bytes.
fn measure_peak<R>(operation: impl FnOnce() -> R) -> (R, usize) {
    METER.with(|cell| {
        cell.set(Meter {
            armed: true,
            live: 0,
            peak: 0,
        })
    });
    let value = operation();
    let peak = METER.with(|cell| {
        let meter = cell.get();
        cell.set(Meter::IDLE);
        meter.peak
    });
    (value, peak)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The declared length a hostile manifest claims.
///
/// Four billion `f32` values is sixteen gigabytes if a decoder believes it, and
/// a small enough number that the multiplication does not overflow — so a
/// decoder that refused only on overflow would still reserve it.
const HOSTILE_LEN: usize = 4_000_000_000;

/// The recipe every case below is built from.
///
/// A bare source, so the container holds exactly one array and each field in
/// the manifest occurs once. That makes a textual edit unambiguous: there is no
/// second `"len"` for a replacement to land on by accident.
fn probe_recipe() -> Recipe {
    Recipe::seeded(64, 4, 5).expect("a valid shape")
}

/// Writes a genuine container and returns the exchange it lives in.
fn materialize(directory: &Path) -> DatasetExchange {
    let exchange = DatasetExchange::new(directory);
    exchange
        .materialize("probe", &probe_recipe())
        .expect("a valid recipe materializes");
    exchange
}

/// Rewrites the manifest, leaving the array file — and therefore its recorded
/// digest — untouched.
///
/// That is what makes these cases reach the decoder at all: the spec digest
/// covers the recipe and the data digest covers the array file, so an edit to
/// the array *table* passes both checks and is answered by the table
/// validation and the cursor rather than by a checksum.
fn edit_manifest(exchange: &DatasetExchange, from: &str, to: &str) -> usize {
    let path = exchange.manifest_path("probe").expect("a valid name");
    let text = std::fs::read_to_string(&path).expect("the manifest was just written");
    let edited = text.replace(from, to);
    assert_ne!(edited, text, "the edit {from:?} did not reach the manifest");
    std::fs::write(&path, &edited).expect("the manifest is writable");
    edited.len()
}

/// The two files' combined length, which is what a budget is scaled against.
fn container_bytes(exchange: &DatasetExchange) -> usize {
    let manifest = exchange.manifest_path("probe").expect("a valid name");
    let data = exchange.data_path("probe").expect("a valid name");
    let length = |path: &Path| {
        std::fs::metadata(path)
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0)
    };
    length(&manifest) + length(&data)
}

fn budget(input_bytes: usize) -> usize {
    ALLOC_BASE_BYTES + ALLOC_INPUT_FACTOR * input_bytes
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

#[test]
fn a_declared_array_length_is_never_reserved_before_its_bytes_are_read() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = materialize(directory.path());
    edit_manifest(
        &exchange,
        "\"len\": 256",
        &format!("\"len\": {HOSTILE_LEN}"),
    );

    let input = container_bytes(&exchange);
    let (outcome, peak) = measure_peak(|| exchange.load("probe"));

    assert!(
        matches!(outcome, Err(ExchangeError::InvalidArrayTable)),
        "a length the file cannot supply has to be refused, got {outcome:?}",
    );
    // The refusal is the cheap half. This is the half no accept/reject test
    // sees: believing the declaration would have reserved sixteen gigabytes
    // from a container of about a kilobyte.
    let believed = HOSTILE_LEN * 4;
    assert!(
        peak <= budget(input),
        "loading {input} bytes allocated {peak}, budget {}; the declaration was {believed} bytes, \
         so the amplification would have been {}x",
        budget(input),
        believed / input.max(1),
    );
}

#[test]
fn every_hostile_array_table_refuses_inside_the_same_budget() {
    // One edit per field the table declares. Each is answered somewhere
    // different — an overflow check, the shape check, the contiguity check, the
    // cursor — and the oracle is the same for all of them, because "which check
    // caught it" is exactly the thing a caller cannot rely on.
    let cases: [(&str, &str, String); 6] = [
        (
            "an inflated length",
            "\"len\": 256",
            format!("\"len\": {HOSTILE_LEN}"),
        ),
        (
            "a length that overflows its byte span",
            "\"len\": 256",
            format!("\"len\": {}", usize::MAX),
        ),
        (
            "an inflated row count",
            "\"rows\": 64",
            format!("\"rows\": {HOSTILE_LEN}"),
        ),
        (
            "an offset past the file",
            "\"byte_offset\": 0",
            format!("\"byte_offset\": {HOSTILE_LEN}"),
        ),
        (
            "a widened element type",
            "\"dtype\": \"f32\"",
            "\"dtype\": \"u64\"".to_owned(),
        ),
        (
            "an inflated file length",
            "\"bytes\": 1024",
            format!("\"bytes\": {HOSTILE_LEN}"),
        ),
    ];

    for (label, from, to) in cases {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let exchange = materialize(directory.path());
        edit_manifest(&exchange, from, &to);

        let input = container_bytes(&exchange);
        let (outcome, peak) = measure_peak(|| exchange.load("probe"));
        assert!(
            outcome.is_err(),
            "{label} was accepted: {:?}",
            outcome.map(|container| container.data_bytes()),
        );
        assert!(
            peak <= budget(input),
            "{label}: loading {input} bytes allocated {peak}, budget {}",
            budget(input),
        );
    }
}

#[test]
fn a_genuine_load_stays_inside_the_same_budget() {
    // Without this the budget could be anything: an oracle nothing legitimate
    // comes close to filling is a number, not a measurement. A real load has to
    // fit the same envelope the hostile ones are held to.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = materialize(directory.path());
    let input = container_bytes(&exchange);

    let (outcome, peak) = measure_peak(|| exchange.load("probe"));
    let container = outcome.expect("a container this crate just wrote reloads");
    assert_eq!(container.spec_digest(), probe_recipe().spec_digest());
    assert!(
        peak <= budget(input),
        "a genuine load of {input} bytes allocated {peak}, budget {}",
        budget(input),
    );

    // And at a size where the fixed part no longer dominates, because that is
    // where the per-byte factor is the whole budget: a load holds the file's
    // bytes and the values decoded from them at once, so the factor is what
    // says how many copies of the input are allowed to be live.
    let wide = Recipe::seeded(4_096, 32, 5).expect("a valid shape");
    let exchange = DatasetExchange::new(directory.path());
    exchange
        .materialize("wide", &wide)
        .expect("a valid recipe materializes");
    let input = std::fs::metadata(exchange.data_path("wide").expect("a valid name"))
        .expect("the file was just written")
        .len() as usize;
    let (outcome, peak) = measure_peak(|| exchange.load("wide"));
    outcome.expect("a container this crate just wrote reloads");
    assert!(
        peak <= budget(input),
        "a genuine load of {input} bytes allocated {peak}, budget {}",
        budget(input),
    );
}

#[test]
fn the_oracle_reports_an_unbounded_reservation() {
    // The control. Every assertion above is of the form "the peak stayed
    // inside a budget", which a meter that had stopped counting would also
    // satisfy. This reserves from exactly the declaration those cases carry and
    // requires the meter to see it.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = materialize(directory.path());
    let input = container_bytes(&exchange);

    let (length, peak) = measure_peak(|| {
        // What the decoder would have done before `bounded_capacity`: size the
        // buffer from the declared count rather than from the bytes present.
        // One element is pushed so the reservation cannot be optimized away.
        let mut values: Vec<f32> = Vec::with_capacity(HOSTILE_LEN);
        values.push(1.0);
        values.len()
    });

    assert_eq!(length, 1);
    assert!(
        peak > budget(input),
        "the meter reported {peak} for a {}-byte reservation, which is inside the {} budget the \
         oracle above relies on being exceeded",
        HOSTILE_LEN * 4,
        budget(input),
    );
}

// ---------------------------------------------------------------------------
// The round trip, through the files rather than around them
// ---------------------------------------------------------------------------

#[test]
fn a_materialized_container_reloads_to_identical_bytes() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = probe_recipe();

    let written = exchange
        .materialize("probe", &recipe)
        .expect("a valid recipe materializes");
    let loaded = exchange.load("probe").expect("the container reloads");

    assert_eq!(loaded, written);
    assert_eq!(
        loaded
            .array("features")
            .and_then(|array| array.f32_values()),
        Some(recipe.generate().features().as_slice()),
    );
    assert_eq!(loaded.data_digest(), written.data_digest());
}
