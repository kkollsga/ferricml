//! Structured fuzzing of every artifact decoder.
//!
//! Artifacts are the one FerricML surface that consumes fully untrusted bytes,
//! and their contract is three claims: a decoder returns a typed
//! [`ArtifactError`] rather than panicking, it never allocates in proportion to
//! a length field it has not validated, and it never reads out of bounds. This
//! file makes all three falsifiable.
//!
//! Two things make it more than a byte-flipper. First, every mutation that
//! touches the checksummed region is *resealed* — the SHA-256 footer is
//! recomputed — so the mutation actually reaches the payload parser instead of
//! bouncing off `ChecksumMismatch`. Second, alongside the mutators there are
//! grammar generators that build envelopes, components, and logical trees from
//! scratch, which reaches states no mutation of a valid artifact reaches.
//!
//! The oracles are:
//!
//! * no panic and no out-of-bounds read (a panic fails the test; reads are
//!   bounded because every decoder goes through `ArtifactCursor`, and a mistake
//!   there aborts under the test harness);
//! * peak live allocation during one decode stays inside
//!   [`ALLOC_BASE_BYTES`] + [`ALLOC_INPUT_FACTOR`] × input length; and
//! * anything accepted re-encodes to exactly the bytes it was decoded from, so
//!   an artifact cannot be malleable — two distinct byte strings decoding to
//!   one model would mean the reader accepts a non-canonical encoding.
//!
//! No fuzzing dependency is involved: the generator is the crate's own
//! SplitMix64 stream, restated here because `src/numeric/rng.rs` is private.

use ferricml::api::AnyRegressor;
use ferricml::artifact::ArtifactError;
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures,
    RandomForestRegressor, RandomForestRegressorParams,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams, StandardScaler,
    StandardScalerParams,
};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
};
use sha2::{Digest, Sha256};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// ---------------------------------------------------------------------------
// Peak-allocation meter
// ---------------------------------------------------------------------------

/// Allocation the decode of a *tiny* artifact may always use.
///
/// Both numbers are deliberately close to what real artifacts measure — the
/// largest ratio observed for a genuine artifact is under 3× — because a
/// budget set far above the truth stops being an oracle. They are still loose
/// enough that ordinary allocator rounding cannot trip them.
pub const ALLOC_BASE_BYTES: usize = 4 * 1024;
/// Multiple of the encoded length a decode may allocate beyond that base.
pub const ALLOC_INPUT_FACTOR: usize = 8;

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
    /// Per-thread so two `#[test]` functions in this binary cannot interfere.
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
// Deterministic generator
// ---------------------------------------------------------------------------

/// SplitMix64, the same stream `src/numeric/rng.rs` defines for the crate.
///
/// It is restated rather than imported because the crate generator is
/// `pub(crate)`; an integration test cannot reach it, and exposing it would
/// enlarge the public API for a test's convenience.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "empty range");
        (self.next_u64() % upper as u64) as usize
    }

    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }

    fn pick<'a, T>(&mut self, values: &'a [T]) -> &'a T {
        &values[self.below(values.len())]
    }
}

/// Values that sit on a bound, just past one, or exercise sign handling.
const INTERESTING_U32: [u32; 19] = [
    0,
    1,
    2,
    3,
    4,
    0x0000_007f,
    0x0000_00ff,
    0x0000_ffff,
    4_096,
    4_097,
    65_536,
    65_537,
    131_071,
    131_072,
    1_000_000,
    1_000_001,
    1_048_576,
    0x7fff_ffff,
    0xffff_ffff,
];

const INTERESTING_BYTES: [u8; 8] = [0x00, 0x01, 0x02, 0x55, 0x7f, 0x80, 0xfe, 0xff];

/// Non-finite and signed-zero `f32` bit patterns plus ordinary magnitudes.
const INTERESTING_F32_BITS: [u32; 10] = [
    0x0000_0000, // 0.0
    0x8000_0000, // -0.0
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x7f80_0000, // inf
    0xff80_0000, // -inf
    0x7fc0_0000, // NaN
    0x7f7f_ffff, // f32::MAX
    0x0080_0000, // smallest normal
    0x0000_0001, // smallest subnormal
];

// ---------------------------------------------------------------------------
// Envelope layout
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 8] = b"FERRICML";
const V2_HEADER_BYTES: usize = 24;
const SCHEMA_RECORD_BYTES: usize = 36;
const CHECKSUM_BYTES: usize = 32;
const COMPONENT_HEADER_BYTES: usize = 8;

const INPUT_SCHEMA: [u8; 32] = [3; 32];
const TRANSFORMED_SCHEMA: [u8; 32] = [4; 32];

/// Every artifact kind the crate reads today, plus the neighbours it must not.
const ARTIFACT_KINDS: [u16; 18] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("two bytes"))
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

/// Recomputes the SHA-256 footer so a mutation reaches the payload parser.
fn reseal(bytes: &mut [u8]) {
    if bytes.len() < CHECKSUM_BYTES {
        return;
    }
    let split = bytes.len() - CHECKSUM_BYTES;
    let digest = Sha256::digest(&bytes[..split]);
    bytes[split..].copy_from_slice(&digest);
}

/// Byte range of the payload inside a well-formed version-2 envelope.
fn payload_span(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.len() < V2_HEADER_BYTES + CHECKSUM_BYTES || &bytes[..8] != MAGIC {
        return None;
    }
    let payload_len = u32_at(bytes, 16) as usize;
    let schema_count = u16_at(bytes, 20) as usize;
    let start = V2_HEADER_BYTES.checked_add(schema_count.checked_mul(SCHEMA_RECORD_BYTES)?)?;
    let end = start.checked_add(payload_len)?;
    (end.checked_add(CHECKSUM_BYTES)? <= bytes.len()).then_some((start, payload_len))
}

/// Offsets of each length-delimited component inside a payload.
fn component_offsets(payload: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut at = 0;
    while at + COMPONENT_HEADER_BYTES <= payload.len() {
        offsets.push(at);
        let length = u32_at(payload, at + 4) as usize;
        match at
            .checked_add(COMPONENT_HEADER_BYTES)
            .and_then(|next| next.checked_add(length))
        {
            Some(next) if next <= payload.len() => at = next,
            _ => break,
        }
    }
    offsets
}

fn component(kind: u16, version: u16, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(COMPONENT_HEADER_BYTES + payload.len());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn envelope(
    kind: u16,
    payload_version: u16,
    schemas: &[(u16, [u8; 32])],
    payload: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(
        V2_HEADER_BYTES + schemas.len() * SCHEMA_RECORD_BYTES + payload.len() + CHECKSUM_BYTES,
    );
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&payload_version.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(schemas.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    for (role, hash) in schemas {
        bytes.extend_from_slice(&role.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(hash);
    }
    bytes.extend_from_slice(payload);
    let digest = Sha256::digest(&bytes);
    bytes.extend_from_slice(&digest);
    bytes
}

// ---------------------------------------------------------------------------
// Decoders under test
// ---------------------------------------------------------------------------

/// What one decoder did with one candidate byte string.
enum Outcome {
    Rejected(ArtifactError),
    /// Accepted, together with what re-encoding the decoded model produced.
    Accepted(Result<Vec<u8>, ArtifactError>),
}

fn accepted<T>(
    decoded: Result<T, ArtifactError>,
    encode: impl FnOnce(&T) -> Result<Vec<u8>, ArtifactError>,
) -> Outcome {
    match decoded {
        Err(error) => Outcome::Rejected(error),
        Ok(model) => Outcome::Accepted(encode(&model)),
    }
}

type Decoder = (&'static str, fn(&[u8]) -> Outcome);

type StagedTwo = StagedPipeline<(MinMaxScaler, StandardScaler), Ridge>;
type StagedThree = StagedPipeline<(MinMaxScaler, StandardScaler, MaxAbsScaler), Ridge>;

fn decoders() -> Vec<Decoder> {
    let table: Vec<Decoder> = vec![
        ("logistic", |bytes| {
            accepted(
                LogisticRegression::from_artifact(bytes, INPUT_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA),
            )
        }),
        ("linear", |bytes| {
            accepted(LinearRegression::from_artifact(bytes, INPUT_SCHEMA), |m| {
                m.to_artifact(INPUT_SCHEMA)
            })
        }),
        ("ridge", |bytes| {
            accepted(Ridge::from_artifact(bytes, INPUT_SCHEMA), |m| {
                m.to_artifact(INPUT_SCHEMA)
            })
        }),
        ("standard-scaler", |bytes| {
            accepted(
                StandardScaler::from_artifact(bytes, INPUT_SCHEMA, TRANSFORMED_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("min-max-scaler", |bytes| {
            accepted(
                MinMaxScaler::from_artifact(bytes, INPUT_SCHEMA, TRANSFORMED_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("max-abs-scaler", |bytes| {
            accepted(
                MaxAbsScaler::from_artifact(bytes, INPUT_SCHEMA, TRANSFORMED_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("pipeline-logistic", |bytes| {
            accepted(
                Pipeline::<StandardScaler, LogisticRegression>::from_artifact(
                    bytes,
                    INPUT_SCHEMA,
                    TRANSFORMED_SCHEMA,
                ),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("pipeline-linear", |bytes| {
            accepted(
                Pipeline::<StandardScaler, LinearRegression>::from_artifact(
                    bytes,
                    INPUT_SCHEMA,
                    TRANSFORMED_SCHEMA,
                ),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("pipeline-ridge", |bytes| {
            accepted(
                Pipeline::<StandardScaler, Ridge>::from_artifact(
                    bytes,
                    INPUT_SCHEMA,
                    TRANSFORMED_SCHEMA,
                ),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("pairwise-ranker", |bytes| {
            accepted(
                PairwiseLinearRanker::from_artifact(bytes, INPUT_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA),
            )
        }),
        ("hist-gradient-boosting", |bytes| {
            accepted(
                HistGradientBoostingRegressor::from_artifact(bytes, INPUT_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA),
            )
        }),
        ("random-forest", |bytes| {
            accepted(
                RandomForestRegressor::from_artifact(bytes, INPUT_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA),
            )
        }),
        ("any-regressor", |bytes| {
            accepted(AnyRegressor::from_artifact(bytes, INPUT_SCHEMA), |m| {
                m.to_artifact(INPUT_SCHEMA)
            })
        }),
        ("staged-two", |bytes| {
            accepted(
                StagedTwo::from_artifact(bytes, INPUT_SCHEMA, TRANSFORMED_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
        ("staged-three", |bytes| {
            accepted(
                StagedThree::from_artifact(bytes, INPUT_SCHEMA, TRANSFORMED_SCHEMA),
                |m| m.to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA),
            )
        }),
    ];
    table
}

// ---------------------------------------------------------------------------
// Valid seed corpus
// ---------------------------------------------------------------------------

/// A version-1 logistic artifact, which no current writer produces.
///
/// The legacy reader is still reachable — any artifact whose version field is
/// not 2 routes into it — so it has to be fuzzed, and the only way to seed it
/// is to build one. The layout is the magic, version, kind, the bare
/// feature-schema hash, the logistic state fields with no component framing,
/// and the checksum.
fn legacy_logistic_seed(n_features: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&INPUT_SCHEMA);
    for value in [
        n_features,
        1,           // fit_intercept
        0x3f80_0000, // C = 1.0
        100,         // max_iter
        0x3727_c5ac, // tol = 1e-5
        7,           // iterations
        0x3dcc_cccd, // intercept = 0.1
        n_features,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for index in 0..n_features {
        bytes.extend_from_slice(&(index as f32 / 8.0 - 0.5).to_bits().to_le_bytes());
    }
    let digest = Sha256::digest(&bytes);
    bytes.extend_from_slice(&digest);
    bytes
}

/// One fitted artifact of every kind the crate writes.
fn seed_corpus() -> Vec<(&'static str, Vec<u8>)> {
    let data = DenseMatrix::new(vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 8.0], 4, 2).unwrap();
    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();

    let logistic = LogisticRegression::fit(
        &data.as_view(),
        &binary,
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let linear = LinearRegression::fit(
        &data.as_view(),
        &regression,
        LinearRegressionParams::default(),
    )
    .unwrap();
    let ridge = Ridge::fit(&data.as_view(), &regression, RidgeParams::default()).unwrap();
    let standard = StandardScaler::fit(&data.as_view(), StandardScalerParams::default()).unwrap();
    let min_max = MinMaxScaler::fit(&data.as_view(), MinMaxScalerParams::default()).unwrap();
    let max_abs = MaxAbsScaler::fit(&data.as_view(), MaxAbsScalerParams).unwrap();
    let transformed = standard.transform(&data.as_view()).unwrap();

    let pipeline_logistic = Pipeline::new(
        standard.clone(),
        LogisticRegression::fit(
            &transformed.as_view(),
            &binary,
            LogisticRegressionParams::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let pipeline_linear = Pipeline::new(
        standard.clone(),
        LinearRegression::fit(
            &transformed.as_view(),
            &regression,
            LinearRegressionParams::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let pipeline_ridge = Pipeline::new(
        standard.clone(),
        Ridge::fit(&transformed.as_view(), &regression, RidgeParams::default()).unwrap(),
    )
    .unwrap();

    let items = DenseMatrix::new(vec![0.0, 0.0, 1.0, 0.25, 2.0, 1.0, 3.0, 2.0], 4, 2).unwrap();
    let pair = |left, right, outcome, weight| {
        PairwiseObservation::new(PairIndex::new(left, right).unwrap(), outcome, weight).unwrap()
    };
    let ranker = PairwiseLinearRanker::fit(
        &items.as_view(),
        &[
            pair(3, 2, PairOutcome::LeftPreferred, 2.0),
            pair(2, 1, PairOutcome::LeftPreferred, 1.0),
            pair(1, 0, PairOutcome::LeftPreferred, 1.0),
            pair(1, 2, PairOutcome::Tie, 0.5),
        ],
        PairwiseLinearRankerParams::default(),
    )
    .unwrap();

    let boosting_data = DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap();
    let boosting_targets =
        RegressionTargets::new(vec![0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]).unwrap();
    let boosting = HistGradientBoostingRegressor::fit(
        &boosting_data.as_view(),
        &boosting_targets,
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(2)
            .with_max_leaf_nodes(4)
            .with_min_samples_leaf(1)
            .with_max_bins(8),
    )
    .unwrap();
    let forest = RandomForestRegressor::fit(
        &data.as_view(),
        &regression,
        RandomForestRegressorParams::default()
            .with_n_estimators(3)
            .with_max_depth(Some(4))
            .with_max_features(MaxFeatures::All)
            .with_random_state(11),
    )
    .unwrap();

    let staged_two: StagedTwo = StagedPipeline::fit(
        &data.as_view(),
        |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
        |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
        |batch| Ridge::fit(batch, &regression, RidgeParams::default()),
    )
    .unwrap();
    let after_first = min_max.transform(&data.as_view()).unwrap();
    let second =
        StandardScaler::fit(&after_first.as_view(), StandardScalerParams::default()).unwrap();
    let after_second = second.transform(&after_first.as_view()).unwrap();
    let third = MaxAbsScaler::fit(&after_second.as_view(), MaxAbsScalerParams).unwrap();
    let after_third = third.transform(&after_second.as_view()).unwrap();
    let staged_three: StagedThree = StagedPipeline::new(
        (min_max.clone(), second, third),
        Ridge::fit(&after_third.as_view(), &regression, RidgeParams::default()).unwrap(),
    )
    .unwrap();

    vec![
        ("logistic", logistic.to_artifact(INPUT_SCHEMA).unwrap()),
        ("linear", linear.to_artifact(INPUT_SCHEMA).unwrap()),
        ("ridge", ridge.to_artifact(INPUT_SCHEMA).unwrap()),
        (
            "standard-scaler",
            standard
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        (
            "min-max-scaler",
            min_max
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        (
            "max-abs-scaler",
            max_abs
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        (
            "pipeline-logistic",
            pipeline_logistic
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        (
            "pipeline-linear",
            pipeline_linear
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        (
            "pipeline-ridge",
            pipeline_ridge
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        ("pairwise-ranker", ranker.to_artifact(INPUT_SCHEMA).unwrap()),
        ("boosting", boosting.to_artifact(INPUT_SCHEMA).unwrap()),
        ("forest", forest.to_artifact(INPUT_SCHEMA).unwrap()),
        (
            "any-forest",
            AnyRegressor::from(forest)
                .to_artifact(INPUT_SCHEMA)
                .unwrap(),
        ),
        (
            "any-ridge",
            AnyRegressor::from(ridge).to_artifact(INPUT_SCHEMA).unwrap(),
        ),
        (
            "any-boosting",
            AnyRegressor::from(boosting)
                .to_artifact(INPUT_SCHEMA)
                .unwrap(),
        ),
        (
            "staged-two",
            staged_two
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        (
            "staged-three",
            staged_three
                .to_artifact(INPUT_SCHEMA, TRANSFORMED_SCHEMA)
                .unwrap(),
        ),
        ("legacy-logistic", legacy_logistic_seed(2)),
        // Degenerate inputs a mutator would rarely shrink all the way to.
        ("empty", Vec::new()),
        ("one-byte", vec![0xff]),
        ("magic-only", MAGIC.to_vec()),
    ]
}

// ---------------------------------------------------------------------------
// Mutators and grammar generators
// ---------------------------------------------------------------------------

/// Structural shape of a synthetic logical tree payload.
fn random_tree_payload(rng: &mut Rng, n_features: u32) -> Vec<u8> {
    let mut records: Vec<[u32; 5]> = Vec::new();
    if rng.chance(3) {
        // Free-form: header and records drawn straight from interesting values.
        let count = rng.below(6);
        for _ in 0..count {
            records.push([
                *rng.pick(&[0, 1, 2, 0xffff_ffff]),
                *rng.pick(&INTERESTING_U32),
                *rng.pick(&INTERESTING_F32_BITS),
                *rng.pick(&INTERESTING_U32),
                *rng.pick(&INTERESTING_U32),
            ]);
        }
        let mut payload = Vec::with_capacity(12 + records.len() * 20);
        payload.extend_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes());
        payload.extend_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes());
        payload.extend_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes());
        for record in &records {
            for field in record {
                payload.extend_from_slice(&field.to_le_bytes());
            }
        }
        return payload;
    }

    // Structurally valid pre-order tree, then a few targeted perturbations.
    let depth_limit = 1 + rng.below(4);
    grow(rng, depth_limit, n_features, &mut records);
    let (leaves, depth) = tree_stats(&records);
    let mut header = [records.len() as u32, leaves, depth];
    if rng.chance(3) {
        header[rng.below(3)] = *rng.pick(&INTERESTING_U32);
    }
    for _ in 0..rng.below(3) {
        if records.is_empty() || rng.chance(2) {
            break;
        }
        let index = rng.below(records.len());
        let field = rng.below(5);
        records[index][field] = if field == 2 {
            *rng.pick(&INTERESTING_F32_BITS)
        } else {
            *rng.pick(&INTERESTING_U32)
        };
    }
    let mut payload = Vec::with_capacity(12 + records.len() * 20);
    for value in header {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for record in &records {
        for field in record {
            payload.extend_from_slice(&field.to_le_bytes());
        }
    }
    payload
}

/// Appends a valid pre-order subtree and returns nothing; `left` is always the
/// next record, which is what the logical-tree contract requires.
fn grow(rng: &mut Rng, depth: usize, n_features: u32, records: &mut Vec<[u32; 5]>) {
    if depth == 0 || rng.chance(3) {
        records.push([0, leaf_bits(rng), 0, 0, 0]);
        return;
    }
    let index = records.len();
    records.push([
        1,
        rng.below(n_features.max(1) as usize) as u32,
        0x3f80_0000,
        0,
        0,
    ]);
    grow(rng, depth - 1, n_features, records);
    let right = records.len() as u32;
    grow(rng, depth - 1, n_features, records);
    records[index][3] = index as u32 + 1;
    records[index][4] = right;
}

/// A leaf value: usually an ordinary finite number, sometimes a boundary bit
/// pattern. Drawing only from the boundary set would make nearly every
/// generated tree non-finite and therefore rejected before the topology is
/// ever examined.
fn leaf_bits(rng: &mut Rng) -> u32 {
    if rng.chance(4) {
        return *rng.pick(&INTERESTING_F32_BITS);
    }
    (rng.below(41) as f32 / 4.0 - 5.0).to_bits()
}

fn tree_stats(records: &[[u32; 5]]) -> (u32, u32) {
    fn walk(records: &[[u32; 5]], index: usize, depth: u32, leaves: &mut u32, max: &mut u32) {
        *max = (*max).max(depth);
        if records[index][0] == 0 {
            *leaves += 1;
            return;
        }
        walk(records, index + 1, depth + 1, leaves, max);
        walk(records, records[index][4] as usize, depth + 1, leaves, max);
    }
    let (mut leaves, mut max) = (0, 0);
    if !records.is_empty() {
        walk(records, 0, 0, &mut leaves, &mut max);
    }
    (leaves, max)
}

/// Forest metadata, coherent with the trees that follow unless perturbed.
///
/// A generator whose metadata is uniformly random never gets past the count
/// cross-checks, so it can only ever exercise rejection. Building metadata that
/// *agrees* with the generated trees and then perturbing at most one field is
/// what lets a synthetic artifact reach the tree decoder, the packed rebuild,
/// and acceptance.
fn forest_metadata(rng: &mut Rng, n_features: u32, tree_count: u32, nodes: u32) -> Vec<u8> {
    let mut fields: Vec<u32> = vec![
        1,          // objective version
        n_features, // fitted width
        tree_count, // n_estimators
        4,          // max_depth
        2,          // min_samples_split
        1,          // min_samples_leaf
        1,          // max_features tag: All
        0,          // max_features count
        0,          // bootstrap
        1,          // n_jobs tag: Serial
        0,          // n_jobs count
        tree_count, // tree count
        nodes,      // total logical nodes
    ];
    perturb(rng, &mut fields);
    let mut payload = Vec::with_capacity(13 * 4 + 8);
    for (index, value) in fields.iter().enumerate() {
        // `random_state` is the one eight-byte field, between bootstrap and
        // the parallelism tag.
        if index == 9 {
            payload.extend_from_slice(&rng.next_u64().to_le_bytes());
        }
        payload.extend_from_slice(&value.to_le_bytes());
    }
    payload
}

/// Boosting metadata, coherent with the trees that follow unless perturbed.
fn boosting_metadata(rng: &mut Rng, n_features: u32, tree_count: u32, nodes: u32) -> Vec<u8> {
    let mut fields: Vec<u32> = vec![
        1,           // objective version
        n_features,  // fitted width
        0x3dcc_cccd, // learning rate 0.1
        tree_count,  // max_iter
        4,           // max_leaf_nodes
        0,           // max_depth: none
        1,           // min_samples_leaf
        0,           // l2 regularization 0.0
        8,           // max_bins
        0,           // baseline 0.0
        tree_count,  // tree count
        nodes,       // total logical nodes
    ];
    perturb(rng, &mut fields);
    let mut payload = Vec::with_capacity(12 * 4);
    for value in fields {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    payload
}

/// Replaces up to one metadata field with a boundary value.
fn perturb(rng: &mut Rng, fields: &mut [u32]) {
    if rng.chance(2) {
        return;
    }
    let index = rng.below(fields.len());
    fields[index] = *rng.pick(&INTERESTING_U32);
}

/// Builds a whole artifact from the grammar rather than mutating a valid one.
fn synthesize(rng: &mut Rng) -> Vec<u8> {
    // Weighted towards the two tree-bearing kinds; the rest cover cross-kind
    // confusion, including kinds this build does not implement.
    let kind = *rng.pick(&[9_u16, 9, 9, 10, 10, 10, 0, 1, 4, 11, 13, 16, 17]);
    // Small widths keep generated feature indices inside the fitted width, so
    // the trees themselves are the variable rather than the metadata.
    let n_features = *rng.pick(&[1_u32, 1, 2, 3, 0, 1_000_001]);

    let mut payload = Vec::new();
    let shape = rng.below(4);
    if shape < 2 {
        let mut trees = Vec::new();
        let mut nodes = 0_u32;
        for _ in 0..1 + rng.below(3) {
            let tree = random_tree_payload(rng, n_features);
            nodes = nodes.saturating_add(if tree.len() >= 4 { u32_at(&tree, 0) } else { 0 });
            trees.push(tree);
        }
        let count = trees.len() as u32;
        let metadata = if shape == 0 {
            forest_metadata(rng, n_features, count, nodes)
        } else {
            boosting_metadata(rng, n_features, count, nodes)
        };
        payload.extend_from_slice(&component(1, 1, &metadata));
        for tree in &trees {
            payload.extend_from_slice(&component(2, 1, tree));
        }
    } else {
        // A free-form component sequence.
        for _ in 0..rng.below(4) {
            let length = rng.below(48);
            let mut body = Vec::with_capacity(length);
            for _ in 0..length {
                body.push(*rng.pick(&INTERESTING_BYTES));
            }
            payload.extend_from_slice(&component(
                *rng.pick(&[0_u16, 1, 2, 3, 4, 0xffff]),
                *rng.pick(&[0_u16, 1, 2, 0xffff]),
                &body,
            ));
        }
    }

    // Weighted towards the shapes a reader accepts: a generator that almost
    // never produces a decodable envelope only ever exercises the header.
    let roles: &[(u16, [u8; 32])] = match rng.below(8) {
        0 => &[],
        1 | 4 | 5 | 6 => &[(1, INPUT_SCHEMA)],
        2 => &[(2, TRANSFORMED_SCHEMA)],
        3 => &[(1, INPUT_SCHEMA), (2, TRANSFORMED_SCHEMA)],
        _ => &[
            (1, INPUT_SCHEMA),
            (1, INPUT_SCHEMA),
            (2, TRANSFORMED_SCHEMA),
        ],
    };
    let mut bytes = envelope(
        kind,
        *rng.pick(&[1_u16, 1, 1, 0, 2, 0xffff]),
        roles,
        &payload,
    );

    // Occasionally corrupt a declared length after sealing so the reader meets
    // a length field that disagrees with the bytes actually present.
    if rng.chance(3) && bytes.len() > 20 {
        let offset = *rng.pick(&[14_usize, 16, 20, 22]);
        match offset {
            16 => bytes[16..20].copy_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes()),
            _ => bytes[offset..offset + 2]
                .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
        }
        reseal(&mut bytes);
    }
    bytes
}

const STRATEGIES: [&str; 10] = [
    "bit-flip",
    "byte-splat",
    "word-overwrite",
    "truncate",
    "extend",
    "splice",
    "header-field",
    "component-field",
    "coherent-count",
    "synthesize",
];

fn mutate(
    rng: &mut Rng,
    strategy: usize,
    seed: &[u8],
    corpus: &[(&'static str, Vec<u8>)],
) -> Vec<u8> {
    let mut bytes = seed.to_vec();
    match STRATEGIES[strategy] {
        "bit-flip" => {
            for _ in 0..1 + rng.below(4) {
                if bytes.is_empty() {
                    break;
                }
                let index = rng.below(bytes.len());
                bytes[index] ^= 1 << rng.below(8);
            }
        }
        "byte-splat" => {
            for _ in 0..1 + rng.below(4) {
                if bytes.is_empty() {
                    break;
                }
                let index = rng.below(bytes.len());
                bytes[index] = *rng.pick(&INTERESTING_BYTES);
            }
        }
        "word-overwrite" => {
            if bytes.len() >= 4 {
                let index = rng.below(bytes.len() - 3);
                bytes[index..index + 4].copy_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes());
            }
        }
        "truncate" => {
            let keep = rng.below(bytes.len() + 1);
            bytes.truncate(keep);
        }
        "extend" => {
            for _ in 0..1 + rng.below(8) {
                bytes.push(*rng.pick(&INTERESTING_BYTES));
            }
        }
        "splice" => {
            let donor = &rng.pick(corpus).1;
            if !bytes.is_empty() && !donor.is_empty() {
                let at = rng.below(bytes.len());
                let from = rng.below(donor.len());
                let length = rng.below((donor.len() - from).min(bytes.len() - at) + 1);
                bytes[at..at + length].copy_from_slice(&donor[from..from + length]);
            }
        }
        "header-field" => {
            if bytes.len() >= V2_HEADER_BYTES {
                match rng.below(6) {
                    0 => bytes[8..10]
                        .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
                    1 => bytes[10..12].copy_from_slice(&rng.pick(&ARTIFACT_KINDS).to_le_bytes()),
                    2 => bytes[12..14]
                        .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
                    3 => bytes[14..16]
                        .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
                    4 => bytes[16..20].copy_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes()),
                    _ => bytes[20..22]
                        .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
                }
            }
        }
        "component-field" => {
            if let Some((start, length)) = payload_span(&bytes) {
                let offsets = component_offsets(&bytes[start..start + length]);
                if !offsets.is_empty() {
                    let at = start + offsets[rng.below(offsets.len())];
                    match rng.below(4) {
                        0 => bytes[at..at + 2]
                            .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
                        1 => bytes[at + 2..at + 4]
                            .copy_from_slice(&(*rng.pick(&INTERESTING_U32) as u16).to_le_bytes()),
                        2 => bytes[at + 4..at + 8]
                            .copy_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes()),
                        _ => {
                            let body = at + COMPONENT_HEADER_BYTES;
                            let limit = start + length;
                            if body + 4 <= limit {
                                let index = body + rng.below(limit - body - 3);
                                bytes[index..index + 4]
                                    .copy_from_slice(&rng.pick(&INTERESTING_U32).to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        // Every count in an artifact is written more than once — a declared
        // width and a repeated element count, a tree count and a node total —
        // and a reader that cross-checks them defeats any single-field edit.
        // Rewriting *every* word that currently holds the same value keeps the
        // cross-checks satisfied, which is what lets a mutant reach the
        // allocation that follows them.
        "coherent-count" => {
            if let Some((start, length)) = payload_span(&bytes) {
                let mut words: Vec<u32> = Vec::new();
                let mut at = start;
                while at + 4 <= start + length {
                    words.push(u32_at(&bytes, at));
                    at += 4;
                }
                if !words.is_empty() {
                    let target = words[rng.below(words.len())];
                    let replacement = *rng.pick(&INTERESTING_U32);
                    let mut at = start;
                    while at + 4 <= start + length {
                        if u32_at(&bytes, at) == target {
                            bytes[at..at + 4].copy_from_slice(&replacement.to_le_bytes());
                        }
                        at += 4;
                    }
                }
            }
        }
        _ => return synthesize(rng),
    }
    // Half of every mutation is resealed so it reaches the payload parser
    // instead of stopping at the integrity footer; the other half proves the
    // footer still rejects what it should.
    if rng.chance(2) {
        reseal(&mut bytes);
    }
    bytes
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

/// Mutants generated per (seed, strategy) pair inside `make gate`.
///
/// Deep local campaigns raise it through `FERRICML_FUZZ_ROUNDS`; the gate keeps
/// the checked-in value so the sweep is a fixed, bounded, reproducible corpus
/// rather than a random-length run.
const ROUNDS: usize = 6;
/// Fixed so the whole sweep is reproducible from the file alone.
const FUZZ_SEED: u64 = 0xf3_77_1c_a1_5e_ed_00_01;

/// How far one candidate got, so the sweep can prove it is not inert.
#[derive(Default)]
struct Reach {
    candidates: usize,
    /// Rejections that can only be reached after the integrity footer passed.
    past_checksum: usize,
    /// Candidates some decoder accepted as a model.
    accepted: usize,
    /// Accepted candidates that are *not* one of the fitted seeds — the
    /// mutants that reached a complete, valid, different model.
    novel: usize,
}

fn check(label: &str, decoder: &Decoder, bytes: &[u8], novel: bool, reach: &mut Reach) {
    let (name, decode) = *decoder;
    let (outcome, peak) = measure_peak(|| decode(bytes));
    let budget = ALLOC_BASE_BYTES + ALLOC_INPUT_FACTOR * bytes.len();
    let length = bytes.len();
    assert!(
        peak <= budget,
        "{label}/{name}: decoding {length} bytes allocated {peak} bytes, budget {budget}"
    );
    match outcome {
        // These three are only reachable once the checksum, magic, version,
        // kind, and schema identities have all been accepted, so counting them
        // proves mutants really are reaching the payload parsers.
        Outcome::Rejected(
            ArtifactError::InvalidPayload
            | ArtifactError::TrailingBytes
            | ArtifactError::UnsupportedPayloadVersion { .. },
        ) => reach.past_checksum += 1,
        Outcome::Rejected(_) => {}
        Outcome::Accepted(reencoded) => {
            reach.accepted += 1;
            reach.novel += usize::from(novel);
            // A version-1 logistic artifact is deliberately re-encoded as
            // version 2, so only current-format acceptances owe canonicity.
            if length >= 10 && bytes[..8] == *MAGIC && u16_at(bytes, 8) == 2 {
                match reencoded {
                    Err(error) => panic!(
                        "{label}/{name}: accepted {length} bytes it cannot re-encode: {error}"
                    ),
                    Ok(again) => assert!(
                        again == bytes,
                        "{label}/{name}: accepted a non-canonical encoding of {length} bytes"
                    ),
                }
            }
        }
    }
}

fn rounds() -> usize {
    std::env::var("FERRICML_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(ROUNDS)
}

#[test]
fn every_decoder_survives_structured_mutation() {
    let corpus = seed_corpus();
    let decoders = decoders();
    let mut rng = Rng::new(FUZZ_SEED);
    let mut reach = Reach::default();
    let rounds = rounds();

    for (seed_name, seed) in &corpus {
        for (strategy, strategy_name) in STRATEGIES.iter().enumerate() {
            for round in 0..rounds {
                let bytes = mutate(&mut rng, strategy, seed, &corpus);
                reach.candidates += 1;
                let label = format!("{seed_name}/{strategy_name}/{round}");
                let novel = corpus.iter().all(|(_, fitted)| *fitted != bytes);
                for decoder in &decoders {
                    check(&label, decoder, &bytes, novel, &mut reach);
                }
            }
        }
    }

    // A fuzzer nobody can see failing is worthless. These floors say the sweep
    // still generates work, still gets past the integrity footer, and still
    // produces inputs a decoder is willing to accept; a mutator that decayed
    // into producing garbage nothing parses would trip them.
    println!(
        "fuzz sweep: {} candidates, {} decoder rejections past the checksum, \
         {} acceptances ({} of models no seed contains)",
        reach.candidates, reach.past_checksum, reach.accepted, reach.novel
    );
    assert!(
        reach.candidates >= 500,
        "the sweep shrank to {} candidates",
        reach.candidates
    );
    assert!(
        reach.past_checksum >= 200,
        "only {} mutants reached a payload parser",
        reach.past_checksum
    );
    assert!(
        reach.accepted >= 10,
        "only {} mutants were accepted as models",
        reach.accepted
    );
    assert!(
        reach.novel >= 5,
        "only {} mutants built a model no seed already encodes; the sweep is \
         only re-testing its own corpus",
        reach.novel
    );
}

#[test]
fn valid_artifacts_decode_within_the_allocation_budget() {
    let decoders = decoders();
    let mut reach = Reach::default();
    for (name, bytes) in seed_corpus() {
        for decoder in &decoders {
            check(name, decoder, &bytes, false, &mut reach);
        }
    }
    assert!(
        reach.accepted >= 18,
        "only {} valid artifacts decoded",
        reach.accepted
    );
}

// ---------------------------------------------------------------------------
// The frozen adversarial corpus
// ---------------------------------------------------------------------------
//
// The sweep above is bounded so it can run inside `make gate`; a defect it
// takes a long campaign to rediscover would therefore escape. The corpus is
// the memory. Each entry is a checked-in byte string with one line of
// provenance, decoded in an ordinary test that asserts the exact typed error
// *and* the allocation bound — both matter, because the two over-allocation
// defects this corpus records returned the correct error the whole time and
// were visible only in what they allocated on the way to it.
//
// It shares this binary with the sweep because the allocation meter is a
// global allocator, which exists once per test binary.

/// Where the frozen bytes live, relative to the crate root.
const CORPUS_DIRECTORY: &str = "tests/fixtures/adversarial-artifacts";

struct Case {
    /// File stem under [`CORPUS_DIRECTORY`].
    name: &'static str,
    /// Why this byte string is in the corpus, in one line.
    provenance: &'static str,
    /// The decoder the case is aimed at.
    decoder: &'static str,
    /// Exactly what that decoder must answer.
    expected: ArtifactError,
    bytes: Vec<u8>,
}

const SCALER_ROLES: [(u16, [u8; 32]); 2] = [(1, INPUT_SCHEMA), (2, TRANSFORMED_SCHEMA)];
const MODEL_ROLES: [(u16, [u8; 32]); 1] = [(1, INPUT_SCHEMA)];

fn words(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// A state component wrapped straight into an envelope: the shape every
/// single-component artifact uses.
fn stated(kind: u16, roles: &[(u16, [u8; 32])], state: &[u8]) -> Vec<u8> {
    envelope(kind, 1, roles, &component(1, 1, state))
}

/// Forest metadata whose every field is the value a fitted model would write.
fn valid_forest_metadata(n_features: u32, trees: u32, nodes: u32) -> Vec<u8> {
    let mut bytes = words(&[1, n_features, trees, 4, 2, 1, 1, 0, 0]);
    bytes.extend_from_slice(&11_u64.to_le_bytes());
    bytes.extend_from_slice(&words(&[1, 0, trees, nodes]));
    bytes
}

/// Boosting metadata whose every field is the value a fitted model would write.
fn valid_boosting_metadata(n_features: u32, trees: u32, nodes: u32) -> Vec<u8> {
    words(&[
        1,
        n_features,
        0x3dcc_cccd,
        trees,
        4,
        0,
        1,
        0,
        8,
        0,
        trees,
        nodes,
    ])
}

fn tree_component(records: &[[u32; 5]], leaves: u32, depth: u32) -> Vec<u8> {
    let mut payload = words(&[records.len() as u32, leaves, depth]);
    for record in records {
        payload.extend_from_slice(&words(record));
    }
    component(2, 1, &payload)
}

fn branch_record(feature: u32, threshold: f32, left: u32, right: u32) -> [u32; 5] {
    [1, feature, threshold.to_bits(), left, right]
}

fn leaf_record(value: f32) -> [u32; 5] {
    [0, value.to_bits(), 0, 0, 0]
}

/// The five-record tree used by the topology cases, in canonical pre-order.
fn canonical_tree() -> [[u32; 5]; 5] {
    [
        branch_record(0, 1.0, 1, 4),
        branch_record(0, 2.0, 2, 3),
        leaf_record(1.0),
        leaf_record(3.0),
        leaf_record(2.0),
    ]
}

fn forest_with(records: &[[u32; 5]], leaves: u32, depth: u32) -> Vec<u8> {
    let mut payload = component(1, 1, &valid_forest_metadata(1, 1, records.len() as u32));
    payload.extend_from_slice(&tree_component(records, leaves, depth));
    envelope(10, 1, &MODEL_ROLES, &payload)
}

/// Copies `base` with `value` written at `offset`, resealing the footer.
fn overwrite(base: &[u8], offset: usize, value: &[u8]) -> Vec<u8> {
    let mut bytes = base.to_vec();
    bytes[offset..offset + value.len()].copy_from_slice(value);
    reseal(&mut bytes);
    bytes
}

fn corpus() -> Vec<Case> {
    let fitted: Vec<(&'static str, Vec<u8>)> = seed_corpus();
    let seed = |name: &str| -> Vec<u8> {
        fitted
            .iter()
            .find(|(seed_name, _)| *seed_name == name)
            .unwrap_or_else(|| panic!("no fitted seed named {name}"))
            .1
            .clone()
    };
    let logistic = seed("logistic");
    let scaler = seed("standard-scaler");
    let forest = seed("forest");
    let staged = seed("staged-two");
    let ranker = seed("pairwise-ranker");
    let any_ridge = seed("any-ridge");
    let (scaler_payload, _) = payload_span(&scaler).expect("scaler payload");
    let (logistic_payload, _) = payload_span(&logistic).expect("logistic payload");
    let (any_payload, _) = payload_span(&any_ridge).expect("dispatch payload");
    let (staged_payload, _) = payload_span(&staged).expect("staged payload");
    let (ranker_payload, _) = payload_span(&ranker).expect("ranker payload");

    // A declared element count far past the bytes present. Before the
    // reservation was clamped, each of these turned roughly 150 bytes into
    // between 4 MB and 32 MB of allocation before reporting `Truncated`.
    let inflated = 1_000_000_u32;
    let mut cases = vec![
        Case {
            name: "standard-scaler-inflated-width",
            provenance: "1e6 declared features, both counts agreeing, no feature bytes",
            decoder: "standard-scaler",
            expected: ArtifactError::Truncated,
            bytes: stated(4, &SCALER_ROLES, &words(&[inflated, 1, 1, inflated])),
        },
        Case {
            name: "min-max-scaler-inflated-width",
            provenance: "same shape against the two-field-per-feature scaler",
            decoder: "min-max-scaler",
            expected: ArtifactError::Truncated,
            bytes: stated(14, &SCALER_ROLES, &words(&[inflated, 0, inflated])),
        },
        Case {
            name: "max-abs-scaler-inflated-width",
            provenance: "same shape against the flagless scaler",
            decoder: "max-abs-scaler",
            expected: ArtifactError::Truncated,
            bytes: stated(15, &SCALER_ROLES, &words(&[inflated, inflated])),
        },
        Case {
            name: "logistic-inflated-coefficients",
            provenance: "1e6 declared coefficients with none encoded",
            decoder: "logistic",
            expected: ArtifactError::Truncated,
            bytes: stated(
                1,
                &MODEL_ROLES,
                &words(&[inflated, 1, 0x3f80_0000, 100, 0x3727_c5ac, 1, 0, inflated]),
            ),
        },
        Case {
            name: "linear-inflated-coefficients",
            provenance: "same shape against ordinary least squares",
            decoder: "linear",
            expected: ArtifactError::Truncated,
            bytes: stated(2, &MODEL_ROLES, &words(&[inflated, 1, 0, 1, 0, inflated])),
        },
        Case {
            name: "ridge-inflated-coefficients",
            provenance: "same shape against ridge",
            decoder: "ridge",
            expected: ArtifactError::Truncated,
            bytes: stated(3, &MODEL_ROLES, &words(&[inflated, 0, 1, 0, inflated])),
        },
        Case {
            name: "legacy-logistic-inflated-coefficients",
            provenance: "the same inflation through the version-1 envelope",
            decoder: "logistic",
            expected: ArtifactError::Truncated,
            bytes: {
                let mut bytes = legacy_logistic_seed(2);
                let truncate = bytes.len() - CHECKSUM_BYTES - 2 * 4;
                bytes.truncate(truncate);
                bytes[44..48].copy_from_slice(&inflated.to_le_bytes());
                bytes[72..76].copy_from_slice(&inflated.to_le_bytes());
                let digest = Sha256::digest(&bytes);
                bytes.extend_from_slice(&digest);
                bytes
            },
        },
        Case {
            name: "forest-inflated-tree-count",
            provenance: "4096 declared trees with no tree component encoded",
            decoder: "random-forest",
            expected: ArtifactError::Truncated,
            bytes: envelope(
                10,
                1,
                &MODEL_ROLES,
                &component(1, 1, &valid_forest_metadata(1, 4_096, 4_096)),
            ),
        },
        Case {
            name: "boosting-inflated-tree-count",
            provenance: "same shape against histogram boosting",
            decoder: "hist-gradient-boosting",
            expected: ArtifactError::Truncated,
            bytes: envelope(
                9,
                1,
                &MODEL_ROLES,
                &component(1, 1, &valid_boosting_metadata(1, 4_096, 4_096)),
            ),
        },
    ];

    // Envelope-level malformations.
    cases.extend([
        Case {
            name: "envelope-truncated-header",
            provenance: "ten bytes: shorter than the fixed header",
            decoder: "logistic",
            expected: ArtifactError::Truncated,
            bytes: logistic[..10].to_vec(),
        },
        Case {
            name: "envelope-bad-magic",
            provenance: "a fitted logistic artifact with one magic byte changed",
            decoder: "logistic",
            expected: ArtifactError::InvalidMagic,
            bytes: overwrite(&logistic, 0, b"X"),
        },
        Case {
            name: "envelope-unsupported-version",
            provenance: "envelope version 3, which no reader understands",
            decoder: "ridge",
            expected: ArtifactError::UnsupportedVersion { found: 3 },
            bytes: overwrite(&seed("ridge"), 8, &3_u16.to_le_bytes()),
        },
        Case {
            name: "envelope-unknown-required-flags",
            provenance: "one required flag bit set that this reader does not implement",
            decoder: "logistic",
            expected: ArtifactError::UnsupportedRequiredFlags { found: 1 },
            bytes: overwrite(&logistic, 14, &1_u16.to_le_bytes()),
        },
        Case {
            name: "envelope-nonzero-reserved",
            provenance: "the reserved header field carrying a value",
            decoder: "logistic",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&logistic, 22, &1_u16.to_le_bytes()),
        },
        Case {
            name: "envelope-wrong-kind",
            provenance: "a fitted logistic model relabelled as a ridge model",
            decoder: "logistic",
            expected: ArtifactError::UnsupportedModelKind { found: 3 },
            bytes: overwrite(&logistic, 10, &3_u16.to_le_bytes()),
        },
        Case {
            name: "envelope-checksum-mismatch",
            provenance: "one payload byte flipped without resealing the footer",
            decoder: "logistic",
            expected: ArtifactError::ChecksumMismatch,
            bytes: {
                let mut bytes = logistic.clone();
                bytes[logistic_payload] ^= 0x01;
                bytes
            },
        },
        Case {
            name: "envelope-payload-length-overflow",
            provenance: "a declared payload length of u32::MAX",
            decoder: "logistic",
            expected: ArtifactError::Truncated,
            bytes: overwrite(&logistic, 16, &u32::MAX.to_le_bytes()),
        },
        Case {
            name: "envelope-payload-length-short",
            provenance: "a payload length four bytes short of the payload present",
            decoder: "logistic",
            expected: ArtifactError::TrailingBytes,
            bytes: overwrite(&logistic, 16, &(u32_at(&logistic, 16) - 4).to_le_bytes()),
        },
        Case {
            name: "envelope-schema-count-inflated",
            provenance: "more schema records declared than the reader requires",
            decoder: "logistic",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&logistic, 20, &4_u16.to_le_bytes()),
        },
        Case {
            name: "envelope-schema-role-swapped",
            provenance: "a scaler's input and transformed schema roles exchanged",
            decoder: "standard-scaler",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut bytes = scaler.clone();
                bytes[24..26].copy_from_slice(&2_u16.to_le_bytes());
                bytes[60..62].copy_from_slice(&1_u16.to_le_bytes());
                reseal(&mut bytes);
                bytes
            },
        },
        Case {
            name: "envelope-schema-flags-set",
            provenance: "a schema record carrying flags the reader does not define",
            decoder: "logistic",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&logistic, 26, &1_u16.to_le_bytes()),
        },
        Case {
            name: "envelope-schema-hash-foreign",
            provenance: "a model bound to a feature schema the caller did not ask for",
            decoder: "logistic",
            expected: ArtifactError::FeatureSchemaMismatch,
            bytes: overwrite(&logistic, 28, &[0xab; 32]),
        },
    ]);

    // Component-level malformations.
    cases.extend([
        Case {
            name: "component-length-overflow",
            provenance: "a component claiming u32::MAX bytes of body",
            decoder: "logistic",
            expected: ArtifactError::Truncated,
            bytes: overwrite(&logistic, logistic_payload + 4, &u32::MAX.to_le_bytes()),
        },
        Case {
            name: "component-unknown-version",
            provenance: "a state component at a version this reader does not implement",
            decoder: "logistic",
            expected: ArtifactError::UnsupportedPayloadVersion { found: 9 },
            bytes: overwrite(&logistic, logistic_payload + 2, &9_u16.to_le_bytes()),
        },
        Case {
            name: "component-unknown-kind",
            provenance: "a state component relabelled as a component kind that is not expected",
            decoder: "logistic",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&logistic, logistic_payload, &7_u16.to_le_bytes()),
        },
        Case {
            name: "scaler-count-fields-disagree",
            provenance: "the declared width and the repeated element count differ",
            decoder: "standard-scaler",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&scaler, scaler_payload + 8, &7_u32.to_le_bytes()),
        },
        Case {
            name: "scaler-zero-width",
            provenance: "a fitted scaler declaring zero features",
            decoder: "standard-scaler",
            expected: ArtifactError::InvalidPayload,
            bytes: stated(4, &SCALER_ROLES, &words(&[0, 1, 1, 0])),
        },
        Case {
            name: "scaler-scale-disagrees-with-variance",
            provenance: "a stored scale that is not the square root of its variance",
            decoder: "standard-scaler",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut state = words(&[1, 1, 1, 1]);
                state.extend_from_slice(&0.0_f64.to_le_bits_vec());
                state.extend_from_slice(&4.0_f64.to_le_bits_vec());
                state.extend_from_slice(&3.0_f64.to_le_bits_vec());
                stated(4, &SCALER_ROLES, &state)
            },
        },
    ]);

    // Logical-tree malformations, all reaching the tree decoder through a
    // metadata component a fitted forest would have written.
    let mut over_limit = words(&[131_072, 65_537, 1]);
    over_limit.extend_from_slice(&words(&leaf_record(0.0)));
    cases.extend([
        Case {
            name: "tree-node-count-over-limit",
            provenance: "a node count one past the format ceiling",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut payload = component(1, 1, &valid_forest_metadata(1, 1, 1));
                payload.extend_from_slice(&component(2, 1, &over_limit));
                envelope(10, 1, &MODEL_ROLES, &payload)
            },
        },
        Case {
            name: "tree-leaf-count-mismatch",
            provenance: "a declared leaf count the records do not produce",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: forest_with(&canonical_tree(), 2, 2),
        },
        Case {
            name: "tree-depth-mismatch",
            provenance: "a declared depth the records do not produce",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: forest_with(&canonical_tree(), 3, 3),
        },
        Case {
            name: "tree-cyclic-child",
            provenance: "a branch whose right child is the root",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[1][4] = 0;
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-left-child-not-adjacent",
            provenance: "a branch whose left child is not the next record",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[1][3] = 3;
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-unreachable-record",
            provenance: "a record no path from the root ever visits",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree().to_vec();
                records.push(leaf_record(9.0));
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-non-finite-leaf",
            provenance: "a leaf holding NaN",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[2] = leaf_record(f32::NAN);
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-non-finite-threshold",
            provenance: "a branch threshold of positive infinity",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[1] = branch_record(0, f32::INFINITY, 2, 3);
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-feature-beyond-fitted-width",
            provenance: "a split on a feature index the fitted width does not have",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[1] = branch_record(9, 2.0, 2, 3);
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-leaf-sentinel-as-feature",
            provenance: "the packed layout's leaf sentinel smuggled in as a branch feature",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[1] = branch_record(u32::MAX, 2.0, 2, 3);
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-unknown-record-tag",
            provenance: "a record tag that is neither leaf nor branch",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[2][0] = 2;
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-leaf-padding-nonzero",
            provenance: "a leaf record whose reserved child fields are not zero",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut records = canonical_tree();
                records[2][2] = 1;
                forest_with(&records, 3, 2)
            },
        },
        Case {
            name: "tree-record-count-short",
            provenance: "a header declaring more records than the component carries",
            decoder: "random-forest",
            expected: ArtifactError::Truncated,
            bytes: {
                let mut payload = component(1, 1, &valid_forest_metadata(1, 1, 5));
                let mut tree = words(&[5, 3, 2]);
                for record in &canonical_tree()[..3] {
                    tree.extend_from_slice(&words(record));
                }
                payload.extend_from_slice(&component(2, 1, &tree));
                envelope(10, 1, &MODEL_ROLES, &payload)
            },
        },
        Case {
            name: "forest-node-total-disagrees",
            provenance: "metadata whose total node count is not what the trees decode to",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut payload = component(1, 1, &valid_forest_metadata(1, 1, 7));
                payload.extend_from_slice(&tree_component(&canonical_tree(), 3, 2));
                envelope(10, 1, &MODEL_ROLES, &payload)
            },
        },
        Case {
            name: "forest-estimator-count-disagrees",
            provenance: "metadata whose estimator count is not the tree count",
            decoder: "random-forest",
            expected: ArtifactError::InvalidPayload,
            bytes: {
                let mut metadata = valid_forest_metadata(1, 1, 5);
                metadata[8..12].copy_from_slice(&2_u32.to_le_bytes());
                let mut payload = component(1, 1, &metadata);
                payload.extend_from_slice(&tree_component(&canonical_tree(), 3, 2));
                envelope(10, 1, &MODEL_ROLES, &payload)
            },
        },
        Case {
            name: "forest-trailing-tree-component",
            provenance: "one more tree than the metadata declares",
            decoder: "random-forest",
            expected: ArtifactError::TrailingBytes,
            bytes: {
                let mut payload = component(1, 1, &valid_forest_metadata(1, 1, 5));
                payload.extend_from_slice(&tree_component(&canonical_tree(), 3, 2));
                payload.extend_from_slice(&tree_component(&canonical_tree(), 3, 2));
                envelope(10, 1, &MODEL_ROLES, &payload)
            },
        },
    ]);

    // Dispatch and composition envelopes.
    cases.extend([
        Case {
            name: "any-regressor-unknown-variant",
            provenance: "a dispatch tag no runtime variant claims",
            decoder: "any-regressor",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&any_ridge, any_payload + 12, &9_u32.to_le_bytes()),
        },
        Case {
            name: "any-regressor-variant-disagrees-with-payload",
            provenance: "a ridge model tagged as a random forest",
            decoder: "any-regressor",
            expected: ArtifactError::UnsupportedModelKind { found: 3 },
            bytes: overwrite(&any_ridge, any_payload + 12, &1_u32.to_le_bytes()),
        },
        Case {
            name: "staged-stage-tag-mismatch",
            provenance: "a composition recording a stage type it does not hold",
            decoder: "staged-two",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&staged, staged_payload + 16, &3_u32.to_le_bytes()),
        },
        Case {
            name: "staged-estimator-tag-mismatch",
            provenance: "a composition recording an estimator type it does not hold",
            decoder: "staged-two",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&staged, staged_payload + 12, &1_u32.to_le_bytes()),
        },
        Case {
            name: "staged-stage-count-mismatch",
            provenance: "a two-stage composition declaring three stages",
            decoder: "staged-two",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&staged, staged_payload + 8, &3_u32.to_le_bytes()),
        },
        Case {
            name: "pairwise-objective-version-unknown",
            provenance: "a ranker declaring an objective version this reader does not implement",
            decoder: "pairwise-ranker",
            expected: ArtifactError::InvalidPayload,
            bytes: overwrite(&ranker, ranker_payload + 8, &9_u32.to_le_bytes()),
        },
    ]);

    // The forest seed is a genuine artifact and belongs in the corpus as the
    // control: a corpus of only rejections cannot show the reader still works.
    cases.push(Case {
        name: "control-fitted-forest",
        provenance: "an unmodified fitted forest artifact, which must still decode",
        decoder: "random-forest",
        expected: ArtifactError::InvalidPayload,
        bytes: forest,
    });
    cases
}

trait LeBits {
    fn to_le_bits_vec(self) -> [u8; 8];
}

impl LeBits for f64 {
    fn to_le_bits_vec(self) -> [u8; 8] {
        self.to_bits().to_le_bytes()
    }
}

/// The one entry that must decode rather than be rejected.
const CONTROL_CASE: &str = "control-fitted-forest";

fn corpus_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(CORPUS_DIRECTORY)
        .join(format!("{name}.bin"))
}

/// Rewrites the checked-in corpus from the builders above.
///
/// Ignored by default: the gate reads the frozen bytes, so a builder that
/// drifted cannot silently change what the corpus tests. Run with
/// `cargo test --test artifact_hardening -- --ignored refresh` and review the
/// diff, exactly like a snapshot.
#[test]
#[ignore = "writes fixtures; the gate reads them instead"]
fn refresh_the_adversarial_corpus() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIRECTORY);
    std::fs::create_dir_all(&directory).expect("corpus directory");
    for case in corpus() {
        std::fs::write(corpus_path(case.name), &case.bytes).expect("write fixture");
    }
}

#[test]
fn the_frozen_adversarial_corpus_decodes_exactly_as_recorded() {
    let decoders = decoders();
    let cases = corpus();
    let mut reach = Reach::default();

    for case in &cases {
        let frozen = std::fs::read(corpus_path(case.name)).unwrap_or_else(|error| {
            panic!(
                "missing corpus fixture {}.bin ({error}); regenerate with \
                 `cargo test --test artifact_hardening -- --ignored refresh`",
                case.name
            )
        });
        assert_eq!(
            frozen, case.bytes,
            "{}: the checked-in bytes and the builder disagree",
            case.name
        );

        // The allocation bound is checked for every decoder, because a hostile
        // artifact aimed at one reader can still be handed to another.
        for decoder in &decoders {
            check(case.name, decoder, &frozen, false, &mut reach);
        }

        let (_, decode) = *decoders
            .iter()
            .find(|(name, _)| *name == case.decoder)
            .unwrap_or_else(|| panic!("{}: no decoder named {}", case.name, case.decoder));
        match decode(&frozen) {
            Outcome::Rejected(error) => {
                assert_ne!(
                    case.name, CONTROL_CASE,
                    "the control artifact stopped decoding: {error}"
                );
                assert_eq!(
                    error, case.expected,
                    "{} ({}): decoded to the wrong error",
                    case.name, case.provenance
                );
            }
            Outcome::Accepted(_) => assert_eq!(
                case.name, CONTROL_CASE,
                "{} ({}) was accepted; it must be rejected with {:?}",
                case.name, case.provenance, case.expected
            ),
        }
    }

    // A corpus that quietly lost entries would still pass every assertion
    // above, so the directory itself is checked against the manifest.
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIRECTORY);
    let on_disk = std::fs::read_dir(&directory)
        .expect("corpus directory")
        .filter(|entry| {
            entry
                .as_ref()
                .is_ok_and(|entry| entry.path().extension().is_some_and(|kind| kind == "bin"))
        })
        .count();
    assert_eq!(
        on_disk,
        cases.len(),
        "the corpus directory holds {on_disk} fixtures but the manifest lists {}",
        cases.len()
    );
    assert!(
        cases.len() >= 38,
        "the corpus shrank to {} cases",
        cases.len()
    );
}
