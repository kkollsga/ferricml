//! Matched, opt-in performance contract against external Rust implementations.
//!
//! The locked FerricML/Rafor lanes compare the same public operation and output
//! shape. FerricML's caller-owned extensions are reported separately because
//! Rafor 0.3 does not expose equivalent allocation-free batch methods.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::ensemble::{MaxFeatures, RandomForestClassifier, RandomForestClassifierParams};
use rafor::prelude::{
    ClassDecode, CommonTrainerBuilder, EnsembleTrainerBuilder, MaxFeaturesPolicy,
};
use std::hint::black_box;

const FEATURES: usize = 64;
const FIT_ROWS: usize = 2_048;
const FIT_TREES: usize = 20;
const PREDICT_TREES: usize = 100;
const PREDICT_ROWS: [usize; 3] = [1, 32, 1_024];
const SEED: u64 = 42;

#[derive(Clone)]
struct Fixture {
    values: Vec<f32>,
    labels: Vec<u8>,
    rows: usize,
    columns: usize,
}

impl Fixture {
    fn ferric_matrix(&self) -> DenseMatrix {
        DenseMatrix::new(self.values.clone(), self.rows, self.columns).unwrap()
    }

    fn ferric_targets(&self) -> BinaryTargets {
        BinaryTargets::new(self.labels.clone()).unwrap()
    }

    fn prefix_matrix(&self, rows: usize) -> DenseMatrix {
        DenseMatrix::new(
            self.values[..rows * self.columns].to_vec(),
            rows,
            self.columns,
        )
        .unwrap()
    }
}

fn fixture(rows: usize, columns: usize, seed: u64) -> Fixture {
    let mut rng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(rows * columns);
    let mut labels = Vec::with_capacity(rows);
    for _ in 0..rows {
        let start = values.len();
        for _ in 0..columns {
            values.push(rng.signed_unit());
        }
        let row = &values[start..];
        let score = 1.4 * row[0] * row[1] + 0.9 * row[2] - 0.8 * row[3] * row[3]
            + 0.5 * (3.0 * row[4]).sin()
            + 0.25 * row[5] * row[6];
        labels.push(u8::from(score > 0.0));
    }
    Fixture {
        values,
        labels,
        rows,
        columns,
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn signed_unit(&mut self) -> f32 {
        let fraction = (self.next() >> 40) as f32 / (1_u32 << 24) as f32;
        fraction * 2.0 - 1.0
    }
}

fn ferric_params(trees: usize) -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(trees)
        .with_max_depth(Some(12))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::Sqrt)
        .with_bootstrap(true)
        .with_random_state(SEED)
}

fn rafor_trainer(trees: usize) -> rafor::ensemble_classifier::Trainer {
    let mut trainer = rafor::rf::Classifier::trainer();
    trainer
        .with_trees(trees)
        .with_threads(1)
        .with_max_depth(12)
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeaturesPolicy::SQRT)
        .with_seed(SEED);
    trainer
}

fn fit_benchmarks(c: &mut Criterion) {
    let fixture = fixture(FIT_ROWS, FEATURES, 0xfeed_beef);
    let ferric_matrix = fixture.ferric_matrix();
    let ferric_targets = fixture.ferric_targets();
    let rafor_labels: Vec<i64> = fixture
        .labels
        .iter()
        .map(|&label| i64::from(label))
        .collect();

    let mut group = c.benchmark_group("forest_contract_fit_2048x64_20t");
    group.throughput(Throughput::Elements((FIT_ROWS * FIT_TREES) as u64));
    group.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
        bencher.iter_batched(
            || ferric_params(FIT_TREES),
            |params| {
                black_box(
                    RandomForestClassifier::fit(&ferric_matrix.as_view(), &ferric_targets, params)
                        .unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function(BenchmarkId::from_parameter("rafor"), |bencher| {
        bencher.iter_batched(
            || rafor_trainer(FIT_TREES),
            |trainer| {
                black_box(trainer.train(&fixture.values, &rafor_labels));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn prediction_benchmarks(c: &mut Criterion) {
    let train = fixture(FIT_ROWS, FEATURES, 0xfeed_beef);
    let test = fixture(*PREDICT_ROWS.last().unwrap(), FEATURES, 0xdeca_fbad);
    let ferric_train = train.ferric_matrix();
    let ferric_targets = train.ferric_targets();
    let ferric = RandomForestClassifier::fit(
        &ferric_train.as_view(),
        &ferric_targets,
        ferric_params(PREDICT_TREES),
    )
    .unwrap();
    let rafor_labels: Vec<i64> = train.labels.iter().map(|&label| i64::from(label)).collect();
    let rafor = rafor_trainer(PREDICT_TREES).train(&train.values, &rafor_labels);
    let rafor_positive_index = rafor
        .get_decode_table()
        .iter()
        .position(|&label| label == 1)
        .unwrap();
    assert_eq!(ferric.classes(), &[0, 1]);
    assert_eq!(rafor.num_classes(), 2);

    for rows in PREDICT_ROWS {
        let ferric_input = test.prefix_matrix(rows);
        let rafor_input = &test.values[..rows * FEATURES];
        let mut labels_into = vec![0_u8; rows];
        let mut full_proba_into = vec![0.0_f32; rows * 2];
        let mut class_proba_into = vec![0.0_f32; rows];

        let mut labels = c.benchmark_group(format!("forest_contract_labels_alloc_{rows}x64_100t"));
        labels.throughput(Throughput::Elements(rows as u64));
        labels.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
            bencher.iter(|| {
                black_box(ferric.predict(black_box(&ferric_input.as_view())).unwrap());
            });
        });
        labels.bench_function(BenchmarkId::from_parameter("rafor"), |bencher| {
            bencher.iter(|| {
                black_box(rafor.predict(black_box(rafor_input), 1));
            });
        });
        labels.finish();

        let mut full_proba =
            c.benchmark_group(format!("forest_contract_full_proba_alloc_{rows}x64_100t"));
        full_proba.throughput(Throughput::Elements(rows as u64));
        full_proba.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
            bencher.iter(|| {
                black_box(
                    ferric
                        .predict_proba(black_box(&ferric_input.as_view()))
                        .unwrap(),
                );
            });
        });
        full_proba.bench_function(BenchmarkId::from_parameter("rafor"), |bencher| {
            bencher.iter(|| {
                black_box(rafor.proba(black_box(rafor_input), 1));
            });
        });
        full_proba.finish();

        let mut class_proba =
            c.benchmark_group(format!("forest_contract_class_proba_alloc_{rows}x64_100t"));
        class_proba.throughput(Throughput::Elements(rows as u64));
        class_proba.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
            bencher.iter(|| {
                black_box(
                    ferric
                        .predict_class_proba(black_box(&ferric_input.as_view()), 1)
                        .unwrap(),
                );
            });
        });
        class_proba.bench_function(BenchmarkId::from_parameter("rafor"), |bencher| {
            bencher.iter(|| {
                let all = rafor.proba(black_box(rafor_input), 1);
                let selected: Vec<f32> = all
                    .chunks_exact(rafor.num_classes())
                    .map(|row| row[rafor_positive_index])
                    .collect();
                black_box(selected);
            });
        });
        class_proba.finish();

        // FerricML-only caller-owned extensions. Rafor 0.3 has no equivalent
        // batch APIs, so these are historical gates rather than head-to-head
        // comparisons.
        let mut into = c.benchmark_group(format!("forest_contract_into_{rows}x64_100t"));
        into.throughput(Throughput::Elements(rows as u64));
        into.bench_function(BenchmarkId::from_parameter("labels"), |bencher| {
            bencher.iter(|| {
                ferric
                    .predict_into(
                        black_box(&ferric_input.as_view()),
                        black_box(&mut labels_into),
                    )
                    .unwrap();
                black_box(&labels_into);
            });
        });
        into.bench_function(BenchmarkId::from_parameter("full_proba"), |bencher| {
            bencher.iter(|| {
                ferric
                    .predict_proba_into(
                        black_box(&ferric_input.as_view()),
                        black_box(&mut full_proba_into),
                    )
                    .unwrap();
                black_box(&full_proba_into);
            });
        });
        into.bench_function(BenchmarkId::from_parameter("class_proba"), |bencher| {
            bencher.iter(|| {
                ferric
                    .predict_class_proba_into(
                        black_box(&ferric_input.as_view()),
                        1,
                        black_box(&mut class_proba_into),
                    )
                    .unwrap();
                black_box(&class_proba_into);
            });
        });
        into.finish();
    }
}

criterion_group!(benches, fit_benchmarks, prediction_benchmarks);
criterion_main!(benches);
