use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::api::ProbabilisticClassifier;
use ferricml::artifact::ModelArtifact;
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::datasets::{BenchmarkFixture, BenchmarkLane, Target};
use ferricml::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams,
};
use std::hint::black_box;

const TRAIN_ROWS: usize = 2_048;
const COLUMNS: usize = 48;
const SCALAR_REPETITIONS: usize = 256;

/// The suite's design matrix and its six-term regression target.
///
/// The generator owns the expression; this is the shape adapter. It moved into
/// `ferricml::datasets` so the bytes every `bench-history` baseline was measured
/// on are pinned by digest in the crate's own test suite rather than living
/// untested in a benchmark file — see
/// `the_absorbed_benchmark_fixtures_reproduce_their_recorded_bytes`.
fn fixture(rows: usize, columns: usize) -> (DenseMatrix, RegressionTargets) {
    let dataset = BenchmarkFixture::new(BenchmarkLane::BoostingRegression, rows, columns)
        .unwrap()
        .generate();
    let targets = match dataset.target() {
        Some(Target::Regression(targets)) => targets.clone(),
        other => panic!("the boosting regression lane produced {other:?}"),
    };
    (dataset.into_features(), targets)
}

fn params(trees: usize, leaves: usize) -> HistGradientBoostingRegressorParams {
    HistGradientBoostingRegressorParams::default()
        .with_max_iter(trees)
        .with_max_leaf_nodes(leaves)
        .with_min_samples_leaf(20)
}

fn u32_at(bytes: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
}

fn artifact_stats(model: &HistGradientBoostingRegressor) -> (usize, usize) {
    let bytes = model.to_artifact([41; 32]).unwrap();
    let payload_end = bytes.len() - 32;
    let metadata_payload_bytes = u32_at(&bytes, 64);
    let declared_nodes = u32_at(&bytes, 112);
    let mut cursor = 60 + 8 + metadata_payload_bytes;
    let mut actual_nodes = 0_usize;
    while cursor < payload_end {
        assert_eq!(
            u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()),
            2
        );
        let component_bytes = u32_at(&bytes, cursor + 4);
        actual_nodes += u32_at(&bytes, cursor + 8);
        cursor += 8 + component_bytes;
    }
    assert_eq!(cursor, payload_end);
    assert_eq!(actual_nodes, declared_nodes);
    (actual_nodes, bytes.len())
}

fn report_metadata(name: &str, trees: usize, leaves: usize, model: &HistGradientBoostingRegressor) {
    let (logical_nodes, artifact_bytes) = artifact_stats(model);
    eprintln!(
        "FERRICML_BENCH_METADATA {{\"model\":\"{name}\",\"trees\":{trees},\"max_leaf_nodes\":{leaves},\"logical_nodes\":{logical_nodes},\"artifact_bytes\":{artifact_bytes}}}"
    );
}

fn inference(c: &mut Criterion) {
    let (training, targets) = fixture(TRAIN_ROWS, COLUMNS);
    for (trees, leaves) in [(32, 7), (64, 7), (64, 15), (128, 15)] {
        let model = HistGradientBoostingRegressor::fit(
            &training.as_view(),
            &targets,
            params(trees, leaves),
        )
        .unwrap();
        let name = format!("{trees}t{leaves}l");
        report_metadata(&name, trees, leaves, &model);
        let row = training.row(0).unwrap();
        let mut group = c.benchmark_group(format!(
            "ferricml_boosting_v1_predict_one_{trees}t{leaves}l"
        ));
        group.bench_function(BenchmarkId::from_parameter("predict"), |bencher| {
            bencher.iter(|| black_box(model.predict_one(black_box(row)).unwrap()));
        });
        group.finish();

        let scalar_rows = training
            .as_slice()
            .chunks_exact(COLUMNS)
            .take(SCALAR_REPETITIONS)
            .collect::<Vec<_>>();
        let mut group = c.benchmark_group(format!(
            "ferricml_boosting_v2_predict_one_256x_{trees}t{leaves}l"
        ));
        group.throughput(Throughput::Elements(SCALAR_REPETITIONS as u64));
        group.bench_function(BenchmarkId::from_parameter("predict"), |bencher| {
            bencher.iter(|| {
                for row in &scalar_rows {
                    black_box(model.predict_one(black_box(row)).unwrap());
                }
            });
        });
        group.finish();

        if (trees, leaves) == (64, 7) {
            for rows in [32, 1_024] {
                let input = DenseMatrix::new(
                    training.as_slice()[..rows * COLUMNS].to_vec(),
                    rows,
                    COLUMNS,
                )
                .unwrap();
                let mut output = vec![0.0; rows];
                let mut group =
                    c.benchmark_group(format!("ferricml_boosting_v1_into_{rows}x48_64t7l"));
                group.throughput(Throughput::Elements(rows as u64));
                group.bench_function(BenchmarkId::from_parameter("predict"), |bencher| {
                    bencher.iter(|| {
                        model
                            .predict_into(black_box(&input.as_view()), black_box(&mut output))
                            .unwrap();
                        black_box(&output);
                    });
                });
                group.finish();
            }
        }
    }
}

fn training(c: &mut Criterion) {
    let (data, targets) = fixture(TRAIN_ROWS, COLUMNS);
    let mut group = c.benchmark_group("ferricml_boosting_v1_fit_2048x48_64t7l");
    group.throughput(Throughput::Elements(TRAIN_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
        bencher.iter_batched(
            || params(64, 7),
            |params| {
                black_box(
                    HistGradientBoostingRegressor::fit(&data.as_view(), &targets, params).unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Weighted fitting beside the unweighted fit of the same workload.
///
/// The histogram scan now accumulates a per-bin weight total where it counted
/// rows, so the unweighted arm is registered alongside the weighted one: it is
/// the arm that must not move.
fn weighted_training(c: &mut Criterion) {
    let (data, targets) = fixture(TRAIN_ROWS, COLUMNS);
    let weights = SampleWeights::new(
        (0..TRAIN_ROWS)
            .map(|row| 0.25 + ((row % 7) as f32) * 0.5)
            .collect(),
    )
    .unwrap();
    let mut group = c.benchmark_group("ferricml_boosting_v2_weighted_fit_2048x48_64t7l");
    group.throughput(Throughput::Elements(TRAIN_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("unweighted"), |bencher| {
        bencher.iter_batched(
            || params(64, 7),
            |params| {
                black_box(
                    HistGradientBoostingRegressor::fit(&data.as_view(), &targets, params).unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function(BenchmarkId::from_parameter("weighted"), |bencher| {
        bencher.iter_batched(
            || params(64, 7),
            |params| {
                black_box(
                    HistGradientBoostingRegressor::fit_weighted(
                        &data.as_view(),
                        &targets,
                        &weights,
                        params,
                    )
                    .unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Binary labels over the same fixture, split at its median-ish threshold.
///
/// Using one fixture for both estimators keeps the classifier lanes comparable
/// with the regressor lanes they sit beside: the binning work and the tree
/// shapes are the same, and what differs is the objective.
fn binary_targets(targets: &RegressionTargets) -> BinaryTargets {
    BinaryTargets::new(
        targets
            .as_slice()
            .iter()
            .map(|&value| u8::from(value > 0.0))
            .collect(),
    )
    .unwrap()
}

fn classifier_params(trees: usize, leaves: usize) -> HistGradientBoostingClassifierParams {
    HistGradientBoostingClassifierParams::default()
        .with_max_iter(trees)
        .with_max_leaf_nodes(leaves)
        .with_min_samples_leaf(20)
}

/// Boosted classification fitting, beside the regressor lane it shares a grower
/// with.
///
/// The classifier is the first consumer of the per-sample hessian arm, so this
/// is the lane that would move if that arm's cost ever leaked into the
/// constant-hessian one, and the regressor's own fit lane is the arm that must
/// not move.
fn classifier_training(c: &mut Criterion) {
    let (data, targets) = fixture(TRAIN_ROWS, COLUMNS);
    let labels = binary_targets(&targets);
    let mut group = c.benchmark_group("ferricml_boosting_v3_classifier_fit_2048x48_64t7l");
    group.throughput(Throughput::Elements(TRAIN_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
        bencher.iter_batched(
            || classifier_params(64, 7),
            |params| {
                black_box(
                    HistGradientBoostingClassifier::fit(&data.as_view(), &labels, params).unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Boosted classification inference: the raw score, its sigmoid, and the
/// allocation-free probability matrix.
fn classifier_inference(c: &mut Criterion) {
    let (training, targets) = fixture(TRAIN_ROWS, COLUMNS);
    let labels = binary_targets(&targets);
    let model =
        HistGradientBoostingClassifier::fit(&training.as_view(), &labels, classifier_params(64, 7))
            .unwrap();

    let scalar_rows = training
        .as_slice()
        .chunks_exact(COLUMNS)
        .take(SCALAR_REPETITIONS)
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("ferricml_boosting_v3_classifier_predict_one_256x_64t7l");
    group.throughput(Throughput::Elements(SCALAR_REPETITIONS as u64));
    group.bench_function(BenchmarkId::from_parameter("predict"), |bencher| {
        bencher.iter(|| {
            for row in &scalar_rows {
                black_box(model.predict_one(black_box(row)).unwrap());
            }
        });
    });
    group.finish();

    for rows in [32, 1_024] {
        let input = DenseMatrix::new(
            training.as_slice()[..rows * COLUMNS].to_vec(),
            rows,
            COLUMNS,
        )
        .unwrap();
        let mut output = vec![0.0; rows * 2];
        let mut group = c.benchmark_group(format!(
            "ferricml_boosting_v3_classifier_proba_into_{rows}x48_64t7l"
        ));
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::from_parameter("predict_proba"), |bencher| {
            bencher.iter(|| {
                ProbabilisticClassifier::predict_proba_into(
                    &model,
                    black_box(&input.as_view()),
                    black_box(&mut output),
                )
                .unwrap();
                black_box(&output);
            });
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    inference,
    training,
    weighted_training,
    classifier_training,
    classifier_inference
);
criterion_main!(benches);
