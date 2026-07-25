use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::ensemble::{
    MaxFeatures, RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use std::hint::black_box;

fn fixture(rows: usize, columns: usize) -> (DenseMatrix, BinaryTargets) {
    let mut values = Vec::with_capacity(rows * columns);
    let mut targets = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut score = 0.0_f32;
        for column in 0..columns {
            let value = (((row * 131 + column * 17) % 1009) as f32 / 504.5) - 1.0;
            values.push(value);
            if column < 4 {
                score += value * (column + 1) as f32;
            }
        }
        targets.push(u8::from(score > 0.0));
    }
    (
        DenseMatrix::new(values, rows, columns).unwrap(),
        BinaryTargets::new(targets).unwrap(),
    )
}

/// Regression targets derived from the shared fixture's separable score, so
/// the regressor lanes measure the same dataset the classifier lanes use.
fn regression_targets(labels: &BinaryTargets) -> RegressionTargets {
    RegressionTargets::new(
        labels
            .as_slice()
            .iter()
            .enumerate()
            .map(|(row, &label)| f32::from(label) * 4.0 + (row % 11) as f32)
            .collect(),
    )
    .unwrap()
}

fn regressor(
    rows: usize,
    columns: usize,
    trees: usize,
    max_depth: usize,
) -> (DenseMatrix, RandomForestRegressor) {
    let (data, labels) = fixture(rows, columns);
    let targets = regression_targets(&labels);
    let model = RandomForestRegressor::fit(
        &data.as_view(),
        &targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(trees)
            .with_max_depth(Some(max_depth))
            .with_max_features(MaxFeatures::All)
            .with_random_state(42),
    )
    .unwrap();
    (data, model)
}

fn classifier(rows: usize, columns: usize, trees: usize) -> (DenseMatrix, RandomForestClassifier) {
    let (data, targets) = fixture(rows, columns);
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(trees)
        .with_max_depth(Some(12))
        .with_max_features(MaxFeatures::Sqrt)
        .with_random_state(42);
    let model = RandomForestClassifier::fit(&data.as_view(), &targets, params).unwrap();
    (data, model)
}

fn inference(c: &mut Criterion) {
    for rows in [1, 32, 1024] {
        let columns = 64;
        let trees = 100;
        let (data, model) = classifier(2048, columns, trees);
        let input_values = data.as_slice()[..rows * columns].to_vec();
        let input = DenseMatrix::new(input_values, rows, columns).unwrap();
        let mut labels = vec![0; rows];
        let mut full_probabilities = vec![0.0; rows * model.classes().len()];
        let mut class_probabilities = vec![0.0; rows];
        let mut group = c.benchmark_group(format!("forest_historical_into_{rows}x64_100t"));
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::from_parameter("labels"), |bencher| {
            bencher.iter(|| {
                model
                    .predict_into(black_box(&input.as_view()), black_box(&mut labels))
                    .unwrap();
                black_box(&labels);
            });
        });
        group.bench_function(BenchmarkId::from_parameter("full_proba"), |bencher| {
            bencher.iter(|| {
                model
                    .predict_proba_into(
                        black_box(&input.as_view()),
                        black_box(&mut full_probabilities),
                    )
                    .unwrap();
                black_box(&full_probabilities);
            });
        });
        group.bench_function(BenchmarkId::from_parameter("class_proba"), |bencher| {
            bencher.iter(|| {
                model
                    .predict_class_proba_into(
                        black_box(&input.as_view()),
                        1,
                        black_box(&mut class_probabilities),
                    )
                    .unwrap();
                black_box(&class_probabilities);
            });
        });
        group.finish();
    }
}

fn training(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_historical_fit_2048x64_20t");
    let (rows, columns, trees) = (2048, 64, 20);
    let (data, targets) = fixture(rows, columns);
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(trees)
        .with_max_depth(Some(12))
        .with_max_features(MaxFeatures::Sqrt)
        .with_random_state(42);
    group.throughput(Throughput::Elements((rows * trees) as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter("ferricml"),
        &rows,
        |bencher, _| {
            bencher.iter_batched(
                || params.clone(),
                |params| {
                    black_box(
                        RandomForestClassifier::fit(&data.as_view(), &targets, params).unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
    group.finish();
}

/// Round-tripping a fitted forest through its artifact: encoding expands every
/// packed tree into logical records, and decoding revalidates each one before
/// rebuilding the packed layout.
fn artifact(c: &mut Criterion) {
    let (_, model) = regressor(512, 16, 32, 8);
    let schema = [42; 32];
    let encoded = model.to_artifact(schema).unwrap();

    let mut group = c.benchmark_group("ferricml_artifact_v1_forest_regressor_512x16_32t");
    group.throughput(Throughput::Bytes(encoded.len() as u64));
    group.bench_function(BenchmarkId::from_parameter("encode"), |bencher| {
        bencher.iter(|| {
            black_box(model.to_artifact(black_box(schema)).unwrap());
        });
    });
    group.bench_function(BenchmarkId::from_parameter("decode"), |bencher| {
        bencher.iter(|| {
            black_box(RandomForestRegressor::from_artifact(black_box(&encoded), schema).unwrap());
        });
    });
    group.finish();
}

/// Weighted fitting, beside the unweighted fit of the same workload.
///
/// Sample weights turned every node statistic from a row count into a weight
/// total on the crate's most benchmarked fitting path, so the two arms are
/// registered together: the unweighted arm is the one that must not move, and
/// the weighted arm is what the new capability costs.
fn weighted_training(c: &mut Criterion) {
    let (rows, columns, trees) = (2048, 64, 20);
    let (data, targets) = fixture(rows, columns);
    let regression = regression_targets(&targets);
    let weights = SampleWeights::new(
        (0..rows)
            .map(|row| 0.25 + ((row % 7) as f32) * 0.5)
            .collect(),
    )
    .unwrap();
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(trees)
        .with_max_depth(Some(12))
        .with_max_features(MaxFeatures::Sqrt)
        .with_random_state(42);
    let regressor_params = RandomForestRegressorParams::default()
        .with_n_estimators(trees)
        .with_max_depth(Some(12))
        .with_max_features(MaxFeatures::Sqrt)
        .with_random_state(42);

    let mut group = c.benchmark_group("ferricml_forest_v2_weighted_fit_2048x64_20t");
    group.throughput(Throughput::Elements((rows * trees) as u64));
    group.bench_function(
        BenchmarkId::from_parameter("classifier_unweighted"),
        |bencher| {
            bencher.iter_batched(
                || params.clone(),
                |params| {
                    black_box(
                        RandomForestClassifier::fit(&data.as_view(), &targets, params).unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
    group.bench_function(
        BenchmarkId::from_parameter("classifier_weighted"),
        |bencher| {
            bencher.iter_batched(
                || params.clone(),
                |params| {
                    black_box(
                        RandomForestClassifier::fit_weighted(
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
        },
    );
    group.bench_function(
        BenchmarkId::from_parameter("regressor_weighted"),
        |bencher| {
            bencher.iter_batched(
                || regressor_params.clone(),
                |params| {
                    black_box(
                        RandomForestRegressor::fit_weighted(
                            &data.as_view(),
                            &regression,
                            &weights,
                            params,
                        )
                        .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
    group.finish();
}

/// Round-tripping a fitted classifier through its artifact, in both leaf
/// representations. The multiclass arm carries a probability block per tree
/// that the binary arm does not, so the two are separate parameters.
fn classifier_artifact(c: &mut Criterion) {
    let (data, targets) = fixture(512, 16);
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(32)
        .with_max_depth(Some(8))
        .with_max_features(MaxFeatures::All)
        .with_random_state(42);
    let binary = RandomForestClassifier::fit(&data.as_view(), &targets, params.clone()).unwrap();
    let classes = ClassTargets::new(
        targets
            .as_slice()
            .iter()
            .enumerate()
            .map(|(row, &label)| if row % 5 == 0 { 10 } else { label * 7 })
            .collect(),
    )
    .unwrap();
    let multiclass =
        RandomForestClassifier::fit_multiclass(&data.as_view(), &classes, params).unwrap();
    let schema = [42; 32];
    let binary_encoded = binary.to_artifact(schema).unwrap();
    let multiclass_encoded = multiclass.to_artifact(schema).unwrap();

    let mut group = c.benchmark_group("ferricml_artifact_v1_forest_classifier_512x16_32t");
    group.throughput(Throughput::Bytes(binary_encoded.len() as u64));
    group.bench_function(BenchmarkId::from_parameter("encode"), |bencher| {
        bencher.iter(|| {
            black_box(binary.to_artifact(black_box(schema)).unwrap());
        });
    });
    group.bench_function(BenchmarkId::from_parameter("decode"), |bencher| {
        bencher.iter(|| {
            black_box(
                RandomForestClassifier::from_artifact(black_box(&binary_encoded), schema).unwrap(),
            );
        });
    });
    group.bench_function(
        BenchmarkId::from_parameter("multiclass_encode"),
        |bencher| {
            bencher.iter(|| {
                black_box(multiclass.to_artifact(black_box(schema)).unwrap());
            });
        },
    );
    group.bench_function(
        BenchmarkId::from_parameter("multiclass_decode"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    RandomForestClassifier::from_artifact(black_box(&multiclass_encoded), schema)
                        .unwrap(),
                );
            });
        },
    );
    group.finish();
}

/// Caller-owned batch regression inference.
///
/// The classifier lanes above cover label and probability prediction, but the
/// regressor averaging path had no lane at all. Averaging is validated once
/// per row, so this measures that check inside the loop rather than around it.
fn regressor_inference(c: &mut Criterion) {
    let columns = 64;
    let trees = 100;
    let (data, model) = regressor(2048, columns, trees, 12);
    for rows in [32, 1024] {
        let input_values = data.as_slice()[..rows * columns].to_vec();
        let input = DenseMatrix::new(input_values, rows, columns).unwrap();
        let mut predictions = vec![0.0; rows];
        let mut group =
            c.benchmark_group(format!("ferricml_forest_v1_regressor_into_{rows}x64_100t"));
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::from_parameter("predict"), |bencher| {
            bencher.iter(|| {
                model
                    .predict_into(black_box(&input.as_view()), black_box(&mut predictions))
                    .unwrap();
                black_box(&predictions);
            });
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    inference,
    training,
    artifact,
    classifier_artifact,
    weighted_training,
    regressor_inference
);
criterion_main!(benches);
