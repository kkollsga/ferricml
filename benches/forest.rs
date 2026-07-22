use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::ensemble::{MaxFeatures, RandomForestClassifier, RandomForestClassifierParams};
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

criterion_group!(benches, inference, training);
criterion_main!(benches);
