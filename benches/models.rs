use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams, Ridge, RidgeParams};
use ferricml::pipeline::Pipeline;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
};
use std::hint::black_box;

const ROWS: usize = 2_048;
const COLUMNS: usize = 48;
const INFERENCE_ROWS: usize = 1_024;
const PAIRS: usize = 1_024;

fn fixture(rows: usize, columns: usize) -> (DenseMatrix, RegressionTargets) {
    let mut state = 0x9e37_79b9_u32;
    let mut values = Vec::with_capacity(rows * columns);
    let mut targets = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut selected = [0.0_f32; 6];
        for column in 0..columns {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            values.push(value);
            if column < selected.len() {
                selected[column] = value;
            }
        }
        let nonlinear = selected[2] * selected[3]
            + if selected[4] > 0.0 { 0.8 } else { -0.8 }
            + 0.25 * ((row % 11) as f32 - 5.0);
        targets.push(1.7 * selected[0] - 0.9 * selected[1] + nonlinear);
    }
    (
        DenseMatrix::new(values, rows, columns).unwrap(),
        RegressionTargets::new(targets).unwrap(),
    )
}

fn observations(targets: &RegressionTargets, count: usize) -> Vec<PairwiseObservation> {
    (0..count)
        .map(|index| {
            let left = index % targets.len();
            let mut right = (index.wrapping_mul(37) + 17) % targets.len();
            if right == left {
                right = (right + 1) % targets.len();
            }
            let outcome = if targets.as_slice()[left] > targets.as_slice()[right] {
                PairOutcome::LeftPreferred
            } else {
                PairOutcome::RightPreferred
            };
            PairwiseObservation::new(PairIndex::new(left, right).unwrap(), outcome, 1.0).unwrap()
        })
        .collect()
}

fn inference(c: &mut Criterion) {
    let (data, targets) = fixture(INFERENCE_ROWS, COLUMNS);
    let linear =
        LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
            .unwrap();
    let ridge = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
    let pair_data = observations(&targets, PAIRS);
    let ranker = PairwiseLinearRanker::fit(
        &data.as_view(),
        &pair_data,
        PairwiseLinearRankerParams::default().with_max_iter(40),
    )
    .unwrap();
    let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default()).unwrap();
    let transformed = scaler.transform(&data.as_view()).unwrap();
    let pipeline_model =
        Ridge::fit(&transformed.as_view(), &targets, RidgeParams::default()).unwrap();
    let pipeline = Pipeline::new(scaler, pipeline_model).unwrap();

    let mut output = vec![0.0; INFERENCE_ROWS];
    let mut workspace = vec![0.0; INFERENCE_ROWS * COLUMNS];
    let mut group = c.benchmark_group("ferricml_models_v1_into_1024x48");
    group.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("linear"), |bencher| {
        bencher.iter(|| {
            linear
                .predict_into(black_box(&data.as_view()), black_box(&mut output))
                .unwrap();
            black_box(&output);
        });
    });
    group.bench_function(BenchmarkId::from_parameter("ridge"), |bencher| {
        bencher.iter(|| {
            ridge
                .predict_into(black_box(&data.as_view()), black_box(&mut output))
                .unwrap();
            black_box(&output);
        });
    });
    group.bench_function(BenchmarkId::from_parameter("ranker_scores"), |bencher| {
        bencher.iter(|| {
            ranker
                .score_items_into(black_box(&data.as_view()), black_box(&mut output))
                .unwrap();
            black_box(&output);
        });
    });
    group.bench_function(
        BenchmarkId::from_parameter("scaler_ridge_pipeline"),
        |bencher| {
            bencher.iter(|| {
                pipeline
                    .predict_into(
                        black_box(&data.as_view()),
                        black_box(&mut workspace),
                        black_box(&mut output),
                    )
                    .unwrap();
                black_box((&workspace, &output));
            });
        },
    );
    group.finish();
}

fn training(c: &mut Criterion) {
    let (data, targets) = fixture(ROWS, COLUMNS);
    let pair_data = observations(&targets, PAIRS);
    let mut group = c.benchmark_group("ferricml_models_v1_fit_2048x48");
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("linear"), |bencher| {
        bencher.iter_batched(
            LinearRegressionParams::default,
            |params| {
                black_box(LinearRegression::fit(&data.as_view(), &targets, params).unwrap());
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function(BenchmarkId::from_parameter("ridge"), |bencher| {
        bencher.iter_batched(
            RidgeParams::default,
            |params| {
                black_box(Ridge::fit(&data.as_view(), &targets, params).unwrap());
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function(
        BenchmarkId::from_parameter("ranker_1024_pairs"),
        |bencher| {
            bencher.iter_batched(
                || PairwiseLinearRankerParams::default().with_max_iter(40),
                |params| {
                    black_box(
                        PairwiseLinearRanker::fit(&data.as_view(), &pair_data, params).unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
    group.bench_function(
        BenchmarkId::from_parameter("scaler_ridge_pipeline"),
        |bencher| {
            bencher.iter(|| {
                let scaler = StandardScaler::fit(
                    black_box(&data.as_view()),
                    StandardScalerParams::default(),
                )
                .unwrap();
                let transformed = scaler.transform(&data.as_view()).unwrap();
                let model =
                    Ridge::fit(&transformed.as_view(), &targets, RidgeParams::default()).unwrap();
                black_box(Pipeline::new(scaler, model).unwrap());
            });
        },
    );
    group.finish();
}

criterion_group!(benches, inference, training);
criterion_main!(benches);
