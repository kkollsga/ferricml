use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::api::Classifier;
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::dummy::{
    DummyClassifier, DummyClassifierParams, DummyRegressor, DummyRegressorParams,
};
use ferricml::ensemble::{MaxFeatures, RandomForestRegressor, RandomForestRegressorParams};
use ferricml::inspection::{PermutationImportanceParams, permutation_importance_regressor_into};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::metrics::{mean_squared_error, roc_auc_score};
use ferricml::model_selection::{
    HoldoutParams, KFold, RegressionScorer, TestSize, cross_validate_regressor,
    stratified_train_test_split, train_test_split,
};
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
const METRIC_ROWS: usize = 4_096;
const CV_ROWS: usize = 256;
const CV_COLUMNS: usize = 12;
const HOLDOUT_ROWS: usize = 1_000_000;
const MANY_CLASS_ROWS: usize = 262_144;
const INSPECTION_ROWS: usize = 256;
const INSPECTION_COLUMNS: usize = 8;
const INSPECTION_REPEATS: usize = 3;

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

fn logistic_and_scaler(c: &mut Criterion) {
    let (training, regression_targets) = fixture(ROWS, COLUMNS);
    let labels = BinaryTargets::new(
        regression_targets
            .as_slice()
            .iter()
            .map(|&target| u8::from(target > 0.0))
            .collect(),
    )
    .unwrap();
    let logistic_params = LogisticRegressionParams::default().with_max_iter(25);
    let logistic =
        LogisticRegression::fit(&training.as_view(), &labels, logistic_params.clone()).unwrap();
    let scaler = StandardScaler::fit(&training.as_view(), StandardScalerParams::default()).unwrap();
    let inference = DenseMatrix::new(
        training.as_slice()[..INFERENCE_ROWS * COLUMNS].to_vec(),
        INFERENCE_ROWS,
        COLUMNS,
    )
    .unwrap();

    eprintln!(
        "FERRICML_BENCH_METADATA {{\"suite\":\"models-v2\",\"rows\":{ROWS},\"features\":{COLUMNS},\"logistic_max_iter\":25}}"
    );

    let mut probabilities = vec![0.0; INFERENCE_ROWS * logistic.classes().len()];
    let mut logistic_group = c.benchmark_group("ferricml_models_v2_logistic_into_1024x48");
    logistic_group.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    logistic_group.bench_function(BenchmarkId::from_parameter("proba"), |bencher| {
        bencher.iter(|| {
            logistic
                .predict_proba_into(
                    black_box(&inference.as_view()),
                    black_box(&mut probabilities),
                )
                .unwrap();
            black_box(&probabilities);
        });
    });
    logistic_group.finish();

    let mut transformed = vec![0.0; INFERENCE_ROWS * COLUMNS];
    let mut scaler_group = c.benchmark_group("ferricml_models_v2_scaler_into_1024x48");
    scaler_group.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    scaler_group.bench_function(BenchmarkId::from_parameter("transform"), |bencher| {
        bencher.iter(|| {
            black_box(
                scaler
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    scaler_group.finish();

    let mut logistic_fit = c.benchmark_group("ferricml_models_v2_logistic_fit_2048x48");
    logistic_fit.throughput(Throughput::Elements(ROWS as u64));
    logistic_fit.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
        bencher.iter_batched(
            || logistic_params.clone(),
            |params| {
                black_box(LogisticRegression::fit(&training.as_view(), &labels, params).unwrap());
            },
            BatchSize::SmallInput,
        );
    });
    logistic_fit.finish();

    let mut scaler_fit = c.benchmark_group("ferricml_models_v2_scaler_fit_2048x48");
    scaler_fit.throughput(Throughput::Elements(ROWS as u64));
    scaler_fit.bench_function(BenchmarkId::from_parameter("ferricml"), |bencher| {
        bencher.iter(|| {
            black_box(
                StandardScaler::fit(
                    black_box(&training.as_view()),
                    StandardScalerParams::default(),
                )
                .unwrap(),
            );
        });
    });
    scaler_fit.finish();
}

fn evaluation(c: &mut Criterion) {
    let expected = (0..METRIC_ROWS)
        .map(|index| ((index % 97) as f32 - 48.0) / 11.0)
        .collect::<Vec<_>>();
    let predicted = expected
        .iter()
        .enumerate()
        .map(|(index, &value)| value + (index % 13) as f32 * 0.01)
        .collect::<Vec<_>>();
    let labels = (0..METRIC_ROWS)
        .map(|index| u8::from(index % 3 == 0))
        .collect::<Vec<_>>();
    let scores = (0..METRIC_ROWS)
        .map(|index| ((index.wrapping_mul(37) % 1_009) as f32) / 1_009.0)
        .collect::<Vec<_>>();

    let mut metrics = c.benchmark_group("ferricml_evaluation_v1_metrics_4096");
    metrics.throughput(Throughput::Elements(METRIC_ROWS as u64));
    metrics.bench_function(
        BenchmarkId::from_parameter("mean_squared_error"),
        |bencher| {
            bencher.iter(|| {
                black_box(mean_squared_error(black_box(&expected), black_box(&predicted)).unwrap());
            });
        },
    );
    metrics.bench_function(BenchmarkId::from_parameter("roc_auc"), |bencher| {
        bencher.iter(|| {
            black_box(roc_auc_score(black_box(&labels), black_box(&scores)).unwrap());
        });
    });
    metrics.bench_function(BenchmarkId::from_parameter("holdout_split"), |bencher| {
        bencher.iter(|| {
            black_box(
                train_test_split(
                    METRIC_ROWS,
                    HoldoutParams::default()
                        .with_test_size(TestSize::Fraction(0.2))
                        .with_random_state(19),
                )
                .unwrap(),
            );
        });
    });
    metrics.finish();

    let (data, targets) = fixture(CV_ROWS, CV_COLUMNS);
    let mut cross_validation = c.benchmark_group("ferricml_evaluation_v1_cv_256x12");
    cross_validation.throughput(Throughput::Elements(CV_ROWS as u64));
    cross_validation.bench_function(BenchmarkId::from_parameter("ridge_5_fold"), |bencher| {
        bencher.iter(|| {
            let splits = KFold::new(5)
                .with_shuffle(true)
                .with_random_state(23)
                .split(CV_ROWS)
                .unwrap();
            black_box(
                cross_validate_regressor(
                    black_box(&data.as_view()),
                    black_box(&targets),
                    splits,
                    RegressionScorer::MeanSquaredError,
                    |train, train_targets| Ridge::fit(train, train_targets, RidgeParams::default()),
                )
                .unwrap(),
            );
        });
    });
    cross_validation.finish();
}

fn split_workloads(c: &mut Criterion) {
    let four_class_labels = (0..HOLDOUT_ROWS)
        .map(|index| (index % 4) as u8)
        .collect::<Vec<_>>();
    let many_class_labels = (0..MANY_CLASS_ROWS)
        .map(|index| (index % 256) as u8)
        .collect::<Vec<_>>();

    let mut holdout = c.benchmark_group("ferricml_model_selection_v2_holdout_1000000");
    holdout.throughput(Throughput::Elements(HOLDOUT_ROWS as u64));
    for (name, params) in [
        (
            "ordinary_shuffled_20pct",
            HoldoutParams::default()
                .with_test_size(TestSize::Fraction(0.2))
                .with_random_state(19),
        ),
        (
            "ordinary_shuffled_80pct",
            HoldoutParams::default()
                .with_test_size(TestSize::Fraction(0.8))
                .with_random_state(19),
        ),
        (
            "ordinary_unshuffled_20pct",
            HoldoutParams::default()
                .with_test_size(TestSize::Fraction(0.2))
                .with_shuffle(false),
        ),
    ] {
        holdout.bench_function(BenchmarkId::from_parameter(name), |bencher| {
            bencher.iter(|| {
                black_box(train_test_split(black_box(HOLDOUT_ROWS), black_box(params)).unwrap());
            });
        });
    }
    holdout.bench_function(
        BenchmarkId::from_parameter("stratified_4_class_20pct"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    stratified_train_test_split(
                        black_box(&four_class_labels),
                        black_box(
                            HoldoutParams::default()
                                .with_test_size(TestSize::Fraction(0.2))
                                .with_random_state(19),
                        ),
                    )
                    .unwrap(),
                );
            });
        },
    );
    holdout.finish();

    let mut many_class = c.benchmark_group("ferricml_model_selection_v2_stratified_262144");
    many_class.throughput(Throughput::Elements(MANY_CLASS_ROWS as u64));
    many_class.bench_function(BenchmarkId::from_parameter("256_class_50pct"), |bencher| {
        bencher.iter(|| {
            black_box(
                stratified_train_test_split(
                    black_box(&many_class_labels),
                    black_box(
                        HoldoutParams::default()
                            .with_test_size(TestSize::Fraction(0.5))
                            .with_random_state(19),
                    ),
                )
                .unwrap(),
            );
        });
    });
    many_class.finish();
}

/// Permutation importance is a scoring loop, not a fitting loop: one baseline
/// score plus `columns * repeats` rescorings over a permuted column, all from
/// a workspace allocated once.
fn inspection(c: &mut Criterion) {
    let (data, targets) = fixture(INSPECTION_ROWS, INSPECTION_COLUMNS);
    let forest = RandomForestRegressor::fit(
        &data.as_view(),
        &targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(16)
            .with_max_depth(Some(8))
            .with_max_features(MaxFeatures::All)
            .with_random_state(31),
    )
    .unwrap();
    let ridge = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
    let params = PermutationImportanceParams::default()
        .with_n_repeats(INSPECTION_REPEATS)
        .with_random_state(31);
    let mut means = vec![0.0; INSPECTION_COLUMNS];
    let mut std_devs = vec![0.0; INSPECTION_COLUMNS];

    let mut group = c.benchmark_group("ferricml_inspection_v1_permutation_256x8_3r");
    group.throughput(Throughput::Elements(
        (INSPECTION_ROWS * INSPECTION_COLUMNS * INSPECTION_REPEATS) as u64,
    ));
    group.bench_function(BenchmarkId::from_parameter("forest_mse"), |bencher| {
        bencher.iter(|| {
            permutation_importance_regressor_into(
                black_box(&forest),
                black_box(&data.as_view()),
                &targets,
                RegressionScorer::MeanSquaredError,
                params,
                black_box(&mut means),
                black_box(&mut std_devs),
            )
            .unwrap();
        });
    });
    group.bench_function(BenchmarkId::from_parameter("ridge_r2"), |bencher| {
        bencher.iter(|| {
            permutation_importance_regressor_into(
                black_box(&ridge),
                black_box(&data.as_view()),
                &targets,
                RegressionScorer::R2,
                params,
                black_box(&mut means),
                black_box(&mut std_devs),
            )
            .unwrap();
        });
    });
    group.finish();
}

/// Baseline inference is the floor every other inference lane is read
/// against: it walks the same batch entry points while doing no per-row work,
/// so a real model's lane can be separated from the contract around it.
fn baselines(c: &mut Criterion) {
    let (data, targets) = fixture(INFERENCE_ROWS, COLUMNS);
    let labels = BinaryTargets::new(
        targets
            .as_slice()
            .iter()
            .map(|&value| u8::from(value > 0.0))
            .collect(),
    )
    .unwrap();
    let classifier = DummyClassifier::fit(&data.as_view(), &labels, DummyClassifierParams).unwrap();
    let regressor = DummyRegressor::fit(&data.as_view(), &targets, DummyRegressorParams).unwrap();

    let mut labels_out = vec![0_u8; INFERENCE_ROWS];
    let mut values_out = vec![0.0_f32; INFERENCE_ROWS];
    let mut group = c.benchmark_group("ferricml_baselines_v1_into_1024x48");
    group.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("dummy_classifier"), |bencher| {
        bencher.iter(|| {
            classifier
                .predict_into(black_box(&data.as_view()), black_box(&mut labels_out))
                .unwrap();
            black_box(&labels_out);
        });
    });
    group.bench_function(BenchmarkId::from_parameter("dummy_regressor"), |bencher| {
        bencher.iter(|| {
            regressor
                .predict_into(black_box(&data.as_view()), black_box(&mut values_out))
                .unwrap();
            black_box(&values_out);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    baselines,
    inference,
    training,
    logistic_and_scaler,
    evaluation,
    split_workloads,
    inspection
);
criterion_main!(benches);
