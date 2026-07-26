use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::api::ProbabilisticClassifier;
use ferricml::artifact::StageArtifact;
use ferricml::calibration::{
    CalibratedClassifier, IsotonicRegression, IsotonicRegressionParams, PlattCalibrator,
    PlattParams,
};
use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets};
use ferricml::dummy::{
    DummyClassifier, DummyClassifierParams, DummyRegressor, DummyRegressorParams,
};
use ferricml::ensemble::{
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::inspection::{PermutationImportanceParams, permutation_importance_regressor_into};
use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LinearRegression, LinearRegressionParams,
    LogisticRegression, LogisticRegressionParams, LogisticSolver, Ridge, RidgeParams,
};
use ferricml::metrics::{
    Average, ConfusionMatrix, average_precision_score, mean_squared_error, roc_auc_score,
};
use ferricml::model_selection::{
    ClassificationScorer, GroupKFold, GroupShuffleSplit, HoldoutParams, KFold, LeaveOneOut,
    ParameterGrid, RegressionScorer, RepeatedKFold, ScorableClassifier, ScoringWorkspace,
    TestGroupSize, TestSize, TimeSeriesSplit, cross_validate_regressor, grid_search_classifier,
    grid_search_regressor, score_regressor, score_regressor_with, stratified_train_test_split,
    train_test_split,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    Binarizer, BinarizerParams, FunctionTransformer, FunctionTransformerParams, MaxAbsScaler,
    MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams, Normalizer, NormalizerParams,
    RobustScaler, RobustScalerParams, StandardScaler, StandardScalerParams,
};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
};
use ferricml::tree::MaxFeatures;
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
const SPLITTER_ROWS: usize = 16_384;
const LEAVE_ONE_OUT_ROWS: usize = 512;
const MULTICLASS_CLASSES: usize = 4;
/// Wide enough that `classes * (features + intercept)` is 2052, one step past
/// the exact solver's 2048-parameter refusal.
const WIDE_MULTICLASS_COLUMNS: usize = 512;
/// Rows for the wide multiclass lane; the cost there is the parameter count.
const WIDE_MULTICLASS_ROWS: usize = 1_024;
/// Strong enough to remove coefficients and weak enough to keep a real fit.
const PENALTY_ALPHA: f32 = 0.05;
const MULTICLASS_FOREST_TREES: usize = 16;

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

/// Non-contiguous, non-zero-based labels derived from the shared fixture.
///
/// The labels are `{3, 7, 10, 20}` rather than `0..4` so the benchmark exercises
/// the same class-lookup path a real caller does.
fn multiclass_targets(targets: &RegressionTargets) -> ClassTargets {
    const LABELS: [u8; MULTICLASS_CLASSES] = [3, 7, 10, 20];
    let mut sorted = targets.as_slice().to_vec();
    sorted.sort_by(f32::total_cmp);
    let cuts: Vec<f32> = (1..MULTICLASS_CLASSES)
        .map(|part| sorted[part * sorted.len() / MULTICLASS_CLASSES])
        .collect();
    ClassTargets::new(
        targets
            .as_slice()
            .iter()
            .map(|&value| LABELS[cuts.iter().filter(|&&cut| value >= cut).count()])
            .collect(),
    )
    .unwrap()
}

/// Multiclass fitting and inference for both estimator families.
///
/// Registered with the sprint that added the capability so the paths are
/// visible to `bench-history` from birth. Each family gets both halves that can
/// drift independently: the fit, whose cost is the multiclass solver or the
/// multiclass split search, and inference, whose cost is the softmax or the
/// per-tree probability averaging.
fn multiclass(c: &mut Criterion) {
    let (training, training_targets) = fixture(ROWS, COLUMNS);
    let (inference, _) = fixture(INFERENCE_ROWS, COLUMNS);
    let labels = multiclass_targets(&training_targets);
    let forest_params = RandomForestClassifierParams::default()
        .with_n_estimators(MULTICLASS_FOREST_TREES)
        .with_max_depth(Some(8))
        .with_random_state(0);

    let logistic = LogisticRegression::fit_multiclass(
        &training.as_view(),
        &labels,
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let forest =
        RandomForestClassifier::fit_multiclass(&training.as_view(), &labels, forest_params.clone())
            .unwrap();

    let mut probabilities = vec![0.0_f32; INFERENCE_ROWS * MULTICLASS_CLASSES];
    let mut predicted = vec![0_u8; INFERENCE_ROWS];
    let mut into = c.benchmark_group("ferricml_multiclass_v1_into_1024x48_4c");
    into.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    into.bench_function(BenchmarkId::from_parameter("logistic_proba"), |bencher| {
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
    into.bench_function(BenchmarkId::from_parameter("logistic_label"), |bencher| {
        bencher.iter(|| {
            logistic
                .predict_into(black_box(&inference.as_view()), black_box(&mut predicted))
                .unwrap();
            black_box(&predicted);
        });
    });
    into.bench_function(BenchmarkId::from_parameter("forest_proba"), |bencher| {
        bencher.iter(|| {
            forest
                .predict_proba_into(
                    black_box(&inference.as_view()),
                    black_box(&mut probabilities),
                )
                .unwrap();
            black_box(&probabilities);
        });
    });
    into.bench_function(BenchmarkId::from_parameter("forest_label"), |bencher| {
        bencher.iter(|| {
            forest
                .predict_into(black_box(&inference.as_view()), black_box(&mut predicted))
                .unwrap();
            black_box(&predicted);
        });
    });
    into.finish();

    let mut fit = c.benchmark_group("ferricml_multiclass_v1_fit_2048x48_4c");
    fit.throughput(Throughput::Elements(ROWS as u64));
    fit.bench_function(BenchmarkId::from_parameter("logistic"), |bencher| {
        bencher.iter(|| {
            black_box(
                LogisticRegression::fit_multiclass(
                    black_box(&training.as_view()),
                    black_box(&labels),
                    LogisticRegressionParams::default(),
                )
                .unwrap(),
            );
        });
    });
    fit.bench_function(BenchmarkId::from_parameter("forest"), |bencher| {
        bencher.iter_batched(
            || forest_params.clone(),
            |params| {
                black_box(
                    RandomForestClassifier::fit_multiclass(
                        black_box(&training.as_view()),
                        black_box(&labels),
                        params,
                    )
                    .unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    fit.finish();
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

/// The penalized linear fits and the matrix-free logistic solver.
///
/// A separate suite version from `..._v1_fit_...` because these lanes are
/// iterative: their cost is a sweep or iteration count, not one factorization,
/// so a change in convergence behaviour shows up here and nowhere else. The
/// penalty strengths are chosen so each lane converges well inside its budget
/// — a lane that refused would measure the refusal path rather than the solver.
fn penalized_and_matrix_free(c: &mut Criterion) {
    let (data, targets) = fixture(ROWS, COLUMNS);
    let labels = BinaryTargets::new(
        targets
            .as_slice()
            .iter()
            .map(|&target| u8::from(target > 0.0))
            .collect(),
    )
    .unwrap();

    let mut fit = c.benchmark_group("ferricml_models_v3_penalized_fit_2048x48");
    fit.throughput(Throughput::Elements(ROWS as u64));
    fit.bench_function(BenchmarkId::from_parameter("lasso"), |bencher| {
        bencher.iter_batched(
            || LassoParams::default().with_alpha(PENALTY_ALPHA),
            |params| {
                black_box(Lasso::fit(&data.as_view(), &targets, params).unwrap());
            },
            BatchSize::SmallInput,
        );
    });
    fit.bench_function(BenchmarkId::from_parameter("elastic_net"), |bencher| {
        bencher.iter_batched(
            || {
                ElasticNetParams::default()
                    .with_alpha(PENALTY_ALPHA)
                    .with_l1_ratio(0.5)
            },
            |params| {
                black_box(ElasticNet::fit(&data.as_view(), &targets, params).unwrap());
            },
            BatchSize::SmallInput,
        );
    });
    fit.bench_function(BenchmarkId::from_parameter("logistic_lbfgs"), |bencher| {
        bencher.iter_batched(
            || LogisticRegressionParams::default().with_solver(LogisticSolver::Lbfgs),
            |params| {
                black_box(LogisticRegression::fit(&data.as_view(), &labels, params).unwrap());
            },
            BatchSize::SmallInput,
        );
    });
    fit.finish();

    // The lane the matrix-free solver exists for: a stacked system the exact
    // path refuses outright, so there is nothing to compare it against.
    let (wide_data, wide_targets) = fixture(WIDE_MULTICLASS_ROWS, WIDE_MULTICLASS_COLUMNS);
    let wide_classes = multiclass_targets(&wide_targets);
    let mut multiclass_fit =
        c.benchmark_group("ferricml_models_v3_matrix_free_multiclass_fit_1024x512_4c");
    multiclass_fit.throughput(Throughput::Elements(WIDE_MULTICLASS_ROWS as u64));
    multiclass_fit.bench_function(BenchmarkId::from_parameter("logistic_lbfgs"), |bencher| {
        bencher.iter_batched(
            || {
                LogisticRegressionParams::default()
                    .with_solver(LogisticSolver::Lbfgs)
                    .with_max_iter(200)
            },
            |params| {
                black_box(
                    LogisticRegression::fit_multiclass(&wide_data.as_view(), &wide_classes, params)
                        .unwrap(),
                );
            },
            BatchSize::SmallInput,
        );
    });
    multiclass_fit.finish();

    let inference_data = fixture(INFERENCE_ROWS, COLUMNS).0;
    let lasso = Lasso::fit(
        &data.as_view(),
        &targets,
        LassoParams::default().with_alpha(PENALTY_ALPHA),
    )
    .unwrap();
    let mut into = c.benchmark_group("ferricml_models_v3_penalized_into_1024x48");
    into.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    into.bench_function(BenchmarkId::from_parameter("lasso"), |bencher| {
        let mut output = vec![0.0_f32; INFERENCE_ROWS];
        bencher.iter(|| {
            lasso
                .predict_into(black_box(&inference_data.as_view()), &mut output)
                .unwrap();
            black_box(&output);
        });
    });
    into.finish();
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

/// The evaluation vocabulary added in the metrics-and-scorer sprint: averaged
/// label-set scores, threshold sweeps, and the allocating versus workspace
/// scoring entry points on one fitted model.
fn evaluation_vocabulary(c: &mut Criterion) {
    let labels = (0..METRIC_ROWS)
        .map(|index| (index % 3) as u8)
        .collect::<Vec<_>>();
    let predicted = (0..METRIC_ROWS)
        .map(|index| (index.wrapping_mul(7) % 3) as u8)
        .collect::<Vec<_>>();
    let binary = (0..METRIC_ROWS)
        .map(|index| u8::from(index % 3 == 0))
        .collect::<Vec<_>>();
    let scores = (0..METRIC_ROWS)
        .map(|index| ((index.wrapping_mul(37) % 1_009) as f32) / 1_009.0)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("ferricml_evaluation_v2_vocabulary_4096");
    group.throughput(Throughput::Elements(METRIC_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("confusion_matrix"), |bencher| {
        bencher.iter(|| {
            black_box(ConfusionMatrix::new(black_box(&labels), black_box(&predicted)).unwrap());
        });
    });
    let matrix = ConfusionMatrix::new(&labels, &predicted).unwrap();
    group.bench_function(BenchmarkId::from_parameter("macro_f1"), |bencher| {
        bencher.iter(|| {
            black_box(black_box(&matrix).f1(Average::Macro).unwrap());
        });
    });
    group.bench_function(
        BenchmarkId::from_parameter("average_precision"),
        |bencher| {
            bencher.iter(|| {
                black_box(average_precision_score(black_box(&binary), black_box(&scores)).unwrap());
            });
        },
    );
    group.finish();

    let (data, targets) = fixture(CV_ROWS, CV_COLUMNS);
    let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
    let mut workspace = ScoringWorkspace::new();
    let mut scoring = c.benchmark_group("ferricml_evaluation_v2_scoring_256x12");
    scoring.throughput(Throughput::Elements(CV_ROWS as u64));
    scoring.bench_function(BenchmarkId::from_parameter("allocating"), |bencher| {
        bencher.iter(|| {
            black_box(
                score_regressor(
                    black_box(&model),
                    black_box(&data.as_view()),
                    &targets,
                    RegressionScorer::MeanSquaredError,
                )
                .unwrap(),
            );
        });
    });
    scoring.bench_function(BenchmarkId::from_parameter("workspace"), |bencher| {
        bencher.iter(|| {
            black_box(
                score_regressor_with(
                    black_box(&model),
                    black_box(&data.as_view()),
                    &targets,
                    RegressionScorer::MeanSquaredError,
                    &mut workspace,
                )
                .unwrap(),
            );
        });
    });
    scoring.finish();
}

/// Splitters added alongside the evaluation vocabulary. Each lane materializes
/// every fold, so the cost of the whole iterator is what is measured.
fn evaluation_splitters(c: &mut Criterion) {
    let groups = (0..SPLITTER_ROWS)
        .map(|index| (index % 128) as u64)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("ferricml_model_selection_v3_splitters_16384");
    group.throughput(Throughput::Elements(SPLITTER_ROWS as u64));
    group.bench_function(BenchmarkId::from_parameter("group_kfold_5"), |bencher| {
        bencher.iter(|| {
            black_box(
                GroupKFold::new(5)
                    .split(black_box(&groups))
                    .unwrap()
                    .count(),
            );
        });
    });
    group.bench_function(BenchmarkId::from_parameter("time_series_5"), |bencher| {
        bencher.iter(|| {
            black_box(
                TimeSeriesSplit::new(5)
                    .split(black_box(SPLITTER_ROWS))
                    .unwrap()
                    .count(),
            );
        });
    });
    group.bench_function(
        BenchmarkId::from_parameter("repeated_kfold_5x3"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    RepeatedKFold::new(5, 3)
                        .with_random_state(19)
                        .split(black_box(SPLITTER_ROWS))
                        .unwrap()
                        .count(),
                );
            });
        },
    );
    group.bench_function(BenchmarkId::from_parameter("leave_one_out"), |bencher| {
        bencher.iter(|| {
            black_box(
                LeaveOneOut::new()
                    .split(black_box(LEAVE_ONE_OUT_ROWS))
                    .unwrap()
                    .count(),
            );
        });
    });
    group.bench_function(
        BenchmarkId::from_parameter("group_shuffle_5x25pct"),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    GroupShuffleSplit::new(5)
                        .with_test_size(TestGroupSize::Fraction(0.25))
                        .with_random_state(19)
                        .split(black_box(&groups))
                        .unwrap()
                        .count(),
                );
            });
        },
    );
    group.finish();
}

/// Typed parameter search. Each lane runs a whole grid, so what is measured is
/// the full candidate loop through cross-validation and the shared scorer, not
/// one fit.
fn parameter_search(c: &mut Criterion) {
    let (data, targets) = fixture(CV_ROWS, CV_COLUMNS);
    let labels =
        BinaryTargets::new((0..CV_ROWS).map(|row| u8::from(row % 3 == 0)).collect()).unwrap();

    let mut group = c.benchmark_group("ferricml_model_selection_v4_search_256x12");
    group.throughput(Throughput::Elements(CV_ROWS as u64));
    group.bench_function(
        BenchmarkId::from_parameter("ridge_grid_4x3fold"),
        |bencher| {
            let grid = ParameterGrid::new(RidgeParams::default())
                .axis([0.01_f32, 1.0], RidgeParams::with_alpha)
                .axis([true, false], RidgeParams::with_fit_intercept);
            bencher.iter(|| {
                let splits = KFold::new(3)
                    .with_shuffle(true)
                    .with_random_state(23)
                    .split(CV_ROWS)
                    .unwrap();
                black_box(
                    grid_search_regressor(
                        black_box(&data.as_view()),
                        black_box(&targets),
                        splits,
                        black_box(&grid),
                        RegressionScorer::MeanSquaredError,
                        |train, train_targets, params| {
                            Ridge::fit(train, train_targets, params.clone())
                        },
                    )
                    .unwrap(),
                );
            });
        },
    );
    group.bench_function(
        BenchmarkId::from_parameter("logistic_grid_2x3fold"),
        |bencher| {
            let grid = ParameterGrid::new(LogisticRegressionParams::default().with_max_iter(25))
                .axis([0.1_f32, 1.0], LogisticRegressionParams::with_c);
            bencher.iter(|| {
                let splits = KFold::new(3)
                    .with_shuffle(true)
                    .with_random_state(23)
                    .split(CV_ROWS)
                    .unwrap();
                black_box(
                    grid_search_classifier(
                        black_box(&data.as_view()),
                        black_box(&labels),
                        splits,
                        black_box(&grid),
                        ClassificationScorer::Accuracy,
                        |train, train_targets, params| {
                            LogisticRegression::fit(train, train_targets, params.clone())
                        },
                        |model| ScorableClassifier::probabilistic(model),
                    )
                    .unwrap(),
                );
            });
        },
    );
    group.finish();
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

/// Transformer family and multi-stage composition workloads.
///
/// Registered so the new user-facing capability is visible to `bench-history`
/// from the sprint that added it. The lanes cover both halves that can drift
/// independently: per-column scaling itself, and the workspace splitting a
/// multi-stage composition adds on top of the stages it runs.
/// The map the elementwise-transformer workload applies.
///
/// A named `fn` because `FunctionTransformer` takes a function pointer, which
/// is what keeps the type nameable and its capability declaration visible.
fn bench_scale(value: f32) -> f32 {
    value * 1.5
}

fn transformers_and_staged_pipelines(c: &mut Criterion) {
    let (training, targets) = fixture(ROWS, COLUMNS);
    let (inference, _) = fixture(INFERENCE_ROWS, COLUMNS);

    let min_max = MinMaxScaler::fit(&training.as_view(), MinMaxScalerParams::default()).unwrap();
    let max_abs = MaxAbsScaler::fit(&training.as_view(), MaxAbsScalerParams).unwrap();
    let robust = RobustScaler::fit(&training.as_view(), RobustScalerParams::default()).unwrap();
    let normalizer = Normalizer::fit(&training.as_view(), NormalizerParams::default()).unwrap();
    let binarizer = Binarizer::fit(&training.as_view(), BinarizerParams::default()).unwrap();
    let elementwise = FunctionTransformer::fit(
        &training.as_view(),
        FunctionTransformerParams::default().with_func(bench_scale),
    )
    .unwrap();

    let mut transformed = vec![0.0; INFERENCE_ROWS * COLUMNS];
    let mut into = c.benchmark_group("ferricml_transformers_v1_into_1024x48");
    into.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    into.bench_function(BenchmarkId::from_parameter("min_max"), |bencher| {
        bencher.iter(|| {
            black_box(
                min_max
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    into.bench_function(BenchmarkId::from_parameter("max_abs"), |bencher| {
        bencher.iter(|| {
            black_box(
                max_abs
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    into.bench_function(BenchmarkId::from_parameter("robust"), |bencher| {
        bencher.iter(|| {
            black_box(
                robust
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    into.bench_function(BenchmarkId::from_parameter("normalizer"), |bencher| {
        bencher.iter(|| {
            black_box(
                normalizer
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    into.bench_function(BenchmarkId::from_parameter("binarizer"), |bencher| {
        bencher.iter(|| {
            black_box(
                binarizer
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    into.bench_function(BenchmarkId::from_parameter("elementwise"), |bencher| {
        bencher.iter(|| {
            black_box(
                elementwise
                    .transform_into(black_box(&inference.as_view()), black_box(&mut transformed))
                    .unwrap(),
            );
        });
    });
    // The inverse direction is a separate lane: it is the only user-facing
    // transform path whose cost is not visible from the forward one.
    into.bench_function(BenchmarkId::from_parameter("robust_inverse"), |bencher| {
        bencher.iter(|| {
            black_box(
                robust
                    .inverse_transform_into(
                        black_box(&inference.as_view()),
                        black_box(&mut transformed),
                    )
                    .unwrap(),
            );
        });
    });
    into.finish();

    let mut fit = c.benchmark_group("ferricml_transformers_v1_fit_2048x48");
    fit.throughput(Throughput::Elements(ROWS as u64));
    fit.bench_function(BenchmarkId::from_parameter("robust"), |bencher| {
        bencher.iter(|| {
            black_box(
                RobustScaler::fit(
                    black_box(&training.as_view()),
                    RobustScalerParams::default(),
                )
                .unwrap(),
            );
        });
    });
    fit.bench_function(BenchmarkId::from_parameter("min_max"), |bencher| {
        bencher.iter(|| {
            black_box(
                MinMaxScaler::fit(
                    black_box(&training.as_view()),
                    MinMaxScalerParams::default(),
                )
                .unwrap(),
            );
        });
    });
    fit.bench_function(BenchmarkId::from_parameter("max_abs"), |bencher| {
        bencher.iter(|| {
            black_box(
                MaxAbsScaler::fit(black_box(&training.as_view()), MaxAbsScalerParams).unwrap(),
            );
        });
    });
    fit.finish();

    let staged: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> = StagedPipeline::fit(
        &training.as_view(),
        |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
        |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
        |batch| Ridge::fit(batch, &targets, RidgeParams::default()),
    )
    .unwrap();
    let three: StagedPipeline<(MinMaxScaler, StandardScaler, MaxAbsScaler), Ridge> = {
        let first = MinMaxScaler::fit(&training.as_view(), MinMaxScalerParams::default()).unwrap();
        let after_first = first.transform(&training.as_view()).unwrap();
        let second =
            StandardScaler::fit(&after_first.as_view(), StandardScalerParams::default()).unwrap();
        let after_second = second.transform(&after_first.as_view()).unwrap();
        let third = MaxAbsScaler::fit(&after_second.as_view(), MaxAbsScalerParams).unwrap();
        let final_batch = third.transform(&after_second.as_view()).unwrap();
        let estimator =
            Ridge::fit(&final_batch.as_view(), &targets, RidgeParams::default()).unwrap();
        StagedPipeline::new((first, second, third), estimator).unwrap()
    };

    let mut two_workspace = vec![0.0; staged.workspace_len(INFERENCE_ROWS).unwrap()];
    let mut three_workspace = vec![0.0; three.workspace_len(INFERENCE_ROWS).unwrap()];
    let mut predictions = vec![0.0; INFERENCE_ROWS];
    let mut staged_into = c.benchmark_group("ferricml_staged_pipeline_v1_into_1024x48");
    staged_into.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    staged_into.bench_function(BenchmarkId::from_parameter("two_stage_ridge"), |bencher| {
        bencher.iter(|| {
            staged
                .with_transformed(
                    black_box(&inference.as_view()),
                    black_box(&mut two_workspace),
                    |model, batch| model.predict_into(batch, &mut predictions),
                )
                .unwrap();
            black_box(&predictions);
        });
    });
    staged_into.bench_function(
        BenchmarkId::from_parameter("three_stage_ridge"),
        |bencher| {
            bencher.iter(|| {
                three
                    .with_transformed(
                        black_box(&inference.as_view()),
                        black_box(&mut three_workspace),
                        |model, batch| model.predict_into(batch, &mut predictions),
                    )
                    .unwrap();
                black_box(&predictions);
            });
        },
    );
    staged_into.finish();

    let mut artifact = c.benchmark_group("ferricml_staged_pipeline_v1_artifact_2048x48");
    artifact.throughput(Throughput::Elements(1));
    artifact.bench_function(BenchmarkId::from_parameter("round_trip"), |bencher| {
        bencher.iter(|| {
            let bytes = staged.to_artifact([7; 32], [8; 32]).unwrap();
            black_box(
                StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
                    black_box(&bytes),
                    [7; 32],
                    [8; 32],
                )
                .unwrap(),
            );
        });
    });
    artifact.finish();
}

/// Calibration fitting and calibrated inference.
///
/// Registered with the sprint that added the capability, so both halves are
/// visible to `bench-history` from birth. The two halves drift independently:
/// a calibrator fit is one sort plus one linear pass over the calibration
/// scores or a handful of Newton passes, while calibrated inference is the
/// wrapped model's own prediction plus one map per row, which is the cost the
/// wrapper has to justify.
fn calibration(c: &mut Criterion) {
    let (training, regression_targets) = fixture(ROWS, COLUMNS);
    let labels = BinaryTargets::new(
        regression_targets
            .as_slice()
            .iter()
            .map(|&target| u8::from(target > 0.0))
            .collect(),
    )
    .unwrap();
    let forest = RandomForestClassifier::fit(
        &training.as_view(),
        &labels,
        RandomForestClassifierParams::default()
            .with_n_estimators(16)
            .with_max_depth(Some(8))
            .with_random_state(0),
    )
    .unwrap();
    let inference = DenseMatrix::new(
        training.as_slice()[..INFERENCE_ROWS * COLUMNS].to_vec(),
        INFERENCE_ROWS,
        COLUMNS,
    )
    .unwrap();

    let scores = forest.predict_class_proba(&training.as_view(), 1).unwrap();
    let mut fit = c.benchmark_group("ferricml_calibration_v1_fit_2048x48");
    fit.throughput(Throughput::Elements(ROWS as u64));
    fit.bench_function(BenchmarkId::from_parameter("isotonic"), |bencher| {
        bencher.iter(|| {
            black_box(
                IsotonicRegression::fit_calibration(
                    black_box(&scores),
                    black_box(&labels),
                    IsotonicRegressionParams,
                )
                .unwrap(),
            );
        });
    });
    fit.bench_function(BenchmarkId::from_parameter("platt"), |bencher| {
        bencher.iter(|| {
            black_box(
                PlattCalibrator::fit(
                    black_box(&scores),
                    black_box(&labels),
                    PlattParams::default(),
                )
                .unwrap(),
            );
        });
    });
    fit.finish();

    let isotonic = CalibratedClassifier::fit_isotonic(
        forest.clone(),
        &training.as_view(),
        &labels,
        IsotonicRegressionParams,
    )
    .unwrap();
    let platt = CalibratedClassifier::fit_platt(
        forest.clone(),
        &training.as_view(),
        &labels,
        PlattParams::default(),
    )
    .unwrap();

    let mut positive = vec![0.0_f32; INFERENCE_ROWS];
    let mut matrix = vec![0.0_f32; INFERENCE_ROWS * 2];
    let mut predicted = vec![0_u8; INFERENCE_ROWS];
    let mut into = c.benchmark_group("ferricml_calibration_v1_into_1024x48");
    into.throughput(Throughput::Elements(INFERENCE_ROWS as u64));
    into.bench_function(BenchmarkId::from_parameter("uncalibrated"), |bencher| {
        bencher.iter(|| {
            forest
                .predict_class_proba_into(
                    black_box(&inference.as_view()),
                    1,
                    black_box(&mut positive),
                )
                .unwrap();
        });
    });
    into.bench_function(BenchmarkId::from_parameter("isotonic_proba"), |bencher| {
        bencher.iter(|| {
            isotonic
                .predict_proba_into(black_box(&inference.as_view()), black_box(&mut matrix))
                .unwrap();
        });
    });
    into.bench_function(BenchmarkId::from_parameter("platt_proba"), |bencher| {
        bencher.iter(|| {
            platt
                .predict_proba_into(black_box(&inference.as_view()), black_box(&mut matrix))
                .unwrap();
        });
    });
    into.bench_function(BenchmarkId::from_parameter("platt_decision"), |bencher| {
        bencher.iter(|| {
            platt
                .decision_function_into(black_box(&inference.as_view()), black_box(&mut positive))
                .unwrap();
        });
    });
    into.bench_function(BenchmarkId::from_parameter("isotonic_predict"), |bencher| {
        bencher.iter(|| {
            isotonic
                .predict_into_with(
                    black_box(&inference.as_view()),
                    black_box(&mut positive),
                    black_box(&mut predicted),
                )
                .unwrap();
        });
    });
    into.finish();
}

criterion_group!(
    benches,
    baselines,
    inference,
    training,
    logistic_and_scaler,
    transformers_and_staged_pipelines,
    evaluation,
    evaluation_vocabulary,
    evaluation_splitters,
    parameter_search,
    split_workloads,
    inspection,
    multiclass,
    calibration,
    penalized_and_matrix_free
);
criterion_main!(benches);
