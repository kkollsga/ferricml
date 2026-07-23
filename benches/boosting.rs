use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::ensemble::{HistGradientBoostingRegressor, HistGradientBoostingRegressorParams};
use std::hint::black_box;

const TRAIN_ROWS: usize = 2_048;
const COLUMNS: usize = 48;

fn fixture(rows: usize, columns: usize) -> (DenseMatrix, RegressionTargets) {
    let mut state = 0x243f_6a88_u32;
    let mut values = Vec::with_capacity(rows * columns);
    let mut targets = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut selected = [0.0_f32; 12];
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
        let target = 2.0 * selected[0] - selected[1]
            + 1.5 * selected[2] * selected[3]
            + if selected[4] > 0.0 { 1.2 } else { -1.2 }
            + if selected[5] + selected[6] > 0.25 {
                0.9
            } else {
                -0.4
            }
            + 0.3 * selected[7] * selected[8]
            + 0.15 * ((row % 17) as f32 - 8.0);
        targets.push(target);
    }
    (
        DenseMatrix::new(values, rows, columns).unwrap(),
        RegressionTargets::new(targets).unwrap(),
    )
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

criterion_group!(benches, inference, training);
criterion_main!(benches);
