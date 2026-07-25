use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures,
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    MinMaxScaler, MinMaxScalerParams, RobustScaler, RobustScalerParams, StandardScaler,
    StandardScalerParams,
};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
};
use sha2::{Digest, Sha256};

fn assert_fingerprint(
    name: &str,
    left: Vec<u8>,
    right: Vec<u8>,
    expected_len: usize,
    expected_digest: [u8; 32],
) {
    assert_eq!(left, right, "{name} encoding changed between calls");
    assert_eq!(left.len(), expected_len, "{name} artifact length changed");
    let digest: [u8; 32] = Sha256::digest(&left).into();
    assert_eq!(digest, expected_digest, "{name} artifact bytes changed");
}

#[test]
fn fitted_artifact_fingerprints_are_frozen() {
    let data = DenseMatrix::new(vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 8.0], 4, 2).unwrap();
    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let input_schema = [3; 32];
    let transformed_schema = [4; 32];

    let linear = LinearRegression::fit(
        &data.as_view(),
        &regression,
        LinearRegressionParams::default(),
    )
    .unwrap();
    assert_fingerprint(
        "linear",
        linear.to_artifact(input_schema).unwrap(),
        linear.to_artifact(input_schema).unwrap(),
        132,
        [
            209, 115, 88, 5, 134, 10, 192, 82, 18, 224, 210, 175, 7, 235, 181, 2, 161, 161, 114,
            112, 73, 95, 181, 197, 7, 190, 134, 15, 123, 42, 242, 173,
        ],
    );

    let ridge = Ridge::fit(&data.as_view(), &regression, RidgeParams::default()).unwrap();
    assert_fingerprint(
        "ridge",
        ridge.to_artifact(input_schema).unwrap(),
        ridge.to_artifact(input_schema).unwrap(),
        128,
        [
            76, 173, 93, 164, 34, 120, 233, 236, 22, 186, 73, 121, 180, 213, 76, 155, 140, 151, 25,
            71, 150, 166, 143, 4, 244, 252, 252, 66, 17, 70, 182, 240,
        ],
    );

    let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default()).unwrap();
    assert_fingerprint(
        "scaler",
        scaler
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        scaler
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        200,
        [
            221, 144, 82, 224, 26, 34, 134, 67, 6, 101, 22, 48, 40, 68, 24, 153, 91, 202, 184, 178,
            63, 13, 17, 18, 180, 252, 187, 143, 124, 32, 238, 199,
        ],
    );
    let robust = RobustScaler::fit(&data.as_view(), RobustScalerParams::default()).unwrap();
    assert_fingerprint(
        "robust-scaler",
        robust
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        robust
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        200,
        [
            131, 38, 226, 223, 251, 78, 202, 173, 153, 102, 41, 221, 136, 202, 24, 140, 25, 136,
            254, 88, 39, 143, 127, 10, 88, 211, 64, 191, 0, 68, 134, 7,
        ],
    );

    let transformed = scaler.transform(&data.as_view()).unwrap();

    let pipeline_linear = Pipeline::new(
        scaler.clone(),
        LinearRegression::fit(
            &transformed.as_view(),
            &regression,
            LinearRegressionParams::default(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_fingerprint(
        "pipeline-linear",
        pipeline_linear
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        pipeline_linear
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        476,
        [
            156, 254, 13, 194, 125, 209, 110, 21, 89, 236, 78, 152, 154, 91, 33, 61, 184, 108, 174,
            47, 120, 151, 24, 220, 24, 231, 0, 157, 183, 57, 109, 134,
        ],
    );

    let pipeline_ridge = Pipeline::new(
        scaler.clone(),
        Ridge::fit(&transformed.as_view(), &regression, RidgeParams::default()).unwrap(),
    )
    .unwrap();
    assert_fingerprint(
        "pipeline-ridge",
        pipeline_ridge
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        pipeline_ridge
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        472,
        [
            115, 241, 109, 152, 128, 225, 187, 85, 197, 206, 115, 168, 5, 44, 107, 86, 220, 241,
            178, 196, 77, 88, 11, 77, 92, 217, 155, 101, 18, 182, 121, 148,
        ],
    );

    let pipeline_logistic = Pipeline::new(
        scaler,
        LogisticRegression::fit(
            &transformed.as_view(),
            &binary,
            LogisticRegressionParams::default(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_fingerprint(
        "pipeline-logistic",
        pipeline_logistic
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        pipeline_logistic
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        484,
        [
            253, 122, 177, 33, 161, 6, 254, 251, 239, 170, 1, 45, 20, 140, 109, 220, 172, 76, 209,
            169, 1, 220, 245, 201, 168, 27, 121, 239, 253, 76, 181, 230,
        ],
    );

    let items = DenseMatrix::new(vec![0.0, 0.0, 1.0, 0.25, 2.0, 1.0, 3.0, 2.0], 4, 2).unwrap();
    let pair = |left, right, outcome, weight| {
        PairwiseObservation::new(PairIndex::new(left, right).unwrap(), outcome, weight).unwrap()
    };
    let observations = vec![
        pair(3, 2, PairOutcome::LeftPreferred, 2.0),
        pair(2, 1, PairOutcome::LeftPreferred, 1.0),
        pair(1, 0, PairOutcome::LeftPreferred, 1.0),
        pair(1, 2, PairOutcome::Tie, 0.5),
    ];
    let ranker = PairwiseLinearRanker::fit(
        &items.as_view(),
        &observations,
        PairwiseLinearRankerParams::default(),
    )
    .unwrap();
    assert_fingerprint(
        "pairwise",
        ranker.to_artifact([9; 32]).unwrap(),
        ranker.to_artifact([9; 32]).unwrap(),
        276,
        [
            144, 27, 27, 246, 176, 226, 36, 24, 177, 122, 232, 148, 177, 120, 30, 154, 0, 201, 132,
            227, 21, 189, 87, 130, 152, 50, 250, 167, 66, 210, 147, 100,
        ],
    );

    let boosting_data = DenseMatrix::new((0..8).map(|value| value as f32).collect(), 8, 1).unwrap();
    let boosting_targets =
        RegressionTargets::new(vec![0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0]).unwrap();
    let boosting = HistGradientBoostingRegressor::fit(
        &boosting_data.as_view(),
        &boosting_targets,
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1)
            .with_max_bins(8),
    )
    .unwrap();
    assert_fingerprint(
        "boosting",
        boosting.to_artifact([23; 32]).unwrap(),
        boosting.to_artifact([23; 32]).unwrap(),
        228,
        [
            113, 238, 99, 34, 64, 238, 241, 189, 162, 243, 18, 101, 151, 94, 20, 11, 136, 11, 11,
            45, 74, 88, 55, 16, 252, 131, 234, 252, 147, 96, 194, 153,
        ],
    );

    // Tier 2: the fitting path evaluates `ln` for the baseline and `exp` once
    // per row per iteration for the gradient, so these bytes are promised on the
    // two tested targets rather than on every IEEE-754 one. Freezing them is
    // still what turns a green `main` into cross-platform evidence.
    let boosted_classifier = HistGradientBoostingClassifier::fit(
        &data.as_view(),
        &binary,
        HistGradientBoostingClassifierParams::default()
            .with_learning_rate(0.5)
            .with_max_iter(3)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1)
            .with_max_bins(4),
    )
    .unwrap();
    assert_fingerprint(
        "boosting-classifier",
        boosted_classifier.to_artifact([23; 32]).unwrap(),
        boosted_classifier.to_artifact([23; 32]).unwrap(),
        388,
        [
            111, 189, 143, 235, 218, 45, 37, 198, 223, 13, 116, 171, 127, 173, 155, 206, 98, 41,
            135, 47, 117, 89, 55, 72, 135, 29, 143, 215, 65, 44, 53, 255,
        ],
    );

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
    assert_fingerprint(
        "forest-regressor",
        forest.to_artifact([5; 32]).unwrap(),
        forest.to_artifact([5; 32]).unwrap(),
        480,
        [
            72, 150, 95, 114, 217, 38, 109, 10, 241, 113, 80, 81, 84, 36, 151, 137, 108, 208, 36,
            188, 169, 255, 246, 212, 233, 192, 21, 124, 241, 7, 33, 217,
        ],
    );

    // Both classifier leaf representations, under one artifact kind. Forests
    // are tier-1 deterministic (arithmetic only), so their bytes are frozen
    // here rather than only round-tripped.
    let classifier_params = RandomForestClassifierParams::default()
        .with_n_estimators(3)
        .with_max_depth(Some(4))
        .with_max_features(MaxFeatures::All)
        .with_random_state(11);
    let forest_classifier =
        RandomForestClassifier::fit(&data.as_view(), &binary, classifier_params.clone()).unwrap();
    assert_fingerprint(
        "forest-classifier",
        forest_classifier.to_artifact([5; 32]).unwrap(),
        forest_classifier.to_artifact([5; 32]).unwrap(),
        416,
        [
            71, 168, 77, 107, 151, 107, 74, 212, 140, 21, 194, 196, 23, 122, 251, 133, 18, 69, 168,
            127, 152, 244, 55, 144, 94, 67, 209, 38, 202, 118, 97, 237,
        ],
    );
    let multiclass_forest = RandomForestClassifier::fit_multiclass(
        &data.as_view(),
        &ClassTargets::new(vec![3, 7, 10, 7]).unwrap(),
        classifier_params,
    )
    .unwrap();
    assert_fingerprint(
        "forest-classifier-multiclass",
        multiclass_forest.to_artifact([5; 32]).unwrap(),
        multiclass_forest.to_artifact([5; 32]).unwrap(),
        592,
        [
            89, 182, 208, 111, 239, 82, 138, 10, 170, 201, 231, 232, 108, 177, 151, 116, 130, 156,
            121, 230, 156, 233, 230, 25, 201, 87, 142, 118, 141, 43, 145, 2,
        ],
    );

    let staged: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> = StagedPipeline::fit(
        &data.as_view(),
        |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
        |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
        |batch| Ridge::fit(batch, &regression, RidgeParams::default()),
    )
    .unwrap();
    assert_fingerprint(
        "staged-pipeline",
        staged
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        staged
            .to_artifact(input_schema, transformed_schema)
            .unwrap(),
        684,
        [
            239, 250, 222, 100, 214, 146, 29, 132, 72, 138, 179, 149, 177, 235, 23, 20, 5, 202, 16,
            102, 16, 59, 93, 207, 114, 219, 35, 191, 204, 194, 105, 7,
        ],
    );
}
