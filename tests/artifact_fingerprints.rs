use ferricml::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, ExtraTreesRegressor,
    ExtraTreesRegressorParams, HistGradientBoostingClassifier,
    HistGradientBoostingClassifierParams, HistGradientBoostingRegressor,
    HistGradientBoostingRegressorParams, MaxFeatures, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor, RandomForestRegressorParams,
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
use ferricml::tree::{
    DecisionTreeClassifier, DecisionTreeClassifierParams, DecisionTreeRegressor,
    DecisionTreeRegressorParams, Splitter,
};

/// Asserts that encoding a fitted model twice yields identical bytes.
///
/// This deliberately does **not** freeze a length or a digest. FerricML is
/// pre-1.0 with no users, so a byte-stability promise would constrain the
/// format for nobody while biasing the design toward whatever is cheap not to
/// change. Determinism is a real guarantee (see `CLAUDE.md`) and is what this
/// keeps; canonicity and round-tripping are asserted in `artifact_hardening`.
/// Re-freeze here when the API and feature set settle.
fn assert_fingerprint(name: &str, left: Vec<u8>, right: Vec<u8>) {
    assert_eq!(left, right, "{name} encoding changed between calls");
    assert!(!left.is_empty(), "{name} produced an empty artifact");
}

#[test]
fn fitted_artifact_encoding_is_deterministic() {
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
    );

    let ridge = Ridge::fit(&data.as_view(), &regression, RidgeParams::default()).unwrap();
    assert_fingerprint(
        "ridge",
        ridge.to_artifact(input_schema).unwrap(),
        ridge.to_artifact(input_schema).unwrap(),
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
    );

    // The randomized ensembles. Their thresholds come out of the crate's own
    // generator rather than out of the data, so freezing them is what turns
    // "the randomized search is deterministic" into a checked claim.
    let extra_trees_regressor = ExtraTreesRegressor::fit(
        &data.as_view(),
        &regression,
        ExtraTreesRegressorParams::default()
            .with_n_estimators(3)
            .with_max_depth(Some(4))
            .with_max_features(MaxFeatures::All)
            .with_random_state(11),
    )
    .unwrap();
    assert_fingerprint(
        "extra-trees-regressor",
        extra_trees_regressor.to_artifact([5; 32]).unwrap(),
        extra_trees_regressor.to_artifact([5; 32]).unwrap(),
    );
    let extra_trees_classifier_params = ExtraTreesClassifierParams::default()
        .with_n_estimators(3)
        .with_max_depth(Some(4))
        .with_max_features(MaxFeatures::All)
        .with_random_state(11);
    let extra_trees_classifier = ExtraTreesClassifier::fit(
        &data.as_view(),
        &binary,
        extra_trees_classifier_params.clone(),
    )
    .unwrap();
    assert_fingerprint(
        "extra-trees-classifier",
        extra_trees_classifier.to_artifact([5; 32]).unwrap(),
        extra_trees_classifier.to_artifact([5; 32]).unwrap(),
    );
    let multiclass_extra_trees = ExtraTreesClassifier::fit_multiclass(
        &data.as_view(),
        &ClassTargets::new(vec![3, 7, 10, 7]).unwrap(),
        extra_trees_classifier_params,
    )
    .unwrap();
    assert_fingerprint(
        "extra-trees-classifier-multiclass",
        multiclass_extra_trees.to_artifact([5; 32]).unwrap(),
        multiclass_extra_trees.to_artifact([5; 32]).unwrap(),
    );

    // The standalone trees, in all three fitted shapes. A tree is grown by the
    // same arithmetic-only code path the forest is, so its bytes are tier-1
    // deterministic and are frozen here rather than only round-tripped.
    let tree_regressor_params = DecisionTreeRegressorParams::default()
        .with_max_depth(Some(4))
        .with_max_features(MaxFeatures::All)
        .with_random_state(11);
    let tree_regressor =
        DecisionTreeRegressor::fit(&data.as_view(), &regression, tree_regressor_params.clone())
            .unwrap();
    assert_fingerprint(
        "decision-tree-regressor",
        tree_regressor.to_artifact([5; 32]).unwrap(),
        tree_regressor.to_artifact([5; 32]).unwrap(),
    );

    // The randomized splitter is a second user-facing fit under the same kind,
    // and its thresholds come straight out of the crate's own generator, so it
    // owes its own frozen bytes rather than inheriting the exhaustive tree's.
    let randomized_tree = DecisionTreeRegressor::fit(
        &data.as_view(),
        &regression,
        tree_regressor_params.with_splitter(Splitter::Random),
    )
    .unwrap();
    assert_fingerprint(
        "decision-tree-regressor-randomized",
        randomized_tree.to_artifact([5; 32]).unwrap(),
        randomized_tree.to_artifact([5; 32]).unwrap(),
    );

    let tree_classifier_params = DecisionTreeClassifierParams::default()
        .with_max_depth(Some(4))
        .with_max_features(MaxFeatures::All)
        .with_random_state(11);
    let tree_classifier =
        DecisionTreeClassifier::fit(&data.as_view(), &binary, tree_classifier_params.clone())
            .unwrap();
    assert_fingerprint(
        "decision-tree-classifier",
        tree_classifier.to_artifact([5; 32]).unwrap(),
        tree_classifier.to_artifact([5; 32]).unwrap(),
    );
    let multiclass_tree = DecisionTreeClassifier::fit_multiclass(
        &data.as_view(),
        &ClassTargets::new(vec![3, 7, 10, 7]).unwrap(),
        tree_classifier_params,
    )
    .unwrap();
    assert_fingerprint(
        "decision-tree-classifier-multiclass",
        multiclass_tree.to_artifact([5; 32]).unwrap(),
        multiclass_tree.to_artifact([5; 32]).unwrap(),
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
    );
}
