use ferricml::api::{AnyRegressor, AnyRegressorParams, Classifier, Regressor};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets, SampleWeights};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor, RandomForestRegressorParams,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::inspection::{PermutationImportanceParams, permutation_importance_regressor};
use ferricml::metrics::accuracy_score;
use ferricml::model_selection::{
    HoldoutParams, KFold, RegressionScorer, TestSize, cross_validate_regressor,
    train_test_split,
};
use ferricml::pipeline::Pipeline;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams,
    PairwiseObservation, kendall_tau_b,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features = DenseMatrix::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0], 4, 2)?;
    let classifier = RandomForestClassifier::fit(
        &features.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1])?,
        RandomForestClassifierParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_random_state(7),
    )?;
    let labels = Classifier::predict(&classifier, &features.as_view())?;
    let probabilities = Classifier::predict_proba(&classifier, &features.as_view())?;
    assert_eq!(labels.len(), features.rows());
    assert_eq!(probabilities.len(), features.rows() * 2);
    assert_eq!(accuracy_score(&[0, 0, 1, 1], &labels)?, 1.0);

    let holdout = train_test_split(
        features.rows(),
        HoldoutParams::default()
            .with_test_size(TestSize::Count(1))
            .with_random_state(7),
    )?;
    assert_eq!(holdout.train_indices().len(), 3);
    assert_eq!(holdout.test_indices().len(), 1);

    let logistic = LogisticRegression::fit_weighted(
        &features.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1])?,
        &SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0])?,
        LogisticRegressionParams::default(),
    )?;
    let mut decisions = vec![0.0; features.rows()];
    logistic.decision_function_into(&features.as_view(), &mut decisions)?;
    assert!(decisions.iter().all(|value| value.is_finite()));
    let schema = [7; 32];
    let encoded = logistic.to_artifact(schema)?;
    let decoded = LogisticRegression::from_artifact(&encoded, schema)?;
    assert_eq!(
        decoded.predict_proba(&features.as_view())?,
        logistic.predict_proba(&features.as_view())?
    );

    let linear = LinearRegression::fit_weighted(
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        &SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0])?,
        LinearRegressionParams::default(),
    )?;
    let linear_encoded = linear.to_artifact(schema)?;
    let linear_decoded = LinearRegression::from_artifact(&linear_encoded, schema)?;
    assert_eq!(
        linear_decoded.predict(&features.as_view())?,
        linear.predict(&features.as_view())?
    );
    let ridge = Ridge::fit(
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        RidgeParams::default(),
    )?;
    let ridge_encoded = ridge.to_artifact(schema)?;
    assert_eq!(
        Ridge::from_artifact(&ridge_encoded, schema)?.predict(&features.as_view())?,
        ridge.predict(&features.as_view())?
    );
    let cross_validation = cross_validate_regressor(
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        KFold::new(2).split(features.rows())?,
        RegressionScorer::RootMeanSquaredError,
        |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
    )?;
    assert_eq!(cross_validation.len(), 2);
    assert!(cross_validation.mean().is_finite());

    let transformed_schema = [8; 32];
    let scaler = StandardScaler::fit(&features.as_view(), StandardScalerParams::default())?;
    let transformed = scaler.transform(&features.as_view())?;
    let pipeline_model = Ridge::fit(
        &transformed.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        RidgeParams::default(),
    )?;
    let pipeline = Pipeline::new(scaler, pipeline_model)?;
    let pipeline_encoded = pipeline.to_artifact(schema, transformed_schema)?;
    let pipeline = Pipeline::<StandardScaler, Ridge>::from_artifact(
        &pipeline_encoded,
        schema,
        transformed_schema,
    )?;
    let mut workspace = vec![0.0; pipeline.workspace_len(features.rows())?];
    let mut pipeline_predictions = vec![0.0; features.rows()];
    pipeline.predict_into(
        &features.as_view(),
        &mut workspace,
        &mut pipeline_predictions,
    )?;
    assert!(pipeline_predictions.iter().all(|value| value.is_finite()));

    let pair_observations = [
        PairwiseObservation::new(
            PairIndex::new(3, 2)?,
            PairOutcome::LeftPreferred,
            1.0,
        )?,
        PairwiseObservation::new(
            PairIndex::new(2, 1)?,
            PairOutcome::LeftPreferred,
            1.0,
        )?,
        PairwiseObservation::new(
            PairIndex::new(1, 0)?,
            PairOutcome::LeftPreferred,
            1.0,
        )?,
    ];
    let ranker = PairwiseLinearRanker::fit(
        &features.as_view(),
        &pair_observations,
        PairwiseLinearRankerParams::default(),
    )?;
    let ranker_encoded = ranker.to_artifact(schema)?;
    let ranker = PairwiseLinearRanker::from_artifact(&ranker_encoded, schema)?;
    let scores = ranker.score_items(&features.as_view())?;
    assert_eq!(scores.len(), features.rows());
    assert_eq!(kendall_tau_b(&scores.iter().map(|&value| f64::from(value)).collect::<Vec<_>>(), &[0.0, 1.0, 2.0, 3.0])?, 1.0);

    let boosted = HistGradientBoostingRegressor::fit(
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(4)
            .with_max_leaf_nodes(3)
            .with_min_samples_leaf(1),
    )?;
    let mut boosted_predictions = vec![0.0; features.rows()];
    boosted.predict_into(&features.as_view(), &mut boosted_predictions)?;
    assert!(boosted_predictions.iter().all(|value| value.is_finite()));
    let boosted_encoded = boosted.to_artifact(schema)?;
    let boosted =
        HistGradientBoostingRegressor::from_artifact(&boosted_encoded, schema)?;
    assert_eq!(
        boosted.predict(&features.as_view())?,
        boosted_predictions
    );
    let boosted: AnyRegressor = boosted.into();
    assert!(matches!(
        boosted.get_params(),
        AnyRegressorParams::HistGradientBoosting(_)
    ));

    let regressor = RandomForestRegressor::fit(
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        RandomForestRegressorParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_random_state(7),
    )?;
    let predictions = Regressor::predict(&regressor, &features.as_view())?;
    assert_eq!(predictions.len(), features.rows());
    assert!(predictions.iter().all(|value| value.is_finite()));

    let forest_encoded = regressor.to_artifact(schema)?;
    assert_eq!(forest_encoded, regressor.to_artifact(schema)?);
    let forest_decoded = RandomForestRegressor::from_artifact(&forest_encoded, schema)?;
    assert_eq!(forest_decoded.n_features_in(), regressor.n_features_in());
    assert_eq!(forest_decoded.get_params(), regressor.get_params());
    assert_eq!(
        Regressor::predict(&forest_decoded, &features.as_view())?,
        predictions
    );
    assert!(RandomForestRegressor::from_artifact(&forest_encoded, transformed_schema).is_err());

    let dispatch: AnyRegressor = forest_decoded.into();
    let dispatch_encoded = dispatch.to_artifact(schema)?;
    let dispatch_decoded = AnyRegressor::from_artifact(&dispatch_encoded, schema)?;
    assert!(matches!(
        dispatch_decoded.get_params(),
        AnyRegressorParams::RandomForest(_)
    ));
    assert_eq!(dispatch_decoded.predict(&features.as_view())?, predictions);
    assert!(AnyRegressor::from_artifact(&forest_encoded, schema).is_err());

    let importance = permutation_importance_regressor(
        &dispatch_decoded,
        &features.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?,
        RegressionScorer::MeanSquaredError,
        PermutationImportanceParams::default()
            .with_n_repeats(3)
            .with_random_state(1),
    )?;
    assert_eq!(importance.n_features(), features.columns());
    assert_eq!(importance.ranked().len(), features.columns());
    assert!(importance.means().iter().all(|value| value.is_finite()));
    Ok(())
}
