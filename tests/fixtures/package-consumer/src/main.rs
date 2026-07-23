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
    Ok(())
}
