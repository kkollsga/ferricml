use ferricml::api::{AnyClassifier, AnyRegressor, AnyRegressorParams, Classifier, Regressor};
use ferricml::data::{
    BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights,
};
use ferricml::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingClassifierParams,
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
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    MaxAbsScaler, MaxAbsScalerParams, MinMaxScaler, MinMaxScalerParams, StandardScaler,
    StandardScalerParams,
};
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

    // Classifier persistence, from outside the crate, for both fitted leaf
    // representations and through the runtime dispatch enum.
    let schema = [7; 32];
    let encoded = classifier.to_artifact(schema)?;
    assert_eq!(encoded, classifier.to_artifact(schema)?);
    let decoded = RandomForestClassifier::from_artifact(&encoded, schema)?;
    assert_eq!(decoded.n_features_in(), classifier.n_features_in());
    assert_eq!(decoded.get_params(), classifier.get_params());
    assert_eq!(decoded.classes(), classifier.classes());
    assert_eq!(Classifier::predict(&decoded, &features.as_view())?, labels);
    assert_eq!(
        Classifier::predict_proba(&decoded, &features.as_view())?,
        probabilities
    );
    assert!(RandomForestClassifier::from_artifact(&encoded, [8; 32]).is_err());

    let multiclass = RandomForestClassifier::fit_multiclass(
        &features.as_view(),
        &ClassTargets::new(vec![3, 7, 10, 7])?,
        RandomForestClassifierParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_random_state(7),
    )?;
    let multiclass_encoded = multiclass.to_artifact(schema)?;
    let multiclass_decoded = RandomForestClassifier::from_artifact(&multiclass_encoded, schema)?;
    assert_eq!(multiclass_decoded.classes(), &[3, 7, 10]);
    assert_eq!(
        Classifier::predict_proba(&multiclass_decoded, &features.as_view())?,
        Classifier::predict_proba(&multiclass, &features.as_view())?
    );
    assert_eq!(multiclass_decoded.to_artifact(schema)?, multiclass_encoded);

    let erased: AnyClassifier = multiclass.into();
    let dispatch_encoded = erased.to_artifact(schema)?;
    let dispatch_decoded = AnyClassifier::from_artifact(&dispatch_encoded, schema)?;
    assert_eq!(dispatch_decoded.classes(), &[3, 7, 10]);
    assert_eq!(
        dispatch_decoded.predict_proba(&features.as_view())?,
        erased.predict_proba(&features.as_view())?
    );
    // A bare classifier artifact is not a dispatch artifact.
    assert!(AnyClassifier::from_artifact(&multiclass_encoded, schema).is_err());

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
    let encoded = logistic.to_artifact(schema)?;
    let decoded = LogisticRegression::from_artifact(&encoded, schema)?;
    assert_eq!(
        decoded.predict_proba(&features.as_view())?,
        logistic.predict_proba(&features.as_view())?
    );

    // The joint multinomial fit persists under its own payload schema.
    let multinomial = LogisticRegression::fit_multiclass(
        &features.as_view(),
        &ClassTargets::new(vec![3, 7, 10, 7])?,
        LogisticRegressionParams::default(),
    )?;
    let multinomial_encoded = multinomial.to_artifact(schema)?;
    let multinomial_decoded = LogisticRegression::from_artifact(&multinomial_encoded, schema)?;
    assert_eq!(multinomial_decoded.classes(), &[3, 7, 10]);
    assert_eq!(multinomial_decoded.n_decision_columns(), 3);
    assert_eq!(
        multinomial_decoded.predict_proba(&features.as_view())?,
        multinomial.predict_proba(&features.as_view())?
    );
    assert_eq!(multinomial_decoded.to_artifact(schema)?, multinomial_encoded);

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

    let staged_targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?;
    let staged: StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> = StagedPipeline::fit(
        &features.as_view(),
        |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
        |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
        |batch| Ridge::fit(batch, &staged_targets, RidgeParams::default()),
    )?;
    let mut staged_workspace = vec![0.0; staged.workspace_len(features.rows())?];
    let mut staged_predictions = vec![0.0; features.rows()];
    staged.with_transformed(&features.as_view(), &mut staged_workspace, |model, batch| {
        model.predict_into(batch, &mut staged_predictions)
    })?;
    let staged_encoded = staged.to_artifact(schema, transformed_schema)?;
    assert_eq!(staged_encoded, staged.to_artifact(schema, transformed_schema)?);
    let staged_decoded = StagedPipeline::<(MinMaxScaler, StandardScaler), Ridge>::from_artifact(
        &staged_encoded,
        schema,
        transformed_schema,
    )?;
    let mut decoded_predictions = vec![0.0; features.rows()];
    staged_decoded.with_transformed(
        &features.as_view(),
        &mut staged_workspace,
        |model, batch| model.predict_into(batch, &mut decoded_predictions),
    )?;
    assert_eq!(decoded_predictions, staged_predictions);
    // A different composition never decodes another one's bytes.
    assert!(
        StagedPipeline::<(MinMaxScaler, MaxAbsScaler), Ridge>::from_artifact(
            &staged_encoded,
            schema,
            transformed_schema
        )
        .is_err()
    );
    let _ = MaxAbsScaler::fit(&features.as_view(), MaxAbsScalerParams)?;

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

    let boosted_classifier = HistGradientBoostingClassifier::fit(
        &features.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1])?,
        HistGradientBoostingClassifierParams::default()
            .with_max_iter(4)
            .with_max_leaf_nodes(3)
            .with_min_samples_leaf(1),
    )?;
    let boosted_scores = boosted_classifier.decision_function(&features.as_view())?;
    let boosted_proba = boosted_classifier.predict_proba(&features.as_view())?;
    let boosted_labels = Classifier::predict(&boosted_classifier, &features.as_view())?;
    assert_eq!(boosted_classifier.classes(), &[0, 1]);
    assert_eq!(boosted_scores.len(), features.rows());
    assert_eq!(boosted_proba.len(), 2 * features.rows());
    assert!(boosted_proba.iter().all(|value| (0.0..=1.0).contains(value)));
    let boosted_classifier_encoded = boosted_classifier.to_artifact(schema)?;
    let boosted_classifier =
        HistGradientBoostingClassifier::from_artifact(&boosted_classifier_encoded, schema)?;
    assert_eq!(
        boosted_classifier.to_artifact(schema)?,
        boosted_classifier_encoded
    );
    assert_eq!(
        Classifier::predict(&boosted_classifier, &features.as_view())?,
        boosted_labels
    );
    assert_eq!(
        boosted_classifier.predict_proba(&features.as_view())?,
        boosted_proba
    );
    assert_eq!(
        boosted_classifier.decision_function(&features.as_view())?,
        boosted_scores
    );

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
