use ferricml::api::{Classifier, Estimator, HasParams, ModelError, Regressor, Transformer};
use ferricml::data::{BinaryTargets, DenseMatrix, MatrixView, RegressionTargets, SampleWeights};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures, NJobs,
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::pipeline::Pipeline;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
    decisive_directional_accuracy, kendall_tau_b, spearman_correlation, three_way_accuracy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IdentityTransformer {
    features: usize,
}

impl Estimator for IdentityTransformer {
    fn n_features_in(&self) -> usize {
        self.features
    }
}

impl Transformer for IdentityTransformer {
    fn n_features_out(&self) -> usize {
        self.features
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        if data.columns() != self.features {
            return Err(ModelError::FeatureDimension {
                expected: self.features,
                actual: data.columns(),
            });
        }
        if output.len() != data.as_slice().len() {
            return Err(ModelError::OutputLength {
                expected: data.as_slice().len(),
                actual: output.len(),
            });
        }
        output.copy_from_slice(data.as_slice());
        Ok(MatrixView::new(output, data.rows(), self.features)
            .expect("copying a validated matrix preserves validation"))
    }
}

fn training_matrix() -> DenseMatrix {
    DenseMatrix::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0], 4, 2).unwrap()
}

fn estimator_width(estimator: &dyn Estimator) -> usize {
    estimator.n_features_in()
}

fn classifier_width(estimator: &dyn Classifier) -> usize {
    estimator.n_features_in()
}

fn regressor_width(estimator: &dyn Regressor) -> usize {
    estimator.n_features_in()
}

fn transformer_width(transformer: &dyn Transformer) -> (usize, usize) {
    (transformer.n_features_in(), transformer.n_features_out())
}

fn retained_params<E, P>(estimator: &E) -> &P
where
    E: HasParams<Params = P>,
{
    estimator.get_params()
}

#[test]
fn classifier_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(3)
        .with_max_depth(Some(4))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::Count(1))
        .with_bootstrap(false)
        .with_random_state(17)
        .with_n_jobs(NJobs::Serial);

    let model = RandomForestClassifier::fit(&matrix.as_view(), &targets, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(classifier_width(&model), 2);
    assert_eq!(model.n_features_in(), 2);
    assert_eq!(model.get_params(), &params);
    assert_eq!(
        retained_params::<_, RandomForestClassifierParams>(&model),
        &params
    );
    assert_eq!(params.n_estimators(), 3);
    assert_eq!(params.max_depth(), Some(4));
    assert_eq!(params.min_samples_split(), 2);
    assert_eq!(params.min_samples_leaf(), 1);
    assert_eq!(params.max_features(), MaxFeatures::Count(1));
    assert!(!params.bootstrap());
    assert_eq!(params.random_state(), 17);
    assert_eq!(params.n_jobs(), NJobs::Serial);

    let mut positive_probabilities = [0.0; 4];
    model
        .predict_positive_proba_into(&matrix.as_view(), &mut positive_probabilities)
        .unwrap();
    assert!(
        positive_probabilities
            .iter()
            .all(|probability| (0.0..=1.0).contains(probability))
    );
    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(model.predict(&matrix.as_view()).unwrap().len(), 4);
    assert_eq!(model.predict_proba(&matrix.as_view()).unwrap().len(), 8);
}

#[test]
fn logistic_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let params = LogisticRegressionParams::default()
        .with_c(0.5)
        .with_fit_intercept(true)
        .with_max_iter(50)
        .with_tol(1.0e-5);
    let model = LogisticRegression::fit(&matrix.as_view(), &targets, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(classifier_width(&model), 2);
    assert_eq!(
        retained_params::<_, LogisticRegressionParams>(&model),
        &params
    );
    assert_eq!(params.c(), 0.5);
    assert!(params.fit_intercept());
    assert_eq!(params.max_iter(), 50);
    assert_eq!(params.tol(), 1.0e-5);
    assert_eq!(model.coefficients().len(), 2);
    assert!(model.intercept().is_finite());
    assert!(model.n_iter() <= 50);
    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(model.predict(&matrix.as_view()).unwrap().len(), 4);
    assert_eq!(model.predict_proba(&matrix.as_view()).unwrap().len(), 8);

    let weights = SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    assert_eq!(weights.len(), matrix.rows());
    assert_eq!(weights.total(), 6.0);
    let weighted =
        LogisticRegression::fit_weighted(&matrix.as_view(), &targets, &weights, params).unwrap();
    let scores = weighted.decision_function(&matrix.as_view()).unwrap();
    let mut score_output = [0.0; 4];
    weighted
        .decision_function_into(&matrix.as_view(), &mut score_output)
        .unwrap();
    assert_eq!(scores, score_output);
    assert_eq!(
        weighted
            .decision_function_one(matrix.row(0).unwrap())
            .unwrap(),
        scores[0]
    );
}

#[test]
fn regressor_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let params = RandomForestRegressorParams::default()
        .with_n_estimators(2)
        .with_max_depth(Some(3))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::All)
        .with_bootstrap(false)
        .with_random_state(23)
        .with_n_jobs(NJobs::Count(1));

    let model = RandomForestRegressor::fit(&matrix.as_view(), &targets, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(regressor_width(&model), 2);
    assert_eq!(model.n_features_in(), 2);
    assert_eq!(model.get_params(), &params);
    assert_eq!(
        retained_params::<_, RandomForestRegressorParams>(&model),
        &params
    );

    let mut predictions = [0.0; 4];
    model
        .predict_into(&matrix.as_view(), &mut predictions)
        .unwrap();
    assert!(predictions.iter().all(|prediction| prediction.is_finite()));
    assert_eq!(model.predict(&matrix.as_view()).unwrap(), predictions);
}

#[test]
fn linear_regression_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let params = LinearRegressionParams::default()
        .with_fit_intercept(true)
        .with_tol(1.0e-6);
    let model = LinearRegression::fit(&matrix.as_view(), &targets, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(regressor_width(&model), 2);
    assert_eq!(
        retained_params::<_, LinearRegressionParams>(&model),
        &params
    );
    assert!(params.fit_intercept());
    assert_eq!(params.tol(), 1.0e-6);
    assert_eq!(model.coefficients().len(), 2);
    assert!(model.intercept().is_finite());
    assert!(model.rank() <= 2);
    let batch = model.predict(&matrix.as_view()).unwrap();
    let mut output = [0.0; 4];
    model.predict_into(&matrix.as_view(), &mut output).unwrap();
    assert_eq!(batch, output);
    assert_eq!(model.predict_one(matrix.row(0).unwrap()).unwrap(), batch[0]);

    let weighted = LinearRegression::fit_weighted(
        &matrix.as_view(),
        &targets,
        &SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0]).unwrap(),
        params,
    )
    .unwrap();
    assert_eq!(weighted.n_features_in(), 2);
}

#[test]
fn ridge_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let params = RidgeParams::default()
        .with_alpha(0.75)
        .with_fit_intercept(true);
    let model = Ridge::fit(&matrix.as_view(), &targets, params.clone()).unwrap();
    assert_eq!(estimator_width(&model), 2);
    assert_eq!(regressor_width(&model), 2);
    assert_eq!(retained_params::<_, RidgeParams>(&model), &params);
    assert_eq!(params.alpha(), 0.75);
    assert!(params.fit_intercept());
    assert_eq!(
        model.predict(&matrix.as_view()).unwrap().len(),
        matrix.rows()
    );
}

#[test]
fn all_models_share_the_public_model_error_surface() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let error = RandomForestClassifier::fit(
        &matrix.as_view(),
        &targets,
        RandomForestClassifierParams::default().with_n_estimators(0),
    )
    .unwrap_err();

    assert_eq!(error, ModelError::InvalidEstimatorCount);
}

#[test]
fn generic_pipeline_keeps_transform_and_estimator_types_static() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let model = RandomForestClassifier::fit(
        &matrix.as_view(),
        &targets,
        RandomForestClassifierParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false),
    )
    .unwrap();
    let expected = model.predict(&matrix.as_view()).unwrap();
    let pipeline = Pipeline::new(IdentityTransformer { features: 2 }, model).unwrap();

    assert_eq!(pipeline.n_features_in(), 2);
    assert_eq!(pipeline.workspace_len(matrix.rows()).unwrap(), 8);
    assert_eq!(transformer_width(pipeline.transformer()), (2, 2));
    assert_eq!(pipeline.transformer().n_features_out(), 2);
    assert_eq!(pipeline.estimator().n_features_in(), 2);

    let mut workspace = vec![0.0; pipeline.workspace_len(matrix.rows()).unwrap()];
    let mut output = vec![0; matrix.rows()];
    pipeline
        .with_transformed(
            &matrix.as_view(),
            &mut workspace,
            |estimator, transformed| estimator.predict_into(transformed, &mut output),
        )
        .unwrap();
    assert_eq!(output, expected);

    let allocated = pipeline.transform(&matrix.as_view()).unwrap();
    assert_eq!(allocated, matrix);
    let (_, model) = pipeline.into_parts();
    assert_eq!(model.n_features_in(), 2);
}

#[test]
fn pipeline_rejects_an_incompatible_feature_handoff() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let model = RandomForestClassifier::fit(
        &matrix.as_view(),
        &targets,
        RandomForestClassifierParams::default().with_n_estimators(1),
    )
    .unwrap();

    assert_eq!(
        Pipeline::new(IdentityTransformer { features: 3 }, model).unwrap_err(),
        ModelError::FeatureDimension {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn standard_scaler_and_typed_pipeline_paths_are_stable() {
    let matrix = training_matrix();
    let params = StandardScalerParams::default()
        .with_mean(true)
        .with_std(true);
    let scaler = StandardScaler::fit(&matrix.as_view(), params.clone()).unwrap();
    assert_eq!(transformer_width(&scaler), (2, 2));
    assert_eq!(retained_params::<_, StandardScalerParams>(&scaler), &params);
    assert!(params.mean_enabled());
    assert!(params.std_enabled());
    assert_eq!(scaler.means().len(), 2);
    assert_eq!(scaler.variances().len(), 2);
    assert_eq!(scaler.scales().len(), 2);

    let transformed = scaler.transform(&matrix.as_view()).unwrap();
    let model = LogisticRegression::fit(
        &transformed.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let pipeline = Pipeline::new(scaler, model).unwrap();
    let mut workspace = vec![0.0; pipeline.workspace_len(matrix.rows()).unwrap()];
    let mut labels = vec![0; matrix.rows()];
    pipeline
        .predict_into(&matrix.as_view(), &mut workspace, &mut labels)
        .unwrap();
    let artifact = pipeline.to_artifact([1; 32], [2; 32]).unwrap();
    let decoded =
        Pipeline::<StandardScaler, LogisticRegression>::from_artifact(&artifact, [1; 32], [2; 32])
            .unwrap();
    let mut decisions = vec![0.0; matrix.rows()];
    decoded
        .decision_function_into(&matrix.as_view(), &mut workspace, &mut decisions)
        .unwrap();
    assert!(decisions.iter().all(|value| value.is_finite()));
}

#[test]
fn pairwise_ranker_and_metric_paths_are_stable() {
    let matrix = training_matrix();
    let pairs = [
        PairwiseObservation::new(
            PairIndex::new(3, 2).unwrap(),
            PairOutcome::LeftPreferred,
            1.0,
        )
        .unwrap(),
        PairwiseObservation::new(
            PairIndex::new(2, 1).unwrap(),
            PairOutcome::LeftPreferred,
            1.0,
        )
        .unwrap(),
        PairwiseObservation::new(PairIndex::new(1, 0).unwrap(), PairOutcome::Tie, 0.5).unwrap(),
    ];
    let params = PairwiseLinearRankerParams::default()
        .with_c(2.0)
        .with_max_iter(80)
        .with_tol(1.0e-5)
        .with_tie_threshold(0.1);
    let model = PairwiseLinearRanker::fit(&matrix.as_view(), &pairs, params.clone()).unwrap();
    assert_eq!(estimator_width(&model), 2);
    assert_eq!(
        retained_params::<_, PairwiseLinearRankerParams>(&model),
        &params
    );
    assert_eq!(model.coefficients().len(), 2);
    let query = [PairIndex::new(3, 0).unwrap()];
    let mut margins = [0.0];
    model
        .pair_margins_into(&matrix.as_view(), &query, &mut margins)
        .unwrap();
    assert_eq!(model.score_items(&matrix.as_view()).unwrap().len(), 4);
    let artifact = model.to_artifact([5; 32]).unwrap();
    assert_eq!(
        PairwiseLinearRanker::from_artifact(&artifact, [5; 32]).unwrap(),
        model
    );

    assert_eq!(
        decisive_directional_accuracy(&[PairOutcome::LeftPreferred], &[PairOutcome::LeftPreferred]),
        Ok(1.0)
    );
    assert_eq!(
        three_way_accuracy(&[PairOutcome::Tie], &[PairOutcome::Tie]),
        Ok(1.0)
    );
    assert_eq!(spearman_correlation(&[1.0, 2.0], &[3.0, 4.0]), Ok(1.0));
    assert_eq!(kendall_tau_b(&[1.0, 2.0], &[4.0, 3.0]), Ok(-1.0));
}

#[test]
fn histogram_boosting_paths_builders_and_traits_are_stable() {
    let matrix = training_matrix();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let params = HistGradientBoostingRegressorParams::default()
        .with_learning_rate(0.2)
        .with_max_iter(4)
        .with_max_leaf_nodes(3)
        .with_max_depth(Some(2))
        .with_min_samples_leaf(1)
        .with_l2_regularization(0.5)
        .with_max_bins(8);
    let model =
        HistGradientBoostingRegressor::fit(&matrix.as_view(), &targets, params.clone()).unwrap();
    assert_eq!(estimator_width(&model), 2);
    assert_eq!(regressor_width(&model), 2);
    assert_eq!(
        retained_params::<_, HistGradientBoostingRegressorParams>(&model),
        &params
    );
    assert_eq!(params.learning_rate(), 0.2);
    assert_eq!(params.max_iter(), 4);
    assert_eq!(params.max_leaf_nodes(), 3);
    assert_eq!(params.max_depth(), Some(2));
    assert_eq!(params.min_samples_leaf(), 1);
    assert_eq!(params.l2_regularization(), 0.5);
    assert_eq!(params.max_bins(), 8);
    assert_eq!(model.n_iter(), 4);
    assert_eq!(model.predict(&matrix.as_view()).unwrap().len(), 4);
}
