use ferricml::api::{Classifier, Estimator, HasParams, ModelError, Regressor, Transformer};
use ferricml::artifact::{ModelArtifact, StageArtifact};
use ferricml::data::{
    BinaryTargets, DenseMatrix, MatrixView, RegressionTargets, SampleWeights, SelectionError,
};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, ExtraTreesRegressor,
    ExtraTreesRegressorParams, HistGradientBoostingClassifier,
    HistGradientBoostingClassifierParams, HistGradientBoostingRegressor,
    HistGradientBoostingRegressorParams, MaxFeatures, NJobs, RandomForestClassifier,
    RandomForestClassifierParams, RandomForestRegressor, RandomForestRegressorParams,
};
use ferricml::linear_model::{
    LinearRegression, LinearRegressionParams, LogisticRegression, LogisticRegressionParams, Ridge,
    RidgeParams,
};
use ferricml::metrics::{
    MetricError, accuracy_score, binary_confusion_matrix, brier_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error,
};
use ferricml::model_selection::{
    ClassificationScorer, CrossValidationError, HoldoutParams, KFold, RegressionScorer,
    ScorableClassifier, ScoringError, Split, SplitError, StratifiedKFold, TestSize,
    cross_validate_classifier, cross_validate_regressor, score_classifier, score_regressor,
    stratified_train_test_split, train_test_split,
};
use ferricml::pipeline::Pipeline;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams, PairwiseObservation,
    decisive_directional_accuracy, kendall_tau_b, spearman_correlation, three_way_accuracy,
};
use ferricml::tree::{
    DecisionTreeClassifier, DecisionTreeClassifierParams, DecisionTreeRegressor,
    DecisionTreeRegressorParams,
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
fn evaluation_metric_paths_and_results_are_stable() {
    let expected = [0, 0, 1, 1];
    let predicted = [0, 1, 1, 1];
    let probabilities = [0.1, 0.8, 0.7, 0.9];
    let confusion = binary_confusion_matrix(&expected, &predicted).unwrap();
    assert_eq!(confusion.true_negatives(), 1);
    assert_eq!(confusion.false_positives(), 1);
    assert_eq!(confusion.false_negatives(), 0);
    assert_eq!(confusion.true_positives(), 2);
    assert_eq!(confusion.total(), 4);
    assert_eq!(accuracy_score(&expected, &predicted), Ok(0.75));
    assert_eq!(precision_score(&expected, &predicted), Ok(2.0 / 3.0));
    assert_eq!(recall_score(&expected, &predicted), Ok(1.0));
    assert_eq!(f1_score(&expected, &predicted), Ok(0.8));
    assert!(brier_score(&expected, &probabilities).unwrap().is_finite());
    assert!(log_loss(&expected, &probabilities).unwrap().is_finite());
    assert_eq!(roc_auc_score(&expected, &probabilities), Ok(0.75));

    let regression_expected = [1.0, 2.0, 3.0];
    let regression_predicted = [1.0, 3.0, 2.0];
    assert!(
        mean_absolute_error(&regression_expected, &regression_predicted)
            .unwrap()
            .is_finite()
    );
    assert!(
        mean_squared_error(&regression_expected, &regression_predicted)
            .unwrap()
            .is_finite()
    );
    assert!(
        root_mean_squared_error(&regression_expected, &regression_predicted)
            .unwrap()
            .is_finite()
    );
    assert!(
        r2_score(&regression_expected, &regression_predicted)
            .unwrap()
            .is_finite()
    );
    assert!(matches!(
        precision_score(&[0], &[0]),
        Err(MetricError::Undefined)
    ));
}

#[test]
fn deterministic_selection_paths_and_materialization_are_stable() {
    let matrix = training_matrix();
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let selected = matrix.select_rows(&[3, 1]).unwrap();
    assert_eq!(selected.as_slice(), &[3.0, 9.0, 1.0, 1.0]);
    assert_eq!(binary.select(&[3, 1]).unwrap().as_slice(), &[1, 0]);
    assert_eq!(regression.select(&[3, 1]).unwrap().as_slice(), &[9.0, 1.0]);
    assert!(matches!(
        matrix.select_rows(&[]),
        Err(SelectionError::Empty)
    ));

    let params = HoldoutParams::default()
        .with_test_size(TestSize::Count(2))
        .with_shuffle(false)
        .with_random_state(7);
    assert_eq!(params.test_size(), TestSize::Count(2));
    assert!(!params.shuffle());
    assert_eq!(params.random_state(), 7);
    let holdout = train_test_split(4, params).unwrap();
    assert_eq!(holdout.train_indices(), &[0, 1]);
    assert_eq!(holdout.test_indices(), &[2, 3]);
    assert_eq!(
        Split::new(4, vec![0, 2], vec![1, 3]).unwrap(),
        Split::new(4, vec![0, 2], vec![1, 3]).unwrap()
    );
    assert!(matches!(
        Split::new(4, vec![0, 1], vec![1, 3]),
        Err(SplitError::OverlappingIndex { index: 1 })
    ));

    let stratified = stratified_train_test_split(
        binary.as_slice(),
        HoldoutParams::default()
            .with_test_size(TestSize::Count(2))
            .with_shuffle(false),
    )
    .unwrap();
    assert_eq!(stratified.train_indices(), &[0, 2]);
    assert_eq!(stratified.test_indices(), &[1, 3]);

    let kfold = KFold::new(2).with_shuffle(true).with_random_state(3);
    assert_eq!(kfold.n_splits(), 2);
    assert!(kfold.shuffle());
    assert_eq!(kfold.random_state(), 3);
    assert_eq!(kfold.split(4).unwrap().len(), 2);

    let stratified_kfold = StratifiedKFold::new(2)
        .with_shuffle(true)
        .with_random_state(3);
    assert_eq!(stratified_kfold.n_splits(), 2);
    assert!(stratified_kfold.shuffle());
    assert_eq!(stratified_kfold.random_state(), 3);
    assert_eq!(stratified_kfold.split(binary.as_slice()).unwrap().len(), 2);
}

#[test]
fn fitted_estimator_scoring_paths_are_stable() {
    let matrix = training_matrix();
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let classifier = RandomForestClassifier::fit(
        &matrix.as_view(),
        &binary,
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false),
    )
    .unwrap();
    for scorer in [
        ClassificationScorer::Accuracy,
        ClassificationScorer::Precision,
        ClassificationScorer::Recall,
        ClassificationScorer::F1,
        ClassificationScorer::Brier,
        ClassificationScorer::LogLoss,
        ClassificationScorer::RocAuc,
    ] {
        assert!(
            score_classifier(
                ScorableClassifier::probabilistic(&classifier),
                &matrix.as_view(),
                &binary,
                scorer
            )
            .is_ok()
        );
    }

    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let regressor = Ridge::fit(&matrix.as_view(), &regression, RidgeParams::default()).unwrap();
    for scorer in [
        RegressionScorer::MeanAbsoluteError,
        RegressionScorer::MeanSquaredError,
        RegressionScorer::RootMeanSquaredError,
        RegressionScorer::R2,
    ] {
        assert!(score_regressor(&regressor, &matrix.as_view(), &regression, scorer).is_ok());
    }
    assert!(matches!(
        score_classifier(
            ScorableClassifier::probabilistic(&classifier),
            &matrix.as_view(),
            &BinaryTargets::new(vec![0, 1]).unwrap(),
            ClassificationScorer::Accuracy,
        ),
        Err(ScoringError::TargetLength {
            rows: 4,
            targets: 2
        })
    ));
}

#[test]
fn closure_based_cross_validation_paths_are_stable() {
    let matrix = training_matrix();
    let binary = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let classifier = cross_validate_classifier(
        &matrix.as_view(),
        &binary,
        StratifiedKFold::new(2).split(binary.as_slice()).unwrap(),
        ClassificationScorer::Accuracy,
        |train, targets| {
            RandomForestClassifier::fit(
                train,
                targets,
                RandomForestClassifierParams::default()
                    .with_n_estimators(1)
                    .with_bootstrap(false),
            )
        },
        |model| ScorableClassifier::probabilistic(model),
    )
    .unwrap();
    assert_eq!(classifier.len(), 2);
    assert!(!classifier.is_empty());
    assert_eq!(classifier.scores().len(), 2);
    assert!(classifier.mean().is_finite());
    assert!(classifier.population_standard_deviation().is_finite());

    let regression = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let regressor = cross_validate_regressor(
        &matrix.as_view(),
        &regression,
        KFold::new(2).split(matrix.rows()).unwrap(),
        RegressionScorer::MeanSquaredError,
        |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
    )
    .unwrap();
    assert_eq!(regressor.len(), 2);
    assert!(matches!(
        cross_validate_regressor::<Ridge, _, _, _>(
            &matrix.as_view(),
            &regression,
            std::iter::empty(),
            RegressionScorer::MeanSquaredError,
            |train, targets| Ridge::fit(train, targets, RidgeParams::default()),
        ),
        Err(CrossValidationError::NoSplits)
    ));
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
fn extra_trees_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let labels = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let params = ExtraTreesClassifierParams::default()
        .with_n_estimators(3)
        .with_max_depth(Some(4))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::Count(1))
        .with_bootstrap(false)
        .with_random_state(17)
        .with_n_jobs(NJobs::Serial);

    let model = ExtraTreesClassifier::fit(&matrix.as_view(), &labels, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(classifier_width(&model), 2);
    assert_eq!(model.get_params(), &params);
    assert_eq!(
        retained_params::<_, ExtraTreesClassifierParams>(&model),
        &params
    );
    // The parameter vocabulary is the random forest's, and `splitter` is
    // deliberately **not** on it: it is what the type means, so a caller cannot
    // set it back and get a random forest under a second name.
    assert_eq!(params.n_estimators(), 3);
    assert_eq!(params.max_depth(), Some(4));
    assert_eq!(params.min_samples_split(), 2);
    assert_eq!(params.min_samples_leaf(), 1);
    assert_eq!(params.max_features(), MaxFeatures::Count(1));
    assert!(!params.bootstrap());
    assert_eq!(params.random_state(), 17);
    assert_eq!(params.n_jobs(), NJobs::Serial);

    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(model.predict(&matrix.as_view()).unwrap().len(), 4);
    assert_eq!(model.predict_proba(&matrix.as_view()).unwrap().len(), 8);
    let mut positive = [0.0; 4];
    model
        .predict_positive_proba_into(&matrix.as_view(), &mut positive)
        .unwrap();
    assert!(positive.iter().all(|value| (0.0..=1.0).contains(value)));

    let regressor_params = ExtraTreesRegressorParams::default()
        .with_n_estimators(2)
        .with_max_depth(Some(3))
        .with_max_features(MaxFeatures::All)
        .with_random_state(23);
    let regressor =
        ExtraTreesRegressor::fit(&matrix.as_view(), &targets, regressor_params.clone()).unwrap();
    assert_eq!(regressor_width(&regressor), 2);
    assert_eq!(regressor.get_params(), &regressor_params);
    let mut predictions = [0.0; 4];
    regressor
        .predict_into(&matrix.as_view(), &mut predictions)
        .unwrap();
    assert!(predictions.iter().all(|prediction| prediction.is_finite()));
    assert_eq!(regressor.predict(&matrix.as_view()).unwrap(), predictions);
}

#[test]
fn decision_tree_classifier_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let params = DecisionTreeClassifierParams::default()
        .with_max_depth(Some(4))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::Count(1))
        .with_random_state(17);

    let model = DecisionTreeClassifier::fit(&matrix.as_view(), &targets, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(classifier_width(&model), 2);
    assert_eq!(model.n_features_in(), 2);
    assert_eq!(model.get_params(), &params);
    assert_eq!(
        retained_params::<_, DecisionTreeClassifierParams>(&model),
        &params
    );
    // A single tree has no ensemble around it, so the parameter surface stops
    // exactly where growing one tree stops: no member count, no bootstrap, no
    // thread count. Reading every accessor back here is what makes a silently
    // widened parameter set a failing test rather than a snapshot-only diff.
    assert_eq!(params.max_depth(), Some(4));
    assert_eq!(params.min_samples_split(), 2);
    assert_eq!(params.min_samples_leaf(), 1);
    assert_eq!(params.max_features(), MaxFeatures::Count(1));
    assert_eq!(params.random_state(), 17);

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
    assert_eq!(
        model
            .predict_proba_one(matrix.row(0).unwrap())
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        model
            .predict_class_proba(&matrix.as_view(), 1)
            .unwrap()
            .len(),
        4
    );
}

#[test]
fn decision_tree_regressor_paths_builders_traits_and_retained_params_are_stable() {
    let matrix = training_matrix();
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let params = DecisionTreeRegressorParams::default()
        .with_max_depth(Some(3))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::All)
        .with_random_state(23);

    let model = DecisionTreeRegressor::fit(&matrix.as_view(), &targets, params.clone()).unwrap();

    assert_eq!(estimator_width(&model), 2);
    assert_eq!(regressor_width(&model), 2);
    assert_eq!(model.n_features_in(), 2);
    assert_eq!(model.get_params(), &params);
    assert_eq!(
        retained_params::<_, DecisionTreeRegressorParams>(&model),
        &params
    );

    let mut predictions = [0.0; 4];
    model
        .predict_into(&matrix.as_view(), &mut predictions)
        .unwrap();
    assert!(predictions.iter().all(|prediction| prediction.is_finite()));
    assert_eq!(model.predict(&matrix.as_view()).unwrap(), predictions);
    assert_eq!(
        model.predict_one(matrix.row(0).unwrap()).unwrap(),
        predictions[0]
    );

    let weights = SampleWeights::new(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let weighted =
        DecisionTreeRegressor::fit_weighted(&matrix.as_view(), &targets, &weights, params).unwrap();
    assert_eq!(weighted.predict(&matrix.as_view()).unwrap(), predictions);
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

    // This file deliberately does not import `ProbabilisticClassifier`, so
    // both of these resolve only through inherent forwarders. The allocating
    // one existed and the caller-owned one did not, which made the
    // allocation-free path the one that needed a trait import — the exact
    // inversion of the crate's preference on hot paths.
    let column = model.predict_class_proba(&matrix.as_view(), 1).unwrap();
    let mut owned_column = [0.0; 4];
    model
        .predict_class_proba_into(&matrix.as_view(), 1, &mut owned_column)
        .unwrap();
    assert_eq!(column, owned_column);

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
    let artifact = model.to_artifact([17; 32]).unwrap();
    let decoded = HistGradientBoostingRegressor::from_artifact(&artifact, [17; 32]).unwrap();
    assert_eq!(decoded, model);
}

#[test]
fn boosted_classifier_paths_builders_and_traits_are_stable() {
    let matrix = training_matrix();
    let targets = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let params = HistGradientBoostingClassifierParams::default()
        .with_learning_rate(0.2)
        .with_max_iter(4)
        .with_max_leaf_nodes(3)
        .with_max_depth(Some(2))
        .with_min_samples_leaf(1)
        .with_l2_regularization(0.5)
        .with_max_bins(8);
    let model =
        HistGradientBoostingClassifier::fit(&matrix.as_view(), &targets, params.clone()).unwrap();
    assert_eq!(estimator_width(&model), 2);
    assert_eq!(classifier_width(&model), 2);
    assert_eq!(
        retained_params::<_, HistGradientBoostingClassifierParams>(&model),
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
    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(model.predict(&matrix.as_view()).unwrap().len(), 4);
    assert_eq!(model.predict_proba(&matrix.as_view()).unwrap().len(), 8);
    assert_eq!(model.decision_function(&matrix.as_view()).unwrap().len(), 4);
    assert_eq!(
        model
            .predict_class_proba(&matrix.as_view(), 1)
            .unwrap()
            .len(),
        4
    );
    let row = matrix.row(0).unwrap();
    assert!(model.decision_function_one(row).unwrap().is_finite());
    assert!((0.0..=1.0).contains(&model.predict_positive_proba_one(row).unwrap()));
    assert!(model.predict_one(row).unwrap() <= 1);
    assert!(model.baseline().is_finite());

    let weights = SampleWeights::new(vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    let weighted =
        HistGradientBoostingClassifier::fit_weighted(&matrix.as_view(), &targets, &weights, params)
            .unwrap();
    assert_eq!(weighted.n_iter(), 4);

    let artifact = model.to_artifact([17; 32]).unwrap();
    let decoded = HistGradientBoostingClassifier::from_artifact(&artifact, [17; 32]).unwrap();
    assert_eq!(decoded, model);
}
