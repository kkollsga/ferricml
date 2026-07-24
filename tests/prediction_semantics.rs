use ferricml::api::{
    AnyClassifier, AnyClassifierParams, AnyRegressor, AnyRegressorParams, Classifier, ModelError,
    Regressor,
};
use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams, MaxFeatures, NJobs,
    RandomForestClassifier, RandomForestClassifierParams, RandomForestRegressor,
    RandomForestRegressorParams,
};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams, Ridge, RidgeParams};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};
use ferricml::ranking::{
    PairIndex, PairOutcome, PairwiseError, PairwiseLinearRanker, PairwiseLinearRankerParams,
    PairwiseObservation,
};

fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
    DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
}

fn classifier(
    data: &DenseMatrix,
    labels: Vec<u8>,
    min_samples_split: usize,
) -> RandomForestClassifier {
    RandomForestClassifier::fit(
        &data.as_view(),
        &BinaryTargets::new(labels).unwrap(),
        RandomForestClassifierParams::default()
            .with_n_estimators(1)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All)
            .with_min_samples_split(min_samples_split),
    )
    .unwrap()
}

fn assert_any_classifier_errors_are_atomic(model: &AnyClassifier, data: &DenseMatrix) {
    let mut labels = [7_u8; 3];
    assert_eq!(
        model.predict_into(&data.as_view(), &mut labels),
        Err(ModelError::OutputLength {
            expected: data.rows(),
            actual: 3,
        })
    );
    assert_eq!(labels, [7; 3]);
    let mut probabilities = [7.0_f32; 7];
    assert_eq!(
        model.predict_proba_into(&data.as_view(), &mut probabilities),
        Err(ModelError::OutputLength {
            expected: data.rows() * model.classes().len(),
            actual: 7,
        })
    );
    assert_eq!(probabilities, [7.0; 7]);
    let mut class = [7.0_f32; 4];
    assert_eq!(
        model.predict_class_proba_into(&data.as_view(), 9, &mut class),
        Err(ModelError::UnknownClass { class: 9 })
    );
    assert_eq!(class, [7.0; 4]);
}

fn assert_any_regressor_errors_are_atomic(model: &AnyRegressor, data: &DenseMatrix) {
    let mut output = [7.0_f32; 3];
    assert_eq!(
        model.predict_into(&data.as_view(), &mut output),
        Err(ModelError::OutputLength {
            expected: data.rows(),
            actual: 3,
        })
    );
    assert_eq!(output, [7.0; 3]);
}

#[test]
fn binary_probabilities_are_row_major_in_sorted_class_order() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let model = classifier(&data, vec![0, 0, 1, 1], 2);

    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(Classifier::classes(&model), &[0, 1]);
    assert_eq!(model.n_features_in(), 1);

    let probabilities = model.predict_proba(&data.as_view()).unwrap();
    assert_eq!(probabilities.len(), data.rows() * model.classes().len());
    for row in probabilities.chunks_exact(2) {
        assert!((row[0] + row[1] - 1.0).abs() <= f32::EPSILON);
    }

    let positive = model.predict_class_proba(&data.as_view(), 1).unwrap();
    let negative = model.predict_class_proba(&data.as_view(), 0).unwrap();
    for ((row, &p1), &p0) in probabilities.chunks_exact(2).zip(&positive).zip(&negative) {
        assert_eq!(row, [p0, p1]);
    }
    assert_eq!(model.predict(&data.as_view()).unwrap(), vec![0, 0, 1, 1]);
}

#[test]
fn exact_probability_tie_selects_the_first_smaller_class() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let model = classifier(&data, vec![0, 1, 0, 1], 5);

    assert_eq!(model.predict_proba_one(&[10.0]).unwrap(), vec![0.5, 0.5]);
    assert_eq!(model.predict_one(&[10.0]).unwrap(), 0);
    assert_eq!(model.predict(&data.as_view()).unwrap(), vec![0; 4]);
}

#[test]
fn single_class_models_use_one_probability_column() {
    let data = matrix(&[0.0, 1.0, 2.0], 3, 1);
    for (label, absent) in [(0, 1), (1, 0)] {
        let model = classifier(&data, vec![label; 3], 2);

        assert_eq!(model.classes(), &[label]);
        assert_eq!(model.predict(&data.as_view()).unwrap(), vec![label; 3]);
        assert_eq!(model.predict_proba(&data.as_view()).unwrap(), vec![1.0; 3]);
        assert_eq!(model.predict_proba_one(&[1.5]).unwrap(), vec![1.0]);
        assert_eq!(
            model.predict_class_proba(&data.as_view(), label).unwrap(),
            vec![1.0; 3]
        );
        assert_eq!(
            model
                .predict_class_proba(&data.as_view(), absent)
                .unwrap_err(),
            ModelError::UnknownClass { class: absent }
        );
        assert_eq!(
            model.predict_positive_proba(&[1.5]).unwrap(),
            f32::from(label)
        );
    }
}

#[test]
fn output_validation_happens_before_writing() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let model = classifier(&data, vec![0, 0, 1, 1], 2);

    let mut labels = [9_u8; 3];
    assert_eq!(
        model
            .predict_into(&data.as_view(), &mut labels)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 4,
            actual: 3
        }
    );
    assert_eq!(labels, [9; 3]);

    let mut probabilities = [9.0_f32; 7];
    assert_eq!(
        model
            .predict_proba_into(&data.as_view(), &mut probabilities)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 8,
            actual: 7
        }
    );
    assert_eq!(probabilities, [9.0; 7]);

    let mut one_row = [9.0_f32; 1];
    assert_eq!(
        model
            .predict_proba_one_into(&[1.0], &mut one_row)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(one_row, [9.0]);

    let wrong_width = matrix(&[0.0, 0.0, 1.0, 1.0], 2, 2);
    let mut output = [9.0_f32; 4];
    assert_eq!(
        model
            .predict_proba_into(&wrong_width.as_view(), &mut output)
            .unwrap_err(),
        ModelError::FeatureDimension {
            expected: 1,
            actual: 2
        }
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn scalar_batch_allocating_and_parallel_models_agree() {
    let data = matrix(
        &[
            0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 7.0, 5.0, 6.0, 6.0, 5.0, 7.0, 4.0,
        ],
        8,
        2,
    );
    let targets = BinaryTargets::new(vec![0, 1, 1, 0, 1, 0, 0, 1]).unwrap();
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(31)
        .with_max_depth(Some(8))
        .with_max_features(MaxFeatures::All)
        .with_random_state(123);
    let serial = RandomForestClassifier::fit(&data.as_view(), &targets, params.clone()).unwrap();
    let parallel = RandomForestClassifier::fit(
        &data.as_view(),
        &targets,
        params.with_n_jobs(NJobs::Count(4)),
    )
    .unwrap();

    assert_eq!(
        serial.predict(&data.as_view()).unwrap(),
        parallel.predict(&data.as_view()).unwrap()
    );
    assert_eq!(
        serial.predict_proba(&data.as_view()).unwrap(),
        parallel.predict_proba(&data.as_view()).unwrap()
    );

    let batch_labels = serial.predict(&data.as_view()).unwrap();
    let batch_probabilities = serial.predict_proba(&data.as_view()).unwrap();
    for (index, row) in data.as_view().iter_rows().enumerate() {
        assert_eq!(serial.predict_one(row).unwrap(), batch_labels[index]);
        assert_eq!(
            serial.predict_proba_one(row).unwrap(),
            batch_probabilities[index * 2..index * 2 + 2]
        );
    }
}

#[test]
fn object_safe_traits_and_owned_enums_dispatch_by_batch() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let concrete_classifier = classifier(&data, vec![0, 0, 1, 1], 2);
    let expected_labels = concrete_classifier.predict(&data.as_view()).unwrap();
    let expected_probabilities = concrete_classifier.predict_proba(&data.as_view()).unwrap();
    let classifier: AnyClassifier = concrete_classifier.into();
    let erased_classifier: &dyn Classifier = &classifier;

    assert_eq!(
        erased_classifier.predict(&data.as_view()).unwrap(),
        expected_labels
    );
    assert_eq!(
        erased_classifier.predict_proba(&data.as_view()).unwrap(),
        expected_probabilities
    );
    let mut labels_into = [9_u8; 4];
    classifier
        .predict_into(&data.as_view(), &mut labels_into)
        .unwrap();
    assert_eq!(labels_into.as_slice(), expected_labels.as_slice());
    let mut probabilities_into = [9.0_f32; 8];
    classifier
        .predict_proba_into(&data.as_view(), &mut probabilities_into)
        .unwrap();
    assert_eq!(
        probabilities_into.as_slice(),
        expected_probabilities.as_slice()
    );
    let mut class_into = [9.0_f32; 4];
    classifier
        .predict_class_proba_into(&data.as_view(), 1, &mut class_into)
        .unwrap();
    assert_eq!(
        class_into.as_slice(),
        classifier
            .predict_class_proba(&data.as_view(), 1)
            .unwrap()
            .as_slice()
    );
    assert!(matches!(
        classifier.get_params(),
        AnyClassifierParams::RandomForest(_)
    ));
    assert_any_classifier_errors_are_atomic(&classifier, &data);

    let logistic = LogisticRegression::fit(
        &data.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
        LogisticRegressionParams::default(),
    )
    .unwrap();
    let expected = logistic.predict_proba(&data.as_view()).unwrap();
    let logistic: AnyClassifier = logistic.into();
    assert_eq!(logistic.predict_proba(&data.as_view()).unwrap(), expected);
    let mut logistic_into = [9.0_f32; 8];
    logistic
        .predict_proba_into(&data.as_view(), &mut logistic_into)
        .unwrap();
    assert_eq!(logistic_into.as_slice(), expected.as_slice());
    let mut logistic_class = [9.0_f32; 4];
    logistic
        .predict_class_proba_into(&data.as_view(), 1, &mut logistic_class)
        .unwrap();
    assert_eq!(
        logistic_class.as_slice(),
        logistic
            .predict_class_proba(&data.as_view(), 1)
            .unwrap()
            .as_slice()
    );
    assert!(matches!(
        logistic.get_params(),
        AnyClassifierParams::LogisticRegression(_)
    ));
    assert_any_classifier_errors_are_atomic(&logistic, &data);

    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let concrete_regressor = RandomForestRegressor::fit(
        &data.as_view(),
        &targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false),
    )
    .unwrap();
    let expected = concrete_regressor.predict(&data.as_view()).unwrap();
    let regressor: AnyRegressor = concrete_regressor.into();
    let erased_regressor: &dyn Regressor = &regressor;

    assert_eq!(erased_regressor.predict(&data.as_view()).unwrap(), expected);
    let mut regressor_into = [9.0_f32; 4];
    regressor
        .predict_into(&data.as_view(), &mut regressor_into)
        .unwrap();
    assert_eq!(regressor_into.as_slice(), expected.as_slice());
    assert!(matches!(
        regressor.get_params(),
        AnyRegressorParams::RandomForest(_)
    ));
    assert_any_regressor_errors_are_atomic(&regressor, &data);
    assert_eq!(classifier.n_features_in(), regressor.n_features_in());

    let linear =
        LinearRegression::fit(&data.as_view(), &targets, LinearRegressionParams::default())
            .unwrap();
    let expected = linear.predict(&data.as_view()).unwrap();
    let linear: AnyRegressor = linear.into();
    assert_eq!(linear.predict(&data.as_view()).unwrap(), expected);
    let mut linear_into = [9.0_f32; 4];
    linear
        .predict_into(&data.as_view(), &mut linear_into)
        .unwrap();
    assert_eq!(linear_into.as_slice(), expected.as_slice());
    assert!(matches!(
        linear.get_params(),
        AnyRegressorParams::LinearRegression(_)
    ));
    assert_any_regressor_errors_are_atomic(&linear, &data);

    let ridge = Ridge::fit(&data.as_view(), &targets, RidgeParams::default()).unwrap();
    let expected = ridge.predict(&data.as_view()).unwrap();
    let ridge: AnyRegressor = ridge.into();
    assert_eq!(ridge.predict(&data.as_view()).unwrap(), expected);
    let mut ridge_into = [9.0_f32; 4];
    ridge
        .predict_into(&data.as_view(), &mut ridge_into)
        .unwrap();
    assert_eq!(ridge_into.as_slice(), expected.as_slice());
    assert!(matches!(ridge.get_params(), AnyRegressorParams::Ridge(_)));
    assert_any_regressor_errors_are_atomic(&ridge, &data);

    let boosted = HistGradientBoostingRegressor::fit(
        &data.as_view(),
        &targets,
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(3)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1),
    )
    .unwrap();
    let expected = boosted.predict(&data.as_view()).unwrap();
    let boosted: AnyRegressor = boosted.into();
    assert_eq!(boosted.predict(&data.as_view()).unwrap(), expected);
    let mut boosted_into = [9.0_f32; 4];
    boosted
        .predict_into(&data.as_view(), &mut boosted_into)
        .unwrap();
    assert_eq!(boosted_into.as_slice(), expected.as_slice());
    assert!(matches!(
        boosted.get_params(),
        AnyRegressorParams::HistGradientBoosting(_)
    ));
    assert_any_regressor_errors_are_atomic(&boosted, &data);
}

#[test]
fn regressor_scalar_batch_and_output_validation_agree() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let model = RandomForestRegressor::fit(
        &data.as_view(),
        &targets,
        RandomForestRegressorParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false),
    )
    .unwrap();

    let allocating = model.predict(&data.as_view()).unwrap();
    let mut into = vec![0.0; data.rows()];
    model.predict_into(&data.as_view(), &mut into).unwrap();
    assert_eq!(allocating, into);
    for (row, &batch_value) in data.as_view().iter_rows().zip(&allocating) {
        assert_eq!(model.predict_one(row).unwrap(), batch_value);
    }

    let mut too_short = [123.0_f32; 3];
    assert_eq!(
        model
            .predict_into(&data.as_view(), &mut too_short)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 4,
            actual: 3
        }
    );
    assert_eq!(too_short, [123.0; 3]);
}

#[test]
fn pairwise_scores_are_raw_antisymmetric_and_batch_validation_is_atomic() {
    let items = matrix(&[0.0, 0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 2.0], 4, 2);
    let observations = [
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
        PairwiseObservation::new(
            PairIndex::new(1, 0).unwrap(),
            PairOutcome::LeftPreferred,
            1.0,
        )
        .unwrap(),
    ];
    let model = PairwiseLinearRanker::fit(
        &items.as_view(),
        &observations,
        PairwiseLinearRankerParams::default().with_c(4.0),
    )
    .unwrap();
    let forward = model
        .pair_margin(&items.as_view(), PairIndex::new(3, 0).unwrap())
        .unwrap();
    let reverse = model
        .pair_margin(&items.as_view(), PairIndex::new(0, 3).unwrap())
        .unwrap();
    assert_eq!(forward.to_bits(), (-reverse).to_bits());
    assert!(forward > 1.0);

    let pairs = [PairIndex::new(3, 0).unwrap(), PairIndex::new(0, 7).unwrap()];
    let mut output = [99.0; 2];
    assert_eq!(
        model.pair_margins_into(&items.as_view(), &pairs, &mut output),
        Err(PairwiseError::PairIndexOutOfBounds {
            pair: 1,
            item: 7,
            items: 4,
        })
    );
    assert_eq!(output, [99.0; 2]);
}

#[test]
fn histogram_boosting_scalar_batch_and_output_validation_agree() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let targets = RegressionTargets::new(vec![0.0, 0.0, 4.0, 4.0]).unwrap();
    let model = HistGradientBoostingRegressor::fit(
        &data.as_view(),
        &targets,
        HistGradientBoostingRegressorParams::default()
            .with_learning_rate(1.0)
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1),
    )
    .unwrap();
    let allocating = model.predict(&data.as_view()).unwrap();
    let mut output = vec![0.0; data.rows()];
    model.predict_into(&data.as_view(), &mut output).unwrap();
    assert_eq!(allocating, output);
    for (row, &expected) in data.iter_rows().zip(&output) {
        assert_eq!(model.predict_one(row).unwrap(), expected);
    }
    let mut untouched = [88.0; 3];
    assert_eq!(
        model.predict_into(&data.as_view(), &mut untouched),
        Err(ModelError::OutputLength {
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(untouched, [88.0; 3]);
}

#[test]
fn scalar_prediction_rejects_non_finite_features_and_outputs() {
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let regression = RegressionTargets::new(vec![0.0, 2.0, 4.0, 6.0]).unwrap();
    let linear = LinearRegression::fit(
        &data.as_view(),
        &regression,
        LinearRegressionParams::default().with_fit_intercept(false),
    )
    .unwrap();
    let ridge = Ridge::fit(
        &data.as_view(),
        &regression,
        RidgeParams::default()
            .with_alpha(0.0)
            .with_fit_intercept(false),
    )
    .unwrap();
    assert_eq!(
        linear.predict_one(&[f32::NAN]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        ridge.predict_one(&[f32::INFINITY]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        linear.predict_one(&[f32::MAX]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    assert_eq!(
        ridge.predict_one(&[f32::MAX]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    let logistic = LogisticRegression::fit(
        &data.as_view(),
        &BinaryTargets::new(vec![0, 0, 1, 1]).unwrap(),
        LogisticRegressionParams::default().with_c(100.0),
    )
    .unwrap();
    assert_eq!(
        logistic.decision_function_one(&[f32::NEG_INFINITY]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        logistic.predict_positive_proba(&[f32::NAN]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        logistic.predict_one(&[f32::NAN]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
    assert_eq!(
        logistic.decision_function_one(&[f32::MAX]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    let items = matrix(&[0.0, 0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 2.0], 4, 2);
    let observations = [
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
    ];
    let ranker = PairwiseLinearRanker::fit(
        &items.as_view(),
        &observations,
        PairwiseLinearRankerParams::default().with_c(4.0),
    )
    .unwrap();
    assert_eq!(
        ranker.score_one(&[f32::NAN, 0.0]),
        Err(PairwiseError::Model(ModelError::NonFiniteFeature {
            row: 0,
            column: 0,
        }))
    );

    let boosted = HistGradientBoostingRegressor::fit(
        &data.as_view(),
        &RegressionTargets::new(vec![0.0, 0.0, 4.0, 4.0]).unwrap(),
        HistGradientBoostingRegressorParams::default()
            .with_max_iter(1)
            .with_max_leaf_nodes(2)
            .with_min_samples_leaf(1),
    )
    .unwrap();
    assert_eq!(
        boosted.predict_one(&[f32::NAN]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 0 })
    );
}

#[test]
fn every_regressor_reports_a_non_finite_prediction_instead_of_returning_it() {
    // Averaging extreme leaf values overflows the f32 accumulator, which is
    // the forest's route to a non-finite prediction from finite inputs. Every
    // regressor must surface that as an error rather than an infinity.
    let data = matrix(&[0.0, 1.0, 2.0, 3.0], 4, 1);
    let extreme = RegressionTargets::new(vec![f32::MAX; 4]).unwrap();
    let forest = RandomForestRegressor::fit(
        &data.as_view(),
        &extreme,
        RandomForestRegressorParams::default()
            .with_n_estimators(4)
            .with_bootstrap(false),
    )
    .unwrap();
    assert_eq!(
        forest.predict_one(&[1.0]),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    assert_eq!(
        forest.predict(&data.as_view()),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    let mut output = [7.0; 4];
    assert_eq!(
        forest.predict_into(&data.as_view(), &mut output),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );
    assert_eq!(
        Regressor::predict(&forest, &data.as_view()),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    let erased: AnyRegressor = forest.into();
    assert_eq!(
        erased.predict(&data.as_view()),
        Err(ModelError::NonFinitePrediction { row: 0 })
    );

    // A finitely-predicting forest keeps returning values, so the new check
    // rejects only the overflowing case.
    let ordinary = RandomForestRegressor::fit(
        &data.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap(),
        RandomForestRegressorParams::default()
            .with_n_estimators(4)
            .with_bootstrap(false),
    )
    .unwrap();
    assert!(
        ordinary
            .predict(&data.as_view())
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}
