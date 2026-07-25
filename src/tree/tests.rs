use super::*;
use crate::api::ModelError;
use crate::artifact::{
    ArtifactError, DECISION_TREE_CLASSIFIER_ARTIFACT_KIND, DECISION_TREE_REGRESSOR_ARTIFACT_KIND,
};
use crate::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};

const SCHEMA: [u8; 32] = [7; 32];
const OTHER_SCHEMA: [u8; 32] = [9; 32];

fn matrix(rows: &[&[f32]]) -> DenseMatrix {
    let columns = rows.first().map_or(0, |row| row.len());
    assert!(rows.iter().all(|row| row.len() == columns));
    let values = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DenseMatrix::new(values, rows.len(), columns).unwrap()
}

/// A small two-column problem with a unique best split at every node, so the
/// tests below assert structure rather than a tie-break.
fn separable() -> (DenseMatrix, Vec<f32>, Vec<u8>) {
    let x = matrix(&[
        &[-3.0, 0.5],
        &[-2.0, 1.5],
        &[-1.0, 0.25],
        &[1.0, 2.5],
        &[2.0, 0.75],
        &[3.0, 3.5],
    ]);
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let labels = vec![0, 0, 0, 1, 1, 1];
    (x, y, labels)
}

fn classifier_params() -> DecisionTreeClassifierParams {
    DecisionTreeClassifierParams::default().with_random_state(11)
}

fn regressor_params() -> DecisionTreeRegressorParams {
    DecisionTreeRegressorParams::default().with_random_state(11)
}

#[test]
fn a_classifier_separates_labels_and_keeps_probabilities_in_range() {
    let (x, _, labels) = separable();
    let targets = BinaryTargets::new(labels.clone()).unwrap();
    let model = DecisionTreeClassifier::fit(&x.as_view(), &targets, classifier_params()).unwrap();

    assert_eq!(model.classes(), &[0, 1]);
    assert_eq!(model.n_features_in(), 2);
    assert_eq!(model.predict(&x.as_view()).unwrap(), labels);
    for probabilities in model.predict_proba(&x.as_view()).unwrap().chunks_exact(2) {
        assert!(
            probabilities
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        assert!((probabilities[0] + probabilities[1] - 1.0).abs() <= f32::EPSILON);
    }
}

#[test]
fn a_regressor_reproduces_a_step_function_it_can_fully_grow() {
    let (x, y, _) = separable();
    let targets = RegressionTargets::new(y.clone()).unwrap();
    let model = DecisionTreeRegressor::fit(&x.as_view(), &targets, regressor_params()).unwrap();
    assert_eq!(model.predict(&x.as_view()).unwrap(), y);
}

#[test]
fn identical_inputs_parameters_and_seed_give_an_identical_model() {
    let (x, y, labels) = separable();
    let targets = BinaryTargets::new(labels).unwrap();
    let first = DecisionTreeClassifier::fit(&x.as_view(), &targets, classifier_params()).unwrap();
    let second = DecisionTreeClassifier::fit(&x.as_view(), &targets, classifier_params()).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.to_artifact(SCHEMA).unwrap(),
        second.to_artifact(SCHEMA).unwrap()
    );

    let regression = RegressionTargets::new(y).unwrap();
    let first = DecisionTreeRegressor::fit(&x.as_view(), &regression, regressor_params()).unwrap();
    let second = DecisionTreeRegressor::fit(&x.as_view(), &regression, regressor_params()).unwrap();
    assert_eq!(first, second);
}

#[test]
fn every_into_method_agrees_with_its_allocating_twin() {
    let (x, y, labels) = separable();
    let view = x.as_view();

    let targets = BinaryTargets::new(labels).unwrap();
    let model = DecisionTreeClassifier::fit(&view, &targets, classifier_params()).unwrap();
    let mut into_labels = vec![0_u8; view.rows()];
    model.predict_into(&view, &mut into_labels).unwrap();
    assert_eq!(into_labels, model.predict(&view).unwrap());

    let mut into_proba = vec![0.0_f32; view.rows() * model.classes().len()];
    model.predict_proba_into(&view, &mut into_proba).unwrap();
    assert_eq!(into_proba, model.predict_proba(&view).unwrap());

    let mut into_column = vec![0.0_f32; view.rows()];
    model
        .predict_class_proba_into(&view, 1, &mut into_column)
        .unwrap();
    assert_eq!(into_column, model.predict_class_proba(&view, 1).unwrap());

    // The per-row entry points must agree with the batch ones too, or a caller
    // scoring one sample would get a different answer from the same model.
    for (index, row) in view.iter_rows().enumerate() {
        assert_eq!(model.predict_one(row).unwrap(), into_labels[index]);
        assert_eq!(
            model.predict_proba_one(row).unwrap(),
            into_proba[index * 2..index * 2 + 2]
        );
        assert_eq!(
            model.predict_class_proba_one(row, 1).unwrap(),
            into_column[index]
        );
    }

    let regression = RegressionTargets::new(y).unwrap();
    let model = DecisionTreeRegressor::fit(&view, &regression, regressor_params()).unwrap();
    let mut into_values = vec![0.0_f32; view.rows()];
    model.predict_into(&view, &mut into_values).unwrap();
    assert_eq!(into_values, model.predict(&view).unwrap());
    for (index, row) in view.iter_rows().enumerate() {
        assert_eq!(model.predict_one(row).unwrap(), into_values[index]);
    }
}

#[test]
fn unit_weights_reproduce_the_unweighted_fit_and_an_integer_weight_repeats_a_row() {
    let (x, y, labels) = separable();
    let view = x.as_view();
    let targets = BinaryTargets::new(labels.clone()).unwrap();
    let unit = SampleWeights::new(vec![1.0; view.rows()]).unwrap();
    assert_eq!(
        DecisionTreeClassifier::fit_weighted(&view, &targets, &unit, classifier_params()).unwrap(),
        DecisionTreeClassifier::fit(&view, &targets, classifier_params()).unwrap()
    );

    // A weight of three is the same fit as three copies of the row. This holds
    // unconditionally only because the node-size bounds count summed weight
    // rather than rows — the recorded divergence from the reference.
    let params = regressor_params().with_min_samples_leaf(2);
    let mut weights = vec![1.0_f32; view.rows()];
    weights[0] = 3.0;
    let weighted = DecisionTreeRegressor::fit_weighted(
        &view,
        &RegressionTargets::new(y.clone()).unwrap(),
        &SampleWeights::new(weights).unwrap(),
        params.clone(),
    )
    .unwrap();

    let mut repeated_rows: Vec<Vec<f32>> = Vec::new();
    let mut repeated_targets = Vec::new();
    for (index, row) in view.iter_rows().enumerate() {
        let copies = if index == 0 { 3 } else { 1 };
        for _ in 0..copies {
            repeated_rows.push(row.to_vec());
            repeated_targets.push(y[index]);
        }
    }
    let borrowed: Vec<&[f32]> = repeated_rows.iter().map(Vec::as_slice).collect();
    let repeated = matrix(&borrowed);
    let expanded = DecisionTreeRegressor::fit(
        &repeated.as_view(),
        &RegressionTargets::new(repeated_targets).unwrap(),
        params,
    )
    .unwrap();
    assert_eq!(
        weighted.predict(&view).unwrap(),
        expanded.predict(&view).unwrap()
    );
}

#[test]
fn a_zero_weight_row_is_absent_rather_than_present_with_no_influence() {
    let (x, y, _) = separable();
    let view = x.as_view();
    let mut weights = vec![1.0_f32; view.rows()];
    weights[5] = 0.0;
    let with_zero = DecisionTreeRegressor::fit_weighted(
        &view,
        &RegressionTargets::new(y.clone()).unwrap(),
        &SampleWeights::new(weights).unwrap(),
        regressor_params(),
    )
    .unwrap();

    let kept: Vec<&[f32]> = (0..5).map(|row| view.row(row).unwrap()).collect();
    let dropped = matrix(&kept);
    let without = DecisionTreeRegressor::fit(
        &dropped.as_view(),
        &RegressionTargets::new(y[..5].to_vec()).unwrap(),
        regressor_params(),
    )
    .unwrap();
    assert_eq!(
        with_zero.predict(&dropped.as_view()).unwrap(),
        without.predict(&dropped.as_view()).unwrap()
    );
}

#[test]
fn artifacts_round_trip_through_every_fitted_shape() {
    let (x, y, labels) = separable();
    let view = x.as_view();

    let binary = DecisionTreeClassifier::fit(
        &view,
        &BinaryTargets::new(labels).unwrap(),
        classifier_params(),
    )
    .unwrap();
    let multiclass = DecisionTreeClassifier::fit_multiclass(
        &view,
        &ClassTargets::new(vec![3, 3, 7, 7, 10, 10]).unwrap(),
        classifier_params(),
    )
    .unwrap();
    let single_class = DecisionTreeClassifier::fit(
        &view,
        &BinaryTargets::new(vec![1; view.rows()]).unwrap(),
        classifier_params(),
    )
    .unwrap();
    // `max_depth = 1` on constant targets leaves the root a leaf, which is the
    // degenerate topology the codec has to survive.
    let root_leaf = DecisionTreeRegressor::fit(
        &view,
        &RegressionTargets::new(vec![2.5; view.rows()]).unwrap(),
        regressor_params(),
    )
    .unwrap();
    let regressor = DecisionTreeRegressor::fit(
        &view,
        &RegressionTargets::new(y).unwrap(),
        regressor_params(),
    )
    .unwrap();

    for model in [&binary, &multiclass, &single_class] {
        let bytes = model.to_artifact(SCHEMA).unwrap();
        let restored = DecisionTreeClassifier::from_artifact(&bytes, SCHEMA).unwrap();
        assert_eq!(&restored, model);
        assert_eq!(restored.get_params(), model.get_params());
        assert_eq!(
            restored.predict_proba(&view).unwrap(),
            model.predict_proba(&view).unwrap()
        );
        // Canonicity: one model has exactly one encoding, so re-encoding what
        // was decoded must reproduce the input byte for byte.
        assert_eq!(restored.to_artifact(SCHEMA).unwrap(), bytes);
    }
    for model in [&root_leaf, &regressor] {
        let bytes = model.to_artifact(SCHEMA).unwrap();
        let restored = DecisionTreeRegressor::from_artifact(&bytes, SCHEMA).unwrap();
        assert_eq!(&restored, model);
        assert_eq!(restored.to_artifact(SCHEMA).unwrap(), bytes);
    }
    assert_eq!(multiclass.classes(), &[3, 7, 10]);
    assert_eq!(single_class.classes(), &[1]);
    assert_eq!(
        single_class.predict_proba_one(&[0.0, 0.0]).unwrap(),
        vec![1.0]
    );
}

#[test]
fn a_decoder_refuses_another_estimators_bytes_and_another_schema() {
    let (x, y, _) = separable();
    let view = x.as_view();
    let model = DecisionTreeRegressor::fit(
        &view,
        &RegressionTargets::new(y).unwrap(),
        regressor_params(),
    )
    .unwrap();
    let bytes = model.to_artifact(SCHEMA).unwrap();

    assert_eq!(
        DecisionTreeRegressor::from_artifact(&bytes, OTHER_SCHEMA),
        Err(ArtifactError::FeatureSchemaMismatch)
    );
    assert_eq!(
        DecisionTreeClassifier::from_artifact(&bytes, SCHEMA),
        Err(ArtifactError::UnsupportedModelKind {
            found: DECISION_TREE_REGRESSOR_ARTIFACT_KIND
        })
    );

    let mut truncated = bytes.clone();
    truncated.truncate(bytes.len() - 1);
    assert!(DecisionTreeRegressor::from_artifact(&truncated, SCHEMA).is_err());

    let mut corrupted = bytes.clone();
    let last = corrupted.len() - 40;
    corrupted[last] ^= 0xff;
    assert_eq!(
        DecisionTreeRegressor::from_artifact(&corrupted, SCHEMA),
        Err(ArtifactError::ChecksumMismatch)
    );
    assert_eq!(
        DECISION_TREE_CLASSIFIER_ARTIFACT_KIND,
        DECISION_TREE_REGRESSOR_ARTIFACT_KIND + 1
    );
}

#[test]
fn invalid_shapes_and_parameters_fail_before_any_training_work() {
    let (x, y, labels) = separable();
    let view = x.as_view();
    let targets = RegressionTargets::new(y).unwrap();

    let cases: [(DecisionTreeRegressorParams, ModelError); 4] = [
        (
            regressor_params().with_max_depth(Some(0)),
            ModelError::InvalidMaxDepth,
        ),
        (
            regressor_params().with_min_samples_split(1),
            ModelError::InvalidMinSamplesSplit,
        ),
        (
            regressor_params().with_min_samples_leaf(0),
            ModelError::InvalidMinSamplesLeaf,
        ),
        (
            regressor_params().with_max_features(MaxFeatures::Count(3)),
            ModelError::InvalidMaxFeatures {
                requested: 3,
                available: 2,
            },
        ),
    ];
    for (params, expected) in cases {
        assert_eq!(
            DecisionTreeRegressor::fit(&view, &targets, params),
            Err(expected)
        );
    }

    assert_eq!(
        DecisionTreeRegressor::fit(
            &view,
            &RegressionTargets::new(vec![0.0; 3]).unwrap(),
            regressor_params()
        ),
        Err(ModelError::TargetLength {
            rows: 6,
            targets: 3
        })
    );
    assert_eq!(
        DecisionTreeClassifier::fit_weighted(
            &view,
            &BinaryTargets::new(labels).unwrap(),
            &SampleWeights::new(vec![1.0; 3]).unwrap(),
            classifier_params()
        ),
        Err(ModelError::SampleWeightLength {
            rows: 6,
            weights: 3
        })
    );
}

#[test]
fn a_multiclass_fit_has_no_positive_class_to_report() {
    let (x, _, _) = separable();
    let model = DecisionTreeClassifier::fit_multiclass(
        &x.as_view(),
        &ClassTargets::new(vec![3, 3, 7, 7, 10, 10]).unwrap(),
        classifier_params(),
    )
    .unwrap();
    assert_eq!(
        model.predict_positive_proba(&[0.0, 0.0]),
        Err(ModelError::MulticlassOutput { columns: 3 })
    );
    assert_eq!(
        model.predict_class_proba_one(&[0.0, 0.0], 4),
        Err(ModelError::UnknownClass { class: 4 })
    );
}

#[test]
fn prediction_rejects_a_row_of_the_wrong_width_before_traversing() {
    let (x, y, _) = separable();
    let model = DecisionTreeRegressor::fit(
        &x.as_view(),
        &RegressionTargets::new(y).unwrap(),
        regressor_params(),
    )
    .unwrap();
    assert_eq!(
        model.predict_one(&[0.0]),
        Err(ModelError::FeatureDimension {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        model.predict_one(&[0.0, f32::NAN]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 1 })
    );
    let mut output = vec![0.0; 2];
    assert_eq!(
        model.predict_into(&x.as_view(), &mut output),
        Err(ModelError::OutputLength {
            expected: 6,
            actual: 2
        })
    );
}
