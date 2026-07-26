use super::*;
use crate::api::ModelError;
use crate::artifact::{
    ArtifactError, EXTRA_TREES_CLASSIFIER_ARTIFACT_KIND, EXTRA_TREES_REGRESSOR_ARTIFACT_KIND,
    ModelArtifact,
};
use crate::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};
use crate::ensemble::{
    MaxFeatures, NJobs, RandomForestClassifier, RandomForestClassifierParams,
    RandomForestRegressor, RandomForestRegressorParams,
};

const SCHEMA: [u8; 32] = [7; 32];
const OTHER_SCHEMA: [u8; 32] = [9; 32];

fn matrix(rows: &[&[f32]]) -> DenseMatrix {
    let columns = rows.first().map_or(0, |row| row.len());
    assert!(rows.iter().all(|row| row.len() == columns));
    let values = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DenseMatrix::new(values, rows.len(), columns).unwrap()
}

/// A separable four-column problem with enough rows for a randomized threshold
/// to have room to land somewhere useful.
fn sample() -> (DenseMatrix, Vec<f32>, Vec<u8>) {
    let mut values = Vec::new();
    let mut targets = Vec::new();
    let mut labels = Vec::new();
    for row in 0..48_usize {
        let base = (row as f32) / 8.0 - 3.0;
        values.extend_from_slice(&[base, base * 0.5 + 1.0, -base, (row % 5) as f32]);
        targets.push(base * 2.0);
        labels.push(u8::from(base > 0.0));
    }
    (DenseMatrix::new(values, 48, 4).unwrap(), targets, labels)
}

fn classifier_params() -> ExtraTreesClassifierParams {
    ExtraTreesClassifierParams::default()
        .with_n_estimators(7)
        .with_max_features(MaxFeatures::All)
        .with_random_state(11)
}

fn regressor_params() -> ExtraTreesRegressorParams {
    ExtraTreesRegressorParams::default()
        .with_n_estimators(7)
        .with_max_features(MaxFeatures::All)
        .with_random_state(11)
}

/// The reference's defaults, restated as a test because they are the one place
/// this family deliberately disagrees with the random forest it shares a
/// parameter vocabulary with.
#[test]
fn the_defaults_are_the_randomized_ensembles_own() {
    let classifier = ExtraTreesClassifierParams::default();
    let regressor = ExtraTreesRegressorParams::default();
    assert_eq!(classifier.max_features(), MaxFeatures::Sqrt);
    assert_eq!(regressor.max_features(), MaxFeatures::All);
    // Trees decorrelate through their thresholds here, so resampling on top of
    // that would only remove training rows.
    assert!(!classifier.bootstrap());
    assert!(!regressor.bootstrap());
    assert!(RandomForestClassifierParams::default().bootstrap());
    assert!(RandomForestRegressorParams::default().bootstrap());
    assert_eq!(classifier.n_estimators(), 100);
    assert_eq!(classifier.min_samples_split(), 2);
    assert_eq!(classifier.min_samples_leaf(), 1);
    assert_eq!(classifier.max_depth(), None);
    assert_eq!(classifier.n_jobs(), NJobs::Serial);
}

#[test]
fn both_estimators_fit_predict_and_stay_reproducible() {
    let (x, y, labels) = sample();
    let view = x.as_view();

    let targets = BinaryTargets::new(labels.clone()).unwrap();
    let first = ExtraTreesClassifier::fit(&view, &targets, classifier_params()).unwrap();
    let second = ExtraTreesClassifier::fit(&view, &targets, classifier_params()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.classes(), &[0, 1]);
    assert_eq!(first.n_features_in(), 4);
    assert_eq!(first.predict(&view).unwrap(), labels);
    for probabilities in first.predict_proba(&view).unwrap().chunks_exact(2) {
        assert!(probabilities.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    let values = RegressionTargets::new(y).unwrap();
    let first = ExtraTreesRegressor::fit(&view, &values, regressor_params()).unwrap();
    let second = ExtraTreesRegressor::fit(&view, &values, regressor_params()).unwrap();
    assert_eq!(first, second);
    assert!(first.predict(&view).unwrap().iter().all(|v| v.is_finite()));
}

/// Randomized thresholds are the whole difference, so an extra-trees fit must
/// not coincide with the random forest of the same shape.
///
/// Bootstrapping is disabled on both sides here, so the only remaining
/// difference is the split search — which makes this a test of that and not of
/// the resampling default.
#[test]
fn a_randomized_ensemble_is_not_the_random_forest_of_the_same_shape() {
    let (x, y, labels) = sample();
    let view = x.as_view();
    let targets = BinaryTargets::new(labels).unwrap();
    let forest_params = RandomForestClassifierParams::default()
        .with_n_estimators(7)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All)
        .with_random_state(11);
    let randomized = ExtraTreesClassifier::fit(&view, &targets, classifier_params()).unwrap();
    let forest = RandomForestClassifier::fit(&view, &targets, forest_params).unwrap();
    assert_ne!(
        randomized.to_artifact(SCHEMA).unwrap(),
        forest.to_artifact(SCHEMA).unwrap()
    );

    let values = RegressionTargets::new(y).unwrap();
    let randomized = ExtraTreesRegressor::fit(&view, &values, regressor_params()).unwrap();
    let forest = RandomForestRegressor::fit(
        &view,
        &values,
        RandomForestRegressorParams::default()
            .with_n_estimators(7)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All)
            .with_random_state(11),
    )
    .unwrap();
    assert_ne!(
        randomized.to_artifact(SCHEMA).unwrap(),
        forest.to_artifact(SCHEMA).unwrap()
    );
}

#[test]
fn every_into_method_agrees_with_its_allocating_twin() {
    let (x, y, labels) = sample();
    let view = x.as_view();

    let targets = BinaryTargets::new(labels).unwrap();
    let model = ExtraTreesClassifier::fit(&view, &targets, classifier_params()).unwrap();
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

    let mut positive = vec![0.0_f32; view.rows()];
    model
        .predict_positive_proba_into(&view, &mut positive)
        .unwrap();
    for (index, row) in view.iter_rows().enumerate() {
        assert_eq!(model.predict_one(row).unwrap(), into_labels[index]);
        assert_eq!(
            model.predict_positive_proba_one(row).unwrap(),
            positive[index]
        );
        assert_eq!(
            model.predict_proba_one(row).unwrap(),
            into_proba[index * 2..index * 2 + 2]
        );
    }

    let regression = RegressionTargets::new(y).unwrap();
    let model = ExtraTreesRegressor::fit(&view, &regression, regressor_params()).unwrap();
    let mut into_values = vec![0.0_f32; view.rows()];
    model.predict_into(&view, &mut into_values).unwrap();
    assert_eq!(into_values, model.predict(&view).unwrap());
    for (index, row) in view.iter_rows().enumerate() {
        assert_eq!(model.predict_one(row).unwrap(), into_values[index]);
    }
}

#[test]
fn unit_weights_reproduce_the_unweighted_fit() {
    let (x, y, labels) = sample();
    let view = x.as_view();
    let unit = SampleWeights::new(vec![1.0; view.rows()]).unwrap();
    let targets = BinaryTargets::new(labels).unwrap();
    assert_eq!(
        ExtraTreesClassifier::fit_weighted(&view, &targets, &unit, classifier_params()).unwrap(),
        ExtraTreesClassifier::fit(&view, &targets, classifier_params()).unwrap()
    );
    let regression = RegressionTargets::new(y).unwrap();
    assert_eq!(
        ExtraTreesRegressor::fit_weighted(&view, &regression, &unit, regressor_params()).unwrap(),
        ExtraTreesRegressor::fit(&view, &regression, regressor_params()).unwrap()
    );
}

#[test]
fn artifacts_round_trip_through_every_fitted_shape() {
    let (x, y, labels) = sample();
    let view = x.as_view();

    let binary = ExtraTreesClassifier::fit(
        &view,
        &BinaryTargets::new(labels).unwrap(),
        classifier_params(),
    )
    .unwrap();
    let classes =
        ClassTargets::new((0..view.rows()).map(|row| [3_u8, 7, 10][row % 3]).collect()).unwrap();
    let multiclass =
        ExtraTreesClassifier::fit_multiclass(&view, &classes, classifier_params()).unwrap();
    for model in [&binary, &multiclass] {
        let bytes = model.to_artifact(SCHEMA).unwrap();
        let restored = ExtraTreesClassifier::from_artifact(&bytes, SCHEMA).unwrap();
        assert_eq!(&restored, model);
        assert_eq!(restored.get_params(), model.get_params());
        // Canonicity: one model has exactly one encoding.
        assert_eq!(restored.to_artifact(SCHEMA).unwrap(), bytes);
    }
    assert_eq!(multiclass.classes(), &[3, 7, 10]);

    let regressor = ExtraTreesRegressor::fit(
        &view,
        &RegressionTargets::new(y).unwrap(),
        regressor_params(),
    )
    .unwrap();
    let bytes = regressor.to_artifact(SCHEMA).unwrap();
    let restored = ExtraTreesRegressor::from_artifact(&bytes, SCHEMA).unwrap();
    assert_eq!(restored, regressor);
    assert_eq!(restored.to_artifact(SCHEMA).unwrap(), bytes);
}

/// Sharing one codec must not let one ensemble read another's bytes.
///
/// The kinds are what separate them, and this is the test that would fail if a
/// facade were expanded with the wrong constant — a mistake the macro makes
/// cheap to make and this makes impossible to ship.
#[test]
fn a_decoder_refuses_another_ensembles_bytes_and_another_schema() {
    let (x, y, labels) = sample();
    let view = x.as_view();
    let regressor = ExtraTreesRegressor::fit(
        &view,
        &RegressionTargets::new(y).unwrap(),
        regressor_params(),
    )
    .unwrap();
    let bytes = regressor.to_artifact(SCHEMA).unwrap();

    assert_eq!(
        ExtraTreesRegressor::from_artifact(&bytes, OTHER_SCHEMA),
        Err(ArtifactError::FeatureSchemaMismatch)
    );
    assert_eq!(
        ExtraTreesClassifier::from_artifact(&bytes, SCHEMA),
        Err(ArtifactError::UnsupportedModelKind {
            found: EXTRA_TREES_REGRESSOR_ARTIFACT_KIND,
        })
    );
    assert_eq!(
        RandomForestRegressor::from_artifact(&bytes, SCHEMA),
        Err(ArtifactError::UnsupportedModelKind {
            found: EXTRA_TREES_REGRESSOR_ARTIFACT_KIND,
        })
    );

    let classifier = ExtraTreesClassifier::fit(
        &view,
        &BinaryTargets::new(labels).unwrap(),
        classifier_params(),
    )
    .unwrap();
    let bytes = classifier.to_artifact(SCHEMA).unwrap();
    assert_eq!(
        RandomForestClassifier::from_artifact(&bytes, SCHEMA),
        Err(ArtifactError::UnsupportedModelKind {
            found: EXTRA_TREES_CLASSIFIER_ARTIFACT_KIND,
        })
    );
}

#[test]
fn invalid_shapes_and_parameters_fail_before_any_training_work() {
    let (x, y, labels) = sample();
    let view = x.as_view();
    let regression = RegressionTargets::new(y).unwrap();
    let targets = BinaryTargets::new(labels).unwrap();

    assert_eq!(
        ExtraTreesRegressor::fit(&view, &regression, regressor_params().with_n_estimators(0)),
        Err(ModelError::InvalidEstimatorCount)
    );
    assert_eq!(
        ExtraTreesRegressor::fit(
            &view,
            &regression,
            regressor_params().with_max_depth(Some(0))
        ),
        Err(ModelError::InvalidMaxDepth)
    );
    assert_eq!(
        ExtraTreesClassifier::fit(
            &view,
            &targets,
            classifier_params().with_min_samples_split(1)
        ),
        Err(ModelError::InvalidMinSamplesSplit)
    );
    assert_eq!(
        ExtraTreesClassifier::fit(
            &view,
            &targets,
            classifier_params().with_min_samples_leaf(0)
        ),
        Err(ModelError::InvalidMinSamplesLeaf)
    );
    assert_eq!(
        ExtraTreesClassifier::fit(
            &view,
            &targets,
            classifier_params().with_max_features(MaxFeatures::Count(9))
        ),
        Err(ModelError::InvalidMaxFeatures {
            requested: 9,
            available: 4,
        })
    );
    assert_eq!(
        ExtraTreesClassifier::fit(
            &view,
            &targets,
            classifier_params().with_n_jobs(NJobs::Count(0))
        ),
        Err(ModelError::InvalidJobCount)
    );

    let short = matrix(&[&[0.0, 1.0, 2.0, 3.0]]);
    assert_eq!(
        ExtraTreesRegressor::fit(&short.as_view(), &regression, regressor_params()),
        Err(ModelError::TargetLength {
            rows: 1,
            targets: 48,
        })
    );
}

#[test]
fn a_multiclass_fit_has_no_positive_class_to_report() {
    let (x, _, _) = sample();
    let view = x.as_view();
    let classes =
        ClassTargets::new((0..view.rows()).map(|row| [3_u8, 7, 10][row % 3]).collect()).unwrap();
    let model = ExtraTreesClassifier::fit_multiclass(&view, &classes, classifier_params()).unwrap();
    assert_eq!(
        model.predict_positive_proba_one(view.row(0).unwrap()),
        Err(ModelError::MulticlassOutput { columns: 3 })
    );
}

/// A serial fit and a parallel fit are the same ensemble.
///
/// Randomized thresholds do not weaken this: every member's generator is
/// derived from its index alone, so thread count still cannot reach a fitted
/// value.
#[test]
fn parallel_and_serial_fits_are_identical() {
    let (x, y, _) = sample();
    let view = x.as_view();
    let regression = RegressionTargets::new(y).unwrap();
    let serial = ExtraTreesRegressor::fit(
        &view,
        &regression,
        regressor_params().with_n_jobs(NJobs::Serial),
    )
    .unwrap();
    let parallel = ExtraTreesRegressor::fit(
        &view,
        &regression,
        regressor_params().with_n_jobs(NJobs::Count(4)),
    )
    .unwrap();
    assert_eq!(serial.core.trees, parallel.core.trees);
}
