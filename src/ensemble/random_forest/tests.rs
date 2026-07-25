use super::super::forest::model::Forest;
use super::*;
use crate::api::ModelError;
use crate::artifact::{
    ArtifactError, RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND, RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
};
use crate::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets, SampleWeights};
use crate::ensemble::HistGradientBoostingRegressor;
use crate::ensemble::{MaxFeatures, NJobs};
use crate::linear_model::Ridge;
use crate::tree::{FEATURE_MASK, LEFT_IS_LEAF, PackedNode, RIGHT_IS_LEAF};
use sha2::{Digest, Sha256};

fn matrix(rows: &[&[f32]]) -> DenseMatrix {
    let cols = rows.first().map_or(0, |row| row.len());
    assert!(rows.iter().all(|row| row.len() == cols));
    let values = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DenseMatrix::new(values, rows.len(), cols).unwrap()
}

fn classifier_params(random_state: u64) -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(31)
        .with_max_depth(Some(8))
        .with_max_features(MaxFeatures::All)
        .with_random_state(random_state)
}

fn regressor_params(random_state: u64) -> RandomForestRegressorParams {
    RandomForestRegressorParams::default()
        .with_n_estimators(31)
        .with_max_depth(Some(8))
        .with_max_features(MaxFeatures::All)
        .with_random_state(random_state)
}

#[test]
fn classifies_separable_data_and_probabilities_are_bounded() {
    let x = matrix(&[
        &[-3.0],
        &[-2.0],
        &[-1.0],
        &[-0.5],
        &[0.5],
        &[1.0],
        &[2.0],
        &[3.0],
    ]);
    let y = BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).unwrap();
    let forest = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(4)).unwrap();
    let mut predictions = vec![0.0; x.rows()];
    forest
        .predict_positive_proba_into(&x.as_view(), &mut predictions)
        .unwrap();
    assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
    assert!(predictions[..4].iter().all(|&p| p < 0.5));
    assert!(predictions[4..].iter().all(|&p| p > 0.5));
}

#[test]
fn nonlinear_forest_learns_repeated_xor() {
    let x = matrix(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ]);
    let y = BinaryTargets::new(vec![0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]).unwrap();
    let cfg = classifier_params(99).with_n_estimators(101);
    let forest = RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap();
    for (row, &expected) in y.as_slice().iter().take(4).enumerate() {
        let p = forest.predict_positive_proba(x.row(row).unwrap()).unwrap();
        assert_eq!(p >= 0.5, expected == 1, "row {row}: {p}");
    }
}

#[test]
fn regresses_piecewise_values() {
    let x = matrix(&[
        &[-3.0],
        &[-2.0],
        &[-1.0],
        &[0.0],
        &[1.0],
        &[2.0],
        &[3.0],
        &[4.0],
    ]);
    let y = RegressionTargets::new(vec![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]).unwrap();
    let cfg = regressor_params(7).with_n_estimators(61);
    let forest = RandomForestRegressor::fit(&x.as_view(), &y, cfg).unwrap();
    let mut output = vec![0.0; x.rows()];
    forest.predict_into(&x.as_view(), &mut output).unwrap();
    let mae = output
        .iter()
        .zip(y.as_slice())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / output.len() as f32;
    assert!(mae < 1.5, "mae={mae}, predictions={output:?}");
}

#[test]
fn reference_defaults_distinguish_classification_and_regression() {
    assert_eq!(
        RandomForestClassifierParams::default().max_features(),
        MaxFeatures::Sqrt
    );
    assert_eq!(
        RandomForestRegressorParams::default().max_features(),
        MaxFeatures::All
    );
}

#[test]
fn exact_classifier_split_and_leaf_probabilities_match_the_oracle() {
    let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let y = BinaryTargets::new(vec![0, 0, 1, 1]).unwrap();
    let cfg = RandomForestClassifierParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All);
    let forest = RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap();
    let tree = &forest.binary_trees()[0];
    assert_eq!(tree.nodes.len(), 1);
    let root = &tree.nodes[0];
    assert_eq!(root.feature_and_flags & FEATURE_MASK, 0);
    assert_eq!(root.threshold, 1.5);
    assert_ne!(root.feature_and_flags & LEFT_IS_LEAF, 0);
    assert_ne!(root.feature_and_flags & RIGHT_IS_LEAF, 0);
    assert_eq!(f32::from_bits(root.left), 0.0);
    assert_eq!(f32::from_bits(root.right), 1.0);

    let leaf_cfg = RandomForestClassifierParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_min_samples_split(5);
    let leaf = RandomForestClassifier::fit(&x.as_view(), &y, leaf_cfg).unwrap();
    assert_eq!(leaf.predict_positive_proba(&[100.0]).unwrap(), 0.5);
}

#[test]
fn exact_regression_leaf_is_the_target_mean() {
    let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let y = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let cfg = RandomForestRegressorParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_min_samples_split(5);
    let forest = RandomForestRegressor::fit(&x.as_view(), &y, cfg).unwrap();
    assert_eq!(forest.predict_one(&[-50.0]).unwrap(), 2.5);

    let stump_cfg = RandomForestRegressorParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_max_depth(Some(1));
    let stump = RandomForestRegressor::fit(&x.as_view(), &y, stump_cfg).unwrap();
    assert_eq!(stump.core.trees[0].nodes[0].threshold, 1.5);
    assert_eq!(stump.predict_one(&[-50.0]).unwrap(), 1.5);
    assert_eq!(stump.predict_one(&[100.0]).unwrap(), 3.5);
}

#[test]
fn model_is_identical_across_repeats_and_thread_counts() {
    let x = matrix(&[
        &[0.0, 3.0],
        &[1.0, 2.0],
        &[2.0, 1.0],
        &[3.0, 0.0],
        &[4.0, 7.0],
        &[5.0, 6.0],
        &[6.0, 5.0],
        &[7.0, 4.0],
    ]);
    let y = BinaryTargets::new(vec![0, 1, 1, 0, 1, 0, 0, 1]).unwrap();
    let one = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(123)).unwrap();
    let repeat = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(123)).unwrap();
    let parallel_config = classifier_params(123).with_n_jobs(crate::ensemble::NJobs::Count(4));
    let parallel = RandomForestClassifier::fit(&x.as_view(), &y, parallel_config).unwrap();
    assert_eq!(one.to_bytes(), repeat.to_bytes());
    assert_eq!(one.to_bytes(), parallel.to_bytes());
}

#[test]
fn packed_classifier_and_regressor_fingerprints_are_frozen() {
    let x = matrix(&[
        &[0.0, 3.0],
        &[1.0, 2.0],
        &[2.0, 1.0],
        &[3.0, 0.0],
        &[4.0, 7.0],
        &[5.0, 6.0],
        &[6.0, 5.0],
        &[7.0, 4.0],
    ]);
    let classifier = RandomForestClassifier::fit(
        &x.as_view(),
        &BinaryTargets::new(vec![0, 1, 1, 0, 1, 0, 0, 1]).unwrap(),
        classifier_params(123),
    )
    .unwrap();
    let regressor = RandomForestRegressor::fit(
        &x.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 1.5, 2.5, 8.0, 7.0, 6.0, 5.0]).unwrap(),
        regressor_params(123),
    )
    .unwrap();
    let regressor_repeat = RandomForestRegressor::fit(
        &x.as_view(),
        &RegressionTargets::new(vec![0.0, 1.0, 1.5, 2.5, 8.0, 7.0, 6.0, 5.0]).unwrap(),
        regressor_params(123),
    )
    .unwrap();
    assert_eq!(regressor.to_bytes(), regressor_repeat.to_bytes());

    for (name, bytes, expected_len, expected_digest) in [
        (
            "classifier",
            classifier.to_bytes(),
            1595,
            [
                180, 124, 71, 225, 4, 107, 44, 127, 181, 142, 154, 67, 201, 35, 134, 98, 57, 65,
                187, 73, 172, 213, 231, 42, 36, 177, 233, 251, 92, 178, 60, 101,
            ],
        ),
        (
            "regressor",
            regressor.to_bytes(),
            2587,
            [
                100, 242, 214, 182, 27, 5, 82, 121, 64, 157, 253, 240, 23, 181, 188, 179, 232, 105,
                178, 228, 17, 225, 213, 116, 97, 196, 21, 239, 13, 206, 129, 77,
            ],
        ),
    ] {
        assert_eq!(bytes.len(), expected_len, "{name} packed bytes changed");
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        assert_eq!(digest, expected_digest, "{name} packed bytes changed");
    }
}

#[test]
fn bootstrap_and_seed_are_deterministic_but_seed_affects_model() {
    let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
    let y = BinaryTargets::new(vec![0, 1, 0, 1, 0, 1]).unwrap();
    let a = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(8)).unwrap();
    let b = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(8)).unwrap();
    let c = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(9)).unwrap();
    assert_eq!(a.to_bytes(), b.to_bytes());
    assert_ne!(a.to_bytes(), c.to_bytes());
}

// ------------------------------------------------------------ sample weights

/// Eight rows of exactly representable features and integer targets.
///
/// Every weighted accumulation over this fixture is a sum of small integers
/// scaled by small integers, so the weighted and duplicated arithmetic below is
/// exact rather than merely close. That is what lets the equivalences be
/// asserted on fitted bytes instead of on a tolerance.
fn weight_fixture() -> (DenseMatrix, Vec<u8>, Vec<f32>) {
    let x = matrix(&[
        &[0.0, 3.0],
        &[1.0, 2.0],
        &[2.0, 1.0],
        &[3.0, 0.0],
        &[4.0, 7.0],
        &[5.0, 6.0],
        &[6.0, 5.0],
        &[7.0, 4.0],
    ]);
    let labels = vec![0, 1, 1, 0, 1, 0, 0, 1];
    let values = vec![0.0, 1.0, 2.0, 3.0, 8.0, 7.0, 6.0, 5.0];
    (x, labels, values)
}

/// The same fixture with row `repeat` present `times` times, in place.
fn duplicated_fixture(repeat: usize, times: usize) -> (DenseMatrix, Vec<u8>, Vec<f32>) {
    let (x, labels, values) = weight_fixture();
    let mut rows = Vec::new();
    let mut duplicated_labels = Vec::new();
    let mut duplicated_values = Vec::new();
    for row in 0..labels.len() {
        let copies = if row == repeat { times } else { 1 };
        for _ in 0..copies {
            rows.extend_from_slice(x.as_view().row(row).unwrap());
            duplicated_labels.push(labels[row]);
            duplicated_values.push(values[row]);
        }
    }
    let columns = x.as_view().columns();
    let matrix = DenseMatrix::new(rows, duplicated_labels.len(), columns).unwrap();
    (matrix, duplicated_labels, duplicated_values)
}

/// Weights of exactly one must not perturb one bit of the unweighted fit.
///
/// Asserted on the packed model bytes and on the whole fitted value, not on
/// predictions: a difference small enough to leave every prediction unchanged
/// would still mean the weighted path is a second implementation of the
/// unweighted one, which is exactly what the declaration promises it is not.
#[test]
fn unit_weights_reproduce_the_unweighted_fit_bit_for_bit() {
    let (x, labels, values) = weight_fixture();
    let binary = BinaryTargets::new(labels.clone()).unwrap();
    let classes = ClassTargets::new(labels).unwrap();
    let regression = RegressionTargets::new(values).unwrap();
    let ones = SampleWeights::new(vec![1.0; x.as_view().rows()]).unwrap();

    for bootstrap in [false, true] {
        let classifier_params = classifier_params(123).with_bootstrap(bootstrap);
        let regressor_params = regressor_params(123).with_bootstrap(bootstrap);

        let plain =
            RandomForestClassifier::fit(&x.as_view(), &binary, classifier_params.clone()).unwrap();
        let weighted = RandomForestClassifier::fit_weighted(
            &x.as_view(),
            &binary,
            &ones,
            classifier_params.clone(),
        )
        .unwrap();
        assert_eq!(
            plain.to_bytes(),
            weighted.to_bytes(),
            "binary classifier, bootstrap = {bootstrap}"
        );
        assert_eq!(plain, weighted);

        let plain = RandomForestClassifier::fit_multiclass(
            &x.as_view(),
            &classes,
            classifier_params.clone(),
        )
        .unwrap();
        let weighted = RandomForestClassifier::fit_multiclass_weighted(
            &x.as_view(),
            &classes,
            &ones,
            classifier_params,
        )
        .unwrap();
        assert_eq!(
            plain, weighted,
            "multiclass classifier, bootstrap = {bootstrap}"
        );

        let plain = RandomForestRegressor::fit(&x.as_view(), &regression, regressor_params.clone())
            .unwrap();
        let weighted =
            RandomForestRegressor::fit_weighted(&x.as_view(), &regression, &ones, regressor_params)
                .unwrap();
        assert_eq!(
            plain.to_bytes(),
            weighted.to_bytes(),
            "regressor, bootstrap = {bootstrap}"
        );
        assert_eq!(
            plain.to_artifact([5; 32]).unwrap(),
            weighted.to_artifact([5; 32]).unwrap()
        );
    }
}

/// An integer weight is the same fit as repeating the row that many times.
///
/// Bootstrapping is off, because a resample draws from the rows it is given:
/// adding rows changes what is drawn, so the two datasets would stop being the
/// same training problem before any weight was applied.
#[test]
fn an_integer_weight_is_the_same_fit_as_repeating_the_row() {
    let repeat = 2;
    let times = 3;
    let (x, labels, values) = weight_fixture();
    let (duplicated_x, duplicated_labels, duplicated_values) = duplicated_fixture(repeat, times);
    let mut weights = vec![1.0_f32; labels.len()];
    weights[repeat] = times as f32;
    let weights = SampleWeights::new(weights).unwrap();

    let classifier_params = classifier_params(123).with_bootstrap(false);
    let regressor_params = regressor_params(123).with_bootstrap(false);

    let weighted = RandomForestClassifier::fit_weighted(
        &x.as_view(),
        &BinaryTargets::new(labels.clone()).unwrap(),
        &weights,
        classifier_params.clone(),
    )
    .unwrap();
    let repeated = RandomForestClassifier::fit(
        &duplicated_x.as_view(),
        &BinaryTargets::new(duplicated_labels.clone()).unwrap(),
        classifier_params.clone(),
    )
    .unwrap();
    assert_eq!(
        weighted.to_bytes(),
        repeated.to_bytes(),
        "binary classifier"
    );

    let weighted = RandomForestClassifier::fit_multiclass_weighted(
        &x.as_view(),
        &ClassTargets::new(labels).unwrap(),
        &weights,
        classifier_params.clone(),
    )
    .unwrap();
    let repeated = RandomForestClassifier::fit_multiclass(
        &duplicated_x.as_view(),
        &ClassTargets::new(duplicated_labels).unwrap(),
        classifier_params,
    )
    .unwrap();
    assert_eq!(weighted, repeated, "multiclass classifier");

    let weighted = RandomForestRegressor::fit_weighted(
        &x.as_view(),
        &RegressionTargets::new(values).unwrap(),
        &weights,
        regressor_params.clone(),
    )
    .unwrap();
    let repeated = RandomForestRegressor::fit(
        &duplicated_x.as_view(),
        &RegressionTargets::new(duplicated_values).unwrap(),
        regressor_params,
    )
    .unwrap();
    assert_eq!(weighted.to_bytes(), repeated.to_bytes(), "regressor");
}

/// A weight of zero removes the row from training entirely.
///
/// This is the `times = 0` case of the duplication rule, and it is also why a
/// bootstrap resample draws from the positively weighted rows only: a row that
/// is not in the sample cannot consume one of its draws.
#[test]
fn a_zero_weight_row_is_the_same_fit_as_a_deleted_row() {
    let (x, labels, values) = weight_fixture();
    let dropped = 4;
    let mut weights = vec![1.0_f32; labels.len()];
    weights[dropped] = 0.0;
    let weights = SampleWeights::new(weights).unwrap();

    let mut kept_rows = Vec::new();
    let mut kept_values = Vec::new();
    for (row, &value) in values.iter().enumerate() {
        if row != dropped {
            kept_rows.extend_from_slice(x.as_view().row(row).unwrap());
            kept_values.push(value);
        }
    }
    let kept = DenseMatrix::new(kept_rows, labels.len() - 1, x.as_view().columns()).unwrap();

    for bootstrap in [false, true] {
        let params = regressor_params(123).with_bootstrap(bootstrap);
        let weighted = RandomForestRegressor::fit_weighted(
            &x.as_view(),
            &RegressionTargets::new(values.clone()).unwrap(),
            &weights,
            params.clone(),
        )
        .unwrap();
        let deleted = RandomForestRegressor::fit(
            &kept.as_view(),
            &RegressionTargets::new(kept_values.clone()).unwrap(),
            params,
        )
        .unwrap();
        assert_eq!(
            weighted.to_bytes(),
            deleted.to_bytes(),
            "bootstrap = {bootstrap}"
        );
    }
}

#[test]
fn weighted_fits_stay_deterministic_across_repeats_and_thread_counts() {
    let (x, labels, values) = weight_fixture();
    let weights = SampleWeights::new(vec![0.5, 2.0, 1.25, 3.0, 0.25, 1.0, 4.0, 0.75]).unwrap();
    let regression = RegressionTargets::new(values).unwrap();
    let classes = ClassTargets::new(labels).unwrap();

    let serial = regressor_params(77);
    let parallel = regressor_params(77).with_n_jobs(crate::ensemble::NJobs::Count(4));
    let one =
        RandomForestRegressor::fit_weighted(&x.as_view(), &regression, &weights, serial.clone())
            .unwrap();
    let repeat =
        RandomForestRegressor::fit_weighted(&x.as_view(), &regression, &weights, serial).unwrap();
    let threaded =
        RandomForestRegressor::fit_weighted(&x.as_view(), &regression, &weights, parallel).unwrap();
    assert_eq!(one.to_bytes(), repeat.to_bytes());
    assert_eq!(one.to_bytes(), threaded.to_bytes());

    let serial = multiclass_params(77);
    let parallel = multiclass_params(77).with_n_jobs(crate::ensemble::NJobs::Count(4));
    let one =
        RandomForestClassifier::fit_multiclass_weighted(&x.as_view(), &classes, &weights, serial)
            .unwrap();
    let threaded =
        RandomForestClassifier::fit_multiclass_weighted(&x.as_view(), &classes, &weights, parallel)
            .unwrap();
    // The retained thread count is part of the fitted parameters, so the two
    // values differ there by construction; the fitted trees must not.
    assert_eq!(
        one.predict_proba(&x.as_view()).unwrap(),
        threaded.predict_proba(&x.as_view()).unwrap()
    );
}

/// A length mismatch is reported before any training work begins.
#[test]
fn weighted_fitting_rejects_a_length_mismatch_before_training() {
    let (x, labels, values) = weight_fixture();
    let short = SampleWeights::new(vec![1.0; labels.len() - 1]).unwrap();
    let expected = ModelError::SampleWeightLength {
        rows: labels.len(),
        weights: labels.len() - 1,
    };
    assert_eq!(
        RandomForestClassifier::fit_weighted(
            &x.as_view(),
            &BinaryTargets::new(labels.clone()).unwrap(),
            &short,
            classifier_params(1),
        )
        .unwrap_err(),
        expected
    );
    assert_eq!(
        RandomForestClassifier::fit_multiclass_weighted(
            &x.as_view(),
            &ClassTargets::new(labels).unwrap(),
            &short,
            classifier_params(1),
        )
        .unwrap_err(),
        expected
    );
    assert_eq!(
        RandomForestRegressor::fit_weighted(
            &x.as_view(),
            &RegressionTargets::new(values).unwrap(),
            &short,
            regressor_params(1),
        )
        .unwrap_err(),
        expected
    );
}

/// Weights change the model. Without this the equivalences above would be
/// satisfied by an implementation that ignored its weights entirely.
#[test]
fn weights_are_not_inert() {
    let (x, labels, values) = weight_fixture();
    let regression = RegressionTargets::new(values).unwrap();
    let mut skewed = vec![1.0_f32; labels.len()];
    skewed[0] = 9.0;
    let skewed = SampleWeights::new(skewed).unwrap();
    let plain = RandomForestRegressor::fit(&x.as_view(), &regression, regressor_params(5)).unwrap();
    let weighted = RandomForestRegressor::fit_weighted(
        &x.as_view(),
        &regression,
        &skewed,
        regressor_params(5),
    )
    .unwrap();
    assert_ne!(plain.to_bytes(), weighted.to_bytes());
}

// --------------------------------------------------- classifier persistence

fn classifier_artifact_fixture() -> (DenseMatrix, RandomForestClassifier, RandomForestClassifier) {
    let (x, labels, _) = weight_fixture();
    let params = classifier_params(21)
        .with_n_estimators(4)
        .with_max_depth(Some(3));
    let binary = RandomForestClassifier::fit(
        &x.as_view(),
        &BinaryTargets::new(labels.clone()).unwrap(),
        params.clone(),
    )
    .unwrap();
    // Non-contiguous labels, so a decoder that reconstructed the class list
    // instead of reading it would relabel every prediction.
    let multiclass = RandomForestClassifier::fit_multiclass(
        &x.as_view(),
        &ClassTargets::new(vec![3, 7, 10, 3, 7, 10, 3, 7]).unwrap(),
        params,
    )
    .unwrap();
    (x, binary, multiclass)
}

#[test]
fn classifier_artifacts_round_trip_both_leaf_representations() {
    let (x, binary, multiclass) = classifier_artifact_fixture();
    let schema = [17; 32];
    for (name, model) in [("binary", &binary), ("multiclass", &multiclass)] {
        let bytes = model.to_artifact(schema).unwrap();
        assert_eq!(
            bytes,
            model.to_artifact(schema).unwrap(),
            "{name} is stable"
        );
        let decoded = RandomForestClassifier::from_artifact(&bytes, schema).unwrap();
        assert_eq!(&decoded, model, "{name} round trip");
        assert_eq!(decoded.classes(), model.classes(), "{name} classes");
        assert_eq!(
            decoded.predict(&x.as_view()).unwrap(),
            model.predict(&x.as_view()).unwrap(),
            "{name} labels"
        );
        assert_eq!(
            decoded.predict_proba(&x.as_view()).unwrap(),
            model.predict_proba(&x.as_view()).unwrap(),
            "{name} probabilities"
        );
        assert_eq!(
            decoded.to_artifact(schema).unwrap(),
            bytes,
            "{name} re-encodes to exactly the bytes it decoded from"
        );
        assert_eq!(
            RandomForestClassifier::from_artifact(&bytes, [18; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch,
            "{name} is schema-bound"
        );
    }

    // The two flavours are different models and their artifacts differ; the
    // regressor's kind is a different kind again.
    assert_ne!(
        binary.to_artifact(schema).unwrap(),
        multiclass.to_artifact(schema).unwrap()
    );
    assert_eq!(
        RandomForestRegressor::from_artifact(&binary.to_artifact(schema).unwrap(), schema)
            .unwrap_err(),
        ArtifactError::UnsupportedModelKind {
            found: RANDOM_FOREST_CLASSIFIER_ARTIFACT_KIND
        }
    );
}

#[test]
fn classifier_artifacts_restore_every_retained_parameter() {
    let (x, labels, _) = weight_fixture();
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(3)
        .with_max_depth(Some(5))
        .with_min_samples_split(3)
        .with_min_samples_leaf(2)
        .with_max_features(MaxFeatures::Count(2))
        .with_bootstrap(false)
        .with_random_state(4_242)
        .with_n_jobs(crate::ensemble::NJobs::Count(2));
    let model = RandomForestClassifier::fit(
        &x.as_view(),
        &BinaryTargets::new(labels).unwrap(),
        params.clone(),
    )
    .unwrap();
    let bytes = model.to_artifact([1; 32]).unwrap();
    let decoded = RandomForestClassifier::from_artifact(&bytes, [1; 32]).unwrap();
    assert_eq!(decoded.get_params(), &params);
    assert_eq!(decoded.n_features_in(), model.n_features_in());
}

/// A single observed class round trips in both flavours, which is the shape
/// with one probability column of `1.0`.
#[test]
fn classifier_artifacts_round_trip_single_class_and_root_leaf_trees() {
    let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let params = RandomForestClassifierParams::default()
        .with_n_estimators(2)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All)
        .with_random_state(3);
    let binary = RandomForestClassifier::fit(
        &x.as_view(),
        &BinaryTargets::new(vec![1, 1, 1, 1]).unwrap(),
        params.clone(),
    )
    .unwrap();
    assert_eq!(binary.classes(), [1]);
    let bytes = binary.to_artifact([2; 32]).unwrap();
    let decoded = RandomForestClassifier::from_artifact(&bytes, [2; 32]).unwrap();
    assert_eq!(decoded, binary);
    assert_eq!(decoded.predict_proba(&x.as_view()).unwrap(), vec![1.0; 4]);

    let multiclass = RandomForestClassifier::fit_multiclass(
        &x.as_view(),
        &ClassTargets::new(vec![9, 9, 9, 9]).unwrap(),
        params,
    )
    .unwrap();
    assert_eq!(multiclass.classes(), [9]);
    let bytes = multiclass.to_artifact([2; 32]).unwrap();
    let decoded = RandomForestClassifier::from_artifact(&bytes, [2; 32]).unwrap();
    assert_eq!(decoded, multiclass);
    assert_eq!(decoded.predict_proba(&x.as_view()).unwrap(), vec![1.0; 4]);
    assert_eq!(decoded.to_artifact([2; 32]).unwrap(), bytes);
}

#[test]
fn classifier_artifact_rejects_invalid_metadata_and_framing() {
    let (_, binary, multiclass) = classifier_artifact_fixture();
    let schema = [17; 32];
    let bytes = binary.to_artifact(schema).unwrap();
    let multiclass_bytes = multiclass.to_artifact(schema).unwrap();

    // Metadata words, in the order the writer emits them.
    const OBJECTIVE: usize = 0;
    const FLAVOUR: usize = 4;
    const N_FEATURES: usize = 8;
    const N_ESTIMATORS: usize = 12;
    const MIN_SAMPLES_SPLIT: usize = 20;
    const MIN_SAMPLES_LEAF: usize = 24;
    const MAX_FEATURES_TAG: usize = 28;
    const CLASS_COUNT: usize = 64;
    const FIRST_CLASS: usize = 68;

    for (name, offset, value) in [
        ("objective version", OBJECTIVE, 9_u32),
        ("forest flavour", FLAVOUR, 3),
        ("zero feature width", N_FEATURES, 0),
        ("estimator count disagrees with tree count", N_ESTIMATORS, 9),
        ("min_samples_split below two", MIN_SAMPLES_SPLIT, 1),
        ("zero min_samples_leaf", MIN_SAMPLES_LEAF, 0),
        ("unknown max_features tag", MAX_FEATURES_TAG, 9),
        ("zero class count", CLASS_COUNT, 0),
        ("a binary class label outside {0, 1}", FIRST_CLASS, 5),
    ] {
        let mut mutated = bytes.clone();
        metadata_u32(&mut mutated, offset, value);
        assert_eq!(
            RandomForestClassifier::from_artifact(&mutated, schema).unwrap_err(),
            ArtifactError::InvalidPayload,
            "{name} was accepted"
        );
    }

    // Relabelling a binary forest as multiclass leaves it without the leaf
    // probability component the multiclass reader requires, so the two
    // flavours cannot be crossed by rewriting the tag.
    let mut crossed = bytes.clone();
    metadata_u32(&mut crossed, FLAVOUR, 2);
    assert!(RandomForestClassifier::from_artifact(&crossed, schema).is_err());

    // A multiclass class list must stay strictly increasing: [3, 7, 10] with
    // the first label raised to 20 is no longer sorted.
    let mut unsorted = multiclass_bytes.clone();
    metadata_u32(&mut unsorted, FIRST_CLASS, 20);
    assert_eq!(
        RandomForestClassifier::from_artifact(&unsorted, schema).unwrap_err(),
        ArtifactError::InvalidPayload
    );

    assert_eq!(
        RandomForestClassifier::from_artifact(&bytes[..bytes.len() - 1], schema).unwrap_err(),
        ArtifactError::ChecksumMismatch
    );
    let mut trailing = bytes;
    trailing.push(0);
    assert_eq!(
        RandomForestClassifier::from_artifact(&trailing, schema).unwrap_err(),
        ArtifactError::ChecksumMismatch
    );
}

/// The multiclass leaf record carries a reserved zero where a scalar leaf
/// carries its value, so a nonzero there would be a second encoding of one
/// model.
#[test]
fn multiclass_leaf_records_reserve_their_scalar_slot() {
    let (_, _, multiclass) = classifier_artifact_fixture();
    let schema = [17; 32];
    let bytes = multiclass.to_artifact(schema).unwrap();

    // Skip the metadata component, whose declared length is its own header's
    // third word, and find the first leaf record in the first tree.
    let metadata_len = u32::from_le_bytes(
        bytes[PAYLOAD_START + 4..PAYLOAD_START + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let first_tree = PAYLOAD_START + 8 + metadata_len;
    let leaf = (first_tree..bytes.len() - 20)
        .step_by(4)
        .find(|&offset| bytes[offset..offset + 20] == [0_u8; 20])
        .expect("a leaf record with its reserved zero");

    let mut mutated = bytes;
    mutated[leaf + 4..leaf + 8].copy_from_slice(&0x3f80_0000_u32.to_le_bytes());
    resign(&mut mutated);
    assert_eq!(
        RandomForestClassifier::from_artifact(&mutated, schema).unwrap_err(),
        ArtifactError::InvalidPayload
    );
}

#[test]
fn rejects_invalid_configuration_and_data() {
    let x = matrix(&[&[0.0], &[1.0]]);
    let y = BinaryTargets::new(vec![0, 1]).unwrap();
    let cfg = classifier_params(1).with_n_estimators(0);
    assert_eq!(
        RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap_err(),
        ModelError::InvalidEstimatorCount
    );
    let cfg = classifier_params(1).with_min_samples_split(1);
    assert_eq!(
        RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap_err(),
        ModelError::InvalidMinSamplesSplit
    );
    let cfg = classifier_params(1).with_min_samples_leaf(0);
    assert_eq!(
        RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap_err(),
        ModelError::InvalidMinSamplesLeaf
    );
    let cfg = classifier_params(1).with_max_features(MaxFeatures::Count(2));
    assert!(matches!(
        RandomForestClassifier::fit(&x.as_view(), &y, cfg),
        Err(ModelError::InvalidMaxFeatures { .. })
    ));
}

#[test]
fn checks_prediction_dimensions_and_output_size() {
    let x = matrix(&[&[0.0], &[1.0]]);
    let y = BinaryTargets::new(vec![0, 1]).unwrap();
    let forest = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(2)).unwrap();
    assert!(matches!(
        forest.predict_positive_proba(&[0.0, 1.0]),
        Err(ModelError::FeatureDimension { .. })
    ));
    let mut too_short = [0.0];
    assert!(matches!(
        forest.predict_positive_proba_into(&x.as_view(), &mut too_short),
        Err(ModelError::OutputLength { .. })
    ));
}

#[test]
fn every_packed_tree_has_valid_topology() {
    let x = matrix(&[
        &[0.0, 2.0],
        &[1.0, 3.0],
        &[2.0, 0.0],
        &[3.0, 1.0],
        &[4.0, 6.0],
        &[5.0, 7.0],
        &[6.0, 4.0],
        &[7.0, 5.0],
    ]);
    let y = BinaryTargets::new(vec![0, 0, 1, 1, 0, 1, 0, 1]).unwrap();
    let forest = RandomForestClassifier::fit(&x.as_view(), &y, classifier_params(55)).unwrap();
    assert_eq!(std::mem::size_of::<PackedNode>(), 16);
    for tree in forest.binary_trees() {
        assert!(!tree.nodes.is_empty() || tree.root_leaf.is_some());
        assert!(tree.root_leaf.is_none_or(f32::is_finite));
        for node in &tree.nodes {
            assert!(((node.feature_and_flags & FEATURE_MASK) as usize) < forest.n_features_in());
            assert!(node.threshold.is_finite());
            if node.feature_and_flags & LEFT_IS_LEAF != 0 {
                assert!(f32::from_bits(node.left).is_finite());
            } else {
                assert!((node.left as usize) < tree.nodes.len());
            }
            if node.feature_and_flags & RIGHT_IS_LEAF != 0 {
                assert!(f32::from_bits(node.right).is_finite());
            } else {
                assert!((node.right as usize) < tree.nodes.len());
            }
        }
    }
}

#[test]
fn pathological_deep_tree_uses_an_explicit_builder_stack() {
    const ROWS: usize = 4096;
    let values = (0..ROWS).map(|row| row as f32).collect();
    let labels = (0..ROWS).map(|row| (row & 1) as u8).collect();
    let x = DenseMatrix::new(values, ROWS, 1).unwrap();
    let y = BinaryTargets::new(labels).unwrap();
    let cfg = RandomForestClassifierParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All);
    let forest = RandomForestClassifier::fit(&x.as_view(), &y, cfg).unwrap();
    assert_eq!(forest.binary_trees()[0].nodes.len(), ROWS - 1);
}

/// Offsets into a version-2 envelope carrying one input schema: the fixed
/// header is 24 bytes, one schema record is 36, and each component adds an
/// 8-byte header.
const PAYLOAD_START: usize = 24 + 36;
const METADATA_START: usize = PAYLOAD_START + 8;
const METADATA_FIELD_BYTES: usize = 13 * 4 + 8;
const FIRST_TREE_START: usize = METADATA_START + METADATA_FIELD_BYTES;

// Byte offsets of each metadata field, relative to the component payload.
// `RANDOM_STATE` is the only 64-bit field, so later offsets are not multiples
// of four.
const OBJECTIVE_VERSION: usize = 0;
const N_FEATURES: usize = 4;
const N_ESTIMATORS: usize = 8;
const MAX_DEPTH: usize = 12;
const MIN_SAMPLES_SPLIT: usize = 16;
const MIN_SAMPLES_LEAF: usize = 20;
const MAX_FEATURES_TAG: usize = 24;
const MAX_FEATURES_COUNT: usize = 28;
const BOOTSTRAP: usize = 32;
const N_JOBS_TAG: usize = 44;
const N_JOBS_COUNT: usize = 48;
const TREE_COUNT: usize = 52;
const TOTAL_NODES: usize = 56;

fn write_metadata_u32(bytes: &mut [u8], offset: usize, value: u32) {
    let offset = METADATA_START + offset;
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn metadata_u32(bytes: &mut [u8], offset: usize, value: u32) {
    write_metadata_u32(bytes, offset, value);
    resign(bytes);
}

fn resign(bytes: &mut [u8]) {
    let checksum_offset = bytes.len() - 32;
    let checksum = Sha256::digest(&bytes[..checksum_offset]);
    bytes[checksum_offset..].copy_from_slice(&checksum);
}

fn artifact_fixture() -> (DenseMatrix, RandomForestRegressor) {
    let x = matrix(&[
        &[0.0, 3.0],
        &[1.0, 2.0],
        &[2.0, 1.0],
        &[3.0, 0.0],
        &[4.0, 7.0],
        &[5.0, 6.0],
        &[6.0, 5.0],
        &[7.0, 4.0],
    ]);
    let y = RegressionTargets::new(vec![0.0, 1.0, 1.5, 2.5, 8.0, 7.0, 6.0, 5.0]).unwrap();
    let model = RandomForestRegressor::fit(
        &x.as_view(),
        &y,
        RandomForestRegressorParams::default()
            .with_n_estimators(4)
            .with_max_depth(Some(4))
            .with_max_features(MaxFeatures::All)
            .with_random_state(5),
    )
    .unwrap();
    (x, model)
}

#[test]
fn regressor_artifact_round_trip_is_deterministic_schema_bound_and_kind_isolated() {
    let (x, model) = artifact_fixture();
    let schema = [17; 32];
    let left = model.to_artifact(schema).unwrap();
    assert_eq!(left, model.to_artifact(schema).unwrap());

    let decoded = RandomForestRegressor::from_artifact(&left, schema).unwrap();
    assert_eq!(decoded, model);
    assert_eq!(decoded.to_bytes(), model.to_bytes());
    assert_eq!(
        decoded.predict(&x.as_view()).unwrap(),
        model.predict(&x.as_view()).unwrap()
    );
    assert_eq!(decoded.to_artifact(schema).unwrap(), left);

    assert_eq!(
        RandomForestRegressor::from_artifact(&left, [18; 32]).unwrap_err(),
        ArtifactError::FeatureSchemaMismatch
    );
    assert_eq!(
        Ridge::from_artifact(&left, schema).unwrap_err(),
        ArtifactError::UnsupportedModelKind {
            found: RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
        }
    );
    assert_eq!(
        HistGradientBoostingRegressor::from_artifact(&left, schema).unwrap_err(),
        ArtifactError::UnsupportedModelKind {
            found: RANDOM_FOREST_REGRESSOR_ARTIFACT_KIND,
        }
    );

    let mut corrupted = left;
    corrupted[FIRST_TREE_START + 20] ^= 1;
    assert_eq!(
        RandomForestRegressor::from_artifact(&corrupted, schema).unwrap_err(),
        ArtifactError::ChecksumMismatch
    );
}

#[test]
fn regressor_artifact_restores_every_retained_parameter() {
    let x = matrix(&[&[0.0, 1.0], &[1.0, 0.0], &[2.0, 3.0], &[3.0, 2.0]]);
    let y = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let schema = [2; 32];
    for params in [
        RandomForestRegressorParams::default().with_n_estimators(3),
        RandomForestRegressorParams::default()
            .with_n_estimators(2)
            .with_max_depth(None)
            .with_min_samples_split(3)
            .with_min_samples_leaf(2)
            .with_max_features(MaxFeatures::Sqrt)
            .with_bootstrap(false)
            .with_random_state(u64::MAX)
            .with_n_jobs(NJobs::Count(2)),
        RandomForestRegressorParams::default()
            .with_n_estimators(2)
            .with_max_depth(Some(1))
            .with_max_features(MaxFeatures::Count(2))
            .with_n_jobs(NJobs::All),
    ] {
        let model = RandomForestRegressor::fit(&x.as_view(), &y, params.clone()).unwrap();
        let decoded =
            RandomForestRegressor::from_artifact(&model.to_artifact(schema).unwrap(), schema)
                .unwrap();
        assert_eq!(decoded.get_params(), &params);
        assert_eq!(decoded.n_features_in(), model.n_features_in());
        assert_eq!(decoded, model);
    }
}

#[test]
fn regressor_artifact_round_trips_single_leaf_trees() {
    let x = matrix(&[&[1.0], &[1.0], &[1.0], &[1.0]]);
    let y = RegressionTargets::new(vec![2.5, 2.5, 2.5, 2.5]).unwrap();
    let model = RandomForestRegressor::fit(
        &x.as_view(),
        &y,
        RandomForestRegressorParams::default().with_n_estimators(3),
    )
    .unwrap();
    let schema = [6; 32];
    let bytes = model.to_artifact(schema).unwrap();
    // Three one-node logical trees: every tree collapsed to a root leaf.
    let total_nodes = u32::from_le_bytes(
        bytes[METADATA_START + TOTAL_NODES..METADATA_START + TOTAL_NODES + 4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(total_nodes, 3);
    let decoded = RandomForestRegressor::from_artifact(&bytes, schema).unwrap();
    assert_eq!(decoded, model);
    assert_eq!(decoded.predict_one(&[1.0]).unwrap(), 2.5);
}

#[test]
fn regressor_artifact_rejects_invalid_metadata_tree_records_and_framing() {
    let (_, model) = artifact_fixture();
    let schema = [29; 32];
    let bytes = model.to_artifact(schema).unwrap();
    assert!(RandomForestRegressor::from_artifact(&bytes, schema).is_ok());

    // Metadata field offset, and the value that breaks it.
    for (name, offset, value) in [
        ("objective version", OBJECTIVE_VERSION, 2),
        ("zero feature width", N_FEATURES, 0),
        ("feature width beyond the ceiling", N_FEATURES, 1_000_001),
        (
            "estimator count disagreeing with the trees",
            N_ESTIMATORS,
            3,
        ),
        ("max_depth beyond the ceiling", MAX_DEPTH, 1_048_577),
        ("min_samples_split below two", MIN_SAMPLES_SPLIT, 1),
        ("zero min_samples_leaf", MIN_SAMPLES_LEAF, 0),
        ("unknown max_features tag", MAX_FEATURES_TAG, 9),
        (
            "max_features count set for the All tag",
            MAX_FEATURES_COUNT,
            3,
        ),
        ("non-boolean bootstrap flag", BOOTSTRAP, 2),
        ("unknown n_jobs tag", N_JOBS_TAG, 9),
        ("n_jobs count set for the Serial tag", N_JOBS_COUNT, 1),
        ("tree count disagreeing with the estimators", TREE_COUNT, 3),
        ("declared node total below the tree count", TOTAL_NODES, 3),
        (
            "declared node total beyond the ceiling",
            TOTAL_NODES,
            1_048_577,
        ),
    ] {
        let mut corrupted = bytes.clone();
        metadata_u32(&mut corrupted, offset, value);
        assert_eq!(
            RandomForestRegressor::from_artifact(&corrupted, schema).unwrap_err(),
            ArtifactError::InvalidPayload,
            "{name} was accepted"
        );
    }

    // A max_features count must stay inside the fitted width.
    let mut inconsistent = bytes.clone();
    write_metadata_u32(&mut inconsistent, MAX_FEATURES_TAG, 3);
    write_metadata_u32(&mut inconsistent, MAX_FEATURES_COUNT, 3);
    resign(&mut inconsistent);
    assert_eq!(
        RandomForestRegressor::from_artifact(&inconsistent, schema).unwrap_err(),
        ArtifactError::InvalidPayload
    );

    let mut component_kind = bytes.clone();
    component_kind[FIRST_TREE_START..FIRST_TREE_START + 2].copy_from_slice(&3_u16.to_le_bytes());
    resign(&mut component_kind);
    assert_eq!(
        RandomForestRegressor::from_artifact(&component_kind, schema).unwrap_err(),
        ArtifactError::InvalidPayload
    );

    // The first logical record of the first tree: tag, then feature index.
    let record = FIRST_TREE_START + 8 + 12;
    assert_eq!(
        u32::from_le_bytes(bytes[record..record + 4].try_into().unwrap()),
        1,
        "fixture must start with a branch record"
    );
    let mut feature = bytes.clone();
    feature[record + 4..record + 8].copy_from_slice(&7_u32.to_le_bytes());
    resign(&mut feature);
    assert_eq!(
        RandomForestRegressor::from_artifact(&feature, schema).unwrap_err(),
        ArtifactError::InvalidPayload
    );

    assert_eq!(
        RandomForestRegressor::from_artifact(&bytes[..bytes.len() - 1], schema).unwrap_err(),
        ArtifactError::ChecksumMismatch
    );

    let mut trailing = bytes.clone();
    trailing.insert(bytes.len() - 32, 0);
    resign(&mut trailing);
    assert_eq!(
        RandomForestRegressor::from_artifact(&trailing, schema).unwrap_err(),
        ArtifactError::TrailingBytes
    );

    assert_eq!(
        RandomForestRegressor::from_artifact(&[], schema).unwrap_err(),
        ArtifactError::Truncated
    );
}

#[test]
fn regressor_artifact_refuses_models_whose_averaged_prediction_overflows() {
    let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let y = RegressionTargets::new(vec![f32::MAX; 4]).unwrap();
    let model = RandomForestRegressor::fit(
        &x.as_view(),
        &y,
        RandomForestRegressorParams::default()
            .with_n_estimators(4)
            .with_bootstrap(false),
    )
    .unwrap();
    assert_eq!(
        model.to_artifact([1; 32]).unwrap_err(),
        ArtifactError::InvalidPayload
    );
}

// --------------------------------------------------------------- multiclass

/// Twelve rows, two features, three classes with clearly separated regions.
fn three_class_problem() -> (DenseMatrix, ClassTargets) {
    let x = matrix(&[
        &[0.0, 0.0],
        &[0.5, 0.2],
        &[0.2, 0.6],
        &[1.0, 0.3],
        &[2.0, 0.1],
        &[1.8, 0.5],
        &[2.2, 0.9],
        &[0.3, 2.0],
        &[0.8, 2.4],
        &[1.2, 2.2],
        &[1.0, 3.0],
        &[0.1, 1.0],
    ]);
    (
        x,
        ClassTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0]).unwrap(),
    )
}

fn multiclass_params(random_state: u64) -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(17)
        .with_max_features(MaxFeatures::All)
        .with_random_state(random_state)
}

#[test]
fn the_ensemble_averages_per_tree_probability_vectors_rather_than_voting() {
    // Four depth-one trees over three classes. A hard vote over four trees can
    // only ever produce multiples of a quarter, so any other value is decisive
    // evidence that whole distributions are being averaged.
    let (x, y) = three_class_problem();
    let model = RandomForestClassifier::fit_multiclass(
        &x.as_view(),
        &y,
        RandomForestClassifierParams::default()
            .with_n_estimators(4)
            .with_max_depth(Some(1))
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();

    let query = matrix(&[&[0.2, 0.3], &[2.0, 0.4], &[1.0, 3.0], &[1.0, 1.2]]);
    let probabilities = model.predict_proba(&query.as_view()).unwrap();
    assert_eq!(probabilities.len(), query.rows() * 3);

    let quarter = 0.25_f32;
    let off_grid = probabilities
        .iter()
        .filter(|value| (*value / quarter).fract() != 0.0)
        .count();
    assert!(
        off_grid > 0,
        "every value was a multiple of 1/4, which a hard vote could also have \
         produced: {probabilities:?}"
    );

    // And the averaging identity itself: the ensemble row is the mean of the
    // per-tree rows, to the rounding of one division.
    for (row_index, row) in query.iter_rows().enumerate() {
        let mut expected = [0.0_f64; 3];
        let Forest::Multiclass(trees) = &model.core.forest else {
            unreachable!("multiclass fit");
        };
        for tree in trees {
            for (slot, &value) in expected.iter_mut().zip(tree.probabilities(row)) {
                *slot += f64::from(value);
            }
        }
        for (class, expected) in expected.iter().enumerate() {
            let actual = f64::from(probabilities[row_index * 3 + class]);
            assert!(
                (actual - expected / 4.0).abs() <= 1.0e-6,
                "row {row_index} class {class}: {actual} vs {}",
                expected / 4.0
            );
        }
    }
}

#[test]
fn multiclass_labels_never_disagree_with_the_probability_argmax() {
    for classes in 2..=6_u8 {
        let rows = 240;
        let mut values = Vec::with_capacity(rows * 3);
        let mut labels = Vec::with_capacity(rows);
        let mut state = 0x9e37_79b9_u64 + u64::from(classes);
        for row in 0..rows {
            for _ in 0..3 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                values.push(((state >> 33) as f32 / (1_u32 << 31) as f32) * 4.0 - 2.0);
            }
            labels.push((row % usize::from(classes)) as u8);
        }
        let data = DenseMatrix::new(values, rows, 3).unwrap();
        let targets = ClassTargets::new(labels).unwrap();
        let model =
            RandomForestClassifier::fit_multiclass(&data.as_view(), &targets, multiclass_params(3))
                .unwrap();

        let width = usize::from(classes);
        let predicted = model.predict(&data.as_view()).unwrap();
        let probabilities = model.predict_proba(&data.as_view()).unwrap();
        let mut into = vec![0_u8; rows];
        model.predict_into(&data.as_view(), &mut into).unwrap();
        assert_eq!(into, predicted);
        for (index, (label, row)) in predicted
            .iter()
            .zip(probabilities.chunks_exact(width))
            .enumerate()
        {
            let mut best = 0;
            for class in 1..width {
                if row[class] > row[best] {
                    best = class;
                }
            }
            assert_eq!(*label, model.classes()[best], "row {index}");
            assert_eq!(
                model.predict_one(data.row(index).unwrap()).unwrap(),
                *label,
                "scalar path, row {index}"
            );
        }
    }
}

#[test]
fn multiclass_probability_rows_stay_inside_the_frozen_tolerance() {
    let (x, y) = three_class_problem();
    for estimators in [1_usize, 3, 17] {
        let model = RandomForestClassifier::fit_multiclass(
            &x.as_view(),
            &y,
            multiclass_params(1).with_n_estimators(estimators),
        )
        .unwrap();
        for row in model
            .predict_proba(&x.as_view())
            .unwrap()
            .chunks_exact(model.classes().len())
        {
            let sum = row.iter().sum::<f32>();
            // Not renormalized: one is reached only to `n_classes` ulps.
            assert!(
                (sum - 1.0).abs() <= model.classes().len() as f32 * f32::EPSILON,
                "{estimators} trees, row {row:?} sums to {sum}"
            );
            assert!(row.iter().all(|&value| (0.0..=1.0).contains(&value)));
        }
    }
}

#[test]
fn multiclass_columns_follow_sorted_labels_with_no_contiguity_assumption() {
    let (x, y) = three_class_problem();
    let base =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &y, multiclass_params(5)).unwrap();
    assert_eq!(base.classes(), &[0, 1, 2]);

    let relabelled = ClassTargets::new(
        y.as_slice()
            .iter()
            .map(|&label| match label {
                0 => 7,
                1 => 3,
                _ => 10,
            })
            .collect(),
    )
    .unwrap();
    let permuted =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &relabelled, multiclass_params(5))
            .unwrap();
    assert_eq!(permuted.classes(), &[3, 7, 10]);

    // The trees are identical — only the class column order changed — so the
    // probability rows must be an exact permutation, not merely similar.
    let base_probabilities = base.predict_proba(&x.as_view()).unwrap();
    let permuted_probabilities = permuted.predict_proba(&x.as_view()).unwrap();
    for (base, permuted) in base_probabilities
        .chunks_exact(3)
        .zip(permuted_probabilities.chunks_exact(3))
    {
        assert_eq!(permuted, [base[1], base[0], base[2]]);
    }
    assert_eq!(
        permuted.predict(&x.as_view()).unwrap(),
        base.predict(&x.as_view())
            .unwrap()
            .into_iter()
            .map(|label| match label {
                0 => 7,
                1 => 3,
                _ => 10,
            })
            .collect::<Vec<_>>()
    );
    for (index, &label) in permuted.classes().iter().enumerate() {
        let column = permuted.predict_class_proba(&x.as_view(), label).unwrap();
        for (row, &value) in column.iter().enumerate() {
            assert_eq!(value, permuted_probabilities[row * 3 + index]);
        }
        assert_eq!(
            permuted
                .predict_class_proba_one(x.row(0).unwrap(), label)
                .unwrap(),
            permuted_probabilities[index]
        );
    }
    assert_eq!(
        permuted.predict_class_proba(&x.as_view(), 0).unwrap_err(),
        ModelError::UnknownClass { class: 0 }
    );
}

#[test]
fn a_strict_subset_tie_selects_the_lowest_tied_class_not_the_first() {
    // One feature held constant, so no split can separate anything and every
    // tree is a single leaf holding the class frequencies: 1/5, 2/5, 2/5.
    let x = matrix(&[&[1.0], &[1.0], &[1.0], &[1.0], &[1.0]]);
    let targets = ClassTargets::new(vec![5, 20, 20, 9, 9]).unwrap();
    let model = RandomForestClassifier::fit_multiclass(
        &x.as_view(),
        &targets,
        RandomForestClassifierParams::default()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_max_features(MaxFeatures::All)
            .with_random_state(0),
    )
    .unwrap();
    assert_eq!(model.classes(), &[5, 9, 20]);

    let probabilities = model.predict_proba_one(&[1.0]).unwrap();
    assert_eq!(probabilities, vec![0.2, 0.4, 0.4]);
    assert_eq!(
        probabilities[1].to_bits(),
        probabilities[2].to_bits(),
        "the tie must be exact, not near"
    );
    // Lowest *tied* index: label 9, not the first class 5.
    assert_eq!(model.predict_one(&[1.0]).unwrap(), 9);
    assert_eq!(model.predict(&x.as_view()).unwrap(), vec![9; 5]);
}

#[test]
fn a_single_observed_class_fits_and_returns_one_all_ones_column() {
    let (x, _) = three_class_problem();
    for label in [0_u8, 3, 200] {
        let targets = ClassTargets::new(vec![label; x.rows()]).unwrap();
        let model =
            RandomForestClassifier::fit_multiclass(&x.as_view(), &targets, multiclass_params(9))
                .unwrap();
        assert_eq!(model.classes(), &[label]);
        let probabilities = model.predict_proba(&x.as_view()).unwrap();
        assert_eq!(probabilities, vec![1.0; x.rows()]);
        assert_eq!(model.predict(&x.as_view()).unwrap(), vec![label; x.rows()]);
        assert_eq!(
            model.predict_proba_one(x.row(0).unwrap()).unwrap(),
            vec![1.0]
        );
        assert_eq!(
            model.predict_class_proba(&x.as_view(), label).unwrap(),
            vec![1.0; x.rows()]
        );
    }
}

#[test]
fn multiclass_fits_are_deterministic_across_thread_counts() {
    let (x, y) = three_class_problem();
    let serial =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &y, multiclass_params(11)).unwrap();
    let repeat =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &y, multiclass_params(11)).unwrap();
    let parallel = RandomForestClassifier::fit_multiclass(
        &x.as_view(),
        &y,
        multiclass_params(11).with_n_jobs(NJobs::Count(4)),
    )
    .unwrap();
    assert_eq!(serial, repeat);
    // The thread count is retained in the parameters, so the trees themselves
    // are what must match — a parallel fit is the same forest, not the same
    // configuration.
    assert_eq!(serial.core.forest, parallel.core.forest);
    let different =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &y, multiclass_params(12)).unwrap();
    assert_ne!(serial.core.forest, different.core.forest);
}

#[test]
fn a_multiclass_fit_has_no_positive_class_and_validates_before_writing() {
    let (x, y) = three_class_problem();
    let model =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &y, multiclass_params(2)).unwrap();
    let expected = ModelError::MulticlassOutput { columns: 3 };
    assert_eq!(
        model.predict_positive_proba(x.row(0).unwrap()).unwrap_err(),
        expected
    );
    let mut sentinel = vec![9.0_f32; x.rows()];
    assert_eq!(
        model
            .predict_positive_proba_into(&x.as_view(), &mut sentinel)
            .unwrap_err(),
        expected
    );
    assert!(sentinel.iter().all(|&value| value == 9.0));

    let wrong_width = DenseMatrix::new(vec![1.0; x.rows() * 3], x.rows(), 3).unwrap();
    let mut probabilities = vec![9.0_f32; x.rows() * 3];
    assert_eq!(
        model
            .predict_proba_into(&wrong_width.as_view(), &mut probabilities)
            .unwrap_err(),
        ModelError::FeatureDimension {
            expected: 2,
            actual: 3
        }
    );
    assert!(probabilities.iter().all(|&value| value == 9.0));

    let mut short = vec![9.0_f32; 2];
    assert_eq!(
        model
            .predict_proba_into(&x.as_view(), &mut short)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: x.rows() * 3,
            actual: 2
        }
    );
    assert!(short.iter().all(|&value| value == 9.0));
}

#[test]
fn multiclass_fitting_validates_targets_and_configuration_before_training() {
    let (x, _) = three_class_problem();
    assert_eq!(
        RandomForestClassifier::fit_multiclass(
            &x.as_view(),
            &ClassTargets::new(vec![0, 1, 2]).unwrap(),
            multiclass_params(0),
        )
        .unwrap_err(),
        ModelError::TargetLength {
            rows: 12,
            targets: 3
        }
    );
    let targets = ClassTargets::new(vec![0; x.rows()]).unwrap();
    assert_eq!(
        RandomForestClassifier::fit_multiclass(
            &x.as_view(),
            &targets,
            multiclass_params(0).with_n_estimators(0),
        )
        .unwrap_err(),
        ModelError::InvalidEstimatorCount
    );
    assert_eq!(
        RandomForestClassifier::fit_multiclass(
            &x.as_view(),
            &targets,
            multiclass_params(0).with_min_samples_split(1),
        )
        .unwrap_err(),
        ModelError::InvalidMinSamplesSplit
    );
}

#[test]
fn separable_multiclass_regions_are_recovered() {
    let (x, y) = three_class_problem();
    let model =
        RandomForestClassifier::fit_multiclass(&x.as_view(), &y, multiclass_params(4)).unwrap();
    assert_eq!(model.predict(&x.as_view()).unwrap(), y.as_slice());
}
