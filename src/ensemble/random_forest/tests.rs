use super::*;
use crate::api::ModelError;
use crate::data::{BinaryTargets, DenseMatrix, RegressionTargets};
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
    let tree = &forest.trees[0];
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
    assert_eq!(stump.trees[0].nodes[0].threshold, 1.5);
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
    for tree in &forest.trees {
        assert!(!tree.nodes.is_empty() || tree.root_leaf.is_some());
        assert!(tree.root_leaf.is_none_or(f32::is_finite));
        for node in &tree.nodes {
            assert!(((node.feature_and_flags & FEATURE_MASK) as usize) < forest.n_features_in);
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
    assert_eq!(forest.trees[0].nodes.len(), ROWS - 1);
}
