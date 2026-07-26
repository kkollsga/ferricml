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
    let (x, _, labels) = separable();
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
    //
    // The claim needs data where the bound *binds*, which is what this test
    // previously lacked: on `separable()` the optimal tree already has
    // three-row leaves, so `min_samples_leaf(2)` refused nothing and the
    // equality below would have held under row counting too. Here the weighted
    // row is the only one carrying its target, so reproducing that target
    // requires a leaf holding one row and weight three — admissible under the
    // weight bound and inadmissible under a row bound of two.
    let bound_x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0]]);
    let bound_view = bound_x.as_view();
    let bound_y = vec![0.0, 0.0, 9.0, 1.0, 1.0];
    let params = regressor_params().with_min_samples_leaf(2);
    let weighted = DecisionTreeRegressor::fit_weighted(
        &bound_view,
        &RegressionTargets::new(bound_y.clone()).unwrap(),
        &SampleWeights::new(vec![1.0, 1.0, 3.0, 1.0, 1.0]).unwrap(),
        params.clone(),
    )
    .unwrap();

    // Non-vacuity: the one-row, weight-three leaf really is formed. Under a row
    // bound it is inadmissible, its parent stays whole, and every row in that
    // parent reads one shared weighted mean instead of its own target.
    assert_eq!(weighted.predict(&bound_view).unwrap(), bound_y);

    // And the guard above is a guard: raise the bound past the row's weight and
    // the same fit loses that leaf, so the assertion is testing the bound
    // rather than restating that a tree can fit five points.
    let blocked = DecisionTreeRegressor::fit_weighted(
        &bound_view,
        &RegressionTargets::new(bound_y.clone()).unwrap(),
        &SampleWeights::new(vec![1.0, 1.0, 3.0, 1.0, 1.0]).unwrap(),
        regressor_params().with_min_samples_leaf(4),
    )
    .unwrap();
    assert_ne!(blocked.predict(&bound_view).unwrap(), bound_y);

    let mut repeated_rows: Vec<Vec<f32>> = Vec::new();
    let mut repeated_targets = Vec::new();
    for (index, row) in bound_view.iter_rows().enumerate() {
        let copies = if index == 2 { 3 } else { 1 };
        for _ in 0..copies {
            repeated_rows.push(row.to_vec());
            repeated_targets.push(bound_y[index]);
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
        weighted.predict(&bound_view).unwrap(),
        expanded.predict(&bound_view).unwrap()
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

/// The randomized splitter still fits a usable tree, and it is a different
/// tree from the exhaustive one.
///
/// Both halves matter: a `Splitter` that silently fell back to the exhaustive
/// search would pass every other test in this file.
#[test]
fn the_randomized_splitter_fits_a_different_tree_that_still_separates() {
    let (x, y, labels) = separable();
    let view = x.as_view();
    let random = classifier_params().with_splitter(Splitter::Random);
    assert_eq!(random.splitter(), Splitter::Random);
    assert_eq!(classifier_params().splitter(), Splitter::Best);

    let targets = BinaryTargets::new(labels.clone()).unwrap();
    let randomized = DecisionTreeClassifier::fit(&view, &targets, random.clone()).unwrap();
    let exhaustive = DecisionTreeClassifier::fit(&view, &targets, classifier_params()).unwrap();
    assert_eq!(randomized.predict(&view).unwrap(), labels);
    assert_ne!(randomized, exhaustive);

    let values = RegressionTargets::new(y).unwrap();
    let randomized = DecisionTreeRegressor::fit(
        &view,
        &values,
        regressor_params().with_splitter(Splitter::Random),
    )
    .unwrap();
    let exhaustive = DecisionTreeRegressor::fit(&view, &values, regressor_params()).unwrap();
    assert_ne!(randomized, exhaustive);
    assert!(
        randomized
            .predict(&view)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    // The multiclass builder takes the same two arms, so it owes the same
    // proof: the flavour must not quietly keep optimizing within a column.
    let classes = ClassTargets::new(vec![3, 3, 7, 7, 10, 10]).unwrap();
    assert_ne!(
        DecisionTreeClassifier::fit_multiclass(&view, &classes, random).unwrap(),
        DecisionTreeClassifier::fit_multiclass(&view, &classes, classifier_params()).unwrap()
    );
}

/// A randomized fit is as reproducible as an exhaustive one, and the artifact
/// carries the setting that produced it.
#[test]
fn a_randomized_fit_is_reproducible_and_its_splitter_survives_a_round_trip() {
    let (x, y, _) = separable();
    let view = x.as_view();
    let params = regressor_params().with_splitter(Splitter::Random);
    let targets = RegressionTargets::new(y).unwrap();
    let first = DecisionTreeRegressor::fit(&view, &targets, params.clone()).unwrap();
    let second = DecisionTreeRegressor::fit(&view, &targets, params).unwrap();
    assert_eq!(first, second);

    let bytes = first.to_artifact(SCHEMA).unwrap();
    let restored = DecisionTreeRegressor::from_artifact(&bytes, SCHEMA).unwrap();
    assert_eq!(restored.get_params().splitter(), Splitter::Random);
    assert_eq!(restored, first);
    assert_eq!(restored.to_artifact(SCHEMA).unwrap(), bytes);

    // The two settings are different models under one artifact kind, so the
    // stored tag has to be what tells them apart rather than the topology.
    let exhaustive = DecisionTreeRegressor::fit(&view, &targets, regressor_params()).unwrap();
    assert_ne!(exhaustive.to_artifact(SCHEMA).unwrap(), bytes);
}

/// An inadmissible draw is discarded, never redrawn.
///
/// One column holding `0..19` at `min_samples_leaf = 8` admits a partition only
/// for thresholds in `[7, 12)` — five of the nineteen gaps, about 26% of the
/// draw range. Redrawing until an admissible threshold appeared would make
/// *every* seed split; discarding makes the split rate track the admissible
/// share, and bounds the work a node with a tiny admissible region can cost.
#[test]
fn an_inadmissible_random_draw_is_discarded_rather_than_redrawn() {
    let data = DenseMatrix::new((0..20).map(|value| value as f32).collect(), 20, 1).unwrap();
    let view = data.as_view();
    let targets = RegressionTargets::new((0..20).map(|value| value as f32).collect()).unwrap();

    let mut split = 0;
    let seeds = 400;
    for seed in 0..seeds {
        let model = DecisionTreeRegressor::fit(
            &view,
            &targets,
            DecisionTreeRegressorParams::default()
                .with_splitter(Splitter::Random)
                .with_max_depth(Some(1))
                .with_min_samples_leaf(8)
                .with_random_state(seed),
        )
        .unwrap();
        let predictions = model.predict(&view).unwrap();
        let left = predictions.iter().filter(|&&v| v == predictions[0]).count();
        if left == predictions.len() {
            continue;
        }
        split += 1;
        // Both children respect the leaf bound, so an accepted draw really was
        // admissible rather than nudged into admissibility.
        assert!(
            (8..=12).contains(&left),
            "seed {seed} produced a left child of {left} rows"
        );
    }
    assert!(
        (seeds / 8..seeds / 2).contains(&split),
        "{split} of {seeds} seeds split; redrawing would make it {seeds}"
    );
}

/// A drawn column that is constant inside the node **consumes** the quota.
///
/// This is a recorded divergence, and it is asserted here rather than left to
/// the contract table because it is invisible in the parameter region the
/// existing fixtures use: at `MaxFeatures::All` every column is drawn anyway,
/// so the quota question never arises and a change to this rule would move
/// nothing that is currently checked.
///
/// The reference skips a constant column and keeps drawing, so with one
/// informative column beside nineteen constant ones at a quota of one it splits
/// on the informative column every time. FerricML draws one column and stops,
/// so it splits only when the informative column is the one drawn — about one
/// seed in twenty. A test asserting the reference's behaviour would fail, and
/// correctly.
#[test]
fn a_constant_column_consumes_the_feature_quota_unlike_the_reference() {
    let rows = 24_usize;
    let columns = 20_usize;
    let mut values = vec![0.5_f32; rows * columns];
    for row in 0..rows {
        values[row * columns] = row as f32;
    }
    let data = DenseMatrix::new(values, rows, columns).unwrap();
    let view = data.as_view();
    let targets =
        RegressionTargets::new((0..rows).map(|row| f32::from(row >= 12)).collect()).unwrap();

    let seeds = 400_u64;
    let mut split = 0;
    for seed in 0..seeds {
        let model = DecisionTreeRegressor::fit(
            &view,
            &targets,
            DecisionTreeRegressorParams::default()
                .with_max_features(MaxFeatures::Count(1))
                .with_max_depth(Some(1))
                .with_random_state(seed),
        )
        .unwrap();
        let predictions = model.predict(&view).unwrap();
        if predictions.iter().any(|&value| value != predictions[0]) {
            split += 1;
        }
    }
    // The reference's rule would give `seeds`; ours gives roughly `seeds / 20`.
    assert!(
        split > 0 && split < seeds / 4,
        "{split} of {seeds} seeds split; the reference's skip-and-redraw rule          would give {seeds}"
    );
}

/// Two values are distinct by **exact** comparison.
///
/// The other recorded divergence the existing fixtures cannot show, for the
/// same reason: near-duplicate values never appear in them. The reference
/// treats two values as the same unless they are separated by roughly `1e-7`
/// absolute, so at magnitude 1.0 a column whose classes are separated by one
/// `f32` ulp yields a depth-0 leaf there. FerricML splits it.
#[test]
fn adjacent_float_values_are_distinct_unlike_the_reference() {
    let low = 1.0_f32;
    let high = f32::from_bits(low.to_bits() + 1);
    assert!(
        high - low < 2.0e-7,
        "the gap must be inside the reference's rule"
    );
    let data = matrix(&[&[low], &[low], &[high], &[high]]);
    let view = data.as_view();
    let model = DecisionTreeRegressor::fit(
        &view,
        &RegressionTargets::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap(),
        regressor_params(),
    )
    .unwrap();
    // A depth-0 leaf would predict the mean everywhere; a genuine split
    // reproduces the step. The reference produces the former.
    assert_eq!(model.predict(&view).unwrap(), vec![0.0, 0.0, 1.0, 1.0]);
}

/// Among exactly-tied splits the **first drawn** column wins — not the
/// lowest-indexed one.
///
/// This is the crate's cross-column tie-break, and until now nothing asserted
/// it. `TIE_TRAIN_X` is a single column, so a *cross*-column rule cannot show
/// in it at all, and the whole reference-semantics suite passes unchanged when
/// the rule is replaced by lowest-column-index. Two tests do notice, both
/// incidentally: `the_identity_is_of_member_zero_and_not_of_the_public_seed`
/// loses the *inequality* it asserts once every tree picks the same column,
/// and the frozen adversarial artifact corpus is byte-frozen. Neither states
/// the rule, so neither tells a reader what changed.
///
/// The construction gives the rule something to decide: four bit-identical
/// columns produce the same sorted order, the same threshold and the same
/// score, so every candidate ties exactly and the winner is decided purely by
/// visit order. `MaxFeatures::Count(1)` draws exactly one column and consumes
/// the generator identically to the first draw of `MaxFeatures::All`, so it
/// names the first drawn column over a path where no tie-break can be
/// involved — the rule is checked against the draw itself, not against a
/// second copy of the draw written in the test.
#[test]
fn the_first_drawn_column_wins_an_exactly_tied_split() {
    const COLUMNS: usize = 4;
    let column = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data = DenseMatrix::new(
        column.iter().flat_map(|&v| [v; COLUMNS]).collect(),
        column.len(),
        COLUMNS,
    )
    .unwrap();
    let targets = RegressionTargets::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    let root_split_column = |max_features: MaxFeatures, seed: u64| {
        let model = DecisionTreeRegressor::fit(
            &data.as_view(),
            &targets,
            DecisionTreeRegressorParams::default()
                .with_max_features(max_features)
                .with_max_depth(Some(1))
                .with_random_state(seed),
        )
        .unwrap();
        let packed = model.packed();
        assert!(packed.root_leaf.is_none(), "seed {seed} grew no split");
        (packed.nodes[0].feature_and_flags & FEATURE_MASK) as usize
    };

    let mut winners = [0_u32; COLUMNS];
    for seed in 0..64 {
        let first_drawn = root_split_column(MaxFeatures::Count(1), seed);
        assert_eq!(
            root_split_column(MaxFeatures::All, seed),
            first_drawn,
            "seed {seed} broke an exact tie somewhere other than the first drawn column"
        );
        winners[first_drawn] += 1;
    }
    // The half that makes the assertion above capable of failing: the draw
    // really does put a column other than column 0 first, so lowest-column-index
    // tie-breaking would disagree rather than coincide. Every column must win
    // somewhere, or the permutation would be closer to the identity than the
    // rule claims.
    assert!(
        winners.iter().all(|&count| count > 0),
        "the seeded draw never put some column first ({winners:?}); \
         a lowest-index rule would be indistinguishable"
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
        model.predict_positive_proba_one(&[0.0, 0.0]),
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
