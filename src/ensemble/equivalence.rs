//! The sprint's headline property: an ensemble of one tree **is** that tree.
//!
//! Two claims live here, deliberately separated so neither can be mistaken for
//! the other.
//!
//! **FerricML's own bar** is *identical structure under the same derived seed*,
//! asserted bit-for-bit on the packed layout. It holds by construction rather
//! than by luck: a forest never hands its public `random_state` to a member, it
//! derives `derive_tree_seed(random_state, index)`, and a standalone tree seeds
//! itself with `derive_tree_seed(random_state, 0)` — the derivation for member
//! zero. Both then enter one grower, under one configuration type, with the
//! generator in the same state, because an unbootstrapped sample consumes no
//! randomness. If this ever fails it is a real seed-derivation difference and a
//! finding, not a tolerance to widen.
//!
//! **The reference's weaker property** is recorded separately below: a
//! one-tree ensemble and a standalone tree agree wherever the best split is
//! unique, and may differ only among exactly-tied splits. FerricML exceeds it,
//! because its cross-column tie-break is reproducible from data, parameters,
//! and seed rather than from an unobservable random permutation. Keeping the
//! two tests apart is what stops the stronger claim from being read as the
//! weaker one.

use crate::data::{BinaryTargets, ClassTargets, DenseMatrix, RegressionTargets};
use crate::ensemble::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, ExtraTreesRegressor,
    ExtraTreesRegressorParams, MaxFeatures, RandomForestClassifier, RandomForestClassifierParams,
    RandomForestRegressor, RandomForestRegressorParams,
};
use crate::tree::{
    DecisionTreeClassifier, DecisionTreeClassifierParams, DecisionTreeRegressor,
    DecisionTreeRegressorParams, Splitter,
};

const SEED: u64 = 19;

/// Continuous data with ties at deep nodes — the case where the reference's
/// forest-of-one and standalone tree diverge in 0 of 50 seeds.
fn tie_heavy() -> (DenseMatrix, RegressionTargets, BinaryTargets, ClassTargets) {
    let rows = 60_usize;
    let columns = 5_usize;
    let mut values = Vec::with_capacity(rows * columns);
    let mut targets = Vec::with_capacity(rows);
    let mut labels = Vec::with_capacity(rows);
    let mut classes = Vec::with_capacity(rows);
    for row in 0..rows {
        for column in 0..columns {
            values.push((((row * 37 + column * 11) % 23) as f32) / 4.0 - 3.0);
        }
        targets.push(((row % 7) as f32) - 3.0);
        labels.push(u8::from(row % 3 == 0));
        classes.push([3_u8, 7, 10][row % 3]);
    }
    (
        DenseMatrix::new(values, rows, columns).unwrap(),
        RegressionTargets::new(targets).unwrap(),
        BinaryTargets::new(labels).unwrap(),
        ClassTargets::new(classes).unwrap(),
    )
}

/// A clean step function of one column: the best split is unique at every node.
fn unique_optimum() -> (DenseMatrix, RegressionTargets, BinaryTargets) {
    let rows = 32_usize;
    let mut values = Vec::with_capacity(rows * 2);
    let mut targets = Vec::with_capacity(rows);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        values.push(row as f32);
        values.push((row * 3) as f32);
        targets.push(if row < 8 {
            0.0
        } else if row < 20 {
            5.0
        } else {
            11.0
        });
        labels.push(u8::from(row >= 16));
    }
    (
        DenseMatrix::new(values, rows, 2).unwrap(),
        RegressionTargets::new(targets).unwrap(),
        BinaryTargets::new(labels).unwrap(),
    )
}

fn one_tree_forest_regressor() -> RandomForestRegressorParams {
    RandomForestRegressorParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All)
        .with_random_state(SEED)
}

fn one_tree_forest_classifier() -> RandomForestClassifierParams {
    RandomForestClassifierParams::default()
        .with_n_estimators(1)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::All)
        .with_random_state(SEED)
}

fn standalone_regressor() -> DecisionTreeRegressorParams {
    DecisionTreeRegressorParams::default()
        .with_max_features(MaxFeatures::All)
        .with_random_state(SEED)
}

fn standalone_classifier() -> DecisionTreeClassifierParams {
    DecisionTreeClassifierParams::default()
        .with_max_features(MaxFeatures::All)
        .with_random_state(SEED)
}

/// FerricML's bar, through the regressor.
#[test]
fn a_forest_of_one_and_a_standalone_tree_are_bitwise_identical() {
    let (x, y, _, _) = tie_heavy();
    let view = x.as_view();
    let forest = RandomForestRegressor::fit(&view, &y, one_tree_forest_regressor()).unwrap();
    let tree = DecisionTreeRegressor::fit(&view, &y, standalone_regressor()).unwrap();
    assert_eq!(forest.core.trees.len(), 1);
    assert_eq!(
        &forest.core.trees[0],
        tree.packed(),
        "the shared grower produced two different trees from one derived seed"
    );
}

/// The same bar through **both** of the forest classifier's fitting entry
/// points, because they run different builders.
///
/// A proof against only the scalar-leaf path would leave the multiclass
/// builder — a genuinely separate sweep — unproven, which is exactly the half
/// that could drift without anything failing.
#[test]
fn both_classifier_leaf_flavours_match_their_forest_of_one() {
    let (x, _, labels, classes) = tie_heavy();
    let view = x.as_view();

    let forest = RandomForestClassifier::fit(&view, &labels, one_tree_forest_classifier()).unwrap();
    let tree = DecisionTreeClassifier::fit(&view, &labels, standalone_classifier()).unwrap();
    assert_eq!(forest.binary_trees().len(), 1);
    assert_eq!(&forest.binary_trees()[0], tree.packed_binary());
    assert_eq!(forest.classes(), tree.classes());

    let forest =
        RandomForestClassifier::fit_multiclass(&view, &classes, one_tree_forest_classifier())
            .unwrap();
    let tree =
        DecisionTreeClassifier::fit_multiclass(&view, &classes, standalone_classifier()).unwrap();
    let crate::ensemble::forest::model::Forest::Multiclass(trees) = &forest.core.forest else {
        unreachable!("multiclass fixture");
    };
    assert_eq!(trees.len(), 1);
    assert_eq!(&trees[0], tree.packed_multiclass());
    assert_eq!(forest.classes(), tree.classes());
}

/// The same bar for the randomized family.
///
/// This one also proves the randomized search consumes the generator
/// identically on both sides: a single extra or missing draw anywhere in the
/// grow would move every threshold after it.
#[test]
fn an_extra_trees_of_one_and_a_randomized_standalone_tree_are_bitwise_identical() {
    let (x, y, labels, _) = tie_heavy();
    let view = x.as_view();

    let ensemble = ExtraTreesRegressor::fit(
        &view,
        &y,
        ExtraTreesRegressorParams::default()
            .with_n_estimators(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(SEED),
    )
    .unwrap();
    let tree = DecisionTreeRegressor::fit(
        &view,
        &y,
        standalone_regressor().with_splitter(Splitter::Random),
    )
    .unwrap();
    assert_eq!(&ensemble.core.trees[0], tree.packed());

    let ensemble = ExtraTreesClassifier::fit(
        &view,
        &labels,
        ExtraTreesClassifierParams::default()
            .with_n_estimators(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(SEED),
    )
    .unwrap();
    let tree = DecisionTreeClassifier::fit(
        &view,
        &labels,
        standalone_classifier().with_splitter(Splitter::Random),
    )
    .unwrap();
    assert_eq!(&ensemble.binary_trees()[0], tree.packed_binary());
}

/// The recorded divergence, asserted rather than assumed.
///
/// A random forest and an extra-trees of one member, both unbootstrapped and
/// considering every column, must **not** agree — the split search is the whole
/// difference between the families, and a test that only ever compared
/// like-for-like would pass just as happily if `Splitter::Random` silently fell
/// back to the exhaustive sweep.
#[test]
fn the_two_families_do_not_collapse_into_each_other() {
    let (x, y, _, _) = tie_heavy();
    let view = x.as_view();
    let forest = RandomForestRegressor::fit(&view, &y, one_tree_forest_regressor()).unwrap();
    let randomized = ExtraTreesRegressor::fit(
        &view,
        &y,
        ExtraTreesRegressorParams::default()
            .with_n_estimators(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(SEED),
    )
    .unwrap();
    assert_ne!(forest.core.trees[0], randomized.core.trees[0]);
}

/// The reference's weaker property, stated as its own test.
///
/// Against the reference a one-tree forest and a standalone tree are identical
/// only where the optimum is unique (30/30 on depth-limited or step-function
/// data) and diverge on generic fully grown data (0/50) — entirely through
/// exactly-tied splits, at which the two picks score bit-for-bit equally.
///
/// FerricML satisfies the weak property here, and the test above shows it also
/// satisfies the strong one on tie-heavy data the reference fails. The two are
/// separate tests so the stronger claim is never read as evidence for the
/// weaker one, or the reverse.
#[test]
fn the_reference_property_holds_where_the_optimum_is_unique() {
    let (x, y, labels) = unique_optimum();
    let view = x.as_view();

    let forest = RandomForestRegressor::fit(&view, &y, one_tree_forest_regressor()).unwrap();
    let tree = DecisionTreeRegressor::fit(&view, &y, standalone_regressor()).unwrap();
    assert_eq!(forest.predict(&view).unwrap(), tree.predict(&view).unwrap());

    let forest = RandomForestClassifier::fit(&view, &labels, one_tree_forest_classifier()).unwrap();
    let tree = DecisionTreeClassifier::fit(&view, &labels, standalone_classifier()).unwrap();
    assert_eq!(forest.predict(&view).unwrap(), tree.predict(&view).unwrap());
    assert_eq!(
        forest.predict_proba(&view).unwrap(),
        tree.predict_proba(&view).unwrap()
    );

    // Depth-limited fits are the reference's other 30/30 case, and the seed
    // must not matter where no node has a tie to break.
    for seed in [0_u64, 1, 2, 3, 4] {
        let forest = RandomForestRegressor::fit(
            &view,
            &y,
            one_tree_forest_regressor()
                .with_random_state(seed)
                .with_max_depth(Some(2)),
        )
        .unwrap();
        let tree = DecisionTreeRegressor::fit(
            &view,
            &y,
            standalone_regressor()
                .with_random_state(seed)
                .with_max_depth(Some(2)),
        )
        .unwrap();
        assert_eq!(&forest.core.trees[0], tree.packed(), "seed {seed}");
    }
}

/// The seed derivation is what makes the identity structural, so it is asserted
/// rather than left to the two sites' comments.
///
/// A standalone tree at public seed `r` matches member **zero** of a forest at
/// public seed `r` — not a forest member at some other index, and not a
/// standalone tree at the raw seed. Comparing against member one is what would
/// catch a derivation that had quietly become the identity function.
#[test]
fn the_identity_is_of_member_zero_and_not_of_the_public_seed() {
    let (x, y, _, _) = tie_heavy();
    let view = x.as_view();
    let forest =
        RandomForestRegressor::fit(&view, &y, one_tree_forest_regressor().with_n_estimators(3))
            .unwrap();
    let tree = DecisionTreeRegressor::fit(&view, &y, standalone_regressor()).unwrap();
    assert_eq!(&forest.core.trees[0], tree.packed());
    assert_ne!(&forest.core.trees[1], tree.packed());
    assert_ne!(&forest.core.trees[2], tree.packed());
}
