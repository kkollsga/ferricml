//! Every artifact kind and composition tag, frozen by number.
//!
//! These two numbers are permanent names in the on-disk format. A kind decides
//! which reader a byte string reaches; a composition tag decides which
//! estimator or stage a composed payload claims to hold. Changing either
//! changes what previously written artifacts mean, and neither has a
//! compile-time reason to stay put — so they are written down here and reviewed
//! like a snapshot.
//!
//! # Why a table rather than a uniqueness assertion
//!
//! A tag now *derives* from its kind, which makes the derivation itself the
//! thing that could silently move. Asserting only that tags are unique would
//! accept a table where every tag had shifted together: still unique, and every
//! existing composed artifact silently reinterpreted. The numbers are therefore
//! spelled out, and uniqueness is checked on top of them rather than instead of
//! them.
//!
//! Six entries have a tag that differs from their kind, because they were
//! tagged before the derivation existed. Those six are the entire reason the
//! move to a derived tag did not rewrite anyone's bytes, so they are marked as
//! such and asserted to differ — an entry that quietly became equal to its kind
//! would mean the legacy table stopped being consulted.

mod support;

use support::api_profile::{self, MODEL_ARTIFACT, STAGE_ARTIFACT};

use ferricml::api::{AnyClassifier, AnyRegressor, HasCapabilities};
use ferricml::artifact::{ModelArtifact, StageArtifact};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
};
use ferricml::linear_model::{
    ElasticNet, ElasticNetParams, Lasso, LassoParams, LinearRegression, LogisticRegression, Ridge,
};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    MaxAbsScaler, MinMaxScaler, PolynomialFeatures, RobustScaler, StandardScaler,
};
use ferricml::ranking::PairwiseLinearRanker;
use ferricml::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

/// One frozen row: the type, its envelope kind, and its composition tag.
struct Row {
    /// The type, spelled as `tests/api-baselines/rust/ferricml-default.txt`
    /// spells it. The long form is what lets these rows be closed against the
    /// frozen API profile rather than against someone's memory.
    name: &'static str,
    kind: u16,
    tag: u16,
}

/// A tag that predates the derivation and therefore differs from its kind.
const LEGACY: bool = true;

fn model_row<M: ModelArtifact>(name: &'static str) -> Row {
    Row {
        name,
        kind: M::ARTIFACT_KIND,
        tag: M::MODEL_TAG,
    }
}

fn stage_row<S: StageArtifact>(name: &'static str) -> Row {
    Row {
        name,
        kind: S::ARTIFACT_KIND,
        tag: S::STAGE_TAG,
    }
}

/// The concrete composition the generic implementation is read through.
type StagedTwo = StagedPipeline<(MinMaxScaler, StandardScaler), Ridge>;

/// Every final estimator that persists, with its frozen numbers.
fn model_rows() -> Vec<(Row, u16, u16, bool)> {
    // (row, expected kind, expected tag, tag predates the derivation)
    vec![
        (
            model_row::<LogisticRegression>("ferricml::linear_model::LogisticRegression"),
            1,
            1,
            !LEGACY,
        ),
        (
            model_row::<LinearRegression>("ferricml::linear_model::LinearRegression"),
            2,
            2,
            !LEGACY,
        ),
        (
            model_row::<Ridge>("ferricml::linear_model::Ridge"),
            3,
            3,
            !LEGACY,
        ),
        (
            model_row::<PairwiseLinearRanker>("ferricml::ranking::PairwiseLinearRanker"),
            8,
            8,
            !LEGACY,
        ),
        (
            model_row::<HistGradientBoostingRegressor>(
                "ferricml::ensemble::HistGradientBoostingRegressor",
            ),
            9,
            5,
            LEGACY,
        ),
        (
            model_row::<RandomForestRegressor>("ferricml::ensemble::RandomForestRegressor"),
            10,
            4,
            LEGACY,
        ),
        (
            model_row::<RandomForestClassifier>("ferricml::ensemble::RandomForestClassifier"),
            11,
            11,
            !LEGACY,
        ),
        (
            model_row::<AnyRegressor>("ferricml::api::AnyRegressor"),
            12,
            12,
            !LEGACY,
        ),
        (
            model_row::<AnyClassifier>("ferricml::api::AnyClassifier"),
            13,
            13,
            !LEGACY,
        ),
        (
            model_row::<HistGradientBoostingClassifier>(
                "ferricml::ensemble::HistGradientBoostingClassifier",
            ),
            20,
            20,
            !LEGACY,
        ),
        (
            model_row::<DecisionTreeRegressor>("ferricml::tree::DecisionTreeRegressor"),
            21,
            21,
            !LEGACY,
        ),
        (
            model_row::<DecisionTreeClassifier>("ferricml::tree::DecisionTreeClassifier"),
            22,
            22,
            !LEGACY,
        ),
        (
            model_row::<ExtraTreesRegressor>("ferricml::ensemble::ExtraTreesRegressor"),
            23,
            23,
            !LEGACY,
        ),
        (
            model_row::<ExtraTreesClassifier>("ferricml::ensemble::ExtraTreesClassifier"),
            24,
            24,
            !LEGACY,
        ),
        (
            model_row::<Lasso>("ferricml::linear_model::Lasso"),
            69,
            69,
            !LEGACY,
        ),
        (
            model_row::<ElasticNet>("ferricml::linear_model::ElasticNet"),
            70,
            70,
            !LEGACY,
        ),
    ]
}

/// Every schema-spanning type that persists, with its frozen numbers.
fn stage_rows() -> Vec<(Row, u16, u16, bool)> {
    vec![
        (
            stage_row::<StandardScaler>("ferricml::preprocessing::StandardScaler"),
            4,
            1,
            LEGACY,
        ),
        (
            stage_row::<Pipeline<StandardScaler, LogisticRegression>>(
                "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, \
                 ferricml::linear_model::LogisticRegression>",
            ),
            5,
            5,
            !LEGACY,
        ),
        (
            stage_row::<Pipeline<StandardScaler, LinearRegression>>(
                "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, \
                 ferricml::linear_model::LinearRegression>",
            ),
            6,
            6,
            !LEGACY,
        ),
        (
            stage_row::<Pipeline<StandardScaler, Ridge>>(
                "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, \
             ferricml::linear_model::Ridge>",
            ),
            7,
            7,
            !LEGACY,
        ),
        (
            stage_row::<MinMaxScaler>("ferricml::preprocessing::MinMaxScaler"),
            14,
            2,
            LEGACY,
        ),
        (
            stage_row::<MaxAbsScaler>("ferricml::preprocessing::MaxAbsScaler"),
            15,
            3,
            LEGACY,
        ),
        (
            stage_row::<StagedTwo>(
                "ferricml::pipeline::StagedPipeline<(ferricml::preprocessing::MinMaxScaler, \
             ferricml::preprocessing::StandardScaler), ferricml::linear_model::Ridge>",
            ),
            16,
            16,
            !LEGACY,
        ),
        (
            stage_row::<PolynomialFeatures>("ferricml::preprocessing::PolynomialFeatures"),
            46,
            46,
            !LEGACY,
        ),
        (
            stage_row::<RobustScaler>("ferricml::preprocessing::RobustScaler"),
            44,
            4,
            LEGACY,
        ),
    ]
}

fn check(rows: Vec<(Row, u16, u16, bool)>, namespace: &str) {
    for (row, kind, tag, legacy) in &rows {
        assert_eq!(
            row.kind, *kind,
            "{}: artifact kind moved, which changes which reader its bytes reach",
            row.name
        );
        assert_eq!(
            row.tag, *tag,
            "{}: composition tag moved, which changes what a composed payload \
             claims to hold",
            row.name
        );
        if *legacy {
            assert_ne!(
                row.tag, row.kind,
                "{}: this tag predates the derivation and must stay different \
                 from its kind; equal means the legacy table stopped being \
                 consulted and existing composed artifacts have been \
                 reinterpreted",
                row.name
            );
        } else {
            assert_eq!(
                row.tag, row.kind,
                "{}: a tag that is not legacy is derived from its kind, so the \
                 two must agree; assigning one by hand is what the derivation \
                 exists to prevent",
                row.name
            );
        }
    }

    let mut kinds: Vec<u16> = rows.iter().map(|(row, ..)| row.kind).collect();
    kinds.sort_unstable();
    let unique = {
        let mut seen = kinds.clone();
        seen.dedup();
        seen.len()
    };
    assert_eq!(unique, kinds.len(), "{namespace}: two types share one kind");

    let mut tags: Vec<u16> = rows.iter().map(|(row, ..)| row.tag).collect();
    tags.sort_unstable();
    let unique_tags = {
        let mut seen = tags.clone();
        seen.dedup();
        seen.len()
    };
    assert_eq!(
        unique_tags,
        tags.len(),
        "{namespace}: two types share one composition tag, so one would decode \
         as the other inside a composition"
    );
}

#[test]
fn every_model_kind_and_composition_tag_is_the_frozen_number() {
    check(model_rows(), "final estimators");
}

#[test]
fn every_stage_kind_and_composition_tag_is_the_frozen_number() {
    check(stage_rows(), "schema-spanning types");
}

/// Kinds 17-19 are reserved and must never be reachable.
///
/// They were assigned to compositions a staged design made unnecessary. A kind
/// is a permanent name, so reusing one is the crossed-schema confusion the
/// adversarial corpus exists to catch — and the cheapest place to catch it is
/// before any bytes are written.
#[test]
fn the_reserved_kinds_are_claimed_by_nothing() {
    let claimed: Vec<u16> = model_rows()
        .iter()
        .map(|(row, ..)| row.kind)
        .chain(stage_rows().iter().map(|(row, ..)| row.kind))
        .collect();
    for reserved in 17..=19_u16 {
        assert!(
            !claimed.contains(&reserved),
            "kind {reserved} is reserved and was recycled"
        );
    }
}

// ---------------------------------------------------------------------------
// The two tables, closed against the frozen API profile
// ---------------------------------------------------------------------------

/// Every `ModelArtifact` implementation has a frozen kind and tag, and no row
/// outlives its impl.
///
/// The table above is hand-maintained. Until this test nothing closed it: a
/// seventeenth persisting estimator that correctly declared
/// `Capabilities::artifact` *and* implemented `ModelArtifact` would pass
/// `tests/capability_snapshot.rs` while its two permanent on-disk numbers were
/// frozen by nothing — free to move under it, silently reinterpreting every
/// composed artifact that already names them. That is the seven-estimator
/// defect's shape in a new place: a list someone has to remember to extend.
#[test]
fn every_model_artifact_impl_has_a_frozen_row_and_no_row_is_stale() {
    assert_closed(MODEL_ARTIFACT, &row_names(model_rows()), "artifact kinds");
}

/// The same, for the schema-spanning half of the persistence surface.
#[test]
fn every_stage_artifact_impl_has_a_frozen_row_and_no_row_is_stale() {
    assert_closed(STAGE_ARTIFACT, &row_names(stage_rows()), "stage kinds");
}

fn row_names(rows: Vec<(Row, u16, u16, bool)>) -> Vec<&'static str> {
    rows.iter().map(|(row, ..)| row.name).collect()
}

/// Both directions, because an impl with no frozen numbers and a frozen number
/// for a type that no longer persists are different defects.
fn assert_closed(trait_path: &str, names: &[&str], namespace: &str) {
    let closure = api_profile::close_against_persistence(&[trait_path], names);
    assert!(
        closure.unlisted.is_empty(),
        "{namespace}: these types implement {trait_path} and no row here \
         freezes their kind and composition tag: {:#?}",
        closure.unlisted
    );
    assert!(
        closure.stale.is_empty(),
        "{namespace}: these rows match no {trait_path} implementation, so they \
         freeze numbers nothing writes: {:#?}",
        closure.stale
    );
}

/// The closure must be able to fail, in both of its directions.
///
/// Driven through the same function the two tests above use, over a table
/// doctored the two ways it can go wrong. Without this the closure would prove
/// the tables are currently complete, not that incompleteness would be seen.
#[test]
fn the_kind_and_tag_closure_detects_a_missing_row_and_a_stale_one() {
    let complete = row_names(model_rows());
    assert!(
        complete.contains(&"ferricml::linear_model::Ridge"),
        "the row the missing-row half removes has to be there to remove"
    );

    let without_ridge: Vec<&str> = complete
        .iter()
        .copied()
        .filter(|name| *name != "ferricml::linear_model::Ridge")
        .collect();
    let missing = api_profile::close_against_persistence(&[MODEL_ARTIFACT], &without_ridge);
    assert_eq!(missing.unlisted, vec!["ferricml::linear_model::Ridge"]);
    assert!(missing.stale.is_empty(), "removing a row is not staleness");

    // `DummyRegressor` is the anchor for the other direction because its lack
    // of persistence is a documented decision rather than a gap: a baseline is
    // refitted, never restored. It can never own an artifact kind.
    let mut with_a_ghost = complete.clone();
    with_a_ghost.push("ferricml::dummy::DummyRegressor");
    let stale = api_profile::close_against_persistence(&[MODEL_ARTIFACT], &with_a_ghost);
    assert_eq!(stale.stale, vec!["ferricml::dummy::DummyRegressor"]);
    assert!(stale.unlisted.is_empty(), "adding a row hides nothing");

    // A stage row is not a model row: the two traits are closed separately, so
    // the model table cannot be satisfied by a `StageArtifact` implementation.
    let scaler = api_profile::close_against_persistence(
        &[MODEL_ARTIFACT],
        &["ferricml::preprocessing::StandardScaler"],
    );
    assert_eq!(
        scaler.stale,
        vec!["ferricml::preprocessing::StandardScaler"]
    );

    assert!(api_profile::close_against_persistence(&[MODEL_ARTIFACT], &complete).is_closed());
}

// ---------------------------------------------------------------------------
// The compositions the old mechanism excluded
// ---------------------------------------------------------------------------

/// Every estimator that could be saved alone but not inside a composition, now
/// composed and round-tripped.
///
/// These seven are the defect this contract was changed to fix. Each shipped a
/// working, fuzz-tested encoder and no membership in the composition list, so a
/// `StagedPipeline` ending in one could not persist *and* had no capability
/// declaration, which dropped it out of the conformance battery entirely.
/// Nothing anywhere noticed, because the two declarations were maintained
/// independently.
///
/// The assertions are deliberately about compositions rather than about the
/// estimators: persisting alone is what they could already do. A test naming
/// only `to_artifact` would have passed the whole time the defect existed.
mod compositions {
    use super::*;
    use ferricml::api::Estimator;
    use ferricml::data::{BinaryTargets, DenseMatrix, RegressionTargets};
    use ferricml::ensemble::{
        ExtraTreesClassifierParams, ExtraTreesRegressorParams,
        HistGradientBoostingClassifierParams, RandomForestClassifierParams,
    };
    use ferricml::preprocessing::{MinMaxScalerParams, StandardScalerParams};
    use ferricml::ranking::{
        PairIndex, PairOutcome, PairwiseLinearRankerParams, PairwiseObservation,
    };
    use ferricml::tree::MaxFeatures;
    use ferricml::tree::{DecisionTreeClassifierParams, DecisionTreeRegressorParams};

    const INPUT: [u8; 32] = [3; 32];
    const TRANSFORMED: [u8; 32] = [4; 32];

    fn data() -> DenseMatrix {
        DenseMatrix::new(
            vec![
                0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 7.0, 5.0, 6.0, 6.0, 5.0, 7.0, 4.0,
            ],
            8,
            2,
        )
        .unwrap()
    }

    fn regression() -> RegressionTargets {
        RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]).unwrap()
    }

    fn binary() -> BinaryTargets {
        BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1]).unwrap()
    }

    /// Fits a two-scaler composition over `estimator`, then asserts the whole
    /// composition declares persistence, round-trips, and decodes to itself.
    fn round_trips<E>(
        fit_estimator: impl FnOnce(&ferricml::data::MatrixView<'_>) -> E,
    ) -> StagedPipeline<(MinMaxScaler, StandardScaler), E>
    where
        E: Estimator + ModelArtifact + HasCapabilities + Clone + PartialEq + std::fmt::Debug,
    {
        let raw = data();
        let pipeline: StagedPipeline<(MinMaxScaler, StandardScaler), E> = StagedPipeline::fit(
            &raw.as_view(),
            |batch| MinMaxScaler::fit(batch, MinMaxScalerParams::default()),
            |batch| StandardScaler::fit(batch, StandardScalerParams::default()),
            |batch| Ok(fit_estimator(batch)),
        )
        .expect("the composition fits");

        assert!(
            <StagedPipeline<(MinMaxScaler, StandardScaler), E> as HasCapabilities>::CAPABILITIES
                .artifact(),
            "a composition over a persisting estimator must declare persistence"
        );

        let bytes = pipeline.to_artifact(INPUT, TRANSFORMED).expect("encode");
        assert_eq!(
            bytes,
            pipeline.to_artifact(INPUT, TRANSFORMED).expect("re-encode"),
            "the composition encodes deterministically"
        );
        let decoded = StagedPipeline::<(MinMaxScaler, StandardScaler), E>::from_artifact(
            &bytes,
            INPUT,
            TRANSFORMED,
        )
        .expect("decode");
        assert_eq!(decoded, pipeline);
        pipeline
    }

    #[test]
    fn a_composition_over_a_forest_classifier_persists() {
        round_trips(|batch| {
            RandomForestClassifier::fit(
                batch,
                &binary(),
                RandomForestClassifierParams::default()
                    .with_n_estimators(2)
                    .with_max_depth(Some(3))
                    .with_max_features(MaxFeatures::All)
                    .with_random_state(11),
            )
            .expect("fit")
        });
    }

    #[test]
    fn a_composition_over_the_randomized_ensembles_persists() {
        round_trips(|batch| {
            ExtraTreesRegressor::fit(
                batch,
                &regression(),
                ExtraTreesRegressorParams::default()
                    .with_n_estimators(2)
                    .with_max_depth(Some(3))
                    .with_max_features(MaxFeatures::All)
                    .with_random_state(11),
            )
            .expect("fit")
        });
        round_trips(|batch| {
            ExtraTreesClassifier::fit(
                batch,
                &binary(),
                ExtraTreesClassifierParams::default()
                    .with_n_estimators(2)
                    .with_max_depth(Some(3))
                    .with_max_features(MaxFeatures::All)
                    .with_random_state(11),
            )
            .expect("fit")
        });
    }

    #[test]
    fn a_composition_over_the_boosted_classifier_persists() {
        round_trips(|batch| {
            HistGradientBoostingClassifier::fit(
                batch,
                &binary(),
                HistGradientBoostingClassifierParams::default()
                    .with_max_iter(2)
                    .with_max_leaf_nodes(4)
                    .with_min_samples_leaf(1)
                    .with_max_bins(8),
            )
            .expect("fit")
        });
    }

    #[test]
    fn a_composition_over_the_standalone_trees_persists() {
        round_trips(|batch| {
            DecisionTreeRegressor::fit(
                batch,
                &regression(),
                DecisionTreeRegressorParams::default()
                    .with_max_depth(Some(3))
                    .with_max_features(MaxFeatures::All)
                    .with_random_state(11),
            )
            .expect("fit")
        });
        round_trips(|batch| {
            DecisionTreeClassifier::fit(
                batch,
                &binary(),
                DecisionTreeClassifierParams::default()
                    .with_max_depth(Some(3))
                    .with_max_features(MaxFeatures::All)
                    .with_random_state(11),
            )
            .expect("fit")
        });
    }

    #[test]
    fn a_composition_over_the_pairwise_ranker_persists() {
        let pair = |left, right| {
            PairwiseObservation::new(
                PairIndex::new(left, right).unwrap(),
                PairOutcome::LeftPreferred,
                1.0,
            )
            .unwrap()
        };
        round_trips(|batch| {
            PairwiseLinearRanker::fit(
                batch,
                &[pair(7, 6), pair(6, 5), pair(5, 4), pair(3, 2)],
                PairwiseLinearRankerParams::default(),
            )
            .expect("fit")
        });
    }

    /// The penalized pair, which had no persistence at all until this phase.
    ///
    /// They cost one codec each and nothing else — no tag, no list — which is
    /// what makes them the check that the contract is structural rather than
    /// seven instances repaired.
    #[test]
    fn a_composition_over_the_penalized_regressors_persists() {
        round_trips(|batch| {
            Lasso::fit(batch, &regression(), LassoParams::default().with_alpha(0.1)).expect("fit")
        });
        round_trips(|batch| {
            ElasticNet::fit(
                batch,
                &regression(),
                ElasticNetParams::default()
                    .with_alpha(0.1)
                    .with_l1_ratio(0.5),
            )
            .expect("fit")
        });
    }
}
