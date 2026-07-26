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

use ferricml::api::{AnyClassifier, AnyRegressor};
use ferricml::artifact::{ModelArtifact, StageArtifact};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
};
use ferricml::linear_model::{ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use ferricml::ranking::PairwiseLinearRanker;
use ferricml::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

/// One frozen row: the type, its envelope kind, and its composition tag.
struct Row {
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
            model_row::<LogisticRegression>("LogisticRegression"),
            1,
            1,
            !LEGACY,
        ),
        (
            model_row::<LinearRegression>("LinearRegression"),
            2,
            2,
            !LEGACY,
        ),
        (model_row::<Ridge>("Ridge"), 3, 3, !LEGACY),
        (
            model_row::<PairwiseLinearRanker>("PairwiseLinearRanker"),
            8,
            8,
            !LEGACY,
        ),
        (
            model_row::<HistGradientBoostingRegressor>("HistGradientBoostingRegressor"),
            9,
            5,
            LEGACY,
        ),
        (
            model_row::<RandomForestRegressor>("RandomForestRegressor"),
            10,
            4,
            LEGACY,
        ),
        (
            model_row::<RandomForestClassifier>("RandomForestClassifier"),
            11,
            11,
            !LEGACY,
        ),
        (model_row::<AnyRegressor>("AnyRegressor"), 12, 12, !LEGACY),
        (model_row::<AnyClassifier>("AnyClassifier"), 13, 13, !LEGACY),
        (
            model_row::<HistGradientBoostingClassifier>("HistGradientBoostingClassifier"),
            20,
            20,
            !LEGACY,
        ),
        (
            model_row::<DecisionTreeRegressor>("DecisionTreeRegressor"),
            21,
            21,
            !LEGACY,
        ),
        (
            model_row::<DecisionTreeClassifier>("DecisionTreeClassifier"),
            22,
            22,
            !LEGACY,
        ),
        (
            model_row::<ExtraTreesRegressor>("ExtraTreesRegressor"),
            23,
            23,
            !LEGACY,
        ),
        (
            model_row::<ExtraTreesClassifier>("ExtraTreesClassifier"),
            24,
            24,
            !LEGACY,
        ),
        (model_row::<Lasso>("Lasso"), 69, 69, !LEGACY),
        (model_row::<ElasticNet>("ElasticNet"), 70, 70, !LEGACY),
    ]
}

/// Every schema-spanning type that persists, with its frozen numbers.
fn stage_rows() -> Vec<(Row, u16, u16, bool)> {
    vec![
        (stage_row::<StandardScaler>("StandardScaler"), 4, 1, LEGACY),
        (
            stage_row::<Pipeline<StandardScaler, LogisticRegression>>(
                "Pipeline<StandardScaler, LogisticRegression>",
            ),
            5,
            5,
            !LEGACY,
        ),
        (
            stage_row::<Pipeline<StandardScaler, LinearRegression>>(
                "Pipeline<StandardScaler, LinearRegression>",
            ),
            6,
            6,
            !LEGACY,
        ),
        (
            stage_row::<Pipeline<StandardScaler, Ridge>>("Pipeline<StandardScaler, Ridge>"),
            7,
            7,
            !LEGACY,
        ),
        (stage_row::<MinMaxScaler>("MinMaxScaler"), 14, 2, LEGACY),
        (stage_row::<MaxAbsScaler>("MaxAbsScaler"), 15, 3, LEGACY),
        (stage_row::<StagedTwo>("StagedPipeline"), 16, 16, !LEGACY),
        (stage_row::<RobustScaler>("RobustScaler"), 44, 4, LEGACY),
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
