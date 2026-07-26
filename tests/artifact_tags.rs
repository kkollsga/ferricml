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
        HistGradientBoostingClassifierParams, MaxFeatures, RandomForestClassifierParams,
    };
    use ferricml::preprocessing::{MinMaxScalerParams, StandardScalerParams};
    use ferricml::ranking::{
        PairIndex, PairOutcome, PairwiseLinearRankerParams, PairwiseObservation,
    };
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
