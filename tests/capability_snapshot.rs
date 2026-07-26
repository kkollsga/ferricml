//! The exact declared capability values, as a reviewable snapshot.
//!
//! `cargo-public-api` records *that* a type declares capabilities and never
//! *which*: the frozen API baseline contains
//! `pub const Ridge::CAPABILITIES: Capabilities` with no value, so flipping
//! `sample_weights(true)` to `sample_weights(false)` is invisible to
//! `make api-check`. That is a property of the tool, not a gap to widen the
//! tool through — a const value is not part of the surface it can see.
//!
//! So the values get their own companion snapshot, generated the way the API
//! profile is: `tests/api-baselines/rust/ferricml-capabilities.txt` sits beside
//! the API baseline, and a declaration flip lands as a diff in the directory a
//! reviewer reads as "the public API changed". It is also an ordinary
//! integration test, so the flip fails `make gate` too.
//!
//! # What makes the snapshot complete
//!
//! A snapshot listing whatever someone remembered to add is not a contract.
//! The completeness check closes it against the thing the tool *can* see:
//! every `impl … HasCapabilities for …` line in the API baseline must be
//! covered by at least one row here, and every row must correspond to such a
//! line. A new estimator therefore cannot declare capabilities without
//! appearing in both files, and a stale row cannot outlive its impl.
//!
//! # Where this sits among the three mechanisms
//!
//! - This file is the **change detector**: exact values, mechanically diffed.
//! - `tests/estimator_capabilities.rs` is the **reasoned record**: why each
//!   declaration is what it is, which a generated table cannot carry.
//! - The conformance battery is the **behavioral proof**: that the declaration
//!   and the estimator agree.
//!
//! One does not substitute for another.
//!
//! # Refreshing
//!
//! `make api-refresh` rewrites the snapshot; `make api-check` compares it.
//! Setting `FERRICML_REFRESH_CAPABILITY_SNAPSHOT=1` does the same directly.
//! Review the delta in the commit that causes it.

use std::fs;
use std::path::PathBuf;

mod support;

use support::api_profile::{
    MODEL_ARTIFACT, PersistenceImpl, STAGE_ARTIFACT, baseline_dir, covers, impl_target,
    persistence_impl, persistence_impls, profile_lines,
};

use ferricml::api::{AnyClassifier, AnyRegressor, Capabilities, HasCapabilities};
use ferricml::calibration::{CalibratedClassifier, IsotonicRegression, PlattCalibrator};
use ferricml::dummy::{DummyClassifier, DummyRegressor};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
};
use ferricml::linear_model::{ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{
    Binarizer, FunctionTransformer, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler,
    StandardScaler,
};
use ferricml::ranking::PairwiseLinearRanker;
use ferricml::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

/// The environment variable that rewrites the snapshot instead of checking it.
const REFRESH: &str = "FERRICML_REFRESH_CAPABILITY_SNAPSHOT";

/// Capability names, in the order the descriptor declares its fields.
///
/// Declaration order rather than alphabetical, so a row reads like the type it
/// describes. What matters is that it is stable: reordering would rewrite every
/// row and drown the one that changed.
fn capability_names(capabilities: Capabilities) -> Vec<&'static str> {
    let mut names = Vec::new();
    if capabilities.sample_weights() {
        names.push("sample_weights");
    }
    if capabilities.artifact() {
        names.push("artifact");
    }
    if capabilities.multiclass() {
        names.push("multiclass");
    }
    if capabilities.decision_function() {
        names.push("decision_function");
    }
    if capabilities.probability() {
        names.push("probability");
    }
    names
}

/// One snapshot row: the type as the API baseline spells it, and its value.
///
/// Generic declaration sites are represented by a named concrete
/// instantiation, which is the only form that has a value at all.
fn declarations() -> Vec<(&'static str, Capabilities)> {
    let mut rows: Vec<(&'static str, Capabilities)> = vec![
        ("ferricml::api::AnyClassifier", AnyClassifier::CAPABILITIES),
        ("ferricml::api::AnyRegressor", AnyRegressor::CAPABILITIES),
        (
            "ferricml::calibration::CalibratedClassifier<ferricml::ensemble::RandomForestClassifier, ferricml::calibration::IsotonicRegression>",
            <CalibratedClassifier<RandomForestClassifier, IsotonicRegression> as HasCapabilities>::CAPABILITIES,
        ),
        (
            "ferricml::calibration::CalibratedClassifier<ferricml::ensemble::RandomForestClassifier, ferricml::calibration::PlattCalibrator>",
            <CalibratedClassifier<RandomForestClassifier, PlattCalibrator> as HasCapabilities>::CAPABILITIES,
        ),
        (
            "ferricml::calibration::IsotonicRegression",
            IsotonicRegression::CAPABILITIES,
        ),
        (
            "ferricml::dummy::DummyClassifier",
            DummyClassifier::CAPABILITIES,
        ),
        (
            "ferricml::dummy::DummyRegressor",
            DummyRegressor::CAPABILITIES,
        ),
        (
            "ferricml::ensemble::HistGradientBoostingClassifier",
            HistGradientBoostingClassifier::CAPABILITIES,
        ),
        (
            "ferricml::ensemble::HistGradientBoostingRegressor",
            HistGradientBoostingRegressor::CAPABILITIES,
        ),
        (
            "ferricml::ensemble::RandomForestClassifier",
            RandomForestClassifier::CAPABILITIES,
        ),
        (
            "ferricml::ensemble::ExtraTreesClassifier",
            ExtraTreesClassifier::CAPABILITIES,
        ),
        (
            "ferricml::ensemble::ExtraTreesRegressor",
            ExtraTreesRegressor::CAPABILITIES,
        ),
        (
            "ferricml::tree::DecisionTreeClassifier",
            DecisionTreeClassifier::CAPABILITIES,
        ),
        (
            "ferricml::tree::DecisionTreeRegressor",
            DecisionTreeRegressor::CAPABILITIES,
        ),
        (
            "ferricml::ensemble::RandomForestRegressor",
            RandomForestRegressor::CAPABILITIES,
        ),
        ("ferricml::linear_model::ElasticNet", ElasticNet::CAPABILITIES),
        ("ferricml::linear_model::Lasso", Lasso::CAPABILITIES),
        (
            "ferricml::linear_model::LinearRegression",
            LinearRegression::CAPABILITIES,
        ),
        (
            "ferricml::linear_model::LogisticRegression",
            LogisticRegression::CAPABILITIES,
        ),
        ("ferricml::linear_model::Ridge", Ridge::CAPABILITIES),
        (
            "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, ferricml::linear_model::LinearRegression>",
            <Pipeline<StandardScaler, LinearRegression> as HasCapabilities>::CAPABILITIES,
        ),
        (
            "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, ferricml::linear_model::LogisticRegression>",
            <Pipeline<StandardScaler, LogisticRegression> as HasCapabilities>::CAPABILITIES,
        ),
        (
            "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, ferricml::linear_model::Ridge>",
            <Pipeline<StandardScaler, Ridge> as HasCapabilities>::CAPABILITIES,
        ),
        (
            "ferricml::pipeline::StagedPipeline<(ferricml::preprocessing::MinMaxScaler, ferricml::preprocessing::StandardScaler), ferricml::linear_model::Ridge>",
            <StagedPipeline<(MinMaxScaler, StandardScaler), Ridge> as HasCapabilities>::CAPABILITIES,
        ),
        // The same generic impl at an instantiation that does *not* persist.
        // One row per declaration site would record only the persisting value
        // and change-detect nothing about the computed one, so both ends of
        // the intersection are pinned.
        (
            "ferricml::pipeline::StagedPipeline<(ferricml::preprocessing::Normalizer, ferricml::preprocessing::StandardScaler), ferricml::linear_model::Ridge>",
            <StagedPipeline<(Normalizer, StandardScaler), Ridge> as HasCapabilities>::CAPABILITIES,
        ),
        (
            "ferricml::preprocessing::MaxAbsScaler",
            MaxAbsScaler::CAPABILITIES,
        ),
        (
            "ferricml::ranking::PairwiseLinearRanker",
            PairwiseLinearRanker::CAPABILITIES,
        ),
        (
            "ferricml::preprocessing::MinMaxScaler",
            MinMaxScaler::CAPABILITIES,
        ),
        ("ferricml::preprocessing::Binarizer", Binarizer::CAPABILITIES),
        (
            "ferricml::preprocessing::FunctionTransformer",
            FunctionTransformer::CAPABILITIES,
        ),
        (
            "ferricml::preprocessing::Normalizer",
            Normalizer::CAPABILITIES,
        ),
        (
            "ferricml::preprocessing::RobustScaler",
            RobustScaler::CAPABILITIES,
        ),
        (
            "ferricml::preprocessing::StandardScaler",
            StandardScaler::CAPABILITIES,
        ),
    ];
    rows.sort_by_key(|(name, _)| *name);
    rows
}

fn rendered() -> String {
    let mut text = String::from(
        "# Declared capability values. Generated by tests/capability_snapshot.rs;\n\
         # refresh with `make api-refresh`. `cargo-public-api` cannot see const\n\
         # values, so this file — not the API profile — is what change-detects a\n\
         # capability declaration.\n",
    );
    for (name, capabilities) in declarations() {
        let names = capability_names(capabilities);
        let value = if names.is_empty() {
            "-".to_owned()
        } else {
            names.join(", ")
        };
        text.push_str(&format!("{name} = {value}\n"));
    }
    text
}

fn snapshot_path() -> PathBuf {
    baseline_dir().join("ferricml-capabilities.txt")
}

#[test]
fn declared_capability_values_match_their_snapshot() {
    let path = snapshot_path();
    let generated = rendered();
    if std::env::var_os(REFRESH).is_some() {
        fs::write(&path, &generated).expect("write the capability snapshot");
        return;
    }
    let recorded = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} could not be read ({error}); run `make api-refresh` to create it",
            path.display()
        )
    });
    assert_eq!(
        recorded, generated,
        "declared capability values changed. This is a public, semver-relevant \
         contract diff: review it, then run `make api-refresh`."
    );
}

// ------------------------------------------------- completeness against the
// ------------------------------------------------- frozen public API profile

/// The trait whose implementations owe a snapshot row.
const HAS_CAPABILITIES: &str = "ferricml::api::HasCapabilities";

/// Every `impl … HasCapabilities for …` target in the API baseline, as a
/// pattern whose generic parameters match anything.
fn declaration_sites() -> Vec<String> {
    let mut sites: Vec<String> = Vec::new();
    for line in profile_lines() {
        let Some(site) = declaration_site(&line) else {
            continue;
        };
        if !sites.contains(&site) {
            sites.push(site);
        }
    }
    assert!(
        !sites.is_empty(),
        "no HasCapabilities impls were found in the API profile, so this check \
         would pass vacuously"
    );
    sites
}

fn declaration_site(line: &str) -> Option<String> {
    impl_target(line, HAS_CAPABILITIES).map(|(target, _)| target)
}

/// Every declaration the frozen API profile records has a value in the
/// snapshot, and every snapshot row corresponds to a real declaration.
///
/// This is what makes the companion snapshot a contract rather than a list.
/// `cargo-public-api` reliably sees the *existence* of a `HasCapabilities`
/// impl even though it cannot see the constant's value, so the two files check
/// each other: an estimator added without a snapshot row fails here, and so
/// does a row left behind by a deleted impl.
#[test]
fn every_declaration_site_in_the_api_profile_has_a_snapshot_row() {
    let sites = declaration_sites();
    let rows = declarations();

    let uncovered: Vec<&String> = sites
        .iter()
        .filter(|site| !rows.iter().any(|(name, _)| covers(site, name)))
        .collect();
    assert!(
        uncovered.is_empty(),
        "these HasCapabilities impls have no capability-snapshot row, so their \
         declared values are change-detected by nothing: {uncovered:#?}"
    );

    let stale: Vec<&str> = rows
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !sites.iter().any(|site| covers(site, name)))
        .collect();
    assert!(
        stale.is_empty(),
        "these capability-snapshot rows match no HasCapabilities impl in the \
         API profile: {stale:#?}"
    );
}

// ------------------------------------------------- the persistence contract,
// ------------------------------------------------- closed against the same two files

/// Whether a capability row declares persistence.
fn declares_artifact(capabilities: Capabilities) -> bool {
    capabilities.artifact()
}

/// `Capabilities::artifact` and the persistence traits mean the same thing, and
/// this is what stops them from drifting apart again.
///
/// They were previously maintained independently and disagreed for seven of
/// twelve estimators: each had a working, fuzz-tested encoder, none was
/// reachable through the composition contract, and nothing anywhere compared
/// the two lists. Both directions are checked, because the two failures are
/// different defects.
///
/// The generic-impl asymmetry is deliberate rather than a weakened rule. A
/// conditional implementation — `StagedPipeline<S, E>` where the parts persist
/// — applies to some instantiations and not others, so its *existence* says
/// nothing about a particular named instantiation. It can therefore satisfy a
/// declaration but cannot compel one; concrete impls, which name exactly one
/// type, are held to both directions.
#[test]
fn declared_persistence_and_the_persistence_traits_agree_in_both_directions() {
    let impls = persistence_impls();
    let rows = declarations();

    let undeclared: Vec<&str> = rows
        .iter()
        .filter(|(_, capabilities)| declares_artifact(*capabilities))
        .map(|(name, _)| *name)
        .filter(|name| {
            !impls
                .iter()
                .any(|candidate| covers(&candidate.target, name))
        })
        .collect();
    assert!(
        undeclared.is_empty(),
        "these types declare `Capabilities::artifact` but implement neither \
         persistence trait, so they promise bytes they cannot write: {undeclared:#?}"
    );

    let unpersisted: Vec<&String> = impls
        .iter()
        .filter(|candidate| !candidate.generic)
        .map(|candidate| &candidate.target)
        .filter(|pattern| {
            !rows.iter().any(|(name, capabilities)| {
                covers(pattern, name) && declares_artifact(*capabilities)
            })
        })
        .collect();
    assert!(
        unpersisted.is_empty(),
        "these types implement a persistence trait but do not declare \
         `Capabilities::artifact`, so their persistence is invisible to every \
         caller that asks: {unpersisted:#?}"
    );
}

/// The persistence closure must be able to fail, in both of its directions.
///
/// Without this it would pass for a tree that happened to be clean and prove
/// nothing — the same defect the layout checker's self-test exists to prevent,
/// and the reason this check exists at all.
#[test]
fn the_persistence_closure_detects_a_missing_impl_and_an_undeclared_one() {
    let concrete = persistence_impl(
        "impl ferricml::artifact::ModelArtifact for ferricml::linear_model::Ridge",
    )
    .expect("a concrete persistence impl parses");
    assert_eq!(
        concrete,
        PersistenceImpl {
            trait_path: MODEL_ARTIFACT,
            target: "ferricml::linear_model::Ridge".to_owned(),
            generic: false,
        }
    );

    let generic = persistence_impl(
        "impl<S, E> ferricml::artifact::StageArtifact for ferricml::pipeline::StagedPipeline<S, E> \
         where S: ferricml::pipeline::TransformerStack",
    )
    .expect("a generic persistence impl parses");
    assert_eq!(generic.trait_path, STAGE_ARTIFACT);
    assert!(
        generic.generic,
        "a parameterized impl is recognized as generic"
    );
    assert!(covers(
        &generic.target,
        "ferricml::pipeline::StagedPipeline<(ferricml::preprocessing::MinMaxScaler, \
         ferricml::preprocessing::StandardScaler), ferricml::linear_model::Ridge>"
    ));

    // A declaring type with no impl is the seven-estimator defect inverted, and
    // a covering impl is what clears it. `DummyRegressor` is the anchor because
    // its lack of persistence is a documented decision rather than a gap: a
    // baseline is refitted, never restored.
    let impls = persistence_impls();
    assert!(
        !impls
            .iter()
            .any(|candidate| covers(&candidate.target, "ferricml::dummy::DummyRegressor")),
        "a type with no persistence must match no impl; this assertion is what \
         keeps the missing-impl direction non-vacuous"
    );
    assert!(
        impls
            .iter()
            .any(|candidate| covers(&candidate.target, "ferricml::linear_model::Ridge"))
    );

    // Neither a non-persistence impl nor a non-impl line is mistaken for one.
    assert!(
        persistence_impl("impl ferricml::api::HasCapabilities for ferricml::linear_model::Ridge")
            .is_none()
    );
    assert!(persistence_impl("pub trait ferricml::artifact::ModelArtifact").is_none());
}

/// The completeness check must be able to fail.
///
/// Without this, a matcher that silently stopped matching would report a clean
/// tree and prove nothing — the same defect class the source-layout checker's
/// self-test exists to prevent.
#[test]
fn the_completeness_check_detects_a_missing_and_a_stale_row() {
    let site = declaration_site(
        "impl<S, E> ferricml::api::HasCapabilities for ferricml::pipeline::StagedPipeline<S, E> \
         where S: ferricml::pipeline::TransformerStack",
    )
    .expect("a generic declaration site parses");
    assert!(covers(
        &site,
        "ferricml::pipeline::StagedPipeline<(ferricml::preprocessing::MinMaxScaler, \
         ferricml::preprocessing::StandardScaler), ferricml::linear_model::Ridge>"
    ));
    assert!(!covers(&site, "ferricml::pipeline::Pipeline<A, B>"));

    let concrete =
        declaration_site("impl ferricml::api::HasCapabilities for ferricml::linear_model::Ridge")
            .expect("a concrete declaration site parses");
    assert_eq!(concrete, "ferricml::linear_model::Ridge");
    assert!(covers(&concrete, "ferricml::linear_model::Ridge"));
    assert!(!covers(&concrete, "ferricml::linear_model::Lasso"));

    // A generic parameter is only ever a whole identifier, never a path
    // segment that happens to share its spelling.
    let shadowed = declaration_site("impl<E> ferricml::api::HasCapabilities for ferricml::api::E")
        .expect("a site whose parameter name also appears as a path segment parses");
    assert_eq!(shadowed, "ferricml::api::E");

    assert!(declaration_site("pub struct ferricml::api::Capabilities").is_none());
    assert!(
        declaration_site("impl ferricml::api::Estimator for ferricml::linear_model::Ridge")
            .is_none()
    );
}
