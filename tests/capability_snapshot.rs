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

use ferricml::api::{AnyClassifier, AnyRegressor, Capabilities, HasCapabilities};
use ferricml::calibration::{CalibratedClassifier, IsotonicRegression, PlattCalibrator};
use ferricml::dummy::{DummyClassifier, DummyRegressor};
use ferricml::ensemble::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
    RandomForestRegressor,
};
use ferricml::linear_model::{ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge};
use ferricml::pipeline::{Pipeline, StagedPipeline};
use ferricml::preprocessing::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use ferricml::ranking::PairwiseLinearRanker;

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

fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("api-baselines")
        .join("rust")
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

/// A generic parameter position in a declaration site, as a match-anything.
const WILDCARD: char = '\u{0}';

/// Every `impl … HasCapabilities for …` target in the API baseline, as a
/// pattern whose generic parameters match anything.
fn declaration_sites() -> Vec<String> {
    let baseline = fs::read_to_string(baseline_dir().join("ferricml-default.txt"))
        .expect("read the frozen public API profile");
    let mut sites: Vec<String> = Vec::new();
    for line in baseline.lines() {
        let Some(site) = declaration_site(line) else {
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
    let line = line.trim();
    let rest = line.strip_prefix("impl")?;
    // Generic parameters, if the impl has any: `<C: Trait>` or `<S, E>`.
    let (parameters, rest) = if let Some(inner) = rest.strip_prefix('<') {
        let end = matching_angle(inner)?;
        (parameter_names(&inner[..end]), &inner[end + 1..])
    } else {
        (Vec::new(), rest)
    };
    let target = rest
        .trim_start()
        .strip_prefix("ferricml::api::HasCapabilities for ")?;
    let target = target.split(" where ").next().unwrap_or(target).trim();
    Some(wildcard_pattern(target, &parameters))
}

/// Index of the `>` closing an already-opened `<`.
fn matching_angle(text: &str) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, character) in text.char_indices() {
        match character {
            '<' => depth += 1,
            '>' if depth == 0 => return Some(index),
            '>' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// The declared names in `C: Trait, S, E`, ignoring their bounds.
fn parameter_names(parameters: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut depth = 0_usize;
    let mut current = String::new();
    for character in parameters.chars() {
        match character {
            '<' | '(' => {
                depth += 1;
                current.push(character);
            }
            '>' | ')' => {
                depth = depth.saturating_sub(1);
                current.push(character);
            }
            ',' if depth == 0 => {
                names.push(current.clone());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    names.push(current);
    names
        .into_iter()
        .filter_map(|name| {
            let name = name.split(':').next()?.trim().to_owned();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Replaces whole-identifier occurrences of the impl's generic parameters with
/// a wildcard, so `StagedPipeline<S, E>` covers any concrete instantiation.
fn wildcard_pattern(target: &str, parameters: &[String]) -> String {
    let mut pattern = String::new();
    let mut identifier = String::new();
    for character in target.chars() {
        if character.is_alphanumeric() || character == '_' {
            identifier.push(character);
            continue;
        }
        push_identifier(&mut pattern, &identifier, parameters);
        identifier.clear();
        pattern.push(character);
    }
    push_identifier(&mut pattern, &identifier, parameters);
    pattern
}

fn push_identifier(pattern: &mut String, identifier: &str, parameters: &[String]) {
    if identifier.is_empty() {
        return;
    }
    // A path segment such as `ferricml` is never a bare generic parameter,
    // because a parameter is only ever a whole identifier between separators.
    if parameters.iter().any(|parameter| parameter == identifier) && !pattern.ends_with("::") {
        pattern.push(WILDCARD);
    } else {
        pattern.push_str(identifier);
    }
}

/// Whether a concrete type name satisfies a wildcard pattern.
fn covers(pattern: &str, name: &str) -> bool {
    let segments: Vec<&str> = pattern.split(WILDCARD).collect();
    if segments.len() == 1 {
        return pattern == name;
    }
    let Some(mut rest) = name.strip_prefix(segments[0]) else {
        return false;
    };
    let last = segments.len() - 1;
    for (index, segment) in segments.iter().enumerate().skip(1) {
        if index == last {
            return segment.is_empty() || rest.ends_with(segment);
        }
        match rest.find(segment) {
            Some(position) => rest = &rest[position + segment.len()..],
            None => return false,
        }
    }
    true
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
