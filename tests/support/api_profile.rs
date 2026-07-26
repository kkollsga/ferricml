//! The frozen public API profile, read as the registry no one maintains.
//!
//! `tests/api-baselines/rust/ferricml-default.txt` is regenerated from the
//! compiler's view of the public surface by `make api-refresh` and compared by
//! `make api-check`, so a type cannot gain or lose a trait implementation
//! without that file moving and a reviewer seeing the diff. That makes it the
//! right thing to close a *hand-maintained* table against: the table is a list
//! someone has to remember to extend, and the profile is not.
//!
//! One reader serves every such closure. It lives here rather than in whichever
//! test needed it first because a second copy of this parser would be exactly
//! the defect the closures exist to prevent — a list to keep in step by hand.
//!
//! Three tables close against it today:
//!
//! * `tests/capability_snapshot.rs` — declared capability values, and the
//!   `Capabilities::artifact` / persistence-trait agreement;
//! * `tests/artifact_hardening.rs` — the decoder table that receives hostile
//!   bytes;
//! * `tests/artifact_tags.rs` — the frozen artifact-kind and composition-tag
//!   numbers.
//!
//! The closure that "a list someone must remember to update" needs is
//! two-directional, and [`close_against_persistence`] is that shape: an impl
//! with no table entry and a table entry with no impl are different defects,
//! and neither is allowed to pass as the other's absence.

use std::fs;
use std::path::PathBuf;

/// The two traits that *are* persistence, as the API profile spells them.
pub const MODEL_ARTIFACT: &str = "ferricml::artifact::ModelArtifact";
pub const STAGE_ARTIFACT: &str = "ferricml::artifact::StageArtifact";

/// Both persistence traits, for a caller that does not distinguish them.
pub const PERSISTENCE_TRAITS: [&str; 2] = [MODEL_ARTIFACT, STAGE_ARTIFACT];

/// A generic parameter position in an impl target, as a match-anything.
///
/// A NUL never appears in a rendered type name, so it cannot collide with one.
pub const WILDCARD: char = '\u{0}';

/// The directory holding the frozen API profile and its companion snapshots.
pub fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("api-baselines")
        .join("rust")
}

/// The frozen public API profile, line by line.
pub fn profile_lines() -> Vec<String> {
    fs::read_to_string(baseline_dir().join("ferricml-default.txt"))
        .expect("read the frozen public API profile")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// One persistence-trait implementation, as the API baseline records it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistenceImpl {
    /// The implemented trait: [`MODEL_ARTIFACT`] or [`STAGE_ARTIFACT`].
    pub trait_path: &'static str,
    /// The implementing type, with the impl's own generic parameters replaced
    /// by [`WILDCARD`], so a conditional impl covers its instantiations.
    pub target: String,
    /// Whether the impl is parameterized — that is, whether it applies to
    /// *some* instantiations rather than to one named type.
    pub generic: bool,
}

/// Every persistence-trait implementation the frozen API profile records.
///
/// The emptiness guard is the vacuity floor: a parser that stopped matching
/// would otherwise report a clean tree and every closure built on it would pass
/// by finding nothing to compare.
pub fn persistence_impls() -> Vec<PersistenceImpl> {
    let mut impls: Vec<PersistenceImpl> = Vec::new();
    for line in profile_lines() {
        let Some(entry) = persistence_impl(&line) else {
            continue;
        };
        if !impls.contains(&entry) {
            impls.push(entry);
        }
    }
    assert!(
        !impls.is_empty(),
        "no persistence impls were found in the API profile, so every closure \
         against it would pass vacuously"
    );
    impls
}

/// One profile line, if it implements a persistence trait.
pub fn persistence_impl(line: &str) -> Option<PersistenceImpl> {
    PERSISTENCE_TRAITS.iter().find_map(|trait_path| {
        impl_target(line, trait_path).map(|(target, generic)| PersistenceImpl {
            trait_path,
            target,
            generic,
        })
    })
}

/// The target of `impl … {trait_path} for …`, as a pattern, and whether the
/// impl is parameterized.
///
/// This is the whole profile parser: every closure that asks "which types
/// implement this trait" goes through it, whichever trait it means.
pub fn impl_target(line: &str, trait_path: &str) -> Option<(String, bool)> {
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
        .strip_prefix(&format!("{trait_path} for "))?;
    let target = target.split(" where ").next().unwrap_or(target).trim();
    let generic = !parameters.is_empty();
    Some((wildcard_pattern(target, &parameters), generic))
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
pub fn covers(pattern: &str, name: &str) -> bool {
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

/// What a two-directional closure found, in the two directions it looked.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Closure {
    /// Implementations in the API profile that no table entry covers.
    pub unlisted: Vec<String>,
    /// Table entries that match no implementation in the API profile.
    pub stale: Vec<String>,
}

impl Closure {
    /// Whether the table and the profile agree in both directions.
    pub fn is_closed(&self) -> bool {
        self.unlisted.is_empty() && self.stale.is_empty()
    }
}

/// Closes a hand-maintained table of type names against the persistence impls
/// the frozen API profile records, in both directions.
///
/// `traits` selects which persistence traits participate, because a table may
/// be about one of them — the artifact-kind rows are split by trait — or about
/// both, as the decoder table is.
///
/// A parameterized impl owes an instantiation here, unlike in the capability
/// closure where its existence deliberately cannot compel a declaration. The
/// question is different: a conditional `StagedPipeline<S, E>` impl really does
/// write bytes, and a table that never names one composition never exercises
/// the code that writes them.
pub fn close_against_persistence(traits: &[&str], entries: &[&str]) -> Closure {
    let impls: Vec<PersistenceImpl> = persistence_impls()
        .into_iter()
        .filter(|entry| traits.contains(&entry.trait_path))
        .collect();
    assert!(
        !impls.is_empty(),
        "no impl of {traits:?} was found in the API profile, so this closure \
         would pass vacuously"
    );
    assert!(
        !entries.is_empty(),
        "the table is empty, so this closure would pass vacuously"
    );
    Closure {
        unlisted: impls
            .iter()
            .filter(|candidate| !entries.iter().any(|entry| covers(&candidate.target, entry)))
            .map(|candidate| candidate.target.clone())
            .collect(),
        stale: entries
            .iter()
            .filter(|entry| {
                !impls
                    .iter()
                    .any(|candidate| covers(&candidate.target, entry))
            })
            .map(|entry| (*entry).to_owned())
            .collect(),
    }
}
