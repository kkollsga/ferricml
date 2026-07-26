//! The documentation samples are tests, and this file is what keeps that true.
//!
//! `src/lib.rs` includes every narrative page under `docs/` as a doctest
//! carrier, so `cargo test` compiles and runs the Rust samples the
//! documentation site publishes. Two ways of losing that guarantee are already
//! impossible: a renamed page breaks `include_str!`, and a broken sample fails
//! its own doctest. The third way is silent, and is what these tests close.
//!
//! 1. **A page is never added to the carrier.** It renders on the site, its
//!    samples look authoritative, and nothing compiles them. Only a check that
//!    walks the directory can see this.
//! 2. **A sample opts out of running.** A ` ```rust,ignore ` or ` ```no_run `
//!    fence is prose wearing a compiler's clothes. Where something genuinely
//!    cannot execute in a doctest, the reason must be written down next to it,
//!    so a reader can tell "verified" from "illustrative".
//!
//! Each check is a pure function over its inputs plus a self-test proving it
//! fires, because a rule that silently stopped matching would otherwise pass
//! both the check and the suite.

use std::fs;
use std::path::{Path, PathBuf};

/// Fence tokens that stop a Rust sample from being compiled *and* run.
///
/// `text` is absent deliberately: a ```` ```text ```` block is not a Rust
/// sample at all, it is a formula or a diagram, and `src/linear_model/` uses
/// several. The tokens here are the ones that look like a tested sample while
/// not being one.
const NON_RUNNABLE_FENCE_TOKENS: [&str; 4] = ["ignore", "no_run", "compile_fail", "should_panic"];

/// The marker that makes a non-runnable sample acceptable. It must carry a
/// reason, so the exemption is an explanation rather than a silencer.
const EXEMPTION_MARKER: &str = "<!-- doctest-exempt:";

/// The same marker for a Rust doc comment, where HTML comments are not the
/// idiom.
const RUST_EXEMPTION_MARKER: &str = "doctest-exempt:";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every markdown page under `docs/`, as a repository-relative path, sorted.
fn narrative_pages(docs_root: &Path) -> Vec<String> {
    let mut pages = Vec::new();
    collect_markdown(docs_root, docs_root, &mut pages);
    pages.sort();
    pages
}

fn collect_markdown(dir: &Path, docs_root: &Path, out: &mut Vec<String>) {
    let entries = fs::read_dir(dir).expect("docs/ is readable");
    for entry in entries {
        let path = entry.expect("directory entry is readable").path();
        if path.is_dir() {
            collect_markdown(&path, docs_root, out);
        } else if path.extension().is_some_and(|extension| extension == "md") {
            let relative = path
                .strip_prefix(docs_root)
                .expect("page is below docs/")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(format!("docs/{relative}"));
        }
    }
}

/// Pages that `lib_source` does not include as a doctest carrier.
///
/// Pure over its inputs so the self-test can drive it with a synthetic source.
fn pages_missing_from_carrier(lib_source: &str, pages: &[String]) -> Vec<String> {
    pages
        .iter()
        .filter(|page| {
            let include = format!("include_str!(\"../{page}\")");
            !lib_source.contains(&include)
        })
        .cloned()
        .collect()
}

/// Non-runnable Rust fences in `text` that carry no written justification.
///
/// Returns one `(line number, fence)` pair per offence. A fence is excused when
/// an exemption marker appears in the five lines above it, which is close
/// enough that a reader meets the reason before the sample.
fn unjustified_non_runnable_fences(text: &str, marker: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut offences = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line
            .trim_start()
            .trim_start_matches("///")
            .trim_start_matches("//!");
        let trimmed = trimmed.trim_start();
        let Some(info) = trimmed.strip_prefix("```") else {
            continue;
        };
        let tokens: Vec<&str> = info
            .split(',')
            .map(|token| token.trim())
            .filter(|token| !token.is_empty())
            .collect();
        if !tokens
            .iter()
            .any(|token| NON_RUNNABLE_FENCE_TOKENS.contains(token))
        {
            continue;
        }
        let window_start = index.saturating_sub(5);
        let justified = lines[window_start..index]
            .iter()
            .any(|above| above.contains(marker));
        if !justified {
            offences.push((index + 1, trimmed.to_string()));
        }
    }
    offences
}

#[test]
fn every_narrative_page_is_compiled_as_a_doctest() {
    let root = repo_root();
    let pages = narrative_pages(&root.join("docs"));
    assert!(
        !pages.is_empty(),
        "docs/ holds no markdown pages, which means this check is inspecting the wrong directory"
    );

    let lib_source = fs::read_to_string(root.join("src/lib.rs")).expect("src/lib.rs is readable");
    let missing = pages_missing_from_carrier(&lib_source, &pages);

    assert!(
        missing.is_empty(),
        "these documentation pages are published but never compiled, so their Rust samples are \
         unverified prose. Add an `#[doc = include_str!(\"../<page>\")]` carrier module to the \
         `doc_pages` module in src/lib.rs for each:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn the_closure_check_detects_a_page_that_was_never_included() {
    let carrier = r#"
        #[cfg(doctest)]
        mod doc_pages {
            #[doc = include_str!("../docs/included.md")]
            mod included {}
        }
    "#;
    let pages = vec![
        "docs/included.md".to_string(),
        "docs/guide/forgotten.md".to_string(),
    ];

    let missing = pages_missing_from_carrier(carrier, &pages);

    assert_eq!(
        missing,
        vec!["docs/guide/forgotten.md".to_string()],
        "the closure check must report exactly the page the carrier omits"
    );
}

#[test]
fn no_documentation_sample_opts_out_of_running_without_saying_why() {
    let root = repo_root();
    let mut offences = Vec::new();

    for page in narrative_pages(&root.join("docs")) {
        let text = fs::read_to_string(root.join(&page)).expect("page is readable");
        for (line, fence) in unjustified_non_runnable_fences(&text, EXEMPTION_MARKER) {
            offences.push(format!("{page}:{line}: {fence}"));
        }
    }

    for source in rust_sources(&root.join("src")) {
        let text = fs::read_to_string(&source).expect("source is readable");
        let relative = source.strip_prefix(&root).unwrap_or(&source);
        for (line, fence) in unjustified_non_runnable_fences(&text, RUST_EXEMPTION_MARKER) {
            offences.push(format!("{}:{line}: {fence}", relative.display()));
        }
    }

    assert!(
        offences.is_empty(),
        "a sample that cannot run is a sample nobody is checking. Either make it executable, or \
         write the reason it cannot be — `{EXEMPTION_MARKER} needs a fitted artifact on disk -->` \
         in markdown, `{RUST_EXEMPTION_MARKER} ...` in a doc comment — within five lines above \
         the fence:\n  {}",
        offences.join("\n  ")
    );
}

#[test]
fn the_opt_out_check_distinguishes_a_justified_fence_from_a_bare_one() {
    let bare = "prose\n```rust,ignore\nlet x = 1;\n```\n";
    assert_eq!(
        unjustified_non_runnable_fences(bare, EXEMPTION_MARKER).len(),
        1,
        "an unexplained non-runnable fence must be reported"
    );

    let justified = "prose\n<!-- doctest-exempt: needs a fitted artifact on disk -->\n```rust,ignore\nlet x = 1;\n```\n";
    assert!(
        unjustified_non_runnable_fences(justified, EXEMPTION_MARKER).is_empty(),
        "a non-runnable fence with a written reason above it is acceptable"
    );

    let ordinary = "prose\n```rust\nlet x = 1;\n```\n";
    assert!(
        unjustified_non_runnable_fences(ordinary, EXEMPTION_MARKER).is_empty(),
        "an ordinary runnable fence is not an offence"
    );

    let formula = "prose\n```text\nloss = |y - Xb|^2\n```\n";
    assert!(
        unjustified_non_runnable_fences(formula, EXEMPTION_MARKER).is_empty(),
        "a text block is a formula, not a Rust sample that opted out"
    );
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    collect_rust(dir, &mut sources);
    sources.sort();
    sources
}

fn collect_rust(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).expect("source directory is readable");
    for entry in entries {
        let path = entry.expect("directory entry is readable").path();
        if path.is_dir() {
            collect_rust(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}
