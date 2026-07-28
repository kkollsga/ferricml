//! The dataset generator's cross-process determinism contract.
//!
//! A recipe is supposed to be the whole identity of the data it produces. The
//! unit tests in `src/datasets/` prove that within one process, which is the
//! weaker half: a generator seeded from a process-lifetime value — an address,
//! a lazily initialized table, a hash seed, a clock read once at start-up —
//! reproduces itself perfectly inside a run and differs between runs. The only
//! way to see that is to compare two runs.
//!
//! So this file re-executes the test binary. The child generates the same
//! recipes and prints their bytes; the parent generates them in-process and
//! compares. Both halves are the same code, so what the comparison isolates is
//! exactly the process boundary.
//!
//! The generator is behind the non-default `datasets` feature, and this file
//! sees it because the crate carries a path-only dev-dependency on itself with
//! that feature on. Nothing here needs a Cargo flag.

use ferricml::datasets::{Recipe, Source};
use std::env;
use std::process::Command;

/// Set on the child, so it prints instead of spawning another child.
const CHILD_MARKER: &str = "FERRICML_DATASET_GENERATOR_CHILD";

/// The recipes the two processes have to agree about.
///
/// One of each source, because the failure mode differs by source: the two
/// generator-backed ones would be broken by a state that varied per process,
/// and the lattice by index arithmetic that depended on anything but the
/// indices.
fn probes() -> Vec<Recipe> {
    vec![
        Recipe::seeded(37, 5, 11).expect("valid shape"),
        Recipe::new(37, 5, Source::Sampled { state: 11 }).expect("valid shape"),
        Recipe::new(
            37,
            5,
            Source::Lattice {
                row_stride: 131,
                column_stride: 17,
                modulus: 1009,
            },
        )
        .expect("valid shape"),
        Recipe::new(37, 5, Source::Xorshift32 { state: 0x9e37_79b9 }).expect("valid shape"),
    ]
}

/// Every probe's digest and design, as text a child process can print and a
/// parent can compare.
///
/// The design is rendered from `f32::to_bits` rather than from a decimal
/// formatting of the value, so the comparison is over the bytes themselves: two
/// distinct `f32` values can share a shortest decimal representation under some
/// formatting choices, and a determinism claim that a formatter could satisfy is
/// not the claim being made.
fn rendered_probes() -> String {
    let mut rendered = String::new();
    let mut buffer = Vec::new();
    for recipe in probes() {
        for byte in recipe.spec_digest() {
            rendered.push_str(&format!("{byte:02x}"));
        }
        rendered.push(' ');
        recipe.design_into(&mut buffer);
        for value in &buffer {
            rendered.push_str(&format!("{:08x}", value.to_bits()));
        }
        rendered.push('\n');
    }
    rendered
}

#[test]
fn the_same_recipe_reproduces_byte_identical_output_across_two_processes() {
    let rendered = rendered_probes();

    // The child arrives here with the marker set, prints, and stops. Everything
    // below is the parent's.
    if env::var_os(CHILD_MARKER).is_some() {
        print!("{rendered}");
        return;
    }

    // The probes have to be worth comparing. An empty or degenerate rendering
    // would make every assertion below true by vacuity, which is the failure
    // this crate keeps finding in checks that pass by not looking.
    assert_eq!(rendered.lines().count(), probes().len());
    for line in rendered.lines() {
        let (digest, design) = line.split_once(' ').expect("a digest and a design");
        assert_eq!(digest.len(), 64, "a SHA-256 digest is 32 bytes");
        assert_eq!(design.len(), 37 * 5 * 8, "one hex word per design value");
        assert!(
            design.chars().filter(|value| *value != '0').count() > 100,
            "the design rendered as almost all zeros, so the comparison is vacuous"
        );
    }

    let child = Command::new(env::current_exe().expect("the test binary's own path"))
        .args([
            "the_same_recipe_reproduces_byte_identical_output_across_two_processes",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD_MARKER, "1")
        .output()
        .expect("re-executing the test binary");
    assert!(
        child.status.success(),
        "the child process failed: {}",
        String::from_utf8_lossy(&child.stderr)
    );

    let printed = String::from_utf8(child.stdout).expect("the child printed text");
    for (index, expected) in rendered.lines().enumerate() {
        assert!(
            printed.contains(expected),
            "probe {index} differed between processes; the child printed:\n{printed}"
        );
    }

    // And the negative control: a line the child could not have printed is not
    // found, so `contains` is discriminating rather than trivially satisfied.
    let mutated = rendered
        .lines()
        .next()
        .expect("a first probe")
        .replace('0', "1");
    assert!(!printed.contains(&mutated));
}

/// A recipe that a caller could construct from data must refuse an impossible
/// request rather than trying it.
///
/// This lives in the integration crate as well as in the unit tests because the
/// public boundary is what a consumer actually reaches: the unit tests can see
/// crate-private constructors, and a validation that only held on the private
/// path would still pass them.
#[test]
fn the_public_boundary_refuses_impossible_requests() {
    use ferricml::datasets::DatasetError;

    assert_eq!(Recipe::seeded(0, 4, 1), Err(DatasetError::ZeroRows));
    assert_eq!(Recipe::seeded(4, 0, 1), Err(DatasetError::ZeroColumns));
    assert_eq!(
        Recipe::new(4, 4, Source::Xorshift32 { state: 0 }),
        Err(DatasetError::ZeroXorshiftState)
    );

    // The error is an ordinary `std::error::Error`, so a consumer can box it
    // alongside the crate's other typed refusals.
    let error: Box<dyn std::error::Error> = Box::new(Recipe::seeded(0, 4, 1).unwrap_err());
    assert!(error.to_string().contains("row count"));
}
