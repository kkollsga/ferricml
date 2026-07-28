//! Materialize FerricML's synthetic dataset suites into exchange containers.
//!
//! This is the writing half of the cross-language boundary. FerricML generates,
//! and every consumer — a Rust benchmark, a Python conformance script, a
//! comparison against another library — reads the same files, so no second
//! implementation of the generator has to be kept byte-identical by hand and no
//! consumer's own libm can move the data underneath a comparison.
//!
//! Each catalogue entry becomes `<name>.manifest.json` and `<name>.bin` in the
//! output directory. The manifest carries the recipe, its spec digest and the
//! determinism envelope; the array file carries the arrays concatenated
//! little-endian. Nothing here writes a format this binary defines — the
//! container is `ferricml::datasets::DatasetExchange`'s, so a reader and a
//! writer cannot disagree.
//!
//! # Why the catalogue is the suites
//!
//! The generator's own answer to "which problems exist" is
//! [`AccuracySuite`] and [`PerformanceGrid`], both held by tests
//! to spanning every family. Taking the catalogue from them rather than from a
//! list here means a family added to the crate appears in this tool without
//! anyone remembering to add it.
//!
//! # Exit status
//!
//! `0` on success, `2` for a usage error, `1` when a container could not be
//! produced. A partially written run reports the first failure and stops,
//! because a caller feeding a comparison wants the whole catalogue or a clear
//! failure rather than a directory it has to audit.

use ferricml::datasets::{
    AccuracySuite, CacheOutcome, DatasetExchange, ExchangeError, PerformanceGrid, Recipe, SuiteCase,
};
use std::process::ExitCode;

/// The usage text, which is also this binary's whole interface.
const USAGE: &str = "\
ferricml-datagen — materialize FerricML's synthetic dataset suites

USAGE:
    ferricml-datagen --out <DIR> [OPTIONS]
    ferricml-datagen --list [--suite <SUITE>]

OPTIONS:
    --out <DIR>       Directory the containers are written to. Created if absent.
    --suite <SUITE>   accuracy, performance, or all. Default: accuracy.
    --name <NAME>     Materialize only the entry with this name.
    --list            Print the catalogue and exit without writing anything.
    --force           Regenerate every entry, even one already current on disk.
    -h, --help        Print this text.

Each entry becomes <name>.manifest.json and <name>.bin. Without --force an
entry whose container already records the same recipe is read back rather than
regenerated, and the run reports which of the two happened.
";

/// Which catalogue a run materializes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Suite {
    Accuracy,
    Performance,
    All,
}

impl Suite {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "accuracy" => Some(Self::Accuracy),
            "performance" => Some(Self::Performance),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    /// Every entry of this catalogue, as a name and the recipe behind it.
    ///
    /// The two suites share a namespace, so each name carries which suite it
    /// came from: an accuracy case and a grid point of the same family are
    /// different problems at different sizes, and a container that overwrote
    /// the other would be a silent one.
    fn entries(self) -> Vec<(String, Recipe)> {
        let mut entries = Vec::new();
        if self != Self::Performance {
            entries.extend(
                AccuracySuite::cases()
                    .iter()
                    .map(|case| (format!("accuracy_{}", label(case)), case.recipe())),
            );
        }
        if self != Self::Accuracy {
            entries.extend(PerformanceGrid::cases().iter().map(|case| {
                (
                    format!(
                        "performance_{}_{}x{}",
                        label(case),
                        case.recipe().rows(),
                        case.recipe().columns(),
                    ),
                    case.recipe(),
                )
            }));
        }
        entries
    }
}

/// A case's family label with its hyphens kept.
///
/// The label is the identity a recorded benchmark row is already filed under,
/// so a container is named the same way as the measurement taken on it.
fn label(case: &SuiteCase) -> &'static str {
    case.name()
}

/// One run's parsed arguments.
struct Arguments {
    out: Option<String>,
    suite: Suite,
    name: Option<String>,
    list: bool,
    force: bool,
}

/// Parses the command line, refusing anything it does not define.
///
/// Hand-rolled because `default = []` is a product boundary in this crate and
/// an argument parser is a dependency. The interface is six flags, so the cost
/// of writing it is smaller than the cost of carrying a crate for it — and
/// refusing an unknown flag outright, rather than ignoring it, is what keeps a
/// mistyped invocation from quietly writing the wrong catalogue.
fn parse(arguments: impl Iterator<Item = String>) -> Result<Option<Arguments>, String> {
    let mut parsed = Arguments {
        out: None,
        suite: Suite::Accuracy,
        name: None,
        list: false,
        force: false,
    };
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(None),
            "--list" => parsed.list = true,
            "--force" => parsed.force = true,
            "--out" => {
                parsed.out = Some(value(&mut arguments, "--out")?);
            }
            "--name" => {
                parsed.name = Some(value(&mut arguments, "--name")?);
            }
            "--suite" => {
                let text = value(&mut arguments, "--suite")?;
                parsed.suite = Suite::parse(&text).ok_or_else(|| {
                    format!("unknown suite {text:?}; expected accuracy, performance or all")
                })?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if !parsed.list && parsed.out.is_none() {
        return Err("--out is required unless --list is given".to_owned());
    }
    Ok(Some(parsed))
}

fn value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn main() -> ExitCode {
    let parsed = match parse(std::env::args().skip(1)) {
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Ok(Some(parsed)) => parsed,
        Err(message) => {
            eprintln!("ferricml-datagen: {message}\n");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    let mut entries = parsed.suite.entries();
    if let Some(name) = &parsed.name {
        entries.retain(|(entry, _)| entry == name);
        if entries.is_empty() {
            eprintln!("ferricml-datagen: no catalogue entry is named {name:?}");
            return ExitCode::from(2);
        }
    }

    if parsed.list {
        for (name, recipe) in &entries {
            println!(
                "{name}\t{}x{}\t{}",
                recipe.rows(),
                recipe.columns(),
                hex(&recipe.spec_digest()),
            );
        }
        return ExitCode::SUCCESS;
    }

    let exchange = DatasetExchange::new(parsed.out.expect("--out is required above"));
    for (name, recipe) in &entries {
        let outcome = if parsed.force {
            exchange
                .materialize(name, recipe)
                .map(|container| (container, CacheOutcome::Generated))
        } else {
            exchange.ensure(name, recipe)
        };
        match outcome {
            Ok((container, outcome)) => println!(
                "{name}\t{}\t{}\t{} bytes",
                match outcome {
                    CacheOutcome::Reused => "reused",
                    _ => "generated",
                },
                hex(&container.spec_digest()),
                container.data_bytes(),
            ),
            Err(error) => {
                report(name, &error);
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

/// Prints a failure with the chain the error carries.
///
/// The source chain matters here: an `Io` failure's own message names the
/// path, and its source names what the filesystem said, and a caller
/// diagnosing a run wants both rather than whichever one the outer type
/// happened to render.
fn report(name: &str, error: &ExchangeError) {
    eprintln!("ferricml-datagen: {name}: {error}");
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        eprintln!("  caused by: {cause}");
        source = cause.source();
    }
}

fn hex(digest: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}
