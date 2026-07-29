//! The exchange container's own contract: canonical text, exact round trip,
//! and a reader that refuses everything it did not write.
//!
//! The allocation oracle lives in `tests/dataset_exchange.rs` rather than here,
//! because measuring peak allocation needs a `#[global_allocator]` and that is
//! a property of a whole test binary. What this file proves is everything a
//! reservation budget cannot see: that the text is canonical, that every family
//! survives the trip value for value, and that each refusal fires on the input
//! it names.

use super::contamination::{Contamination, WeightPattern};
use super::dataset::Dataset;
use super::error::{DatasetError, ExchangeError};
use super::exchange::{
    ArrayDtype, CacheOutcome, DatasetArray, DatasetExchange, Derivation, MaterializedDataset,
    Payload,
};
use super::manifest;
use super::presets::{ReferenceLane, ReferenceQuality, Split};
use super::recipe::{Recipe, Source};
use super::structural::{ClassBalance, ClassGeometry, GroupPattern};
use super::suites::AccuracySuite;
use super::task::{BinaryKind, Family, GlmLink, NonlinearKind, Portability, Task};

/// Writes a container to a fresh directory and reads it back.
///
/// Through the real files rather than around them: the manifest text and the
/// array bytes are exactly what a Python reader would open.
fn round_trip(container: &MaterializedDataset) -> Result<MaterializedDataset, ExchangeError> {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = container.recipe();
    exchange.materialize("probe", &recipe)?;
    exchange.load("probe")
}

/// The rendered manifest of one recipe, as text a test can edit.
fn rendered(recipe: &Recipe) -> String {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    exchange
        .materialize("probe", recipe)
        .expect("a valid recipe materializes");
    std::fs::read_to_string(exchange.manifest_path("probe").expect("a valid name"))
        .expect("the manifest was just written")
}

/// Parses a manifest text directly, which is what an edited file exercises.
fn parse(text: &str) -> Result<(), ExchangeError> {
    manifest::parse(text).map(|_| ())
}

#[test]
fn every_accuracy_family_survives_the_round_trip_value_for_value() {
    for case in AccuracySuite::cases() {
        let recipe = case.recipe();
        let container = MaterializedDataset::new(&recipe);
        let loaded = round_trip(&container).expect("a generated container reloads");
        // The whole container, not a sampled array: the recipe, both digests,
        // the envelope, and every array's name, shape, type and values.
        assert_eq!(loaded, container, "{} did not survive", case.name());
        // And the design really is the design, rather than two equal copies of
        // something else.
        assert_eq!(
            loaded.array("features").and_then(DatasetArray::f32_values),
            Some(recipe.generate().features().as_slice()),
            "{}",
            case.name(),
        );
    }
}

#[test]
fn every_source_and_every_pattern_survives_the_round_trip() {
    let sources = [
        Source::Sampled { state: 11 },
        Source::Lattice {
            row_stride: 131,
            column_stride: 17,
            modulus: 1009,
        },
        Source::Xorshift32 { state: 0x9e37_79b9 },
    ];
    for source in sources {
        let recipe = Recipe::new(48, 6, source).expect("a valid shape");
        let container = MaterializedDataset::new(&recipe);
        assert_eq!(round_trip(&container).expect("reloads"), container);
    }

    let patterns = [
        WeightPattern::Uniform,
        WeightPattern::Ramp {
            low: 0.25,
            high: 4.0,
        },
        WeightPattern::Alternating {
            first: 1.5,
            second: 0.5,
        },
    ];
    for pattern in patterns {
        let recipe = Recipe::seeded(48, 6, 3)
            .and_then(|recipe| recipe.with_weights(pattern))
            .expect("a valid pattern");
        let container = MaterializedDataset::new(&recipe);
        assert_eq!(round_trip(&container).expect("reloads"), container);
    }

    let groupings = [
        GroupPattern::RoundRobin { groups: 6 },
        GroupPattern::Contiguous { groups: 6 },
        GroupPattern::Unbalanced {
            groups: 6,
            ratio: 3.0,
        },
    ];
    for grouping in groupings {
        let recipe = Recipe::seeded(48, 6, 3)
            .and_then(|recipe| recipe.with_groups(grouping))
            .expect("a valid pattern");
        let container = MaterializedDataset::new(&recipe);
        let loaded = round_trip(&container).expect("reloads");
        assert_eq!(loaded, container);
        assert_eq!(
            loaded
                .array("groups")
                .expect("a grouped recipe exports its groups")
                .dtype(),
            ArrayDtype::U64,
        );
    }
}

#[test]
fn a_class_balanced_weighting_and_a_contamination_survive_together() {
    let recipe = Recipe::seeded(128, 6, 9)
        .and_then(|recipe| {
            recipe.with_task(Task::LinearBinary {
                informative: 3,
                separation: 2.0,
                prevalence: 0.25,
            })
        })
        .and_then(|recipe| {
            recipe.with_contamination(
                Contamination::none()
                    .with_label_noise(0.05)
                    .with_constant_columns(1)
                    .with_collinear_pairs(1)
                    .with_feature_scale_spread(2.0),
            )
        })
        .and_then(|recipe| recipe.with_weights(WeightPattern::ClassBalanced))
        .expect("a valid recipe");
    let container = MaterializedDataset::new(&recipe);
    let loaded = round_trip(&container).expect("reloads");
    assert_eq!(loaded, container);
    assert_eq!(loaded.recipe(), recipe);
    assert_eq!(loaded.portability(), Portability::PerRunner);
}

#[test]
fn a_ranking_container_carries_its_pairs_and_its_queries() {
    let recipe = Recipe::seeded(64, 5, 21)
        .and_then(|recipe| {
            recipe.with_task(Task::Ranking {
                queries: 16,
                docs_per_query: 4,
                grades: 3,
                informative: 3,
                coefficient_scale: 1.0,
            })
        })
        .expect("a valid recipe");
    let dataset = recipe.generate();
    let container = MaterializedDataset::new(&recipe);
    let loaded = round_trip(&container).expect("reloads");
    assert_eq!(loaded, container);

    let pairs = dataset.pairs().expect("a ranking family draws pairs");
    let left = loaded.array("pair_left").expect("pairs are exported");
    assert_eq!(left.len(), pairs.len());
    assert_eq!(
        left.u64_values().expect("pair indices are u64")[0],
        pairs[0].pair().left() as u64,
    );
    assert_eq!(
        loaded
            .array("pair_outcome")
            .expect("outcomes are exported")
            .dtype(),
        ArrayDtype::U8,
    );
    assert_eq!(
        loaded
            .array("groups")
            .expect("a ranking family assigns query groups")
            .len(),
        64,
    );
}

#[test]
fn a_clustered_container_has_no_target_and_still_has_its_answer() {
    let recipe = Recipe::seeded(64, 4, 5)
        .and_then(|recipe| {
            recipe.with_task(Task::Clustered {
                blobs: 4,
                spread: 0.2,
            })
        })
        .expect("a valid recipe");
    let loaded = round_trip(&MaterializedDataset::new(&recipe)).expect("reloads");
    assert!(
        loaded.array("target").is_none(),
        "an unsupervised family exports no target",
    );
    assert_eq!(
        loaded
            .array("truth_cluster_assignments")
            .expect("the assignment is the answer")
            .len(),
        64,
    );
    assert_eq!(
        loaded
            .array("truth_cluster_centres")
            .expect("the centres are recorded")
            .columns(),
        4,
    );
}

#[test]
fn the_manifest_text_is_canonical() {
    let recipe = AccuracySuite::cases()[0].recipe();
    assert_eq!(rendered(&recipe), rendered(&recipe));
}

#[test]
fn every_float_field_round_trips_through_the_text_bit_for_bit() {
    // Rust's `Display` for `f32` emits the shortest decimal that parses back to
    // the same value, and `parse` is correctly rounded — so this is a property
    // of the standard library rather than of the format. It is pinned anyway,
    // because a manifest that rendered a float lossily would produce a silently
    // different dataset rather than a parse failure.
    let probes = [
        0.0_f32,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        0.1,
        1.0 / 3.0,
        -1.234_567_8e-20,
    ];
    for probe in probes {
        let recipe = Recipe::seeded(16, 4, 1)
            .and_then(|recipe| {
                recipe.with_task(Task::LinearRegression {
                    informative: 2,
                    coefficient_scale: 1.0,
                    intercept: probe,
                    noise_scale: 0.0,
                })
            })
            .expect("any finite intercept is admissible");
        let parsed = manifest::parse(&rendered(&recipe)).expect("a rendered manifest parses");
        let Some(Task::LinearRegression { intercept, .. }) = parsed.recipe.task() else {
            panic!("the task survived as another family");
        };
        assert_eq!(
            intercept.to_bits(),
            probe.to_bits(),
            "{probe} did not survive the text",
        );
    }
}

#[test]
fn every_family_label_is_the_kind_the_reader_accepts() {
    // The writer spells a task's kind with `Family::label`, and the reader
    // matches the same roster. Pinning the pair here is what stops the two
    // drifting into a format that writes a name nothing reads.
    for case in AccuracySuite::cases() {
        let text = rendered(&case.recipe());
        let quoted = format!("\"kind\": \"{}\"", case.family().label());
        assert!(
            text.contains(&quoted),
            "{} is not written under its own label",
            case.name(),
        );
    }
    assert_eq!(Family::ALL.len(), AccuracySuite::cases().len());
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn an_edited_recipe_no_longer_matches_its_recorded_digest() {
    let recipe = Recipe::seeded(64, 4, 7).expect("a valid shape");
    let text = rendered(&recipe);
    let edited = text.replace("\"rows\": 64", "\"rows\": 65");
    assert_ne!(edited, text, "the edit has to reach the recipe");
    assert!(matches!(
        parse(&edited),
        Err(ExchangeError::SpecDigestMismatch),
    ));
}

#[test]
fn an_edited_determinism_envelope_is_refused() {
    // A transcendental family carried as `bit-exact` would tell a harness
    // comparing two machines that a difference is a defect.
    let recipe = Recipe::seeded(64, 4, 7)
        .and_then(|recipe| {
            recipe.with_task(Task::LinearBinary {
                informative: 2,
                separation: 2.0,
                prevalence: 0.3,
            })
        })
        .expect("a valid recipe");
    let text = rendered(&recipe);
    assert!(text.contains("\"portability\": \"per-runner\""));
    let edited = text.replace("\"per-runner\"", "\"bit-exact\"");
    assert!(matches!(
        parse(&edited),
        Err(ExchangeError::SpecDigestMismatch),
    ));
}

#[test]
fn a_reordered_manifest_is_refused_rather_than_searched() {
    let text = rendered(&Recipe::seeded(16, 4, 1).expect("a valid shape"));
    let edited = text.replace("\"rows\": 16", "\"columns\": 16");
    assert!(matches!(
        parse(&edited),
        Err(ExchangeError::MalformedManifest { .. }),
    ));
}

#[test]
fn an_unknown_format_version_is_refused_before_anything_else() {
    let text = rendered(&Recipe::seeded(16, 4, 1).expect("a valid shape"));
    let edited = text.replace("\"format\": 2", "\"format\": 3");
    assert!(matches!(
        parse(&edited),
        Err(ExchangeError::UnsupportedFormat { found: 3 }),
    ));
    // And the version the reader accepts is the one the writer emits, so this
    // test cannot start passing because it edited a field that is no longer
    // there.
    assert!(text.contains("\"format\": 2"));
}

#[test]
fn a_manifest_describing_an_impossible_recipe_is_refused_as_a_recipe() {
    let text = rendered(&Recipe::seeded(16, 4, 1).expect("a valid shape"));
    let edited = text.replace("\"rows\": 16", "\"rows\": 0");
    assert!(matches!(
        parse(&edited),
        Err(ExchangeError::InvalidRecipe(DatasetError::ZeroRows)),
    ));
}

#[test]
fn a_string_escape_is_refused_rather_than_unescaped() {
    // The writer emits no escape, so accepting one would mean the reader
    // understood a manifest this crate cannot produce.
    let text = rendered(&Recipe::seeded(16, 4, 1).expect("a valid shape"));
    let edited = text.replace("\"features\"", "\"fea\\u0074ures\"");
    assert!(matches!(
        parse(&edited),
        Err(ExchangeError::MalformedManifest { .. }),
    ));
}

#[test]
fn trailing_content_after_the_manifest_is_refused() {
    let mut text = rendered(&Recipe::seeded(16, 4, 1).expect("a valid shape"));
    text.push_str("{}\n");
    assert!(matches!(
        parse(&text),
        Err(ExchangeError::MalformedManifest { .. }),
    ));
}

#[test]
fn a_container_name_that_is_not_a_file_stem_is_refused() {
    let exchange = DatasetExchange::new("target/never-created");
    for name in ["", "../escape", "a/b", "Upper", "with space", "dot.name"] {
        assert!(
            matches!(
                exchange.manifest_path(name),
                Err(ExchangeError::InvalidName)
            ),
            "{name:?} was accepted",
        );
        assert!(matches!(
            exchange.load(name),
            Err(ExchangeError::InvalidName)
        ));
    }
    assert!(exchange.manifest_path("linear-regression_256x8").is_ok());
}

#[test]
fn a_tampered_array_file_fails_its_checksum() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(32, 4, 2).expect("a valid shape");
    exchange
        .materialize("probe", &recipe)
        .expect("a valid recipe materializes");

    let path = exchange.data_path("probe").expect("a valid name");
    let mut bytes = std::fs::read(&path).expect("the file was just written");
    bytes[0] ^= 0xff;
    std::fs::write(&path, &bytes).expect("the file is writable");

    assert!(matches!(
        exchange.load("probe"),
        Err(ExchangeError::DataChecksumMismatch),
    ));
}

#[test]
fn a_truncated_array_file_is_refused() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(32, 4, 2).expect("a valid shape");
    exchange
        .materialize("probe", &recipe)
        .expect("a valid recipe materializes");

    let path = exchange.data_path("probe").expect("a valid name");
    let bytes = std::fs::read(&path).expect("the file was just written");
    std::fs::write(&path, &bytes[..bytes.len() - 8]).expect("the file is writable");

    assert!(matches!(
        exchange.load("probe"),
        Err(ExchangeError::InvalidArrayTable),
    ));
}

#[test]
fn a_missing_container_reports_the_path_it_looked_for() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let Err(ExchangeError::Io { path, .. }) = exchange.load("absent") else {
        panic!("a missing container has to name what was missing");
    };
    assert_eq!(
        path,
        exchange.manifest_path("absent").expect("a valid name")
    );
}

// ---------------------------------------------------------------------------
// Derived containers
// ---------------------------------------------------------------------------

/// The derivation and dataset of one frozen lane split.
fn lane(lane: ReferenceLane, seed: u64, split: Split) -> (ReferenceQuality, Dataset, Derivation) {
    let preset = ReferenceQuality::new(lane, seed);
    let dataset = preset.split(split);
    (
        preset,
        dataset,
        Derivation::ReferenceSplit { lane, seed, split },
    )
}

#[test]
fn every_lane_split_and_seed_survives_the_round_trip_carrying_what_it_holds() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    for reference in ReferenceLane::ALL {
        for seed in ReferenceQuality::SEEDS {
            for split in Split::ALL {
                let (preset, dataset, derivation) = lane(reference, seed, split);
                let written = exchange
                    .materialize_derived("probe", &preset.recipe(), &dataset, derivation)
                    .expect("a lane split materializes");
                let loaded = exchange.load("probe").expect("it reloads");
                assert_eq!(loaded, written, "{reference:?} {seed} {split:?}");
                assert_eq!(loaded.payload(), Payload::Derived(derivation));
                // The arrays really are the split's, rather than equal copies
                // of something else the round trip preserved just as faithfully.
                assert_eq!(
                    loaded.array("features").and_then(DatasetArray::f32_values),
                    Some(dataset.features().as_slice()),
                );
            }
        }
    }
}

#[test]
fn a_derived_container_is_refused_as_regenerable_while_its_digest_says_otherwise() {
    // The whole reason `Payload` exists, stated as the two facts that make it
    // necessary: the digest agrees, and the data does not.
    let (preset, dataset, derivation) = lane(ReferenceLane::NoisyBinary, 22, Split::Test);
    let recipe = preset.recipe();
    let derived = MaterializedDataset::derived(&recipe, &dataset, derivation);
    let generated = MaterializedDataset::new(&recipe);

    assert_eq!(derived.spec_digest(), generated.spec_digest());
    assert_ne!(derived.data_digest(), generated.data_digest());

    assert!(matches!(
        derived.regenerate(),
        Err(ExchangeError::NotRegenerable { derivation: held }) if held == derivation,
    ));
    // And the generated half of the contract holds: a container that claims to
    // be its recipe's output reproduces itself exactly.
    assert_eq!(
        generated
            .regenerate()
            .expect("a generated container regenerates"),
        generated,
    );
}

#[test]
fn ensure_refuses_a_derived_container_rather_than_serving_or_replacing_it() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let (preset, dataset, derivation) = lane(ReferenceLane::Regression, 44, Split::Train);
    let recipe = preset.recipe();
    let written = exchange
        .materialize_derived("probe", &recipe, &dataset, derivation)
        .expect("a lane split materializes");

    // `ensure` is asked for the very recipe the container records, so a
    // digest-keyed cache would call this a hit and hand back the training half.
    assert!(matches!(
        exchange.ensure("probe", &recipe),
        Err(ExchangeError::NotRegenerable { .. }),
    ));
    // Refused rather than overwritten: the recording is still there, and no
    // recipe could have rebuilt it if it were not.
    assert_eq!(exchange.load("probe").expect("still readable"), written);
}

#[test]
fn a_derived_cache_hit_needs_the_lane_the_seed_and_the_split_to_agree() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let (preset, train, train_derivation) = lane(ReferenceLane::SeparableBinary, 11, Split::Train);
    let recipe = preset.recipe();

    let (first, outcome) = exchange
        .ensure_derived("probe", &recipe, &train, train_derivation)
        .expect("writes");
    assert_eq!(outcome, CacheOutcome::Generated);
    let (second, outcome) = exchange
        .ensure_derived("probe", &recipe, &train, train_derivation)
        .expect("reuses");
    assert_eq!(outcome, CacheOutcome::Reused);
    assert_eq!(first, second);

    // The other split of the same lane shares this recipe and therefore this
    // digest, so only the derivation tells the two apart.
    let (_, test, test_derivation) = lane(ReferenceLane::SeparableBinary, 11, Split::Test);
    let (replaced, outcome) = exchange
        .ensure_derived("probe", &recipe, &test, test_derivation)
        .expect("rewrites");
    assert_eq!(outcome, CacheOutcome::Generated);
    assert_eq!(replaced.spec_digest(), first.spec_digest());
    assert_ne!(replaced.data_digest(), first.data_digest());

    // A generated container under the name is replaced rather than refused: it
    // is reproducible from the recipe written inside it, so nothing is lost.
    exchange
        .materialize("probe", &recipe)
        .expect("a generated container");
    let (_, outcome) = exchange
        .ensure_derived("probe", &recipe, &train, train_derivation)
        .expect("rewrites");
    assert_eq!(outcome, CacheOutcome::Generated);
}

#[test]
fn an_edited_payload_block_is_refused_rather_than_partly_understood() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let (preset, dataset, derivation) = lane(ReferenceLane::ImbalancedBinary, 33, Split::Test);
    exchange
        .materialize_derived("probe", &preset.recipe(), &dataset, derivation)
        .expect("a lane split materializes");
    let text = std::fs::read_to_string(exchange.manifest_path("probe").expect("a valid name"))
        .expect("the manifest was just written");

    for (from, to) in [
        ("\"kind\": \"derived\"", "\"kind\": \"invented\""),
        (
            "\"derivation\": \"reference-split\"",
            "\"derivation\": \"other\"",
        ),
        ("\"lane\": \"imbalanced\"", "\"lane\": \"imbalancd\""),
        ("\"split\": \"test\"", "\"split\": \"holdout\""),
    ] {
        let edited = text.replace(from, to);
        assert_ne!(edited, text, "the edit {from:?} did not reach the manifest");
        assert!(
            matches!(parse(&edited), Err(ExchangeError::MalformedManifest { .. })),
            "{to} was accepted",
        );
    }

    // A generated payload is spelled by one word, and the reader accepts only
    // that word — so the two kinds cannot be confused by a one-character edit.
    let generated = rendered(&preset.recipe());
    assert!(generated.contains("\"kind\": \"generated\""));
    assert!(matches!(
        parse(&generated.replace("\"kind\": \"generated\"", "\"kind\": \"generted\"")),
        Err(ExchangeError::MalformedManifest { .. }),
    ));
}

// ---------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------

#[test]
fn a_repeated_request_for_one_recipe_is_a_file_read() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(64, 5, 13).expect("a valid shape");

    let (first, outcome) = exchange.ensure("probe", &recipe).expect("generates");
    assert_eq!(outcome, CacheOutcome::Generated);
    let (second, outcome) = exchange.ensure("probe", &recipe).expect("reuses");
    assert_eq!(outcome, CacheOutcome::Reused);
    assert_eq!(first, second);
}

#[test]
fn a_changed_recipe_regenerates_under_the_same_name() {
    // The name is a label and the digest is the identity. A cache keyed on the
    // name alone would hand back the previous problem for a recipe that has
    // moved, which is the failure this test exists for.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let first = Recipe::seeded(64, 5, 13).expect("a valid shape");
    let second = Recipe::seeded(64, 5, 14).expect("a valid shape");

    let (stored, _) = exchange.ensure("probe", &first).expect("generates");
    let (replaced, outcome) = exchange.ensure("probe", &second).expect("regenerates");
    assert_eq!(outcome, CacheOutcome::Generated);
    assert_ne!(stored.spec_digest(), replaced.spec_digest());
    assert_eq!(replaced.spec_digest(), second.spec_digest());
}

#[test]
fn a_damaged_container_is_regenerated_rather_than_reported() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(64, 5, 13).expect("a valid shape");
    let (expected, _) = exchange.ensure("probe", &recipe).expect("generates");

    std::fs::write(
        exchange.manifest_path("probe").expect("a valid name"),
        b"not a manifest",
    )
    .expect("the file is writable");

    let (recovered, outcome) = exchange.ensure("probe", &recipe).expect("regenerates");
    assert_eq!(outcome, CacheOutcome::Generated);
    assert_eq!(recovered, expected);
}

// ---------------------------------------------------------------------------
// The array table
// ---------------------------------------------------------------------------

#[test]
fn the_array_table_covers_the_file_exactly() {
    let recipe = Recipe::seeded(32, 4, 2)
        .and_then(|recipe| {
            recipe.with_task(Task::Multiclass {
                classes: 3,
                balance: ClassBalance::Imbalanced { ratio: 4.0 },
                geometry: ClassGeometry::Hierarchical,
                separation: 2.0,
            })
        })
        .expect("a valid recipe");
    let container = MaterializedDataset::new(&recipe);

    let mut span = 0;
    for array in container.arrays() {
        assert_eq!(
            array.rows() * array.columns(),
            array.len(),
            "{}",
            array.name()
        );
        span += array.len() * array.dtype().stride();
    }
    assert_eq!(span, container.data_bytes());
    assert!(
        container.arrays().iter().any(|a| a.name() == "features"),
        "every container has a design",
    );
}

#[test]
fn an_inflated_array_length_is_refused() {
    // The manifest is edited to declare an array far longer than the file can
    // hold, keeping both digests valid — the spec digest covers the recipe and
    // the data digest covers the untouched array file, so this reaches the
    // decoder. `tests/dataset_exchange.rs` measures what it allocates on the
    // way to refusing.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(8, 2, 4).expect("a valid shape");
    exchange
        .materialize("probe", &recipe)
        .expect("a valid recipe materializes");

    let path = exchange.manifest_path("probe").expect("a valid name");
    let text = std::fs::read_to_string(&path).expect("the file was just written");
    let edited = text.replace("\"len\": 16", "\"len\": 4000000000");
    assert_ne!(edited, text, "the edit has to reach the table");
    std::fs::write(&path, &edited).expect("the file is writable");

    assert!(matches!(
        exchange.load("probe"),
        Err(ExchangeError::InvalidArrayTable),
    ));
}

#[test]
fn a_declared_length_that_overflows_its_byte_span_is_refused() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(8, 2, 4).expect("a valid shape");
    exchange
        .materialize("probe", &recipe)
        .expect("a valid recipe materializes");

    let path = exchange.manifest_path("probe").expect("a valid name");
    let text = std::fs::read_to_string(&path).expect("the file was just written");
    let edited = text.replace("\"len\": 16", &format!("\"len\": {}", usize::MAX));
    std::fs::write(&path, &edited).expect("the file is writable");

    assert!(matches!(
        exchange.load("probe"),
        Err(ExchangeError::InvalidArrayTable),
    ));
}

#[test]
fn a_non_contiguous_array_table_is_refused() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let exchange = DatasetExchange::new(directory.path());
    let recipe = Recipe::seeded(8, 2, 4)
        .and_then(|recipe| {
            recipe.with_task(Task::LinearRegression {
                informative: 1,
                coefficient_scale: 1.0,
                intercept: 0.0,
                noise_scale: 0.0,
            })
        })
        .expect("a valid recipe");
    exchange
        .materialize("probe", &recipe)
        .expect("a valid recipe materializes");

    let path = exchange.manifest_path("probe").expect("a valid name");
    let text = std::fs::read_to_string(&path).expect("the file was just written");
    let edited = text.replace("\"byte_offset\": 64", "\"byte_offset\": 68");
    assert_ne!(edited, text, "the edit has to reach the table");
    std::fs::write(&path, &edited).expect("the file is writable");

    assert!(matches!(
        exchange.load("probe"),
        Err(ExchangeError::InvalidArrayTable),
    ));
}

#[test]
fn the_word_spelled_vocabularies_survive_the_text() {
    // Every enum the manifest spells as a word, exercised through one recipe
    // each: a label the reader does not know is a refusal, and a label the
    // writer spells differently from the reader is a failed round trip.
    for kind in [
        NonlinearKind::Interaction,
        NonlinearKind::Piecewise,
        NonlinearKind::Sinusoid,
        NonlinearKind::Friedman,
    ] {
        let recipe = Recipe::seeded(32, 6, 1)
            .and_then(|recipe| {
                recipe.with_task(Task::NonlinearRegression {
                    kind,
                    noise_scale: 0.1,
                })
            })
            .expect("a valid recipe");
        assert_eq!(
            manifest::parse(&rendered(&recipe))
                .expect("parses")
                .recipe
                .task(),
            recipe.task(),
        );
    }
    for kind in [
        BinaryKind::Xor,
        BinaryKind::Sinusoid,
        BinaryKind::Circles,
        BinaryKind::Checkerboard,
    ] {
        let recipe = Recipe::seeded(32, 6, 1)
            .and_then(|recipe| {
                recipe.with_task(Task::NonlinearBinary {
                    kind,
                    separation: 2.0,
                    prevalence: 0.4,
                })
            })
            .expect("a valid recipe");
        assert_eq!(
            manifest::parse(&rendered(&recipe))
                .expect("parses")
                .recipe
                .task(),
            recipe.task(),
        );
    }
    for link in [GlmLink::LogCount, GlmLink::LogPositive] {
        let dispersion = match link {
            GlmLink::LogCount => 1.0,
            GlmLink::LogPositive => 0.5,
        };
        let recipe = Recipe::seeded(32, 6, 1)
            .and_then(|recipe| {
                recipe.with_task(Task::GlmRegression {
                    link,
                    informative: 2,
                    coefficient_scale: 0.4,
                    intercept: 0.5,
                    dispersion,
                })
            })
            .expect("a valid recipe");
        assert_eq!(
            manifest::parse(&rendered(&recipe))
                .expect("parses")
                .recipe
                .task(),
            recipe.task(),
        );
    }
}

/// Each typed accessor answers for its own type and refuses the other two.
///
/// `f32_values` was the only one any test reached, because the design is the
/// array every round trip compares. The other two were unasserted outright: a
/// mutation run replaced `u8_values` with `None`, with an empty slice, and with
/// a leaked one-element slice, and every test in this file still passed. That
/// is the accessor a consumer reads a *label* through.
///
/// The refusal half is the part that makes them typed at all. An accessor that
/// answered for every array would hand a caller a reinterpretation of bytes it
/// did not ask for, which is the failure the `dtype` tag exists to prevent, so
/// each array is asked all three questions rather than only its own.
#[test]
fn each_typed_array_accessor_answers_only_for_its_own_type() {
    let recipe = Recipe::seeded(24, 4, 5)
        .and_then(|recipe| {
            recipe.with_task(Task::LinearBinary {
                informative: 2,
                separation: 1.0,
                prevalence: 0.5,
            })
        })
        .and_then(|recipe| recipe.with_groups(GroupPattern::Contiguous { groups: 3 }))
        .expect("a valid recipe");
    let container = MaterializedDataset::new(&recipe);

    let mut seen: Vec<ArrayDtype> = Vec::new();
    for array in container.arrays() {
        let answers = (
            array.f32_values().map(<[f32]>::len),
            array.u8_values().map(<[u8]>::len),
            array.u64_values().map(<[u64]>::len),
        );
        let expected = match array.dtype() {
            ArrayDtype::F32 => (Some(array.len()), None, None),
            ArrayDtype::U8 => (None, Some(array.len()), None),
            ArrayDtype::U64 => (None, None, Some(array.len())),
        };
        assert_eq!(
            answers,
            expected,
            "{} is {:?} and answered {answers:?}",
            array.name(),
            array.dtype()
        );
        // A materialized array holds values, and `is_empty` has to say so:
        // returning a constant would make the emptiness question decorative.
        assert!(!array.is_empty(), "{} decoded as empty", array.name());
        assert_eq!(array.len(), array.rows() * array.columns());
        if !seen.contains(&array.dtype()) {
            seen.push(array.dtype());
        }
    }
    // All three types really are present, so the two arms that used to be
    // unreachable in this file are reached.
    assert_eq!(seen.len(), 3, "the probe container carries {seen:?}");
}

/// Every container refusal says what was wrong, and carries its cause when it
/// has one.
///
/// Neither half was asserted anywhere: `Display` could have rendered the whole
/// enum as the empty string, and `source` could have returned `None` for all of
/// it, with every test in this file still passing. Both matter to a caller —
/// `ferricml-datagen` prints the chain, and an `Io` refusal's own message names
/// the path while its source names what the filesystem said, so a caller
/// diagnosing a run needs both rather than whichever one the outer type
/// happened to render.
///
/// The sibling claim for [`DatasetError`] is
/// `every_error_variant_is_reachable_from_a_constructor` in `tests.rs`; this is
/// the same three properties — non-empty, a sentence fragment, and no two
/// alike — over the type that reads files rather than the one that builds
/// recipes.
#[test]
fn every_container_refusal_says_what_was_wrong_and_carries_its_cause() {
    let refusals = [
        ExchangeError::InvalidName,
        ExchangeError::Io {
            path: std::path::PathBuf::from("/nowhere/probe.bin"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        },
        ExchangeError::SizeLimitExceeded {
            limit: 8,
            actual: 9,
        },
        ExchangeError::MalformedManifest { offset: 17 },
        ExchangeError::UnsupportedFormat { found: 1 },
        ExchangeError::SpecDigestMismatch,
        ExchangeError::DataChecksumMismatch,
        ExchangeError::InvalidArrayTable,
        ExchangeError::InvalidRecipe(DatasetError::ZeroRows),
        ExchangeError::NotRegenerable {
            derivation: Derivation::ReferenceSplit {
                lane: ReferenceLane::Regression,
                seed: 11,
                split: Split::Train,
            },
        },
    ];

    let mut messages: Vec<String> = Vec::new();
    for refusal in &refusals {
        let message = refusal.to_string();
        assert!(!message.is_empty(), "{refusal:?} renders as nothing");
        // A sentence fragment rather than a capitalized sentence, so a caller
        // can embed it in one. Written as "not upper-case" rather than as
        // "lower-case", which is what the sibling claim over `DatasetError`
        // says: an `Io` refusal opens with the path it names, and a path is
        // neither.
        assert!(
            !message.chars().next().is_some_and(char::is_uppercase),
            "refusals read as sentence fragments: {message}"
        );
        assert!(
            !messages.contains(&message),
            "two refusals share the message {message:?}"
        );
        messages.push(message);
    }

    // The chain, where there is one. Both carrying variants are checked against
    // the text their own cause renders, so returning some *other* error would
    // fail as loudly as returning none.
    for refusal in &refusals {
        let cause = std::error::Error::source(refusal).map(ToString::to_string);
        match refusal {
            ExchangeError::Io { source, .. } => {
                assert_eq!(cause.as_deref(), Some(source.to_string().as_str()));
            }
            ExchangeError::InvalidRecipe(error) => {
                assert_eq!(cause.as_deref(), Some(error.to_string().as_str()));
            }
            other => assert!(cause.is_none(), "{other:?} invented a cause: {cause:?}"),
        }
    }
}
