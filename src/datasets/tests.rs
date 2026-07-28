use super::*;
use crate::data::{BinaryTargets, RegressionTargets, SampleWeights};

/// The lattice the absorbed benchmark fixture uses, named once so the frozen
/// values below and the recipe under test cannot drift apart.
const FOREST_LATTICE: Source = Source::Lattice {
    row_stride: 131,
    column_stride: 17,
    modulus: 1009,
};

#[test]
fn an_invalid_shape_is_refused_by_name_before_anything_is_generated() {
    assert_eq!(
        Recipe::seeded(0, 4, 7),
        Err(DatasetError::ZeroRows),
        "an empty row count must not reach a generation buffer"
    );
    assert_eq!(Recipe::seeded(4, 0, 7), Err(DatasetError::ZeroColumns));
    assert_eq!(
        Recipe::seeded(usize::MAX, 2, 7),
        Err(DatasetError::DimensionOverflow {
            rows: usize::MAX,
            columns: 2
        }),
        "a shape whose product cannot be represented must be refused rather than \
         wrapped into a small allocation"
    );

    // Column count is checked after row count, so the error names the first
    // problem a caller would fix rather than an arbitrary one.
    assert_eq!(Recipe::seeded(0, 0, 7), Err(DatasetError::ZeroRows));
}

#[test]
fn an_invalid_source_is_refused_by_name_before_anything_is_generated() {
    assert_eq!(
        Recipe::new(4, 4, Source::Xorshift32 { state: 0 }),
        Err(DatasetError::ZeroXorshiftState),
        "zero is xorshift's fixed point, so the design would be one repeated value"
    );
    for modulus in [0_u64, 1] {
        assert_eq!(
            Recipe::new(
                4,
                4,
                Source::Lattice {
                    row_stride: 1,
                    column_stride: 1,
                    modulus
                }
            ),
            Err(DatasetError::LatticeModulusTooSmall { modulus })
        );
    }
    // `2^24` is the boundary, so both sides of it are asserted: one more than
    // the limit is refused and the limit itself is accepted.
    assert_eq!(
        Recipe::new(
            4,
            4,
            Source::Lattice {
                row_stride: 1,
                column_stride: 1,
                modulus: (1 << 24) + 1
            }
        ),
        Err(DatasetError::LatticeModulusNotRepresentable {
            modulus: (1 << 24) + 1,
            limit: 1 << 24,
        })
    );
    assert!(
        Recipe::new(
            4,
            4,
            Source::Lattice {
                row_stride: 1,
                column_stride: 1,
                modulus: 1 << 24
            }
        )
        .is_ok(),
        "the largest exactly representable modulus must be usable"
    );

    // Every state is a valid SplitMix64 state, including zero, because zero is
    // not a fixed point of a mixed-counter generator.
    assert!(Recipe::new(4, 4, Source::Sampled { state: 0 }).is_ok());
}

/// A refused recipe is refused, and a validated one is not re-checked.
///
/// The property that matters is that validation happens at the constructor and
/// nowhere else, which is what lets [`Recipe::design`] and [`Recipe::generate`]
/// return a value rather than a `Result`. Asserted by exhausting the error
/// space: every variant this module can produce comes out of `new` or `seeded`.
#[test]
fn every_error_variant_is_reachable_from_a_constructor() {
    let observed = [
        Recipe::seeded(0, 4, 7).unwrap_err(),
        Recipe::seeded(4, 0, 7).unwrap_err(),
        Recipe::seeded(usize::MAX, 2, 7).unwrap_err(),
        Recipe::new(4, 4, Source::Xorshift32 { state: 0 }).unwrap_err(),
        Recipe::new(
            4,
            4,
            Source::Lattice {
                row_stride: 1,
                column_stride: 1,
                modulus: 1,
            },
        )
        .unwrap_err(),
        Recipe::new(
            4,
            4,
            Source::Lattice {
                row_stride: 1,
                column_stride: 1,
                modulus: u64::MAX,
            },
        )
        .unwrap_err(),
    ];
    // Every one names itself rather than deferring to a debug spelling, and no
    // two share a message.
    let messages: Vec<String> = observed.iter().map(|error| error.to_string()).collect();
    for message in &messages {
        assert!(!message.is_empty());
        assert!(
            message.chars().next().is_some_and(char::is_lowercase),
            "error messages read as sentence fragments: {message}"
        );
    }
    for (index, message) in messages.iter().enumerate() {
        assert!(
            !messages[index + 1..].contains(message),
            "two error variants share the message {message:?}"
        );
    }
}

/// Every source's first design values, pinned.
///
/// These are the literals a port has to reproduce. Two of the three sources are
/// the streams FerricML's frozen reference and benchmark fixtures were recorded
/// against, and the third is derived through `derive_dataset_stream`, so a
/// change to any of them moves data that other files are frozen to. Compared
/// with `assert_eq!` rather than a tolerance because every operation in the maps
/// is exact and any difference at all would be a fixture change.
#[test]
fn the_first_design_values_of_every_source_are_frozen() {
    // The raw SplitMix64 state the reference lanes use.
    let sampled = Recipe::new(2, 3, Source::Sampled { state: 11 })
        .unwrap()
        .design();
    assert_eq!(
        sampled.as_slice(),
        [
            -0.36751127,
            -0.4752698,
            0.27608466,
            0.009227991,
            -0.6696149,
            0.10387528,
        ]
    );

    // The benchmark lattice, at the shape the pinned values below were read at.
    let lattice = Recipe::new(2, 4, FOREST_LATTICE).unwrap().design();
    assert_eq!(
        lattice.as_slice(),
        [
            -1.0,
            -0.9663033,
            -0.9326065,
            -0.8989098,
            -0.74033695,
            -0.70664024,
            -0.6729435,
            -0.6392468,
        ]
    );

    // The two benchmark xorshift states.
    let models = Recipe::new(1, 4, Source::Xorshift32 { state: 0x9e37_79b9 })
        .unwrap()
        .design();
    assert_eq!(
        models.as_slice(),
        [-0.36681294, 0.75141394, -0.0333997, -0.9881696]
    );
    let boosting = Recipe::new(1, 4, Source::Xorshift32 { state: 0x243f_6a88 })
        .unwrap()
        .design();
    assert_eq!(
        boosting.as_slice(),
        [0.78661466, 0.36769938, -0.4735734, 0.7154708]
    );
}

/// The lattice reproduces the benchmark fixture's arithmetic exactly.
///
/// Written as the fixture's own expression rather than as more literals,
/// because what has to be preserved is the *expression*: reducing the two
/// stride terms separately, or forming the divisor differently, gives values
/// that agree almost everywhere and differ where it matters.
#[test]
fn the_lattice_source_reproduces_the_benchmark_fixture_expression() {
    let (rows, columns) = (37, 11);
    let generated = Recipe::new(rows, columns, FOREST_LATTICE).unwrap().design();
    let expected: Vec<f32> = (0..rows)
        .flat_map(|row| {
            (0..columns)
                .map(move |column| (((row * 131 + column * 17) % 1009) as f32 / 504.5) - 1.0)
        })
        .collect();
    assert_eq!(generated.as_slice(), expected);
}

/// The row-major fill order is part of the contract, not an implementation
/// detail.
///
/// A generator-backed source advances its stream once per element, so
/// transposing the loop would permute every value while leaving the multiset of
/// draws identical — a difference no distributional check can see, and exactly
/// the shape of change the reference fixtures exist to catch.
#[test]
fn a_generated_design_is_filled_row_by_row() {
    let wide = Recipe::new(2, 6, Source::Sampled { state: 11 })
        .unwrap()
        .design();
    let tall = Recipe::new(6, 2, Source::Sampled { state: 11 })
        .unwrap()
        .design();
    assert_eq!(
        wide.as_slice(),
        tall.as_slice(),
        "the same stream must fill both shapes in the same order"
    );
    assert_eq!(wide.row(0).unwrap(), &tall.as_slice()[..6]);

    // The lattice has no stream, and its row-major definition is checkable
    // directly: cell (1, 0) is one row stride from the origin and cell (0, 1)
    // one column stride, and both strides are below the modulus so the
    // reduction is the identity at these two cells.
    let lattice = Recipe::new(2, 2, FOREST_LATTICE).unwrap().design();
    assert_eq!(lattice.get(1, 0), Some((131.0_f32 / 504.5) - 1.0));
    assert_eq!(lattice.get(0, 1), Some((17.0_f32 / 504.5) - 1.0));
    assert_ne!(lattice.get(1, 0), lattice.get(0, 1));
}

#[test]
fn every_source_stays_inside_the_signed_unit_interval() {
    let sources = [
        Source::Sampled { state: 11 },
        FOREST_LATTICE,
        Source::Xorshift32 { state: 0x9e37_79b9 },
    ];
    for source in sources {
        let design = Recipe::new(512, 8, source).unwrap().design();
        for (index, &value) in design.as_slice().iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&value),
                "{source:?} escaped [-1, 1] at {index}: {value}"
            );
        }
    }

    // The lattice's lower endpoint is attained exactly — residue zero maps to
    // `-1` — which is what makes the affine map's offset checkable rather than
    // merely plausible.
    let lattice = Recipe::new(1, 1, FOREST_LATTICE).unwrap().design();
    assert_eq!(lattice.as_slice(), [-1.0]);
}

/// A caller-owned buffer produces the same values as the allocating form, and
/// is refilled rather than appended to.
#[test]
fn the_caller_owned_form_matches_the_allocating_one_and_reuses_its_buffer() {
    let recipe = Recipe::seeded(64, 5, 3).unwrap();
    let allocated = recipe.design();

    let mut buffer = vec![f32::MAX; 1_000];
    recipe.design_into(&mut buffer);
    assert_eq!(buffer, allocated.as_slice());

    // A second fill with a different recipe replaces the contents; a buffer
    // that were appended to would be twice as long and start with stale values.
    let other = Recipe::seeded(8, 2, 4).unwrap();
    other.design_into(&mut buffer);
    assert_eq!(buffer.len(), 16);
    assert_eq!(buffer, other.design().as_slice());

    // And the allocation survives, which is the whole reason the form exists.
    let capacity = buffer.capacity();
    recipe.design_into(&mut buffer);
    assert_eq!(buffer.capacity(), capacity);
}

/// Regenerating a recipe inside one process gives identical bytes.
///
/// The cross-process half of this claim is `tests/dataset_generator.rs`, which
/// re-executes the test binary; this half covers the sources whose state is a
/// local generator, where a leaked `static` or a lazily initialized table would
/// show up on the second call rather than the first.
#[test]
fn regenerating_a_recipe_reproduces_its_bytes() {
    for source in [
        Source::Sampled { state: 11 },
        FOREST_LATTICE,
        Source::Xorshift32 { state: 0x243f_6a88 },
    ] {
        let recipe = Recipe::new(128, 7, source).unwrap();
        let first = recipe.generate();
        let second = recipe.generate();
        assert_eq!(first.features(), second.features(), "{source:?}");
        assert_eq!(first.spec_digest(), second.spec_digest(), "{source:?}");
        assert_eq!(first, second, "{source:?}");
    }
}

/// The digest separates every field of a recipe, including the ones a naive
/// concatenation would let bleed into each other.
#[test]
fn the_spec_digest_distinguishes_recipes_that_differ_anywhere() {
    let recipes = [
        Recipe::new(16, 4, Source::Sampled { state: 11 }).unwrap(),
        Recipe::new(4, 16, Source::Sampled { state: 11 }).unwrap(),
        Recipe::new(16, 4, Source::Sampled { state: 12 }).unwrap(),
        // Same numeric state, different source: the discriminant is what has to
        // separate these, since the field bytes are identical.
        Recipe::new(16, 4, Source::Xorshift32 { state: 11 }).unwrap(),
        Recipe::new(
            16,
            4,
            Source::Lattice {
                row_stride: 131,
                column_stride: 17,
                modulus: 1009,
            },
        )
        .unwrap(),
        // Strides swapped. A digest that summed its fields would miss this.
        Recipe::new(
            16,
            4,
            Source::Lattice {
                row_stride: 17,
                column_stride: 131,
                modulus: 1009,
            },
        )
        .unwrap(),
    ];
    for (index, left) in recipes.iter().enumerate() {
        assert_eq!(
            left.spec_digest(),
            left.spec_digest(),
            "the digest is not a function of the recipe alone"
        );
        for right in &recipes[index + 1..] {
            assert_ne!(
                left.spec_digest(),
                right.spec_digest(),
                "{left:?} and {right:?} share a digest"
            );
        }
    }

    // The digest is carried on the generated dataset, so data that has left the
    // recipe behind can still say where it came from.
    let recipe = recipes[0];
    assert_eq!(recipe.generate().spec_digest(), recipe.spec_digest());
}

/// A seeded recipe's stream is the derived one, not the raw seed.
///
/// The disjointness itself is asserted in `src/numeric/rng.rs` against the
/// estimator derivations. What belongs here is the other half: that `seeded`
/// actually routes through that derivation instead of handing the number
/// straight to the generator, which is the transcription error that would make
/// design matrices correlate with model randomness while every test still
/// passed.
#[test]
fn a_seeded_recipe_does_not_reuse_the_raw_state() {
    for seed in [0_u64, 1, 11, 42, u64::MAX] {
        let seeded = Recipe::seeded(16, 4, seed).unwrap();
        let raw = Recipe::new(16, 4, Source::Sampled { state: seed }).unwrap();
        assert_ne!(seeded.source(), raw.source(), "seed {seed}");
        assert_ne!(
            seeded.design().as_slice(),
            raw.design().as_slice(),
            "seed {seed} generated the raw-state design"
        );
    }
}

#[test]
fn recipe_accessors_report_what_was_asked_for() {
    let recipe = Recipe::new(9, 3, FOREST_LATTICE).unwrap();
    assert_eq!(recipe.rows(), 9);
    assert_eq!(recipe.columns(), 3);
    assert_eq!(recipe.source(), FOREST_LATTICE);

    let design = recipe.design();
    assert_eq!(design.rows(), 9);
    assert_eq!(design.columns(), 3);
    assert_eq!(design.as_slice().len(), 27);
}

/// A design-only dataset says it has no task, and a populated one reports every
/// part it was assembled from.
///
/// The populated half uses the crate-private constructor the task families will
/// call, so the container's accessors are exercised before any family exists to
/// exercise them — otherwise P1 would ship five accessors that only ever return
/// `None`.
#[test]
fn a_dataset_reports_the_parts_it_was_assembled_from() {
    let recipe = Recipe::seeded(4, 2, 5).unwrap();
    let design_only = recipe.generate();
    assert!(design_only.target().is_none());
    assert!(design_only.weights().is_none());
    assert!(design_only.groups().is_none());
    assert_eq!(design_only.truth(), &Truth::DesignOnly);
    assert_eq!(design_only.features().rows(), 4);
    assert_eq!(design_only.clone().into_features(), *design_only.features());

    let populated = Dataset::from_parts(
        recipe.design(),
        Some(Target::Binary(
            BinaryTargets::new(vec![0, 1, 1, 0]).unwrap(),
        )),
        Some(SampleWeights::new(vec![1.0, 2.0, 0.5, 1.0]).unwrap()),
        Truth::DesignOnly,
        Some(vec![7, 7, 9, 9]),
        recipe.spec_digest(),
    );
    assert_eq!(
        populated.target(),
        Some(&Target::Binary(
            BinaryTargets::new(vec![0, 1, 1, 0]).unwrap()
        ))
    );
    assert_eq!(
        populated.weights().unwrap().as_slice(),
        [1.0, 2.0, 0.5, 1.0]
    );
    assert_eq!(populated.groups(), Some(&[7, 7, 9, 9][..]));
    assert_eq!(populated.spec_digest(), recipe.spec_digest());

    // The three target vocabularies are all reachable through one container.
    let regression = Dataset::from_parts(
        recipe.design(),
        Some(Target::Regression(
            RegressionTargets::new(vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        )),
        None,
        Truth::DesignOnly,
        None,
        recipe.spec_digest(),
    );
    assert_ne!(regression.target(), populated.target());
}

/// A generated dataset feeds the crate's own containers without an adapter.
///
/// The point of owning the generator is that its output is the input the
/// estimators already take, so this asserts the type-level claim by using it:
/// the design goes to a `MatrixView` and the groups to the `&[u64]` the grouped
/// splitters accept.
#[test]
fn a_generated_design_is_the_input_the_estimators_already_take() {
    let recipe = Recipe::seeded(32, 4, 2).unwrap();
    let dataset = recipe.generate();
    let view = dataset.features().as_view();
    assert_eq!(view.rows(), 32);
    assert_eq!(view.iter_rows().count(), 32);
    assert!(view.as_slice().iter().all(|value| value.is_finite()));
}

/// A debug build refuses a `Dataset` whose parts disagree about the row count.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "target length disagrees with the design's row count")]
fn a_dataset_whose_target_is_the_wrong_length_is_a_defect() {
    let recipe = Recipe::seeded(4, 2, 5).unwrap();
    let _ = Dataset::from_parts(
        recipe.design(),
        Some(Target::Binary(BinaryTargets::new(vec![0, 1]).unwrap())),
        None,
        Truth::DesignOnly,
        None,
        recipe.spec_digest(),
    );
}
