use super::*;
use crate::data::{BinaryTargets, RegressionTargets, SampleWeights};
// The lattice the absorbed benchmark fixture draws from, taken from the module
// that owns it rather than restated, so the frozen values below and the recipe
// under test cannot drift apart.
use super::benchmarks::FOREST_LATTICE;
// The field classification, read out of the encoder that owns it rather than
// restated here: a sweep that carried its own copy of the partition would prove
// only that the copy agrees with itself.
use super::recipe::task_field_counts;
use sha2::{Digest, Sha256};

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
        None,
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
        None,
        recipe.spec_digest(),
    );
}

/// Every absorbed lane's design split, at every recorded seed.
///
/// `(seed, first four training values, first four test values)`. The test
/// values are the load-bearing half: they are the stream *continuing* past the
/// training half's `768 * 12` draws rather than restarting, which is what the
/// lanes did and what a preset generating two independent matrices would get
/// wrong while every distributional check stayed green.
const ABSORBED_DESIGN_HEADS: [(u64, [f32; 4], [f32; 4]); 5] = [
    (
        11,
        [-0.36751127, -0.4752698, 0.27608466, 0.009227991],
        [-0.96033, -0.71677804, 0.7148739, 0.5721407],
    ),
    (
        22,
        [0.56292343, 0.8718723, -0.84690225, 0.34488785],
        [0.27923572, -0.7704772, 0.28242874, 0.6317868],
    ),
    (
        33,
        [-0.65582097, -0.84953594, -0.40368485, -0.68200266],
        [0.87418437, 0.59509933, 0.49366927, -0.52509165],
    ),
    (
        44,
        [0.9630481, 0.13257527, -0.22201598, 0.5625949],
        [-0.88748443, -0.3912977, 0.7844597, 0.47235596],
    ),
    (
        55,
        [-0.14003551, -0.9380609, 0.65851283, 0.8496196],
        [-0.042396903, 0.77195096, -0.9948883, -0.5530999],
    ),
];

/// `(lane, seed, training fold, test fold, training positives, test positives)`
/// for every absorbed classification lane.
///
/// The folds cover all `768` and `384` labels; the positive counts are carried
/// beside them because a fold says only *that* something moved and a prevalence
/// says *what*, which is the first thing a reader of a failure wants.
const ABSORBED_BINARY_LABELS: [(ReferenceLane, u64, u64, u64, usize, usize); 20] = [
    (
        ReferenceLane::NonlinearBinary,
        11,
        0x7795_0006_415e_4b9d,
        0x6e71_2035_961b_bd7d,
        458,
        200,
    ),
    (
        ReferenceLane::NonlinearBinary,
        22,
        0xc045_289f_e807_c5f7,
        0xab4a_4564_f91f_0cb0,
        466,
        225,
    ),
    (
        ReferenceLane::NonlinearBinary,
        33,
        0xdad2_b19d_a98e_89bc,
        0x033f_4ce0_6dc8_a8f9,
        417,
        210,
    ),
    (
        ReferenceLane::NonlinearBinary,
        44,
        0xa6e1_14e4_2c6b_f700,
        0x085e_53ce_f8ff_b406,
        429,
        229,
    ),
    (
        ReferenceLane::NonlinearBinary,
        55,
        0x798c_e185_a011_484d,
        0x5aa3_ef67_93ac_35bf,
        428,
        222,
    ),
    (
        ReferenceLane::SeparableBinary,
        11,
        0x1ecd_f024_7d6a_3b14,
        0x8193_69ac_b040_139b,
        377,
        214,
    ),
    (
        ReferenceLane::SeparableBinary,
        22,
        0x2021_13d9_9c14_a2ba,
        0xba12_e06c_23b5_2bcf,
        379,
        186,
    ),
    (
        ReferenceLane::SeparableBinary,
        33,
        0x99d8_222a_6495_6ac0,
        0x66f4_5382_dad1_617f,
        379,
        192,
    ),
    (
        ReferenceLane::SeparableBinary,
        44,
        0x8d36_31aa_76c4_ed09,
        0xf3f2_5d4c_8b49_bdbd,
        408,
        198,
    ),
    (
        ReferenceLane::SeparableBinary,
        55,
        0x2f66_6c0d_2496_7d76,
        0x628d_5571_0a8e_5a66,
        361,
        183,
    ),
    (
        ReferenceLane::ImbalancedBinary,
        11,
        0xc1af_532e_7d1b_2493,
        0xac6b_fa7a_a1c0_e542,
        44,
        21,
    ),
    (
        ReferenceLane::ImbalancedBinary,
        22,
        0x4b2f_17a1_3033_bc98,
        0x9f3d_ca02_c600_32fa,
        43,
        35,
    ),
    (
        ReferenceLane::ImbalancedBinary,
        33,
        0x49ab_ca78_46c0_691b,
        0x0f11_0d24_3607_3bcd,
        42,
        18,
    ),
    (
        ReferenceLane::ImbalancedBinary,
        44,
        0xd634_76bb_34dc_8620,
        0x8f1e_5716_42fb_99b3,
        59,
        22,
    ),
    (
        ReferenceLane::ImbalancedBinary,
        55,
        0x4f3c_34d4_a74e_c65d,
        0x771e_44ce_4194_45a0,
        44,
        13,
    ),
    (
        ReferenceLane::NoisyBinary,
        11,
        0xdb47_0405_c479_9a98,
        0x3e13_accc_f567_eaad,
        383,
        192,
    ),
    (
        ReferenceLane::NoisyBinary,
        22,
        0xa128_7cfb_20ac_7ccf,
        0x407d_0ad6_2d53_f05d,
        382,
        188,
    ),
    (
        ReferenceLane::NoisyBinary,
        33,
        0xde1b_2582_6aa1_2ea7,
        0xfd58_e30a_cc43_1181,
        380,
        186,
    ),
    (
        ReferenceLane::NoisyBinary,
        44,
        0x7032_5df5_8e41_b7f5,
        0x8cb2_6518_4295_d0e1,
        386,
        194,
    ),
    (
        ReferenceLane::NoisyBinary,
        55,
        0xc11b_815f_9cd7_4693,
        0x32cc_f403_0775_4539,
        384,
        192,
    ),
];

/// `(seed, training fold, test fold)` over the regression lane's target bits.
const ABSORBED_REGRESSION_TARGETS: [(u64, u64, u64); 5] = [
    (11, 0x1160_f627_8a9f_460b, 0x83bb_becc_e30d_a8ce),
    (22, 0xefac_d5f8_172d_c2f3, 0x09e7_244e_1933_60aa),
    (33, 0xe256_c637_fd8a_4f7c, 0xb499_529d_fd38_ab6b),
    (44, 0x45aa_3065_fee3_d9eb, 0xae0e_d639_30a9_aa72),
    (55, 0x7cf6_d731_f6ab_21ba, 0x555f_9ab8_877a_62b8),
];

/// FNV-1a over whole `u64` words, used only to pin a long vector by one number.
///
/// It is order-sensitive and value-sensitive, which is the whole requirement: a
/// permuted or altered label vector gives a different fold. Nothing depends on
/// its cryptographic strength — the folds below were captured from the lane
/// functions this module replaced, and they exist so a `1152`-value vector can
/// be frozen without pasting `1152` literals.
fn fold_u64(values: impl Iterator<Item = u64>) -> u64 {
    let mut accumulator: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        accumulator = (accumulator ^ value).wrapping_mul(0x0100_0000_01b3);
    }
    accumulator
}

/// The labels of a dataset a binary lane produced.
fn binary_labels(dataset: &Dataset) -> &[u8] {
    match dataset.target() {
        Some(Target::Binary(targets)) => targets.as_slice(),
        other => panic!("a binary lane produced {other:?}"),
    }
}

/// The targets of a dataset the regression lane produced.
fn regression_values(dataset: &Dataset) -> &[f32] {
    match dataset.target() {
        Some(Target::Regression(targets)) => targets.as_slice(),
        other => panic!("the regression lane produced {other:?}"),
    }
}

/// The absorbed lanes reproduce, by value, what the functions they replaced
/// produced.
///
/// This is the evidence the port rests on, and it is deliberately not an
/// aggregate one. The lanes it replaces are consumed by quality tests that
/// compare accuracy and Brier against a frozen reference within `0.02`, so a
/// preset emitting a *different but similarly distributed* stream would pass
/// every one of them while changing every design matrix and every label — the
/// trap `tests/reference_semantics.rs` records at the frozen-stream test. Every
/// literal below was captured by running the outgoing functions before they were
/// deleted, in the commit that deleted them.
#[test]
fn the_absorbed_lanes_reproduce_their_recorded_values() {
    for (seed, train_head, test_head) in ABSORBED_DESIGN_HEADS {
        for lane in [
            ReferenceLane::NonlinearBinary,
            ReferenceLane::SeparableBinary,
            ReferenceLane::ImbalancedBinary,
            ReferenceLane::NoisyBinary,
            ReferenceLane::Regression,
        ] {
            // The design is the lane's own stream and does not depend on which
            // task is drawn over it, which is asserted here rather than assumed
            // because it is what lets one recipe describe all five lanes.
            let preset = ReferenceQuality::new(lane, seed);
            let train = preset.train();
            let test = preset.test();
            assert_eq!(train.features().rows(), ReferenceQuality::TRAIN_ROWS);
            assert_eq!(test.features().rows(), ReferenceQuality::TEST_ROWS);
            assert_eq!(
                &train.features().as_slice()[..4],
                train_head,
                "{lane:?} seed {seed} training design moved"
            );
            assert_eq!(
                &test.features().as_slice()[..4],
                test_head,
                "{lane:?} seed {seed} test design moved — the test half must continue \
                 the training half's stream, not restart it"
            );
            assert_eq!(train.truth(), &Truth::Unrecorded);
            assert_eq!(train.spec_digest(), preset.recipe().spec_digest());
            assert_eq!(test.spec_digest(), preset.recipe().spec_digest());
        }
    }

    for (lane, seed, train_fold, test_fold, train_positives, test_positives) in
        ABSORBED_BINARY_LABELS
    {
        let preset = ReferenceQuality::new(lane, seed);
        let train = preset.train();
        let test = preset.test();
        let (train_labels, test_labels) = (binary_labels(&train), binary_labels(&test));
        assert_eq!(train_labels.len(), ReferenceQuality::TRAIN_ROWS);
        assert_eq!(test_labels.len(), ReferenceQuality::TEST_ROWS);
        assert_eq!(
            train_labels.iter().filter(|&&label| label == 1).count(),
            train_positives,
            "{lane:?} seed {seed} training prevalence moved"
        );
        assert_eq!(
            test_labels.iter().filter(|&&label| label == 1).count(),
            test_positives,
            "{lane:?} seed {seed} test prevalence moved"
        );
        assert_eq!(
            fold_u64(train_labels.iter().map(|&label| u64::from(label))),
            train_fold,
            "{lane:?} seed {seed} training labels moved"
        );
        assert_eq!(
            fold_u64(test_labels.iter().map(|&label| u64::from(label))),
            test_fold,
            "{lane:?} seed {seed} test labels moved"
        );
    }

    for (seed, train_fold, test_fold) in ABSORBED_REGRESSION_TARGETS {
        let preset = ReferenceQuality::new(ReferenceLane::Regression, seed);
        let train = preset.train();
        let test = preset.test();
        assert_eq!(
            fold_u64(
                regression_values(&train)
                    .iter()
                    .map(|value| u64::from(value.to_bits()))
            ),
            train_fold,
            "regression seed {seed} training targets moved"
        );
        assert_eq!(
            fold_u64(
                regression_values(&test)
                    .iter()
                    .map(|value| u64::from(value.to_bits()))
            ),
            test_fold,
            "regression seed {seed} test targets moved"
        );
    }
}

/// The head of every lane's label and target vector, spelled out.
///
/// The folds above pin the whole vectors and this pins the first twelve values
/// of each, at the first recorded seed. It exists because a fold failure says
/// only that something moved: these literals say *what* the lane is supposed to
/// emit, and are what a reader compares against when one of them does move.
#[test]
fn the_absorbed_lanes_emit_their_recorded_first_values() {
    let heads: [(ReferenceLane, [u8; 12], [u8; 12]); 4] = [
        (
            ReferenceLane::NonlinearBinary,
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
        ),
        (
            ReferenceLane::SeparableBinary,
            [1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
        ),
        (
            ReferenceLane::ImbalancedBinary,
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        // The noisy lane's two splits agree on all twelve, and that is a
        // property rather than a coincidence: its noise term is four times the
        // linear signal and depends only on the row index and the seed, both of
        // which restart with the split. A preset indexing the test rows
        // continuously from 768 would disagree here on the first row.
        (
            ReferenceLane::NoisyBinary,
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        ),
    ];
    for (lane, train_head, test_head) in heads {
        let preset = ReferenceQuality::new(lane, 11);
        assert_eq!(binary_labels(&preset.train())[..12], train_head, "{lane:?}");
        assert_eq!(binary_labels(&preset.test())[..12], test_head, "{lane:?}");
    }

    let regression = ReferenceQuality::new(ReferenceLane::Regression, 11);
    assert_eq!(
        regression_values(&regression.train())[..6],
        [
            -0.9404716,
            0.3970253,
            0.05040382,
            -1.2404964,
            1.5710618,
            0.061550044
        ]
    );
    assert_eq!(
        regression_values(&regression.test())[..6],
        [
            -1.5648069, -1.180205, 0.89713323, 1.1617364, 0.24731648, -0.5884442
        ]
    );
}

/// A preset's design is exactly the recipe it names, split by row.
///
/// The two splits are halves of one generated matrix rather than two separately
/// seeded ones, and this asserts the identity directly instead of inferring it
/// from the pinned heads: concatenating them reproduces the recipe's own output
/// value for value.
#[test]
fn a_preset_splits_one_generated_design_rather_than_seeding_two() {
    let preset = ReferenceQuality::new(ReferenceLane::NonlinearBinary, 33);
    let recipe = preset.recipe();
    assert_eq!(recipe.source(), Source::Sampled { state: 33 });
    assert_eq!(
        recipe.rows(),
        ReferenceQuality::TRAIN_ROWS + ReferenceQuality::TEST_ROWS
    );
    assert_eq!(recipe.columns(), ReferenceQuality::COLUMNS);

    let whole = recipe.design();
    let mut rejoined = preset.train().into_features().as_slice().to_vec();
    rejoined.extend_from_slice(preset.test().features().as_slice());
    assert_eq!(rejoined, whole.as_slice());
}

/// A preset's stream is the raw seed, not the derived one.
///
/// This is the single transcription error that would move every frozen fixture
/// while leaving the port looking correct: `Recipe::seeded` is the right
/// constructor for new work and the wrong one here.
#[test]
fn a_preset_names_the_raw_state_and_not_a_derived_stream() {
    for seed in [11_u64, 22, 33, 44, 55] {
        let preset = ReferenceQuality::new(ReferenceLane::SeparableBinary, seed);
        assert_eq!(preset.seed(), seed);
        assert_eq!(preset.recipe().source(), Source::Sampled { state: seed });
        assert_ne!(
            preset.recipe().source(),
            Recipe::seeded(16, 4, seed).unwrap().source(),
            "seed {seed} reached the derivation, which would move every fixture"
        );
    }
}

/// Each lane draws a different target over the same design.
///
/// Without this the five lanes could collapse onto one another — the same
/// expression reached through five variants — and every pinned fold would still
/// be whatever that one expression emits.
#[test]
fn the_lanes_are_distinct_tasks_over_one_design() {
    let seed = 11;
    let binary = [
        ReferenceLane::NonlinearBinary,
        ReferenceLane::SeparableBinary,
        ReferenceLane::ImbalancedBinary,
        ReferenceLane::NoisyBinary,
    ];
    let designs: Vec<Vec<f32>> = binary
        .iter()
        .map(|&lane| {
            ReferenceQuality::new(lane, seed)
                .train()
                .into_features()
                .as_slice()
                .to_vec()
        })
        .collect();
    for design in &designs[1..] {
        assert_eq!(design, &designs[0], "the lanes must share one design");
    }

    let labels: Vec<Vec<u8>> = binary
        .iter()
        .map(|&lane| binary_labels(&ReferenceQuality::new(lane, seed).train()).to_vec())
        .collect();
    for (index, left) in labels.iter().enumerate() {
        for right in &labels[index + 1..] {
            assert_ne!(left, right, "two lanes produced the same labels");
        }
    }

    let regression = ReferenceQuality::new(ReferenceLane::Regression, seed).train();
    assert_eq!(
        regression.features().as_slice(),
        designs[0],
        "the regression lane must share the same design"
    );
    assert!(
        matches!(regression.target(), Some(Target::Regression(_))),
        "the regression lane must not threshold"
    );
}

/// The absorbed lanes carry targets and still report no ground truth.
///
/// `DesignOnly` would be a false statement — a task *was* drawn — and inventing
/// a coefficient vector for a thresholded polynomial would be a claim this
/// module cannot support. The third variant is what says so, and this is what
/// keeps a later family from quietly reusing it.
#[test]
fn an_absorbed_lane_reports_a_task_without_a_recorded_truth() {
    let preset = ReferenceQuality::new(ReferenceLane::NoisyBinary, 44);
    let train = preset.train();
    assert!(train.target().is_some());
    assert_eq!(train.truth(), &Truth::Unrecorded);
    assert_ne!(train.truth(), &Truth::DesignOnly);
    assert!(train.weights().is_none());
    assert!(train.groups().is_none());

    // A bare recipe over the same stream draws no task at all, which is the
    // other statement.
    assert_eq!(preset.recipe().generate().truth(), &Truth::DesignOnly);
}

/// Generating a preset twice gives the same bytes.
#[test]
fn regenerating_a_preset_reproduces_its_bytes() {
    let preset = ReferenceQuality::new(ReferenceLane::Regression, 22);
    assert_eq!(preset.train(), preset.train());
    assert_eq!(preset.test(), preset.test());
    assert_eq!(preset, ReferenceQuality::new(ReferenceLane::Regression, 22));
    assert_ne!(preset, ReferenceQuality::new(ReferenceLane::Regression, 23));
    assert_ne!(
        preset,
        ReferenceQuality::new(ReferenceLane::NonlinearBinary, 22)
    );
}

/// Every absorbed benchmark fixture, digested at every shape its suite calls it
/// at.
///
/// `(lane, rows, columns, design digest, target digest)`. The design digest is
/// SHA-256 over the whole value vector's little-endian `f32` bytes, in row-major
/// order; the target digest is over the label bytes for a binary lane and over
/// the little-endian `f32` bytes for a regression one.
///
/// Every literal here was captured by running the private `fixture` functions in
/// `benches/forest.rs`, `benches/models.rs` and `benches/boosting.rs` — extracted
/// mechanically from those files rather than retyped — before this module
/// replaced them, in the commit that deleted them. The shapes are the ones the
/// benches actually call: `2048x64` and `512x16` for the forest suite, the five
/// shapes between `256x8` and `1024x512` the model suite reaches, and `2048x48`
/// for the boosted one.
const ABSORBED_BENCHMARK_DIGESTS: [(BenchmarkLane, usize, usize, &str, &str); 10] = [
    (
        BenchmarkLane::ForestBinary,
        2048,
        64,
        "8e72b704312cd30f8dd8ddcd322099134b46e9653e308268d09cfa5a16e59b3b",
        "d4abb5ae935dc4f94fa3653ddad024bd8aff57e83df56fd5730766446edcccf1",
    ),
    (
        BenchmarkLane::ForestRegression,
        2048,
        64,
        "8e72b704312cd30f8dd8ddcd322099134b46e9653e308268d09cfa5a16e59b3b",
        "e1ec9d23210ffc7e899e4c65352de893538b00fae2e5b4d42d081e8dd4b2acf0",
    ),
    (
        BenchmarkLane::ForestBinary,
        512,
        16,
        "8185887127313368be00e385f6aee76f5161a799c775e67700a50b37122e2779",
        "bca462f253414db5361faa5fd48a5189024e58972ad4e1e313f342d49b9a3956",
    ),
    (
        BenchmarkLane::ForestRegression,
        512,
        16,
        "8185887127313368be00e385f6aee76f5161a799c775e67700a50b37122e2779",
        "9d78970fca27880448b6176b9c018e453b4b30be282734ebd7166e0153d00262",
    ),
    (
        BenchmarkLane::ModelsRegression,
        2048,
        48,
        "025f1786748d2a5c2b03c827582b984b98023074f02807633370b5cad4e55560",
        "55cf8811d3ac7d3c86bdc96473aeefae750838561cb54f902cff14c87fd13a9a",
    ),
    (
        BenchmarkLane::ModelsRegression,
        1024,
        48,
        "34b7d132a47abb19125a2c1e70a4a1c862919ec730aa74f40ca6b1b85e1dcbb1",
        "6e28d765fa4544e4535edcd4f3aefa4f71aac2788deded4377625bca71ab3f01",
    ),
    (
        BenchmarkLane::ModelsRegression,
        1024,
        512,
        "767748fc61311a61b8f29b9d6ddda43749eea888d0b1fcbf0b661be1814fab38",
        "7e67a5e78ee278c5f4c011db966655ecd8beb47d840314ebcea0d7279af99cf0",
    ),
    (
        BenchmarkLane::ModelsRegression,
        256,
        12,
        "e1f456af845dc0e8fdbeecfce81203845e0125d294321cb79c7ca812061ff3a5",
        "c18efa2cf9fc614f18807a1e552b87d72acb6641d6d42795474c79bd55915236",
    ),
    (
        BenchmarkLane::ModelsRegression,
        256,
        8,
        "881a3ba08880b65fa46f0a57139d07cc98839a006377d54bbcf77a277ecbb19e",
        "9cfd72e3f14f69aedd7e696bd1f9b936e034d3cab05d50bc1dc469e5041d23d0",
    ),
    (
        BenchmarkLane::BoostingRegression,
        2048,
        48,
        "873e751520bec55ca6e145b51b0e8c57743bd97de78d90a728ae31bf27299d68",
        "e8e61ce111f762e4074de7a30a4b5e4eaf412b3543682ae71e67882f6226cd04",
    ),
];

/// Lowercase hexadecimal, so a moved digest reads as a digest in the failure.
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// SHA-256 over a design matrix's little-endian `f32` bytes, in row-major order.
fn design_digest(dataset: &Dataset) -> String {
    let mut digest = Sha256::new();
    for value in dataset.features().as_slice() {
        digest.update(value.to_le_bytes());
    }
    hex(digest.finalize().into())
}

/// SHA-256 over a target vector, in whichever vocabulary the lane drew.
fn target_digest(dataset: &Dataset) -> String {
    let mut digest = Sha256::new();
    match dataset.target() {
        Some(Target::Binary(targets)) => digest.update(targets.as_slice()),
        Some(Target::Regression(targets)) => {
            for value in targets.as_slice() {
                digest.update(value.to_le_bytes());
            }
        }
        other => panic!("a benchmark lane produced {other:?}"),
    }
    hex(digest.finalize().into())
}

/// The absorbed benchmark fixtures emit the bytes their originals emitted.
///
/// This is the whole evidence P3 rests on, and it has to be byte identity rather
/// than anything weaker. `bench-history` compares each release against immutable
/// per-release results at a `1.10` ratio limit, so a fixture that changed by one
/// value would leave every historical baseline non-comparable — and unlike a
/// quality lane, a *timing* lane cannot notice that at all: a differently
/// distributed design of the same shape runs at very nearly the same speed while
/// meaning something else entirely.
#[test]
fn the_absorbed_benchmark_fixtures_reproduce_their_recorded_bytes() {
    for (lane, rows, columns, design, target) in ABSORBED_BENCHMARK_DIGESTS {
        let dataset = BenchmarkFixture::new(lane, rows, columns)
            .unwrap()
            .generate();
        assert_eq!(dataset.features().rows(), rows);
        assert_eq!(dataset.features().columns(), columns);
        assert_eq!(
            design_digest(&dataset),
            design,
            "{lane:?} {rows}x{columns} design moved — every bench-history baseline \
             measured on it is now non-comparable"
        );
        assert_eq!(
            target_digest(&dataset),
            target,
            "{lane:?} {rows}x{columns} target moved"
        );
        assert_eq!(dataset.truth(), &Truth::Unrecorded);
    }
}

/// The head of every absorbed benchmark fixture, spelled out.
///
/// The digests above pin the whole vectors; these say *what* each lane is
/// supposed to emit, which is what a reader compares against when a digest does
/// move. The forest heads are the same at both shapes because the lattice's first
/// row is the same cells at any width — an identity worth seeing rather than
/// rediscovering.
#[test]
fn the_absorbed_benchmark_fixtures_emit_their_recorded_first_values() {
    let forest_design_head = [
        -1.0, -0.9663033, -0.9326065, -0.8989098, -0.8652131, -0.8315164, -0.7978196, -0.7641229,
    ];
    for (rows, columns, positives) in [(2048_usize, 64_usize, 1023_usize), (512, 16, 253)] {
        let binary = BenchmarkFixture::new(BenchmarkLane::ForestBinary, rows, columns)
            .unwrap()
            .generate();
        assert_eq!(&binary.features().as_slice()[..8], forest_design_head);
        let labels = match binary.target() {
            Some(Target::Binary(targets)) => targets.as_slice(),
            other => panic!("the forest binary lane produced {other:?}"),
        };
        assert_eq!(&labels[..12], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(
            labels.iter().filter(|&&label| label == 1).count(),
            positives,
            "the forest fixture's prevalence at {rows}x{columns} moved"
        );

        let regression = BenchmarkFixture::new(BenchmarkLane::ForestRegression, rows, columns)
            .unwrap()
            .generate();
        assert_eq!(
            regression.features().as_slice(),
            binary.features().as_slice(),
            "the two forest lanes must share one design"
        );
        let values = match regression.target() {
            Some(Target::Regression(targets)) => targets.as_slice(),
            other => panic!("the forest regression lane produced {other:?}"),
        };
        // Four units of class separation on top of a sawtooth in the row index:
        // rows 0-3 are negative and rows 4-7 positive, so the second group is the
        // first group plus four.
        assert_eq!(&values[..8], [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]);
    }

    let models = BenchmarkFixture::new(BenchmarkLane::ModelsRegression, 2048, 48)
        .unwrap()
        .generate();
    assert_eq!(
        &models.features().as_slice()[..8],
        [
            -0.36681294,
            0.75141394,
            -0.0333997,
            -0.9881696,
            0.7984444,
            0.94951844,
            0.45425916,
            0.9764383,
        ]
    );
    assert_eq!(
        match models.target() {
            Some(Target::Regression(targets)) => &targets.as_slice()[..8],
            other => panic!("the models lane produced {other:?}"),
        },
        [
            -1.7168499, -2.4857368, -3.8307328, 1.2398754, 0.8283434, -2.685411, -1.7159778,
            0.19300494,
        ]
    );

    let boosting = BenchmarkFixture::new(BenchmarkLane::BoostingRegression, 2048, 48)
        .unwrap()
        .generate();
    assert_eq!(
        &boosting.features().as_slice()[..8],
        [
            0.78661466,
            0.36769938,
            -0.4735734,
            0.7154708,
            -0.82400614,
            0.5394697,
            0.3655858,
            -0.81354034,
        ]
    );
    assert_eq!(
        match boosting.target() {
            Some(Target::Regression(targets)) => &targets.as_slice()[..8],
            other => panic!("the boosting lane produced {other:?}"),
        },
        [
            -1.0230119,
            -1.7656684,
            2.1884053,
            -1.4441595,
            -0.42575365,
            1.5665402,
            1.3162644,
            -3.4956913,
        ]
    );
}

/// A shorter fixture is a prefix of a longer one at the same width.
///
/// The benches rely on this without stating it: `benches/models.rs` trains on
/// `2048x48` and measures inference on `1024x48`, and `benches/boosting.rs`
/// slices the first `32` and `1024` rows out of its training matrix. Both are
/// only measuring the same data because every source fills row by row from a
/// fixed start.
#[test]
fn a_shorter_benchmark_fixture_is_a_prefix_of_a_longer_one() {
    for lane in [
        BenchmarkLane::ForestBinary,
        BenchmarkLane::ForestRegression,
        BenchmarkLane::ModelsRegression,
        BenchmarkLane::BoostingRegression,
    ] {
        let long = BenchmarkFixture::new(lane, 2048, 48).unwrap().generate();
        let short = BenchmarkFixture::new(lane, 1024, 48).unwrap().generate();
        assert_eq!(
            short.features().as_slice(),
            &long.features().as_slice()[..1024 * 48],
            "{lane:?} is not a prefix of itself at a taller shape"
        );
    }

    // The width is not a prefix dimension: widening the design moves every value
    // after the first row, because the source advances once per element.
    let narrow = BenchmarkFixture::new(BenchmarkLane::ModelsRegression, 4, 8)
        .unwrap()
        .generate();
    let wide = BenchmarkFixture::new(BenchmarkLane::ModelsRegression, 4, 12)
        .unwrap()
        .generate();
    assert_eq!(
        narrow.features().row(0).unwrap(),
        &wide.features().row(0).unwrap()[..8]
    );
    assert_ne!(
        narrow.features().row(1).unwrap(),
        &wide.features().row(1).unwrap()[..8]
    );
}

/// A benchmark fixture reports what it was asked for and refuses what it cannot
/// generate.
#[test]
fn a_benchmark_fixture_validates_its_shape_and_names_its_lane() {
    let fixture = BenchmarkFixture::new(BenchmarkLane::ForestBinary, 512, 16).unwrap();
    assert_eq!(fixture.lane(), BenchmarkLane::ForestBinary);
    assert_eq!(fixture.recipe().rows(), 512);
    assert_eq!(fixture.recipe().columns(), 16);
    assert_eq!(fixture.recipe().source(), FOREST_LATTICE);
    assert_eq!(
        fixture.generate().spec_digest(),
        fixture.recipe().spec_digest()
    );

    // The two xorshift lanes are different streams, which is what stops the two
    // suites from measuring the same matrix under two names.
    let models = BenchmarkFixture::new(BenchmarkLane::ModelsRegression, 4, 8).unwrap();
    let boosting = BenchmarkFixture::new(BenchmarkLane::BoostingRegression, 4, 8).unwrap();
    assert_ne!(models.recipe().source(), boosting.recipe().source());
    assert_ne!(
        models.generate().features().as_slice(),
        boosting.generate().features().as_slice()
    );

    // Validation is the recipe's, so a refused shape is refused by name before
    // anything is generated.
    assert_eq!(
        BenchmarkFixture::new(BenchmarkLane::ModelsRegression, 0, 8),
        Err(DatasetError::ZeroRows)
    );
    assert_eq!(
        BenchmarkFixture::new(BenchmarkLane::ForestBinary, 8, 0),
        Err(DatasetError::ZeroColumns)
    );
}

/// The two forest lanes draw different tasks over one design.
///
/// Without this they could collapse onto each other and both digests would still
/// be whatever the surviving expression emits.
#[test]
fn the_forest_lanes_are_two_tasks_over_one_design() {
    let binary = BenchmarkFixture::new(BenchmarkLane::ForestBinary, 64, 8)
        .unwrap()
        .generate();
    let regression = BenchmarkFixture::new(BenchmarkLane::ForestRegression, 64, 8)
        .unwrap()
        .generate();
    assert_eq!(binary.features(), regression.features());
    assert!(matches!(binary.target(), Some(Target::Binary(_))));
    assert!(matches!(regression.target(), Some(Target::Regression(_))));

    // The regression target is derived from the binary lane's own labels, so the
    // class separation is recoverable from it exactly.
    let labels = match binary.target() {
        Some(Target::Binary(targets)) => targets.as_slice(),
        other => panic!("{other:?}"),
    };
    let values = match regression.target() {
        Some(Target::Regression(targets)) => targets.as_slice(),
        other => panic!("{other:?}"),
    };
    for (row, (&label, &value)) in labels.iter().zip(values).enumerate() {
        assert_eq!(value, f32::from(label) * 4.0 + (row % 11) as f32);
    }
}

/// Which side of the task partition a field falls on, and what that costs the
/// design.
///
/// Three values rather than two, because two dials do reach the design matrix
/// and calling them structural would hide the property that makes them dials:
/// the draw underneath them is unchanged, and what moved is a closed-form
/// transform of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// In the stream digest. The two recipes are two different problems.
    Structural,
    /// Out of the stream digest, and the design is byte-identical across it.
    Dial,
    /// Out of the stream digest, and applied to the design as a transform of an
    /// unchanged draw. The transform itself is asserted by
    /// `family_tests::a_conditioning_dial_scales_a_fixed_draw` and
    /// `structural_tests::a_spread_dial_scatters_a_fixed_draw`.
    DesignDial,
}

/// One field of one task, moved on its own.
struct FieldMove {
    /// The field's name, for the failure message.
    field: &'static str,
    /// The task before the move.
    left: Task,
    /// The task after it.
    right: Task,
    /// Which side of the partition the field is on.
    role: Role,
    /// How many fields this move covers. One, except where two fields cannot be
    /// varied independently at a fixed shape.
    covers: usize,
}

impl FieldMove {
    /// A move of exactly one field.
    fn one(field: &'static str, left: Task, right: Task, role: Role) -> Self {
        Self {
            field,
            left,
            right,
            role,
            covers: 1,
        }
    }
}

/// Every field of one family, each moved on its own, over one shape.
struct FamilySweep {
    rows: usize,
    columns: usize,
    /// The unmoved task, whose field counts the sweep is held to.
    base: Task,
    moves: Vec<FieldMove>,
}

/// A dial holds the stream and a structural field moves it, for every field of
/// every family.
///
/// This is the assertion the partition exists to make, and it is made per field
/// rather than sampled. For a dial: the two recipes share a stream digest, the
/// design is byte-identical, and the dataset still says something different —
/// the knob moved what it names and nothing else. For a structural field: the
/// stream digest differs, which is the honest answer when the two recipes
/// describe different problems. For both: the *spec* digest differs, because the
/// data does and a cache keyed on it must not serve one for the other.
///
/// # Why it cannot go stale
///
/// The table below is data, and data rots. What stops it is that
/// [`task_field_counts`] reads the classification out of `encode_task` itself:
/// each family's moves must cover exactly as many structural fields and exactly
/// as many dials as the encoder classified. A field added to a `Task` variant
/// does not compile until `encode_task` classifies it, and the moment it does,
/// this test fails until the table covers it. The compiler enforces that a field
/// *has* a role; this enforces that the role was checked against behaviour.
///
/// # The two joint moves
///
/// Two fields cannot be varied alone at a fixed shape, and both are declared as
/// covering what they move rather than pretended otherwise.
///
/// `Task::Ranking`'s `queries` and `docs_per_query` multiply to the row count, so
/// no recipe differs in one of them alone. They are swept as a swapped pair
/// covering both fields.
///
/// `Task::GlmRegression`'s `link` and `dispersion` have disjoint admissible
/// ranges — a Poisson count cannot be under-dispersed and a positive continuous
/// response cannot be over-dispersed by one — so no recipe differs in the link
/// alone either. That pair is still conclusive rather than merely suggestive:
/// the `dispersion` move directly below it holds the stream fixed on its own, so
/// the stream that the joint move disturbs was disturbed by the link.
#[test]
fn every_task_dial_holds_the_stream_and_every_structural_field_moves_it() {
    let sweeps = task_partition_sweeps();

    // The catalogue rule the suites follow: a family that exists is a family
    // that is swept, and the roster is the compiler-backed one.
    for family in Family::ALL {
        assert!(
            sweeps.iter().any(|sweep| sweep.base.family() == family),
            "{family:?} has no field sweep"
        );
    }

    for sweep in &sweeps {
        let family = sweep.base.family();
        let counts = task_field_counts(sweep.base);
        let covered = |role: Role| -> usize {
            sweep
                .moves
                .iter()
                .filter(|entry| entry.role == role)
                .map(|entry| entry.covers)
                .sum()
        };
        assert_eq!(
            covered(Role::Structural),
            counts.structural,
            "{family:?} sweeps {} structural fields, the encoder classified {}",
            covered(Role::Structural),
            counts.structural
        );
        assert_eq!(
            covered(Role::Dial) + covered(Role::DesignDial),
            counts.dials,
            "{family:?} sweeps {} dials, the encoder classified {}",
            covered(Role::Dial) + covered(Role::DesignDial),
            counts.dials
        );

        for entry in &sweep.moves {
            let name = entry.field;
            let left = Recipe::seeded(sweep.rows, sweep.columns, 41)
                .unwrap()
                .with_task(entry.left)
                .unwrap();
            let right = Recipe::seeded(sweep.rows, sweep.columns, 41)
                .unwrap()
                .with_task(entry.right)
                .unwrap();

            // Whatever the role, the recipe's *identity* moves: the data does,
            // so a cache keyed on the digest must not serve one for the other.
            assert_ne!(
                left.spec_digest(),
                right.spec_digest(),
                "{family:?}.{name} left the spec digest where it was"
            );

            match entry.role {
                Role::Structural => assert_ne!(
                    left.stream_digest(),
                    right.stream_digest(),
                    "{family:?}.{name} is structural and must redraw"
                ),
                Role::Dial | Role::DesignDial => assert_eq!(
                    left.stream_digest(),
                    right.stream_digest(),
                    "{family:?}.{name} is a dial and reseeded the streams"
                ),
            }

            let (left_design, right_design) = (left.design(), right.design());
            match entry.role {
                Role::Dial => {
                    assert_eq!(
                        left_design.as_slice(),
                        right_design.as_slice(),
                        "{family:?}.{name} is a dial and moved the design"
                    );
                    // "The target changed" is too narrow for one dial and would
                    // have to be excepted rather than stated: `Task::Ranking`'s
                    // `coefficient_scale` multiplies every utility by one
                    // positive constant, and a sort cannot notice that, so the
                    // grades and the pairs are identical and only the recorded
                    // utilities move. What every dial must do is change *what
                    // the dataset says*, in the target or in the truth.
                    let (left_data, right_data) = (left.generate(), right.generate());
                    assert!(
                        left_data.target() != right_data.target()
                            || left_data.truth() != right_data.truth(),
                        "{family:?}.{name} is a dial that changed nothing"
                    );
                }
                Role::DesignDial => assert_ne!(
                    left_design.as_slice(),
                    right_design.as_slice(),
                    "{family:?}.{name} shapes the design and left it unmoved"
                ),
                Role::Structural => {}
            }
        }
    }
}

/// One move of every field of every family.
///
/// The shapes are wide enough that a dial which moved only a handful of drawn
/// labels still shows up as a changed target vector, and small enough that the
/// whole sweep is a few milliseconds.
fn task_partition_sweeps() -> Vec<FamilySweep> {
    const ROWS: usize = 256;
    const COLUMNS: usize = 6;

    let linear = |informative, coefficient_scale, intercept, noise_scale| Task::LinearRegression {
        informative,
        coefficient_scale,
        intercept,
        noise_scale,
    };
    let glm = |link, informative, coefficient_scale, intercept, dispersion| Task::GlmRegression {
        link,
        informative,
        coefficient_scale,
        intercept,
        dispersion,
    };
    let conditioned =
        |condition_number, rank, coefficient_scale, noise_scale| Task::IllConditioned {
            condition_number,
            rank,
            coefficient_scale,
            noise_scale,
        };
    let binary = |informative, separation, prevalence| Task::LinearBinary {
        informative,
        separation,
        prevalence,
    };
    let curved = |kind, separation, prevalence| Task::NonlinearBinary {
        kind,
        separation,
        prevalence,
    };
    let multiclass = |classes, balance, geometry, separation| Task::Multiclass {
        classes,
        balance,
        geometry,
        separation,
    };
    let timed = |informative, coefficient_scale, drift, intercept, noise_scale| Task::TimeOrdered {
        informative,
        coefficient_scale,
        drift,
        intercept,
        noise_scale,
    };
    let ranked = |queries, docs_per_query, grades, informative, coefficient_scale| Task::Ranking {
        queries,
        docs_per_query,
        grades,
        informative,
        coefficient_scale,
    };

    vec![
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: linear(2, 1.0, 0.25, 0.1),
            moves: vec![
                FieldMove::one(
                    "informative",
                    linear(2, 1.0, 0.25, 0.1),
                    linear(3, 1.0, 0.25, 0.1),
                    Role::Structural,
                ),
                FieldMove::one(
                    "coefficient_scale",
                    linear(2, 1.0, 0.25, 0.1),
                    linear(2, 2.0, 0.25, 0.1),
                    Role::Dial,
                ),
                FieldMove::one(
                    "intercept",
                    linear(2, 1.0, 0.25, 0.1),
                    linear(2, 1.0, 0.75, 0.1),
                    Role::Dial,
                ),
                FieldMove::one(
                    "noise_scale",
                    linear(2, 1.0, 0.25, 0.1),
                    linear(2, 1.0, 0.25, 0.3),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: Task::NonlinearRegression {
                kind: NonlinearKind::Interaction,
                noise_scale: 0.1,
            },
            moves: vec![
                FieldMove::one(
                    "kind",
                    Task::NonlinearRegression {
                        kind: NonlinearKind::Interaction,
                        noise_scale: 0.1,
                    },
                    Task::NonlinearRegression {
                        kind: NonlinearKind::Piecewise,
                        noise_scale: 0.1,
                    },
                    Role::Structural,
                ),
                FieldMove::one(
                    "noise_scale",
                    Task::NonlinearRegression {
                        kind: NonlinearKind::Interaction,
                        noise_scale: 0.1,
                    },
                    Task::NonlinearRegression {
                        kind: NonlinearKind::Interaction,
                        noise_scale: 0.3,
                    },
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.3),
            moves: vec![
                FieldMove::one(
                    "link",
                    glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.3),
                    glm(GlmLink::LogCount, 2, 0.5, 0.0, 1.5),
                    Role::Structural,
                ),
                FieldMove::one(
                    "informative",
                    glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.3),
                    glm(GlmLink::LogPositive, 3, 0.5, 0.0, 0.3),
                    Role::Structural,
                ),
                FieldMove::one(
                    "coefficient_scale",
                    glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.3),
                    glm(GlmLink::LogPositive, 2, 1.0, 0.0, 0.3),
                    Role::Dial,
                ),
                FieldMove::one(
                    "intercept",
                    glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.3),
                    glm(GlmLink::LogPositive, 2, 0.5, 0.5, 0.3),
                    Role::Dial,
                ),
                FieldMove::one(
                    "dispersion",
                    glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.3),
                    glm(GlmLink::LogPositive, 2, 0.5, 0.0, 0.7),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: conditioned(10.0, 4, 1.0, 0.1),
            moves: vec![
                FieldMove::one(
                    "condition_number",
                    conditioned(10.0, 4, 1.0, 0.1),
                    conditioned(1000.0, 4, 1.0, 0.1),
                    Role::DesignDial,
                ),
                FieldMove::one(
                    "rank",
                    conditioned(10.0, 4, 1.0, 0.1),
                    conditioned(10.0, 5, 1.0, 0.1),
                    Role::Structural,
                ),
                FieldMove::one(
                    "coefficient_scale",
                    conditioned(10.0, 4, 1.0, 0.1),
                    conditioned(10.0, 4, 2.0, 0.1),
                    Role::Dial,
                ),
                FieldMove::one(
                    "noise_scale",
                    conditioned(10.0, 4, 1.0, 0.1),
                    conditioned(10.0, 4, 1.0, 0.3),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: binary(2, 1.0, 0.3),
            moves: vec![
                FieldMove::one(
                    "informative",
                    binary(2, 1.0, 0.3),
                    binary(3, 1.0, 0.3),
                    Role::Structural,
                ),
                FieldMove::one(
                    "separation",
                    binary(2, 1.0, 0.3),
                    binary(2, 2.0, 0.3),
                    Role::Dial,
                ),
                FieldMove::one(
                    "prevalence",
                    binary(2, 1.0, 0.3),
                    binary(2, 1.0, 0.6),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: curved(BinaryKind::Xor, 1.0, 0.3),
            moves: vec![
                FieldMove::one(
                    "kind",
                    curved(BinaryKind::Xor, 1.0, 0.3),
                    curved(BinaryKind::Circles, 1.0, 0.3),
                    Role::Structural,
                ),
                FieldMove::one(
                    "separation",
                    curved(BinaryKind::Xor, 1.0, 0.3),
                    curved(BinaryKind::Xor, 2.0, 0.3),
                    Role::Dial,
                ),
                FieldMove::one(
                    "prevalence",
                    curved(BinaryKind::Xor, 1.0, 0.3),
                    curved(BinaryKind::Xor, 1.0, 0.6),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: multiclass(3, ClassBalance::Balanced, ClassGeometry::Blob, 1.0),
            moves: vec![
                FieldMove::one(
                    "classes",
                    multiclass(3, ClassBalance::Balanced, ClassGeometry::Blob, 1.0),
                    multiclass(4, ClassBalance::Balanced, ClassGeometry::Blob, 1.0),
                    Role::Structural,
                ),
                FieldMove::one(
                    "geometry",
                    multiclass(3, ClassBalance::Balanced, ClassGeometry::Blob, 1.0),
                    multiclass(3, ClassBalance::Balanced, ClassGeometry::Hierarchical, 1.0),
                    Role::Structural,
                ),
                FieldMove::one(
                    "balance",
                    multiclass(3, ClassBalance::Balanced, ClassGeometry::Blob, 1.0),
                    multiclass(
                        3,
                        ClassBalance::Imbalanced { ratio: 4.0 },
                        ClassGeometry::Blob,
                        1.0,
                    ),
                    Role::Dial,
                ),
                FieldMove::one(
                    "separation",
                    multiclass(3, ClassBalance::Balanced, ClassGeometry::Blob, 1.0),
                    multiclass(3, ClassBalance::Balanced, ClassGeometry::Blob, 3.0),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: Task::Clustered {
                blobs: 3,
                spread: 0.1,
            },
            moves: vec![
                FieldMove::one(
                    "blobs",
                    Task::Clustered {
                        blobs: 3,
                        spread: 0.1,
                    },
                    Task::Clustered {
                        blobs: 4,
                        spread: 0.1,
                    },
                    Role::Structural,
                ),
                FieldMove::one(
                    "spread",
                    Task::Clustered {
                        blobs: 3,
                        spread: 0.1,
                    },
                    Task::Clustered {
                        blobs: 3,
                        spread: 0.4,
                    },
                    Role::DesignDial,
                ),
            ],
        },
        FamilySweep {
            rows: ROWS,
            columns: COLUMNS,
            base: timed(2, 1.0, 0.5, 0.25, 0.1),
            moves: vec![
                FieldMove::one(
                    "informative",
                    timed(2, 1.0, 0.5, 0.25, 0.1),
                    timed(3, 1.0, 0.5, 0.25, 0.1),
                    Role::Structural,
                ),
                FieldMove::one(
                    "coefficient_scale",
                    timed(2, 1.0, 0.5, 0.25, 0.1),
                    timed(2, 2.0, 0.5, 0.25, 0.1),
                    Role::Dial,
                ),
                FieldMove::one(
                    "drift",
                    timed(2, 1.0, 0.5, 0.25, 0.1),
                    timed(2, 1.0, 1.5, 0.25, 0.1),
                    Role::Dial,
                ),
                FieldMove::one(
                    "intercept",
                    timed(2, 1.0, 0.5, 0.25, 0.1),
                    timed(2, 1.0, 0.5, 0.75, 0.1),
                    Role::Dial,
                ),
                FieldMove::one(
                    "noise_scale",
                    timed(2, 1.0, 0.5, 0.25, 0.1),
                    timed(2, 1.0, 0.5, 0.25, 0.3),
                    Role::Dial,
                ),
            ],
        },
        FamilySweep {
            rows: 240,
            columns: COLUMNS,
            base: ranked(60, 4, 3, 2, 1.0),
            moves: vec![
                FieldMove {
                    field: "queries and docs_per_query",
                    left: ranked(60, 4, 3, 2, 1.0),
                    right: ranked(40, 6, 3, 2, 1.0),
                    role: Role::Structural,
                    covers: 2,
                },
                FieldMove::one(
                    "grades",
                    ranked(60, 4, 3, 2, 1.0),
                    ranked(60, 4, 4, 2, 1.0),
                    Role::Structural,
                ),
                FieldMove::one(
                    "informative",
                    ranked(60, 4, 3, 2, 1.0),
                    ranked(60, 4, 3, 3, 1.0),
                    Role::Structural,
                ),
                FieldMove::one(
                    "coefficient_scale",
                    ranked(60, 4, 3, 2, 1.0),
                    ranked(60, 4, 3, 2, 2.0),
                    Role::Dial,
                ),
            ],
        },
    ]
}

/// The frozen presets and benchmark fixtures carry no task at all, so no dial of
/// theirs could have moved.
///
/// Verified rather than assumed. The partition changed what a recipe *carrying a
/// task* draws, and the whole safety argument for the absorbed streams is that
/// none of them carries one: `encode_task` writes tag zero and nothing else for
/// `None`, and `Recipe::design_into` and `Recipe::generate` never form a stream
/// digest without a task, so a taskless recipe cannot read the changed encoding
/// even by accident.
///
/// The pinned literals in this file are the other half of the evidence and are
/// byte-unchanged, which only means something because this test says the reason.
#[test]
fn the_frozen_presets_and_benchmark_fixtures_carry_no_task() {
    for lane in ReferenceLane::ALL {
        for seed in [11_u64, 22, 33, 44, 55] {
            let preset = ReferenceQuality::new(lane, seed);
            assert_eq!(
                preset.recipe().task(),
                None,
                "{lane:?} at seed {seed} acquired a task"
            );
        }
    }
    for (lane, rows, columns, _, _) in ABSORBED_BENCHMARK_DIGESTS {
        let fixture = BenchmarkFixture::new(lane, rows, columns).unwrap();
        assert_eq!(
            fixture.recipe().task(),
            None,
            "{lane:?} at {rows}x{columns} acquired a task"
        );
    }
}
