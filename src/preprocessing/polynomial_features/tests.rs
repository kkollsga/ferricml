use super::*;
use crate::data::DenseMatrix;

fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
    DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
}

/// The row every frozen-order assertion below reads: distinct primes, so every
/// monomial is a distinct product and no two columns can be confused.
fn three_features() -> DenseMatrix {
    matrix(&[2.0, 3.0, 5.0], 1, 3)
}

fn fitted(data: &DenseMatrix, params: PolynomialFeaturesParams) -> PolynomialFeatures {
    PolynomialFeatures::fit(&data.as_view(), params).unwrap()
}

fn expanded(params: PolynomialFeaturesParams) -> Vec<f32> {
    let data = three_features();
    fitted(&data, params)
        .transform(&data.as_view())
        .unwrap()
        .as_slice()
        .to_vec()
}

/// Column order is frozen contract, so it is asserted as literals.
///
/// Each expected vector is the expansion of `[2, 3, 5]` written out in the
/// order the documentation promises. Any reordering of the blocks, of the
/// tuples inside a block, or of the bias column moves at least one of these
/// values, because the row's factors are distinct primes.
mod frozen_column_order {
    use super::*;

    #[test]
    fn the_full_expansion_blocks_by_ascending_degree() {
        let base = PolynomialFeaturesParams::default();
        assert_eq!(
            expanded(base.with_degree(1)),
            // 1, x0, x1, x2
            vec![1.0, 2.0, 3.0, 5.0]
        );
        assert_eq!(
            expanded(base.with_degree(2)),
            // 1, x0, x1, x2, x0^2, x0x1, x0x2, x1^2, x1x2, x2^2
            vec![1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0]
        );
        assert_eq!(
            expanded(base.with_degree(3)),
            vec![
                1.0, 2.0, 3.0, 5.0, // degree 0 and 1
                4.0, 6.0, 10.0, 9.0, 15.0, 25.0, // degree 2
                8.0, 12.0, 20.0, 18.0, 30.0, 50.0, 27.0, 45.0, 75.0, 125.0, // degree 3
            ]
        );
    }

    #[test]
    fn interaction_terms_keep_the_order_and_drop_the_repeats() {
        let base = PolynomialFeaturesParams::default().with_interaction_only(true);
        assert_eq!(
            expanded(base.with_degree(2)),
            // 1, x0, x1, x2, x0x1, x0x2, x1x2 — no squares
            vec![1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 15.0]
        );
        assert_eq!(
            expanded(base.with_degree(3)),
            vec![1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 15.0, 30.0]
        );
    }

    #[test]
    fn disabling_the_bias_removes_the_leading_column_and_nothing_else() {
        let base = PolynomialFeaturesParams::default().with_include_bias(false);
        assert_eq!(
            expanded(base.with_degree(2)),
            vec![2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0]
        );
        assert_eq!(
            expanded(base.with_degree(2).with_interaction_only(true)),
            vec![2.0, 3.0, 5.0, 6.0, 10.0, 15.0]
        );

        // Stated as a relation rather than only as two literals: the biased
        // expansion is the unbiased one with a `1.0` in front, whatever the
        // parameters, which is the property a reader would actually rely on.
        for degree in 1..=3 {
            let with = expanded(PolynomialFeaturesParams::default().with_degree(degree));
            let without = expanded(
                PolynomialFeaturesParams::default()
                    .with_degree(degree)
                    .with_include_bias(false),
            );
            assert_eq!(with[0], 1.0);
            assert_eq!(&with[1..], without.as_slice(), "at degree {degree}");
        }
    }

    #[test]
    fn a_single_feature_expands_to_its_own_powers() {
        let data = matrix(&[3.0], 1, 1);
        let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(4));
        assert_eq!(
            model.transform(&data.as_view()).unwrap().as_slice(),
            &[1.0, 3.0, 9.0, 27.0, 81.0]
        );
    }

    #[test]
    fn every_row_is_expanded_independently_and_in_order() {
        let data = matrix(&[2.0, 3.0, 0.0, -1.0], 2, 2);
        let model = fitted(&data, PolynomialFeaturesParams::default());
        assert_eq!(
            model.transform(&data.as_view()).unwrap().as_slice(),
            &[
                1.0, 2.0, 3.0, 4.0, 6.0, 9.0, // row 0
                1.0, 0.0, -1.0, 0.0, -0.0, 1.0, // row 1
            ]
        );
    }
}

/// The width formula is public contract, and the ceiling is where it says.
mod width {
    use super::*;

    fn width(n_features: usize, degree: u32, interaction_only: bool) -> Option<usize> {
        PolynomialFeaturesParams::default()
            .with_degree(degree)
            .with_interaction_only(interaction_only)
            .expanded_width(n_features)
    }

    /// `C(n + d, d)` for the full expansion and `sum C(n, k)` for interactions.
    ///
    /// Both are computed here from an independent definition — a factorial-free
    /// product for the first and an explicit sum for the second — so this
    /// compares two derivations rather than restating one.
    #[test]
    fn the_documented_formulas_hold_across_a_grid() {
        fn choose(n: usize, k: usize) -> usize {
            (1..=k).fold(1, |result, step| result * (n - k + step) / step)
        }
        for n_features in [1_usize, 2, 3, 5, 8] {
            for degree in 1_u32..=4 {
                assert_eq!(
                    width(n_features, degree, false),
                    Some(choose(n_features + degree as usize, degree as usize)),
                    "full expansion at n={n_features}, d={degree}"
                );
                let interactions: usize = (0..=degree as usize)
                    .filter(|&k| k <= n_features)
                    .map(|k| choose(n_features, k))
                    .sum();
                assert_eq!(
                    width(n_features, degree, true),
                    Some(interactions),
                    "interaction expansion at n={n_features}, d={degree}"
                );
            }
        }
    }

    #[test]
    fn the_generated_term_count_is_the_formula_it_promised() {
        // The `debug_assert!` inside `describe` says this too; asserting it
        // here makes it a claim in release builds as well, and the two agreeing
        // is the point — a generator that drifted from the formula would give a
        // fitted model whose `n_features_out` lied about its own terms.
        for n_features in 1_usize..=4 {
            for degree in 0_u32..=3 {
                for interaction_only in [false, true] {
                    for include_bias in [false, true] {
                        let params = PolynomialFeaturesParams::default()
                            .with_degree(degree)
                            .with_interaction_only(interaction_only)
                            .with_include_bias(include_bias);
                        if params.validate().is_err() {
                            continue;
                        }
                        let data = matrix(&vec![1.0; n_features], 1, n_features);
                        let model = fitted(&data, params);
                        assert_eq!(
                            model.term_offsets.len() - 1,
                            model.n_features_out(),
                            "n={n_features} d={degree} inter={interaction_only} \
                             bias={include_bias}"
                        );
                        assert_eq!(
                            model.transform(&data.as_view()).unwrap().as_slice().len(),
                            model.n_features_out()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_degree_zero_expansion_is_the_bias_column_alone() {
        let data = three_features();
        let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(0));
        assert_eq!(model.n_features_out(), 1);
        assert_eq!(model.transform(&data.as_view()).unwrap().as_slice(), &[1.0]);
    }

    /// The ceiling is a documented bound, so both sides of it are asserted.
    #[test]
    fn the_ceiling_is_exactly_where_it_is_documented() {
        assert_eq!(width(999_999, 1, false), Some(1_000_000));
        assert_eq!(
            width(1_000_000, 1, false),
            None,
            "one column past the bound is refused"
        );
        assert_eq!(
            width(999_998, 1, false),
            Some(999_999),
            "and one below it is not"
        );
    }

    /// The failure mode this transformer is scheduled around.
    ///
    /// Fifty features at degree ten is not an exotic request, and it asks for
    /// seventy-five billion output columns. The point of the assertion is that
    /// it is refused *at fit*, with both numbers, rather than at the allocation
    /// that would follow.
    #[test]
    fn an_impossible_width_is_refused_at_fit_with_both_numbers() {
        let data = matrix(&[1.0; 50], 1, 50);
        assert_eq!(
            PolynomialFeatures::fit(
                &data.as_view(),
                PolynomialFeaturesParams::default().with_degree(10)
            )
            .unwrap_err(),
            ModelError::FeatureExpansionOverflow {
                n_features: 50,
                degree: 10,
            }
        );

        // A thousand features at merely degree three is refused for the same
        // reason: 167 million columns.
        let wide = matrix(&vec![1.0; 1000], 1, 1000);
        assert_eq!(
            PolynomialFeatures::fit(
                &wide.as_view(),
                PolynomialFeaturesParams::default().with_degree(3)
            )
            .unwrap_err(),
            ModelError::FeatureExpansionOverflow {
                n_features: 1000,
                degree: 3,
            }
        );

        // And the same width restricted to interactions is buildable, so the
        // refusal is about the width rather than about the parameters.
        assert!(
            PolynomialFeatures::fit(
                &wide.as_view(),
                PolynomialFeaturesParams::default()
                    .with_degree(2)
                    .with_interaction_only(true)
            )
            .is_ok()
        );
    }

    #[test]
    fn an_expansion_with_no_columns_is_refused_where_it_is_described() {
        let data = three_features();
        assert_eq!(
            PolynomialFeatures::fit(
                &data.as_view(),
                PolynomialFeaturesParams::default()
                    .with_degree(0)
                    .with_include_bias(false)
            )
            .unwrap_err(),
            ModelError::EmptyFeatureExpansion
        );
        // Either parameter alone is fine, which is why the check is on the
        // combination.
        assert!(
            PolynomialFeatures::fit(
                &data.as_view(),
                PolynomialFeaturesParams::default().with_degree(0)
            )
            .is_ok()
        );
        assert!(
            PolynomialFeatures::fit(
                &data.as_view(),
                PolynomialFeaturesParams::default().with_include_bias(false)
            )
            .is_ok()
        );
    }
}

/// The finiteness contract, both paths through it.
mod finiteness {
    use super::*;
    use crate::preprocessing::expansion::STACK_SCREEN_FEATURES;

    #[test]
    fn an_overflowing_monomial_is_reported_before_anything_is_written() {
        let data = matrix(&[1.0], 1, 1);
        let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(3));
        // 1e13 squares to 1e26, which is a finite `f32`, and cubes to 1e39,
        // which is not — so the first offending cell is the fourth column and
        // not the third, which is what makes the reported location meaningful.
        let extreme = matrix(&[1e13], 1, 1);
        let mut output = [73.0; 4];
        assert_eq!(
            model
                .transform_into(&extreme.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 0, column: 3 }
        );
        assert_eq!(output, [73.0; 4], "a rejected batch writes nothing");
    }

    #[test]
    fn the_offending_row_is_the_first_one_in_row_major_order() {
        let data = matrix(&[1.0], 1, 1);
        let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(2));
        let batch = matrix(&[2.0, 1e30, 3.0], 3, 1);
        let mut output = [91.0; 9];
        assert_eq!(
            model
                .transform_into(&batch.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 1, column: 2 }
        );
        assert_eq!(output, [91.0; 9]);
    }

    /// A batch too wide for the stack screen still transforms, through the
    /// validation pass.
    ///
    /// This is the only way to reach that pass with a batch that succeeds, and
    /// it matters because the two paths write through different code: one
    /// writes after proving nothing can fail, the other after proving it by
    /// evaluating every cell. Their outputs have to agree.
    #[test]
    fn a_batch_wider_than_the_stack_screen_still_transforms_correctly() {
        let width = STACK_SCREEN_FEATURES + 8;
        let row: Vec<f32> = (0..width).map(|index| index as f32).collect();
        let data = matrix(&row, 1, width);
        let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(1));

        assert!(
            !model.batch_is_proven_finite(&data.as_view()),
            "this batch has to reach the validation pass for the test to mean \
             anything"
        );
        let transformed = model.transform(&data.as_view()).unwrap();
        let mut expected = vec![1.0_f32];
        expected.extend_from_slice(&row);
        assert_eq!(transformed.as_slice(), expected.as_slice());
    }

    #[test]
    fn the_screen_is_an_upper_bound_rather_than_an_estimate() {
        let data = matrix(&[1.0], 1, 1);
        let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(2));
        // Comfortably inside `f32` once squared: the screen proves it.
        assert!(model.batch_is_proven_finite(&matrix(&[1e3, -1e3], 2, 1).as_view()));
        // Squares past `f32::MAX`: the screen refuses to prove it, and the
        // transform then reports the cell rather than writing an infinity.
        assert!(!model.batch_is_proven_finite(&matrix(&[1e30], 1, 1).as_view()));
    }
}

#[test]
fn the_caller_owned_path_matches_the_allocating_one() {
    let data = matrix(&[2.0, 3.0, 5.0, 1.0, -2.0, 0.5], 2, 3);
    let model = fitted(&data, PolynomialFeaturesParams::default().with_degree(3));
    let allocating = model.transform(&data.as_view()).unwrap();
    let mut into = vec![f32::MAX; allocating.as_slice().len()];
    let view = model.transform_into(&data.as_view(), &mut into).unwrap();
    assert_eq!(view.as_slice(), allocating.as_slice());
    assert_eq!(view.rows(), 2);
    assert_eq!(view.columns(), model.n_features_out());
}

#[test]
fn validates_width_and_output_length_before_writing() {
    let data = three_features();
    let model = fitted(&data, PolynomialFeaturesParams::default());
    assert_eq!(model.n_features_out(), 10);

    let mut short = [91.0; 9];
    assert_eq!(
        model
            .transform_into(&data.as_view(), &mut short)
            .unwrap_err(),
        ModelError::OutputLength {
            expected: 10,
            actual: 9
        }
    );
    assert_eq!(short, [91.0; 9]);

    let narrow = matrix(&[1.0, 2.0], 1, 2);
    let mut output = [91.0; 10];
    assert_eq!(
        model
            .transform_into(&narrow.as_view(), &mut output)
            .unwrap_err(),
        ModelError::FeatureDimension {
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(output, [91.0; 10]);
}

#[test]
fn refitting_the_same_width_is_deterministic() {
    let data = three_features();
    let params = PolynomialFeaturesParams::default()
        .with_degree(3)
        .with_interaction_only(true);
    let first = fitted(&data, params);
    let second = fitted(&data, params);
    assert_eq!(first, second);
    assert_eq!(first.get_params(), &params);

    // The fit reads a width and nothing else, so different values of the same
    // shape give the same model. Stating it keeps "fitted" honest.
    let other = fitted(&matrix(&[-1.0, 0.0, 7.5], 1, 3), params);
    assert_eq!(first, other);
}

mod persistence {
    use super::*;

    fn round_tripped(params: PolynomialFeaturesParams) -> PolynomialFeatures {
        let model = fitted(&three_features(), params);
        let bytes = model.to_artifact([1; 32], [2; 32]).unwrap();
        PolynomialFeatures::from_artifact(&bytes, [1; 32], [2; 32]).unwrap()
    }

    #[test]
    fn artifact_is_deterministic_and_schema_bound() {
        let params = PolynomialFeaturesParams::default()
            .with_degree(3)
            .with_interaction_only(true)
            .with_include_bias(false);
        let model = fitted(&three_features(), params);
        let bytes = model.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(bytes, model.to_artifact([1; 32], [2; 32]).unwrap());
        assert_eq!(round_tripped(params), model);
        assert_eq!(
            PolynomialFeatures::from_artifact(&bytes, [3; 32], [2; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
        assert_eq!(
            PolynomialFeatures::from_artifact(&bytes, [1; 32], [9; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );
    }

    #[test]
    fn every_claimed_configuration_round_trips_with_its_terms() {
        for degree in 0_u32..=3 {
            for interaction_only in [false, true] {
                for include_bias in [false, true] {
                    let params = PolynomialFeaturesParams::default()
                        .with_degree(degree)
                        .with_interaction_only(interaction_only)
                        .with_include_bias(include_bias);
                    if params.validate().is_err() {
                        continue;
                    }
                    let model = fitted(&three_features(), params);
                    let decoded = round_tripped(params);
                    assert_eq!(decoded, model, "{params:?}");
                    // The terms are regenerated rather than stored, so the
                    // decoded model has to produce the same columns — that is
                    // the whole claim the omission rests on.
                    let data = three_features();
                    assert_eq!(
                        decoded.transform(&data.as_view()).unwrap().as_slice(),
                        model.transform(&data.as_view()).unwrap().as_slice()
                    );
                }
            }
        }
    }

    #[test]
    fn a_decoded_artifact_re_encodes_to_the_bytes_it_came_from() {
        let params = PolynomialFeaturesParams::default().with_degree(3);
        let model = fitted(&three_features(), params);
        let bytes = model.to_artifact([1; 32], [2; 32]).unwrap();
        let decoded = PolynomialFeatures::from_artifact(&bytes, [1; 32], [2; 32]).unwrap();
        assert_eq!(
            decoded.to_artifact([1; 32], [2; 32]).unwrap(),
            bytes,
            "one fitted expansion has exactly one valid byte string"
        );
    }

    #[test]
    fn artifact_rejects_a_configuration_no_fit_could_have_produced() {
        // A degree whose expansion is past the ceiling. Encoded directly,
        // because `fit` refuses to build such a model in the first place —
        // which is exactly why the decoder cannot rely on having been given
        // one that it did build.
        let hostile = PolynomialFeatures {
            n_features_in: 50,
            params: PolynomialFeaturesParams::default().with_degree(10),
            n_features_out: 1,
            term_features: Vec::new(),
            term_offsets: vec![0, 0],
        };
        let bytes = hostile.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            PolynomialFeatures::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let empty = PolynomialFeatures {
            n_features_in: 3,
            params: PolynomialFeaturesParams::default()
                .with_degree(0)
                .with_include_bias(false),
            n_features_out: 0,
            term_features: Vec::new(),
            term_offsets: vec![0],
        };
        let bytes = empty.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            PolynomialFeatures::from_artifact(&bytes, [1; 32], [2; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );
    }

    #[test]
    fn artifact_rejects_a_flag_word_that_is_neither_zero_nor_one() {
        let model = fitted(&three_features(), PolynomialFeaturesParams::default());
        let mut bytes = model.to_artifact([1; 32], [2; 32]).unwrap();
        // The truncation check comes first for a corrupted tail, so this only
        // needs to establish that a non-boolean flag cannot decode; the fuzz
        // corpus pins the resealed form.
        let flag = bytes.len() - 1;
        bytes[flag] ^= 0x02;
        assert!(PolynomialFeatures::from_artifact(&bytes, [1; 32], [2; 32]).is_err());
    }

    #[test]
    fn a_truncated_artifact_is_rejected() {
        let model = fitted(&three_features(), PolynomialFeaturesParams::default());
        let bytes = model.to_artifact([1; 32], [2; 32]).unwrap();
        assert_eq!(
            PolynomialFeatures::from_artifact(&bytes[..bytes.len() - 1], [1; 32], [2; 32])
                .unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }
}
