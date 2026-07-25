use super::*;

#[test]
fn matrix_view_exposes_shape_rows_elements_and_storage() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = MatrixView::new(&values, 2, 3).unwrap();

    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 3);
    assert_eq!(matrix.as_slice(), &values);
    assert_eq!(matrix.row(0), Some(&values[0..3]));
    assert_eq!(matrix.row(1), Some(&values[3..6]));
    assert_eq!(matrix.row(2), None);
    assert_eq!(matrix.get(1, 2), Some(6.0));
    assert_eq!(matrix.get(2, 0), None);
    assert_eq!(matrix.get(0, 3), None);
    assert_eq!(
        matrix.iter_rows().collect::<Vec<_>>(),
        vec![&values[0..3], &values[3..6]]
    );
}

#[test]
fn matrix_rejects_zero_dimensions_before_other_validation() {
    assert_eq!(MatrixView::new(&[], 0, 1), Err(DataError::ZeroRows));
    assert_eq!(MatrixView::new(&[], 1, 0), Err(DataError::ZeroColumns));
    assert_eq!(MatrixView::new(&[], 0, 0), Err(DataError::ZeroRows));
}

#[test]
fn matrix_rejects_dimension_overflow() {
    assert_eq!(
        MatrixView::new(&[], usize::MAX, 2),
        Err(DataError::DimensionOverflow {
            rows: usize::MAX,
            columns: 2,
        })
    );
}

#[test]
fn matrix_requires_exact_buffer_length() {
    assert_eq!(
        MatrixView::new(&[1.0, 2.0, 3.0], 2, 2),
        Err(DataError::LengthMismatch {
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(
        MatrixView::new(&[1.0; 5], 2, 2),
        Err(DataError::LengthMismatch {
            expected: 4,
            actual: 5,
        })
    );
}

#[test]
fn matrices_reject_every_kind_of_non_finite_value() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            MatrixView::new(&[0.0, invalid], 1, 2),
            Err(DataError::NonFiniteValue { index: 1 })
        );
        assert_eq!(
            DenseMatrix::new(vec![0.0, invalid], 1, 2),
            Err(DataError::NonFiniteValue { index: 1 })
        );
    }
}

#[test]
fn dense_matrix_borrows_without_copying_and_can_return_storage() {
    let matrix = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let storage_address = matrix.as_slice().as_ptr();
    let view = matrix.as_view();

    assert_eq!(view.as_slice().as_ptr(), storage_address);
    assert_eq!(matrix.row(1), Some(&[3.0, 4.0][..]));
    assert_eq!(matrix.get(0, 1), Some(2.0));
    assert_eq!(matrix.iter_rows().len(), 2);
    assert_eq!(matrix.into_values(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn dense_matrix_applies_shape_validation() {
    assert_eq!(DenseMatrix::new(vec![], 0, 1), Err(DataError::ZeroRows));
    assert_eq!(DenseMatrix::new(vec![], 1, 0), Err(DataError::ZeroColumns));
    assert_eq!(
        DenseMatrix::new(vec![1.0], 1, 2),
        Err(DataError::LengthMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn binary_targets_accept_only_nonempty_zeroes_and_ones() {
    let targets = BinaryTargets::new(vec![0, 1, 1, 0]).unwrap();
    assert_eq!(targets.len(), 4);
    assert!(!targets.is_empty());
    assert_eq!(targets.as_slice(), &[0, 1, 1, 0]);
    assert_eq!(targets.get(2), Some(1));
    assert_eq!(targets.get(4), None);

    assert_eq!(BinaryTargets::new(vec![]), Err(DataError::EmptyTargets));
    assert_eq!(
        BinaryTargets::new(vec![0, 1, 2, 3]),
        Err(DataError::InvalidBinaryTarget { index: 2, value: 2 })
    );
}

#[test]
fn binary_targets_return_their_storage() {
    let values = vec![1, 0, 1];
    assert_eq!(
        BinaryTargets::new(values.clone()).unwrap().into_values(),
        values
    );
}

#[test]
fn regression_targets_are_nonempty_and_finite() {
    let targets = RegressionTargets::new(vec![-1.5, 0.0, 2.5]).unwrap();
    assert_eq!(targets.len(), 3);
    assert!(!targets.is_empty());
    assert_eq!(targets.as_slice(), &[-1.5, 0.0, 2.5]);
    assert_eq!(targets.get(0), Some(-1.5));
    assert_eq!(targets.get(3), None);

    assert_eq!(RegressionTargets::new(vec![]), Err(DataError::EmptyTargets));
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            RegressionTargets::new(vec![1.0, invalid]),
            Err(DataError::NonFiniteValue { index: 1 })
        );
    }
}

#[test]
fn regression_targets_return_their_storage() {
    let values = vec![1.25, -3.5];
    assert_eq!(
        RegressionTargets::new(values.clone())
            .unwrap()
            .into_values(),
        values
    );
}

#[test]
fn row_and_target_selection_preserve_order_and_repetition() {
    let matrix = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3, 2).unwrap();
    let selected = matrix.select_rows(&[2, 0, 2]).unwrap();
    assert_eq!(selected.rows(), 3);
    assert_eq!(selected.columns(), 2);
    assert_eq!(selected.as_slice(), &[4.0, 5.0, 0.0, 1.0, 4.0, 5.0]);

    let binary = BinaryTargets::new(vec![0, 1, 0]).unwrap();
    assert_eq!(binary.select(&[2, 1, 2]).unwrap().as_slice(), &[0, 1, 0]);
    let regression = RegressionTargets::new(vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(
        regression.select(&[2, 0, 2]).unwrap().as_slice(),
        &[3.0, 1.0, 3.0]
    );
}

#[test]
fn selection_validates_every_index_before_allocating() {
    let matrix = DenseMatrix::new(vec![0.0, 1.0], 2, 1).unwrap();
    assert_eq!(matrix.select_rows(&[]), Err(SelectionError::Empty));
    assert_eq!(
        matrix.select_rows(&[0, 2]),
        Err(SelectionError::IndexOutOfBounds {
            position: 1,
            index: 2,
            available: 2,
        })
    );
    let targets = BinaryTargets::new(vec![0, 1]).unwrap();
    assert_eq!(targets.select(&[]), Err(SelectionError::Empty));
    assert_eq!(
        targets.select(&[2]),
        Err(SelectionError::IndexOutOfBounds {
            position: 0,
            index: 2,
            available: 2,
        })
    );
}

#[test]
fn class_targets_accept_any_label_and_record_the_sorted_observed_set() {
    let targets = ClassTargets::new(vec![7, 3, 10, 3, 7]).unwrap();
    assert_eq!(targets.len(), 5);
    assert!(!targets.is_empty());
    assert_eq!(targets.as_slice(), &[7, 3, 10, 3, 7]);
    assert_eq!(targets.get(1), Some(3));
    assert_eq!(targets.get(5), None);
    // Sorted unique, with gaps preserved and no zero-base assumption.
    assert_eq!(targets.classes(), &[3, 7, 10]);
    assert_eq!(targets.n_classes(), 3);
    assert_eq!(targets.into_values(), vec![7, 3, 10, 3, 7]);

    assert_eq!(ClassTargets::new(Vec::new()), Err(DataError::EmptyTargets));
}

#[test]
fn class_target_columns_are_looked_up_by_label_not_by_value() {
    let targets = ClassTargets::new(vec![5, 20, 9]).unwrap();
    assert_eq!(targets.classes(), &[5, 9, 20]);
    for (index, &label) in targets.classes().iter().enumerate() {
        assert_eq!(targets.class_index(label), Some(index));
    }
    for absent in [0_u8, 4, 6, 10, 19, 21, 255] {
        assert_eq!(targets.class_index(absent), None, "label {absent}");
    }
}

#[test]
fn class_target_extremes_and_full_label_range_are_ordered_numerically() {
    let all = ClassTargets::new((0..=u8::MAX).collect()).unwrap();
    assert_eq!(all.n_classes(), 256);
    assert_eq!(all.classes()[0], 0);
    assert_eq!(all.classes()[255], 255);

    // A single observed class is valid and reports exactly one column.
    let single = ClassTargets::new(vec![200; 4]).unwrap();
    assert_eq!(single.classes(), &[200]);
    assert_eq!(single.n_classes(), 1);

    // Ordering is numeric, so a high label never sorts before a low one.
    let mixed = ClassTargets::new(vec![255, 0, 128]).unwrap();
    assert_eq!(mixed.classes(), &[0, 128, 255]);
}

#[test]
fn class_target_selection_recomputes_the_observed_set() {
    let targets = ClassTargets::new(vec![3, 7, 10, 7]).unwrap();
    let selected = targets.select(&[1, 3, 1]).unwrap();
    assert_eq!(selected.as_slice(), &[7, 7, 7]);
    // The selection saw only one label, so it reports only that label.
    assert_eq!(selected.classes(), &[7]);
    assert_eq!(
        targets.select(&[4]),
        Err(SelectionError::IndexOutOfBounds {
            position: 0,
            index: 4,
            available: 4,
        })
    );
    assert_eq!(targets.select(&[]), Err(SelectionError::Empty));
}

#[test]
fn binary_targets_widen_into_class_targets_without_inventing_a_class() {
    let both = ClassTargets::from(BinaryTargets::new(vec![0, 1, 1]).unwrap());
    assert_eq!(both.as_slice(), &[0, 1, 1]);
    assert_eq!(both.classes(), &[0, 1]);

    // The observed set records what was seen, not what a binary vector allows.
    let one_sided = ClassTargets::from(BinaryTargets::new(vec![0, 0]).unwrap());
    assert_eq!(one_sided.classes(), &[0]);
}

#[test]
fn sample_weights_are_nonempty_finite_nonnegative_and_positive() {
    let weights = SampleWeights::new(vec![0.0, 1.5, 2.5]).unwrap();
    assert_eq!(weights.len(), 3);
    assert!(!weights.is_empty());
    assert_eq!(weights.as_slice(), &[0.0, 1.5, 2.5]);
    assert_eq!(weights.get(1), Some(1.5));
    assert_eq!(weights.get(3), None);
    assert_eq!(weights.total(), 4.0);

    assert_eq!(
        SampleWeights::new(vec![]),
        Err(DataError::EmptySampleWeights)
    );
    assert_eq!(
        SampleWeights::new(vec![1.0, f32::NAN]),
        Err(DataError::NonFiniteSampleWeight { index: 1 })
    );
    assert_eq!(
        SampleWeights::new(vec![1.0, -0.5]),
        Err(DataError::NegativeSampleWeight { index: 1 })
    );
    assert_eq!(
        SampleWeights::new(vec![0.0, 0.0]),
        Err(DataError::ZeroTotalSampleWeight)
    );
}

#[test]
fn sample_weights_return_their_storage() {
    let values = vec![0.5, 1.5];
    assert_eq!(
        SampleWeights::new(values.clone()).unwrap().into_values(),
        values
    );
}

#[test]
fn errors_have_actionable_messages() {
    assert_eq!(
        DataError::ZeroRows.to_string(),
        "matrix row count must be non-zero"
    );
    assert_eq!(
        DataError::InvalidBinaryTarget { index: 4, value: 7 }.to_string(),
        "binary target at index 4 must be 0 or 1, got 7"
    );
}
