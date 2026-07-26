//! The validated containers are a guarantee, not a convention.
//!
//! `MatrixView`, `DenseMatrix`, `BinaryTargets` and `RegressionTargets` exist
//! so that an estimator does not re-derive what its input already promises.
//! That promise is only worth having if no safe caller can produce a container
//! that breaks it, so these tests attack the promise from outside the crate:
//! through the one public trait an external crate can implement, and through
//! every constructor the containers expose.
//!
//! They also pin, deliberately, which `ModelError` variants a caller can still
//! observe. Two of them describe conditions the containers rule out *and* are
//! nevertheless reachable, because several public entry points take a bare
//! `&[f32]` rather than a container. A later reader who removes the guards
//! behind them because "the container already checked" will fail here.

use ferricml::api::{Estimator, ModelError, Transformer};
use ferricml::calibration::{PlattCalibrator, PlattParams};
use ferricml::data::{BinaryTargets, DenseMatrix, MatrixView, RegressionTargets};
use ferricml::dummy::{DummyClassifier, DummyClassifierParams};

/// Finite storage a dishonest transformer can point at instead of its output.
static DECOY: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

/// A transformer that writes one thing and reports another.
///
/// Every line of it is safe code an external crate could write. It exists to
/// hold `Transformer::transform`'s default body to its own documented contract
/// — "a validated view over exactly the values they wrote" — rather than to
/// whatever an implementation left in the buffer it was lent.
struct DishonestTransformer;

impl Estimator for DishonestTransformer {
    fn n_features_in(&self) -> usize {
        2
    }
}

impl Transformer for DishonestTransformer {
    fn n_features_out(&self) -> usize {
        2
    }

    fn transform_into<'output>(
        &self,
        _data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        for slot in output.iter_mut() {
            *slot = f32::NAN;
        }
        Ok(MatrixView::new(&DECOY, 2, 2).expect("the decoy is a valid matrix"))
    }
}

#[test]
fn an_allocating_transform_never_returns_an_unvalidated_buffer() {
    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 2, 2).expect("valid input");

    let produced = DishonestTransformer
        .transform(&data.as_view())
        .expect("the shapes agree, so the call succeeds");

    // The values are the ones the implementation validated, not the ones it
    // wrote into the lent buffer. Both halves matter: the matrix is finite,
    // and it is finite because it came from a `MatrixView`.
    assert_eq!(produced.as_slice(), &DECOY[..]);
    assert!(produced.as_slice().iter().all(|value| value.is_finite()));
}

#[test]
fn a_transformed_matrix_is_accepted_by_an_estimator_without_rescanning_it() {
    use ferricml::tree::{DecisionTreeRegressor, DecisionTreeRegressorParams};

    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 2, 2).expect("valid input");
    let transformed = DishonestTransformer
        .transform(&data.as_view())
        .expect("the shapes agree");
    let targets = RegressionTargets::new(vec![0.0, 1.0]).expect("valid targets");

    // Before the allocating default was fixed this fit saw NaN features and
    // was rejected by a full rescan of the training matrix. The container now
    // carries the guarantee, so the rescan has nothing left to find.
    DecisionTreeRegressor::fit(
        &transformed.as_view(),
        &targets,
        DecisionTreeRegressorParams::default(),
    )
    .expect("a validated matrix is fittable");
}

#[test]
fn empty_data_is_still_reachable_through_a_bare_slice_entry_point() {
    // Calibration fits on decision scores, which are a `&[f32]` rather than a
    // container: nothing has validated them. This is why `EmptyData` stays.
    let targets = BinaryTargets::new(vec![0, 1]).expect("valid targets");
    assert_eq!(
        PlattCalibrator::fit(&[], &targets, PlattParams::default()),
        Err(ModelError::EmptyData)
    );
}

#[test]
fn non_finite_features_are_still_reachable_through_bare_slice_entry_points() {
    let targets = BinaryTargets::new(vec![0, 1]).expect("valid targets");

    // Same reason as above: an unvalidated score slice.
    assert_eq!(
        PlattCalibrator::fit(&[0.5, f32::NAN], &targets, PlattParams::default()),
        Err(ModelError::NonFiniteFeature { row: 1, column: 0 })
    );

    // And the whole `predict_one` family takes a row, not a matrix.
    let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 2, 2).expect("valid input");
    let model =
        DummyClassifier::fit(&data.as_view(), &targets, DummyClassifierParams).expect("valid fit");
    assert_eq!(
        model.predict_one(&[0.0, f32::INFINITY]),
        Err(ModelError::NonFiniteFeature { row: 0, column: 1 })
    );
}
