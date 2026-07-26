//! Caller-supplied elementwise transformation.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::data::MatrixView;

use super::scaling::{inverse_transform_allocating, validate_transform_request};

/// A caller-supplied elementwise map.
///
/// A plain function pointer rather than a boxed closure, deliberately — see
/// [`FunctionTransformer`] for why.
pub type ElementwiseFn = fn(f32) -> f32;

/// Parameters for [`FunctionTransformer`].
///
/// Deliberately **not** `PartialEq`: see [`FunctionTransformer`].
#[derive(Clone, Copy, Debug)]
pub struct FunctionTransformerParams {
    func: ElementwiseFn,
    inverse_func: Option<ElementwiseFn>,
}

/// The identity, which is what a [`FunctionTransformer`] applies by default.
fn identity(value: f32) -> f32 {
    value
}

impl Default for FunctionTransformerParams {
    /// The identity in both directions.
    ///
    /// A default that did anything to the data would be a transformation nobody
    /// asked for, so the useful default is the one that changes nothing and is
    /// exactly invertible.
    fn default() -> Self {
        Self {
            func: identity,
            inverse_func: Some(identity),
        }
    }
}

impl FunctionTransformerParams {
    /// Sets the map applied to every value.
    ///
    /// Setting a forward map clears any previously supplied inverse, because an
    /// inverse belongs to the map it inverts; keeping the old one would pair a
    /// new function with an inverse of a different one.
    #[must_use]
    pub const fn with_func(mut self, func: ElementwiseFn) -> Self {
        self.func = func;
        self.inverse_func = None;
        self
    }

    /// Sets the map that undoes [`FunctionTransformerParams::with_func`].
    ///
    /// FerricML does not check that the two are actually inverse to each other.
    /// Verifying it would mean choosing a tolerance and a set of probe points on
    /// the caller's behalf, and being wrong about either would either reject a
    /// correct pair or accept an incorrect one. The pairing is the caller's
    /// claim; what FerricML guarantees is that `inverse_transform` applies
    /// exactly the function supplied here and nothing else.
    #[must_use]
    pub const fn with_inverse_func(mut self, inverse_func: ElementwiseFn) -> Self {
        self.inverse_func = Some(inverse_func);
        self
    }

    /// Returns the map applied to every value.
    #[must_use]
    pub const fn func(&self) -> ElementwiseFn {
        self.func
    }

    /// Returns the map that undoes it, when one was supplied.
    #[must_use]
    pub const fn inverse_func(&self) -> Option<ElementwiseFn> {
        self.inverse_func
    }
}

/// Applies a caller-supplied function to every value.
///
/// # Why a function pointer rather than a closure
///
/// The map is a `fn(f32) -> f32`, so this type is not generic over a closure.
/// That is a deliberate narrowing, for reasons that are worth stating because
/// the generic version looks more Rust-native at first glance:
///
/// - **A capability declaration lives on a nameable type.**
///   [`HasCapabilities::CAPABILITIES`] is an associated constant, and FerricML's
///   capability snapshot asserts that every public type declaring capabilities
///   appears in it by name. A type instantiated at an unnameable closure type
///   could not be listed, so it would silently fall out of the snapshot's
///   coverage — a contract quietly losing a member is worse than a contract that
///   covers slightly less.
/// - **Fitted types are `Clone` and `Debug`**, which pipelines, the conformance
///   battery, and the artifact tests all rely on. Function pointers are both,
///   and are `Copy` besides; closure types are neither reliably.
/// - **A closure can capture state.** Two values of one type could then behave
///   differently, which would quietly break the promise that identical data and
///   parameters produce an identical fitted result. A bare function pointer
///   captures nothing.
///
/// Nothing is lost by the restriction. [`Transformer`] is a public trait, so a
/// caller who genuinely needs captured state, or a map that reads a whole row,
/// implements it on their own type — which is the honest way to say "this
/// transformation is mine, not FerricML's".
///
/// # Determinism is the caller's obligation
///
/// FerricML promises an identical fitted result for identical data, parameters,
/// seed, and thread count. For this transformer that promise covers the framing
/// and **not** the supplied function: values are visited in a fixed row-major
/// order, the output length and feature width are validated before anything is
/// written, and a finite input mapping to a non-finite output is reported as
/// [`ModelError::NonFiniteTransform`] naming the first offending cell rather
/// than being written into the caller's buffer.
///
/// What FerricML cannot guarantee is that the supplied function is pure. A
/// function reading a clock, a global, or an environment variable will produce a
/// transformer that is not reproducible, and no part of this type can detect
/// that. Supplying a deterministic function is a caller obligation, stated here
/// so it is a documented boundary rather than an assumption.
///
/// # Persistence
///
/// There is none, and this type declares no artifact capability. A function
/// pointer is an address in the current process image: encoding one would
/// produce bytes that mean nothing in another build, and could not be validated
/// on the way back in. A caller needing a persistable transformation uses a
/// fitted transformer whose parameters are values.
///
/// # Why there is no `PartialEq`
///
/// Every other fitted type in the crate compares by value. This one cannot, and
/// rather than derive an implementation that only looks like it does, it has
/// none. Comparing function pointers compares *addresses*, and one function is
/// not guaranteed to have one address: code placement moves with optimization
/// settings and across crate boundaries, so two transformers built from the
/// same `fn` could compare unequal for reasons that have nothing to do with
/// what they compute. The compiler warns about precisely this.
///
/// An equality that is right in the common case and quietly wrong at a boundary
/// is worse than no equality at all, because a caller reaching for `==` wants
/// to know whether two transformers *do the same thing*, which is not a
/// question an address can answer. Compare behaviour instead: transform a batch
/// with each and compare the outputs.
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::preprocessing::{FunctionTransformer, FunctionTransformerParams};
///
/// fn double(value: f32) -> f32 {
///     value * 2.0
/// }
/// fn halve(value: f32) -> f32 {
///     value / 2.0
/// }
///
/// let data = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let transformer = FunctionTransformer::fit(
///     &data.as_view(),
///     FunctionTransformerParams::default()
///         .with_func(double)
///         .with_inverse_func(halve),
/// )?;
///
/// let doubled = transformer.transform(&data.as_view())?;
/// assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
///
/// let restored = transformer.inverse_transform(&doubled.as_view())?;
/// assert_eq!(restored.as_slice(), data.as_slice());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// The map is **elementwise**. A transformation that must read a whole row or
/// column is expressed by implementing [`Transformer`](crate::api::Transformer)
/// directly, which is the honest way to say the transformation is the caller's.
#[derive(Clone, Copy, Debug)]
pub struct FunctionTransformer {
    n_features_in: usize,
    params: FunctionTransformerParams,
}

impl FunctionTransformer {
    /// Records the width this transformer accepts.
    ///
    /// Nothing is estimated from `data`; the supplied function is applied as
    /// given.
    pub fn fit(
        data: &MatrixView<'_>,
        params: FunctionTransformerParams,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            n_features_in: data.columns(),
            params,
        })
    }

    /// Returns the fitted input width.
    #[must_use]
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the fitted output width.
    #[must_use]
    pub const fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    /// Returns the exact transformation parameters.
    #[must_use]
    pub const fn get_params(&self) -> &FunctionTransformerParams {
        &self.params
    }

    /// Transforms a batch into caller-owned row-major storage.
    pub fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        <Self as Transformer>::transform_into(self, data, output)
    }

    /// Transforms a batch into a newly allocated dense matrix.
    pub fn transform(&self, data: &MatrixView<'_>) -> Result<crate::data::DenseMatrix, ModelError> {
        <Self as Transformer>::transform(self, data)
    }

    /// Undoes [`FunctionTransformer::transform`] into caller-owned storage.
    ///
    /// Returns [`ModelError::NoInverseFunction`] when no inverse was supplied.
    /// A missing inverse is refused rather than silently treated as the
    /// identity, which would return the transformed values while looking like a
    /// recovery of the originals.
    ///
    /// FerricML applies exactly the function supplied and does not verify that
    /// it inverts the forward map; that pairing is the caller's claim.
    pub fn inverse_transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        let inverse = self
            .params
            .inverse_func
            .ok_or(ModelError::NoInverseFunction)?;
        validate_transform_request(self.n_features_in, data, output)?;
        self.apply(inverse, data, output)
    }

    /// Undoes [`FunctionTransformer::transform`], allocating the output matrix.
    pub fn inverse_transform(
        &self,
        data: &MatrixView<'_>,
    ) -> Result<crate::data::DenseMatrix, ModelError> {
        if self.params.inverse_func.is_none() {
            return Err(ModelError::NoInverseFunction);
        }
        inverse_transform_allocating(self.n_features_in, data, |batch, output| {
            self.inverse_transform_into(batch, output).map(|_| ())
        })
    }

    /// Applies every value in fixed row-major order, after proving the whole
    /// batch stays finite, and returns a validated view over what it wrote.
    ///
    /// A caller-supplied function is not necessarily monotone, so the extrema
    /// screen the per-column scalers use would not be sound here: a map can be
    /// perfectly finite at both ends of a column and infinite in between. Every
    /// value is therefore checked before any value is written.
    ///
    /// This is the code that proves finiteness for this transformer, so it is
    /// also the only place in it that may reach
    /// [`MatrixView::from_validated_parts`].
    fn apply<'output>(
        &self,
        map: ElementwiseFn,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        for (row_index, row) in data.iter_rows().enumerate() {
            for (column, value) in row.iter().enumerate() {
                if !map(*value).is_finite() {
                    return Err(ModelError::NonFiniteTransform {
                        row: row_index,
                        column,
                    });
                }
            }
        }
        for (row, output_row) in data
            .iter_rows()
            .zip(output.chunks_exact_mut(self.n_features_in))
        {
            for (value, slot) in row.iter().zip(output_row) {
                *slot = map(*value);
            }
        }
        Ok(MatrixView::from_validated_parts(
            output,
            data.rows(),
            self.n_features_in,
        ))
    }
}

impl Estimator for FunctionTransformer {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for FunctionTransformer {
    /// A function pointer cannot be persisted meaningfully, and nothing is
    /// fitted that a per-sample weight could move.
    const CAPABILITIES: Capabilities = Capabilities::NONE;
}

impl HasParams for FunctionTransformer {
    type Params = FunctionTransformerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for FunctionTransformer {
    fn n_features_out(&self) -> usize {
        self.n_features_in
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_transform_request(self.n_features_in, data, output)?;
        self.apply(self.params.func, data, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;

    fn matrix(values: &[f32], rows: usize, columns: usize) -> DenseMatrix {
        DenseMatrix::new(values.to_vec(), rows, columns).unwrap()
    }

    fn double(value: f32) -> f32 {
        value * 2.0
    }

    fn halve(value: f32) -> f32 {
        value / 2.0
    }

    fn reciprocal(value: f32) -> f32 {
        1.0 / value
    }

    fn fitted(params: FunctionTransformerParams) -> FunctionTransformer {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        FunctionTransformer::fit(&data.as_view(), params).unwrap()
    }

    #[test]
    fn the_supplied_function_is_applied_to_every_value() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let transformer = fitted(FunctionTransformerParams::default().with_func(double));
        assert_eq!(
            transformer.transform(&data.as_view()).unwrap().as_slice(),
            &[2.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    fn the_default_is_the_identity_in_both_directions() {
        let data = matrix(&[1.0, -2.5, 0.0, 4.0], 2, 2);
        let transformer = fitted(FunctionTransformerParams::default());
        assert_eq!(
            transformer.transform(&data.as_view()).unwrap().as_slice(),
            data.as_slice()
        );
        assert!(transformer.get_params().inverse_func().is_some());
    }

    #[test]
    fn supplying_a_forward_map_clears_an_inverse_belonging_to_another_map() {
        // An inverse belongs to the function it inverts. Carrying one across a
        // change of `func` would pair a map with the inverse of a different one,
        // which is worse than having no inverse at all.
        let params = FunctionTransformerParams::default()
            .with_inverse_func(halve)
            .with_func(double);
        assert!(params.inverse_func().is_none());

        let params = params.with_inverse_func(halve);
        assert!(params.inverse_func().is_some());
    }

    #[test]
    fn a_non_finite_result_is_reported_at_its_first_row_major_cell() {
        // The reciprocal of zero is where a caller-supplied map most easily
        // leaves the finite range, and the input is finite so nothing earlier
        // in the pipeline would have caught it.
        let data = matrix(&[1.0, 2.0, 0.0, 4.0], 2, 2);
        let transformer = fitted(FunctionTransformerParams::default().with_func(reciprocal));
        let mut output = [91.0; 4];
        assert_eq!(
            transformer
                .transform_into(&data.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 1, column: 0 }
        );
        assert_eq!(output, [91.0; 4], "nothing is written when anything fails");
    }

    #[test]
    fn a_non_monotone_map_is_still_checked_at_every_value() {
        // A map can be finite at both ends of a column and infinite in between,
        // which is exactly why the per-column extrema screen the scalers use
        // would be unsound here.
        fn spike(value: f32) -> f32 {
            if value == 2.0 { f32::INFINITY } else { value }
        }
        let data = matrix(&[1.0, 2.0, 3.0], 3, 1);
        let transformer = fitted(FunctionTransformerParams::default().with_func(spike));
        let narrow = matrix(&[1.0, 2.0, 3.0], 3, 1);
        let transformer =
            FunctionTransformer::fit(&narrow.as_view(), *transformer.get_params()).unwrap();
        let mut output = [7.0; 3];
        assert_eq!(
            transformer
                .transform_into(&data.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NonFiniteTransform { row: 1, column: 0 }
        );
        assert_eq!(output, [7.0; 3]);
    }

    #[test]
    fn refitting_the_same_batch_is_deterministic() {
        // Determinism is asserted on *behaviour*, not on comparing two fitted
        // values. This type has no `PartialEq` precisely because comparing
        // function pointers compares addresses, and a test that asserted two
        // separately built transformers were equal would be testing the linker
        // rather than the transformer.
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let params = FunctionTransformerParams::default().with_func(double);
        let first = fitted(params).transform(&data.as_view()).unwrap();
        let second = fitted(params).transform(&data.as_view()).unwrap();
        assert_eq!(first.as_slice(), second.as_slice());
        assert_eq!(fitted(params).n_features_in(), 2);
    }

    #[test]
    fn the_inverse_applies_exactly_the_supplied_function() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let transformer = fitted(
            FunctionTransformerParams::default()
                .with_func(double)
                .with_inverse_func(halve),
        );
        let transformed = transformer.transform(&data.as_view()).unwrap();
        let recovered = transformer
            .inverse_transform(&transformed.as_view())
            .unwrap();
        assert_eq!(recovered.as_slice(), data.as_slice());
    }

    #[test]
    fn a_missing_inverse_is_refused_rather_than_treated_as_the_identity() {
        // Silently returning the transformed values would look exactly like a
        // successful recovery of the originals, which is the worst available
        // failure mode.
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let transformer = fitted(FunctionTransformerParams::default().with_func(double));
        assert_eq!(
            transformer.inverse_transform(&data.as_view()).unwrap_err(),
            ModelError::NoInverseFunction
        );
        let mut output = [91.0; 4];
        assert_eq!(
            transformer
                .inverse_transform_into(&data.as_view(), &mut output)
                .unwrap_err(),
            ModelError::NoInverseFunction
        );
        assert_eq!(output, [91.0; 4]);
    }

    #[test]
    fn a_wrong_inverse_is_applied_as_given_rather_than_checked() {
        // FerricML does not verify that the pair actually inverts: doing so
        // would mean picking a tolerance and probe points for the caller. The
        // guarantee is that exactly the supplied function is applied.
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let transformer = fitted(
            FunctionTransformerParams::default()
                .with_func(double)
                .with_inverse_func(double),
        );
        assert_eq!(
            transformer
                .inverse_transform(&data.as_view())
                .unwrap()
                .as_slice(),
            &[2.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    fn validates_width_and_workspace_before_writing() {
        let data = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let transformer = fitted(FunctionTransformerParams::default().with_func(double));

        let mut short = [91.0; 3];
        assert_eq!(
            transformer
                .transform_into(&data.as_view(), &mut short)
                .unwrap_err(),
            ModelError::OutputLength {
                expected: 4,
                actual: 3
            }
        );
        assert_eq!(short, [91.0; 3]);

        let narrow = matrix(&[1.0, 2.0, 3.0], 1, 3);
        let mut narrow_output = [91.0; 3];
        assert_eq!(
            transformer
                .transform_into(&narrow.as_view(), &mut narrow_output)
                .unwrap_err(),
            ModelError::FeatureDimension {
                expected: 2,
                actual: 3
            }
        );
        assert_eq!(narrow_output, [91.0; 3]);
    }
}
