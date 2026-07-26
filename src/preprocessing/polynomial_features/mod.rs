//! Deterministic dense polynomial and interaction expansion.

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError, Transformer};
use crate::artifact::{ArtifactError, POLYNOMIAL_FEATURES_ARTIFACT_KIND, StageArtifact};
use crate::data::MatrixView;

use super::expansion::{
    MAX_EXPANDED_FEATURES, binomial, column_magnitude_bounds, expand_allocating,
    expand_preflighted, validate_expansion_request,
};
use super::scaling::{
    BASE_PAYLOAD_VERSION, ScalerHeader, ScalerParameters, decode_flag, decode_scaler_artifact,
    encode_scaler_artifact,
};

/// Parameters for [`PolynomialFeatures`].
///
/// FerricML claims the degree and the two shape toggles. It deliberately does
/// **not** claim a `(min_degree, max_degree)` pair — a lower cut-off changes
/// which blocks appear but not their order or their contents, so it is an
/// additive payload version whenever a caller needs one, rather than a decision
/// this type has to carry from the start. Nor does it claim a memory-order
/// knob: FerricML's dense matrices are row-major by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolynomialFeaturesParams {
    degree: u32,
    interaction_only: bool,
    include_bias: bool,
}

impl Default for PolynomialFeaturesParams {
    fn default() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
        }
    }
}

impl PolynomialFeaturesParams {
    /// Sets the highest total degree of the generated terms.
    ///
    /// Degree zero generates the bias column alone. The parameter is validated
    /// when a transformer is fitted, not here, so a value is never rejected
    /// before the caller has finished describing it.
    #[must_use]
    pub const fn with_degree(mut self, degree: u32) -> Self {
        self.degree = degree;
        self
    }

    /// Restricts the expansion to terms whose factors are distinct features.
    ///
    /// With this set, `x0 x1` is generated and `x0^2` is not, so the width
    /// collapses from `C(n + d, d)` to a sum of `C(n, k)` — for eight features
    /// at degree four, from 495 columns to 163.
    #[must_use]
    pub const fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Enables or disables the leading constant column.
    #[must_use]
    pub const fn with_include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Returns the highest total degree of the generated terms.
    #[must_use]
    pub const fn degree(&self) -> u32 {
        self.degree
    }

    /// Returns whether terms are restricted to distinct features.
    #[must_use]
    pub const fn interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Returns whether a leading constant column is generated.
    #[must_use]
    pub const fn include_bias(&self) -> bool {
        self.include_bias
    }

    /// Rejects a configuration that would produce no columns at all.
    ///
    /// The one reachable case, and it is reachable only in combination: degree
    /// zero generates nothing but the bias column, so disabling the bias as
    /// well asks for the empty expansion. Each parameter alone is fine, which
    /// is why this is validated here rather than on either setter.
    const fn validate(&self) -> Result<(), ModelError> {
        if self.degree == 0 && !self.include_bias {
            return Err(ModelError::EmptyFeatureExpansion);
        }
        Ok(())
    }

    /// Width of the expansion over `n_features`, or `None` if it is too wide.
    ///
    /// Evaluated in checked arithmetic against the ceiling **before** anything
    /// is reserved, which is the whole point: the expanded width is
    /// `C(n + d, d)`, and that grows fast enough that an unremarkable-looking
    /// request is an impossible allocation.
    fn expanded_width(&self, n_features: usize) -> Option<usize> {
        let mut total = usize::from(self.include_bias);
        for degree in 1..=self.degree as usize {
            let block = if self.interaction_only {
                binomial(n_features, degree)?
            } else if n_features == 0 {
                0
            } else {
                binomial(n_features + degree - 1, degree)?
            };
            total = total.checked_add(block)?;
            if total > MAX_EXPANDED_FEATURES {
                return None;
            }
        }
        Some(total)
    }
}

/// Fitted polynomial and interaction expansion of a dense batch.
///
/// Each output column is one monomial in the input features — a product of
/// `degree` or fewer of them — evaluated per row. The transformer is fitted
/// only in the sense that it observes a width: there are no statistics to
/// learn, so the same parameters over the same width are the same model.
///
/// # Column order is frozen contract
///
/// The bias column comes first when present. Then the terms appear in blocks of
/// ascending total degree, and within a block in the lexicographic order of
/// their non-decreasing feature-index tuples. For three features at degree
/// three that is
///
/// ```text
/// 1, x0, x1, x2,
/// x0^2, x0 x1, x0 x2, x1^2, x1 x2, x2^2,
/// x0^3, x0^2 x1, x0^2 x2, x0 x1^2, x0 x1 x2, x0 x2^2, x1^3, x1^2 x2, x1 x2^2, x2^3
/// ```
///
/// and with `interaction_only` the same order over strictly increasing tuples.
/// A caller that persists a fitted downstream model against this expansion is
/// relying on the order, so it is pinned by test rather than left to follow
/// from how the terms happen to be generated.
///
/// # Width is public contract, and is checked before anything is allocated
///
/// The output width is `C(n + d, d)`, or `sum over k in 0..=d of C(n, k)` with
/// `interaction_only`, less one where the bias is disabled. It is evaluated in
/// checked arithmetic at fit time, so a request that cannot be built fails
/// with [`ModelError::FeatureExpansionOverflow`] naming both numbers rather
/// than by attempting the allocation.
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::preprocessing::{PolynomialFeatures, PolynomialFeaturesParams};
///
/// let data = DenseMatrix::new(vec![2.0, 3.0], 1, 2)?;
/// let expansion =
///     PolynomialFeatures::fit(&data.as_view(), PolynomialFeaturesParams::default())?;
///
/// // 1, x0, x1, x0^2, x0 x1, x1^2
/// assert_eq!(expansion.n_features_out(), 6);
/// assert_eq!(
///     expansion.transform(&data.as_view())?.as_slice(),
///     &[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialFeatures {
    n_features_in: usize,
    params: PolynomialFeaturesParams,
    n_features_out: usize,
    /// Feature indices of every term, concatenated in column order.
    term_features: Vec<u32>,
    /// Where each term starts in `term_features`; one longer than the width.
    term_offsets: Vec<u32>,
}

impl PolynomialFeatures {
    /// Fits the expansion to a batch's width.
    ///
    /// Parameters and the resulting width are validated before the term table
    /// is reserved, so a request that cannot be built allocates nothing.
    pub fn fit(
        data: &MatrixView<'_>,
        params: PolynomialFeaturesParams,
    ) -> Result<Self, ModelError> {
        Self::describe(data.columns(), params)
    }

    /// Builds the fitted expansion for a width, without a batch.
    ///
    /// Shared by [`Self::fit`] and the decoder, so a decoded model is generated
    /// by exactly the code that generated the encoded one and the two cannot
    /// come to disagree about column order.
    fn describe(
        n_features_in: usize,
        params: PolynomialFeaturesParams,
    ) -> Result<Self, ModelError> {
        params.validate()?;
        let n_features_out =
            params
                .expanded_width(n_features_in)
                .ok_or(ModelError::FeatureExpansionOverflow {
                    n_features: n_features_in,
                    degree: params.degree,
                })?;

        let mut term_features = Vec::new();
        let mut term_offsets = Vec::with_capacity(n_features_out + 1);
        term_offsets.push(0);
        if params.include_bias {
            term_offsets.push(0);
        }
        for degree in 1..=params.degree as usize {
            push_terms(
                n_features_in,
                degree,
                params.interaction_only,
                &mut term_features,
                &mut term_offsets,
            );
        }
        debug_assert_eq!(
            term_offsets.len() - 1,
            n_features_out,
            "the generated term count left the width formula"
        );

        Ok(Self {
            n_features_in,
            params,
            n_features_out,
            term_features,
            term_offsets,
        })
    }

    /// Returns the fitted input width.
    #[must_use]
    pub const fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Returns the expanded output width.
    #[must_use]
    pub const fn n_features_out(&self) -> usize {
        self.n_features_out
    }

    /// Returns the exact expansion parameters.
    #[must_use]
    pub const fn get_params(&self) -> &PolynomialFeaturesParams {
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
        expand_allocating(self.n_features_out, data, |batch, output| {
            self.transform_into(batch, output).map(|_| ())
        })
    }

    /// Evaluates one output column of one row.
    ///
    /// Accumulated in `f64` and narrowed once, per rule 1 of the accumulation
    /// policy, and over the term's factors in the order they were generated,
    /// per rule 2. The empty product is the bias column's `1.0`.
    fn monomial(&self, row: &[f32], column: usize) -> f32 {
        let start = self.term_offsets[column] as usize;
        let end = self.term_offsets[column + 1] as usize;
        let mut product = 1.0_f64;
        for &feature in &self.term_features[start..end] {
            product *= f64::from(row[feature as usize]);
        }
        product as f32
    }

    /// Whether every value this batch produces is provably finite.
    ///
    /// A monomial is a product of its factors, so its magnitude is bounded by
    /// the product of those factors' largest magnitudes in the batch. Proving
    /// that bound finite for every column proves the whole batch finite, which
    /// is what lets the values be written in a single pass. Returning `false`
    /// only costs the validation pass; it never returns a wrong answer, because
    /// the bound is an upper bound rather than an estimate.
    fn batch_is_proven_finite(&self, data: &MatrixView<'_>) -> bool {
        let Some(bounds) = column_magnitude_bounds(data) else {
            return false;
        };
        (0..self.n_features_out).all(|column| {
            let start = self.term_offsets[column] as usize;
            let end = self.term_offsets[column + 1] as usize;
            let mut product = 1.0_f64;
            for &feature in &self.term_features[start..end] {
                product *= bounds[feature as usize];
            }
            (product as f32).is_finite()
        })
    }
}

/// Appends every term of one total degree, in the frozen column order.
///
/// The tuples are generated in place and advanced lexicographically: find the
/// rightmost position still below its ceiling, raise it, and refill everything
/// to its right from it. The two modes differ only in that ceiling and that
/// refill — non-decreasing tuples repeat the raised value, strictly increasing
/// ones step up from it — which is why one generator serves both rather than
/// two that could order their blocks differently.
fn push_terms(
    n_features: usize,
    degree: usize,
    interaction_only: bool,
    term_features: &mut Vec<u32>,
    term_offsets: &mut Vec<u32>,
) {
    if n_features == 0 || (interaction_only && degree > n_features) {
        return;
    }
    let mut tuple: Vec<usize> = if interaction_only {
        (0..degree).collect()
    } else {
        vec![0; degree]
    };
    loop {
        for &feature in &tuple {
            term_features.push(feature as u32);
        }
        term_offsets.push(term_features.len() as u32);

        let mut position = degree;
        let advanced = loop {
            if position == 0 {
                break false;
            }
            position -= 1;
            let ceiling = if interaction_only {
                n_features - degree + position
            } else {
                n_features - 1
            };
            if tuple[position] < ceiling {
                tuple[position] += 1;
                for later in position + 1..degree {
                    tuple[later] = if interaction_only {
                        tuple[later - 1] + 1
                    } else {
                        tuple[position]
                    };
                }
                break true;
            }
        };
        if !advanced {
            return;
        }
    }
}

impl StageArtifact for PolynomialFeatures {
    const ARTIFACT_KIND: u16 = POLYNOMIAL_FEATURES_ARTIFACT_KIND;

    /// Encodes the fitted width and parameters.
    ///
    /// The term table is **not** stored. It is a pure function of the width and
    /// the parameters, so storing it would be storing a derivation — and a
    /// derivation on disk is a second definition of the column order that could
    /// disagree with the one in this file. Regenerating it on decode is also
    /// what makes the encoding canonical: there is exactly one byte string for
    /// one fitted expansion, whatever its width.
    fn to_artifact(
        &self,
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Vec<u8>, ArtifactError> {
        encode_scaler_artifact(
            Self::ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            self.n_features_in,
            ScalerParameters {
                version: BASE_PAYLOAD_VERSION,
                flags: &[
                    u32::from(self.params.interaction_only),
                    u32::from(self.params.include_bias),
                    self.params.degree,
                ],
                reals: &[],
            },
            0,
            |_, _| {},
        )
    }

    /// Decodes the fitted width and parameters, and regenerates the terms.
    fn from_artifact(
        bytes: &[u8],
        input_schema: [u8; 32],
        transformed_schema: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let ScalerHeader {
            n_features_in,
            flags,
            state,
            ..
        } = decode_scaler_artifact(
            bytes,
            Self::ARTIFACT_KIND,
            input_schema,
            transformed_schema,
            BASE_PAYLOAD_VERSION,
            3,
            0,
        )?;
        if !state.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        let params = PolynomialFeaturesParams {
            interaction_only: decode_flag(flags[0])?,
            include_bias: decode_flag(flags[1])?,
            degree: flags[2],
        };
        // A configuration fitting would have refused describes a model that
        // could not have been produced, so it is rejected on the way back in
        // rather than trusted because it is already encoded. The width is
        // recomputed here for the same reason it is not stored.
        Self::describe(n_features_in, params).map_err(|_| ArtifactError::InvalidPayload)
    }
}

impl Estimator for PolynomialFeatures {
    fn n_features_in(&self) -> usize {
        self.n_features_in
    }
}

impl HasCapabilities for PolynomialFeatures {
    /// The fitted width and parameters persist; there is nothing to weight.
    ///
    /// An expansion learns no statistic from the rows it is fitted on — it
    /// reads their width and nothing else — so a per-sample weight has nothing
    /// to move and there is no weighted entry point to declare.
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_artifact(true);
}

impl HasParams for PolynomialFeatures {
    type Params = PolynomialFeaturesParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

impl Transformer for PolynomialFeatures {
    fn n_features_out(&self) -> usize {
        self.n_features_out
    }

    fn transform_into<'output>(
        &self,
        data: &MatrixView<'_>,
        output: &'output mut [f32],
    ) -> Result<MatrixView<'output>, ModelError> {
        validate_expansion_request(self.n_features_in, self.n_features_out, data, output)?;
        expand_preflighted(
            data,
            output,
            self.n_features_out,
            self.batch_is_proven_finite(data),
            |row, column| self.monomial(row, column),
        )
    }
}

#[cfg(test)]
mod tests;
