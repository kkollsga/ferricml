//! Pairwise linear ranking over one validated item matrix.

use std::error::Error;
use std::fmt;

use crate::api::{Capabilities, Estimator, HasCapabilities, HasParams, ModelError};
use crate::artifact::{
    ArtifactError, ArtifactPayloadWriter, PAIRWISE_LINEAR_RANKER_ARTIFACT_KIND, SchemaRole,
    decode_component, decode_v2_envelope, encode_component, encode_v2_envelope,
};
use crate::data::{BinaryTargets, DenseMatrix, MatrixView, SampleWeights};
use crate::linear_model::{LogisticRegression, LogisticRegressionParams};

const PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const MODEL_COMPONENT_KIND: u16 = 2;
const COMPONENT_VERSION: u16 = 1;
const OBJECTIVE_VERSION: u32 = 1;
const NORMALIZATION_VERSION: u32 = 1;
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;

/// A typed observed or predicted pair outcome.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PairOutcome {
    /// The left item is preferred to the right item.
    LeftPreferred,
    /// The right item is preferred to the left item.
    RightPreferred,
    /// Neither item is preferred.
    Tie,
}

impl PairOutcome {
    const fn reversed(self) -> Self {
        match self {
            Self::LeftPreferred => Self::RightPreferred,
            Self::RightPreferred => Self::LeftPreferred,
            Self::Tie => Self::Tie,
        }
    }
}

/// Errors encountered while constructing pair data.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PairDataError {
    /// A pair references the same item on both sides.
    SameItem {
        /// The item index given on both sides of the pair.
        index: usize,
    },
    /// A pair weight is NaN or infinite.
    NonFiniteWeight,
    /// A pair weight is negative.
    NegativeWeight,
}

impl fmt::Display for PairDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SameItem { index } => write!(f, "pair references item {index} twice"),
            Self::NonFiniteWeight => f.write_str("pair weight must be finite"),
            Self::NegativeWeight => f.write_str("pair weight must be non-negative"),
        }
    }
}

impl Error for PairDataError {}

/// Two distinct item indices.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PairIndex {
    left: usize,
    right: usize,
}

impl PairIndex {
    /// Creates a pair of distinct indices.
    pub fn new(left: usize, right: usize) -> Result<Self, PairDataError> {
        if left == right {
            return Err(PairDataError::SameItem { index: left });
        }
        Ok(Self { left, right })
    }

    /// Returns the left item index.
    pub const fn left(&self) -> usize {
        self.left
    }

    /// Returns the right item index.
    pub const fn right(&self) -> usize {
        self.right
    }
}

/// One weighted pairwise training observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairwiseObservation {
    pair: PairIndex,
    outcome: PairOutcome,
    weight: f32,
}

impl PairwiseObservation {
    /// Creates an observation with a finite, non-negative pair weight.
    pub fn new(pair: PairIndex, outcome: PairOutcome, weight: f32) -> Result<Self, PairDataError> {
        if !weight.is_finite() {
            return Err(PairDataError::NonFiniteWeight);
        }
        if weight < 0.0 {
            return Err(PairDataError::NegativeWeight);
        }
        Ok(Self {
            pair,
            outcome,
            weight,
        })
    }

    /// Returns the two item indices.
    pub const fn pair(&self) -> PairIndex {
        self.pair
    }

    /// Returns the observed preference outcome.
    pub const fn outcome(&self) -> PairOutcome {
        self.outcome
    }

    /// Returns the pair's objective weight.
    pub const fn weight(&self) -> f32 {
        self.weight
    }

    fn canonical(self) -> Self {
        if self.pair.left < self.pair.right {
            self
        } else {
            Self {
                pair: PairIndex {
                    left: self.pair.right,
                    right: self.pair.left,
                },
                outcome: self.outcome.reversed(),
                weight: self.weight,
            }
        }
    }
}

/// Errors produced while fitting or using a pairwise ranker.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PairwiseError {
    /// No pair observations were supplied.
    EmptyPairs,
    /// Every pair weight is zero.
    ZeroTotalPairWeight,
    /// A pair references an item outside the fitted matrix.
    PairIndexOutOfBounds {
        /// Zero-based position of the offending pair observation.
        pair: usize,
        /// The item index it referenced.
        item: usize,
        /// Number of items the fitted matrix holds.
        items: usize,
    },
    /// Expanding pair observations would overflow a matrix dimension.
    PairMatrixOverflow,
    /// Dividing a positive pair weight by its normalization underflowed.
    PairWeightUnderflow {
        /// Zero-based position of the offending pair observation.
        pair: usize,
    },
    /// A finite item difference cannot be represented as `f32`.
    NonFinitePairDifference {
        /// Zero-based position of the offending pair observation.
        pair: usize,
        /// Zero-based feature column whose difference is unrepresentable.
        column: usize,
    },
    /// The tie threshold is not finite and non-negative.
    InvalidTieThreshold,
    /// A lower-level fitted-model contract failed.
    Model(ModelError),
}

impl fmt::Display for PairwiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPairs => f.write_str("pairwise fitting requires at least one pair"),
            Self::ZeroTotalPairWeight => {
                f.write_str("pairwise fitting requires a positive total pair weight")
            }
            Self::PairIndexOutOfBounds { pair, item, items } => write!(
                f,
                "pair {pair} references item {item}, but the item matrix has {items} rows"
            ),
            Self::PairMatrixOverflow => f.write_str("expanded pairwise matrix dimensions overflow"),
            Self::PairWeightUnderflow { pair } => {
                write!(f, "normalizing positive weight for pair {pair} underflowed")
            }
            Self::NonFinitePairDifference { pair, column } => write!(
                f,
                "feature difference for pair {pair}, column {column} is not finite"
            ),
            Self::InvalidTieThreshold => {
                f.write_str("pairwise tie threshold must be finite and non-negative")
            }
            Self::Model(error) => error.fmt(f),
        }
    }
}

impl Error for PairwiseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for PairwiseError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

/// Parameters for [`PairwiseLinearRanker`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseLinearRankerParams {
    c: f32,
    max_iter: usize,
    tol: f32,
    tie_threshold: f32,
}

impl Default for PairwiseLinearRankerParams {
    fn default() -> Self {
        Self {
            c: 1.0,
            max_iter: 100,
            tol: 1.0e-4,
            tie_threshold: 0.0,
        }
    }
}

impl PairwiseLinearRankerParams {
    /// Sets inverse L2 regularization strength.
    #[must_use]
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Sets the maximum optimization iteration count.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the maximum absolute coefficient update used for convergence.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the inclusive absolute-margin threshold mapped to a tie.
    #[must_use]
    pub fn with_tie_threshold(mut self, tie_threshold: f32) -> Self {
        self.tie_threshold = tie_threshold;
        self
    }

    /// Returns inverse L2 regularization strength.
    pub const fn c(&self) -> f32 {
        self.c
    }

    /// Returns the maximum optimization iteration count.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    pub const fn tol(&self) -> f32 {
        self.tol
    }

    /// Returns the inclusive absolute-margin tie threshold.
    pub const fn tie_threshold(&self) -> f32 {
        self.tie_threshold
    }
}

/// Linear item scorer fitted from mirrored weighted pair observations.
///
/// ```
/// use ferricml::data::DenseMatrix;
/// use ferricml::ranking::{
///     PairIndex, PairOutcome, PairwiseLinearRanker, PairwiseLinearRankerParams,
///     PairwiseObservation,
/// };
///
/// // Four items, one feature. Higher is better.
/// let items = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1)?;
///
/// let observations = vec![
///     PairwiseObservation::new(PairIndex::new(3, 0)?, PairOutcome::LeftPreferred, 1.0)?,
///     PairwiseObservation::new(PairIndex::new(2, 1)?, PairOutcome::LeftPreferred, 1.0)?,
///     PairwiseObservation::new(PairIndex::new(3, 1)?, PairOutcome::LeftPreferred, 1.0)?,
/// ];
///
/// let ranker = PairwiseLinearRanker::fit(
///     &items.as_view(),
///     &observations,
///     PairwiseLinearRankerParams::default(),
/// )?;
///
/// // Item scores are raw objective values, not probabilities.
/// let scores = ranker.score_items(&items.as_view())?;
/// assert!(scores[0] < scores[3]);
///
/// // The margin is score(left) - score(right).
/// let margin = ranker.pair_margin(&items.as_view(), PairIndex::new(3, 0)?)?;
/// assert!((margin - (scores[3] - scores[0])).abs() < 1e-5);
/// assert_eq!(
///     ranker.compare(&items.as_view(), PairIndex::new(3, 0)?)?,
///     PairOutcome::LeftPreferred,
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseLinearRanker {
    params: PairwiseLinearRankerParams,
    model: LogisticRegression,
}

impl PairwiseLinearRanker {
    /// Fits a deterministic no-intercept pairwise logistic objective.
    pub fn fit(
        items: &MatrixView<'_>,
        observations: &[PairwiseObservation],
        params: PairwiseLinearRankerParams,
    ) -> Result<Self, PairwiseError> {
        validate_params(&params)?;
        let (expanded, targets, weights) = expand_observations(items, observations)?;
        let model = LogisticRegression::fit_weighted(
            &expanded.as_view(),
            &targets,
            &weights,
            LogisticRegressionParams::default()
                .with_c(params.c)
                .with_fit_intercept(false)
                .with_max_iter(params.max_iter)
                .with_tol(params.tol),
        )?;
        debug_assert_eq!(model.intercept().to_bits(), 0.0_f32.to_bits());
        Ok(Self { params, model })
    }

    /// Returns fitted item-feature coefficients.
    pub fn coefficients(&self) -> &[f32] {
        self.model.coefficients()
    }

    /// Returns the exact ranker parameters.
    pub const fn get_params(&self) -> &PairwiseLinearRankerParams {
        &self.params
    }

    /// Returns the fitted input width.
    pub fn n_features_in(&self) -> usize {
        self.model.n_features_in()
    }

    /// Returns the number of objective iterations performed.
    pub fn n_iter(&self) -> usize {
        self.model.n_iter()
    }

    /// Scores one item with the fitted raw linear objective.
    pub fn score_one(&self, item: &[f32]) -> Result<f32, PairwiseError> {
        Ok(self.model.decision_function_one(item)?)
    }

    /// Scores every item, allocating one raw score per row.
    pub fn score_items(&self, items: &MatrixView<'_>) -> Result<Vec<f32>, PairwiseError> {
        // Before the buffer, not inside the fitted model's `_into` form after
        // it, which repeats the check against the same fitted width.
        if items.columns() != self.n_features_in() {
            return Err(ModelError::FeatureDimension {
                expected: self.n_features_in(),
                actual: items.columns(),
            }
            .into());
        }
        let mut output = vec![0.0; items.rows()];
        self.score_items_into(items, &mut output)?;
        Ok(output)
    }

    /// Scores every item into caller-owned storage.
    pub fn score_items_into(
        &self,
        items: &MatrixView<'_>,
        output: &mut [f32],
    ) -> Result<(), PairwiseError> {
        Ok(self.model.decision_function_into(items, output)?)
    }

    /// Returns `score(left) - score(right)` for one checked pair.
    pub fn pair_margin(
        &self,
        items: &MatrixView<'_>,
        pair: PairIndex,
    ) -> Result<f32, PairwiseError> {
        validate_pair_index(pair, 0, items.rows())?;
        let left = items.row(pair.left).expect("validated pair index");
        let right = items.row(pair.right).expect("validated pair index");
        Ok(self.score_one(left)? - self.score_one(right)?)
    }

    /// Maps one raw pair margin to left, right, or tie.
    pub fn compare(
        &self,
        items: &MatrixView<'_>,
        pair: PairIndex,
    ) -> Result<PairOutcome, PairwiseError> {
        let margin = self.pair_margin(items, pair)?;
        if margin.abs() <= self.params.tie_threshold {
            Ok(PairOutcome::Tie)
        } else if margin > 0.0 {
            Ok(PairOutcome::LeftPreferred)
        } else {
            Ok(PairOutcome::RightPreferred)
        }
    }

    /// Writes raw margins for checked pairs without allocating.
    pub fn pair_margins_into(
        &self,
        items: &MatrixView<'_>,
        pairs: &[PairIndex],
        output: &mut [f32],
    ) -> Result<(), PairwiseError> {
        validate_pair_batch(items, pairs, output.len())?;
        for (&pair, slot) in pairs.iter().zip(output) {
            *slot = self.pair_margin(items, pair)?;
        }
        Ok(())
    }

    /// Writes three-way comparisons for checked pairs without allocating.
    pub fn compare_into(
        &self,
        items: &MatrixView<'_>,
        pairs: &[PairIndex],
        output: &mut [PairOutcome],
    ) -> Result<(), PairwiseError> {
        validate_pair_batch(items, pairs, output.len())?;
        for (&pair, slot) in pairs.iter().zip(output) {
            *slot = self.compare(items, pair)?;
        }
        Ok(())
    }

    /// Encodes the rank objective, normalization, and fitted linear model.
    pub fn to_artifact(&self, schema: [u8; 32]) -> Result<Vec<u8>, ArtifactError> {
        if self.n_features_in() > MAX_ARTIFACT_FEATURES {
            return Err(ArtifactError::InvalidPayload);
        }
        let n_features =
            u32::try_from(self.n_features_in()).map_err(|_| ArtifactError::InvalidPayload)?;
        let max_iter =
            u32::try_from(self.params.max_iter).map_err(|_| ArtifactError::InvalidPayload)?;
        let mut metadata = ArtifactPayloadWriter::with_capacity(7 * 4);
        metadata.u32(OBJECTIVE_VERSION);
        metadata.u32(NORMALIZATION_VERSION);
        metadata.u32(n_features);
        metadata.f32(self.params.c);
        metadata.u32(max_iter);
        metadata.f32(self.params.tol);
        metadata.f32(self.params.tie_threshold);
        let metadata = encode_component(
            METADATA_COMPONENT_KIND,
            COMPONENT_VERSION,
            &metadata.finish(),
        )?;
        let model = encode_component(
            MODEL_COMPONENT_KIND,
            COMPONENT_VERSION,
            &self.model.to_artifact(schema)?,
        )?;
        let mut payload = Vec::with_capacity(metadata.len() + model.len());
        payload.extend_from_slice(&metadata);
        payload.extend_from_slice(&model);
        encode_v2_envelope(
            PAIRWISE_LINEAR_RANKER_ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
            &payload,
        )
    }

    /// Decodes a ranker after checking objective, normalization, and schema.
    pub fn from_artifact(bytes: &[u8], schema: [u8; 32]) -> Result<Self, ArtifactError> {
        let mut envelope = decode_v2_envelope(
            bytes,
            PAIRWISE_LINEAR_RANKER_ARTIFACT_KIND,
            PAYLOAD_VERSION,
            &[(SchemaRole::Input, schema)],
        )?;
        let mut metadata =
            decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
        let model = decode_component(&mut envelope, MODEL_COMPONENT_KIND, COMPONENT_VERSION)?;
        if !envelope.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        let objective_version = metadata.u32()?;
        let normalization_version = metadata.u32()?;
        let n_features_in = metadata.u32()? as usize;
        let params = PairwiseLinearRankerParams {
            c: metadata.f32()?,
            max_iter: metadata.u32()? as usize,
            tol: metadata.f32()?,
            tie_threshold: metadata.f32()?,
        };
        if !metadata.is_empty()
            || objective_version != OBJECTIVE_VERSION
            || normalization_version != NORMALIZATION_VERSION
            || n_features_in == 0
            || n_features_in > MAX_ARTIFACT_FEATURES
            || validate_params(&params).is_err()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        let model = LogisticRegression::from_artifact(model.remaining(), schema)?;
        let model_params = model.get_params();
        if model.n_features_in() != n_features_in
            || model_params.fit_intercept()
            || model.intercept().to_bits() != 0.0_f32.to_bits()
            || model_params.c().to_bits() != params.c.to_bits()
            || model_params.max_iter() != params.max_iter
            || model_params.tol().to_bits() != params.tol.to_bits()
        {
            return Err(ArtifactError::InvalidPayload);
        }
        Ok(Self { params, model })
    }
}

impl Estimator for PairwiseLinearRanker {
    fn n_features_in(&self) -> usize {
        self.model.n_features_in()
    }
}

/// A ranker persists, and declares nothing else.
///
/// `sample_weights` is deliberately absent even though fitting is weighted:
/// the weight belongs to a *pair observation*, not to a row of the item
/// matrix, and there is no `SampleWeights` entry point for a caller to reach.
/// Declaring it would answer a question about per-sample weighting that this
/// estimator does not have. `multiclass` has no meaning without a class set.
///
/// `decision_function` is absent for the reason that keeps the field honest:
/// it records that a *classifier* exposes a raw score whose squashing is its
/// probability. A ranker has no probability to squash to, and ranking is
/// documented as distinct from classification — raw scores and pair margins
/// are not probabilities. Declaring it would make one field mean two things.
impl HasCapabilities for PairwiseLinearRanker {
    const CAPABILITIES: Capabilities = Capabilities::NONE.with_artifact(true);
}

impl HasParams for PairwiseLinearRanker {
    type Params = PairwiseLinearRankerParams;

    fn get_params(&self) -> &Self::Params {
        &self.params
    }
}

fn validate_params(params: &PairwiseLinearRankerParams) -> Result<(), PairwiseError> {
    if !params.tie_threshold.is_finite() || params.tie_threshold < 0.0 {
        return Err(PairwiseError::InvalidTieThreshold);
    }
    if !params.c.is_finite() || params.c <= 0.0 {
        return Err(ModelError::InvalidRegularization.into());
    }
    if params.max_iter == 0 {
        return Err(ModelError::InvalidIterationCount.into());
    }
    if !params.tol.is_finite() || params.tol <= 0.0 {
        return Err(ModelError::InvalidTolerance.into());
    }
    Ok(())
}

fn validate_pair_index(
    pair: PairIndex,
    pair_number: usize,
    items: usize,
) -> Result<(), PairwiseError> {
    for item in [pair.left, pair.right] {
        if item >= items {
            return Err(PairwiseError::PairIndexOutOfBounds {
                pair: pair_number,
                item,
                items,
            });
        }
    }
    Ok(())
}

fn validate_pair_batch(
    items: &MatrixView<'_>,
    pairs: &[PairIndex],
    output_len: usize,
) -> Result<(), PairwiseError> {
    if output_len != pairs.len() {
        return Err(ModelError::OutputLength {
            expected: pairs.len(),
            actual: output_len,
        }
        .into());
    }
    for (index, &pair) in pairs.iter().enumerate() {
        validate_pair_index(pair, index, items.rows())?;
    }
    Ok(())
}

fn expand_observations(
    items: &MatrixView<'_>,
    observations: &[PairwiseObservation],
) -> Result<(DenseMatrix, BinaryTargets, SampleWeights), PairwiseError> {
    if observations.is_empty() {
        return Err(PairwiseError::EmptyPairs);
    }
    // Every check below is one pass over the caller's own slice and needs
    // nothing the canonicalized copy provides, so it runs first: a batch that
    // will be refused costs neither the copy nor the sort that follows it.
    let mut row_count = 0_usize;
    let mut total_weight = 0.0_f64;
    for (index, observation) in observations.iter().enumerate() {
        validate_pair_index(observation.pair, index, items.rows())?;
        row_count = row_count
            .checked_add(if observation.outcome == PairOutcome::Tie {
                4
            } else {
                2
            })
            .ok_or(PairwiseError::PairMatrixOverflow)?;
        total_weight += f64::from(observation.weight);
    }
    if total_weight <= 0.0 {
        return Err(PairwiseError::ZeroTotalPairWeight);
    }
    let mut canonical = observations
        .iter()
        .copied()
        .enumerate()
        .map(|(index, observation)| (index, observation.canonical()))
        .collect::<Vec<_>>();
    canonical.sort_by_key(|(_, observation)| {
        (
            observation.pair,
            observation.outcome,
            observation.weight.to_bits(),
        )
    });
    let value_count = row_count
        .checked_mul(items.columns())
        .ok_or(PairwiseError::PairMatrixOverflow)?;
    let mut values = Vec::with_capacity(value_count);
    let mut targets = Vec::with_capacity(row_count);
    let mut weights = Vec::with_capacity(row_count);
    for &(source_index, observation) in &canonical {
        let divisor = if observation.outcome == PairOutcome::Tie {
            4.0
        } else {
            2.0
        };
        let normalized_weight = observation.weight / divisor;
        if observation.weight > 0.0 && normalized_weight == 0.0 {
            return Err(PairwiseError::PairWeightUnderflow { pair: source_index });
        }
        match observation.outcome {
            PairOutcome::LeftPreferred => {
                push_difference(
                    items,
                    observation.pair,
                    false,
                    1,
                    normalized_weight,
                    source_index,
                    &mut values,
                    &mut targets,
                    &mut weights,
                )?;
                push_difference(
                    items,
                    observation.pair,
                    true,
                    0,
                    normalized_weight,
                    source_index,
                    &mut values,
                    &mut targets,
                    &mut weights,
                )?;
            }
            PairOutcome::RightPreferred => {
                push_difference(
                    items,
                    observation.pair,
                    false,
                    0,
                    normalized_weight,
                    source_index,
                    &mut values,
                    &mut targets,
                    &mut weights,
                )?;
                push_difference(
                    items,
                    observation.pair,
                    true,
                    1,
                    normalized_weight,
                    source_index,
                    &mut values,
                    &mut targets,
                    &mut weights,
                )?;
            }
            PairOutcome::Tie => {
                for (reverse, target) in [(false, 1), (false, 0), (true, 1), (true, 0)] {
                    push_difference(
                        items,
                        observation.pair,
                        reverse,
                        target,
                        normalized_weight,
                        source_index,
                        &mut values,
                        &mut targets,
                        &mut weights,
                    )?;
                }
            }
        }
    }
    let matrix = DenseMatrix::new(values, row_count, items.columns())
        .expect("pair expansion validates shape and finiteness");
    let targets = BinaryTargets::new(targets).expect("pair expansion emits binary targets");
    let weights = SampleWeights::new(weights).expect("pair expansion validates positive weight");
    Ok((matrix, targets, weights))
}

#[allow(clippy::too_many_arguments)]
fn push_difference(
    items: &MatrixView<'_>,
    pair: PairIndex,
    reverse: bool,
    target: u8,
    weight: f32,
    pair_number: usize,
    values: &mut Vec<f32>,
    targets: &mut Vec<u8>,
    weights: &mut Vec<f32>,
) -> Result<(), PairwiseError> {
    let left = items.row(pair.left).expect("validated pair index");
    let right = items.row(pair.right).expect("validated pair index");
    for (column, (&left, &right)) in left.iter().zip(right).enumerate() {
        let difference = if reverse {
            f64::from(right) - f64::from(left)
        } else {
            f64::from(left) - f64::from(right)
        } as f32;
        if !difference.is_finite() {
            return Err(PairwiseError::NonFinitePairDifference {
                pair: pair_number,
                column,
            });
        }
        values.push(difference);
    }
    targets.push(target);
    weights.push(weight);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn items() -> DenseMatrix {
        DenseMatrix::new(vec![0.0, 0.0, 1.0, 0.25, 2.0, 1.0, 3.0, 2.0], 4, 2).unwrap()
    }

    fn observation(
        left: usize,
        right: usize,
        outcome: PairOutcome,
        weight: f32,
    ) -> PairwiseObservation {
        PairwiseObservation::new(PairIndex::new(left, right).unwrap(), outcome, weight).unwrap()
    }

    fn training_pairs() -> Vec<PairwiseObservation> {
        vec![
            observation(3, 2, PairOutcome::LeftPreferred, 2.0),
            observation(2, 1, PairOutcome::LeftPreferred, 1.0),
            observation(1, 0, PairOutcome::LeftPreferred, 1.0),
            observation(1, 2, PairOutcome::Tie, 0.5),
        ]
    }

    #[test]
    fn validates_pair_construction_indices_weights_and_params() {
        assert_eq!(
            PairIndex::new(2, 2),
            Err(PairDataError::SameItem { index: 2 })
        );
        let pair = PairIndex::new(0, 1).unwrap();
        assert_eq!(
            PairwiseObservation::new(pair, PairOutcome::Tie, f32::NAN),
            Err(PairDataError::NonFiniteWeight)
        );
        assert_eq!(
            PairwiseObservation::new(pair, PairOutcome::Tie, -1.0),
            Err(PairDataError::NegativeWeight)
        );
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items().as_view(),
                &[],
                PairwiseLinearRankerParams::default()
            ),
            Err(PairwiseError::EmptyPairs)
        );
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items().as_view(),
                &[observation(0, 7, PairOutcome::LeftPreferred, 1.0)],
                PairwiseLinearRankerParams::default()
            ),
            Err(PairwiseError::PairIndexOutOfBounds {
                pair: 0,
                item: 7,
                items: 4
            })
        );
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items().as_view(),
                &[observation(0, 1, PairOutcome::Tie, 1.0)],
                PairwiseLinearRankerParams::default().with_tie_threshold(-1.0)
            ),
            Err(PairwiseError::InvalidTieThreshold)
        );
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items().as_view(),
                &[observation(0, 1, PairOutcome::Tie, 0.0)],
                PairwiseLinearRankerParams::default()
            ),
            Err(PairwiseError::ZeroTotalPairWeight)
        );
        assert_eq!(
            PairwiseLinearRanker::fit(
                &items().as_view(),
                &[observation(
                    0,
                    1,
                    PairOutcome::LeftPreferred,
                    f32::from_bits(1)
                )],
                PairwiseLinearRankerParams::default()
            ),
            Err(PairwiseError::PairWeightUnderflow { pair: 0 })
        );

        let extreme = DenseMatrix::new(vec![f32::MAX, -f32::MAX], 2, 1).unwrap();
        assert_eq!(
            PairwiseLinearRanker::fit(
                &extreme.as_view(),
                &[observation(0, 1, PairOutcome::LeftPreferred, 1.0)],
                PairwiseLinearRankerParams::default()
            ),
            Err(PairwiseError::NonFinitePairDifference { pair: 0, column: 0 })
        );
    }

    #[test]
    fn mirrored_expansion_preserves_each_pair_weight() {
        let observations = [
            observation(0, 1, PairOutcome::LeftPreferred, 2.0),
            observation(1, 2, PairOutcome::Tie, 3.0),
        ];
        let (expanded, targets, weights) =
            expand_observations(&items().as_view(), &observations).unwrap();
        assert_eq!(expanded.rows(), 6);
        assert_eq!(targets.as_slice(), &[1, 0, 1, 0, 1, 0]);
        assert_eq!(weights.total(), 5.0);
        assert_eq!(&weights.as_slice()[..2], &[1.0, 1.0]);
        assert_eq!(&weights.as_slice()[2..], &[0.75; 4]);
        for column in 0..expanded.columns() {
            assert_eq!(
                expanded.get(0, column),
                expanded.get(1, column).map(|value| -value)
            );
        }
    }

    #[test]
    fn fit_is_permutation_and_orientation_invariant() {
        let pairs = training_pairs();
        let first = PairwiseLinearRanker::fit(
            &items().as_view(),
            &pairs,
            PairwiseLinearRankerParams::default(),
        )
        .unwrap();
        let mut reversed_order = pairs.clone();
        reversed_order.reverse();
        let second = PairwiseLinearRanker::fit(
            &items().as_view(),
            &reversed_order,
            PairwiseLinearRankerParams::default(),
        )
        .unwrap();
        assert_eq!(first, second);

        let reoriented = pairs
            .iter()
            .map(|observation| {
                PairwiseObservation::new(
                    PairIndex::new(observation.pair.right, observation.pair.left).unwrap(),
                    observation.outcome.reversed(),
                    observation.weight,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let third = PairwiseLinearRanker::fit(
            &items().as_view(),
            &reoriented,
            PairwiseLinearRankerParams::default(),
        )
        .unwrap();
        assert_eq!(first, third);
    }

    #[test]
    fn raw_scores_margins_and_threshold_boundary_are_exact() {
        let base = PairwiseLinearRanker::fit(
            &items().as_view(),
            &training_pairs(),
            PairwiseLinearRankerParams::default().with_c(4.0),
        )
        .unwrap();
        let pair = PairIndex::new(3, 0).unwrap();
        let margin = base.pair_margin(&items().as_view(), pair).unwrap();
        let reverse = base
            .pair_margin(&items().as_view(), PairIndex::new(0, 3).unwrap())
            .unwrap();
        assert_eq!(margin.to_bits(), (-reverse).to_bits());
        assert!(
            margin > 1.0,
            "raw rank score must not be probability-shaped"
        );

        let thresholded = PairwiseLinearRanker::fit(
            &items().as_view(),
            &training_pairs(),
            PairwiseLinearRankerParams::default()
                .with_c(4.0)
                .with_tie_threshold(margin.abs()),
        )
        .unwrap();
        assert_eq!(
            thresholded.compare(&items().as_view(), pair),
            Ok(PairOutcome::Tie)
        );
        assert_eq!(base.coefficients().len(), 2);
        assert!((base.coefficients()[0] - 1.311_493_6).abs() < 1.0e-5);
    }

    #[test]
    fn batch_pair_validation_happens_before_writes() {
        let model = PairwiseLinearRanker::fit(
            &items().as_view(),
            &training_pairs(),
            PairwiseLinearRankerParams::default(),
        )
        .unwrap();
        let pairs = [PairIndex::new(0, 1).unwrap(), PairIndex::new(1, 9).unwrap()];
        let mut output = [77.0; 2];
        assert_eq!(
            model.pair_margins_into(&items().as_view(), &pairs, &mut output),
            Err(PairwiseError::PairIndexOutOfBounds {
                pair: 1,
                item: 9,
                items: 4
            })
        );
        assert_eq!(output, [77.0; 2]);
    }

    #[test]
    fn artifact_round_trip_checks_schema_objective_and_corruption() {
        let model = PairwiseLinearRanker::fit(
            &items().as_view(),
            &training_pairs(),
            PairwiseLinearRankerParams::default(),
        )
        .unwrap();
        let bytes = model.to_artifact([9; 32]).unwrap();
        assert_eq!(bytes, model.to_artifact([9; 32]).unwrap());
        assert_eq!(
            PairwiseLinearRanker::from_artifact(&bytes, [9; 32]).unwrap(),
            model
        );
        assert_eq!(
            PairwiseLinearRanker::from_artifact(&bytes, [8; 32]).unwrap_err(),
            ArtifactError::FeatureSchemaMismatch
        );

        let mut objective = bytes.clone();
        objective[68..72].copy_from_slice(&2_u32.to_le_bytes());
        let checksum_start = objective.len() - 32;
        let checksum = Sha256::digest(&objective[..checksum_start]);
        objective[checksum_start..].copy_from_slice(&checksum);
        assert_eq!(
            PairwiseLinearRanker::from_artifact(&objective, [9; 32]).unwrap_err(),
            ArtifactError::InvalidPayload
        );

        let mut corrupted = bytes;
        corrupted[75] ^= 1;
        assert_eq!(
            PairwiseLinearRanker::from_artifact(&corrupted, [9; 32]).unwrap_err(),
            ArtifactError::ChecksumMismatch
        );
    }
}
