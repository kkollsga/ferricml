//! Fit validation, prediction shape checks, and the shared artifact metadata
//! block that both standalone tree estimators write.
//!
//! Keeping these here rather than beside one estimator is what stops the two
//! from drifting into spelling the same parameter two ways on disk — a model
//! that has two valid encodings is a model an artifact reader cannot call
//! canonical.

use super::grower::GrowerConfig;
use super::packed::FEATURE_MASK;
use super::parameters::{MaxFeatures, Splitter, decode_max_features, decode_splitter};
use crate::api::ModelError;
use crate::artifact::{ArtifactCursor, ArtifactError, ArtifactPayloadWriter};
use crate::data::MatrixView;
use crate::numeric::derive_tree_seed;

/// Ceilings applied identically when encoding and decoding, so an artifact this
/// crate produced always decodes and a hostile one allocates nothing unbounded.
pub(super) const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
pub(super) const MAX_ARTIFACT_TOTAL_NODES: usize = 1_048_576;

// A decoded feature index must never reach the packed layout's flag bits.
const _: () = assert!(MAX_ARTIFACT_FEATURES < FEATURE_MASK as usize);

/// The generator seed one standalone tree starts from.
///
/// This is deliberately the ensemble's derivation for member `0` rather than
/// the raw `random_state`. A forest does not hand its public seed to a member;
/// it derives one per index, so a standalone tree that used the raw seed would
/// be a *different* tree from a one-tree forest at the same `random_state`, and
/// the equivalence between the two would hold only by coincidence. Deriving the
/// same way makes it hold by construction — the claim is about the grower, not
/// about two public seeds happening to coincide.
pub(super) fn tree_seed(random_state: u64) -> u64 {
    derive_tree_seed(random_state, 0)
}

/// Validates shapes and parameters before any allocation or training work.
pub(super) fn validate_fit(
    data: &MatrixView<'_>,
    target_len: usize,
    sample_weight_len: Option<usize>,
    config: &GrowerConfig,
) -> Result<(), ModelError> {
    if data.rows() == 0 || data.columns() == 0 {
        return Err(ModelError::EmptyData);
    }
    if target_len != data.rows() {
        return Err(ModelError::TargetLength {
            rows: data.rows(),
            targets: target_len,
        });
    }
    if let Some(weights) = sample_weight_len
        && data.rows() != weights
    {
        return Err(ModelError::SampleWeightLength {
            rows: data.rows(),
            weights,
        });
    }
    if config.max_depth == Some(0) {
        return Err(ModelError::InvalidMaxDepth);
    }
    if config.min_samples_split < 2 {
        return Err(ModelError::InvalidMinSamplesSplit);
    }
    if config.min_samples_leaf == 0 {
        return Err(ModelError::InvalidMinSamplesLeaf);
    }
    if data.rows() > u32::MAX as usize {
        return Err(ModelError::TooManyRows);
    }
    if data.columns() > FEATURE_MASK as usize {
        return Err(ModelError::TooManyFeatures);
    }
    if let MaxFeatures::Count(requested) = config.max_features
        && (requested == 0 || requested > data.columns())
    {
        return Err(ModelError::InvalidMaxFeatures {
            requested,
            available: data.columns(),
        });
    }
    // No finiteness scan of `data`. Every value in a `MatrixView` is finite by
    // construction, and that is now true of the crate-internal constructor as
    // well as the public one, so a rescan here could only ever re-derive the
    // container's own invariant at O(rows × columns) on every fit.
    Ok(())
}

pub(super) fn check_row(row: &[f32], expected: usize) -> Result<(), ModelError> {
    if row.len() != expected {
        return Err(ModelError::FeatureDimension {
            expected,
            actual: row.len(),
        });
    }
    if let Some(column) = row.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteFeature { row: 0, column });
    }
    Ok(())
}

pub(super) fn check_prediction_data(
    data: &MatrixView<'_>,
    output_len: usize,
    expected_output_len: usize,
    expected_features: usize,
) -> Result<(), ModelError> {
    if data.columns() != expected_features {
        return Err(ModelError::FeatureDimension {
            expected: expected_features,
            actual: data.columns(),
        });
    }
    check_output_len(output_len, expected_output_len)
}

pub(super) fn check_output_len(actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::OutputLength { expected, actual });
    }
    Ok(())
}

/// The fixed words every standalone tree artifact opens with.
pub(super) struct CommonMetadata {
    pub(super) n_features_in: usize,
    pub(super) max_depth: Option<usize>,
    pub(super) min_samples_split: usize,
    pub(super) min_samples_leaf: usize,
    pub(super) max_features: MaxFeatures,
    pub(super) splitter: Splitter,
    pub(super) random_state: u64,
    pub(super) node_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_common_metadata(
    metadata: &mut ArtifactPayloadWriter,
    objective_version: u32,
    n_features_in: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: (u32, u32),
    splitter: u32,
    random_state: u64,
    node_count: usize,
) -> Result<(), ArtifactError> {
    let narrow = |value: usize| u32::try_from(value).map_err(|_| ArtifactError::InvalidPayload);
    metadata.u32(objective_version);
    metadata.u32(narrow(n_features_in)?);
    // `0` is not a legal depth, so it is free to mean "unbounded" and the
    // encoding needs no separate presence flag.
    metadata.u32(max_depth.map(narrow).transpose()?.unwrap_or(0));
    metadata.u32(narrow(min_samples_split)?);
    metadata.u32(narrow(min_samples_leaf)?);
    metadata.u32(max_features.0);
    metadata.u32(max_features.1);
    metadata.u32(splitter);
    metadata.u64(random_state);
    metadata.u32(narrow(node_count)?);
    Ok(())
}

pub(super) fn read_common_metadata(
    metadata: &mut ArtifactCursor<'_>,
    expected_objective_version: u32,
) -> Result<CommonMetadata, ArtifactError> {
    let objective_version = metadata.u32()?;
    let n_features_in = metadata.u32()? as usize;
    let encoded_depth = metadata.u32()? as usize;
    let min_samples_split = metadata.u32()? as usize;
    let min_samples_leaf = metadata.u32()? as usize;
    let max_features_tag = metadata.u32()?;
    let max_features_count = metadata.u32()?;
    let splitter_tag = metadata.u32()?;
    let random_state = metadata.u64()?;
    let node_count = metadata.u32()? as usize;
    if objective_version != expected_objective_version
        || n_features_in == 0
        || n_features_in > MAX_ARTIFACT_FEATURES
        || encoded_depth > MAX_ARTIFACT_TOTAL_NODES
        || min_samples_split < 2
        || min_samples_leaf == 0
        || node_count == 0
        || node_count > MAX_ARTIFACT_TOTAL_NODES
    {
        return Err(ArtifactError::InvalidPayload);
    }
    let (Some(max_features), Some(splitter)) = (
        decode_max_features(max_features_tag, max_features_count, n_features_in),
        decode_splitter(splitter_tag),
    ) else {
        return Err(ArtifactError::InvalidPayload);
    };
    Ok(CommonMetadata {
        n_features_in,
        max_depth: (encoded_depth != 0).then_some(encoded_depth),
        min_samples_split,
        min_samples_leaf,
        max_features,
        splitter,
        random_state,
        node_count,
    })
}
