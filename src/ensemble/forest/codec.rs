//! The artifact codec every bagged tree ensemble shares.
//!
//! One encoder and one decoder per fitted shape, parameterised by the artifact
//! kind the estimator owns. The field order, the constants, and the ceilings
//! are the on-disk contract, so they are stated once: two ensembles writing the
//! same metadata block through two copies of this code would be two places for
//! that contract to drift, and a drift no test outside the estimator that moved
//! would see.

use super::model::{Forest, MAX_CLASSES, prediction_bound_is_finite};
use super::parameters::{ForestFields, decode_n_jobs, encode_n_jobs};
use crate::artifact::{
    ArtifactCursor, ArtifactError, ArtifactPayloadWriter, LogicalTreeNode, MIN_ENCODED_TREE_BYTES,
    SchemaRole, decode_component, decode_logical_tree, decode_v2_envelope, encode_component,
    encode_logical_tree, encode_v2_envelope,
};
use crate::tree::{ClassTree, FEATURE_MASK, PackedTree, decode_max_features, encode_max_features};

pub(crate) const REGRESSOR_PAYLOAD_VERSION: u16 = 1;
pub(crate) const CLASSIFIER_PAYLOAD_VERSION: u16 = 1;
const METADATA_COMPONENT_KIND: u16 = 1;
const TREE_COMPONENT_KIND: u16 = 2;
/// Per-tree leaf distributions, in pre-order leaf rank. Written only by the
/// multiclass flavour, immediately after that tree's topology component.
const LEAF_PROBABILITY_COMPONENT_KIND: u16 = 3;
const COMPONENT_VERSION: u16 = 1;
const REGRESSOR_OBJECTIVE_VERSION: u32 = 1;
const CLASSIFIER_OBJECTIVE_VERSION: u32 = 1;
const METADATA_BYTES: usize = 13 * 4 + 8;
/// The classifier metadata's fixed words, before its class list: the regressor
/// fields plus a forest-flavour tag and a class count.
const CLASSIFIER_METADATA_BYTES: usize = 15 * 4 + 8;

/// Which leaf arithmetic the encoded forest uses. The two are different models,
/// so the tag is read before any tree is, and neither flavour's trees are ever
/// handed to the other's builder.
const FOREST_BINARY: u32 = 1;
const FOREST_MULTICLASS: u32 = 2;

/// Ceilings applied identically when encoding and decoding, so an artifact
/// that this crate produced always decodes and a hostile one allocates
/// nothing unbounded.
const MAX_ARTIFACT_FEATURES: usize = 1_000_000;
const MAX_ARTIFACT_TREES: usize = 4_096;
const MAX_ARTIFACT_TOTAL_NODES: usize = 1_048_576;

// A decoded feature index must never reach the packed layout's flag bits.
const _: () = assert!(MAX_ARTIFACT_FEATURES < FEATURE_MASK as usize);

pub(crate) fn encode_classifier(
    kind: u16,
    fields: &ForestFields,
    classes: &[u8],
    forest: &Forest,
    schema: [u8; 32],
) -> Result<Vec<u8>, ArtifactError> {
    let (flavour, tree_count, total_nodes) = match forest {
        Forest::Binary(trees) => (
            FOREST_BINARY,
            trees.len(),
            trees.iter().try_fold(0_usize, |total, tree| {
                total
                    .checked_add(tree.logical_node_count())
                    .ok_or(ArtifactError::InvalidPayload)
            })?,
        ),
        Forest::Multiclass(trees) => (
            FOREST_MULTICLASS,
            trees.len(),
            trees.iter().try_fold(0_usize, |total, tree| {
                total
                    .checked_add(tree.logical_node_count())
                    .ok_or(ArtifactError::InvalidPayload)
            })?,
        ),
    };
    if tree_count > MAX_ARTIFACT_TREES
        || total_nodes > MAX_ARTIFACT_TOTAL_NODES
        || fields.n_features_in > MAX_ARTIFACT_FEATURES
        || classes.is_empty()
        || classes.len() > MAX_CLASSES
    {
        return Err(ArtifactError::InvalidPayload);
    }

    let n_features =
        u32::try_from(fields.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
    let n_estimators =
        u32::try_from(fields.n_estimators).map_err(|_| ArtifactError::InvalidPayload)?;
    let max_depth = fields
        .max_depth
        .map(u32::try_from)
        .transpose()
        .map_err(|_| ArtifactError::InvalidPayload)?
        .unwrap_or(0);
    let min_samples_split =
        u32::try_from(fields.min_samples_split).map_err(|_| ArtifactError::InvalidPayload)?;
    let min_samples_leaf =
        u32::try_from(fields.min_samples_leaf).map_err(|_| ArtifactError::InvalidPayload)?;
    let (max_features_tag, max_features_count) = encode_max_features(fields.max_features)?;
    let (n_jobs_tag, n_jobs_count) = encode_n_jobs(fields.n_jobs)?;
    let tree_count = u32::try_from(tree_count).map_err(|_| ArtifactError::InvalidPayload)?;
    let total_nodes = u32::try_from(total_nodes).map_err(|_| ArtifactError::InvalidPayload)?;
    let class_count = u32::try_from(classes.len()).map_err(|_| ArtifactError::InvalidPayload)?;

    let mut metadata =
        ArtifactPayloadWriter::with_capacity(CLASSIFIER_METADATA_BYTES + classes.len() * 4);
    metadata.u32(CLASSIFIER_OBJECTIVE_VERSION);
    metadata.u32(flavour);
    metadata.u32(n_features);
    metadata.u32(n_estimators);
    metadata.u32(max_depth);
    metadata.u32(min_samples_split);
    metadata.u32(min_samples_leaf);
    metadata.u32(max_features_tag);
    metadata.u32(max_features_count);
    metadata.u32(u32::from(fields.bootstrap));
    metadata.u64(fields.random_state);
    metadata.u32(n_jobs_tag);
    metadata.u32(n_jobs_count);
    metadata.u32(tree_count);
    metadata.u32(total_nodes);
    metadata.u32(class_count);
    for &class in classes {
        metadata.u32(u32::from(class));
    }
    let mut payload = encode_component(
        METADATA_COMPONENT_KIND,
        COMPONENT_VERSION,
        &metadata.finish(),
    )?;
    match forest {
        Forest::Binary(trees) => {
            for tree in trees {
                payload.extend_from_slice(&encode_component(
                    TREE_COMPONENT_KIND,
                    COMPONENT_VERSION,
                    &encode_logical_tree(&tree.to_logical_nodes())?,
                )?);
            }
        }
        Forest::Multiclass(trees) => {
            for tree in trees {
                let (nodes, probabilities) = tree.to_logical_nodes();
                payload.extend_from_slice(&encode_component(
                    TREE_COMPONENT_KIND,
                    COMPONENT_VERSION,
                    &encode_logical_tree(&nodes)?,
                )?);
                let mut block = ArtifactPayloadWriter::with_capacity(8 + probabilities.len() * 4);
                block.u32(
                    u32::try_from(probabilities.len() / classes.len())
                        .map_err(|_| ArtifactError::InvalidPayload)?,
                );
                block.u32(class_count);
                for &value in &probabilities {
                    block.f32(value);
                }
                payload.extend_from_slice(&encode_component(
                    LEAF_PROBABILITY_COMPONENT_KIND,
                    COMPONENT_VERSION,
                    &block.finish(),
                )?);
            }
        }
    }
    encode_v2_envelope(
        kind,
        CLASSIFIER_PAYLOAD_VERSION,
        &[(SchemaRole::Input, schema)],
        &payload,
    )
}

pub(crate) fn decode_classifier(
    kind: u16,
    bytes: &[u8],
    schema: [u8; 32],
) -> Result<(ForestFields, Vec<u8>, Forest), ArtifactError> {
    let mut envelope = decode_v2_envelope(
        bytes,
        kind,
        CLASSIFIER_PAYLOAD_VERSION,
        &[(SchemaRole::Input, schema)],
    )?;
    let mut metadata = decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
    let objective_version = metadata.u32()?;
    let flavour = metadata.u32()?;
    let n_features_in = metadata.u32()? as usize;
    let n_estimators = metadata.u32()? as usize;
    let encoded_depth = metadata.u32()? as usize;
    let min_samples_split = metadata.u32()? as usize;
    let min_samples_leaf = metadata.u32()? as usize;
    let max_features_tag = metadata.u32()?;
    let max_features_count = metadata.u32()?;
    let bootstrap = metadata.u32()?;
    let random_state = metadata.u64()?;
    let n_jobs_tag = metadata.u32()?;
    let n_jobs_count = metadata.u32()?;
    let tree_count = metadata.u32()? as usize;
    let declared_total_nodes = metadata.u32()? as usize;
    let class_count = metadata.u32()? as usize;
    if objective_version != CLASSIFIER_OBJECTIVE_VERSION
        || (flavour != FOREST_BINARY && flavour != FOREST_MULTICLASS)
        || n_features_in == 0
        || n_features_in > MAX_ARTIFACT_FEATURES
        || n_estimators == 0
        || n_estimators != tree_count
        || tree_count > MAX_ARTIFACT_TREES
        || encoded_depth > MAX_ARTIFACT_TOTAL_NODES
        || min_samples_split < 2
        || min_samples_leaf == 0
        || bootstrap > 1
        || declared_total_nodes < tree_count
        || declared_total_nodes > MAX_ARTIFACT_TOTAL_NODES
        || class_count == 0
        || class_count > MAX_CLASSES
    {
        return Err(ArtifactError::InvalidPayload);
    }
    let mut classes: Vec<u8> = Vec::with_capacity(metadata.bounded_capacity(class_count, 4));
    for _ in 0..class_count {
        let label = u8::try_from(metadata.u32()?).map_err(|_| ArtifactError::InvalidPayload)?;
        if classes.last().is_some_and(|&previous| previous >= label) {
            return Err(ArtifactError::InvalidPayload);
        }
        classes.push(label);
    }
    // A binary fit is asymmetric: its scalar leaf is the probability of
    // class `1`, and prediction reads the label straight out of that
    // comparison. Only `[0]`, `[1]`, and `[0, 1]` mean anything there.
    if flavour == FOREST_BINARY && classes.iter().any(|&label| label > 1) {
        return Err(ArtifactError::InvalidPayload);
    }
    if !metadata.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    let (Some(max_features), Some(n_jobs)) = (
        decode_max_features(max_features_tag, max_features_count, n_features_in),
        decode_n_jobs(n_jobs_tag, n_jobs_count),
    ) else {
        return Err(ArtifactError::InvalidPayload);
    };
    let fields = ForestFields {
        n_features_in,
        n_estimators,
        max_depth: (encoded_depth != 0).then_some(encoded_depth),
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap: bootstrap == 1,
        random_state,
        n_jobs,
    };

    let mut actual_total_nodes = 0_usize;
    let forest = if flavour == FOREST_BINARY {
        let mut trees =
            Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
        for _ in 0..tree_count {
            let logical = decode_logical_tree(decode_component(
                &mut envelope,
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
            )?)?;
            actual_total_nodes =
                accumulate_nodes(actual_total_nodes, logical.len(), declared_total_nodes)?;
            // A fitted binary leaf is a probability, so nothing else is a
            // model this crate could have produced.
            if logical.iter().any(|node| {
                matches!(node, LogicalTreeNode::Leaf { value } if !(0.0..=1.0).contains(value))
            }) {
                return Err(ArtifactError::InvalidPayload);
            }
            trees.push(PackedTree::from_logical_nodes(&logical, n_features_in)?);
        }
        Forest::Binary(trees)
    } else {
        let mut trees =
            Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
        for _ in 0..tree_count {
            let logical = decode_logical_tree(decode_component(
                &mut envelope,
                TREE_COMPONENT_KIND,
                COMPONENT_VERSION,
            )?)?;
            actual_total_nodes =
                accumulate_nodes(actual_total_nodes, logical.len(), declared_total_nodes)?;
            let probabilities = decode_leaf_probabilities(
                decode_component(
                    &mut envelope,
                    LEAF_PROBABILITY_COMPONENT_KIND,
                    COMPONENT_VERSION,
                )?,
                class_count,
            )?;
            trees.push(ClassTree::from_logical_nodes(
                &logical,
                &probabilities,
                class_count,
                n_features_in,
            )?);
        }
        Forest::Multiclass(trees)
    };
    if !envelope.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    if actual_total_nodes != declared_total_nodes {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok((fields, classes, forest))
}

pub(crate) fn encode_regressor(
    kind: u16,
    fields: &ForestFields,
    trees: &[PackedTree],
    schema: [u8; 32],
) -> Result<Vec<u8>, ArtifactError> {
    let n_features =
        u32::try_from(fields.n_features_in).map_err(|_| ArtifactError::InvalidPayload)?;
    let n_estimators =
        u32::try_from(fields.n_estimators).map_err(|_| ArtifactError::InvalidPayload)?;
    let max_depth = fields
        .max_depth
        .map(u32::try_from)
        .transpose()
        .map_err(|_| ArtifactError::InvalidPayload)?
        .unwrap_or(0);
    let min_samples_split =
        u32::try_from(fields.min_samples_split).map_err(|_| ArtifactError::InvalidPayload)?;
    let min_samples_leaf =
        u32::try_from(fields.min_samples_leaf).map_err(|_| ArtifactError::InvalidPayload)?;
    let (max_features_tag, max_features_count) = encode_max_features(fields.max_features)?;
    let (n_jobs_tag, n_jobs_count) = encode_n_jobs(fields.n_jobs)?;
    let tree_count = u32::try_from(trees.len()).map_err(|_| ArtifactError::InvalidPayload)?;
    let total_nodes = trees.iter().try_fold(0_usize, |total, tree| {
        total
            .checked_add(tree.logical_node_count())
            .ok_or(ArtifactError::InvalidPayload)
    })?;
    if trees.len() > MAX_ARTIFACT_TREES
        || total_nodes > MAX_ARTIFACT_TOTAL_NODES
        || fields.n_features_in > MAX_ARTIFACT_FEATURES
        || !prediction_bound_is_finite(trees)
    {
        return Err(ArtifactError::InvalidPayload);
    }
    let total_nodes = u32::try_from(total_nodes).map_err(|_| ArtifactError::InvalidPayload)?;

    let mut metadata = ArtifactPayloadWriter::with_capacity(METADATA_BYTES);
    metadata.u32(REGRESSOR_OBJECTIVE_VERSION);
    metadata.u32(n_features);
    metadata.u32(n_estimators);
    metadata.u32(max_depth);
    metadata.u32(min_samples_split);
    metadata.u32(min_samples_leaf);
    metadata.u32(max_features_tag);
    metadata.u32(max_features_count);
    metadata.u32(u32::from(fields.bootstrap));
    metadata.u64(fields.random_state);
    metadata.u32(n_jobs_tag);
    metadata.u32(n_jobs_count);
    metadata.u32(tree_count);
    metadata.u32(total_nodes);
    let mut payload = encode_component(
        METADATA_COMPONENT_KIND,
        COMPONENT_VERSION,
        &metadata.finish(),
    )?;
    for tree in trees {
        payload.extend_from_slice(&encode_component(
            TREE_COMPONENT_KIND,
            COMPONENT_VERSION,
            &encode_logical_tree(&tree.to_logical_nodes())?,
        )?);
    }
    encode_v2_envelope(
        kind,
        REGRESSOR_PAYLOAD_VERSION,
        &[(SchemaRole::Input, schema)],
        &payload,
    )
}

pub(crate) fn decode_regressor(
    kind: u16,
    bytes: &[u8],
    schema: [u8; 32],
) -> Result<(ForestFields, Vec<PackedTree>), ArtifactError> {
    let mut envelope = decode_v2_envelope(
        bytes,
        kind,
        REGRESSOR_PAYLOAD_VERSION,
        &[(SchemaRole::Input, schema)],
    )?;
    let mut metadata = decode_component(&mut envelope, METADATA_COMPONENT_KIND, COMPONENT_VERSION)?;
    let objective_version = metadata.u32()?;
    let n_features_in = metadata.u32()? as usize;
    let n_estimators = metadata.u32()? as usize;
    let encoded_depth = metadata.u32()? as usize;
    let min_samples_split = metadata.u32()? as usize;
    let min_samples_leaf = metadata.u32()? as usize;
    let max_features_tag = metadata.u32()?;
    let max_features_count = metadata.u32()?;
    let bootstrap = metadata.u32()?;
    let random_state = metadata.u64()?;
    let n_jobs_tag = metadata.u32()?;
    let n_jobs_count = metadata.u32()?;
    let tree_count = metadata.u32()? as usize;
    let declared_total_nodes = metadata.u32()? as usize;
    if !metadata.is_empty()
        || objective_version != REGRESSOR_OBJECTIVE_VERSION
        || n_features_in == 0
        || n_features_in > MAX_ARTIFACT_FEATURES
        || n_estimators == 0
        || n_estimators != tree_count
        || tree_count > MAX_ARTIFACT_TREES
        || encoded_depth > MAX_ARTIFACT_TOTAL_NODES
        || min_samples_split < 2
        || min_samples_leaf == 0
        || bootstrap > 1
        || declared_total_nodes < tree_count
        || declared_total_nodes > MAX_ARTIFACT_TOTAL_NODES
    {
        return Err(ArtifactError::InvalidPayload);
    }
    let (Some(max_features), Some(n_jobs)) = (
        decode_max_features(max_features_tag, max_features_count, n_features_in),
        decode_n_jobs(n_jobs_tag, n_jobs_count),
    ) else {
        return Err(ArtifactError::InvalidPayload);
    };
    let fields = ForestFields {
        n_features_in,
        n_estimators,
        max_depth: (encoded_depth != 0).then_some(encoded_depth),
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap: bootstrap == 1,
        random_state,
        n_jobs,
    };

    let mut trees =
        Vec::with_capacity(envelope.bounded_capacity(tree_count, MIN_ENCODED_TREE_BYTES));
    let mut actual_total_nodes = 0_usize;
    for _ in 0..tree_count {
        let logical = decode_logical_tree(decode_component(
            &mut envelope,
            TREE_COMPONENT_KIND,
            COMPONENT_VERSION,
        )?)?;
        actual_total_nodes = actual_total_nodes
            .checked_add(logical.len())
            .ok_or(ArtifactError::InvalidPayload)?;
        if actual_total_nodes > declared_total_nodes {
            return Err(ArtifactError::InvalidPayload);
        }
        trees.push(PackedTree::from_logical_nodes(&logical, n_features_in)?);
    }
    if !envelope.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    if actual_total_nodes != declared_total_nodes || !prediction_bound_is_finite(&trees) {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok((fields, trees))
}

/// Adds one decoded tree's records to the running total, refusing to pass the
/// count the metadata declared before the next tree is even read.
fn accumulate_nodes(total: usize, added: usize, declared: usize) -> Result<usize, ArtifactError> {
    let total = total
        .checked_add(added)
        .ok_or(ArtifactError::InvalidPayload)?;
    if total > declared {
        return Err(ArtifactError::InvalidPayload);
    }
    Ok(total)
}

/// Reads one tree's leaf distributions, in pre-order leaf rank.
///
/// The declared leaf count is checked against the class count and against the
/// bytes actually present before anything is reserved, and every value must be
/// the finite `0..=1` a fitted leaf distribution holds.
fn decode_leaf_probabilities(
    mut cursor: ArtifactCursor<'_>,
    class_count: usize,
) -> Result<Vec<f32>, ArtifactError> {
    let leaves = cursor.u32()? as usize;
    let declared_classes = cursor.u32()? as usize;
    let expected = leaves.checked_mul(class_count);
    if leaves == 0 || declared_classes != class_count || expected.is_none() {
        return Err(ArtifactError::InvalidPayload);
    }
    let expected = expected.expect("checked above");
    let mut probabilities = Vec::with_capacity(cursor.bounded_capacity(expected, 4));
    for _ in 0..expected {
        let value = cursor.f32()?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ArtifactError::InvalidPayload);
        }
        probabilities.push(value);
    }
    if !cursor.is_empty() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(probabilities)
}
