//! Public random-forest parameter types.

use super::super::forest::forest_params;
use crate::tree::{MaxFeatures, Splitter};

forest_params!(
    RandomForestClassifierParams,
    MaxFeatures::Sqrt,
    Splitter::Best
);
forest_params!(
    RandomForestRegressorParams,
    MaxFeatures::All,
    Splitter::Best
);
