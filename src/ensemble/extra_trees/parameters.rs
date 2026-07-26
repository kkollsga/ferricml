//! Public extra-trees parameter types.
//!
//! The same vocabulary a random forest takes, with two different defaults. The
//! split search is **not** a knob here: it is what the type means, so it is
//! fixed by the macro argument rather than exposed as a parameter a caller
//! could set to the value that would turn the estimator into a random forest
//! under a second name.

use super::super::forest::forest_params;
use crate::tree::{MaxFeatures, Splitter};

forest_params!(
    ExtraTreesClassifierParams,
    MaxFeatures::Sqrt,
    Splitter::Random
);
forest_params!(
    ExtraTreesRegressorParams,
    MaxFeatures::All,
    Splitter::Random
);
