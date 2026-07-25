use super::super::forest::{forest_classifier, forest_regressor};
use super::parameters::{ExtraTreesClassifierParams, ExtraTreesRegressorParams};
use crate::artifact::{EXTRA_TREES_CLASSIFIER_ARTIFACT_KIND, EXTRA_TREES_REGRESSOR_ARTIFACT_KIND};

forest_classifier!(
    ExtraTreesClassifier,
    ExtraTreesClassifierParams,
    EXTRA_TREES_CLASSIFIER_ARTIFACT_KIND,
    "An extremely randomized tree classifier.",
    "",
    "Each member tree draws **one uniform threshold per candidate column**,",
    "over that column's own range within the node, and keeps the best-scoring",
    "draw — rather than evaluating every boundary between adjacent distinct",
    "values. The candidate columns themselves are drawn exactly as a random",
    "forest draws them. Trees therefore decorrelate through their thresholds",
    "instead of through resampling, which is why `bootstrap` defaults to",
    "`false` here and to `true` on a random forest.",
);

forest_regressor!(
    ExtraTreesRegressor,
    ExtraTreesRegressorParams,
    EXTRA_TREES_REGRESSOR_ARTIFACT_KIND,
    crate::tree::Regression,
    "An extremely randomized tree regressor. Predictions are averages of tree",
    "leaf means.",
    "",
    "Each member tree draws one uniform threshold per candidate column rather",
    "than optimizing within it; see [`ExtraTreesClassifier`] for what that",
    "changes and what it leaves alone.",
);
