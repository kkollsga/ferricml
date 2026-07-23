# Histogram gradient-boosted regression

`HistGradientBoostingRegressor` implements deterministic serial squared-error
boosting for finite dense `f32` data and one scalar target. The fitted baseline
is the fixed-order `f64` target mean. Each iteration fits residuals with a
globally best positive-gain histogram tree and applies learning-rate shrinkage.

The supported typed parameters are `learning_rate`, `max_iter`,
`max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`, and
`max_bins`. Binning is deterministic, split ties retain feature/bin/node order,
and L2 regularization applies only to leaf denominators. Iterations, leaves,
depth, and bins have hard bounds before training work begins.

Public fitting and prediction live in the `ensemble` facade. Private
`boosting` modules separately own fitted bin thresholds, training-only mutable
growth, and canonical compact prediction trees. This mirrors the responsibility
boundaries of the scikit-learn reference without copying its implementation or
exposing runtime tree layout.

The initial scope intentionally excludes other losses, missing or categorical
features, sample weights, feature subsampling, monotonic/interactions
constraints, early stopping, validation sets, warm starts, and parallel
training.
