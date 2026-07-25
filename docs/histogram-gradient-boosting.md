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

`fit_weighted` accepts validated per-row sample weights. A weight scales that
row's gradient and its share of every node's weight total, so the baseline is a
weighted mean, every leaf value and split gain is a weighted Newton step, and
`min_samples_leaf` bounds weight rather than rows. That last point is what makes
an integer weight the same fitted model as repeating the row that many times.
The bin grid is deliberately not weighted: it is fitted from the distinct
observed feature values, which neither a weight nor a repeated row changes, and
which is also why the two agree. Weights of exactly one reproduce the unweighted
fit bit for bit.

Public fitting and prediction live in the `ensemble` facade. The private
histogram-boosting estimator family separately owns fitted bin thresholds,
training-only mutable growth, persistence conversion, and compact prediction
trees. This separation keeps the public estimator contract independent of
private training and storage details.

Fitted models support deterministic, schema-bound, checksummed artifacts and
owned runtime switching through `AnyRegressor`. Persistence uses canonical
logical tree records, so compact prediction storage can evolve independently.
Decode validates all metadata, component framing, topology, feature indices,
numeric finiteness, and aggregate allocation bounds before exposing a model.

The initial scope intentionally excludes other losses, missing or categorical
features, feature subsampling, monotonic/interactions
constraints, early stopping, validation sets, warm starts, and parallel
training.
