# Changelog

All notable changes to FerricML are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Report `ModelError::NonFinitePrediction` from `RandomForestRegressor` instead
  of returning a non-finite averaged prediction, matching every other
  regressor.

## [0.1.2] - 2026-07-24

### Added

- Deterministic classification and regression metrics with explicit validation
  and undefined-result semantics.
- Checked row/target selection plus deterministic holdout, K-fold, and
  label-stratified dataset splitting.
- Batch-level fitted classifier and regressor scoring across built-in metrics.
- Deterministic serial cross-validation with typed fit closures, ordered fold
  scores, and fold-attributed errors.

### Changed

- Keep third-party provenance and regeneration tooling in local development
  state while retaining FerricML-owned frozen behavior and quality contracts.
- Organize implementation modules by capability and estimator family while
  keeping public paths and model artifacts stable.
- Accelerate exact train/test splitting, stratified quota assignment, standard
  scaling, ridge preprocessing, and logistic Newton fitting.
- Stabilize first-party performance history with repeated scalar inference and
  dedicated ordinary and stratified split workloads.

### Fixed

- Reject non-finite scalar prediction features and non-finite accumulated
  outputs consistently across linear, ridge, logistic, pairwise-ranking, and
  histogram-boosting models.

## [0.1.1] - 2026-07-23

### Added

- Validated sample weights, weighted logistic fitting, and allocation-free raw
  logistic decision scores.
- Bounded version-2 model artifacts with independent payload/component
  versions while retaining strict legacy logistic decoding.
- Dense weighted ordinary least-squares regression with deterministic
  minimum-norm SVD solutions and schema-bound artifacts.
- Dense weighted ridge regression plus runtime switching across forest, linear,
  and ridge regressors.
- Deterministic weighted standard scaling and schema-bound serialized
  scaler-to-logistic, scaler-to-linear, and scaler-to-ridge pipelines with
  caller-owned inference workspace.
- Pairwise linear ranking with explicit tie observations and thresholds,
  schema-bound artifacts, and tie-aware accuracy, Spearman, and Kendall tau-b
  metrics.
- Deterministic dense squared-error histogram gradient boosting with bounded
  bins, leaves, depth, iterations, allocation-free batch prediction, owned
  runtime switching, and schema-bound logical-tree artifacts.

### Fixed

- Honor `LogisticRegressionParams::with_fit_intercept(false)` without silently
  centering features or fitting a folded intercept.

## [0.1.0] - 2026-07-22

### Added

- Validated dense `f32` matrices and classification/regression targets.
- Deterministic random-forest classifiers and regressors with allocation-free
  batch inference.
- Deterministic logistic regression with stable typed parameters and
  allocation-free prediction.
- Versioned, checksummed logistic-regression artifacts with strict feature
  schema verification.
- Frozen estimator and prediction semantics for the supported API subset.
- Generic static-dispatch pipeline and transformer growth seams.
- Frozen correctness fixtures and an on-demand Rust implementation benchmark.

[Unreleased]: https://github.com/kkollsga/ferricml/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/kkollsga/ferricml/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kkollsga/ferricml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kkollsga/ferricml/releases/tag/v0.1.0
