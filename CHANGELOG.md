# Changelog

All notable changes to FerricML are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Schema-bound `RandomForestRegressor` artifacts that persist backend-neutral
  logical trees and revalidate every decoded topology, count, and parameter.
- `AnyRegressor` dispatch artifacts that record the fitted runtime variant and
  nest the estimator's own complete, independently validated artifact.
- Model-agnostic permutation feature importance over any fitted classifier or
  regressor, with a seeded permutation count, allocating and caller-owned entry
  points, and per-feature mean and dispersion of the scorer's quality loss.
- A compile-time estimator capability descriptor, `api::Capabilities`, carried
  by the `api::HasCapabilities` trait, so meta-layers query declared support
  for weighted fitting and artifact persistence instead of matching on concrete
  estimator types. The default declares nothing.
- Capability declarations on every fitted estimator and transformer FerricML
  ships, so callers can ask a type whether it supports weighted fitting or
  artifact persistence without matching on its concrete type.
- `dummy::DummyClassifier` and `dummy::DummyRegressor`, baseline estimators
  that ignore their features and predict the majority class or the training
  mean. They give a quality floor to compare a real estimator against.
- `preprocessing::MinMaxScaler`, a schema-bound transformer that maps every
  fitted feature onto `0.0..=1.0`, carries a column with no spread to `0.0`
  instead of dividing by zero, and can optionally clip later batches into the
  fitted range.
- `preprocessing::MaxAbsScaler`, a schema-bound transformer that divides each
  fitted feature by its largest observed magnitude, preserving sign and zeros,
  and passing an all-zero column through unchanged instead of dividing by zero.
- `pipeline::StagedPipeline`, a trainable multi-stage typed pipeline composing
  two or more fitted transform stages with one fitted estimator. `fit` trains
  every part in order on the previous stage's output, `new` validates every
  feature-width handoff before the composition exists, and inference runs
  through one caller-owned workspace that every stage writes a disjoint
  segment of.
- `pipeline::TransformerStack`, the statically dispatched stage-list contract
  `StagedPipeline` composes over, implemented for tuples of two and three
  fitted transformers.
- Schema-bound `StagedPipeline` artifacts under one artifact kind that records
  which concrete stage types the composition holds, in order, and which
  estimator type it ends in, so a composition never decodes as a different one.
  One capability declaration now covers every composition whose parts all
  persist, instead of one declaration per concrete composition.
- `pipeline::StageArtifact`, `pipeline::ModelArtifact`, and
  `pipeline::PersistedStack`, the persistence contracts a composition is
  generic over.
- `AnyClassifier::capabilities` and `AnyRegressor::capabilities`, reporting the
  selected runtime variant's capabilities. The declared constant on each
  dispatch enum and each concrete pipeline is composed from its variants or
  parts, so it promises only what holds whichever one is held.

### Fixed

- Report `ModelError::NonFinitePrediction` from `RandomForestRegressor` instead
  of returning a non-finite averaged prediction, matching every other
  regressor.
- Bound what every artifact decoder reserves by the bytes actually present
  rather than by a declared element count. A hostile artifact of roughly 150
  bytes could previously make a scaler, linear model, or forest reserve up to
  32 MB before reporting the truncation it was always going to report.
- Reject a logical tree whose records are laid out in any order other than the
  canonical pre-order the writer produces. Such a layout described a model that
  already had an encoding, so one fitted forest or boosted ensemble had more
  than one accepted artifact.
- Apply the documented 32 MiB reader limit to the legacy version-1 envelope as
  well as version 2. An oversized buffer whose version field read 1 was
  checksummed in full before being rejected.

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
