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
- `metrics::ConfusionMatrix`, counting one classification result over the
  sorted union of the observed labels, and the `metrics::Average` vocabulary
  that combines its per-class precision, recall, F1, and F-beta as a binary,
  micro, macro, or support-weighted score. Micro-averaging a single-label
  result equals accuracy, and binary averaging equals the standalone binary
  functions exactly.
- `metrics::Averaging` and `metrics::ZeroDivision`, making the treatment of a
  class with an empty denominator an explicit typed choice. The default reports
  `MetricError::Undefined` instead of substituting a value.
- `ConfusionMatrix::balanced_accuracy`, mean recall over the classes that have
  true rows, and `ConfusionMatrix::matthews_correlation`, which correlates
  expected and predicted labels over any number of classes and is undefined
  rather than zero when either side is constant.
- `metrics::roc_curve`, `metrics::precision_recall_curve`, and
  `metrics::average_precision_score`, which sweep the decision threshold over
  the same tie-aware score ordering ROC AUC uses, so curve and scalar results
  agree by construction.
- `metrics::median_absolute_error`, `metrics::explained_variance_score`, and
  `metrics::mean_absolute_percentage_error`. The percentage error treats every
  expected value as a denominator, so a single expected zero is
  `MetricError::Undefined` rather than a silently clamped floor.
- `model_selection::TimeSeriesSplit`, a forward-chaining splitter for ordered
  observations. Each fold trains on a prefix and tests on the window that
  follows it, with an optional `gap`, so no fold is ever fitted on a row that
  comes after the rows it is evaluated on.
- `model_selection::LeaveOneOut`, which holds out one sample per split.
- `model_selection::GroupKFold`, which assigns whole groups to folds so no
  group is ever on both sides of a split. Assignment is deterministic and needs
  no seed: largest group first, into the fold holding the fewest rows.
- `model_selection::RepeatedKFold`, which runs shuffled K-fold several times
  with a per-repeat seed derived from one configured seed, so a caller can
  separate model variance from partition variance reproducibly.
- `Split::partial`, for a split that deliberately leaves rows out of both
  partitions — what a forward-chaining fold needs so that the rows after its
  test window leak into neither side — plus `Split::covered_samples`.
  `Split::sample_count` now reports the dataset size a split was built for,
  which is unchanged for every complete split.
- `model_selection::ClassificationScore` and `model_selection::RegressionScore`,
  the open scorer contract that batch scoring, cross-validation, and
  permutation importance all consume identically. The existing
  `ClassificationScorer` and `RegressionScorer` enums remain the built-in set
  and now implement these traits, including a declared `greater_is_better`
  orientation, so a caller can score on a metric FerricML does not enumerate.
- `model_selection::ScoringWorkspace` with `score_classifier_with` and
  `score_regressor_with`, the allocation-free scoring entry points. Reusing one
  workspace across calls of the same shape allocates only on the first call.
- `model_selection::ClassifierOutput` and `model_selection::ClassifierOutputKind`,
  which let a classification score declare whether it reads predicted labels or
  positive-class probabilities; being given another kind is the new
  `ScoringError::UnsupportedOutput` rather than a substituted value.
- `model_selection::GroupShuffleSplit`, which draws whole groups at random for
  each of `n_splits` independent holdouts, so no group is ever on both sides of
  a split. Splits are independent draws rather than a partition, and each one's
  draw is seeded from the configured seed and the split index.
- `model_selection::TestGroupSize`, the holdout size for a grouped split. It
  counts **groups**, deliberately as a type distinct from `TestSize`, which
  counts rows: rows only move a whole group at a time, so a row target could
  only be approximated, while a group target is exact. A size that would empty
  either side is the new `SplitError::InvalidTestGroupCount`, which names groups
  rather than reusing the row-counting `InvalidTestCount`.
- `model_selection::ParameterGrid`, an ordered set of typed hyperparameter
  candidates. An axis is a parameter type's own `with_*` builder method plus the
  values to pass it, so there are no string keys and a misnamed parameter is a
  compile error. Different axes carry different value types, the axis added last
  varies fastest, and `from_candidates` takes an explicit list for parameters
  that are not independent.
- `model_selection::grid_search_classifier` and
  `model_selection::grid_search_regressor`, serial typed hyperparameter search.
  The split iterator is drained once, so every candidate is cross-validated over
  exactly the same folds, and each candidate runs through the existing
  cross-validation and scorer path rather than a second evaluation path. The
  result reports every candidate's parameters and every fold's score through
  `model_selection::SearchResult` and `model_selection::CandidateScores`; the
  winner is the best mean fold score in the direction the score declares, with
  an exact tie keeping the earliest candidate in grid order. Search does not
  refit.
- `model_selection::SearchError`, which separates a call that is unusable before
  any fitting (`Setup`) from a candidate's own failure (`Candidate`, keeping the
  fold index) and from a score that returns a value no ranking can order
  (`NonFiniteScore`).
- `api::Capabilities::multiclass`, declaring that an estimator offers a
  multiclass fitting entry point over `ClassTargets`. `LogisticRegression` and
  `RandomForestClassifier` declare it; `AnyClassifier` declares it away
  structurally, because it owns fitted models and no fitting entry point.
  The estimator conformance battery drives a new
  `multiclass_declaration_matches_behavior` obligation from it.
- `metrics::multiclass_log_loss` and `metrics::multiclass_brier_score`, which
  score a whole row-major probability matrix against a sorted class list.
  Neither renormalizes a row, because FerricML's rows sum to one only within
  the documented `f32` tolerance. `multiclass_log_loss` agrees with `log_loss`
  at two classes; `multiclass_brier_score` squares every column where
  `brier_score` squares only the positive one, so it is exactly twice the
  binary value there — stated rather than left to be discovered. The new
  `MetricError::InvalidClassSet` and `MetricError::UnknownClass` report a class
  list that cannot name columns and a label with no column.
- `RandomForestClassifier::fit_multiclass`, a natively multiclass forest whose
  trees split on multiclass Gini impurity and store one probability per class
  at every leaf. The ensemble probability is the mean of the per-tree
  probability vectors — soft averaging, not a majority vote of per-tree labels
  — and the predicted label is the argmax of exactly those probabilities. A
  single observed class fits and returns one all-ones column. Binary `fit`
  keeps its scalar-leaf representation and every fitted value it had.
- `LogisticRegression::fit_multiclass` and `fit_multiclass_weighted`, a joint
  multinomial fit over `data::ClassTargets`. It is one optimization over all
  classes, not a wrapper around per-class binary models: probabilities are the
  softmax of one centred score vector, and no class is pinned as a reference.
  `decision_function` now returns `n_decision_columns` values per row, which is
  one for a binary fit — that shape, and every binary fitted value, is
  unchanged. `intercepts` and `n_decision_columns` are new; `intercept` reports
  the first score row and is no longer `const`. Probability rows are **not**
  renormalized: they sum to one only within `n_classes` `f32` ulps.
- `data::ClassTargets`, validated general classification targets over arbitrary
  `u8` labels. It carries the sorted, deduplicated set of labels actually
  observed, which is the probability column order of any classifier fitted on
  it. Labels are never assumed contiguous or zero-based, so `{7, 3, 10}` gives
  classes `[3, 7, 10]`, and selecting a subset recomputes the observed set.
- `model_selection::ClassifierOutput::ProbabilityMatrix`, a batch output
  carrying a whole row-major probability matrix together with the class list
  naming its columns, and the matching
  `ClassifierOutputKind::ProbabilityMatrix`. A score reading it is independent
  of the binary class layouts, so it works for any observed class set.
- `model_selection::ClassificationScorer::MulticlassLogLoss` and
  `MulticlassBrier`, which read that matrix. Cross-entropy agrees with the
  binary `LogLoss` on two classes; the multiclass Brier score squares every
  column and is therefore exactly twice the binary one, as documented.
- `model_selection::score_multiclass_classifier` and
  `score_multiclass_classifier_with`, scoring a fitted classifier against
  `data::ClassTargets`. They share one implementation with the binary entry
  points, so there is still a single prediction and class-layout path.
- `calibration::IsotonicRegression`, a deterministic pool-adjacent-violators
  monotone fit. It is both the non-parametric probability calibrator and a
  standalone monotone regressor over a single-column matrix. Observations that
  share an input value are pooled into their mean before pooling adjacent
  violators, so the fit depends on the multiset of observations and not on their
  order; prediction interpolates linearly between fitted points and holds the
  end values outside the fitted range.
- `calibration::PlattCalibrator` and `calibration::PlattParams`, the parametric
  calibrator: a two-parameter logistic fit of model scores onto labels, solved
  through the crate's shared binary log-loss objective rather than a third
  logistic solver. It regresses on Platt's prior-corrected targets, so a
  perfectly separating score still has a finite fit and calibrated
  probabilities never collapse to exactly zero or one.
- `calibration::Calibrator`, the fitted monotone score-to-probability map
  contract, with an in-place batch form so calibrated prediction needs no second
  buffer.

### Changed

- Permutation importance takes any score implementing the new scorer traits and
  runs through the shared allocation-free scoring path, so it no longer carries
  its own copy of the scorer dispatch, the singleton-class probability
  handling, or the per-metric orientation table. Its proven allocation bound is
  unchanged.
- `score_classifier`, `score_regressor`, `cross_validate_classifier`, and
  `cross_validate_regressor` are generic over the new scorer traits instead of
  taking the built-in enums. Calls that pass a built-in scorer are unaffected;
  a turbofished `cross_validate_*` call gains one inferred type argument.

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
