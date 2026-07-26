# Changelog

All notable changes to FerricML are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scripts/check_accessor_pairing.py`, run by `make gate`, which enforces that
  an `X_into` method and its allocating `X` partner are actually a pair: the
  caller-owned form takes exactly the allocating form's arguments plus its
  output buffers, every inherent `_into` has an inherent allocating partner,
  and a type forwarding one form inherently forwards the other too. The
  contract was written down and unenforced, which is how a single-row method
  came to hold a batch method's name on five classifiers and get copied into a
  sixth. Its `--self-test` proves each rule against a synthetic violation,
  proves that losing its input is reported rather than passed, and reconstructs
  the four defects it was written for from the baseline rows they occupied.

- `linear_model::LogisticRegression::predict_class_proba_into` is now reachable
  as an inherent method, matching the allocating `predict_class_proba` forwarder
  it already had and the pair every other probabilistic classifier ships.
  Reaching the allocation-free form previously required importing
  `api::ProbabilisticClassifier` while the allocating one did not, which
  inverted the crate's stated preference on hot paths. Behaviour is unchanged:
  the inherent method delegates to the same trait implementation.

- `pipeline::Pipeline::fit`, which fits the transformer and then the estimator
  on what the transformer produced. The two pipeline types disagreed on this:
  `StagedPipeline` could compose *and* fit, `Pipeline` could only compose. The
  one-transformer case is not a special case — it is the shortest composition —
  and fitting the estimator on untransformed rows is the one handoff error that
  yields a silently wrong model rather than a width mismatch.
- `pipeline::TransformerStack` and `pipeline::PersistedStack` are now
  implemented for every flat tuple of fitted transformers from one stage to
  twelve, published as `pipeline::MAX_STAGES`; they previously stopped at two
  and three. `StagedPipeline::new`, width-handoff validation, the single split
  workspace, the capability declaration, and the tagged artifact therefore all
  work at any of those lengths. Nothing about a shipped composition changes:
  the impls are additive and `StagedPipeline<(A, B), E>` is the same type it
  was. A right-nested `(A, (B, (C,)))` stack was considered and rejected — it
  conflicts (`E0119`) with a flat impl rather than joining it, it would rewrite
  the type parameters of every shipped composition, and its nesting carries no
  meaning; a fixed ceiling over flat tuples is what the standard library does.
  One-call fitting stays bounded at two stages, with the two measured attempts
  to lift it recorded on `StagedPipeline::fit`.
- `linear_model::Lasso` and `linear_model::ElasticNet` now persist, under their
  own artifact kinds. They were the last tunable regressors that could be fitted
  but not saved, which is backwards for an L1 model: a sparse coefficient vector
  is chosen precisely because it is the thing worth shipping. Both artifacts
  store the mixing weight and the coordinate-descent sweep count alongside the
  coefficients rather than re-deriving either, because both are readable on a
  fitted model. Neither has an `api::AnyRegressor` variant yet; that remains a
  dispatch gap rather than a contract gap, and adding one later will not touch
  either estimator's bytes.
- `api::ModelError::InvalidTreeStructure`, for a fitted tree whose topology or
  values the packed node format cannot represent at any size. It is separate
  from `TreeTooLarge`, which is a size bound; see the note under Changed for
  what now reports it.
- `tree::DecisionTreeClassifier` and `tree::DecisionTreeRegressor`, standalone
  decision trees over the same grower a random forest uses. Both support
  weighted fitting and persist under new artifact kinds; the classifier fits
  binary or natively multiclass targets and declares genuine probabilities,
  because a leaf *is* a distribution over the training rows that reached it.
  `MaxFeatures` is now also reachable as `tree::MaxFeatures`; the existing
  `ensemble::MaxFeatures` path is unchanged and names the same type.
- `tree::Splitter`, set through `with_splitter` on either standalone tree's
  parameters. `Splitter::Best` (the default) evaluates every boundary between
  adjacent distinct values in each candidate column; `Splitter::Random` draws
  one threshold uniformly inside each candidate column's own range within the
  node and keeps the best-scoring draw, which is what makes an *extremely
  randomized* tree. The candidate columns are drawn identically either way, an
  inadmissible draw is discarded rather than redrawn, and a column that is
  constant within the node consumes no draw at all — so the generator's stream
  does not depend on which columns happen to be constant. Random forests are
  unaffected: they keep the exhaustive search and their artifact bytes are
  unchanged.
- `ensemble::ExtraTreesClassifier` and `ensemble::ExtraTreesRegressor`,
  extremely randomized tree ensembles over the same core a random forest uses.
  Each member draws one uniform threshold per candidate column instead of
  optimizing within it; the candidate columns themselves are drawn exactly as a
  random forest draws them. `bootstrap` therefore defaults to `false` here and
  stays `true` on a random forest — trees decorrelate through their thresholds,
  so resampling on top of that would only remove training rows. Both persist
  under new artifact kinds, fit with or without sample weights, and the
  classifier fits binary or natively multiclass targets. An ensemble of one
  member is bit-identical to the corresponding standalone tree at the same
  seed, which is asserted rather than assumed.
- A narrative documentation site, built with MkDocs from the markdown already
  under `docs/` and configured for Read the Docs. Seven new guide pages — a
  quickstart, data and targets, linear models, trees and forests, preprocessing
  and pipelines, calibration and inspection, and saving and loading models —
  sit in front of the existing contract documents rather than replacing them.
  The site is the narrative guide and deliberately reproduces no API listings:
  the symbol-level reference is rustdoc on docs.rs, which regenerates from the
  code and cannot drift from it. Site machinery lives outside `docs/` and is
  excluded from the published crate, so the archive gains only markdown.
- Rustdoc usage examples on every public estimator that lacked one, covering
  the data containers, all five linear models, all eight tree-based estimators,
  all seven transformers, both pipelines, both dummies, the calibrators, the
  pairwise ranker, permutation importance and the runtime dispatch layer. These
  are doctests, so each is a compiled and executing test of the exact call a
  caller reaches for; the suite goes from 13 to 97.
- Every Rust sample in the narrative documentation is also a doctest. The pages
  under `docs/` are compiled into the test suite under `cfg(doctest)`, so a
  sample that stops compiling — or stops producing the value it claims — fails
  the ordinary gate. `tests/doc_examples.rs` fails if a page is left out of
  that mechanism, and rejects a sample marked `ignore` or `no_run` that carries
  no written reason, so the difference between a verified sample and an
  illustrative one is always visible to a reader.


- `preprocessing::RobustScaler` and `RobustScalerParams`: per-feature scaling by
  a median and a quantile spread. Both statistics are order statistics, so a
  handful of extreme rows move them far less than they move a mean and a
  standard deviation. `with_quantile_range` selects the percentile pair whose
  difference is removed — the interquartile range by default — and
  `with_centering` / `with_scaling` select which statistic the transform
  removes. Quantiles use linear interpolation between the two bracketing order
  statistics (Hyndman–Fan type 7), applied uniformly including at the median.

  A column with no spread keeps a divisor of one and survives as a constant,
  the same exact-zero rule the other three scalers already use. A column whose
  spread is merely *small* is scaled normally; if that overflows `f32` the
  batch is rejected with the offending row and column before anything is
  written, rather than being silently left unscaled.

  `unit_variance` is deliberately not claimed: it needs an inverse-normal-CDF
  primitive with its own accuracy contract, which is not worth adding to serve
  one optional flag.

  The scaler persists through `to_artifact` / `from_artifact` under artifact
  kind `44`, and composes into a `StagedPipeline` as a persisted stage. The raw
  spread is what is stored and the divisor is recomputed on decode, so a fitted
  model has exactly one valid byte string.
- `preprocessing::Normalizer`, `NormalizerParams`, and `Norm`: row-wise scaling
  so each row has unit `L1`, `L2`, or `Max` norm, where `Max` is the largest
  *magnitude*. A zero row has no direction to preserve, so it keeps a divisor
  of one and passes through unchanged.
- `preprocessing::Binarizer` and `BinarizerParams`: every value above a
  threshold becomes `1.0` and every other value `0.0`. The comparison is
  strictly greater-than, so a value exactly at the threshold becomes `0.0` and
  the two output classes are `(-inf, t]` and `(t, +inf)`.

  Both are stateless — they estimate nothing from the data beyond the width a
  pipeline hands them — and both therefore declare **no** capabilities at all,
  including no artifact. There is no fitted value to persist, so a persistence
  promise would be about something that does not exist. This is the same
  reasoning the baseline estimators already use.
- `preprocessing::FunctionTransformer`, `FunctionTransformerParams`, and the
  `ElementwiseFn` alias: a caller-supplied `fn(f32) -> f32` applied to every
  value, with an optional inverse.

  The map is a **function pointer, not a generic closure**. A capability
  declaration is an associated constant on a nameable type, and the capability
  snapshot asserts that every declaring public type appears in it by name — a
  type instantiated at an unnameable closure type would silently fall out of
  that coverage. A function pointer also captures no state, so two values of
  the type cannot behave differently. A caller who needs captured state, or a
  map that reads a whole row, implements `api::Transformer` directly.

  **Determinism of the supplied function is the caller's obligation.** FerricML
  guarantees the framing — fixed row-major order, validation before any write,
  and `ModelError::NonFiniteTransform` naming the first cell where a finite
  input maps to a non-finite output — but cannot guarantee the supplied
  function is pure.

  It declares no capabilities, including no artifact: a function pointer is an
  address in the current process image. It also has no `PartialEq`, because
  comparing function pointers compares addresses and one function is not
  guaranteed to have one address; an equality that is quietly wrong at a
  boundary is worse than none. Compare behaviour instead.
- `inverse_transform` and `inverse_transform_into` on `StandardScaler`,
  `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and `FunctionTransformer`,
  recovering pre-transform values into an allocated matrix or caller-owned
  storage.

  Exactness is stated rather than implied. The round trip is exact by
  construction only where no lossy operation happens — both statistics
  disabled, or a degenerate column whose divisor was substituted to one — and
  elsewhere is exact only when the arithmetic happens to be, since dividing by
  a scale and multiplying back is not a floating-point identity. `MinMaxScaler`
  with clipping enabled is deliberately **not** invertible in the usual sense:
  clipping is a projection, so inverting a clamped value recovers the fitted
  bound rather than the original.

  `FunctionTransformer::inverse_transform` returns
  `ModelError::NoInverseFunction` when no inverse was supplied, rather than
  silently applying the identity — which would look exactly like a successful
  recovery.
- `MinMaxScalerParams::with_feature_range` and `feature_range`, choosing the
  interval each column's fitted range is mapped onto. The default is unchanged
  at `0.0..=1.0`, `clip` now clamps into the configured interval, and a
  zero-range column lands on the interval's lower bound. An empty or inverted
  range is `ModelError::InvalidFeatureRange`, raised before any allocation.

  **Existing `MinMaxScaler` artifacts are byte-identical.** The output range is
  written only when it is one an older reader could not have assumed, so a
  default-configured scaler emits exactly the bytes it emitted before this
  parameter existed and every previously frozen artifact is unmoved. Older
  payloads are read, not rejected, and decode to an identical model. Each
  fitted model still has exactly one valid encoding, because the payload
  version is a function of the parameters rather than a choice — a default
  range written at the newer version is refused.
- `api::ModelError::InvalidFeatureRange`, raised when a min-max output range is
  not a finite interval with its minimum strictly below its maximum.
- `api::ModelError::NoInverseFunction`, raised when an inverse transformation is
  requested of a transformer that was not given one.
- `api::ModelError::InvalidThreshold`, raised when a decision threshold is not
  finite.
- `api::ModelError::InvalidQuantileRange`, raised when a quantile range is not
  two percentiles in `0.0..=100.0` with the lower value first. Equal
  percentiles are accepted and produce a zero spread, which is a legitimate way
  to ask for centering alone.
- `model_selection::ScorableClassifier`, the view the scoring layer takes of a
  fitted classifier: `probabilistic` for one that produces probabilities,
  `labels_only` for one that does not. A label metric works for either; a
  probability metric applied to a labels-only view is
  `ScoringError::UnsupportedOutput` naming what was required and what was
  supplied, never a substituted value. One type rather than a parallel family
  of entry points, and it makes "the labels and the probabilities come from the
  same model" true by construction.
- `model_selection::cross_validate_classifier_labels` and
  `model_selection::grid_search_classifier_labels`, for cross-validating and
  searching a classifier that produces labels but no probabilities. These build
  the model themselves, so the requirement is expressed in the bound rather
  than in an argument.
- `api::Capabilities::probability`, declaring whether a fitted classifier
  produces a probability per class. It is queryable on a runtime dispatch value
  through `capabilities()`, which is where a compile-time bound is unavailable.

- A capability declaration on `ranking::PairwiseLinearRanker`, which was the
  one fitted estimator that could not answer a capability query. It declares
  artifact persistence and nothing else: its weights belong to a pair
  observation rather than to a row, and a ranking score is not a probability,
  so it exposes neither a per-sample weighted fit nor a decision function.
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
- `ensemble::HistGradientBoostingClassifier`, a deterministic serial histogram
  gradient-boosted binary classifier fitted against binary log loss. It shares
  the regressor's binner, grower, and seven growth controls, and differs in
  dividing each leaf by the summed curvature of its rows rather than by their
  count. It reports a raw decision score, probabilities in `classes()` order,
  weighted fitting whose integer weights equal repeated rows, and schema-bound
  artifacts under an artifact kind of its own whose objective field names the
  loss the leaves were fitted to descend. Fitting requires both class labels.
- `api::AnyClassifier::HistGradientBoosting`, so a boosted classifier can be
  selected at runtime and persisted through the dispatch envelope like every
  other variant. The enum's declared capabilities are still the intersection
  over its variants, so adding one that offers no multiclass fit does not
  quietly widen what the enum promises.
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
- `calibration::CalibratedClassifier`, a fitted classifier composed with a
  fitted calibrator. It is an ordinary `Classifier`, so it scores and
  cross-validates through the existing paths unchanged. It calibrates the
  wrapped model's positive-class probability, takes its calibration rows as an
  explicit parameter rather than reusing training rows, and predicts labels from
  its own calibrated probabilities. A Platt-calibrated composition additionally
  exposes `decision_function`.
- `api::Capabilities::decision_function`, declaring whether a fitted classifier
  exposes a raw, unsquashed decision score. Producing probabilities is required
  of every `Classifier` and is not what this records.
- `RandomForestRegressor::fit_weighted`, `RandomForestClassifier::fit_weighted`,
  and `RandomForestClassifier::fit_multiclass_weighted`, taking validated
  per-row `data::SampleWeights`. A weight scales the row's contribution to every
  impurity, leaf statistic, and leaf distribution, and composes with the
  bootstrap replication count. Weights of exactly one reproduce the unweighted
  fit bit for bit, an integer weight is the same fit as repeating the row that
  many times, and a weight of zero removes the row — including from the
  bootstrap resample, which draws only among positively weighted rows. Both
  forests now declare `sample_weights` in their capability descriptor. The
  minimum split and leaf sizes bound **weight** rather than rows, which is a
  deliberate divergence from the reference contract taken so the integer-weight
  equivalence holds unconditionally; unweighted fitting is unaffected.
- `HistGradientBoostingRegressor::fit_weighted`. A weight scales the row's
  gradient and its share of every node's weight total, so the baseline is a
  weighted mean and the minimum leaf size counts weight rather than rows.
  Weights of exactly one reproduce the unweighted fit bit for bit, and an
  integer weight is the same fit as repeating the row. The bin grid stays
  unweighted: it is fitted from the distinct observed feature values, which
  neither a weight nor a repeated row changes.
- Artifact persistence for a joint multinomial `LogisticRegression` fit, under
  a second payload schema of the existing estimator kind. It stores the
  observed class list, one intercept per class, and one coefficient row per
  class; decoding selects the reader from the recorded payload version, so a
  binary and a multiclass artifact never decode as each other. Binary artifacts
  keep their exact bytes. `LogisticRegression` now persists every fit it
  offers, so its declared `artifact` capability no longer depends on which
  entry point was used, and the conformance battery asserts that a classifier
  declaring both persistence and multiclass fitting round trips a multiclass
  fit as well.
- Schema-bound `RandomForestClassifier` artifacts covering both fitted leaf
  representations under one artifact kind. The payload records which leaf
  arithmetic it holds and the reader refuses to build the other; a binary fit
  reuses the scalar logical-tree records unchanged, and a multiclass fit writes
  the same topology with a reserved zero in the scalar slot plus one per-tree
  leaf-distribution block ordered by pre-order leaf rank, so the encoding stays
  a unique name for the model. The classifier now declares `artifact`.
- `AnyClassifier` dispatch artifacts that record the fitted runtime variant and
  nest the estimator's own complete, independently validated artifact, mirroring
  `AnyRegressor`. A variant that carries more than one payload schema of its own
  keeps choosing between them itself, so restoring a dispatch artifact restores
  the variant *and* the fit it held. `AnyClassifier` now declares `artifact` by
  composition rather than declaring it away.
- `linear_model::LogisticSolver` and `LogisticRegressionParams::with_solver`,
  selecting the update rule a logistic fit uses. The default is and stays
  `Newton`, the exact second-order path every existing fitted model was
  produced by; `Lbfgs` is a matrix-free limited-memory quasi-Newton path whose
  storage is linear rather than quadratic in the parameter count. Both minimize
  the same penalized objective, so they agree on its minimizer, but `tol` means
  the largest coefficient update under `Newton` and the mean objective's
  gradient norm under `Lbfgs`. Neither payload schema records a solver, so a
  model fitted under a non-default one reports
  `ArtifactError::UnsupportedModelState` rather than writing bytes that would
  decode as a model claiming `Newton` provenance.
- Joint multinomial logistic fits above the exact solver's parameter ceiling,
  through `LogisticSolver::Lbfgs`. The ceiling is a property of the selected
  solver's storage rather than of the model, so the exact path keeps refusing
  above 2048 stacked parameters and keeps producing the identical fit below it,
  while the matrix-free path accepts 131 072 within the same storage budget.
  `ModelError::MulticlassSystemTooLarge` now reports whichever limit applied.
- `linear_model::Lasso`, a dense L1-regularized regressor fitted by cyclic
  coordinate descent. Coefficients it removes are exactly `0.0` and positively
  signed, so `coefficients` reads as a feature selection. Its objective divides
  the weighted squared error by twice the total sample weight, matching the
  reference contract's documented parametrization — so its `alpha` is a
  different quantity from `Ridge`'s, and the penalty applies to raw-scale
  coefficients because fitting centers but does not rescale the design.
  Sample weights are fractional row counts, and only their ratios matter. It
  declares weighted fitting and, deliberately, no artifact.
- `linear_model::ElasticNet`, the same coordinate-descent solver under a mixed
  L1 and L2 penalty, parametrized by `alpha` and `l1_ratio` exactly as the
  reference contract documents. `l1_ratio = 1` reproduces `Lasso` bit for bit
  at the same `alpha`; `l1_ratio = 0` is the ridge objective, but at
  `Ridge`'s `alpha * total_weight` rather than at the same number. The L2 term
  restores strict convexity, which both spreads weight across correlated
  features and makes designs converge that a pure L1 penalty does not.
- `ModelError::InvalidPenaltyAlpha` and `ModelError::InvalidL1Ratio`, reported
  at the public boundary before any allocation or fitting work.
- `ModelError::SolverDidNotConverge`, reported when an iterative solver
  exhausts `max_iter` — or is asked for a tolerance below what the objective's
  own numerical resolution can certify — instead of returning the last iterate
  as though it were a fitted model.

### Changed

- A `predict_class_proba_into` call that is invalid in *both* its batch width
  and its requested class now reports `ModelError::FeatureDimension` on every
  classifier. `tree::DecisionTreeClassifier`, the forests,
  `linear_model::LogisticRegression`,
  `ensemble::HistGradientBoostingClassifier` and
  `calibration::CalibratedClassifier` previously reported
  `ModelError::UnknownClass` for that call while their own allocating
  `predict_class_proba` — and `dummy::DummyClassifier` — reported the width.
  **A caller matching on the error of a doubly-invalid call gets a different
  variant than before**; a call that is invalid in only one way is unaffected,
  and no valid call changes at all. The rule is now stated once and uniformly:
  validation checks the shape of the input before the content of the request,
  because the width must hold before the matrix can be indexed at all. The
  divergence appeared when the batch-width check was hoisted into the
  allocating trait defaults without the caller-owned primitives underneath
  being aligned, and it survived because no test in the suite made a call that
  was invalid twice. The conformance battery now carries a
  `width_precedes_class` obligation driven by exactly that call, proven to fire
  by a probe that swaps the two checks.
- **Breaking.** `MaxFeatures` has one public path again, and it is
  `tree::MaxFeatures`. The `ensemble::MaxFeatures` re-export is removed; callers
  importing it — including callers of a *forest's* `with_max_features`, which
  takes this type — change the import path and nothing else, because it was
  always the same type. Two paths to one type left rustdoc free to pick the
  canonical one, and it picked `ensemble`, so
  `tree::DecisionTreeClassifierParams::max_features` rendered as returning
  `ferricml::ensemble::MaxFeatures` — a standalone tree's own parameter
  documented as an ensemble type, reading as though `tree` depended on
  `ensemble` when the `tree-below-estimators` layout rule enforces the
  reverse. The type is defined beside the grower that consumes it and now
  publishes from there alone.
- `api::AnyClassifier` and `api::AnyRegressor` now document their variant lists
  as a deliberate, curated set and say what decides membership. The API document
  claimed the regressor variants "cover forests", which was never true of
  `ensemble::ExtraTreesRegressor` and had drifted further as estimators shipped;
  the enums cover 3 of 6 classifiers and 4 of 10 regressors. No variant was
  added, because a dispatch enum declares the *intersection* of its variants'
  capabilities: `AnyRegressor` declares persistence only because all four of its
  variants persist, so admitting a variant that declares nothing — `DummyRegressor`
  and `IsotonicRegression` both do — would silently withdraw that declaration
  from every existing caller. An enum tracking every estimator would end up
  declaring nothing at all. Both enums stay `#[non_exhaustive]`, so a variant can
  still be admitted later without touching any existing estimator's bytes.
- **Breaking.** `calibration::IsotonicRegression` is now fitted like every other
  FerricML estimator. `calibration::IsotonicRegressionParams` is a new empty
  parameter type — the same shape `dummy::DummyClassifierParams` and
  `preprocessing::MaxAbsScalerParams` already ship — and it is a required final
  argument of `IsotonicRegression::fit`, `IsotonicRegression::fit_calibration`
  and `calibration::CalibratedClassifier::fit_isotonic`. The estimator also
  implements `api::HasParams` and exposes inherent `get_params`,
  `n_features_in`, `predict` and `predict_into`; the last two were previously
  reachable only by importing `api::Regressor`, which inverted the crate's
  preference for the caller-owned form being the easiest to reach. It was the
  only concrete leaf estimator in the crate missing any of these. Nothing about
  a fit changes: the parameter type carries no state, and the inherent
  prediction methods forward to the same trait implementation. Adding an
  out-of-range policy or a decreasing direction later is now an additive change
  rather than a `fit` signature break, which is what the empty-params
  convention exists to buy.
- **Breaking.** `inspection::permutation_importance_classifier` and
  `inspection::permutation_importance_classifier_into` are generic over the
  target vocabulary through the same sealed `data::ClassificationTargets`
  trait, instead of taking `data::BinaryTargets` alone. A natively multiclass
  classifier is now inspected through the crate's only classifier
  permutation-importance entry point, with the orientation, workspace reuse and
  caller-owned output the binary path already had. Nothing becomes more
  permissive: a binary positive-probability metric asked for over a wider class
  set is still `ScoringError::UnsupportedClasses`. Existing binary calls
  compile unchanged unless they name the type parameter explicitly, which now
  takes the target type first and the score second.
- **Breaking.** `predict_positive_proba` is now the allocating **batch**
  method on every classifier that carries a positive class, and the
  single-row form it used to name is `predict_positive_proba_one`. Callers
  must rewrite `model.predict_positive_proba(row)` as
  `model.predict_positive_proba_one(row)`; the argument type changes from
  `&[f32]` to `&data::MatrixView` and the return from `f32` to `Vec<f32>`, so
  a missed call site is a compile error rather than a silent reinterpretation.
  Affects `ensemble::RandomForestClassifier`,
  `ensemble::ExtraTreesClassifier`,
  `ensemble::HistGradientBoostingClassifier`,
  `tree::DecisionTreeClassifier` and `linear_model::LogisticRegression`.
  The old pairing was the crate's only shape mismatch between an allocating
  method and its `_into` partner: `predict_positive_proba` took one row while
  `predict_positive_proba_into` took a matrix, which left the caller-owned
  batch form with no allocating partner and put a single-row method under the
  name the batch form owns. Renaming it also gives the batch form the
  allocating partner it never had, on all five classifiers rather than the two
  that happened to expose `_into`. Nothing about the fitted models changed and
  no artifact byte moved.

- **Added, and breaking.** `ranking::PairwiseLinearRanker::pair_margins`
  returns raw margins for a slice of pairs, allocating the output;
  `pair_margins_into` was the only caller-owned method in the crate with no
  allocating partner at all. In the same family, `compare` is now the
  allocating **batch** comparison over a slice of pairs and the single-pair
  form is `compare_one`, so callers must rewrite
  `ranker.compare(&items, pair)` as `ranker.compare_one(&items, pair)`. The
  `compare` collision is the same defect as `predict_positive_proba` and was
  missed by the API audit's original sweep, which compared only each method's
  first argument — `&MatrixView` on both.

- **Breaking.** `model_selection::cross_validate_classifier` and
  `model_selection::grid_search_classifier` are generic over the target
  vocabulary, through the new sealed
  `data::ClassificationTargets` trait, instead of taking
  `data::BinaryTargets` alone. `data::ClassTargets` now folds through exactly
  the same entry point, so a natively multiclass estimator can be
  cross-validated and tuned with the `CrossValidationError` fold attribution,
  the `ScoringWorkspace` reuse, and the split and class-layout guards a
  hand-rolled fold loop gives up. The loop branches on
  classifier-versus-regressor and on nothing else: label arity is a property of
  the metric — `ClassificationScorer::MulticlassLogLoss` and `MulticlassBrier`
  already read a whole probability matrix over any observed class set — so
  there is no multiclass entry point to add. Nothing becomes more permissive: a
  binary positive-probability metric asked for on a wider class set is still
  `CrossValidationError::UnsupportedClasses`. Existing binary calls compile
  unchanged; the trait is sealed because `select` must preserve the
  container's construction-time guarantees, so a new target shape arrives as a
  new `data` container with its implementation.

- **Breaking.** `model_selection::cross_validate_classifier` and
  `model_selection::grid_search_classifier` take a final `view` argument, and
  `model_selection::cross_validate_classifier_labels` and
  `model_selection::grid_search_classifier_labels` are removed. `view` says how
  each fold's fitted model presents itself to the scoring layer, exactly as the
  scoring and permutation-importance entry points already asked:
  `|model| ScorableClassifier::probabilistic(model)` for a model that produces
  probabilities, `|model| ScorableClassifier::labels_only(model)` for one that
  does not. `model_selection` was answering "does this classifier give
  probabilities?" two ways — a `ScorableClassifier` value in one half, a
  duplicated function pair in the other — and now answers it one way. The
  constructor is passed rather than a `ScorableClassifier` value because the
  fitting closure returns an owned model per fold and the view borrows it, so
  the borrow has to be taken inside the fold loop. Neither entry point is
  bounded on `api::ProbabilisticClassifier` any more; the view carries that
  requirement, so a probability metric under a labels-only view is
  `CrossValidationError::UnsupportedOutput` at run time rather than a compile
  error the caller cannot work around.

- **Breaking.** Persistence is now a trait. `to_artifact` and `from_artifact`
  moved from inherent methods onto `artifact::ModelArtifact` (estimators, one
  feature schema) and `artifact::StageArtifact` (transformers and compositions,
  an input and an output schema), so calling either needs the trait in scope —
  `use ferricml::artifact::ModelArtifact;` — exactly as calling `predict` needs
  `api::Estimator`. No artifact's bytes change.

  This closes a gap rather than moving a name. Persisting used to require two
  independent declarations: writing the encoder, and separately listing the
  type as composable. Seven estimators had the first and not the second, so
  `ensemble::RandomForestClassifier`, both extra-trees models,
  `ensemble::HistGradientBoostingClassifier`, both standalone trees and
  `ranking::PairwiseLinearRanker` could be saved on their own but not inside a
  `pipeline::StagedPipeline`. All seven now compose, as do `api::AnyRegressor`
  and `api::AnyClassifier`. The traits are no longer re-exported from
  `ferricml::pipeline`; `ferricml::artifact` is the one path.
- A `pipeline::StagedPipeline` now declares capabilities whatever it holds,
  computing `artifact` from its parts instead of requiring every part to
  persist before it can declare anything. A composition that does not persist
  previously had no capability declaration at all, which also kept it out of
  the conformance battery. `pipeline::TransformerStack` gains
  `STAGES_PERSIST`, and its tuple implementations now require each stage to
  declare capabilities.
- Histogram-boosting fits report four distinct failures where they previously
  reported two. `api::ModelError::NumericalOverflow` used to stand for both a
  non-finite residual and a residual-length mismatch, and
  `api::ModelError::TreeTooLarge` for both an oversized tree and a structurally
  invalid one. A residual-length mismatch is now
  `api::ModelError::OutputLength` — it is a shape bug, not an overflow — and a
  structurally invalid tree is the new `api::ModelError::InvalidTreeStructure`.
  A caller matching on `NumericalOverflow` or `TreeTooLarge` from a
  `HistGradientBoosting*` fit may now see the other variant instead. The
  errors do not carry the residual's index: no public FerricML error names a
  row or an observation, and this is not the place to start.

- The allocating defaults on `api::Classifier`, `api::ProbabilisticClassifier`,
  `api::Regressor`, and `api::Transformer` — `predict`, `predict_proba`,
  `predict_class_proba`, and `transform` — now check the batch width against
  `Estimator::n_features_in` *before* sizing their output buffer, rather than
  allocating it and discovering the mismatch inside the `_into` primitive they
  delegate to. The error is unchanged in kind and in values; what changes is
  that a rejected call now allocates nothing at all. An implementor whose
  `_into` method accepted a width other than its declared `n_features_in`
  would see the default reject that call.

- `RandomForestClassifier::predict` and `ExtraTreesClassifier::predict` check
  the batch width on all three of their fitted-shape branches before
  allocating, rather than on the single-class branch only. The error a
  wrong-width batch receives is the same on every branch and is unchanged;
  what changes is that the binary and multiclass branches no longer allocate
  their output first.

- `ranking::PairwiseLinearRanker::fit` checks every pair index and the total
  pair weight before it copies and sorts the observation batch, rather than
  after. A batch that will be refused no longer pays for a full copy and an
  `O(n log n)` sort first. The errors and the fitted model are unchanged.

- Six further entry points found by sweeping for the same shape check the batch
  width before allocating rather than after: `LogisticRegression`,
  `HistGradientBoostingClassifier` and `CalibratedClassifier`'s
  `decision_function`, `StagedPipeline::transform`,
  `PairwiseLinearRanker::score_items`, and `CalibratedClassifier`'s
  `Classifier::predict_into` — which is an `_into` method, so the scratch
  buffer it no longer allocates for a refused batch was its only allocation.
  Every error is unchanged in kind and in values.

- **Breaking.** Producing probabilities is no longer required of every
  classifier. `predict_proba`, `predict_proba_into`, `predict_class_proba`, and
  `predict_class_proba_into` move off `api::Classifier` onto a new
  dyn-compatible sub-trait, `api::ProbabilisticClassifier`. **Callers that
  invoke any of those four through a generic bound or a trait object must
  require `ProbabilisticClassifier` instead of `Classifier`**; concrete calls
  on a shipped estimator are unaffected, since every classifier FerricML ships
  today implements the sub-trait. Trait upcasting means a
  `&dyn ProbabilisticClassifier` is still accepted wherever a `&dyn Classifier`
  is wanted.

  The split exists because margin-based classifiers — ridge classification,
  discriminant analysis, discrete boosting — have a natural output that is a
  score rather than a distribution. A required probability method would have
  forced each of them either to fabricate a number it never earned or to fail
  at run time on a method the type system promised. A caller that needs
  probabilities now says so in its bounds and gets a compile error rather than
  a surprise.

  Consequently `score_classifier`, `score_classifier_with`,
  `score_multiclass_classifier`, `score_multiclass_classifier_with`,
  `permutation_importance_classifier`,
  `permutation_importance_classifier_into`, and the classifier
  cross-validation and search entry points now take a probability-producing
  classifier, and `CalibratedClassifier` requires one — a calibrator maps a
  probability, so there is nothing to calibrate without one. The classifier
  scoring and permutation-importance entry points take a
  `ScorableClassifier` view rather than a bare reference, so a label-only
  classifier remains scorable on a label metric.

- **Breaking.** `AnyClassifier` no longer exposes `predict_proba`,
  `predict_proba_into`, `predict_class_proba`, or `predict_class_proba_into`
  directly, and deliberately does **not** implement
  `api::ProbabilisticClassifier`. Reach probabilities through
  `AnyClassifier::as_probabilistic`, which returns
  `Option<&dyn ProbabilisticClassifier>`. Runtime dispatch is the one place the
  concrete type is erased by construction, so it is the one place the question
  can only be asked rather than proven in the bounds — and the fallible
  accessor is what lets a future margin-based variant be added without
  breaking this surface a second time.


- Permutation importance takes any score implementing the new scorer traits and
  runs through the shared allocation-free scoring path, so it no longer carries
  its own copy of the scorer dispatch, the singleton-class probability
  handling, or the per-metric orientation table. Its proven allocation bound is
  unchanged.
- `score_classifier`, `score_regressor`, `cross_validate_classifier`, and
  `cross_validate_regressor` are generic over the new scorer traits instead of
  taking the built-in enums. Calls that pass a built-in scorer are unaffected;
  a turbofished `cross_validate_*` call gains one inferred type argument.

- **Breaking.** `api::ModelError` no longer has `EmptyTargets`,
  `InvalidBinaryTarget` or `NonFiniteTarget`. No public entry point could
  produce them: every estimator that checked for those conditions was handed a
  `data::BinaryTargets` or `data::RegressionTargets`, and those containers have
  no unchecked constructor — `new` refuses each case as a `data::DataError`,
  `select` preserves what `new` established and refuses an empty selection, and
  `From<BinaryTargets> for ClassTargets` widens without weakening. A caller
  matching on one of the three was matching on a state the type system already
  ruled out; the corresponding `DataError` variants are where the condition is
  actually reported, and they are unchanged. `ModelError` documents the absence
  so the variants are not reintroduced.
  `EmptyData` and `NonFiniteFeature` deliberately remain: `predict_one` and
  calibration take a bare `&[f32]`, which nothing has validated.
- Tree and forest fitting no longer rescans the training matrix for non-finite
  features. Every value in a `data::MatrixView` is finite by construction, so
  the scan was re-deriving the container's own invariant at O(rows × columns)
  on every fit. No performance claim is attached to this: it has not been
  measured.

### Fixed

- `calibration::PlattCalibrator::fit` no longer returns an unconverged Newton
  iterate as a fitted calibrator. The loop `break`s when the largest parameter
  update falls to `tol` and otherwise fell through to `Ok`, with only `n_iter`
  to say the tolerance was never met. It is **not** the last solver in the
  crate that does this — `LogisticRegression`'s Newton path does the same on
  both target shapes, and is tracked separately; the coordinate-descent and
  L-BFGS seams do report `ModelError::SolverDidNotConverge`. **This is a breaking behaviour change on
  degenerate calibration samples: a call that previously returned a model can
  now return an error.** No fitted value moves — a fit that is returned has
  exactly the parameters it had before, bit for bit.
  <br>
  Exhaustion by itself is *not* the new test, and making it the test would have
  been a worse defect than the one it replaced. `tol` bounds a parameter update
  in parameter units, and those units have no fixed scale. A calibration sample
  whose scores are nearly equal identifies its slope only through their spread,
  so a spread of `1e-6` puts the maximum-likelihood slope near `1e6`; the two-
  parameter Newton determinant is then a difference of nearly equal products
  that keeps a median of two significant digits, and the computed step inherits
  a rounding floor of roughly the parameter magnitude times that lost precision.
  Measured on the reported sample the floor is `1.9e-5` and does not move after
  100,000 iterations, while the gradient at that point is `4.6e-12` and the
  objective is at its minimum to the last bit. Over 7,725 sampled fits from that
  region, *every one* of the exhausted iterates was at the minimum — worst
  objective gap `5.1e-9` — so refusing on exhaustion alone would have converted
  the whole region into spurious errors.
  <br>
  The acceptance test at exhaustion is therefore a quantity that does not change
  with the parameter scale: the Newton decrement, the last step's inner product
  with the gradient it was computed from, which is twice the objective's own
  estimate of the distance above the minimum. It accepts all 7,725; a
  scale-relative step test accepts 7,627 and, applied as the loop's stopping
  rule instead, moves 24 of 166,925 fits that converge today, which is why the
  loop's own rule is untouched. `PlattCalibrator::n_iter` may therefore equal
  `max_iter` on a returned fit, and when it does the fit is at the minimum
  rather than merely the last thing tried.
  <br>
  Two tests pin the rule from both sides over a generated near-constant-score
  region rather than one fixture: refusing on plain exhaustion fails the one
  that watches the region fit, and an acceptance that never refuses fails the
  one that starves the same region of iterations.
- The artifact fuzz sweep's reach floors now detect their own mutator dying.
  They previously could not: with the mutator completely disabled three of the
  four floors still passed, and with nine of ten mutation strategies disabled
  all four did. One floor was anti-correlated with mutator health — an
  unmutated artifact is a valid artifact, so acceptances *rose* from 137 to
  2160 as the mutator died, and a floor on that number rewarded the failure it
  was supposed to catch. The floors now measure the mutator directly, requiring
  each strategy to change the bytes it is given and to produce distinct
  outputs, and measure depth by modelling the version-2 envelope instead of by
  classifying outcomes at the top of the stack, so "these bytes reached a
  payload parser" is decided rather than inferred. Per-decoder reach is floored
  too, which measures the coverage claim the sweep rests on for the first time.
  `the_reach_floors_fail_when_any_one_mutation_strategy_dies` kills each of the
  ten strategies in turn and requires a floor naming it to fail, and
  `the_envelope_model_agrees_with_the_real_decoder` holds the depth model to
  the decoder it models over every envelope field. The comment claiming the old
  counter proved payload-parser reach was false — all three error variants it
  counted are raised by the envelope itself — and is gone.
- The documented ranking guarantee on `calibration` is corrected: it claimed
  that "calibration is monotone, so the **ranking** of any two rows is preserved
  exactly — a threshold-based score such as ROC AUC is unchanged by
  calibration", and that is false. Monotone is weaker than ranking-preserving.
  A `PlattCalibrator` whose fitted slope is negative is a strictly *decreasing*
  map and takes ROC AUC to `1.0 - auc`; `IsotonicRegression` pools, so distinct
  scores can tie, and a fold whose labels run opposite to its scores collapses
  the map to a constant and ROC AUC to `0.5`. Both are reachable through the
  public API, on the held-out calibration fold the same documentation
  *requires*. `Calibrator`, `CalibratedClassifier`, `PlattCalibrator::slope`,
  `IsotonicRegression` and the calibration guide now state the three cases and
  the condition — a strictly increasing map — under which ROC AUC is unchanged.
  The doctest that asserted the false general claim asserted it on the benign
  training-rows case; it now calibrates on held-out rows, asserts the positive
  slope its conclusion rests on, and scores against labels the model does not
  reproduce exactly, so the AUC equality could fail if calibration reordered.
  **Behaviour is unchanged and no fit is rejected.** A negative slope is the
  exact maximum-likelihood answer for its sample: it carries the sign of that
  sample's class mean gap, which `PlattCalibrator::slope` now documents as the
  ranking contract, and the fitted parameters are public so a caller who
  depends on ranking can check them.
- Every one of the crate's 29 capability declarations now carries a doc comment
  saying what it claims and, where a capability is deliberately absent, why —
  up from 8. Four of them contradicted the declaration they sat above and are
  corrected: `CalibratedClassifier<_, IsotonicRegression>` read "Nothing" over a
  declaration that produces probabilities, and `MaxAbsScaler`, `MinMaxScaler`
  and `RobustScaler` each explained an absent capability without ever naming the
  persistence they declare. `scripts/check_documentation_truth.py` gained a rule
  requiring the doc comment, and now reads it from either position rustdoc
  renders — above the `impl` or above the `const CAPABILITIES` — which is where
  those four had been hiding.
- `Pipeline<StandardScaler, LogisticRegression>` now declares the
  `decision_function` capability it has. It exposes `decision_function_into`,
  but computed its declaration by intersecting both parts — and a transformer
  never has a decision function, so the intersection made that field
  structurally unable to be true for any pipeline. A raw decision score is a
  property of the final estimator alone, so it is now taken from there, as
  weighted fitting and multiclass fitting already were. A caller querying the
  capability to decide whether to threshold on raw scores was previously told
  no for a composition that could.
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
- `api::Transformer::transform` builds its `DenseMatrix` from the validated
  view the implementation returned rather than from the buffer it was lent.
  `Transformer` is public and unsealed, so the buffer's contents were whatever
  an arbitrary implementation put there: safe external code could write `NaN`
  into it, return a validated view over unrelated storage, and obtain a
  `DenseMatrix` — the crate's validated container — holding non-finite values.
  That matrix was then accepted anywhere a fitted model takes features. The
  trait already documented the returned view as covering "exactly the values
  they wrote"; the default body now relies on that view instead of restating
  the claim over the raw buffer. `StagedPipeline::transform` already worked
  this way, and the two allocating pipeline entry points now agree.

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
