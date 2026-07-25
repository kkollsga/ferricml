# API and growth structure

FerricML uses a stable estimator vocabulary expressed as safe, typed Rust:
fitted estimators expose `n_features_in`, exact retained
parameters through `get_params`, and the operations `fit`, `predict`,
`predict_proba`, and `transform`. Allocating convenience methods delegate to
caller-owned `_into` primitives. Classification probabilities are row-major,
with one column per sorted entry in `classes`.

The supported contract and deliberate differences are frozen in
[`reference-semantics.md`](reference-semantics.md). FerricML avoids dynamic
object mutation, implicit numeric coercion, backend-native persistence, and
magic parameter strings. Typed parameter builders and validated dense `f32`
containers are intentional Rust interfaces, while their names and observable
estimator meaning follow the reference contract.

## Extension structure

- `api` owns backend-independent estimator categories, errors, retained
  parameter access, batch-level runtime model enums, and the compile-time
  capability descriptor. `Capabilities` records only what varies between
  estimator types and is not already guaranteed by the type system — weighted
  fitting, artifact persistence, and multiclass fitting — so it never
  becomes a second parameter system; it is carried by `HasCapabilities`, a
  generic trait rather than an associated constant on the object-safe
  categories, which must stay dyn-compatible.
- `data` owns validated row-major inputs, targets, and sample weights.
- `ensemble` owns public ensemble estimators and parameter types; each private
  estimator family owns its validation, training, persistence conversion, and
  compact representation below the public facade.
- `pipeline` composes fitted transformers and an estimator generically. Its
  `with_transformed` path uses caller-owned workspace and static dispatch.
  `Pipeline` holds one transformer; concrete standard-scaler pipelines provide
  allocation-free prediction and explicit persistence for logistic, linear, and
  ridge estimators. `StagedPipeline` holds two or more stages as a
  `TransformerStack` tuple and can fit the whole composition in one pass, each
  stage on the previous stage's output. Every handoff is validated before the
  composition exists, and one caller-owned workspace is split into a disjoint
  segment per stage, so multi-stage inference allocates nothing. Prediction
  stays on the generic callback rather than per-category convenience methods,
  which cannot coexist as inherent methods of one name.
- `linear_model` separates estimator facades from private numerical seams.
- `metrics` owns deterministic classification and regression measures with
  explicit errors for invalid or undefined inputs.
- `model_selection` owns validated index partitions, deterministic holdout and
  fold iterators, batch estimator scoring, serial typed cross-validation, and
  typed parameter search. Splitters remain independent of estimator internals,
  while fitting stays in caller-provided closures. A search grid is built from
  the parameter type's own builder methods rather than string keys, and search
  evaluates candidates through cross-validation and the shared scorer contract
  rather than carrying an evaluation path of its own.
- `preprocessing` owns fitted transformer implementations and their state.
  `StandardScaler` uses deterministic two-pass population statistics and
  accepts sample weights; `MinMaxScaler` and `MaxAbsScaler` fit order
  statistics, which no per-sample weight can move, so they declare no weighted
  entry point. Each carries a degenerate column explicitly — a constant column
  scales by one and a zero-magnitude column passes through — rather than
  dividing by an empty range. The shared non-finite preflight is stated once
  for the family, so a finite input that scales to a non-finite `f32` is
  reported at its first row-major location before anything is written.
- `ranking` owns pair construction, the pairwise linear estimator, and
  denominator-safe rank metrics. It remains distinct from `Classifier`: raw
  ranking scores and pair margins are not probabilities.
- `artifact` owns stable envelope identity, bounded decoding primitives, and
  artifact errors. It does not expose backend tree layouts.
- `dummy` owns baseline estimators that ignore their features: the
  majority-class classifier, whose probabilities are the observed class
  frequencies, and the mean regressor. They are the quality floor a real
  estimator has to beat and the reference implementation of the estimator
  contract, so they carry no tunable behavior, no weighted entry point, and no
  artifact kind.
- `calibration` owns post-hoc probability calibration. A calibrator is a fitted
  monotone map of one model score onto a probability, so it changes how
  confident a prediction is without changing which way round two rows are
  ordered. `IsotonicRegression` is the non-parametric one and is also a
  standalone monotone regressor over a single-column matrix. It states its tie
  convention rather than inheriting one: observations sharing an input value are
  pooled into their mean *before* pool-adjacent-violators runs, which is forced
  rather than chosen — a function of one input takes one value at one input —
  and it makes the fit independent of observation order. Prediction interpolates
  linearly between fitted points and holds the end values outside the fitted
  range rather than extrapolating a trend the fit never observed.
  `PlattCalibrator` is the parametric one, a two-parameter logistic fit reached
  through the shared objective contract rather than a third logistic solver. It
  regresses on Platt's prior-corrected targets rather than on the raw labels,
  which is what keeps the fit finite when the score separates the classes
  perfectly — with raw labels that problem has no finite maximum-likelihood
  solution, and the resulting map would assert exactly the certainty
  calibration exists to remove.
- `inspection` owns model-agnostic attribution. Permutation importance works
  through the public batch prediction and scoring contracts only, so it needs
  no estimator cooperation and exposes no model internals. It holds no scoring
  logic of its own: it calls the same caller-owned-buffer scoring entry point
  cross-validation does, and takes the orientation of the result from the
  score's own declaration. Its per-feature
  values are quality losses, oriented so a larger number always means a more
  important feature whichever direction the underlying metric improves in.

Classification covers an arbitrary observed class set. `ClassTargets` carries
the sorted, deduplicated labels a fit observed, and that set is the probability
column order; nothing assumes the labels are contiguous or zero-based.
`LogisticRegression::fit_multiclass` is one joint multinomial optimization
whose probabilities are the softmax of a centred score vector with no pinned
reference class, and `RandomForestClassifier::fit_multiclass` averages per-tree
probability vectors rather than voting on per-tree labels. Both keep their
original binary fit unchanged beside the new one, including its asymmetric
single-row decision score, because the two parametrizations are different
models rather than two spellings of one. Probability rows are never
renormalized: they sum to one only within `n_classes` `f32` ulps, which is a
frozen part of the contract rather than a tolerance to tighten later.

`AnyClassifier` and `AnyRegressor` remain the owned runtime-swap layer. They
match once per batch; the regressor variants cover forests, linear regression,
ridge, and histogram gradient boosting. Generic estimators and pipelines
remain the primary zero-overhead layer.

Meta-layers compose capabilities rather than restating them. A dispatch enum
and a fitted pipeline declare the intersection of their variants' or parts'
capabilities, so an undispatched caller is never promised more than it gets,
while `capabilities` on a dispatch value reports the variant actually held.
Capabilities a composition cannot have at all — weighted fitting, when the
composition owns only already-fitted parts — are declared away structurally
instead of being inherited.

Artifact support composes through a bound rather than a list. A
`StagedPipeline` declares persistence exactly where every stage and its
estimator really have a schema-bound artifact, so one declaration covers every
such composition and asking one that cannot persist is a compile error. That is
possible because a staged composition uses a single artifact kind and records
which concrete parts it holds inside the payload: order, estimator type, and
stage count are all checked on decode, so one composition never decodes as
another. `Pipeline`'s three concrete compositions predate that scheme and keep
their own artifact kinds, so their declarations stay per composition.
