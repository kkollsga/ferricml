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
  estimator types and is not already guaranteed by the type system, so it never
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
  fold iterators, batch estimator scoring, and serial typed cross-validation.
  Splitters remain independent of estimator internals, while fitting stays in
  caller-provided closures.
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
- `inspection` owns model-agnostic attribution. Permutation importance works
  through the public batch prediction and scoring contracts only, so it needs
  no estimator cooperation and exposes no model internals. Its per-feature
  values are quality losses, oriented so a larger number always means a more
  important feature whichever direction the underlying metric improves in.

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
