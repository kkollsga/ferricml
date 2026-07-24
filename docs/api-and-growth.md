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
  parameter access, and batch-level runtime model enums.
- `data` owns validated row-major inputs, targets, and sample weights.
- `ensemble` owns public ensemble estimators and parameter types; each private
  estimator family owns its validation, training, persistence conversion, and
  compact representation below the public facade.
- `pipeline` composes a fitted `Transformer` and estimator generically. Its
  `with_transformed` path uses caller-owned workspace and static dispatch.
  Concrete standard-scaler pipelines provide allocation-free prediction and
  explicit persistence for logistic, linear, and ridge estimators.
- `linear_model` separates estimator facades from private numerical seams.
- `metrics` owns deterministic classification and regression measures with
  explicit errors for invalid or undefined inputs.
- `model_selection` owns validated index partitions, deterministic holdout and
  fold iterators, batch estimator scoring, and serial typed cross-validation.
  Splitters remain independent of estimator internals, while fitting stays in
  caller-provided closures.
- `preprocessing` owns fitted transformer implementations and their state;
  `StandardScaler` uses deterministic two-pass population statistics.
  Training-time pipeline composition and more than one transform step remain
  deliberately deferred until those use cases are real.
- `ranking` owns pair construction, the pairwise linear estimator, and
  denominator-safe rank metrics. It remains distinct from `Classifier`: raw
  ranking scores and pair margins are not probabilities.
- `artifact` owns stable envelope identity, bounded decoding primitives, and
  artifact errors. It does not expose backend tree layouts.
- `inspection` owns model-agnostic attribution. Permutation importance works
  through the public batch prediction and scoring contracts only, so it needs
  no estimator cooperation and exposes no model internals. Its per-feature
  values are quality losses, oriented so a larger number always means a more
  important feature whichever direction the underlying metric improves in.

`AnyClassifier` and `AnyRegressor` remain the owned runtime-swap layer. They
match once per batch; the regressor variants cover forests, linear regression,
ridge, and histogram gradient boosting. Generic estimators and pipelines
remain the primary zero-overhead layer.
