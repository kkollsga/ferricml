# API and growth structure

FerricML follows scikit-learn's estimator vocabulary where it maps cleanly to
safe, typed Rust: fitted estimators expose `n_features_in`, exact retained
parameters through `get_params`, and the operations `fit`, `predict`,
`predict_proba`, and `transform`. Allocating convenience methods delegate to
caller-owned `_into` primitives. Classification probabilities are row-major,
with one column per sorted entry in `classes`.

The supported compatibility subset and deliberate differences are frozen in
[`sklearn-conformance.md`](sklearn-conformance.md). FerricML does not imitate
Python object mutation, NumPy coercion, pickle, or magic parameter strings.
Typed parameter builders and validated dense `f32` containers are intentional
Rust interfaces, while their names and observable estimator meaning follow the
reference contract.

## Extension structure

- `api` owns backend-independent estimator categories, errors, retained
  parameter access, and batch-level runtime model enums.
- `data` owns validated row-major inputs, targets, and sample weights.
- `ensemble` owns public forest estimators and parameter types; private tree
  training and packed representation stay outside the public module tree. The
  histogram-boosting facade validates and orchestrates while private `boosting`
  modules own binning, mutable growth, and compact prediction separately.
- `pipeline` composes a fitted `Transformer` and estimator generically. Its
  `with_transformed` path uses caller-owned workspace and static dispatch.
  Concrete standard-scaler pipelines provide allocation-free prediction and
  explicit persistence for logistic, linear, and ridge estimators.
- `linear_model` separates estimator facades from private numerical seams.
- `preprocessing` owns fitted transformer implementations and their state;
  `StandardScaler` uses deterministic two-pass population statistics.
  Training-time pipeline composition and more than one transform step remain
  deliberately deferred until those use cases are real.
- `ranking` owns pair construction, the pairwise linear estimator, and
  denominator-safe rank metrics. It remains distinct from `Classifier`: raw
  ranking scores and pair margins are not probabilities.
- `artifact` owns stable envelope identity, bounded decoding primitives, and
  artifact errors. It does not expose backend tree layouts.

`AnyClassifier` and `AnyRegressor` remain the owned runtime-swap layer. They
match once per batch; the regressor variants cover forests, linear regression,
ridge, and histogram gradient boosting. Generic estimators and pipelines
remain the primary zero-overhead layer.
