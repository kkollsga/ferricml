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
- `data` owns validated row-major inputs and targets.
- `ensemble` owns public forest estimators and parameter types; private tree
  training and packed representation stay outside the public module tree.
- `pipeline` composes a fitted `Transformer` and estimator generically. Its
  `with_transformed` path uses caller-owned workspace and static dispatch, so a
  future `StandardScaler` can feed logistic or ridge estimators without a
  virtual call or intermediate allocation per batch.
- Future `preprocessing` and `linear_model` modules land only with their first
  concrete estimators. Training-time pipeline composition and more than one
  transform step are deliberately deferred until those use cases are real.
- A future `artifact` module will own a backend-neutral envelope. It will not
  expose backend tree layouts.

`AnyClassifier` and `AnyRegressor` remain the owned runtime-swap layer. They
match once per batch. Generic estimators and pipelines remain the primary
zero-overhead layer.
