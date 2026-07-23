# Scikit-learn conformance contract

FerricML uses scikit-learn 1.9.0 as a black-box semantic and quality reference.
The reference script imports only public estimator classes and reads only
public methods or fitted attributes. Frozen outputs are project-owned test
fixtures; no scikit-learn model, pickle, source, or private implementation
detail is embedded.

The reference environment is Python 3.12 with the exact package versions in
`requirements/sklearn-conformance.txt`. A reproducible verification run is:

```console
uv venv --python 3.12 dev-docs/temp/sklearn-1.9
uv pip install --python dev-docs/temp/sklearn-1.9/bin/python \
  -r requirements/sklearn-conformance.txt
dev-docs/temp/sklearn-1.9/bin/python scripts/sklearn_conformance.py
```

Verification is read-only. Exact public-API outputs must match byte for byte.
Randomized forest topology can differ slightly across supported operating
systems even with a fixed public `random_state`, so five-seed quality values
must stay within a narrow per-seed portability envelope: 0.01 accuracy, 0.002
Brier score, and 0.002 normalized RMSE. These limits are strictly smaller than
FerricML's model-quality allowances and do not relax the Rust accuracy,
calibration, or regression gates. `--update` is the only fixture-writing mode
and is reserved for an intentional, reviewed contract change.

## Locked common subset

The contract covers the scikit parameter names `n_estimators`, `max_depth`,
`min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`,
`random_state`, and `n_jobs`; sorted `classes_`; `n_features_in_`; label and
row-major probability outputs; one-column single-class probability shape;
regression output; logistic sample weights, raw decision scores, and exact-zero
no-intercept semantics; first-class tie selection; and common parameter/shape
validation. Exact forest fixtures use one tree, all features, and no bootstrap,
and must match within `1e-6`; iterative logistic outputs use a reviewed
`2e-5` tolerance.

Quality is an implementation-level comparison rather than tree identity. Five
fixed seeds cover nonlinear, separable, imbalanced, noisy classification and
regression. Each lane is evaluated by its five-seed mean. FerricML may trail by
at most 0.02 accuracy, add at most 0.02 Brier score, or add at most 5% normalized
RMSE.

## Intentional divergences

- FerricML accepts validated, finite dense `f32` data. Missing/categorical
  values, sparse matrices, and implicit numeric conversion are outside scope.
- Targets are binary `u8` or scalar `f32`; multiclass, multilabel, multi-output,
  and arbitrary class-label types are outside scope.
- Logistic regression accepts validated sample weights. Forest sample weights,
  class weights, OOB estimates, pruning, monotonic constraints, warm starts,
  and impurity-based inspection attributes are not implemented.
- Typed `MaxFeatures` and `NJobs` replace magic strings and integer sentinels.
  FerricML defaults to deterministic `random_state = 0` and serial execution;
  scikit defaults both settings to `None`. These are deliberate reproducibility
  defaults, while all other supported defaults have equivalent meaning.
- FerricML owns its RNG and does not promise randomized tree identity. Pickle
  compatibility and backend-native serialization are explicitly not contracts.
