# Evaluation and model selection

FerricML provides deterministic evaluation building blocks without owning data
loading, parameter search, or application-specific policy. The public surface
is split between `metrics`, which evaluates predictions, and
`model_selection`, which creates index partitions and evaluates fitted models.

## Metrics

Classification includes exact accuracy, a binary confusion matrix, precision,
recall, F1, Brier score, binary logarithmic loss, and tie-aware ROC AUC.
Regression includes mean absolute error, mean squared error, root mean squared
error, and R2. Calculations promote `f32` inputs to deterministic `f64`
accumulation.

Every metric rejects empty or mismatched inputs. Binary metrics reject labels
outside zero and one; probability metrics also reject non-finite values and
values outside `0..=1`. Precision, recall, F1, R2, and ROC AUC return
`MetricError::Undefined` when their required denominator or class distribution
does not exist. Binary logarithmic loss clips valid endpoint probabilities to
`1e-15..=1-1e-15` before taking logarithms.

## Deterministic splits

`train_test_split` and `stratified_train_test_split` return validated `Split`
values containing sorted, disjoint train and test indices with complete sample
coverage. `TestSize` accepts an exact count or a fraction rounded upward.
Seeded shuffling uses a stable crate-owned algorithm, so identical parameters
produce identical membership across supported platforms.

`KFold` and `StratifiedKFold` validate before returning lazy, exact-size fold
iterators. Fold sizes differ by at most one. Stratification accepts arbitrary
`u8` labels and requires enough members of every observed class to place that
class in every requested partition.

Use `DenseMatrix::select_rows`, `BinaryTargets::select`, and
`RegressionTargets::select` when manually applying a split:

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::model_selection::{HoldoutParams, TestSize, train_test_split};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?;
let split = train_test_split(
    data.rows(),
    HoldoutParams::default()
        .with_test_size(TestSize::Count(1))
        .with_random_state(7),
)?;
let train = data.select_rows(split.train_indices())?;
let train_targets = targets.select(split.train_indices())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Scoring and cross-validation

`score_classifier` and `score_regressor` map typed scorer enums to public
metrics and make one batch prediction call. Probability-based binary scorers
accept fitted class layouts `[0]`, `[1]`, and `[0, 1]`; other layouts produce
an explicit error.

Cross-validation consumes any iterator of validated `Split` values, fits one
typed model per fold through a caller closure, and returns scores in iterator
order. It is deliberately serial: a fixed split order, fit parameters, seed,
and thread count produce repeatable fitted artifacts and scores. Errors retain
the zero-based fold index for fitting, prediction, metric, or class-layout
failures.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};
use ferricml::model_selection::{
    KFold, RegressionScorer, cross_validate_regressor,
};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?;
let splits = KFold::new(2).split(data.rows())?;
let result = cross_validate_regressor(
    &data.as_view(),
    &targets,
    splits,
    RegressionScorer::RootMeanSquaredError,
    |train, train_targets| Ridge::fit(train, train_targets, RidgeParams::default()),
)?;
assert_eq!(result.len(), 2);
assert!(result.mean().is_finite());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Parameter grids, repeated validation, parallel fold scheduling, group-aware
splits, and nested model selection remain outside this initial contract. They
can be added without exposing fitted model internals or weakening deterministic
split semantics.
