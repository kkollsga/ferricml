# Install and quickstart

## Install

FerricML is one crate with no default features:

```toml
[dependencies]
ferricml = "0.1"
```

Or from the command line:

```console
cargo add ferricml
```

There is nothing else to configure. `default = []` is a product boundary rather
than an accident: enabling no feature is the supported configuration, and the
crate's own dependencies are a numerical backend and a hash function.

## Fit a regressor

Everything in FerricML follows the same four steps: validate the data, build a
parameter value, fit, predict.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};
use ferricml::metrics::r2_score;

// Six rows, one feature. Values are row-major.
let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0])?;

let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())?;

let predictions = model.predict(&data.as_view())?;
assert_eq!(predictions.len(), 6);

let quality = r2_score(targets.as_slice(), &predictions)?;
assert!(quality > 0.99);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Four things in that snippet are worth naming, because they are the same
everywhere in the crate.

**Data is validated before it is a value.** `DenseMatrix::new` checks the shape,
the exact buffer length and the finiteness of every element. By the time
`Ridge::fit` is called, there is nothing left for it to re-check per row, and a
malformed input was refused before any allocation or training work happened.

**Parameters are a typed value, not a bag of strings.** `RidgeParams::default()`
is a real type with `with_*` builder methods. A misspelled parameter does not
compile, and there is no string to get wrong.

**Fitting borrows.** `data.as_view()` produces a `MatrixView`, which is `Copy`
and allocation-free, so one owned matrix serves fitting, prediction and scoring.

**Everything returns `Result`.** Fitting, predicting and scoring each have their
own error type — `ModelError`, `DataError`, `MetricError` — and each names what
was wrong. Nothing panics on bad input and nothing substitutes a value for a
result it could not compute.

## Fit a classifier

A classifier is the same shape, with `BinaryTargets` instead of
`RegressionTargets` and probabilities available beside labels.

```rust
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};
use ferricml::metrics::accuracy_score;

let data = DenseMatrix::new(vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 6, 1)?;
let labels = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;

let model = LogisticRegression::fit(
    &data.as_view(),
    &labels,
    LogisticRegressionParams::default(),
)?;

let predicted = model.predict(&data.as_view())?;
assert_eq!(accuracy_score(labels.as_slice(), &predicted)?, 1.0);

// Probabilities are row-major, one column per observed class.
let probabilities = model.predict_proba(&data.as_view())?;
assert_eq!(probabilities.len(), 6 * model.classes().len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Predict without allocating

Every allocating prediction method delegates to a caller-owned `_into`
primitive. In a loop, on a hot path, or anywhere the output buffer can be
reused, call the primitive directly and the prediction allocates nothing at all:

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;
let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())?;

// One buffer, reused for as many batches as you like.
let mut output = vec![0.0_f32; 4];
model.predict_into(&data.as_view(), &mut output)?;

assert_eq!(output, model.predict(&data.as_view())?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The allocating method is written in terms of the caller-owned one rather than
the other way round, so the fast path is the real path and the two cannot
disagree — as the last line asserts.

## Where to go next

- [Data and targets](data.md) — the validated containers every estimator takes,
  and why validation happens where it does.
- [Linear models](linear-models.md) — ordinary least squares, ridge, the
  penalized fits, and logistic regression.
- [Trees and forests](trees-and-forests.md) — one tree, many trees, and boosted
  trees.
- [Preprocessing and pipelines](preprocessing-and-pipelines.md) — scalers, and
  composing them with a model into one fitted object.
- [Calibration and inspection](calibration-and-inspection.md) — baselines,
  trustworthy probabilities, and which features a model actually uses.
- [Saving and loading models](persistence.md) — versioned, checksummed,
  schema-bound binary artifacts.
- [Evaluation and model selection](../evaluation-and-model-selection.md) —
  metrics, splitters, cross-validation and typed parameter search.

For the exact signature of anything named here, see
[docs.rs/ferricml](https://docs.rs/ferricml). This site does not reproduce API
listings; see [the docs.rs boundary](../index.md#this-site-and-docsrs).
