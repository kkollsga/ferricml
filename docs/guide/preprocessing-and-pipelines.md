# Preprocessing and pipelines

Some models care about the scale of your features and some do not. A tree splits
on thresholds, so multiplying a column by a thousand changes nothing. A
penalized linear model measures its penalty against the coefficients, so the
units a feature is recorded in decide how hard it gets penalized. Preprocessing
is how you take that decision back.

## The seven transformers

| Transformer | Fits | Does |
| --- | --- | --- |
| `StandardScaler` | Mean and population standard deviation | Centres and scales each column |
| `MinMaxScaler` | Column minimum and maximum | Maps each column onto `0..=1` |
| `MaxAbsScaler` | Largest magnitude | Divides each column, keeping zero at zero |
| `RobustScaler` | Median and interquartile range | Scales without letting outliers dominate |
| `Normalizer` | Nothing | Scales each **row** to unit norm |
| `Binarizer` | Nothing | Thresholds every value to `0.0` or `1.0` |
| `FunctionTransformer` | Nothing | Applies an elementwise `fn(f32) -> f32` |

All seven share the estimator shape: `fit` on a `MatrixView`, then `transform`
or the caller-owned `transform_into`.

## Standard scaling

The default choice in front of a linear model. Each column is centred on its
fitted mean and divided by its fitted population standard deviation:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};

// One column in metres, one in millimetres — the same measurement twice.
let data = DenseMatrix::new(
    vec![1.0, 1000.0, 2.0, 2000.0, 3.0, 3000.0, 4.0, 4000.0],
    4,
    2,
)?;

let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default())?;
let scaled = scaler.transform(&data.as_view())?;

// After scaling the two columns are identical: the units are gone.
for row in 0..4 {
    let left = scaled.get(row, 0).expect("in bounds");
    let right = scaled.get(row, 1).expect("in bounds");
    assert!((left - right).abs() < 1e-5);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

That is the whole reason to put a scaler in front of `Lasso` or `ElasticNet`. As
[linear models](linear-models.md) explains, the L1 penalty applies to raw-scale
coefficients, so scale-free feature selection means scaling first — explicitly,
where the transformation persists with the model rather than living in a script
someone forgets to re-run.

## One degeneracy rule, for every scaler

A column with a spread of **exactly** zero keeps a divisor of one. A constant
feature therefore survives as a constant rather than becoming a non-finite
value:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};

let data = DenseMatrix::new(vec![5.0, 5.0, 5.0, 5.0], 4, 1)?;
let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default())?;

let scaled = scaler.transform(&data.as_view())?;
assert!(scaled.as_slice().iter().all(|value| value.is_finite()));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The test is exact equality with zero, and it is the same test for the standard,
min-max, max-abs and robust scalers. A column whose spread is merely *small* is
real data: it is scaled normally, and if that overflows `f32` the batch is
refused with the offending row and column before anything is written.

FerricML deliberately does not use a magnitude threshold here. A threshold would
silently decline to scale a legitimately tiny-scaled column, and it would give
the crate two degeneracy rules where one will do.

## Range scaling, and invertibility

`MinMaxScaler` maps each column onto `0..=1`, and the map inverts:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{MinMaxScaler, MinMaxScalerParams};

let data = DenseMatrix::new(vec![10.0, 20.0, 30.0, 50.0], 4, 1)?;
let scaler = MinMaxScaler::fit(&data.as_view(), MinMaxScalerParams::default())?;

let scaled = scaler.transform(&data.as_view())?;
assert_eq!(scaled.as_slice()[0], 0.0);
assert_eq!(scaled.as_slice()[3], 1.0);

let restored = scaler.inverse_transform(&scaled.as_view())?;
assert!((restored.as_slice()[2] - 30.0).abs() < 1e-4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`MinMaxScaler` and `MaxAbsScaler` fit **order statistics** — a minimum, a
maximum, a largest magnitude — and no per-sample weight can move an order
statistic. That is why they declare no weighted entry point, rather than
offering one that would quietly do nothing. `StandardScaler` does accept sample
weights, because a weighted mean and variance are meaningful.

## Robust scaling

When a column has outliers, the mean and standard deviation chase them.
`RobustScaler` uses the median and the interquartile range instead:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{
    RobustScaler, RobustScalerParams, StandardScaler, StandardScalerParams,
};

// Eight ordinary values and one extreme outlier.
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1000.0];
let data = DenseMatrix::new(values, 9, 1)?;

let robust = RobustScaler::fit(&data.as_view(), RobustScalerParams::default())?;
let standard = StandardScaler::fit(&data.as_view(), StandardScalerParams::default())?;

// The outlier inflates the standard deviation, squashing every ordinary value
// toward zero. The interquartile range barely notices it.
let robust_scaled = robust.transform(&data.as_view())?;
let standard_scaled = standard.transform(&data.as_view())?;
assert!(robust_scaled.as_slice()[0].abs() > standard_scaled.as_slice()[0].abs());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Its quantiles use linear interpolation between the two bracketing order
statistics — Hyndman–Fan type 7 — applied uniformly, including at the median.
Small samples do not contain the value a percentile asks for, so the
interpolation rule is a documented semantic choice rather than an implementation
detail.

`RobustScaler` deliberately does **not** offer scaling the quantile spread to
that of a standard normal distribution: that needs an inverse-normal-CDF
primitive with its own accuracy contract, and one optional flag does not justify
one. FerricML records parameters it does not claim rather than leaving them as
gaps.

## The stateless three

`Normalizer` scales each **row**, not each column, so it is the odd one out:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{Norm, Normalizer, NormalizerParams};

// Two rows pointing the same way at different lengths.
let data = DenseMatrix::new(vec![3.0, 4.0, 30.0, 40.0], 2, 2)?;

let normalizer = Normalizer::fit(
    &data.as_view(),
    NormalizerParams::default().with_norm(Norm::L2),
)?;
let unit = normalizer.transform(&data.as_view())?;

// Both rows land on the same unit vector.
assert_eq!(unit.row(0), unit.row(1));
assert!((unit.as_slice()[0] - 0.6).abs() < 1e-6);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Binarizer` thresholds. The comparison is **strictly** greater-than, so a value
exactly at the threshold becomes `0.0` and the two output classes are
`(-inf, t]` and `(t, +inf)` rather than leaving the boundary to rounding:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{Binarizer, BinarizerParams};

let data = DenseMatrix::new(vec![-1.0, 0.0, 1.0, 2.0], 4, 1)?;
let binarizer = Binarizer::fit(
    &data.as_view(),
    BinarizerParams::default().with_threshold(1.0),
)?;

assert_eq!(
    binarizer.transform(&data.as_view())?.as_slice(),
    &[0.0, 0.0, 0.0, 1.0],
);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`FunctionTransformer` applies an elementwise `fn(f32) -> f32`, with an optional
inverse:

```rust
use ferricml::data::DenseMatrix;
use ferricml::preprocessing::{FunctionTransformer, FunctionTransformerParams};

fn double(value: f32) -> f32 {
    value * 2.0
}
fn halve(value: f32) -> f32 {
    value / 2.0
}

let data = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
let transformer = FunctionTransformer::fit(
    &data.as_view(),
    FunctionTransformerParams::default()
        .with_func(double)
        .with_inverse_func(halve),
)?;

let doubled = transformer.transform(&data.as_view())?;
assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
assert_eq!(
    transformer.inverse_transform(&doubled.as_view())?.as_slice(),
    data.as_slice(),
);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The map is **elementwise** by design. A transformation that must read a whole
row or column is expressed by implementing `api::Transformer` directly, which is
the honest way to say the transformation is yours rather than FerricML's.

All three are stateless: nothing about the fitting batch influences a later one,
so a row transforms identically whether it was in the fitting batch or arrives
years later. They consequently have no artifact — there would be nothing in it
but a feature width the pipeline already validates.

## Pipelines: make the transformation part of the model

Fitting a scaler and then forgetting to apply it at prediction time is the
classic mistake, and it fails quietly: the model produces numbers, they are just
wrong. A `Pipeline` removes the opportunity by making the transformation part of
the fitted object.

```rust
use ferricml::api::Estimator;
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};
use ferricml::pipeline::Pipeline;
use ferricml::preprocessing::{StandardScaler, StandardScalerParams};

let data = DenseMatrix::new(
    vec![1.0, 1000.0, 2.0, 3000.0, 3.0, 2000.0, 4.0, 5000.0],
    4,
    2,
)?;
let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;

// Fit the scaler, then fit the model on what the scaler produced.
let scaler = StandardScaler::fit(&data.as_view(), StandardScalerParams::default())?;
let scaled = scaler.transform(&data.as_view())?;
let model = Ridge::fit(&scaled.as_view(), &targets, RidgeParams::default())?;

// One object from here on. Construction checks the feature-width handoff.
let pipeline = Pipeline::new(scaler, model)?;
assert_eq!(pipeline.n_features_in(), 2);

// Inference takes raw rows and scales them on the way through.
let mut workspace = vec![0.0_f32; pipeline.workspace_len(4)?];
let mut predictions = vec![0.0_f32; 4];
pipeline.predict_into(&data.as_view(), &mut workspace, &mut predictions)?;

assert!(predictions.iter().all(|value| value.is_finite()));
# Ok::<(), Box<dyn std::error::Error>>(())
```

Note the two buffers. `workspace_len` reports how much scratch the
transformation needs for a batch of that size; the caller allocates it once and
reuses it, so repeated inference allocates nothing at all.

## More than one stage

`Pipeline` holds exactly one transformer, and fits it and the estimator in one
pass with `Pipeline::fit`. `StagedPipeline` holds a `TransformerStack` of one to
`pipeline::MAX_STAGES` stages plus the estimator, and fits a two-stage
composition in one pass — each stage on the previous stage's output:

```rust
use ferricml::api::Estimator;
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};
use ferricml::pipeline::StagedPipeline;
use ferricml::preprocessing::{
    MinMaxScaler, MinMaxScalerParams, StandardScaler, StandardScalerParams,
};

let data = DenseMatrix::new(
    vec![1.0, 1000.0, 2.0, 3000.0, 3.0, 2000.0, 4.0, 5000.0],
    4,
    2,
)?;
let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;

let pipeline = StagedPipeline::fit(
    &data.as_view(),
    |view| StandardScaler::fit(view, StandardScalerParams::default()),
    |view| MinMaxScaler::fit(view, MinMaxScalerParams::default()),
    |view| Ridge::fit(view, &targets, RidgeParams::default()),
)?;

assert_eq!(pipeline.n_features_in(), 2);

// One workspace, split into a disjoint segment per stage.
let mut workspace = vec![0.0_f32; pipeline.workspace_len(4)?];
let predictions = pipeline.with_transformed(
    &data.as_view(),
    &mut workspace,
    |model, view| model.predict(view),
)?;

assert_eq!(predictions.len(), 4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Every handoff is validated before the composition exists, and the whole
composition is monomorphized: there is no per-row dynamic dispatch, no parameter
erasure, and no string registry of stages.

Longer chains compose and persist the same way, through `StagedPipeline::new`
after fitting each stage on its predecessor's output. One-call fitting stops at
two stages; the reason, and the two measured attempts to lift it, are recorded
on `StagedPipeline::fit`.

Prediction goes through the generic `with_transformed` callback rather than
per-category convenience methods, because several such methods cannot coexist as
inherent methods of one name on one generic type.

## What persists

`StagedPipeline` declares persistence exactly where every stage and its
estimator really have a schema-bound artifact, so asking one that cannot persist
is a **compile error** rather than a runtime surprise. `Pipeline`'s three
concrete compositions — standard scaler with linear regression, ridge, or
logistic regression — predate that scheme and keep their own artifact kinds.

This is intentionally not a generic serialization trait. Unsupported pipeline
shapes have no persistence API at all. See [the artifact
envelope](../artifact-envelope.md).

## Next

- [Evaluation and model selection](../evaluation-and-model-selection.md) —
  cross-validate the whole pipeline, not just the model.
