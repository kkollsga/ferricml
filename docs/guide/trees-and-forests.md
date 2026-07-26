# Trees and forests

A linear model draws one line. A tree draws a staircase, so it represents a
threshold effect exactly where a linear fit can only approximate one. FerricML
has eight tree-based estimators, and they form a progression: one tree, many
independent trees, and many sequential trees.

| Estimator | Shape | Choose it when |
| --- | --- | --- |
| `DecisionTreeClassifier`, `DecisionTreeRegressor` | One tree | You want the fit to be inspectable, or you are building something on top of it |
| `RandomForestClassifier`, `RandomForestRegressor` | Many bagged trees | The general-purpose default |
| `ExtraTreesClassifier`, `ExtraTreesRegressor` | Many bagged trees, random thresholds | You want more variance reduction, or faster fitting |
| `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` | Sequential trees on residuals | You want the strongest fit and will tune for it |

## One tree

A tree splits the feature space on thresholds. On a step function, it is exact:

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::tree::{DecisionTreeRegressor, DecisionTreeRegressorParams};

// Everything below 2.5 is worth 0; everything above is worth 10.
let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
let targets = RegressionTargets::new(vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0])?;

let model = DecisionTreeRegressor::fit(
    &data.as_view(),
    &targets,
    DecisionTreeRegressorParams::default(),
)?;

assert_eq!(model.predict(&data.as_view())?, vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`max_depth` bounds how fine the staircase can get. A depth-one tree is a single
split, so it has exactly two possible outputs no matter what the data does:

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::tree::{DecisionTreeRegressor, DecisionTreeRegressorParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![0.0, 1.0, 2.0, 3.0])?;

let stump = DecisionTreeRegressor::fit(
    &data.as_view(),
    &targets,
    DecisionTreeRegressorParams::default().with_max_depth(Some(1)),
)?;

let mut distinct = stump.predict(&data.as_view())?;
distinct.dedup();
assert_eq!(distinct.len(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A standalone tree is not a reimplementation of the one a forest grows: both call
one grower under one configuration type, and a tree fitted here is bit-identical
to the single member of a one-tree, no-bootstrap, all-columns forest at the same
seed. That is asserted in the crate's tests rather than assumed.

`DecisionTreeClassifier` has the same two shapes every FerricML classifier has —
a binary `fit` and a `fit_multiclass` over any observed class set:

```rust
use ferricml::api::Classifier;
use ferricml::data::{ClassTargets, DenseMatrix};
use ferricml::tree::{DecisionTreeClassifier, DecisionTreeClassifierParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
let labels = ClassTargets::new(vec![3, 3, 7, 7, 10, 10])?;

let model = DecisionTreeClassifier::fit_multiclass(
    &data.as_view(),
    &labels,
    DecisionTreeClassifierParams::default(),
)?;

assert_eq!(model.classes(), &[3, 7, 10]);
assert_eq!(model.predict(&data.as_view())?, vec![3, 3, 7, 7, 10, 10]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The two are genuinely different models even on two-class data: a binary leaf
stores the probability of class `1` as a scalar, a multiclass leaf stores a full
distribution. Both persist, under one artifact kind that records which leaf
arithmetic it holds.

## Many trees: the random forest

One tree overfits. A forest fits many trees, each on a bootstrap resample of the
rows and each considering a random subset of columns at every split, then
averages them. For a classifier, that average is of per-tree **probability
vectors** — soft averaging, not a majority vote of per-tree labels.

```rust
use ferricml::api::Classifier;
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::ensemble::{RandomForestClassifier, RandomForestClassifierParams};

let values: Vec<f32> = (0..40).map(|index| index as f32).collect();
let labels: Vec<u8> = (0..40).map(|index| u8::from(index >= 20)).collect();
let data = DenseMatrix::new(values, 40, 1)?;
let labels = BinaryTargets::new(labels)?;

let model = RandomForestClassifier::fit(
    &data.as_view(),
    &labels,
    RandomForestClassifierParams::default()
        .with_n_estimators(16)
        .with_random_state(7),
)?;

assert_eq!(model.predict(&data.as_view())?, labels.as_slice().to_vec());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Bootstrapping is on by default, which is worth knowing when experimenting: each
tree sees a resample, so on a handful of rows the ensemble is noisy. Give it
enough rows for the averaging to mean something.

### `n_jobs` does not change the fit

Forest training is FerricML's only parallel fitting path, and it is built so
that the worker count is invisible in the result. Tree `i`'s seed is derived
from `i` alone, and finished trees are sorted back into index order before
packing — so a four-worker fit is the *same model* as a serial one, not merely a
similar one.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::ensemble::{NJobs, RandomForestRegressor, RandomForestRegressorParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0])?;

let params = RandomForestRegressorParams::default()
    .with_n_estimators(8)
    .with_random_state(11);

let serial = RandomForestRegressor::fit(&data.as_view(), &targets, params.clone())?;
let parallel = RandomForestRegressor::fit(
    &data.as_view(),
    &targets,
    params.with_n_jobs(NJobs::Count(4)),
)?;

assert_eq!(serial.predict(&data.as_view())?, parallel.predict(&data.as_view())?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

This is a stronger property than the crate-wide determinism promise, which fixes
the thread count. See [determinism](../determinism.md) for the full tiering.

### Feature subsets

`MaxFeatures` controls how many columns a split considers: `All`, `Sqrt`, or an
exact `Count`. Fewer columns per split means more decorrelated trees and a
faster fit; `Sqrt` is the conventional choice for classification.

## Extra-trees: randomize the threshold too

`ExtraTreesClassifier` and `ExtraTreesRegressor` draw the candidate columns
exactly as a random forest does, then draw **one uniform threshold per candidate
column** over that column's range within the node, keeping the best-scoring
draw — rather than evaluating every boundary between adjacent distinct values.

Trees therefore decorrelate through their thresholds instead of through
resampling, which is why `bootstrap` defaults to `false` here and `true` on a
random forest:

```rust
use ferricml::api::Classifier;
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::ensemble::{
    ExtraTreesClassifier, ExtraTreesClassifierParams, RandomForestClassifierParams,
};

// The defaults differ, and the difference is the point.
assert!(!ExtraTreesClassifierParams::default().bootstrap());
assert!(RandomForestClassifierParams::default().bootstrap());

let values: Vec<f32> = (0..40).map(|index| index as f32).collect();
let labels: Vec<u8> = (0..40).map(|index| u8::from(index >= 20)).collect();
let data = DenseMatrix::new(values, 40, 1)?;
let labels = BinaryTargets::new(labels)?;

let model = ExtraTreesClassifier::fit(
    &data.as_view(),
    &labels,
    ExtraTreesClassifierParams::default()
        .with_n_estimators(32)
        .with_random_state(5),
)?;

assert_eq!(model.predict(&data.as_view())?, labels.as_slice().to_vec());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Randomized splitting is also available on a standalone tree, through
`Splitter::Random` — it is one typed parameter rather than two more public types
and two more permanent artifact names.

## Boosting: trees in sequence

A forest grows trees independently and averages them. Boosting grows them in
sequence: each iteration fits a tree to what the previous ones got wrong and
adds a shrunk share of it. `learning_rate` and `max_iter` trade against each
other — a smaller rate needs more iterations and usually generalizes better.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::ensemble::{HistGradientBoostingRegressor, HistGradientBoostingRegressorParams};
use ferricml::metrics::mean_squared_error;

let values: Vec<f32> = (0..64).map(|index| index as f32).collect();
let squares: Vec<f32> = values.iter().map(|value| value * value).collect();
let data = DenseMatrix::new(values, 64, 1)?;
let targets = RegressionTargets::new(squares)?;

let brief = HistGradientBoostingRegressor::fit(
    &data.as_view(),
    &targets,
    HistGradientBoostingRegressorParams::default().with_max_iter(2),
)?;
let longer = HistGradientBoostingRegressor::fit(
    &data.as_view(),
    &targets,
    HistGradientBoostingRegressorParams::default().with_max_iter(60),
)?;

let brief_error = mean_squared_error(targets.as_slice(), &brief.predict(&data.as_view())?)?;
let longer_error = mean_squared_error(targets.as_slice(), &longer.predict(&data.as_view())?)?;
assert!(longer_error < brief_error);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### The one default that will surprise you

`min_samples_leaf` defaults to **20** on both boosted estimators. A node can
only split when both sides keep at least that many rows, so on a dataset smaller
than about forty rows *no split is ever admissible* — and the fit succeeds
anyway, returning a model that is its baseline and nothing else.

That is the parameter doing its job, and it matches the reference default. What
makes it worth stating is that it is **silent**: there is no error and no
warning separating "boosting had nothing left to add" from "boosting was never
able to start", so a small experiment produces a constant prediction with no
explanation. If a boosted model predicts one value everywhere, check the row
count against `min_samples_leaf` first. The crate pins this behaviour in a test
so it cannot change unnoticed.

The two boosted estimators differ in what a leaf *means* rather than in what a
tree looks like. The regressor descends squared error and its leaves are means;
the classifier descends binary log loss, works in raw score space, and divides
each leaf by the summed curvature of its rows rather than by their count. With a
constant hessian the two denominators coincide; with a varying one they do not.

Only two-class boosting exists. Multiclass boosting is a separate model with a
separate objective rather than a widening of this one, and is deliberately
absent rather than approximated.

For the boosted regressor's full parameter subset and what it deliberately
excludes, see [histogram gradient
boosting](../histogram-gradient-boosting.md).

## Weighted fitting

Every estimator on this page takes sample weights through `fit_weighted`. A
forest treats a weight as a fractional row count: it multiplies the bootstrap
replication count, and that product is what every impurity, split threshold,
leaf mean and leaf distribution accumulates.

The consequence to know is that `min_samples_split` and `min_samples_leaf` bound
the node's total **weight**, not its row count. That is a deliberate divergence
from the reference, taken so an integer sample weight is the same fitted model
as repeating that row — unconditionally, rather than only while the constraint
does not bind. A row of weight zero is not in the training sample at all, so it
is excluded from the bootstrap draw rather than consuming one of it.

Unweighted fitting is unaffected: a node's weight is its row count when every
weight is one.

## Next

- [Preprocessing and pipelines](preprocessing-and-pipelines.md) — trees do not
  need scaled features, but the models you compare them against do.
- [Evaluation and model selection](../evaluation-and-model-selection.md) — how
  to tune `n_estimators`, `max_depth` and `learning_rate` honestly.
