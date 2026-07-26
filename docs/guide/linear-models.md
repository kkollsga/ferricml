# Linear models

FerricML has five linear estimators. Four regress and one classifies, and the
choice between them is a choice about what you want the penalty to do.

| Estimator | Objective | Choose it when |
| --- | --- | --- |
| `LinearRegression` | Least squares, no penalty | You want the unbiased fit and have enough rows to support it |
| `Ridge` | Least squares + L2 | Features are correlated, or the fit is unstable |
| `Lasso` | Least squares + L1 | You want features *removed*, not merely shrunk |
| `ElasticNet` | Least squares + L1 and L2 | You want selection, but correlated features should share rather than compete |
| `LogisticRegression` | L2-penalized log loss | The target is a class, not a number |

They share the same shape as everything else in the crate: `fit` takes a
`MatrixView`, targets, and a typed parameter value, and returns a `Result`.

## Ordinary least squares

`LinearRegression` is the unpenalized fit. Its one interesting property is what
happens when the problem has no unique answer.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams};

// y = 2x + 1
let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;

let model = LinearRegression::fit(
    &data.as_view(),
    &targets,
    LinearRegressionParams::default(),
)?;

assert!((model.coefficients()[0] - 2.0).abs() < 1e-4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Fitting takes a thin SVD, so a rank-deficient or underdetermined input does not
fail and does not pick a coefficient vector arbitrarily. It returns the
**minimum-norm** solution: with two identical columns, the effect is split
evenly between them rather than assigned to whichever the arithmetic reached
first. That is what makes the fit reproducible on a degenerate design.

Degenerate designs are also where a decomposition is easiest to get wrong, so
the crate checks rather than assumes. The linear-model tests sweep exactly
rank-deficient tall designs and assert on each one that the returned
coefficients zero the normal-equation gradient — that they really are a
least-squares solution — because a previous backend produced coefficients that
did not, on this exact shape, while still reporting the right rank and still
splitting the duplicated pair evenly.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams};

// Column 1 duplicates column 0.
let data = DenseMatrix::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 3, 2)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0])?;

let model = LinearRegression::fit(
    &data.as_view(),
    &targets,
    LinearRegressionParams::default(),
)?;

// Split evenly, and still summing to the effect the data supports.
let coefficients = model.coefficients();
assert!((coefficients[0] - coefficients[1]).abs() < 1e-5);
assert!((coefficients[0] + coefficients[1] - 2.0).abs() < 1e-4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Ridge: shrink everything

`Ridge` adds an L2 penalty. Every coefficient moves toward zero and none reaches
it. Larger `alpha` shrinks harder:

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;

let weak = Ridge::fit(&data.as_view(), &targets, RidgeParams::default().with_alpha(0.01))?;
let strong = Ridge::fit(&data.as_view(), &targets, RidgeParams::default().with_alpha(100.0))?;

assert!(strong.coefficients()[0].abs() < weak.coefficients()[0].abs());
// Shrunk hard, but never to exactly zero.
assert!(strong.coefficients()[0] != 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Lasso and ElasticNet: remove features

An L1 penalty is not differentiable at zero, and that is exactly why it produces
coefficients that are *exactly* zero rather than merely small. Both `Lasso` and
`ElasticNet` are fitted by cyclic coordinate descent, which is the solver an L1
penalty requires.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Lasso, LassoParams};

// Column 0 drives the target; column 1 is noise the fit should discard.
let data = DenseMatrix::new(
    vec![0.0, 1.0, 1.0, -1.0, 2.0, 1.0, 3.0, -1.0, 4.0, 1.0, 5.0, -1.0],
    6,
    2,
)?;
let targets = RegressionTargets::new(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0])?;

let model = Lasso::fit(
    &data.as_view(),
    &targets,
    LassoParams::default().with_alpha(0.5),
)?;

// Removed, not shrunk: exactly zero.
assert_eq!(model.coefficients()[1], 0.0);
assert!(model.coefficients()[0] > 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A removed coefficient is **positively signed** zero, even when it was shrunk
from below. A coefficient the fit removed has no sign left to carry, and a
signed zero would be a different byte pattern for a mathematically identical
model. This is a deliberate divergence recorded in [frozen reference
semantics](../reference-semantics.md).

`ElasticNet` mixes the two penalties through `l1_ratio`: `1.0` is pure L1 and
behaves as `Lasso`, `0.0` is pure L2. The mixture matters when features are
correlated, where pure L1 tends to pick one arbitrarily and drop the rest.

### Two things about `alpha` that will bite otherwise

**`alpha` does not mean the same thing here as it does on `Ridge`.** `Lasso` and
`ElasticNet` minimize the squared error *divided by twice the total sample
weight*; `Ridge`'s penalty accompanies an undivided squared-error term. The two
agree at `ridge_alpha = alpha * total_weight`. Both scales are conventional;
FerricML states the relationship rather than silently reconciling them.

**The penalty applies to raw-scale coefficients.** Fitting centers the design
when an intercept is requested and never rescales the columns, so a feature
measured in millimetres is penalized differently from the same feature in
metres. Scale-free selection comes from putting a `StandardScaler` in front,
where the transformation is explicit and persists with the model — see
[preprocessing and pipelines](preprocessing-and-pipelines.md).

### Convergence is a result, not a hint

Coordinate descent stops when the largest absolute coefficient change across one
full sweep falls under `tol`. If `max_iter` runs out first, the fit returns
`ModelError::SolverDidNotConverge` — it does not return a model that stopped
part way and looks like one that arrived.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Lasso, LassoParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
let targets = RegressionTargets::new(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0])?;

let refused = Lasso::fit(
    &data.as_view(),
    &targets,
    LassoParams::default().with_alpha(0.01).with_max_iter(1).with_tol(1e-12),
);

assert!(refused.is_err());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Logistic regression

The classifier. It has two shapes, and they are different models rather than two
spellings of one.

A **binary** fit is asymmetric: one coefficient row, one raw score per row,
whose sigmoid is the probability of class `1`.

```rust
use ferricml::api::Classifier;
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};

let data = DenseMatrix::new(vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 6, 1)?;
let labels = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;

let model = LogisticRegression::fit(
    &data.as_view(),
    &labels,
    LogisticRegressionParams::default(),
)?;

assert_eq!(model.predict(&data.as_view())?, vec![0, 0, 0, 1, 1, 1]);

// The raw score's sign is the predicted label.
let scores = model.decision_function(&data.as_view())?;
assert!(scores[0] < 0.0 && scores[5] > 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A **multiclass** fit is one joint multinomial optimization — not one binary
model per class — with no pinned reference class, so its raw scores are centred
and sum to approximately zero per row.

```rust
use ferricml::api::Classifier;
use ferricml::data::{ClassTargets, DenseMatrix};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};

let data = DenseMatrix::new(vec![-2.0, -1.5, 0.0, 0.1, 1.5, 2.0], 6, 1)?;
let labels = ClassTargets::new(vec![3, 3, 7, 7, 10, 10])?;

let model = LogisticRegression::fit_multiclass(
    &data.as_view(),
    &labels,
    LogisticRegressionParams::default(),
)?;

// The observed labels, sorted — never renumbered into a dense range.
assert_eq!(model.classes(), &[3, 7, 10]);

let probabilities = model.predict_proba(&data.as_view())?;
assert_eq!(probabilities.len(), 6 * 3);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Probability columns are ordered by `classes()`, for both shapes. Rows are
**never renormalized**: a row sums to one only within `n_classes` `f32` ulps,
and that bound is part of the contract rather than a tolerance to be tightened.
A normalizing pass would move mass without measuring anything, and would cost a
second pass over every prediction row that the allocation-free inference
contract exists to avoid.

### Choosing a solver

`LogisticSolver::Newton` is the default and stays the default. It takes the
exact second-order step and converges in single-digit iterations. Its cost is
one `parameters x parameters` factorization per iteration, and a joint
multinomial system is `classes * parameters` square — so on a wide multiclass
problem the exact path refuses with `ModelError::MulticlassSystemTooLarge`
rather than allocating something it cannot hold.

`LogisticSolver::Lbfgs` never forms that system. Its storage is linear in the
parameter count, so it accepts far larger multiclass problems. Select it when
the exact path refuses — **and select it on a large or strongly penalized
binary problem too, where the default is simply the expensive one.** Newton
accumulates and factorizes a `parameters x parameters` system over every row,
so its per-iteration cost grows as `rows * parameters^2` against the
matrix-free path's `rows * parameters`, and a smaller `C` buys more iterations
to pay it on. Fitting 50,000 rows by 50 columns, same data, same `tol`, same
`max_iter`, both arms this crate (Apple M4, release, median of five after one
warmup):

| `C` | `tol` | `Newton` | `Lbfgs` |
|---|---|---|---|
| 1.0 | 1e-4 | 68.3 ms | 16.6 ms |
| 0.1 | 1e-4 | 83.7 ms | 16.6 ms |
| 1.0 | 1e-8 | 83.5 ms | 25.6 ms |
| 0.1 | 1e-8 | **375.4 ms** | **25.4 ms** |

Three to fifteen times, for coefficients agreeing to six decimals at `1e-8`.
The advantage belongs to the shape and not to the solver in general: at
5,000 x 20 the same comparison is 2.8 ms against 1.8 ms, and on small data the
exact step's single-digit iteration count wins outright. That is why this is a
reason to select `Lbfgs`, not a reason to move the default.

```rust
use ferricml::api::HasParams;
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams, LogisticSolver};

let data = DenseMatrix::new(vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 6, 1)?;
let labels = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;

let model = LogisticRegression::fit(
    &data.as_view(),
    &labels,
    LogisticRegressionParams::default().with_solver(LogisticSolver::Lbfgs),
)?;

assert_eq!(model.get_params().solver(), LogisticSolver::Lbfgs);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Two consequences to know before selecting it:

- **The bound is a property of the solver, not the model.** A shape the exact
  path accepts still takes the exact path and produces the identical fit.
- **`tol` measures a different quantity per solver.** Under `Newton` it is the
  largest absolute coefficient update; under `Lbfgs` it is the gradient norm of
  the mean penalized objective. They are documented apart rather than conflated,
  because pretending they were the same would make one of them wrong.
- **A non-default solver has no artifact representation.** Neither logistic
  payload schema records which solver ran, so rather than writing bytes that
  would decode as a `Newton`-provenance model, a model fitted under `Lbfgs`
  cannot be persisted at all. See [the artifact
  envelope](../artifact-envelope.md).

## Weighted fitting

Every linear estimator has a `fit_weighted` entry point taking `SampleWeights`.
This is also how class weighting is expressed — see [data and
targets](data.md#sample-weights-are-the-only-weighting-concept).

## Next

- [Trees and forests](trees-and-forests.md) for the non-linear estimators.
- [Evaluation and model selection](../evaluation-and-model-selection.md) to
  choose `alpha` by cross-validated search rather than by eye.
