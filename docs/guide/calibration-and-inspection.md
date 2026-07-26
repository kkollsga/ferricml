# Calibration and inspection

Two questions come after a model fits. *Do I believe its probabilities?* and
*what is it actually using?* FerricML answers the first with calibration and the
second with permutation importance, and both are built to work through the
public contract rather than through model internals.

## Baselines first

Before either question is worth asking, a model has to beat doing nothing.
`dummy` is that floor, and it is deliberately the most boring code in the crate:
no tunable behaviour, no weighted entry point, no artifact kind.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::dummy::{DummyRegressor, DummyRegressorParams};
use ferricml::metrics::r2_score;

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![1.0, 2.0, 3.0, 4.0])?;

let baseline = DummyRegressor::fit(
    &data.as_view(),
    &targets,
    DummyRegressorParams::default(),
)?;

// The training mean, for every row.
let predictions = baseline.predict(&data.as_view())?;
assert_eq!(predictions, vec![2.5, 2.5, 2.5, 2.5]);

// Which is exactly what an R-squared of zero means, by definition.
assert!(r2_score(targets.as_slice(), &predictions)?.abs() < 1e-6);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`DummyClassifier` predicts the majority class, and its probabilities are the
observed class frequencies — identical on every row:

```rust
use ferricml::api::{Classifier, ProbabilisticClassifier};
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::dummy::{DummyClassifier, DummyClassifierParams};

// Three of class 0, one of class 1.
let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let labels = BinaryTargets::new(vec![0, 0, 0, 1])?;

let baseline = DummyClassifier::fit(
    &data.as_view(),
    &labels,
    DummyClassifierParams::default(),
)?;

assert_eq!(baseline.predict(&data.as_view())?, vec![0, 0, 0, 0]);

let probabilities = baseline.predict_proba(&data.as_view())?;
assert_eq!(&probabilities[0..2], &[0.75, 0.25]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Matching the dummy means the features contributed nothing. These are also the
reference implementation of the estimator contract, which is why they are worth
reading if you are implementing one.

## Calibration: fixing overconfidence

A classifier can order rows correctly and still be wrong about *how* confident
it is. A calibrator is a fitted monotone map of one model score onto a
probability, so it changes how confident a prediction is without changing which
way round two rows are ordered.

FerricML has two, and they differ in what they assume.

**`PlattCalibrator`** is parametric: a two-parameter logistic fit. It regresses
on Platt's prior-corrected targets rather than on the raw labels, which is what
keeps the fit finite when the score separates the classes perfectly. With raw
labels that problem has no finite maximum-likelihood solution, and the resulting
map would assert exactly the certainty calibration exists to remove.

```rust
use ferricml::calibration::{Calibrator, PlattCalibrator, PlattParams};
use ferricml::data::BinaryTargets;

// Perfectly separated scores.
let scores = [-3.0_f32, -2.0, -1.0, 1.0, 2.0, 3.0];
let labels = BinaryTargets::new(vec![0, 0, 0, 1, 1, 1])?;

let calibrator = PlattCalibrator::fit(&scores, &labels, PlattParams::default())?;

// A probability, never an exact certainty.
let high = calibrator.calibrate(3.0);
assert!(high > 0.5 && high < 1.0);

// Monotone, so it never reorders two rows.
assert!(calibrator.calibrate(1.0) < calibrator.calibrate(2.0));
# Ok::<(), Box<dyn std::error::Error>>(())
```

**`IsotonicRegression`** is the non-parametric one, and is also a standalone
monotone regressor over a single-column matrix. It states its tie convention
rather than inheriting one: observations sharing an input value are pooled into
their mean *before* pool-adjacent-violators runs. That is forced rather than
chosen — a function of one input takes one value at one input — and it makes the
fit independent of observation order. Prediction interpolates linearly between
fitted points and holds the end values outside the fitted range, rather than
extrapolating a trend the fit never observed.

### Composing a calibrated classifier

`CalibratedClassifier` wraps an already-fitted classifier with an already-fitted
calibrator, and is itself an ordinary `Classifier` — so it reaches the scorer,
cross-validation and permutation-importance paths without any of them learning
that calibration exists.

```rust
use ferricml::api::{Classifier, ProbabilisticClassifier};
use ferricml::calibration::{CalibratedClassifier, PlattParams};
use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};
use ferricml::metrics::roc_auc_score;

let values: Vec<f32> = (0..20).map(|index| index as f32 - 10.0).collect();
let labels: Vec<u8> = (0..20).map(|index| u8::from(index >= 10)).collect();
let data = DenseMatrix::new(values, 20, 1)?;
let labels = BinaryTargets::new(labels)?;

let model = LogisticRegression::fit(
    &data.as_view(),
    &labels,
    LogisticRegressionParams::default(),
)?;
let before = model.predict_proba(&data.as_view())?;

let calibrated = CalibratedClassifier::fit_platt(
    model,
    &data.as_view(),
    &labels,
    PlattParams::default(),
)?;
let after = calibrated.predict_proba(&data.as_view())?;

// Calibration is monotone, so any threshold-sweeping score is unchanged.
let positive_before: Vec<f32> = before.chunks(2).map(|row| row[1]).collect();
let positive_after: Vec<f32> = after.chunks(2).map(|row| row[1]).collect();
assert_eq!(
    roc_auc_score(labels.as_slice(), &positive_before)?,
    roc_auc_score(labels.as_slice(), &positive_after)?,
);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Four things about that composition are decisions rather than accidents:

- **The calibration rows are always a parameter**, never the wrapped model's own
  training rows taken implicitly. Calibrating on the data the model was fitted
  on is a mistake FerricML will let you make, but not one it will make for you.
- **The score being calibrated is the wrapped model's positive-class
  probability**, which is the one score the `ProbabilisticClassifier` contract
  requires. That is what lets the wrapper be generic over that public contract
  rather than over the estimators FerricML happens to ship.
- **Predicted labels are the argmax of the calibrated probabilities**, so a row
  whose probability crosses the decision point does change label. A classifier
  whose labels disagreed with its own probabilities would be a silent wrong
  answer.
- **Capabilities are declared per calibrator, not inherited.** The composition
  owns already-fitted parts, so weighted fitting, persistence and multiclass
  fitting are declared away structurally. Both calibrators declare
  probabilities — producing a calibrated probability is what the wrapper is
  for — and a Platt composition additionally gains a `decision_function` the
  wrapped model may never have had.

## Permutation importance: what the model uses

Shuffle one column, re-score, and see how much worse the model gets. That
difference is how much it was relying on that column.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::inspection::{PermutationImportanceParams, permutation_importance_regressor};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams};
use ferricml::model_selection::RegressionScorer;

// Column 0 drives the target; column 1 is noise.
let mut values = Vec::new();
let mut targets = Vec::new();
for index in 0..24 {
    let signal = index as f32;
    values.push(signal);
    values.push(if index % 2 == 0 { 1.0 } else { -1.0 });
    targets.push(2.0 * signal);
}
let data = DenseMatrix::new(values, 24, 2)?;
let targets = RegressionTargets::new(targets)?;

let model = LinearRegression::fit(
    &data.as_view(),
    &targets,
    LinearRegressionParams::default(),
)?;

let importance = permutation_importance_regressor(
    &model,
    &data.as_view(),
    &targets,
    RegressionScorer::MeanSquaredError,
    PermutationImportanceParams::default().with_random_state(4),
)?;

assert!(importance.means()[0] > importance.means()[1]);
assert_eq!(importance.ranked()[0], 0); // most important first
# Ok::<(), Box<dyn std::error::Error>>(())
```

This is **model-agnostic**: it works through the public batch prediction and
scoring contracts only, so it needs no estimator cooperation and exposes no
internals. It holds no scoring logic of its own either — it calls the same
caller-owned-buffer scoring entry point cross-validation does.

Two properties are worth relying on:

- **Values are quality losses, always oriented the same way.** A larger number
  means a more important feature, whichever direction the underlying metric
  improves in. The orientation comes from the score's own `greater_is_better`
  declaration rather than from a guess about the metric's name.
- **Repeats are cheap.** Prediction and permutation workspace are allocated
  once, so the cost of extra repeats is scoring alone. `with_n_repeats` and
  `with_random_state` make the result reproducible.

There is a matching `permutation_importance_classifier`, and `_into` variants of
both that write into caller-owned buffers.

## Next

- [Saving and loading models](persistence.md) — putting a fitted model on disk.
- [Evaluation and model selection](../evaluation-and-model-selection.md) — the
  scorer contract these paths share.
