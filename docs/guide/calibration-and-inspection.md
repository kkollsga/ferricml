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
probability, so its main job is to change how confident a prediction is.

Monotone is weaker than ranking-preserving, and it is worth being precise about
the difference before reaching for either calibrator, because both can reach
the weaker case on a calibration sample you might plausibly supply:

- A **strictly increasing** map preserves the ranking of any two rows exactly,
  so a threshold-sweeping score such as ROC AUC is unchanged. `PlattCalibrator`
  is this case exactly when its fitted `slope()` is positive.
- A **non-decreasing** map may send two distinct scores to one value. It never
  inverts a pair, but a tied pair no longer contributes a full correct
  ordering, so ROC AUC is not guaranteed unchanged. `IsotonicRegression` is
  this case whenever it pools; pooled into one block it is constant and ROC AUC
  becomes `0.5`.
- A **decreasing** map is monotone too, and reverses every pairwise comparison:
  ROC AUC becomes `1.0 - auc`. `PlattCalibrator` is this case when its fitted
  slope is negative.

A negative Platt slope is not a fitting failure. The fit is the exact
maximum-likelihood answer, and its sign is the sign of the calibration sample's
class mean gap — the mean score over its positive rows minus the mean over its
negative rows. A small held-out fold whose few positive rows happen to score
low produces one honestly, for a model that ranks well everywhere else. Neither
calibrator rejects such a sample, because the sample is what it is; the fitted
parameters are public so that a caller who depends on ranking can check them.

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

// A positive slope, so the map is strictly increasing and cannot reorder two
// rows. The sign belongs to the calibration sample, so it is checked.
assert!(calibrator.slope() > 0.0);
assert!(calibrator.calibrate(1.0) < calibrator.calibrate(2.0));

// The mirror-image sample fits the mirror-image map, which inverts every pair.
let mirrored = PlattCalibrator::fit(
    &[3.0_f32, 2.0, 1.0, -1.0, -2.0, -3.0],
    &labels,
    PlattParams::default(),
)?;
assert!(mirrored.slope() < 0.0);
assert!(mirrored.calibrate(1.0) > mirrored.calibrate(2.0));
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

// Rows held out of the fit above: the workflow calibration is written for.
let holdout = DenseMatrix::new(
    vec![-7.5_f32, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5],
    8,
    1,
)?;
let holdout_labels = BinaryTargets::new(vec![0, 0, 0, 0, 1, 1, 1, 1])?;
let calibrated = CalibratedClassifier::fit_platt(
    model,
    &holdout.as_view(),
    &holdout_labels,
    PlattParams::default(),
)?;
let after = calibrated.predict_proba(&data.as_view())?;

// This fold fitted a positive slope, so the map is strictly increasing and
// every threshold-sweeping score is unchanged. That condition is asserted
// rather than assumed: it is what the guarantee rests on.
assert!(calibrated.calibrator().slope() > 0.0);

// Scored against labels the model does not reproduce exactly, so both AUCs sit
// strictly between 0.5 and 1 and would differ if calibration could reorder.
let noisy = BinaryTargets::new(
    (0..20).map(|index| u8::from((index >= 10) != (index == 8 || index == 11))).collect(),
)?;
let positive_before: Vec<f32> = before.chunks(2).map(|row| row[1]).collect();
let positive_after: Vec<f32> = after.chunks(2).map(|row| row[1]).collect();
let raw = roc_auc_score(noisy.as_slice(), &positive_before)?;
assert!(raw > 0.5 && raw < 1.0);
assert_eq!(raw, roc_auc_score(noisy.as_slice(), &positive_after)?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Five things about that composition are decisions rather than accidents:

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
- **The ranking guarantee is conditional, and the condition is reachable.**
  `calibrator()` exposes the fitted map, so a caller who depends on ranking
  reads `slope()` for a Platt composition or `values()` for an isotonic one.
  Nothing is rejected at fit time: an inverting calibration fold is a fact
  about the fold, and turning it into an error would make honest small folds
  unfittable.
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

There is a matching `permutation_importance_classifier` — one entry point over
any `data::ClassificationTargets`, so `BinaryTargets` and `ClassTargets` are
inspected the same way and only the metric has to know how many classes there
are — and `_into` variants of both that write into caller-owned buffers.

## Next

- [Saving and loading models](persistence.md) — putting a fitted model on disk.
- [Evaluation and model selection](../evaluation-and-model-selection.md) — the
  scorer contract these paths share.
