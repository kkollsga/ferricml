# Evaluation and model selection

FerricML provides deterministic evaluation building blocks without owning data
loading, parameter search, or application-specific policy. The public surface
is split between `metrics`, which evaluates predictions, and
`model_selection`, which creates index partitions and evaluates fitted models.

## Metrics

Classification includes exact accuracy, a binary confusion matrix, precision,
recall, F1, Brier score, binary logarithmic loss, and tie-aware ROC AUC.
Regression includes mean absolute error, mean squared error, root mean squared
error, R2, median absolute error, explained variance, and mean absolute
percentage error. Calculations promote `f32` inputs to deterministic `f64`
accumulation. Explained variance ignores a constant prediction offset that R2
penalizes, and percentage error reports `MetricError::Undefined` when any
expected value is zero rather than clamping the denominator.

`roc_curve`, `precision_recall_curve`, and `average_precision_score` sweep the
decision threshold over the same score ordering ROC AUC uses, so the curves and
the scalar scores agree by construction. Both curves are ordered by decreasing
threshold, one point per distinct score however many rows share it; `RocCurve`
additionally starts at the operating point that predicts nothing positive,
whose threshold is above every score and is reported as `f32::INFINITY`.
Average precision sums each threshold's precision weighted by the recall it
gained, without interpolating between operating points.

`multiclass_log_loss` and `multiclass_brier_score` score a whole probability
matrix: row-major, one column per entry of a sorted class list, exactly as a
fitted classifier's `predict_proba` produces it. Neither renormalizes a row.
FerricML's probability rows sum to one only within the documented `f32`
tolerance, and a metric that quietly rescaled them would be scoring numbers no
model produced. Cross-entropy reads only the true class's column, so
`multiclass_log_loss` agrees with `log_loss` on two-class predictions. The
Brier scores deliberately do not coincide: `brier_score` squares the positive
column alone while `multiclass_brier_score` squares every column, so on the
same two-class predictions the multiclass value is exactly twice the binary
one. Both conventions are standard; they are named apart and the relationship
is stated rather than left to be discovered.

Every metric rejects empty or mismatched inputs. Binary metrics reject labels
outside zero and one; probability metrics also reject non-finite values and
values outside `0..=1`. Precision, recall, F1, R2, and ROC AUC return
`MetricError::Undefined` when their required denominator or class distribution
does not exist. Binary logarithmic loss clips valid endpoint probabilities to
`1e-15..=1-1e-15` before taking logarithms.

## Averaging

`ConfusionMatrix` counts one classification result over the sorted union of the
labels observed in either input, with expected labels as rows and predicted
labels as columns. Precision, recall, F1, and F-beta are derived from that one
validated pass and combined through the `Average` vocabulary, so binary and
multiclass evaluation share a single set of names:

- `Binary` reports the positive class, label `1`, alone. A wider label set is
  `MetricError::NotBinary` rather than a silently reinterpreted one-vs-rest
  score, and the resulting values equal the standalone binary functions
  exactly.
- `Micro` pools every class into one count pair. For single-label predictions
  micro-averaged precision, recall, and F-score therefore all equal accuracy.
- `Macro` takes the unweighted mean of the per-class scores.
- `Weighted` weighs each per-class score by that class's true support, so a
  class with no true rows carries no weight.

`balanced_accuracy` and `matthews_correlation` read the same matrix and need no
averaging choice. Balanced accuracy is mean recall over the classes that have
true rows, so it is always defined. Matthews correlation runs from `-1.0` to
`1.0` over any number of classes and is `MetricError::Undefined` — not zero —
when either side of the result is constant and so has no variance.

A class that is never predicted has no precision, and a class with no true rows
has no recall. FerricML reports that as `MetricError::Undefined` by default
rather than substituting a value. `Averaging::with_zero_division` states the
alternative explicitly: `ZeroDivision::Zero` scores the affected class zero and
keeps it in the average, and `ZeroDivision::Skip` removes it from both the sum
and the divisor.

## Deterministic splits

`train_test_split` and `stratified_train_test_split` return validated `Split`
values containing sorted, disjoint train and test indices with complete sample
coverage. `TestSize` accepts an exact count or a fraction rounded upward.
Seeded shuffling uses a stable crate-owned algorithm, so identical parameters
produce identical membership across supported platforms.

`TimeSeriesSplit` treats rows as ordered, index zero oldest. Each fold trains
on a prefix and tests on the window immediately after it, so no fold is fitted
on a row that comes after the rows it is evaluated on, and later folds train on
strictly more history. Test windows all hold `samples / (n_splits + 1)` rows and
end at the last row, so any remainder lengthens the first training window.
`with_gap` drops rows between each training window and its test window for
targets that are not knowable immediately. Every fold except the last therefore
leaves later rows out of both partitions: a `Split::partial`, which keeps every
other split guarantee and still reports the dataset size through
`sample_count`, so cross-validation validates it exactly as it does a complete
split. `LeaveOneOut` holds out one sample per split.

`KFold` and `StratifiedKFold` validate before returning lazy, exact-size fold
iterators. Fold sizes differ by at most one. Stratification accepts arbitrary
`u8` labels and requires enough members of every observed class to place that
class in every requested partition.

`GroupKFold` takes one group identifier per row and assigns whole groups to
folds, so no group is ever on both sides of a split. It needs no seed: groups
are taken largest first, ties by increasing identifier, each into the fold
holding the fewest rows. Fold sizes are then as even as whole groups allow.
`RepeatedKFold` runs shuffled K-fold `n_repeats` times, deriving each repeat's
shuffle seed from the configured one, and yields every fold of every repeat in
order.

`GroupShuffleSplit` draws whole groups at random for each of `n_splits`
independent holdouts, so no group is ever on both sides of a split. It is not a
partition: two splits may hold out the same group, and no row is promised a
turn. Its size is stated in `TestGroupSize`, which counts **groups**, not the
rows `TestSize` counts everywhere else. That is a deliberate second type rather
than a second meaning for one: rows only move a whole group at a time, so a row
target would silently become an approximation, while a group target stays exact.
Every split therefore holds out exactly the requested number of whole groups,
and its row count follows from which groups were drawn. `Count` is that number
directly; `Fraction` is a fraction of the distinct groups, rounded upward. Each
split's group draw is seeded from the configured seed and the split index, so
identical parameters reproduce identical membership.

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

`score_classifier` and `score_regressor` make one batch prediction call and
hand the result to a score. Probability-based binary scores accept fitted class
layouts `[0]`, `[1]`, and `[0, 1]`; other layouts produce an explicit error.
That handling lives in the scoring functions alone, so no consumer repeats it.

A score is the `ClassificationScore` or `RegressionScore` trait, not a closed
list. `ClassificationScorer` and `RegressionScorer` remain the built-in set and
are ordinary implementations of those traits, so a caller can score on a metric
FerricML has not enumerated by implementing the trait instead of reimplementing
prediction. A classification score declares which batch output it reads through
`output_kind`, and receiving any other kind is `ScoringError::UnsupportedOutput`
rather than a substituted value. Every score also declares `greater_is_better`,
which is what lets permutation importance orient its result without knowing
which metric it holds.

A classification score can also read a whole probability matrix.
`ClassifierOutput::ProbabilityMatrix` carries the row-major matrix together
with the classifier's own sorted class list, so the columns are *named* rather
than assumed to be `[0, 1]`, and a score reading it works for any observed class
set. `ClassificationScorer::MulticlassLogLoss` and `MulticlassBrier` are the
built-in scores that do. `score_multiclass_classifier` is `score_classifier`
over `ClassTargets` instead of `BinaryTargets`; both reach the model through one
implementation, so the prediction call, the class-layout handling, and the
workspace reuse exist exactly once. The `[0]`, `[1]`, and `[0, 1]`
positive-probability layouts are unchanged, and a wider class set asked for one
of them is still `ScoringError::UnsupportedClasses` rather than a reinterpreted
column.

`score_classifier_with` and `score_regressor_with` take a `ScoringWorkspace`
holding the batch output. Reusing one workspace across calls of the same shape
allocates on the first call only, which is what makes repeated scoring —
cross-validation across folds, permutation importance across repeats — free of
per-call allocation. Cross-validation and permutation importance both consume
exactly these entry points, so there is one implementation of scoring.

Cross-validation consumes any iterator of validated `Split` values, fits one
typed model per fold through a caller closure, and returns scores in iterator
order. It is deliberately serial: a fixed split order, fit parameters, seed,
and thread count produce repeatable fitted artifacts and scores. Errors retain
the zero-based fold index for fitting, prediction, metric, or class-layout
failures.

`cross_validate_classifier` is the only classifier entry point, and the two
choices a caller makes are carried rather than duplicated into more functions.
The target vocabulary is a type parameter: any `ClassificationTargets` —
`BinaryTargets` or `ClassTargets` — folds through the same loop, because label
arity is a property of the metric and not of the loop, and `MulticlassLogLoss`
and `MulticlassBrier` already read a whole probability matrix over any observed
class set. The model's scoring capability is a value: a final `view` argument,
`ScorableClassifier::probabilistic` or `ScorableClassifier::labels_only`, which
is the same mechanism `score_classifier` and permutation importance take. The
two axes compose instead of multiplying, so a further target shape or a further
view is an implementation rather than a fourth entry point.
`grid_search_classifier` forwards both unchanged. Neither buys leniency: a
binary positive-probability metric asked for on a wider class set is still
`CrossValidationError::UnsupportedClasses`, and a probability metric under a
labels-only view is still `CrossValidationError::UnsupportedOutput`.

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

## Typed parameter search

`ParameterGrid` holds candidates of a real parameter type. An axis is that
type's own `with_*` builder method plus the values to pass it, so there are no
string keys, a misnamed parameter is a compile error, and one grid can cross a
`usize` axis with an `f32` axis without erasing either. Candidates are
materialized as each axis is added, so the axis added last varies fastest and
the order is fixed at construction. `from_candidates` takes an explicit list for
parameters that are not independent, and an axis with no values empties the grid
rather than being ignored.

`grid_search_classifier` and `grid_search_regressor` evaluate that grid. The
split iterator is drained once, so every candidate is cross-validated over
exactly the same folds and the comparison between candidates is not confounded
by a re-drawn partition. Each candidate then runs through `cross_validate_*`,
which runs through the same caller-owned scoring entry point batch scoring and
permutation importance use, so search adds no second evaluation path and a
caller-defined score behaves in search exactly as it does anywhere else.

The result reports every candidate's parameters and every fold's score, not only
the winner's summary. The winner is the candidate with the best mean fold score
— largest when the score declares `greater_is_better`, smallest otherwise — and
the comparison is strict, so an exact tie keeps the earliest candidate in grid
order. Search does not refit: the winning parameters go back through the
caller's own fitting closure, which keeps the refit policy in the caller's code.

Failures stay attributed. Everything that is wrong with the call itself — target
length, no splits, a split built for another dataset — is `SearchError::Setup`
and costs no fitting; a candidate's failure is `SearchError::Candidate`, keeping
the underlying fold index; and a score that returns a non-finite value is
`SearchError::NonFiniteScore` rather than an unorderable comparison silently
deciding a winner.

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};
use ferricml::model_selection::{
    KFold, ParameterGrid, RegressionScorer, grid_search_regressor,
};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6, 1)?;
let targets = RegressionTargets::new(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0])?;
let grid = ParameterGrid::new(RidgeParams::default())
    .axis([0.01_f32, 1.0, 100.0], RidgeParams::with_alpha);
let result = grid_search_regressor(
    &data.as_view(),
    &targets,
    KFold::new(3).split(data.rows())?,
    &grid,
    RegressionScorer::MeanSquaredError,
    |train, train_targets, params| Ridge::fit(train, train_targets, params.clone()),
)?;
assert_eq!(result.len(), 3);
assert_eq!(result.best_params().alpha(), 0.01);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Randomized and Bayesian search, parallel candidate evaluation, early stopping,
and nested model selection remain outside this contract. They can be added
without exposing fitted model internals or weakening deterministic split
semantics.
