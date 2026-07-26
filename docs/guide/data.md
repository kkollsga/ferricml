# Data and targets

FerricML does not accept raw slices. Every estimator takes a validated
container, and construction of that container is where all input validation
happens. There are six of them and they divide cleanly:

| Type | Holds | Used for |
| --- | --- | --- |
| `DenseMatrix` | Owned row-major `f32` features | The feature matrix you own |
| `MatrixView<'a>` | Borrowed row-major `f32` features | What estimators actually take |
| `BinaryTargets` | `u8` labels, each `0` or `1` | Two-class classification |
| `ClassTargets` | Arbitrary `u8` labels + observed class set | Multiclass classification |
| `RegressionTargets` | Finite `f32` values | Regression |
| `SampleWeights` | Finite, non-negative `f32`, positive total | Weighted fitting of any kind |

## Why validation lives here

A container that exists is a container whose invariants hold. `DenseMatrix::new`
checks the shape, the exact buffer length, and the finiteness of every element
before returning; after that, no estimator re-checks them, and no prediction
path pays for a check per row.

The second reason is where failure happens. Validation at the boundary means an
invalid input is refused *before* training work or allocation begins, rather
than partway through a fit that then has to be unwound.

Errors are typed and name the location of the problem:

```rust
use ferricml::data::{DataError, DenseMatrix};

// A buffer that does not fill the requested shape.
assert_eq!(
    DenseMatrix::new(vec![1.0, 2.0, 3.0], 2, 2),
    Err(DataError::LengthMismatch { expected: 4, actual: 3 }),
);

// A non-finite value, reported at its flat row-major index.
assert_eq!(
    DenseMatrix::new(vec![1.0, 2.0, f32::NAN, 4.0], 2, 2),
    Err(DataError::NonFiniteValue { index: 2 }),
);
```

Note what is *not* offered: there is no option to drop the offending row, impute
it, or coerce it. FerricML's inputs are finite and dense, and deciding what a
missing value means is the caller's problem, because it is a modelling decision
rather than a numerical one.

## Matrices are row-major

Element `(row, column)` lives at flat index `row * columns + column`. A row is
therefore contiguous, which is what lets row iteration and single-row prediction
avoid allocating.

```rust
use ferricml::data::DenseMatrix;

// Three samples, two features each.
let data = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)?;

assert_eq!(data.row(0), Some(&[1.0, 2.0][..]));
assert_eq!(data.row(2), Some(&[5.0, 6.0][..]));
assert_eq!(data.get(1, 1), Some(4.0));

let rows: Vec<&[f32]> = data.iter_rows().collect();
assert_eq!(rows.len(), 3);
# Ok::<(), ferricml::data::DataError>(())
```

## Owned matrix, borrowed view

Estimators take `&MatrixView`, not `&DenseMatrix`. `MatrixView` is small,
`Copy`, and validated on construction just as the owned matrix is, so one
allocation serves fitting, prediction and scoring.

Use `DenseMatrix::as_view` when you own the data, and `MatrixView::new` when the
data already lives in a buffer you would rather not copy:

```rust
use ferricml::data::{DenseMatrix, MatrixView};

let owned = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
let from_owned = owned.as_view();

let buffer = vec![1.0_f32, 2.0, 3.0, 4.0];
let from_slice = MatrixView::new(&buffer, 2, 2)?;

assert_eq!(from_owned, from_slice);
# Ok::<(), ferricml::data::DataError>(())
```

## Targets come in three kinds

**`BinaryTargets`** guarantees every label is `0` or `1`. Anything else is
rejected with its index and value.

**`ClassTargets`** accepts any `u8` label and additionally records the *observed
class set*: the sorted, deduplicated labels actually present. That set is the
column order of every probability matrix a model fitted on those targets
produces. It is deliberately not assumed contiguous and not assumed zero-based,
so a caller never renumbers labels to fit a dense range:

```rust
use ferricml::data::ClassTargets;

let targets = ClassTargets::new(vec![7, 3, 10, 3])?;

assert_eq!(targets.classes(), &[3, 7, 10]);
assert_eq!(targets.n_classes(), 3);
// Column j of a probability matrix is the probability of classes()[j].
assert_eq!(targets.class_index(10), Some(2));
assert_eq!(targets.class_index(0), None);
# Ok::<(), ferricml::data::DataError>(())
```

**`RegressionTargets`** holds finite `f32` values and rejects `NaN` and
infinities, so no estimator has to decide what fitting against one would mean.

`BinaryTargets` converts into `ClassTargets`, which is how a binary problem
reaches a multiclass-shaped API without restating its labels:

```rust
use ferricml::data::{BinaryTargets, ClassTargets};

let binary = BinaryTargets::new(vec![0, 1, 1, 0])?;
let as_classes = ClassTargets::from(binary);

assert_eq!(as_classes.classes(), &[0, 1]);
assert_eq!(as_classes.n_classes(), 2);
# Ok::<(), ferricml::data::DataError>(())
```

## Selecting rows

`select_rows` and `select` copy a subset in the requested order and return the
same validated type, which is what makes it safe to feed a cross-validation
fold's indices straight back into a fit. They are also how you apply a split by
hand:

```rust
use ferricml::data::{DenseMatrix, RegressionTargets};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![0.0, 1.0, 4.0, 9.0])?;

let train = data.select_rows(&[0, 2, 3])?;
let train_targets = targets.select(&[0, 2, 3])?;

assert_eq!(train.as_slice(), &[0.0, 2.0, 3.0]);
assert_eq!(train_targets.as_slice(), &[0.0, 4.0, 9.0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The splitters in [evaluation and model
selection](../evaluation-and-model-selection.md) produce the index lists these
methods consume.

## Sample weights are the only weighting concept

FerricML has exactly one way to weight a fit: `SampleWeights`, passed to a
`fit_weighted` entry point. There is no `class_weight` parameter on any
estimator, and that absence is a decision rather than a gap.

A per-class weight is a function of the label, which is to say it is already a
per-row weight. Giving it a second spelling would mean two weighting systems
that every estimator, every capability declaration, every artifact and every
validation order would have to agree about. One concept means one thing to
implement and one thing to freeze.

So the balanced rule is a recipe you write, not a string you pass:

```rust
use ferricml::data::{ClassTargets, SampleWeights};

let targets = ClassTargets::new(vec![0, 0, 0, 1])?;

// Inverse class frequency, scaled so the total weight stays the row count.
let n_classes = targets.classes().len() as f32;
let rows = targets.len() as f32;
let mut counts = vec![0_usize; targets.n_classes()];
for &label in targets.as_slice() {
    counts[targets.class_index(label).expect("label was observed")] += 1;
}

let weights = SampleWeights::new(
    targets
        .as_slice()
        .iter()
        .map(|&label| {
            let index = targets.class_index(label).expect("label was observed");
            rows / (n_classes * counts[index] as f32)
        })
        .collect(),
)?;

// The minority row carries three times the weight of a majority row.
assert_eq!(weights.as_slice(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0]);
assert!((weights.total() - 4.0).abs() < 1e-6);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A caller wanting a different balancing rule writes a different closure, instead
of waiting for another accepted parameter value.

An important consequence, stated in [frozen reference
semantics](../reference-semantics.md): a tree's `min_samples_split` and
`min_samples_leaf` bound the node's total *weight*, not its row count. That is
what makes an integer sample weight the same fitted model as repeating the row
that many times, unconditionally. A weight of exactly zero removes the row from
the training sample entirely, including from a forest's bootstrap draw.

## Which estimators accept weights

Not all of them, and the difference is meaningful rather than incidental.
`MinMaxScaler` and `MaxAbsScaler` fit order statistics — a minimum and a
maximum — which no per-sample weight can move, so they declare no weighted entry
point rather than offering one that would silently do nothing. Whether a given
estimator supports weighted fitting is declared in its `Capabilities`, which is
part of the public API and change-detected by its own snapshot.

See [API and growth](../api-and-growth.md) for what `Capabilities` records and
why it records only what varies.
