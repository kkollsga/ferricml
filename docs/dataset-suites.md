# Synthetic dataset suites

FerricML generates the data it measures itself against. `ferricml::datasets`
turns a validated `Recipe` into a `Dataset` — a design matrix, whatever task was
drawn over it, and the `Truth` behind that task — and this page is about the two
catalogues built on top of it:

- **`AccuracySuite`** is every task family as one small, clean problem whose
  correct answer is recorded. It answers *is the fit right*.
- **`PerformanceGrid`** is every task family at every point of a rows × columns
  sweep. It answers *what does generation cost*.

Both live behind the non-default `datasets` feature:

```toml
[dependencies]
ferricml = { version = "0.2", features = ["datasets"] }
```

> `default = []` is a product boundary in this crate. A consumer fitting models
> pays nothing for a generator it never calls, and the feature carries no
> dependency of its own — the streams come from the crate's existing generator
> kernels and the spec digest from the `sha2` already present for artifact
> checksums.

## Why suites, rather than a page of examples

Anyone can write a sweep over the families by hand, once. The reason FerricML
ships one instead is that a hand-written sweep decays in exactly one direction:
the crate gains a family, nobody remembers the sweep, and the sweep keeps
passing while covering less of the generator than its title claims. That is not
a failure anybody sees, because nothing about it looks like a failure.

So the catalogues live beside the families and are checked against them.
`Family` is the vocabulary — a task family with its parameters removed — and
`Family::ALL` is the roster the suites are held to:

```rust
use ferricml::datasets::{AccuracySuite, Family};

let cases = AccuracySuite::cases();
assert_eq!(cases.len(), Family::COUNT);

// Case order is roster order, so two runs read the same family at the same
// index.
let names: Vec<&str> = cases.iter().map(|case| case.name()).collect();
assert_eq!(names[0], "linear-regression");
assert_eq!(names[Family::COUNT - 1], "ranking");
```

## What closes the loop

Adding a family without adding a case has to be *loud*. Four separate things
fail before such a change can reach a reader, and the first three are the
compiler:

1. A new `Task` variant does not compile until `Task::family` names the family
   it joins.
2. A new `Family` variant does not compile until the crate-internal
   declaration-order walk places it.
3. Placing it moves `Family::COUNT`, which is the declared length of
   `Family::ALL`, so the roster literal stops matching its own type.
4. Only then does anything run — and `every_family_has_an_accuracy_case` in
   `src/datasets/suite_tests.rs` fails by name, because both `cases` functions
   are written-out tables rather than a `match` over the roster.

Step 4 is deliberate. A `match` over `Family` would make the suites span every
family *by construction* and the test incapable of failing, which would hide the
only question worth asking: whether somebody thought about what a meaningful
case for the new family looks like.

> The one gap left is a family declared to follow nothing while another family
> already does — an actively wrong total order rather than an omission. Rust
> cannot enumerate an enum's variants, so a roster is data in the end; what the
> chain above buys is that every *omission* is caught and only a deliberate
> mis-declaration is not. It is written down rather than papered over.

## The accuracy suite

Ten cases at 256 × 8, low noise, no contamination. What "right" means differs by
family, and each case's `Truth` is what carries it:

| Case | Family records | Why this case |
| --- | --- | --- |
| `linear-regression` | drawn coefficients | the case a least-squares path has no excuse on |
| `nonlinear-regression` | a conditional mean | an interaction no linear model reproduces |
| `glm-regression` | coefficients and `E[y \| x]` | an exactly Poisson count response at dispersion one |
| `ill-conditioned` | coefficients, at full rank | condition number 100: a stable solver against a normal-equations one |
| `linear-binary` | the Bayes probability | prevalence 0.3, so a majority-class predictor is visible |
| `nonlinear-binary` | the Bayes probability | exclusive-or, which a linear classifier cannot represent |
| `multiclass` | the whole probability row | four balanced blobs, so confusions come from the geometry |
| `clustered` | assignment and centres | the family with no target at all |
| `time-ordered` | both ends of a drifting predictor | drift 1.0, large against the noise and therefore measurable |
| `ranking` | the utility behind every grade | four grades over eight documents, so pairs tie as judgements do |

Because the answer is recorded, "how good is the fit" is a question about
correctness rather than about agreement with another implementation:

```rust
use ferricml::datasets::{AccuracySuite, Family, Target};
use ferricml::linear_model::{LinearRegression, LinearRegressionParams};

let case = AccuracySuite::cases()
    .into_iter()
    .find(|case| case.family() == Family::LinearRegression)
    .expect("the suite spans every family");
let dataset = case.generate();

let Some(Target::Regression(targets)) = dataset.target() else {
    unreachable!("a linear regression case draws a continuous target")
};
let fit = LinearRegression::fit(
    &dataset.features().as_view(),
    targets,
    LinearRegressionParams::default(),
)?;

// Not "close to another library's answer" — close to the coefficients the
// target was actually drawn from.
let truth = dataset
    .truth()
    .coefficients()
    .expect("a linear family records its beta");
let worst = fit
    .coefficients()
    .iter()
    .zip(truth)
    .map(|(fitted, drawn)| (fitted - drawn).abs())
    .fold(0.0_f32, f32::max);
assert!(worst < 0.02, "worst coefficient error was {worst}");
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Contamination is not in the suite

`Contamination` composes with every family — label noise, outliers, heavy tails,
heteroscedasticity, duplicated rows, constant columns, collinear pairs, a
per-column scale spread — so a robustness sweep is the cross product of a suite
with a contamination ladder, not a third table. Take a case's recipe and add the
knob:

```rust
use ferricml::datasets::{AccuracySuite, Contamination, Family};

let case = AccuracySuite::cases()
    .into_iter()
    .find(|case| case.family() == Family::LinearRegression)
    .expect("the suite spans every family");

let contaminated = case
    .recipe()
    .with_contamination(Contamination::none().with_outlier_fraction(0.05))?;

// The contamination is part of the recipe's identity, so the two datasets can
// never be confused for each other in a cache or a record.
assert_ne!(contaminated.spec_digest(), case.recipe().spec_digest());

// And it is an overlay rather than a reseed: the clean draw is still there
// underneath, displaced only where the knob says.
let clean = case.generate();
let dirty = contaminated.generate();
assert_eq!(clean.features().as_slice(), dirty.features().as_slice());
# Ok::<(), ferricml::datasets::DatasetError>(())
```

A knob the current family cannot carry is refused at the constructor with a
typed error rather than silently ignored, because a sweep reporting a model
robust to a contamination it never received would be worse than a build failure.

## Determinism: half the suite is per-runner, and says so

Every *source* in `ferricml::datasets` is transcendental-free and therefore
bit-exact on every target. Most task *families* are not: a Bayes probability is
a logistic function, a log-link mean is an exponential, a softmax is a sum of
exponentials, and a requested condition number is a real power. No libm rounds
any of those correctly, so their last bits are an implementation choice.

`Portability` is that distinction as a value rather than a paragraph, and a
suite spanning every family has to report it rather than average it away:

```rust
use ferricml::datasets::{AccuracySuite, Portability};

let per_runner: Vec<&str> = AccuracySuite::cases()
    .iter()
    .filter(|case| case.portability() == Portability::PerRunner)
    .map(|case| case.name())
    .collect();

assert_eq!(
    per_runner,
    [
        "glm-regression",
        "ill-conditioned",
        "linear-binary",
        "nonlinear-binary",
        "multiclass",
    ],
);
```

The other five reproduce byte for byte anywhere. Both halves are held to
matching evidence: a bit-exact family is pinned by literal values, a per-runner
one by properties and tolerances, because a pinned literal would be a promise
this crate cannot keep across a libm change it does not control.

> The list above is pinned in `src/datasets/suite_tests.rs` as well as written
> here. A family changing its envelope is a real event — a link replaced by a
> rational approximation, or a bit-exact family acquiring a transcendental — and
> it has to move the test and this page together.

## Handing the data to another language

A per-runner recipe is not enough to give another language the same problem, and
reimplementing the generator there would be a second thing to keep
byte-identical by hand. So the *file* is the boundary: generate once, write it
down, and let every consumer read the same bytes.

A container is two files sharing a stem. `<name>.manifest.json` is text — the
recipe in full, its spec digest, the determinism envelope, and a table of
`{name, dtype, rows, columns, byte_offset, len}` — and `<name>.bin` is those
arrays concatenated little-endian: `f32` for features, targets and truth, `u8`
for labels, `u64` for groups and indices. Nothing in the array file needs
parsing, which is why it carries no header: the pair opens with `json.load` and
`numpy.memmap` and needs no FerricML code at all.

```rust
use ferricml::datasets::{AccuracySuite, CacheOutcome, DatasetExchange};

let exchange = DatasetExchange::new(std::env::temp_dir().join("ferricml-docs-exchange"));
let case = AccuracySuite::cases()[0];

let (container, _) = exchange.ensure("accuracy_linear-regression", &case.recipe())?;

// Every array is named and shaped, so a reader slices the file without
// knowing which family produced it.
let features = container.array("features").expect("every container has a design");
assert_eq!((features.rows(), features.columns()), (256, 8));
assert_eq!(features.dtype().label(), "f32");

// The answer travels with the data, which is the whole reason for the format.
assert!(container.array("truth_coefficients").is_some());

// Asking again for the same recipe is a file read.
let (again, outcome) = exchange.ensure("accuracy_linear-regression", &case.recipe())?;
assert_eq!(outcome, CacheOutcome::Reused);
assert_eq!(again, container);
# Ok::<(), ferricml::datasets::ExchangeError>(())
```

The cache is keyed on the recipe's digest rather than on the name.
`DatasetExchange::ensure` reuses a container only when the recipe recorded in it
is the recipe being asked for, so a repeated request costs a file read and a
changed knob regenerates under the same name — the failure a name-keyed cache
would have is handing back the previous problem.

`ferricml-datagen` is the same thing from a shell. It builds only with
`--features datasets`, takes its catalogue from the two suites on this page, and
reports for each entry whether it generated or reused:

```text
$ cargo run --release --features datasets --bin ferricml-datagen -- --out data --suite all
accuracy_linear-regression	generated	ad78ef4e…	10276 bytes
```

> A container is untrusted input, and `src/datasets/exchange.rs` reads it the
> way `src/artifact/` reads a model. The recipe is checked against its recorded
> digest, so editing it cannot quietly redefine what the data is; the array file
> is checked against its own; the table has to describe the file exactly; and no
> allocation is ever sized from a declared length before the bytes behind it are
> read. `tests/dataset_exchange.rs` measures that last one rather than asserting
> it, because the defect it guards against returns the *correct* error and only
> the cost is wrong.

## The performance grid

Nine grid points, ten families, ninety cases. Generation cost is not one number:
a source draws once per element, a linear family's target is a dot product per
row, a multiclass family's is a softmax over classes per row, a ranking family
sorts within each query block, and the ill-conditioned family rescales and
duplicates whole columns. A sweep in one dimension would attribute all of that
to the wrong axis.

```rust
use ferricml::datasets::{Family, PerformanceGrid};

let cases = PerformanceGrid::cases();
assert_eq!(
    cases.len(),
    PerformanceGrid::ROWS.len() * PerformanceGrid::COLUMNS.len() * Family::COUNT,
);

// A recorded row is filed under its family's label and the point it sits on.
let first = cases[0];
let id = format!(
    "{}/{}x{}",
    first.name(),
    first.recipe().rows(),
    first.recipe().columns(),
);
assert_eq!(id, "linear-regression/256x8");

// The whole sweep runs through one buffer: `design_into` clears and refills it,
// so generating n designs costs n fills rather than n allocations.
let mut design = Vec::new();
let mut cells = 0_usize;
for case in cases
    .iter()
    .filter(|case| case.recipe().rows() == PerformanceGrid::ROWS[0])
{
    case.recipe().design_into(&mut design);
    cells += design.len();
}
assert_eq!(cells, 256 * (8 + 32 + 128) * Family::COUNT);
```

The grid exists to be measured, not asserted. FerricML's performance protocol
records numbers on a registered runner against its own history, so what belongs
in a test is that every case exists and is valid; the throughput itself belongs
in `dev-docs/bench/results/`, and the claim it supports is that generation is
negligible against the smallest fit it feeds.

## Adding a family

1. Add the `Task` variant, its validation, and its drawing.
2. Add the `Family` variant, place it in the declaration order, and give it a
   label. The label is what recorded results are filed under, so renaming one
   later orphans every historical record that named it.
3. Add the roster entry.
4. Add an accuracy case — small, clean, with a recoverable answer — and a grid
   row.
5. If the family evaluates a transcendental, say so in `Task::portability`, and
   move both the pinned list in `src/datasets/suite_tests.rs` and the list on
   this page.

Steps 1 to 3 are enforced by the compiler and step 4 by a test. Step 5 is the
one a reviewer has to read, which is why it is last and why the envelope is
pinned in two places.
