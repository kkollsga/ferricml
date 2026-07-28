# Changelog

All notable changes to FerricML are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A synthetic dataset generator, behind the new non-default `datasets`
  feature.** `ferricml::datasets` turns a validated `Recipe` into a `Dataset`:
  a design matrix, whatever task was drawn over it, and — the part no generator
  in this repository had — the `Truth` behind that task. Nothing is enabled by
  default and nothing new enters the dependency graph; the streams come from the
  crate's existing generator kernels and the spec digest from the `sha2` already
  present for artifact checksums.

  This first piece is the kernel: the three deterministic sources
  (`Source::Sampled`, `Source::Lattice`, `Source::Xorshift32`), the `Recipe`
  that validates a shape and a source *before* allocating anything, the
  `Dataset`/`Truth` containers, and `Recipe::spec_digest`. Task families arrive
  next; a recipe with none produces a design matrix and says
  `Truth::DesignOnly` rather than implying a correct answer it does not have.

  **A recipe is the whole identity of its data.** Every source is
  transcendental-free — integer arithmetic and exact or correctly-rounded
  conversions — so the same recipe gives the same bytes on a rerun, in another
  process, and on another machine. The cross-process half of that is a test
  rather than a claim: `tests/dataset_generator.rs` re-executes the test binary
  and compares `f32::to_bits` of every generated value, which is the only way to
  catch a generator that took anything from process state.

  **A dataset seed is disjoint from an estimator seed, by construction.** A
  design matrix drawn from seed `s` must not walk the sequence a forest fitted
  with seed `s` walks, or the data is correlated with the model's own
  randomness. `Recipe::seeded` mixes the caller's number through a derivation
  that lives beside the generator it derives from, and the disjointness is
  asserted against pinned probes: the derivation's mixer is a bijection, so
  *exactly one* tree index and *exactly one* repetition index reach the dataset
  state, and both are past `10^19`.

  **Validation is at the constructor, before allocation.** `Recipe::new` and
  `Recipe::seeded` refuse an empty or unrepresentable shape, a zero xorshift
  state — zero is that generator's fixed point, so the design would be one
  repeated value — and a lattice modulus outside `2..=2^24`, above which
  distinct residues collapse onto the same `f32`. A `Recipe` that exists
  describes data that can be produced, which is why `generate` and `design`
  return a value rather than a `Result`.

- **The frozen conformance lanes are now part of the generator.**
  `datasets::ReferenceQuality` reproduces the design matrices and targets
  FerricML's reference suite is recorded against — a nonlinear, a separable, an
  imbalanced and a noisy binary lane, plus a regression lane, at the recorded
  seeds — and the private lane functions in `tests/reference_semantics.rs` are
  gone. The test crate now consumes the same generator a downstream caller can.

  **Ported by value, not by resemblance.** The lanes feed quality tests that
  compare aggregate accuracy and Brier against the reference within `0.02`, so a
  generator emitting a *different but similarly distributed* stream would pass
  every one of them while changing every design matrix. The outgoing values were
  therefore captured first: `the_absorbed_lanes_reproduce_their_recorded_values`
  pins each lane's design head, its label prevalence, and a fold over all 1152
  labels or targets, at all five seeds, against literals read out of the
  functions this commit deletes. `tests/fixtures/reference_semantics_v1.rs` is
  unchanged, and the frozen-stream test now compares the crate's generator and
  the test crate's against each other as well as against the recorded literals.

  These presets name a **raw** stream state rather than routing through
  `Recipe::seeded`'s derivation, because that is what they were recorded
  against; the derivation stays the right default for new recipes.

  A dataset with targets and no recorded ground truth is a third thing, and
  `Truth::Unrecorded` is how it says so. The absorbed lanes were written to
  compare two implementations against each other, so no coefficient vector or
  noise-free target was ever kept — reporting `Truth::DesignOnly` would deny
  that a task exists, and inventing a coefficient vector would claim more than
  the lanes support.

- **The benchmark fixtures are now part of the generator too.**
  `datasets::BenchmarkFixture` reproduces the three private `fixture` functions
  FerricML's own benchmark suite used — the forest suite's lattice design with
  its separable labels and the regression target derived from them, and the
  xorshift32 designs and targets behind the model and boosting suites — at any
  shape. The originals in `benches/` are gone, and the benches call the
  generator.

  **Byte identity was the whole requirement.** `bench-history` gates a release
  against immutable per-release results at a `1.10` ratio limit, and a timing
  lane cannot notice a changed fixture: a differently distributed design of the
  same shape runs at very nearly the same speed while meaning something else, so
  every historical baseline would silently become non-comparable. The outgoing
  bytes were therefore captured first, at every shape the benches actually call:
  `the_absorbed_benchmark_fixtures_reproduce_their_recorded_bytes` pins a
  SHA-256 over each design's full value vector and over each target vector, ten
  `(lane, shape)` pairs in all, against digests read out of the functions this
  commit deletes.

  Like the conformance presets, these are transcriptions rather than recipes:
  `f32` throughout, association order preserved, and a design narrower than a
  target expression's score columns still sums only the columns it has.

- **Regression and binary-classification task families, with recorded ground
  truth.** `datasets::Task` draws six families over any recipe's design: a
  linear regression with a known `β`, four nonlinear regression shapes, a
  generalized linear count or positive response with a known rate, an
  ill-conditioned design built to a requested condition number and rank, a
  logistic binary problem at a requested prevalence, and four nonlinear binary
  boundaries. **All four of those boundaries defeat a linear rule**, which is
  measured rather than asserted: the family's own instrument scores the best
  least-squares linear rule against each boundary's Bayes accuracy, and the
  smallest gap is `0.188`. `BinaryKind::Sinusoid` is named for the curve it
  draws — `x₂ = sin(2π x₁)`, one full period across the design's support — and
  is the boundary that reading forced into its present shape. Every one of them
  records what it knows through `Truth` —
  coefficients, an intercept, a per-row conditional mean, a per-row Bayes
  probability, an exact algebraic rank — which is what turns "where do two
  libraries disagree" into "which one is closer to right".

  `Truth` gained `coefficients`, `intercept`, `conditional_mean`,
  `probabilities` and `rank` accessors so a consumer can ask one question of any
  family, and five variants so a family that knows less says so with a variant
  rather than with an empty vector. A nonlinear shape reports its conditional
  mean and *no* coefficients, because none produce it.

  **A dial moves the difficulty; a structural field moves the problem.** Two
  recipes differing only in a dial draw from the same streams, so a ladder over
  one of them is a ladder over one problem rather than over a sequence of
  unrelated draws. `informative` and `rank` are dials, which is a claim about the
  implementation and is tested as one: widening `informative` leaves the
  coefficients of the columns that already mattered bit-identical and switches
  further ones on, and `rank` never reaches the coefficient draw at all.

  **Prevalence is a knob, not an outcome.** The binary families solve for the
  intercept by bisection so the mean Bayes probability equals the request
  exactly; the realized rate is then a binomial draw around it, within four
  standard deviations at every prevalence swept.

- **`datasets::Contamination`: label noise, outliers, heavy tails,
  heteroscedasticity, duplicated rows, constant columns, collinear pairs and a
  per-column scale spread.** Orthogonal to the task, so a robustness sweep holds
  the family fixed and moves the contamination.

  **A knob the current task cannot carry is refused, not ignored.** Label noise
  on a regression target, or a heavy tail on a count response, comes back as a
  typed error. A sweep that silently received clean data would have reported a
  model as robust to a contamination it never got.

  **Contamination is an overlay, not a reseed.** The families' auxiliary streams
  are seeded from a digest over the shape, source and task only, so switching a
  knob on changes exactly what the knob describes and leaves every other draw
  identical. Seeding them from the full recipe digest made a five-percent
  label-noise request flip fifty-six percent of the labels, because the clean and
  contaminated datasets were then two independent draws.

- **`datasets::WeightPattern`,** four deterministic per-row weight patterns for
  the `fit_weighted` surface, including a class-balancing one that gives each
  class the same total weight — which is what turns a controlled prevalence into
  a controlled imbalance experiment.

- **`datasets::Portability`, the determinism envelope as a value.** Every source
  in the module is transcendental-free and bit-exact everywhere. Most task
  families are not: a Bayes probability is a logistic, a log-link mean is an
  exponential, a requested condition number is a real power, and no libm rounds
  any of those correctly. `Task::portability` and `Recipe::portability` report
  which of the two statements a caller is entitled to, and the two envelopes
  carry different evidence — a bit-exact family is pinned by literal values, a
  per-runner one by properties and derived tolerances.

- **`Recipe::target_values` and `Recipe::target_values_into`,** the numeric view
  of a task's targets with a caller-owned form that reuses its buffer.

- **Multiclass and structural task families.** `datasets::Task` gained four
  more: `Multiclass` over a chosen balance and geometry, `Clustered`,
  `TimeOrdered` with a controlled drift, and `Ranking` over query blocks. Each
  records real ground truth rather than reaching for `Truth::Unrecorded` —
  `Truth::MulticlassBayes` carries the whole probability *row* of every
  observation, `Truth::ClusterAssignment` the assignment and the centres,
  `Truth::DriftingPredictor` both ends of the coefficient vector and every row's
  time, `Truth::RankingUtility` the utility behind every relevance grade — with
  `class_probabilities`, `classes`, `cluster_assignments`, `cluster_centres`,
  `blobs`, `start_coefficients`, `end_coefficients`, `times`, `utilities` and
  `grades` accessors to read them.

  **Balance and geometry are two knobs, not one.** `ClassBalance` is how often
  each class occurs — solved for through the softmax offsets, so the requested
  marginal is a property of the *correct answer* rather than an outcome, in
  exactly the sense the binary family's prevalence already was. `ClassGeometry`
  is which classes are confusable with which: blob centres confuse whichever
  pairs land near each other, while a hierarchy's expected score gap to a cousin
  is exactly twice its gap to a sibling, so the confusion is nested. Realized
  class rates land within four binomial deviations of the request at every
  swept case, and the offset solver's own residual is six orders of magnitude
  smaller than that.

  **A clustered dataset has no target, and says so.** `Dataset::target` is an
  `Option` for this family: an unsupervised problem has no target vector, and
  handing back an empty or all-zero one would claim otherwise. The assignment is
  still recorded, and is recoverable from the design by nearest centre wherever
  the requested spread leaves the clusters separated.

  **Drift is measurable, not merely present.** Row order is time order and every
  row's time is recorded, so `TimeSeriesSplit` is correct on this data
  unadapted. Because both ends of the coefficient vector are recorded, a fit
  over any window predicts a value the record names in advance: fitting the
  first and last quarters recovers the drift between them to within `0.03` on a
  signal of `0.75`, and a `drift = 0` recipe measures as stationary under the
  same bound.

- **`datasets::GroupPattern` and `Recipe::with_groups`.** Three deterministic
  groupings — round-robin, contiguous, and contiguous with linearly falling
  sizes — that **partition** the rows: one label per row, exactly `0..groups`
  with none unused, and no group empty. `Dataset::groups` is `&[u64]` because
  that is what `GroupKFold::split` and `GroupShuffleSplit::split` take, so a
  generated dataset feeds them with no adapter between. Sizes interpolate
  linearly rather than geometrically so a grouping stays transcendental-free and
  cannot weaken a bit-exact recipe's determinism envelope.

- **`Dataset::pairs`, preference pairs in the crate's own vocabulary.** The
  ranking family emits `ranking::PairwiseObservation`s directly, so they go into
  `PairwiseLinearRanker::fit` unchanged; the pairs are carried on the dataset
  rather than derived at the call site, because which pairs exist and which way
  each points is what the family *produced*. Every pair lies inside one query
  and its outcome agrees with the recorded utility order exactly, so the data is
  separable by the recorded coefficients by construction.

- **`datasets::Family`, `AccuracySuite` and `PerformanceGrid`: two catalogues
  that span the generator.** `Family` is a task family with its parameters
  removed, `Task::family` is the projection, and `Family::ALL` is the roster.
  `AccuracySuite::cases` is every family as one small, clean problem at `256x8`
  whose answer is recorded; `PerformanceGrid::cases` is every family at every
  point of a `256/1024/4096` × `8/32/128` sweep, because a source draws per
  element, a linear target costs a dot product per row, a ranking family sorts
  within each query block, and a one-dimensional sweep would attribute all of
  that to the wrong axis. Both hand back `SuiteCase`s carrying a `Recipe`, so a
  robustness run is a suite crossed with a `Contamination` ladder rather than a
  third table.

  **A family added without a case is loud.** Four things fail before such a
  change reaches a reader, and the first three are the compiler: a new `Task`
  variant does not compile until `Task::family` names its family; a new `Family`
  variant does not compile until the crate-internal declaration-order walk places
  it; placing it moves `Family::COUNT`, which is the declared length of
  `Family::ALL`, so the roster literal stops matching its own type. Only then
  does anything run, and `every_family_has_an_accuracy_case` fails by name —
  because both `cases` functions are written-out tables rather than a `match`
  over the roster, which would have made the test incapable of failing. Verified
  by planting each link in turn, including the eleventh-variant case.

  **Half the accuracy suite is `Portability::PerRunner`, and says which half.**
  The GLM, ill-conditioned, linear-binary, nonlinear-binary and multiclass cases
  evaluate a transcendental; the other five reproduce byte for byte anywhere.
  The split is pinned in a test as well as documented, so a family changing its
  envelope has to move both.

- **`docs/dataset-suites.md`,** the narrative page for the above, wired into the
  documentation site's navigation and into the `doc_pages` doctest carrier — so
  its Rust samples are compiled and executed by `cargo test` like every other
  sample FerricML publishes, rather than being illustrations.

- **An exchange container, so the *file* is the cross-language boundary.**
  `datasets::DatasetExchange` writes a recipe to `<name>.manifest.json` plus
  `<name>.bin` and reads the pair back as a `MaterializedDataset`. The manifest
  is text — the recipe in full, its spec digest, the determinism envelope, and a
  table of `{name, dtype, rows, columns, byte_offset, len}` — and the array file
  is those arrays concatenated little-endian, `f32` for features, targets and
  truth, `u8` for labels, `u64` for groups and indices. It carries no header of
  its own, because the point is that `json.load` and `numpy.memmap` are enough
  to read it.

  **Why a file rather than a recipe.** Most families evaluate a transcendental,
  so their bytes are `Portability::PerRunner` and a recipe alone cannot hand the
  same problem to another machine or another language. Generating once and
  shipping the bytes closes that, and `MaterializedDataset::portability` travels
  with them so a reader knows which of the two statements it holds.

  **The cache is the digest, not the name.** `DatasetExchange::ensure` reuses a
  container only when the recipe recorded inside it is the recipe being asked
  for, so a repeated request is a file read and a changed knob regenerates under
  the same name. A name-keyed cache would have handed back the previous problem.

  **Read the way a model artifact is read.** The recipe is checked against its
  recorded digest — an edited recipe still hashes to something, and what makes
  the edit visible is that it no longer hashes to the value beside it — the
  array file against its own digest, and the array table has to describe the
  file exactly: contiguous, in order, uniquely named, ending on the last byte.
  Above all, **no allocation is sized from a declared length before the bytes
  behind it are read.** That is the Sprint R defect class, where a 148-byte
  artifact reserved 32 MB while returning the correct error, so
  `tests/dataset_exchange.rs` measures peak allocation rather than asserting a
  refusal: six hostile array tables, each declaring four billion `f32` values,
  each refused inside a budget of 4 KiB plus three times the container's length,
  with a control that proves the meter still fires.

  No new dependency: there is no `serde` and no JSON library in the graph, so
  the manifest is written and parsed by hand in `src/datasets/manifest.rs`. The
  reader accepts exactly the schema the writer emits, in exactly its order, and
  borrows every string out of the text rather than copying it.

- **`ferricml-datagen`, the exchange writer as a command.** A `[[bin]]` with
  `required-features = ["datasets"]`, so it does not exist when the generator
  does not. Its catalogue is `AccuracySuite` and `PerformanceGrid` rather than a
  list of its own, so a family added to the crate appears in the tool without
  anyone remembering to add it; `--list` prints the catalogue with each entry's
  digest, and a run reports for every entry whether it generated or reused.

- **A Python reader, and the end of the hand-mirrored generator.**
  `python/ferricml_datasets` opens a container with NumPy and nothing else:
  `load()` maps every array with `numpy.memmap` at the offset the manifest's
  table names, so opening one costs the manifest text and a handful of `mmap`
  calls whatever the arrays weigh, and `generate()` is the same with a
  `cargo run --release … ferricml-datagen` behind it when the container is not
  there yet. Truth is exposed as arrays under its short names, so what the
  answer *was* travels with the data into the other language.

  It is committed to the repository and excluded from the crate archive
  (`/python` in `package.exclude`, which `scripts/check_packaged_crate.sh`
  already reads rather than repeats): a Rust consumer cannot call it and
  `cargo` cannot build it.

  **What it replaced was a real duplicate.** FerricML's local conformance
  script carried a hand-mirrored SplitMix64 and a NumPy rewrite of all five
  frozen quality lanes, kept byte-identical to the Rust original by inspection.
  Nothing checked that pairing, and the lanes it fed compare aggregate accuracy
  and Brier within `0.02` — so a mirror off by one rounding step would have
  emitted a different but similarly distributed design, passed every check, and
  silently moved the data behind all 35 frozen reference rows. The mirror was
  compared against the materialized containers before deletion and agreed on all
  50 of them, bit pattern for bit pattern; regenerating the fixture from
  containers reproduced `tests/fixtures/reference_semantics_v1.rs` byte for byte.

- **A container now says whether its arrays are its recipe's output.** Reading
  the frozen lanes needed something the format did not have. A
  `ReferenceQuality` split is not `Recipe::generate`'s output — the preset builds
  one 1152-row design, slices it, and draws the lane's own targets over the slice
  — and both halves record the digest of the recipe they were cut from, so the
  digest cannot tell a training split apart from the design it came out of.

  `MaterializedDataset::derived` materializes an arbitrary `(Recipe, Dataset)`
  pair, and the manifest gains a `payload` block carrying `Payload::Generated`
  or `Payload::Derived(Derivation)` with the derivation's identity — lane, seed,
  split. The reading side refuses to guess rather than assuming the common case:
  `MaterializedDataset::regenerate` returns the new
  `ExchangeError::NotRegenerable` for a derived container instead of producing
  the recipe's output under the derived container's digest, and
  `DatasetExchange::ensure` refuses one rather than serving it as a cache hit or
  overwriting it. `Container.regenerable_recipe()` raises on the Python side for
  the same reason. `DatasetExchange::materialize_derived` and `ensure_derived`
  are the writing half; `ferricml-datagen --suite reference` is the catalogue.

  `ReferenceLane` gained `ALL` and `label()`, `Split` and its two variants are
  public, and `ReferenceQuality` gained `SEEDS` and `split()`. `label()` returns
  exactly the strings `QUALITY_REFERENCES` keys its rows on, so the fixture key,
  the manifest field and the container's file name are now provably one string.

### Changed

- **The exchange container format is version `2`.** The `payload` block is
  required rather than optional: a version-1 container has no way to say what
  its arrays are, so a reader meeting one would have to assume they are its
  recipe's output — which is the assumption the block exists to prevent.
  Nothing outside this crate has written a version-1 container; the feature is
  unreleased.

- **`Recipe::spec_digest` is now `ferricml.dataset.spec.v4`.** The encoding
  gained a task, a contamination, a weight pattern and a group pattern, each
  hashed under a discriminant that fixes the field layout so the encoding stays
  injective. The version moves whenever the encoding does, rather than fields
  being appended silently, which is what the domain tag exists for — and it
  moved once more without an encoding change, because the *data* behind an
  unchanged encoding moved when the task dials left the stream digest (below).
  One identifier must not name two datasets, in either direction. Nothing
  outside this crate has recorded an earlier digest — the feature is unreleased.

- **`WeightPattern::ClassBalanced` generalizes past two classes.** Each
  *observed* class now carries a total weight of `rows / classes`; at two
  classes that is the half-each it always was. The class count is the observed
  one rather than the requested one, because a class the draw never produced has
  no rows to weigh and inventing a share for it would down-weight every class
  that does exist.

- **The exact public-API snapshot now covers feature-gated surface.**
  `scripts/rust_api_profiles.py` captured one profile, the default one, and
  rejected a second by construction — so a public item behind a Cargo feature
  sat outside the snapshot contract entirely. It now captures an `all-features`
  profile alongside the default one, with its own frozen baseline, and its
  self-test fails if that baseline ever stops recording rows the default one
  does not.

### Fixed

- **A `Task` difficulty dial no longer redraws the problem it was supposed to
  make harder.** Every field of a task fed the digest the family's auxiliary
  streams are seeded from, so two recipes differing only in `separation` drew
  different coefficients: a difficulty sweep measured the gap between unrelated
  draws rather than the effect of the knob. Bayes accuracy over
  `Task::LinearBinary` at `20000 x 8`, seed `31`, read `0.6198 / 0.5543 /
  0.6707` across separation `0.9 / 1.0 / 1.1` — non-monotone in the one
  parameter whose whole purpose is monotonicity, with a step-to-step reversal
  six times the size of the knob's own effect across that interval. The same
  ladder now reads `0.5976 / 0.6076 / 0.6173`. `Task::Multiclass` behaved the
  same way and now climbs step by step as well.

  This is the defect `Contamination` was already excluded from that digest to
  avoid — a five-percent label-noise request had flipped fifty-six percent of
  the labels — and the task had simply been left on the other side of the
  exclusion.

  A task's fields are now partitioned by role. **Structural** fields —
  `informative`, `classes`, `blobs`, `queries`, `docs_per_query`, `grades`,
  `geometry`, `kind`, `link`, `rank` — change what is drawn and stay in the
  stream digest; two recipes differing in one of them are two different
  problems. **Dials** — `separation`, `prevalence`, `noise_scale`, `drift`,
  `spread`, `coefficient_scale`, `intercept`, `condition_number`, `dispersion`,
  `balance` — modulate a fixed draw and leave it. `Recipe::spec_digest` still
  moves for both, because the data does.

  The partition is compiler-enforced rather than maintained by hand: one
  encoding serves both digests and destructures every task variant with no rest
  pattern, so a new task field does not compile until it is classified. Every
  field of every family is swept as an assertion, and the sweep is held to the
  encoder's own field counts, so a field that is classified but never checked is
  a red test.

  **Generated data therefore moved for every recipe carrying a task.** The
  frozen reference lanes and the absorbed benchmark fixtures carry no task at
  all, which is asserted rather than assumed, so their streams and every pinned
  literal and digest they are held to are byte-unchanged.

## [0.2.1] - 2026-07-27

### Fixed

- **Breaking (fitted values). `LinearRegression` returns a least-squares
  solution on exactly rank-deficient designs, where it previously often did
  not.** The SVD this crate depended on returns a factorization that does not
  reconstruct its own input on such designs, and the coefficients derived from
  it were consequently not least squares in any sense: they did not zero the
  normal-equation gradient.

  **The defect, measured.** Tall designs with one exactly duplicated column,
  300 draws per shape, comparing each decomposition against its own input:
  55/300 failed to reconstruct at 64x3, 121/300 at 256x6, and 146/300 at
  1024x6, with a worst *relative* reconstruction error of 18.5 — not a rounding
  discrepancy but a wrong answer. The failures are quiet by construction. The
  factors come back orthonormal, the singular values look plausible, and the
  reported rank is correct on every draw measured, so nothing short of
  reconstructing the input or checking the gradient can see it.

  **What FerricML now asserts.** `least_squares.rs` sweeps that defect class —
  three shapes, 75 draws — and checks three independent properties per draw:
  the scaled normal-equation gradient vanishes, the rank is the true one, and
  the duplicated pair splits evenly. Only the gradient check catches this: run
  against the previous backend it fails on **27 of 75** draws with a worst
  scaled gradient of 5.4e-2 against a 1e-12 limit, while the rank and
  minimum-norm checks pass on all 75. A test that had asserted rank or
  even-splitting alone would have watched this ship.

  **The fix is the backend.** `nalgebra 0.34.2` is replaced by
  `faer 0.24.4`, which is correct on the same corpus: 0 failures in 3600
  decompositions, worst relative reconstruction error 2.7e-14. FerricML now has
  two runtime dependencies, `faer` and `sha2`, and there is one code path — no
  Cargo feature selects a backend and none is planned. `LinearRegression`'s
  minimum-norm promise is now a promise the crate keeps rather than one it
  states.

  **The reference fixtures do not move, and that was measured rather than
  hoped.** Instrumenting every numeric assertion in `tests/reference_semantics.rs`
  and running both backends single-threaded gives **446 deviations that are
  identical value for value**, worst case 91% of tolerance under each. No
  fixture was regenerated.

  **What did move is a guarantee, and it moved for a good reason.**
  `docs/determinism.md` had `LinearRegression` (artifact kind 2) and `Ridge`
  (kind 3) in tier 1, byte-identical on every IEEE-754 target. They are now
  **tier M — identical per machine**, a new named scope, because the new
  kernels select instructions from the CPU features they detect at run time and
  no target triple bounds that. Read the tier-1 listing as having been wrong
  rather than as having been true and then weakened: tier 1's evidence is a
  search of `src/` plus the accumulation policy, and no dense decomposition has
  ever lived in `src/` under either backend. The claim never covered the code
  that computed those two models. Refitting still reproduces the model, there is
  still no thread-count axis — the backend's parallelism is pinned to `Par::Seq`
  at every fit and its `rayon` feature is never enabled — and every predict path
  remains a hand-written row loop in `src/`, so inference is unaffected.

  Rule 2 of the accumulation policy in `src/numeric/mod.rs` gains a matching
  exception. It is scoped to the calls that enter the backend from
  `least_squares.rs` and to its **first** clause only: a blocked kernel reorders
  a reduction for speed by construction, while the second clause — no order that
  depends on how work was scheduled — stands unamended and is what the
  sequential pin buys. Turning the SVD back into coefficients is written out in
  ascending index order through `sum_in_order` rather than delegated, so the
  exempt region ends at the factorization.

  Two smaller things fall out. The centered design is now a plain
  `Vec<f64>` that FerricML owns, declared column-major with no padding, instead
  of a backend matrix type: `coordinate_descent` slices that buffer by column,
  a padded or transposed layout would silently fit a different model rather
  than fail to compile, and a new test pins the layout from both ends. And
  `docs/security.md` withdraws its claim that replacing the numerical backend
  "would carry disproportionate numerical and compatibility risk" — exactly
  backwards, as it turned out; `RUSTSEC-2024-0436`'s unmaintained `paste` is
  reached through the new backend too, so that conclusion survives on a
  different fact than the one it rested on.

- **Breaking (predicted class).** `DummyClassifier` picks its majority class by
  comparing the class **counts**, instead of the `f32` frequencies narrowed from
  them. `class_priors` is an `f32` view of a ratio of `usize` counts, and
  narrowing is not order-preserving: once the relative gap between two counts
  falls below half an ulp they collapse onto one `f32`, the strict `>` in the
  scan sees a tie, the tie rule sends it to the smaller label, and the estimator
  predicts the class that occurs *less* often.

  Measured, and searched rather than guessed: the first pair that collapses is
  `(16_777_216, 16_777_217)` at `33_554_433` rows, where both ratios round to
  exactly `0.5` — bit pattern `0x3f00_0000` for each. At every smaller
  power-of-two total, down to `1_048_577`, the same one-row majority is still
  strictly larger as an `f32`, so this is the first collapse and not one of
  many; from `33_554_433` rows onwards it is the whole regime. Fitted on that
  training set, `predict` returned class `0` for every row while class `1`
  occurred once more often.

  Counts are exact, so the comparison on them is exact. **The tie rule does not
  move**: the scan is still ascending with a strict `>`, so a genuine tie still
  resolves towards the smaller class label, and `class_priors` is unchanged in
  both value and type. One consequence is deliberate and now documented on both
  methods: at such a total the predicted label is no longer the first maximum of
  the reported probabilities, because the reported probabilities are equal where
  the counts are not. The prediction follows the counts.

- **Breaking (fitted values).** Histogram gradient boosting fits its bin grid
  over the rows that are in the training sample, instead of over every row it
  was handed. Both boosted estimators document that "an integer weight is the
  same fit as repeating that row that many times", and zero is an integer: the
  sibling families implement and test that case by name
  (`random_forest`'s `a_zero_weight_row_is_the_same_fit_as_a_deleted_row`,
  `tree`'s `a_zero_weight_row_is_absent_rather_than_present_with_no_influence`),
  and `tree::grower` states the rule as "a row of zero weight is not in the
  training sample at all". Boosting was the silent exception.

  The statistics already ignored a zero-weight row, so it influenced the fit
  through the grid and through nothing else: a row that was supposed to be
  absent still contributed a distinct feature value for a bin edge to land on.
  Measured on nine rows over one column with the row holding the unique value
  `4.0` zero-weighted, the zero-weighted fit and the deleted-row fit disagreed
  by up to **`1.35` on a target range of `0..20`**, at every `max_bins` from
  `3` to `16` — across both branches of the grid, the per-adjacent-pair one and
  the quantile one.

  The grid is still not *weighted*: a weight does not move a threshold, which
  is why weighting a row and repeating it agree. What the weights now decide is
  membership. Only a weighted fit containing a zero weight moves; unweighted
  fits and fits whose weights are all positive are bit-identical, and
  `make reference-check` is green without regeneration.

- `GroupKFold::split` documented that "group identifiers carry no order or
  meaning beyond equality", and `GroupShuffleSplit` restated it as
  "`GroupKFold`, whose assignment depends only on group sizes". Both were
  false: the size tie-break is `left.0.cmp(&right.0)`, the identifier itself,
  so equal-sized groups are ordered by name. Behaviour is unchanged — the
  documentation was the defect — and both pages now say that identifiers order
  equal-sized groups, so the partition survives any order-*preserving*
  renaming and a reversing one can move it.

  The test that named the property renamed with `g -> u32::MAX + 1000g`, which
  is strictly increasing and therefore left the tie-break reading the same
  order: it could not fail. Measured over 4,500 equal-sized-group
  configurations, an order-reversing rename moves 3,500 of them, the smallest
  being `[0, 1]` at two folds. The test now asserts the invariance over
  renamings that really preserve it and pins the reversing case as observable.

- `CalibratedClassifier` saturates its calibrated probabilities at the boundary
  that produces them, instead of forwarding whatever the calibrator returned.
  `Calibrator::calibrate` documents a result in `0.0..=1.0`, and the wrapper
  promised probabilities on top of it, but neither promise was enforced
  anywhere. The trait is open by design, and even the shipped
  `IsotonicRegression` implements it for *both* of its constructors while only
  `fit_calibration` averages `0`/`1` labels — one fitted through `Regressor`
  over unbounded targets is a regression surface, and `CalibratedClassifier::new`
  accepts it. Measured on such a composition, `predict_proba` returned rows like
  `[38.71, -37.71]`: **every one of 12 probability slots outside `0.0..=1.0`**,
  worst excess `37.7`, and `predict` reported class `0` for every row while the
  class-`1` column read `-37.7`.

  Nothing detected it because a row is written as `[1 - p, p]` and therefore
  sums to exactly `1.0` for any `p` at all, so the conformance battery's
  probability obligation — a row-sum check — passes on `[-48.0, 49.0]`. There
  was no per-slot bound assertion anywhere, and `CalibratedClassifier::new` was
  called by no test.

  The clamp sits in the one routine every probability path on the wrapper
  reaches, per rule 5 of the accumulation policy, so `predict_proba`,
  `predict_class_proba` and `predict` now agree on a bounded value. **No fitted
  value moves**: it is a no-op for `fit_isotonic` and `fit_platt`, which already
  produce probabilities in range, and that is asserted bit-for-bit rather than
  to a tolerance. `make reference-check` is green without regeneration.

- **Breaking (fitted values).** The multinomial Newton path supplies curvature
  for the whole subspace the softmax cannot see, instead of for the one member
  of it that no penalty reaches. Adding the same vector to every class's
  parameter row leaves every probability unchanged, so the loss curvature is
  exactly zero along one direction per parameter coordinate — feature
  coordinates included, and with or without an intercept. Only the intercept
  direction was supplied with curvature; the feature directions were left to the
  L2 penalty `1/(C * scale^2)`, which on a weakly regularized, badly scaled
  design is `1e-12` or smaller against a largest curvature of order the sample
  weight. Two user-visible defects followed, both measured over a 576-case
  ill-conditioned region:

  - `fit_multiclass` refused **191 of 576** designs with
    `ModelError::LinearSolveFailed`, because the Cholesky pivot in a direction
    the answer does not use was decided by rounding. It now refuses 13, and all
    563 fits land at a local minimum of the penalized objective in the caller's
    own feature space. The 13 that remain are a genuine collapse of the *data's*
    curvature — every one is twelve rows over four columns across five classes —
    and are left refusing.
  - **231 of the 385** designs that did fit were returned **uncentred**,
    violating the frozen semantic that a multinomial fit is one centred
    coefficient row and intercept per class with no pinned reference class. The
    worst was off by twice its own largest coefficient. Probabilities and
    predicted labels were unaffected — the softmax is shift-invariant — but
    `coefficients`, `intercepts` and `decision_function` reported a different
    representative from the one the contract names. Every fit is now centred to
    within `n_classes` `f32` ulps.

  No reference class is pinned and no semantic changed: the added blocks are
  orthogonal to the subspace the answer lives in, so in exact arithmetic they
  change no fitted value. Fitted values do move in floating point. On
  well-conditioned data they do not move at all — the frozen reference fixture's
  multinomial fit is bit-identical, and `make reference-check` is green without
  regeneration — but `n_iter` falls on 96 of the region's 385 previously-fitted
  designs as the better-conditioned solve converges sooner, and coefficients on
  ill-conditioned designs move to their centred representative.

- `ModelError::SolverDidNotConverge` renders as "solver stopped after `{n}`
  iterations without meeting tol", instead of "solver reached max_iter after
  `{n}` iterations without converging". The old sentence named a cause the
  variant cannot know. Measured on a 12x2 binary logistic fit under L-BFGS with
  `C = 0.1`, `tol = 1e-12` and `max_iter = 500`, it rendered as *"solver reached
  max_iter after 9 iterations without converging"* — a budget of five hundred
  with nine of them used, and the same shape as the 50000x50 report it was found
  on.

  The mechanism is that one variant has always carried two stopping causes.
  L-BFGS maps both an exhausted budget and a collapsed line-search bracket onto
  it, and the two Newton paths break out of their loop when backtracking finds
  no descending step, so three of the five construction sites can report far
  below their budget. A collapsed bracket is the observable form of a `tol`
  below the objective's own numerical resolution, near `1e-9` for a log-loss of
  order one, which makes raising `max_iter` — the one action the old sentence
  pointed at — the one action that cannot help.

  **No mapping and no refusal changed**: to a caller both causes mean the fit is
  not converged, which is why one variant reports both. What changed is the
  sentence and the field's documentation, which now state only what is true at
  every site. The causes stay separable from `iterations`, in one direction:
  it never exceeds the `max_iter` that was set, so `iterations < max_iter`
  proves the budget was not the constraint and the remedy is a looser `tol`,
  while equality leaves both possible. That is now documented on the variant,
  and `tests/solver_refusal_message.rs` renders a refusal at every construction
  site and fails on a message that names either cause.

- **The install snippet in the guide asks for the crate that exists.**
  `docs/guide/quickstart.md` still said `ferricml = "0.1"` after the 0.2.0
  release. Measured rather than reasoned about: a scratch crate carrying
  exactly that requirement resolves, against the live index, to
  `ferricml v0.1.2 (available: v0.2.0)` and builds without a warning. The
  mechanism is that below 1.0 Cargo treats the minor as the breaking
  component, so `"0.1"` is `>=0.1.0, <0.2.0` and cannot reach 0.2.0 at all.
  Nothing fails: the reader gets a silently older crate and then works through
  a page describing an API their build does not contain, and 0.2.0 carried
  breaking changes, so the two genuinely disagree. The requirement is now
  `"0.2"`.

  The string was the smaller half. `scripts/check_documentation_truth.py`
  gained a sixth rule, `documented-dependency-requirement-resolves`, which
  reads every documented `ferricml = "<req>"` — TOML block, dependency table,
  or `cargo add` line, in `docs/`, in rustdoc, and in `README.md` — and
  *evaluates* it the way Cargo does against the manifest's own version rather
  than string-matching the current one. It fires in both directions, on a
  requirement that excludes the released crate and on one naming a release
  that does not exist yet, and it reports rather than passes when it loses its
  input: an empty or missing page, a page with no snippet left in it, and a
  missing or unreadable manifest are each a finding. Twelve accepted spellings
  are proven accepted in `--self-test`, because a rule that rejects everything
  would also have caught this.

### Added

- **A declared minimum supported Rust version: `rust-version = "1.88"`**, plus
  an `msrv` CI lane that builds and tests on exactly that toolchain. The floor
  was always real; it was invisible. A consumer on an older toolchain got a wall
  of edition-2024 parse errors rather than Cargo's "requires rustc 1.88", and
  nothing stopped a merged patch from raising it silently.

  **Measured, not assumed**, by installing each candidate and building against
  the committed `Cargo.lock`. 1.85 and 1.86 do not resolve at all: `nalgebra
  0.34.2` declares `rust-version = "1.87.0"`, the highest in the locked graph.
  1.87 resolves and then fails to compile FerricML itself, with **16 instances
  of `error[E0658]: let expressions in this position are unstable` over 15
  sites** in `src/tree`, `src/linear_model`, `src/ensemble` and
  `src/preprocessing` — `let` chains stabilized in 1.88 on edition 2024. So the
  crate's own source sets the floor, one release above what its dependencies
  ask for and three above what `edition = "2024"` alone would allow.

  **Build and test agree on 1.88; clippy does not, and the lane does not run
  it.** `clippy::nonminimal_bool` reports
  `src/preprocessing/robust_scaler/mod.rs:103` on 1.88, 1.90 and 1.92 and stops
  reporting it on 1.97. Lint output is not a compatibility contract, and pinning
  the floor to whichever clippy is quietest would raise the version a consumer
  needs for a reason no consumer can observe.

  **The lane is demonstrated non-vacuous rather than assumed to work.** It reads
  the floor out of `Cargo.toml` instead of keeping a second copy, because a lane
  with its own hardcoded version passes happily after someone raises
  `rust-version` — the one thing it exists to catch. It was then shown to fail
  in both directions: declaring `1.87` reproduced the 15-site `E0658` wall, and
  adding an inferred array length (`[u8; _]`, unstable on 1.88 and stable by
  1.90) failed the lane with `E0658: using _ for array lengths is unstable`
  **while building cleanly on stable** — a floor raise that every existing gate
  would have passed.

- `PolynomialFeatures`, the first transformer whose output is wider than its
  input. It claims `degree`, `interaction_only` and `include_bias`, persists as
  artifact kind `46`, and composes as a `StagedPipeline` stage.

  **The output width is public contract**: `C(n + d, d)` for the full
  expansion, `sum over k in 0..=d of C(n, k)` with `interaction_only`, less one
  where the bias column is disabled. It is evaluated in checked arithmetic at
  *fit* time, before the expansion's term table is reserved, so a request that
  cannot be built returns `ModelError::FeatureExpansionOverflow` naming the
  feature count and the degree rather than attempting the allocation. Width is
  this transformer's failure mode rather than an unlikely edge of it: fifty
  features at degree ten is an unremarkable-looking request for seventy-five
  billion output columns, and the reference FerricML is measured against
  reports that number from `fit` without complaint.

  **Column order is frozen contract**: the bias column first, then blocks of
  ascending total degree, and within a block the lexicographic order of
  non-decreasing feature-index tuples — `1, x0, x1, x0^2, x0 x1, x1^2` and so
  on, with `interaction_only` taking the strictly increasing tuples of the same
  order. A caller that persists a downstream model against this expansion is
  relying on that order, so it is pinned by test rather than left to follow
  from how the terms happen to be generated.

  Not claimed, and recorded as such rather than as a divergence:
  `degree=(min_degree, max_degree)`, and the memory-order knob — FerricML's
  dense matrices are row-major by construction.

- Two `ModelError` variants for the above: `FeatureExpansionOverflow`, and
  `EmptyFeatureExpansion` for the one configuration that describes no columns
  at all, degree zero with the bias disabled.

### Changed

- `LinearRegression` and `Ridge` at `alpha = 0` reduce a tall design through the
  `R` of its thin QR before decomposing, rather than decomposing the design
  itself. **Fitted values do not move** — every `f32` coefficient and intercept
  is bit-identical across 62 fits from `8x3` to `400x300`, weighted and
  unweighted, intercept both ways, including both frozen reference designs — so
  this is recorded for its cost rather than for its result. Release build,
  sequential, medians of five interleaved runs on an 80-86% idle machine:
  `50000x50` 38.3 ms to **25.9 ms**, `1000x300` 23.2 ms to **17.1 ms**,
  `50000x300` 640.2 ms to **310.4 ms**.

  The reduction is exact rather than approximate, which is why the contract is
  unchanged: `A = QR` with `Q` an isometry, so `R` and `A` have identical
  singular values and the rank cutoff is applied to the same numbers, while
  `(QR)⁺ = R⁺Qᵀ` makes the minimum-norm least-squares solution of `Rx = Qᵀb`
  *be* the minimum-norm least-squares solution of `Ax = b`. Rank deficiency
  needs no special case.

  It is taken only where it pays. Break-even is at `rows/columns ~ 1.20`
  measured at `columns = 300` — 13.33 ms against 13.36 ms — and below it the
  reduction *costs*: 10% at `rows == columns`. The guard is
  `4 * rows >= 5 * columns`, integer so no rounding decides it, and set at 1.25
  rather than at break-even because at break-even there is nothing to save. One
  cost is recorded rather than hidden: break-even drifts with `columns`, sitting
  nearer 1.5 at `columns = 100`, so a small-`columns` design between 1.25 and
  1.5 pays up to 4% of 1.2 ms. A `columns`-dependent constant would buy that
  back and cost a second tuning parameter.

- `LogisticSolver` and the linear-models guide now state that the **default
  solver is the expensive one at scale**, with the measurement rather than the
  adjective. `LogisticSolver::Lbfgs` was documented only as the escape from
  `ModelError::MulticlassSystemTooLarge`, so a caller on a large binary problem
  had no reason to look at it. Newton accumulates and factorizes a
  `parameters x parameters` system over every row, so its per-iteration cost
  grows as `rows * parameters^2` where the matrix-free path grows as
  `rows * parameters`, and a smaller `C` buys more iterations to pay it on. At
  50,000 rows by 50 columns, same data, same `tol`, same `max_iter`, both arms
  this crate: `C = 1.0` is 68.3 ms against 16.6 ms at `tol = 1e-4` and 83.5 ms
  against 25.6 ms at `1e-8`; `C = 0.1` is 83.7 ms against 16.6 ms and
  **375.4 ms against 25.4 ms**, for coefficients agreeing to six decimals. The
  advantage is a property of the shape and not of the solver — at 5,000 x 20 it
  is 2.8 ms against 1.8 ms, and on small data the exact step's single-digit
  iteration count wins — so the default is unchanged and no fitted artifact
  moves.
- `StratifiedKFold` and `GroupKFold` now say that **fold membership is not a
  parity claim**, on the types and in the evaluation guide. What each promises
  is its defining property plus reproducibility from the stated inputs, not
  which fold a given row lands in: `StratifiedKFold` deals each class
  round-robin where another library assigns contiguous blocks of the same
  stratum, and `GroupKFold` states the tie-break between two equally light
  folds where another library leaves it unobservable. Both differences preserve
  everything claimed — measured against the comparison reference, per-fold
  positive rates agree to four decimals and `GroupKFold`'s fold sizes are
  identical (`[194, 210, 209, 193, 194]` on both sides) with no group straddling
  a split. No assignment changed; the sentence exists so a membership diff is
  not read as a defect, and the properties it names are the ones
  `stratified_folds_balance_every_class_and_global_size`,
  `no_group_appears_on_both_sides_of_any_split` and
  `folds_are_as_even_as_whole_groups_allow` already assert.
- `ExtraTreesClassifier` and the trees guide now record that an extra-trees
  member is a **higher-variance estimator, so a narrow `MaxFeatures` needs more
  members**. Measured on 2,000 training and 1,000 held-out rows over 10 columns
  with a nonlinear binary target, five data seeds, accuracy as the five-seed
  mean: at 50 members and `MaxFeatures::Sqrt`, ten values of `random_state`
  give mean 0.7716, sd 0.0036, range 0.7650–0.7782, against 0.7787 for
  `MaxFeatures::All` and 0.7790 for `RandomForestClassifier` at `Sqrt`. Holding
  `random_state` at its default `0`, the `Sqrt` lane climbs 0.7650 → 0.7704 →
  0.7764 → 0.7784 at 50, 100, 200 and 800 members. So the shortfall at 50
  members is under-averaging plus one seed draw — the default `random_state = 0`
  is the lowest of the ten measured — and not a biased split rule. Nothing in
  the fitted model changed.

## [0.2.0] - 2026-07-26

### Added

- `scripts/check_solver_registration.py`, run by `make gate`, which enforces
  that every params type exposing both `with_max_iter` and `with_tol` is named
  in `tests/solver_convergence_contract.rs`. That battery is the mechanism
  behind the claim below that *every* iterative solver refuses an exhausted
  budget, and it caught a listed solver regressing while saying nothing about a
  new one arriving unlisted — a budgeted estimator could ship with the claim
  unenforced for it and nothing anywhere reporting the omission. The rule is
  deliberately syntactic, so it cannot be satisfied by weakening a tolerance or
  by accepting an unconverged iterate; the only thing that satisfies it is a
  row, which then has to pass the battery on its own merits. Requiring *both*
  builders is what keeps the predicate honest: a budget without a convergence
  test is a fixed round count, as on the two boosted estimators, and a tolerance
  without a budget is a rank threshold inside a direct solve, as on
  `LinearRegressionParams` — neither has an iterate to refuse. Its `--self-test`
  proves each rule against a synthetic violation, proves two of them again
  against a violation in a child module a facade-only reader would miss, proves
  that losing either input is reported rather than passed, and reconstructs the
  omission it was written for.

- `scripts/check_accessor_pairing.py`, run by `make gate`, which enforces that
  an `X_into` method and its allocating `X` partner are actually a pair: the
  caller-owned form takes exactly the allocating form's arguments plus its
  output buffers, every inherent `_into` has an inherent allocating partner,
  and a type forwarding one form inherently forwards the other too. The
  contract was written down and unenforced, which is how a single-row method
  came to hold a batch method's name on five classifiers and get copied into a
  sixth. Its `--self-test` proves each rule against a synthetic violation,
  proves that losing its input is reported rather than passed, and reconstructs
  the four defects it was written for from the baseline rows they occupied.

- `linear_model::LogisticRegression::predict_class_proba_into` is now reachable
  as an inherent method, matching the allocating `predict_class_proba` forwarder
  it already had and the pair every other probabilistic classifier ships.
  Reaching the allocation-free form previously required importing
  `api::ProbabilisticClassifier` while the allocating one did not, which
  inverted the crate's stated preference on hot paths. Behaviour is unchanged:
  the inherent method delegates to the same trait implementation.

- `pipeline::Pipeline::fit`, which fits the transformer and then the estimator
  on what the transformer produced. The two pipeline types disagreed on this:
  `StagedPipeline` could compose *and* fit, `Pipeline` could only compose. The
  one-transformer case is not a special case — it is the shortest composition —
  and fitting the estimator on untransformed rows is the one handoff error that
  yields a silently wrong model rather than a width mismatch.
- `pipeline::TransformerStack` and `pipeline::PersistedStack` are now
  implemented for every flat tuple of fitted transformers from one stage to
  twelve, published as `pipeline::MAX_STAGES`; they previously stopped at two
  and three. `StagedPipeline::new`, width-handoff validation, the single split
  workspace, the capability declaration, and the tagged artifact therefore all
  work at any of those lengths. Nothing about a shipped composition changes:
  the impls are additive and `StagedPipeline<(A, B), E>` is the same type it
  was. A right-nested `(A, (B, (C,)))` stack was considered and rejected — it
  conflicts (`E0119`) with a flat impl rather than joining it, it would rewrite
  the type parameters of every shipped composition, and its nesting carries no
  meaning; a fixed ceiling over flat tuples is what the standard library does.
  One-call fitting stays bounded at two stages, with the two measured attempts
  to lift it recorded on `StagedPipeline::fit`.
- `linear_model::Lasso` and `linear_model::ElasticNet` now persist, under their
  own artifact kinds. They were the last tunable regressors that could be fitted
  but not saved, which is backwards for an L1 model: a sparse coefficient vector
  is chosen precisely because it is the thing worth shipping. Both artifacts
  store the mixing weight and the coordinate-descent sweep count alongside the
  coefficients rather than re-deriving either, because both are readable on a
  fitted model. Neither has an `api::AnyRegressor` variant yet; that remains a
  dispatch gap rather than a contract gap, and adding one later will not touch
  either estimator's bytes.
- `api::ModelError::InvalidTreeStructure`, for a fitted tree whose topology or
  values the packed node format cannot represent at any size. It is separate
  from `TreeTooLarge`, which is a size bound; see the note under Changed for
  what now reports it.
- `tree::DecisionTreeClassifier` and `tree::DecisionTreeRegressor`, standalone
  decision trees over the same grower a random forest uses. Both support
  weighted fitting and persist under new artifact kinds; the classifier fits
  binary or natively multiclass targets and declares genuine probabilities,
  because a leaf *is* a distribution over the training rows that reached it.
  `MaxFeatures` is now also reachable as `tree::MaxFeatures`; the existing
  `ensemble::MaxFeatures` path is unchanged and names the same type.
- `tree::Splitter`, set through `with_splitter` on either standalone tree's
  parameters. `Splitter::Best` (the default) evaluates every boundary between
  adjacent distinct values in each candidate column; `Splitter::Random` draws
  one threshold uniformly inside each candidate column's own range within the
  node and keeps the best-scoring draw, which is what makes an *extremely
  randomized* tree. The candidate columns are drawn identically either way, an
  inadmissible draw is discarded rather than redrawn, and a column that is
  constant within the node consumes no draw at all — so the generator's stream
  does not depend on which columns happen to be constant. Random forests are
  unaffected: they keep the exhaustive search and their artifact bytes are
  unchanged.
- `ensemble::ExtraTreesClassifier` and `ensemble::ExtraTreesRegressor`,
  extremely randomized tree ensembles over the same core a random forest uses.
  Each member draws one uniform threshold per candidate column instead of
  optimizing within it; the candidate columns themselves are drawn exactly as a
  random forest draws them. `bootstrap` therefore defaults to `false` here and
  stays `true` on a random forest — trees decorrelate through their thresholds,
  so resampling on top of that would only remove training rows. Both persist
  under new artifact kinds, fit with or without sample weights, and the
  classifier fits binary or natively multiclass targets. An ensemble of one
  member is bit-identical to the corresponding standalone tree at the same
  seed, which is asserted rather than assumed.
- A narrative documentation site, built with MkDocs from the markdown already
  under `docs/` and configured for Read the Docs. Seven new guide pages — a
  quickstart, data and targets, linear models, trees and forests, preprocessing
  and pipelines, calibration and inspection, and saving and loading models —
  sit in front of the existing contract documents rather than replacing them.
  The site is the narrative guide and deliberately reproduces no API listings:
  the symbol-level reference is rustdoc on docs.rs, which regenerates from the
  code and cannot drift from it. Site machinery lives outside `docs/` and is
  excluded from the published crate, so the archive gains only markdown.
- Rustdoc usage examples on every public estimator that lacked one, covering
  the data containers, all five linear models, all eight tree-based estimators,
  all seven transformers, both pipelines, both dummies, the calibrators, the
  pairwise ranker, permutation importance and the runtime dispatch layer. These
  are doctests, so each is a compiled and executing test of the exact call a
  caller reaches for; the suite goes from 13 to 97.
- Every Rust sample in the narrative documentation is also a doctest. The pages
  under `docs/` are compiled into the test suite under `cfg(doctest)`, so a
  sample that stops compiling — or stops producing the value it claims — fails
  the ordinary gate. `tests/doc_examples.rs` fails if a page is left out of
  that mechanism, and rejects a sample marked `ignore` or `no_run` that carries
  no written reason, so the difference between a verified sample and an
  illustrative one is always visible to a reader.


- `preprocessing::RobustScaler` and `RobustScalerParams`: per-feature scaling by
  a median and a quantile spread. Both statistics are order statistics, so a
  handful of extreme rows move them far less than they move a mean and a
  standard deviation. `with_quantile_range` selects the percentile pair whose
  difference is removed — the interquartile range by default — and
  `with_centering` / `with_scaling` select which statistic the transform
  removes. Quantiles use linear interpolation between the two bracketing order
  statistics (Hyndman–Fan type 7), applied uniformly including at the median.

  A column with no spread keeps a divisor of one and survives as a constant,
  the same exact-zero rule the other three scalers already use. A column whose
  spread is merely *small* is scaled normally; if that overflows `f32` the
  batch is rejected with the offending row and column before anything is
  written, rather than being silently left unscaled.

  `unit_variance` is deliberately not claimed: it needs an inverse-normal-CDF
  primitive with its own accuracy contract, which is not worth adding to serve
  one optional flag.

  The scaler persists through `to_artifact` / `from_artifact` under artifact
  kind `44`, and composes into a `StagedPipeline` as a persisted stage. The raw
  spread is what is stored and the divisor is recomputed on decode, so a fitted
  model has exactly one valid byte string.
- `preprocessing::Normalizer`, `NormalizerParams`, and `Norm`: row-wise scaling
  so each row has unit `L1`, `L2`, or `Max` norm, where `Max` is the largest
  *magnitude*. A zero row has no direction to preserve, so it keeps a divisor
  of one and passes through unchanged.
- `preprocessing::Binarizer` and `BinarizerParams`: every value above a
  threshold becomes `1.0` and every other value `0.0`. The comparison is
  strictly greater-than, so a value exactly at the threshold becomes `0.0` and
  the two output classes are `(-inf, t]` and `(t, +inf)`.

  Both are stateless — they estimate nothing from the data beyond the width a
  pipeline hands them — and both therefore declare **no** capabilities at all,
  including no artifact. There is no fitted value to persist, so a persistence
  promise would be about something that does not exist. This is the same
  reasoning the baseline estimators already use.
- `preprocessing::FunctionTransformer`, `FunctionTransformerParams`, and the
  `ElementwiseFn` alias: a caller-supplied `fn(f32) -> f32` applied to every
  value, with an optional inverse.

  The map is a **function pointer, not a generic closure**. A capability
  declaration is an associated constant on a nameable type, and the capability
  snapshot asserts that every declaring public type appears in it by name — a
  type instantiated at an unnameable closure type would silently fall out of
  that coverage. A function pointer also captures no state, so two values of
  the type cannot behave differently. A caller who needs captured state, or a
  map that reads a whole row, implements `api::Transformer` directly.

  **Determinism of the supplied function is the caller's obligation.** FerricML
  guarantees the framing — fixed row-major order, validation before any write,
  and `ModelError::NonFiniteTransform` naming the first cell where a finite
  input maps to a non-finite output — but cannot guarantee the supplied
  function is pure.

  It declares no capabilities, including no artifact: a function pointer is an
  address in the current process image. It also has no `PartialEq`, because
  comparing function pointers compares addresses and one function is not
  guaranteed to have one address; an equality that is quietly wrong at a
  boundary is worse than none. Compare behaviour instead.
- `inverse_transform` and `inverse_transform_into` on `StandardScaler`,
  `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and `FunctionTransformer`,
  recovering pre-transform values into an allocated matrix or caller-owned
  storage.

  Exactness is stated rather than implied. The round trip is exact by
  construction only where no lossy operation happens — both statistics
  disabled, or a degenerate column whose divisor was substituted to one — and
  elsewhere is exact only when the arithmetic happens to be, since dividing by
  a scale and multiplying back is not a floating-point identity. `MinMaxScaler`
  with clipping enabled is deliberately **not** invertible in the usual sense:
  clipping is a projection, so inverting a clamped value recovers the fitted
  bound rather than the original.

  `FunctionTransformer::inverse_transform` returns
  `ModelError::NoInverseFunction` when no inverse was supplied, rather than
  silently applying the identity — which would look exactly like a successful
  recovery.
- `MinMaxScalerParams::with_feature_range` and `feature_range`, choosing the
  interval each column's fitted range is mapped onto. The default is unchanged
  at `0.0..=1.0`, `clip` now clamps into the configured interval, and a
  zero-range column lands on the interval's lower bound. An empty or inverted
  range is `ModelError::InvalidFeatureRange`, raised before any allocation.

  **Existing `MinMaxScaler` artifacts are byte-identical.** The output range is
  written only when it is one an older reader could not have assumed, so a
  default-configured scaler emits exactly the bytes it emitted before this
  parameter existed and every previously frozen artifact is unmoved. Older
  payloads are read, not rejected, and decode to an identical model. Each
  fitted model still has exactly one valid encoding, because the payload
  version is a function of the parameters rather than a choice — a default
  range written at the newer version is refused.
- `api::ModelError::InvalidFeatureRange`, raised when a min-max output range is
  not a finite interval with its minimum strictly below its maximum.
- `api::ModelError::NoInverseFunction`, raised when an inverse transformation is
  requested of a transformer that was not given one.
- `api::ModelError::InvalidThreshold`, raised when a decision threshold is not
  finite.
- `api::ModelError::InvalidQuantileRange`, raised when a quantile range is not
  two percentiles in `0.0..=100.0` with the lower value first. Equal
  percentiles are accepted and produce a zero spread, which is a legitimate way
  to ask for centering alone.
- `model_selection::ScorableClassifier`, the view the scoring layer takes of a
  fitted classifier: `probabilistic` for one that produces probabilities,
  `labels_only` for one that does not. A label metric works for either; a
  probability metric applied to a labels-only view is
  `ScoringError::UnsupportedOutput` naming what was required and what was
  supplied, never a substituted value. One type rather than a parallel family
  of entry points, and it makes "the labels and the probabilities come from the
  same model" true by construction.
- `model_selection::cross_validate_classifier_labels` and
  `model_selection::grid_search_classifier_labels`, for cross-validating and
  searching a classifier that produces labels but no probabilities. These build
  the model themselves, so the requirement is expressed in the bound rather
  than in an argument.
- `api::Capabilities::probability`, declaring whether a fitted classifier
  produces a probability per class. It is queryable on a runtime dispatch value
  through `capabilities()`, which is where a compile-time bound is unavailable.

- A capability declaration on `ranking::PairwiseLinearRanker`, which was the
  one fitted estimator that could not answer a capability query. It declares
  artifact persistence and nothing else: its weights belong to a pair
  observation rather than to a row, and a ranking score is not a probability,
  so it exposes neither a per-sample weighted fit nor a decision function.
- Schema-bound `RandomForestRegressor` artifacts that persist backend-neutral
  logical trees and revalidate every decoded topology, count, and parameter.
- `AnyRegressor` dispatch artifacts that record the fitted runtime variant and
  nest the estimator's own complete, independently validated artifact.
- Model-agnostic permutation feature importance over any fitted classifier or
  regressor, with a seeded permutation count, allocating and caller-owned entry
  points, and per-feature mean and dispersion of the scorer's quality loss.
- A compile-time estimator capability descriptor, `api::Capabilities`, carried
  by the `api::HasCapabilities` trait, so meta-layers query declared support
  for weighted fitting and artifact persistence instead of matching on concrete
  estimator types. The default declares nothing.
- Capability declarations on every fitted estimator and transformer FerricML
  ships, so callers can ask a type whether it supports weighted fitting or
  artifact persistence without matching on its concrete type.
- `dummy::DummyClassifier` and `dummy::DummyRegressor`, baseline estimators
  that ignore their features and predict the majority class or the training
  mean. They give a quality floor to compare a real estimator against.
- `preprocessing::MinMaxScaler`, a schema-bound transformer that maps every
  fitted feature onto `0.0..=1.0`, carries a column with no spread to `0.0`
  instead of dividing by zero, and can optionally clip later batches into the
  fitted range.
- `preprocessing::MaxAbsScaler`, a schema-bound transformer that divides each
  fitted feature by its largest observed magnitude, preserving sign and zeros,
  and passing an all-zero column through unchanged instead of dividing by zero.
- `pipeline::StagedPipeline`, a trainable multi-stage typed pipeline composing
  two or more fitted transform stages with one fitted estimator. `fit` trains
  every part in order on the previous stage's output, `new` validates every
  feature-width handoff before the composition exists, and inference runs
  through one caller-owned workspace that every stage writes a disjoint
  segment of.
- `pipeline::TransformerStack`, the statically dispatched stage-list contract
  `StagedPipeline` composes over, implemented for tuples of two and three
  fitted transformers.
- Schema-bound `StagedPipeline` artifacts under one artifact kind that records
  which concrete stage types the composition holds, in order, and which
  estimator type it ends in, so a composition never decodes as a different one.
  One capability declaration now covers every composition whose parts all
  persist, instead of one declaration per concrete composition.
- `pipeline::StageArtifact`, `pipeline::ModelArtifact`, and
  `pipeline::PersistedStack`, the persistence contracts a composition is
  generic over.
- `AnyClassifier::capabilities` and `AnyRegressor::capabilities`, reporting the
  selected runtime variant's capabilities. The declared constant on each
  dispatch enum and each concrete pipeline is composed from its variants or
  parts, so it promises only what holds whichever one is held.
- `ensemble::HistGradientBoostingClassifier`, a deterministic serial histogram
  gradient-boosted binary classifier fitted against binary log loss. It shares
  the regressor's binner, grower, and seven growth controls, and differs in
  dividing each leaf by the summed curvature of its rows rather than by their
  count. It reports a raw decision score, probabilities in `classes()` order,
  weighted fitting whose integer weights equal repeated rows, and schema-bound
  artifacts under an artifact kind of its own whose objective field names the
  loss the leaves were fitted to descend. Fitting requires both class labels.
- `api::AnyClassifier::HistGradientBoosting`, so a boosted classifier can be
  selected at runtime and persisted through the dispatch envelope like every
  other variant. The enum's declared capabilities are still the intersection
  over its variants, so adding one that offers no multiclass fit does not
  quietly widen what the enum promises.
- `metrics::ConfusionMatrix`, counting one classification result over the
  sorted union of the observed labels, and the `metrics::Average` vocabulary
  that combines its per-class precision, recall, F1, and F-beta as a binary,
  micro, macro, or support-weighted score. Micro-averaging a single-label
  result equals accuracy, and binary averaging equals the standalone binary
  functions exactly.
- `metrics::Averaging` and `metrics::ZeroDivision`, making the treatment of a
  class with an empty denominator an explicit typed choice. The default reports
  `MetricError::Undefined` instead of substituting a value.
- `ConfusionMatrix::balanced_accuracy`, mean recall over the classes that have
  true rows, and `ConfusionMatrix::matthews_correlation`, which correlates
  expected and predicted labels over any number of classes and is undefined
  rather than zero when either side is constant.
- `metrics::roc_curve`, `metrics::precision_recall_curve`, and
  `metrics::average_precision_score`, which sweep the decision threshold over
  the same tie-aware score ordering ROC AUC uses, so curve and scalar results
  agree by construction.
- `metrics::median_absolute_error`, `metrics::explained_variance_score`, and
  `metrics::mean_absolute_percentage_error`. The percentage error treats every
  expected value as a denominator, so a single expected zero is
  `MetricError::Undefined` rather than a silently clamped floor.
- `model_selection::TimeSeriesSplit`, a forward-chaining splitter for ordered
  observations. Each fold trains on a prefix and tests on the window that
  follows it, with an optional `gap`, so no fold is ever fitted on a row that
  comes after the rows it is evaluated on.
- `model_selection::LeaveOneOut`, which holds out one sample per split.
- `model_selection::GroupKFold`, which assigns whole groups to folds so no
  group is ever on both sides of a split. Assignment is deterministic and needs
  no seed: largest group first, into the fold holding the fewest rows.
- `model_selection::RepeatedKFold`, which runs shuffled K-fold several times
  with a per-repeat seed derived from one configured seed, so a caller can
  separate model variance from partition variance reproducibly.
- `Split::partial`, for a split that deliberately leaves rows out of both
  partitions — what a forward-chaining fold needs so that the rows after its
  test window leak into neither side — plus `Split::covered_samples`.
  `Split::sample_count` now reports the dataset size a split was built for,
  which is unchanged for every complete split.
- `model_selection::ClassificationScore` and `model_selection::RegressionScore`,
  the open scorer contract that batch scoring, cross-validation, and
  permutation importance all consume identically. The existing
  `ClassificationScorer` and `RegressionScorer` enums remain the built-in set
  and now implement these traits, including a declared `greater_is_better`
  orientation, so a caller can score on a metric FerricML does not enumerate.
- `model_selection::ScoringWorkspace` with `score_classifier_with` and
  `score_regressor_with`, the allocation-free scoring entry points. Reusing one
  workspace across calls of the same shape allocates only on the first call.
- `model_selection::ClassifierOutput` and `model_selection::ClassifierOutputKind`,
  which let a classification score declare whether it reads predicted labels or
  positive-class probabilities; being given another kind is the new
  `ScoringError::UnsupportedOutput` rather than a substituted value.
- `model_selection::GroupShuffleSplit`, which draws whole groups at random for
  each of `n_splits` independent holdouts, so no group is ever on both sides of
  a split. Splits are independent draws rather than a partition, and each one's
  draw is seeded from the configured seed and the split index.
- `model_selection::TestGroupSize`, the holdout size for a grouped split. It
  counts **groups**, deliberately as a type distinct from `TestSize`, which
  counts rows: rows only move a whole group at a time, so a row target could
  only be approximated, while a group target is exact. A size that would empty
  either side is the new `SplitError::InvalidTestGroupCount`, which names groups
  rather than reusing the row-counting `InvalidTestCount`.
- `model_selection::ParameterGrid`, an ordered set of typed hyperparameter
  candidates. An axis is a parameter type's own `with_*` builder method plus the
  values to pass it, so there are no string keys and a misnamed parameter is a
  compile error. Different axes carry different value types, the axis added last
  varies fastest, and `from_candidates` takes an explicit list for parameters
  that are not independent.
- `model_selection::grid_search_classifier` and
  `model_selection::grid_search_regressor`, serial typed hyperparameter search.
  The split iterator is drained once, so every candidate is cross-validated over
  exactly the same folds, and each candidate runs through the existing
  cross-validation and scorer path rather than a second evaluation path. The
  result reports every candidate's parameters and every fold's score through
  `model_selection::SearchResult` and `model_selection::CandidateScores`; the
  winner is the best mean fold score in the direction the score declares, with
  an exact tie keeping the earliest candidate in grid order. Search does not
  refit.
- `model_selection::SearchError`, which separates a call that is unusable before
  any fitting (`Setup`) from a candidate's own failure (`Candidate`, keeping the
  fold index) and from a score that returns a value no ranking can order
  (`NonFiniteScore`).
- `api::Capabilities::multiclass`, declaring that an estimator offers a
  multiclass fitting entry point over `ClassTargets`. `LogisticRegression` and
  `RandomForestClassifier` declare it; `AnyClassifier` declares it away
  structurally, because it owns fitted models and no fitting entry point.
  The estimator conformance battery drives a new
  `multiclass_declaration_matches_behavior` obligation from it.
- `metrics::multiclass_log_loss` and `metrics::multiclass_brier_score`, which
  score a whole row-major probability matrix against a sorted class list.
  Neither renormalizes a row, because FerricML's rows sum to one only within
  the documented `f32` tolerance. `multiclass_log_loss` agrees with `log_loss`
  at two classes; `multiclass_brier_score` squares every column where
  `brier_score` squares only the positive one, so it is exactly twice the
  binary value there — stated rather than left to be discovered. The new
  `MetricError::InvalidClassSet` and `MetricError::UnknownClass` report a class
  list that cannot name columns and a label with no column.
- `RandomForestClassifier::fit_multiclass`, a natively multiclass forest whose
  trees split on multiclass Gini impurity and store one probability per class
  at every leaf. The ensemble probability is the mean of the per-tree
  probability vectors — soft averaging, not a majority vote of per-tree labels
  — and the predicted label is the argmax of exactly those probabilities. A
  single observed class fits and returns one all-ones column. Binary `fit`
  keeps its scalar-leaf representation and every fitted value it had.
- `LogisticRegression::fit_multiclass` and `fit_multiclass_weighted`, a joint
  multinomial fit over `data::ClassTargets`. It is one optimization over all
  classes, not a wrapper around per-class binary models: probabilities are the
  softmax of one centred score vector, and no class is pinned as a reference.
  `decision_function` now returns `n_decision_columns` values per row, which is
  one for a binary fit — that shape, and every binary fitted value, is
  unchanged. `intercepts` and `n_decision_columns` are new; `intercept` reports
  the first score row and is no longer `const`. Probability rows are **not**
  renormalized: they sum to one only within `n_classes` `f32` ulps.
- `data::ClassTargets`, validated general classification targets over arbitrary
  `u8` labels. It carries the sorted, deduplicated set of labels actually
  observed, which is the probability column order of any classifier fitted on
  it. Labels are never assumed contiguous or zero-based, so `{7, 3, 10}` gives
  classes `[3, 7, 10]`, and selecting a subset recomputes the observed set.
- `model_selection::ClassifierOutput::ProbabilityMatrix`, a batch output
  carrying a whole row-major probability matrix together with the class list
  naming its columns, and the matching
  `ClassifierOutputKind::ProbabilityMatrix`. A score reading it is independent
  of the binary class layouts, so it works for any observed class set.
- `model_selection::ClassificationScorer::MulticlassLogLoss` and
  `MulticlassBrier`, which read that matrix. Cross-entropy agrees with the
  binary `LogLoss` on two classes; the multiclass Brier score squares every
  column and is therefore exactly twice the binary one, as documented.
- `model_selection::score_multiclass_classifier` and
  `score_multiclass_classifier_with`, scoring a fitted classifier against
  `data::ClassTargets`. They share one implementation with the binary entry
  points, so there is still a single prediction and class-layout path.
- `calibration::IsotonicRegression`, a deterministic pool-adjacent-violators
  monotone fit. It is both the non-parametric probability calibrator and a
  standalone monotone regressor over a single-column matrix. Observations that
  share an input value are pooled into their mean before pooling adjacent
  violators, so the fit depends on the multiset of observations and not on their
  order; prediction interpolates linearly between fitted points and holds the
  end values outside the fitted range.
- `calibration::PlattCalibrator` and `calibration::PlattParams`, the parametric
  calibrator: a two-parameter logistic fit of model scores onto labels, solved
  through the crate's shared binary log-loss objective rather than a third
  logistic solver. It regresses on Platt's prior-corrected targets, so a
  perfectly separating score still has a finite fit and calibrated
  probabilities never collapse to exactly zero or one.
- `calibration::Calibrator`, the fitted monotone score-to-probability map
  contract, with an in-place batch form so calibrated prediction needs no second
  buffer.
- `calibration::CalibratedClassifier`, a fitted classifier composed with a
  fitted calibrator. It is an ordinary `Classifier`, so it scores and
  cross-validates through the existing paths unchanged. It calibrates the
  wrapped model's positive-class probability, takes its calibration rows as an
  explicit parameter rather than reusing training rows, and predicts labels from
  its own calibrated probabilities. A Platt-calibrated composition additionally
  exposes `decision_function`.
- `api::Capabilities::decision_function`, declaring whether a fitted classifier
  exposes a raw, unsquashed decision score. Producing probabilities is required
  of every `Classifier` and is not what this records.
- `RandomForestRegressor::fit_weighted`, `RandomForestClassifier::fit_weighted`,
  and `RandomForestClassifier::fit_multiclass_weighted`, taking validated
  per-row `data::SampleWeights`. A weight scales the row's contribution to every
  impurity, leaf statistic, and leaf distribution, and composes with the
  bootstrap replication count. Weights of exactly one reproduce the unweighted
  fit bit for bit, an integer weight is the same fit as repeating the row that
  many times, and a weight of zero removes the row — including from the
  bootstrap resample, which draws only among positively weighted rows. Both
  forests now declare `sample_weights` in their capability descriptor. The
  minimum split and leaf sizes bound **weight** rather than rows, which is a
  deliberate divergence from the reference contract taken so the integer-weight
  equivalence holds unconditionally; unweighted fitting is unaffected.
- `HistGradientBoostingRegressor::fit_weighted`. A weight scales the row's
  gradient and its share of every node's weight total, so the baseline is a
  weighted mean and the minimum leaf size counts weight rather than rows.
  Weights of exactly one reproduce the unweighted fit bit for bit, and an
  integer weight is the same fit as repeating the row. The bin grid stays
  unweighted: it is fitted from the distinct observed feature values, which
  neither a weight nor a repeated row changes.
- Artifact persistence for a joint multinomial `LogisticRegression` fit, under
  a second payload schema of the existing estimator kind. It stores the
  observed class list, one intercept per class, and one coefficient row per
  class; decoding selects the reader from the recorded payload version, so a
  binary and a multiclass artifact never decode as each other. Binary artifacts
  keep their exact bytes. `LogisticRegression` now persists every fit it
  offers, so its declared `artifact` capability no longer depends on which
  entry point was used, and the conformance battery asserts that a classifier
  declaring both persistence and multiclass fitting round trips a multiclass
  fit as well.
- Schema-bound `RandomForestClassifier` artifacts covering both fitted leaf
  representations under one artifact kind. The payload records which leaf
  arithmetic it holds and the reader refuses to build the other; a binary fit
  reuses the scalar logical-tree records unchanged, and a multiclass fit writes
  the same topology with a reserved zero in the scalar slot plus one per-tree
  leaf-distribution block ordered by pre-order leaf rank, so the encoding stays
  a unique name for the model. The classifier now declares `artifact`.
- `AnyClassifier` dispatch artifacts that record the fitted runtime variant and
  nest the estimator's own complete, independently validated artifact, mirroring
  `AnyRegressor`. A variant that carries more than one payload schema of its own
  keeps choosing between them itself, so restoring a dispatch artifact restores
  the variant *and* the fit it held. `AnyClassifier` now declares `artifact` by
  composition rather than declaring it away.
- `linear_model::LogisticSolver` and `LogisticRegressionParams::with_solver`,
  selecting the update rule a logistic fit uses. The default is and stays
  `Newton`, the exact second-order path every existing fitted model was
  produced by; `Lbfgs` is a matrix-free limited-memory quasi-Newton path whose
  storage is linear rather than quadratic in the parameter count. Both minimize
  the same penalized objective, so they agree on its minimizer, but `tol` means
  the largest coefficient update under `Newton` and the mean objective's
  gradient norm under `Lbfgs`. Neither payload schema records a solver, so a
  model fitted under a non-default one reports
  `ArtifactError::UnsupportedModelState` rather than writing bytes that would
  decode as a model claiming `Newton` provenance.
- Joint multinomial logistic fits above the exact solver's parameter ceiling,
  through `LogisticSolver::Lbfgs`. The ceiling is a property of the selected
  solver's storage rather than of the model, so the exact path keeps refusing
  above 2048 stacked parameters and keeps producing the identical fit below it,
  while the matrix-free path accepts 131 072 within the same storage budget.
  `ModelError::MulticlassSystemTooLarge` now reports whichever limit applied.
- `linear_model::Lasso`, a dense L1-regularized regressor fitted by cyclic
  coordinate descent. Coefficients it removes are exactly `0.0` and positively
  signed, so `coefficients` reads as a feature selection. Its objective divides
  the weighted squared error by twice the total sample weight, matching the
  reference contract's documented parametrization — so its `alpha` is a
  different quantity from `Ridge`'s, and the penalty applies to raw-scale
  coefficients because fitting centers but does not rescale the design.
  Sample weights are fractional row counts, and only their ratios matter. It
  declares weighted fitting and, deliberately, no artifact.
- `linear_model::ElasticNet`, the same coordinate-descent solver under a mixed
  L1 and L2 penalty, parametrized by `alpha` and `l1_ratio` exactly as the
  reference contract documents. `l1_ratio = 1` reproduces `Lasso` bit for bit
  at the same `alpha`; `l1_ratio = 0` is the ridge objective, but at
  `Ridge`'s `alpha * total_weight` rather than at the same number. The L2 term
  restores strict convexity, which both spreads weight across correlated
  features and makes designs converge that a pure L1 penalty does not.
- `ModelError::InvalidPenaltyAlpha` and `ModelError::InvalidL1Ratio`, reported
  at the public boundary before any allocation or fitting work.
- `ModelError::SolverDidNotConverge`, reported when an iterative solver
  exhausts `max_iter` — or is asked for a tolerance below what the objective's
  own numerical resolution can certify — instead of returning the last iterate
  as though it were a fitted model.

### Changed

- A `predict_class_proba_into` call that is invalid in *both* its batch width
  and its requested class now reports `ModelError::FeatureDimension` on every
  classifier. `tree::DecisionTreeClassifier`, the forests,
  `linear_model::LogisticRegression`,
  `ensemble::HistGradientBoostingClassifier` and
  `calibration::CalibratedClassifier` previously reported
  `ModelError::UnknownClass` for that call while their own allocating
  `predict_class_proba` — and `dummy::DummyClassifier` — reported the width.
  **A caller matching on the error of a doubly-invalid call gets a different
  variant than before**; a call that is invalid in only one way is unaffected,
  and no valid call changes at all. The rule is now stated once and uniformly:
  validation checks the shape of the input before the content of the request,
  because the width must hold before the matrix can be indexed at all. The
  divergence appeared when the batch-width check was hoisted into the
  allocating trait defaults without the caller-owned primitives underneath
  being aligned, and it survived because no test in the suite made a call that
  was invalid twice. The conformance battery now carries a
  `width_precedes_class` obligation driven by exactly that call, proven to fire
  by a probe that swaps the two checks.
- **Breaking.** `MaxFeatures` has one public path again, and it is
  `tree::MaxFeatures`. The `ensemble::MaxFeatures` re-export is removed; callers
  importing it — including callers of a *forest's* `with_max_features`, which
  takes this type — change the import path and nothing else, because it was
  always the same type. Two paths to one type left rustdoc free to pick the
  canonical one, and it picked `ensemble`, so
  `tree::DecisionTreeClassifierParams::max_features` rendered as returning
  `ferricml::ensemble::MaxFeatures` — a standalone tree's own parameter
  documented as an ensemble type, reading as though `tree` depended on
  `ensemble` when the `tree-below-estimators` layout rule enforces the
  reverse. The type is defined beside the grower that consumes it and now
  publishes from there alone.
- `api::AnyClassifier` and `api::AnyRegressor` now document their variant lists
  as a deliberate, curated set and say what decides membership. The API document
  claimed the regressor variants "cover forests", which was never true of
  `ensemble::ExtraTreesRegressor` and had drifted further as estimators shipped;
  the enums cover 3 of 6 classifiers and 4 of 10 regressors. No variant was
  added, because a dispatch enum declares the *intersection* of its variants'
  capabilities: `AnyRegressor` declares persistence only because all four of its
  variants persist, so admitting a variant that declares nothing — `DummyRegressor`
  and `IsotonicRegression` both do — would silently withdraw that declaration
  from every existing caller. An enum tracking every estimator would end up
  declaring nothing at all. Both enums stay `#[non_exhaustive]`, so a variant can
  still be admitted later without touching any existing estimator's bytes.
- **Breaking.** `calibration::IsotonicRegression` is now fitted like every other
  FerricML estimator. `calibration::IsotonicRegressionParams` is a new empty
  parameter type — the same shape `dummy::DummyClassifierParams` and
  `preprocessing::MaxAbsScalerParams` already ship — and it is a required final
  argument of `IsotonicRegression::fit`, `IsotonicRegression::fit_calibration`
  and `calibration::CalibratedClassifier::fit_isotonic`. The estimator also
  implements `api::HasParams` and exposes inherent `get_params`,
  `n_features_in`, `predict` and `predict_into`; the last two were previously
  reachable only by importing `api::Regressor`, which inverted the crate's
  preference for the caller-owned form being the easiest to reach. It was the
  only concrete leaf estimator in the crate missing any of these. Nothing about
  a fit changes: the parameter type carries no state, and the inherent
  prediction methods forward to the same trait implementation. Adding an
  out-of-range policy or a decreasing direction later is now an additive change
  rather than a `fit` signature break, which is what the empty-params
  convention exists to buy.
- **Breaking.** `inspection::permutation_importance_classifier` and
  `inspection::permutation_importance_classifier_into` are generic over the
  target vocabulary through the same sealed `data::ClassificationTargets`
  trait, instead of taking `data::BinaryTargets` alone. A natively multiclass
  classifier is now inspected through the crate's only classifier
  permutation-importance entry point, with the orientation, workspace reuse and
  caller-owned output the binary path already had. Nothing becomes more
  permissive: a binary positive-probability metric asked for over a wider class
  set is still `ScoringError::UnsupportedClasses`. Existing binary calls
  compile unchanged unless they name the type parameter explicitly, which now
  takes the target type first and the score second.
- **Breaking.** `predict_positive_proba` is now the allocating **batch**
  method on every classifier that carries a positive class, and the
  single-row form it used to name is `predict_positive_proba_one`. Callers
  must rewrite `model.predict_positive_proba(row)` as
  `model.predict_positive_proba_one(row)`; the argument type changes from
  `&[f32]` to `&data::MatrixView` and the return from `f32` to `Vec<f32>`, so
  a missed call site is a compile error rather than a silent reinterpretation.
  Affects `ensemble::RandomForestClassifier`,
  `ensemble::ExtraTreesClassifier`,
  `ensemble::HistGradientBoostingClassifier`,
  `tree::DecisionTreeClassifier` and `linear_model::LogisticRegression`.
  The old pairing was the crate's only shape mismatch between an allocating
  method and its `_into` partner: `predict_positive_proba` took one row while
  `predict_positive_proba_into` took a matrix, which left the caller-owned
  batch form with no allocating partner and put a single-row method under the
  name the batch form owns. Renaming it also gives the batch form the
  allocating partner it never had, on all five classifiers rather than the two
  that happened to expose `_into`. Nothing about the fitted models changed and
  no artifact byte moved.

- **Added, and breaking.** `ranking::PairwiseLinearRanker::pair_margins`
  returns raw margins for a slice of pairs, allocating the output;
  `pair_margins_into` was the only caller-owned method in the crate with no
  allocating partner at all. In the same family, `compare` is now the
  allocating **batch** comparison over a slice of pairs and the single-pair
  form is `compare_one`, so callers must rewrite
  `ranker.compare(&items, pair)` as `ranker.compare_one(&items, pair)`. The
  `compare` collision is the same defect as `predict_positive_proba` and was
  missed by the API audit's original sweep, which compared only each method's
  first argument — `&MatrixView` on both.

- **Breaking.** `model_selection::cross_validate_classifier` and
  `model_selection::grid_search_classifier` are generic over the target
  vocabulary, through the new sealed
  `data::ClassificationTargets` trait, instead of taking
  `data::BinaryTargets` alone. `data::ClassTargets` now folds through exactly
  the same entry point, so a natively multiclass estimator can be
  cross-validated and tuned with the `CrossValidationError` fold attribution,
  the `ScoringWorkspace` reuse, and the split and class-layout guards a
  hand-rolled fold loop gives up. The loop branches on
  classifier-versus-regressor and on nothing else: label arity is a property of
  the metric — `ClassificationScorer::MulticlassLogLoss` and `MulticlassBrier`
  already read a whole probability matrix over any observed class set — so
  there is no multiclass entry point to add. Nothing becomes more permissive: a
  binary positive-probability metric asked for on a wider class set is still
  `CrossValidationError::UnsupportedClasses`. Existing binary calls compile
  unchanged; the trait is sealed because `select` must preserve the
  container's construction-time guarantees, so a new target shape arrives as a
  new `data` container with its implementation.

- **Breaking.** `model_selection::cross_validate_classifier` and
  `model_selection::grid_search_classifier` take a final `view` argument, and
  `model_selection::cross_validate_classifier_labels` and
  `model_selection::grid_search_classifier_labels` are removed. `view` says how
  each fold's fitted model presents itself to the scoring layer, exactly as the
  scoring and permutation-importance entry points already asked:
  `|model| ScorableClassifier::probabilistic(model)` for a model that produces
  probabilities, `|model| ScorableClassifier::labels_only(model)` for one that
  does not. `model_selection` was answering "does this classifier give
  probabilities?" two ways — a `ScorableClassifier` value in one half, a
  duplicated function pair in the other — and now answers it one way. The
  constructor is passed rather than a `ScorableClassifier` value because the
  fitting closure returns an owned model per fold and the view borrows it, so
  the borrow has to be taken inside the fold loop. Neither entry point is
  bounded on `api::ProbabilisticClassifier` any more; the view carries that
  requirement, so a probability metric under a labels-only view is
  `CrossValidationError::UnsupportedOutput` at run time rather than a compile
  error the caller cannot work around.

- **Breaking.** Persistence is now a trait. `to_artifact` and `from_artifact`
  moved from inherent methods onto `artifact::ModelArtifact` (estimators, one
  feature schema) and `artifact::StageArtifact` (transformers and compositions,
  an input and an output schema), so calling either needs the trait in scope —
  `use ferricml::artifact::ModelArtifact;` — exactly as calling `predict` needs
  `api::Estimator`. No artifact's bytes change.

  This closes a gap rather than moving a name. Persisting used to require two
  independent declarations: writing the encoder, and separately listing the
  type as composable. Seven estimators had the first and not the second, so
  `ensemble::RandomForestClassifier`, both extra-trees models,
  `ensemble::HistGradientBoostingClassifier`, both standalone trees and
  `ranking::PairwiseLinearRanker` could be saved on their own but not inside a
  `pipeline::StagedPipeline`. All seven now compose, as do `api::AnyRegressor`
  and `api::AnyClassifier`. The traits are no longer re-exported from
  `ferricml::pipeline`; `ferricml::artifact` is the one path.
- A `pipeline::StagedPipeline` now declares capabilities whatever it holds,
  computing `artifact` from its parts instead of requiring every part to
  persist before it can declare anything. A composition that does not persist
  previously had no capability declaration at all, which also kept it out of
  the conformance battery. `pipeline::TransformerStack` gains
  `STAGES_PERSIST`, and its tuple implementations now require each stage to
  declare capabilities.
- Histogram-boosting fits report four distinct failures where they previously
  reported two. `api::ModelError::NumericalOverflow` used to stand for both a
  non-finite residual and a residual-length mismatch, and
  `api::ModelError::TreeTooLarge` for both an oversized tree and a structurally
  invalid one. A residual-length mismatch is now
  `api::ModelError::OutputLength` — it is a shape bug, not an overflow — and a
  structurally invalid tree is the new `api::ModelError::InvalidTreeStructure`.
  A caller matching on `NumericalOverflow` or `TreeTooLarge` from a
  `HistGradientBoosting*` fit may now see the other variant instead. The
  errors do not carry the residual's index: no public FerricML error names a
  row or an observation, and this is not the place to start.

- The allocating defaults on `api::Classifier`, `api::ProbabilisticClassifier`,
  `api::Regressor`, and `api::Transformer` — `predict`, `predict_proba`,
  `predict_class_proba`, and `transform` — now check the batch width against
  `Estimator::n_features_in` *before* sizing their output buffer, rather than
  allocating it and discovering the mismatch inside the `_into` primitive they
  delegate to. The error is unchanged in kind and in values; what changes is
  that a rejected call now allocates nothing at all. An implementor whose
  `_into` method accepted a width other than its declared `n_features_in`
  would see the default reject that call.

- `RandomForestClassifier::predict` and `ExtraTreesClassifier::predict` check
  the batch width on all three of their fitted-shape branches before
  allocating, rather than on the single-class branch only. The error a
  wrong-width batch receives is the same on every branch and is unchanged;
  what changes is that the binary and multiclass branches no longer allocate
  their output first.

- `ranking::PairwiseLinearRanker::fit` checks every pair index and the total
  pair weight before it copies and sorts the observation batch, rather than
  after. A batch that will be refused no longer pays for a full copy and an
  `O(n log n)` sort first. The errors and the fitted model are unchanged.

- Six further entry points found by sweeping for the same shape check the batch
  width before allocating rather than after: `LogisticRegression`,
  `HistGradientBoostingClassifier` and `CalibratedClassifier`'s
  `decision_function`, `StagedPipeline::transform`,
  `PairwiseLinearRanker::score_items`, and `CalibratedClassifier`'s
  `Classifier::predict_into` — which is an `_into` method, so the scratch
  buffer it no longer allocates for a refused batch was its only allocation.
  Every error is unchanged in kind and in values.

- **Breaking.** Producing probabilities is no longer required of every
  classifier. `predict_proba`, `predict_proba_into`, `predict_class_proba`, and
  `predict_class_proba_into` move off `api::Classifier` onto a new
  dyn-compatible sub-trait, `api::ProbabilisticClassifier`. **Callers that
  invoke any of those four through a generic bound or a trait object must
  require `ProbabilisticClassifier` instead of `Classifier`**; concrete calls
  on a shipped estimator are unaffected, since every classifier FerricML ships
  today implements the sub-trait. Trait upcasting means a
  `&dyn ProbabilisticClassifier` is still accepted wherever a `&dyn Classifier`
  is wanted.

  The split exists because margin-based classifiers — ridge classification,
  discriminant analysis, discrete boosting — have a natural output that is a
  score rather than a distribution. A required probability method would have
  forced each of them either to fabricate a number it never earned or to fail
  at run time on a method the type system promised. A caller that needs
  probabilities now says so in its bounds and gets a compile error rather than
  a surprise.

  Consequently `score_classifier`, `score_classifier_with`,
  `score_multiclass_classifier`, `score_multiclass_classifier_with`,
  `permutation_importance_classifier`,
  `permutation_importance_classifier_into`, and the classifier
  cross-validation and search entry points now take a probability-producing
  classifier, and `CalibratedClassifier` requires one — a calibrator maps a
  probability, so there is nothing to calibrate without one. The classifier
  scoring and permutation-importance entry points take a
  `ScorableClassifier` view rather than a bare reference, so a label-only
  classifier remains scorable on a label metric.

- **Breaking.** `AnyClassifier` no longer exposes `predict_proba`,
  `predict_proba_into`, `predict_class_proba`, or `predict_class_proba_into`
  directly, and deliberately does **not** implement
  `api::ProbabilisticClassifier`. Reach probabilities through
  `AnyClassifier::as_probabilistic`, which returns
  `Option<&dyn ProbabilisticClassifier>`. Runtime dispatch is the one place the
  concrete type is erased by construction, so it is the one place the question
  can only be asked rather than proven in the bounds — and the fallible
  accessor is what lets a future margin-based variant be added without
  breaking this surface a second time.


- Permutation importance takes any score implementing the new scorer traits and
  runs through the shared allocation-free scoring path, so it no longer carries
  its own copy of the scorer dispatch, the singleton-class probability
  handling, or the per-metric orientation table. Its proven allocation bound is
  unchanged.
- `score_classifier`, `score_regressor`, `cross_validate_classifier`, and
  `cross_validate_regressor` are generic over the new scorer traits instead of
  taking the built-in enums. Calls that pass a built-in scorer are unaffected;
  a turbofished `cross_validate_*` call gains one inferred type argument.

- **Breaking.** `api::ModelError` no longer has `EmptyTargets`,
  `InvalidBinaryTarget` or `NonFiniteTarget`. No public entry point could
  produce them: every estimator that checked for those conditions was handed a
  `data::BinaryTargets` or `data::RegressionTargets`, and those containers have
  no unchecked constructor — `new` refuses each case as a `data::DataError`,
  `select` preserves what `new` established and refuses an empty selection, and
  `From<BinaryTargets> for ClassTargets` widens without weakening. A caller
  matching on one of the three was matching on a state the type system already
  ruled out; the corresponding `DataError` variants are where the condition is
  actually reported, and they are unchanged. `ModelError` documents the absence
  so the variants are not reintroduced.
  `EmptyData` and `NonFiniteFeature` deliberately remain: `predict_one` and
  calibration take a bare `&[f32]`, which nothing has validated.
- Tree and forest fitting no longer rescans the training matrix for non-finite
  features. Every value in a `data::MatrixView` is finite by construction, so
  the scan was re-deriving the container's own invariant at O(rows × columns)
  on every fit. No performance claim is attached to this: it has not been
  measured.

### Fixed

- `linear_model::LogisticRegression`'s Newton step is now **damped**, so the
  default solver is globally convergent instead of only locally so. The exact
  step minimizes a local quadratic model of the penalized objective; where that
  model is untrustworthy the step overshoots, the next model is built somewhere
  worse, and the failure compounds. On a badly scaled near-separable design with
  a weak penalty, iterates reaching `1e63` were measured. The full step is now
  accepted whenever it sufficiently decreases the objective and halved until it
  does otherwise — Armijo backtracking, in the new `optimize::damping` seam that
  both the binary and the multinomial path consume.
  <br>
  **This is a breaking change: some fitted values move.** They move far less than
  the description suggests, and the measurement is what says so. Over 1,600
  generated well-conditioned binary designs — every one of which takes more than
  one Newton step — 1,573 fitted `f32` coefficient sets are bit-identical to the
  undamped path and no iteration count changes at all; the 27 that move do so
  because the full step failed sufficient decrease somewhere, which is the case
  damping exists for. At `f64`, before narrowing, 108 of the 1,600 moved, by a
  relative displacement of median `4.8e-9` and at most `5.1e-8` — under half an
  `f32` ulp, which is why only 27 survive narrowing. All 108 are at a local
  minimum under both arms, and the damped point has the strictly lower objective
  on 78 of them against 11 the other way. The four designs the frozen reference
  fixture pins are bit-identical, `make reference-check` passes unchanged, and no
  fixture constant moves — that file holds the reference implementation's own
  outputs, so a FerricML solver change cannot move one; what it could break is
  agreement *with* them, and that is what `reference-check` reports. On the
  deliberately ill-conditioned regions the
  move is larger, 342 of 919 binary and 147 of 343 multinomial, which is the
  population whose old iterates were wrong.
  <br>
  What it buys, measured the same way: over a generated ill-conditioned region of
  972 binary designs the undamped step refused 53 as non-convergent, and the
  damped step refuses **none**, with all 972 returned at a local minimum of the
  penalized objective in the caller's own feature space — convergence, not merely
  the absence of an error. The multinomial region's 22 non-convergence refusals
  likewise go to zero, and 18 of its 210 collapsed-curvature `LinearSolveFailed`
  refusals resolve as well, because the damped path never reaches the iterate
  whose curvature had collapsed. Over a wider and harsher 704-design sweep — column
  scales to `1e7`, separations to `12`, `C` to `1e12` — there is no
  non-convergence refusal left at the default budget at all, and 139 fits exhaust
  it and are accepted on the Newton decrement.
  <br>
  The convergence test still reads the **exact** step rather than the damped one.
  The exact step's size is the second-order estimate of the distance to the
  minimum; a step shortened because the local model was untrustworthy is evidence
  about the model, and treating it as convergence would stop the iteration
  wherever the model was worst.
  <br>
  `optimize::line_search`'s strong-Wolfe search was the obvious candidate and was
  rejected on measurement. Its curvature condition exists to keep L-BFGS's stored
  inverse-Hessian approximation positive definite, and a Newton path has no such
  pairs — it refactorizes the exact Hessian every iteration, and that
  factorization succeeding *is* the certificate. Counted directly: over the 1,600
  well-conditioned designs the curvature condition rejects the exact step on
  **none**, so it would move exactly the population sufficient decrease already
  moves; over the ill-conditioned region it rejects a further 273 of 972 that
  sufficient decrease alone already rescues. It would move more fitted values for
  no additional capability.
  <br>
  Two tests replace ones whose premise the fix falsified, and both replacements
  are stronger. `every_binary_fit_in_the_ill_conditioned_region_reaches_its_minimum`
  asserts the whole region is returned *and* at a local minimum — either half
  alone is satisfiable by a bad solver — and undamping the step fails it.
  `an_exhausted_multinomial_budget_that_reached_its_minimum_is_fitted_not_refused`
  can no longer reach an exhausted budget at the default `max_iter`, because the
  region now converges in at most 53 of its hundred iterations; it sets the budget
  to one iteration short of what each fit needs instead, which exercises both
  answers — 348 of 385 accepted on the decrement and 37 refused — where the
  previous construction only ever watched acceptances. Deleting the decrement
  certificate refuses all 385; making it unconditional accepts all 385, and also
  fails five starved-budget tests.
  <br>
  `docs/determinism.md` gains the two `ln` entries this adds to the Newton fitting
  paths, and the argument that a halving sequence of exact powers of two is a
  narrower determinism risk than the bisection it sits beside: its next trial does
  not depend even on a bracket, only on the halving index.

- `calibration::PlattCalibrator` now stores its map in **centred** form, so a fit
  on a near-constant score column returns the line it solved instead of a
  narrowing that lost the answer. The map was stored as `slope` and `intercept`
  and evaluated as `slope * score + intercept`. A calibration sample whose scores
  are nearly equal identifies its slope only through their spread, so a spread of
  `1e-6` puts both stored fields near `1e6`, where an `f32` ulp is `0.0625` —
  while their sum is `O(1)`. Every bit of the cancellation was charged to a
  quantity six orders of magnitude smaller than its operands. The solve was
  already correct and already tested to be at the minimum; the storage was not.
  <br>
  **This is a breaking change: calibrated probabilities move.** Nothing else
  does. The centred pair is stored *beside* `slope` and `intercept` rather than
  replacing them, so both accessors return the same bits they always returned and
  the public API is byte-identical under `api-check` — `intercept` also stays a
  single narrowing of the `f64` answer, which is strictly more accurate than
  recovering it from two already-narrowed fields would have been. What moves is
  what a caller *evaluates*: `calibrate` and `decision_score` change in their last
  bits on about a third of scores for well-conditioned samples, and change
  substantively on the degenerate ones, which is the point. The centre is
  deliberately not exposed — two `f32` accessors were never enough to reconstruct
  this map, which is the defect rather than an omission, and a third would invite
  a caller to rebuild the line by hand and get the cancellation straight back.
  `decision_score` is the map.
  <br>
  Measured over 6,330 fits from the near-constant region, against the same
  objective solved independently in centred and scaled coordinates at `f64`: the
  shipped line's log-loss gap above the minimum had median `1.7e-5`, 99th
  percentile `8.6e-1` and maximum `5.0` nats, and its worst calibrated
  probability was off by `0.65`. Centred, the same region's gap is median `0`,
  99th percentile `3.5e-8`, maximum `6.5e-8`, and the worst probability error is
  `8.3e-8`. On 2,000 well-conditioned samples the two forms are
  indistinguishable — worst gap `8.3e-8` against `7.8e-8` — so this costs nothing
  where there was nothing to fix.
  <br>
  Storing better-rounded values in the old two fields was not an option, and that
  is a measurement rather than a judgement: searching +-2 ulp in **both** stored
  fields, the uncentred form still cannot get its worst case below `4.2` nats,
  while the centred form's achieved `6.5e-8` is already at its own +-2 ulp floor
  of `5.8e-8`. The defect is which two numbers are stored. For the same reason the
  filed "only 1,296 of 7,725 fits are a minimum among their eight one-ulp
  neighbours" statistic is not the defect and barely moves (1,500 of 6,330 becomes
  1,720): narrowing an `f64` minimizer lands beside a marginally better grid point
  whatever the parametrization, and what matters is whether the miss costs `1e-8`
  nats or `5`.
  <br>
  No artifact format changes and no payload version moves, because
  `PlattCalibrator` has no artifact representation at all — there is no
  calibration entry in the artifact-kind table, `CalibratedClassifier` documents
  that it has no artifact kind, and `PairwiseLinearRanker` persists a nested
  `LogisticRegression` rather than a calibrator. The 128 adversarial artifact
  fixtures are untouched and `MinMaxScaler`'s emit-a-version-only-when-needed
  precedent has nothing to apply to.
  <br>
  The order of the two narrowings is load-bearing and tested. The centre is
  narrowed to `f32` first and the stored intercept computed at *that* centre,
  because evaluation subtracts the stored `f32` centre and nothing else; folding
  in the `f64` centre instead leaves `slope * (mean64 - mean32)` unaccounted for,
  which at these slopes is a raw-score error around `0.03`.
  `a_near_constant_fit_evaluates_the_probability_its_solve_found` fails on that
  one-line change and on reverting `decision_score` to the uncentred form, and it
  asserts that the region still reaches slopes of `1e5` so the bound cannot hold
  vacuously. A companion test pins the arithmetic property the accuracy rests on —
  that the centring subtraction's error stays *relative* to its own result, at most
  half an ulp, often exactly zero by Sterbenz's lemma — and counts both the exact
  and the rounded population so neither half is asserted over an empty set.

- `linear_model::LogisticRegression` no longer returns an unconverged Newton
  iterate as a fitted model, on **either** target shape. `fit` and
  `fit_multiclass` broke out of the iteration when the largest standardized
  coefficient update fell to `tol` and otherwise fell through to `Ok`, so an
  exhausted `max_iter` was reported only by `n_iter`. Their `LogisticSolver::Lbfgs`
  sibling, on the same data and the same parameters, already returned
  `ModelError::SolverDidNotConverge`; the two solvers are documented to agree
  on the minimizer and disagreed on whether one had been found. **This is a
  breaking behaviour change on badly conditioned designs: a call that
  previously returned a model can now return an error.** No fitted value
  moves — a fit that is returned has exactly the coefficients and intercepts it
  had before, bit for bit, and the frozen reference fixtures and artifact
  fingerprints are unchanged.
  <br>
  Exhaustion by itself is *not* the new test, and the measurement is what
  decided that. Over 57,600 sampled binary fits, 3,838 exhaust `max_iter`, and
  3,382 of those sit on the minimum — the absolute test accepts **none** of
  them, so refusing on exhaustion alone would have converted every one into a
  spurious error. The cause is conditioning rather than scale: the standardized
  system carries an L2 penalty of `1 / (C * scale^2)` on the feature diagonal,
  nothing at all on the intercept, and a curvature `p (1 - p)` that collapses
  towards its floor wherever the fit separates a row confidently. At an
  observed condition number reaching `2e26`, the last digits of a gradient
  already down at `1e-13` are amplified into a coefficient step of `1e-2`, far
  above a `tol` of `1e-4`, and no further iteration removes it. The multinomial
  system is worse conditioned, not better — median `1e13` against the binary
  median of `13` — because its unpenalized intercept block is singular in the
  direction that shifts every class alike.
  <br>
  The acceptance test at exhaustion is therefore the Newton decrement, the last
  step's inner product with the gradient it was computed from, which is twice
  the objective's own estimate of the distance above the minimum and is
  unchanged by rescaling a design column. Measured against an independently
  conditioned damped-Newton solve, it accepts all 3,382 at-minimum fits and no
  fit that is more than `7.3e-8` relative above the minimum, while the fits it
  refuses are `2.7e5` relative above it — five orders of magnitude of clear
  air. The three candidates it was compared with all fail: a relative step test
  refuses 2,025 of the 3,382 *and* accepts a fit `3.1e7` above the minimum; a
  gradient infinity norm accepts one `2.7e5` above it, and the mean gradient
  norm one `1.5e8` above it, because at this conditioning a small gradient is
  no evidence of a small distance. The loop's own stopping rule is untouched,
  which is why nothing that converges today moves.
  `LogisticRegression::n_iter` may therefore equal `max_iter` on a returned
  fit, and when it does the fit is at the minimum rather than merely the last
  thing tried.
  <br>
  Four tests pin the rule over generated ill-conditioned regions rather than
  one fixture, two per target shape: refusing on plain exhaustion fails the one
  that watches the region fit — 163 of 919 binary fits and 52 of 344
  multinomial ones exhaust their budget and are accepted — and an acceptance
  that never refuses fails the one that starves the same region, where every
  reachable fit is refused. A third holds the genuinely divergent fits in that
  region to a refusal at the *full* default budget, not only a starved one,
  because the undamped exact step is not globally convergent and this is the
  population the old code returned as a model.
- `calibration::PlattCalibrator::fit` no longer returns an unconverged Newton
  iterate as a fitted calibrator. The loop `break`s when the largest parameter
  update falls to `tol` and otherwise fell through to `Ok`, with only `n_iter`
  to say the tolerance was never met. `LogisticRegression`'s Newton path had
  the same defect on both target shapes and is fixed in its own entry below;
  with those two, every iterative solver in the crate now reports
  `ModelError::SolverDidNotConverge` rather than returning an unconverged
  iterate. **This is a breaking behaviour change on
  degenerate calibration samples: a call that previously returned a model can
  now return an error.** No fitted value moves — a fit that is returned has
  exactly the parameters it had before, bit for bit.
  <br>
  Exhaustion by itself is *not* the new test, and making it the test would have
  been a worse defect than the one it replaced. `tol` bounds a parameter update
  in parameter units, and those units have no fixed scale. A calibration sample
  whose scores are nearly equal identifies its slope only through their spread,
  so a spread of `1e-6` puts the maximum-likelihood slope near `1e6`; the two-
  parameter Newton determinant is then a difference of nearly equal products
  that keeps a median of two significant digits, and the computed step inherits
  a rounding floor of roughly the parameter magnitude times that lost precision.
  Measured on the reported sample the floor is `1.9e-5` and does not move after
  100,000 iterations, while the gradient at that point is `4.6e-12` and the
  objective is at its minimum to the last bit. Over 7,725 sampled fits from that
  region, *every one* of the exhausted iterates was at the minimum — worst
  objective gap `5.1e-9` — so refusing on exhaustion alone would have converted
  the whole region into spurious errors.
  <br>
  The acceptance test at exhaustion is therefore a quantity that does not change
  with the parameter scale: the Newton decrement, the last step's inner product
  with the gradient it was computed from, which is twice the objective's own
  estimate of the distance above the minimum. It accepts all 7,725; a
  scale-relative step test accepts 7,627 and, applied as the loop's stopping
  rule instead, moves 24 of 166,925 fits that converge today, which is why the
  loop's own rule is untouched. `PlattCalibrator::n_iter` may therefore equal
  `max_iter` on a returned fit, and when it does the fit is at the minimum
  rather than merely the last thing tried.
  <br>
  Two tests pin the rule from both sides over a generated near-constant-score
  region rather than one fixture: refusing on plain exhaustion fails the one
  that watches the region fit, and an acceptance that never refuses fails the
  one that starves the same region of iterations.
- The artifact fuzz sweep's reach floors now detect their own mutator dying.
  They previously could not: with the mutator completely disabled three of the
  four floors still passed, and with nine of ten mutation strategies disabled
  all four did. One floor was anti-correlated with mutator health — an
  unmutated artifact is a valid artifact, so acceptances *rose* from 137 to
  2160 as the mutator died, and a floor on that number rewarded the failure it
  was supposed to catch. The floors now measure the mutator directly, requiring
  each strategy to change the bytes it is given and to produce distinct
  outputs, and measure depth by modelling the version-2 envelope instead of by
  classifying outcomes at the top of the stack, so "these bytes reached a
  payload parser" is decided rather than inferred. Per-decoder reach is floored
  too, which measures the coverage claim the sweep rests on for the first time.
  `the_reach_floors_fail_when_any_one_mutation_strategy_dies` kills each of the
  ten strategies in turn and requires a floor naming it to fail, and
  `the_envelope_model_agrees_with_the_real_decoder` holds the depth model to
  the decoder it models over every envelope field. The comment claiming the old
  counter proved payload-parser reach was false — all three error variants it
  counted are raised by the envelope itself — and is gone.
- The documented ranking guarantee on `calibration` is corrected: it claimed
  that "calibration is monotone, so the **ranking** of any two rows is preserved
  exactly — a threshold-based score such as ROC AUC is unchanged by
  calibration", and that is false. Monotone is weaker than ranking-preserving.
  A `PlattCalibrator` whose fitted slope is negative is a strictly *decreasing*
  map and takes ROC AUC to `1.0 - auc`; `IsotonicRegression` pools, so distinct
  scores can tie, and a fold whose labels run opposite to its scores collapses
  the map to a constant and ROC AUC to `0.5`. Both are reachable through the
  public API, on the held-out calibration fold the same documentation
  *requires*. `Calibrator`, `CalibratedClassifier`, `PlattCalibrator::slope`,
  `IsotonicRegression` and the calibration guide now state the three cases and
  the condition — a strictly increasing map — under which ROC AUC is unchanged.
  The doctest that asserted the false general claim asserted it on the benign
  training-rows case; it now calibrates on held-out rows, asserts the positive
  slope its conclusion rests on, and scores against labels the model does not
  reproduce exactly, so the AUC equality could fail if calibration reordered.
  **Behaviour is unchanged and no fit is rejected.** A negative slope is the
  exact maximum-likelihood answer for its sample: it carries the sign of that
  sample's class mean gap, which `PlattCalibrator::slope` now documents as the
  ranking contract, and the fitted parameters are public so a caller who
  depends on ranking can check them.
- Every one of the crate's 29 capability declarations now carries a doc comment
  saying what it claims and, where a capability is deliberately absent, why —
  up from 8. Four of them contradicted the declaration they sat above and are
  corrected: `CalibratedClassifier<_, IsotonicRegression>` read "Nothing" over a
  declaration that produces probabilities, and `MaxAbsScaler`, `MinMaxScaler`
  and `RobustScaler` each explained an absent capability without ever naming the
  persistence they declare. `scripts/check_documentation_truth.py` gained a rule
  requiring the doc comment, and now reads it from either position rustdoc
  renders — above the `impl` or above the `const CAPABILITIES` — which is where
  those four had been hiding.
- `Pipeline<StandardScaler, LogisticRegression>` now declares the
  `decision_function` capability it has. It exposes `decision_function_into`,
  but computed its declaration by intersecting both parts — and a transformer
  never has a decision function, so the intersection made that field
  structurally unable to be true for any pipeline. A raw decision score is a
  property of the final estimator alone, so it is now taken from there, as
  weighted fitting and multiclass fitting already were. A caller querying the
  capability to decide whether to threshold on raw scores was previously told
  no for a composition that could.
- Report `ModelError::NonFinitePrediction` from `RandomForestRegressor` instead
  of returning a non-finite averaged prediction, matching every other
  regressor.
- Bound what every artifact decoder reserves by the bytes actually present
  rather than by a declared element count. A hostile artifact of roughly 150
  bytes could previously make a scaler, linear model, or forest reserve up to
  32 MB before reporting the truncation it was always going to report.
- Reject a logical tree whose records are laid out in any order other than the
  canonical pre-order the writer produces. Such a layout described a model that
  already had an encoding, so one fitted forest or boosted ensemble had more
  than one accepted artifact.
- Apply the documented 32 MiB reader limit to the legacy version-1 envelope as
  well as version 2. An oversized buffer whose version field read 1 was
  checksummed in full before being rejected.
- `api::Transformer::transform` builds its `DenseMatrix` from the validated
  view the implementation returned rather than from the buffer it was lent.
  `Transformer` is public and unsealed, so the buffer's contents were whatever
  an arbitrary implementation put there: safe external code could write `NaN`
  into it, return a validated view over unrelated storage, and obtain a
  `DenseMatrix` — the crate's validated container — holding non-finite values.
  That matrix was then accepted anywhere a fitted model takes features. The
  trait already documented the returned view as covering "exactly the values
  they wrote"; the default body now relies on that view instead of restating
  the claim over the raw buffer. `StagedPipeline::transform` already worked
  this way, and the two allocating pipeline entry points now agree.

## [0.1.2] - 2026-07-24

### Added

- Deterministic classification and regression metrics with explicit validation
  and undefined-result semantics.
- Checked row/target selection plus deterministic holdout, K-fold, and
  label-stratified dataset splitting.
- Batch-level fitted classifier and regressor scoring across built-in metrics.
- Deterministic serial cross-validation with typed fit closures, ordered fold
  scores, and fold-attributed errors.

### Changed

- Keep third-party provenance and regeneration tooling in local development
  state while retaining FerricML-owned frozen behavior and quality contracts.
- Organize implementation modules by capability and estimator family while
  keeping public paths and model artifacts stable.
- Accelerate exact train/test splitting, stratified quota assignment, standard
  scaling, ridge preprocessing, and logistic Newton fitting.
- Stabilize first-party performance history with repeated scalar inference and
  dedicated ordinary and stratified split workloads.

### Fixed

- Reject non-finite scalar prediction features and non-finite accumulated
  outputs consistently across linear, ridge, logistic, pairwise-ranking, and
  histogram-boosting models.

## [0.1.1] - 2026-07-23

### Added

- Validated sample weights, weighted logistic fitting, and allocation-free raw
  logistic decision scores.
- Bounded version-2 model artifacts with independent payload/component
  versions while retaining strict legacy logistic decoding.
- Dense weighted ordinary least-squares regression with deterministic
  minimum-norm SVD solutions and schema-bound artifacts.
- Dense weighted ridge regression plus runtime switching across forest, linear,
  and ridge regressors.
- Deterministic weighted standard scaling and schema-bound serialized
  scaler-to-logistic, scaler-to-linear, and scaler-to-ridge pipelines with
  caller-owned inference workspace.
- Pairwise linear ranking with explicit tie observations and thresholds,
  schema-bound artifacts, and tie-aware accuracy, Spearman, and Kendall tau-b
  metrics.
- Deterministic dense squared-error histogram gradient boosting with bounded
  bins, leaves, depth, iterations, allocation-free batch prediction, owned
  runtime switching, and schema-bound logical-tree artifacts.

### Fixed

- Honor `LogisticRegressionParams::with_fit_intercept(false)` without silently
  centering features or fitting a folded intercept.

## [0.1.0] - 2026-07-22

### Added

- Validated dense `f32` matrices and classification/regression targets.
- Deterministic random-forest classifiers and regressors with allocation-free
  batch inference.
- Deterministic logistic regression with stable typed parameters and
  allocation-free prediction.
- Versioned, checksummed logistic-regression artifacts with strict feature
  schema verification.
- Frozen estimator and prediction semantics for the supported API subset.
- Generic static-dispatch pipeline and transformer growth seams.
- Frozen correctness fixtures and an on-demand Rust implementation benchmark.

[Unreleased]: https://github.com/kkollsga/ferricml/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kkollsga/ferricml/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/kkollsga/ferricml/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kkollsga/ferricml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kkollsga/ferricml/releases/tag/v0.1.0
