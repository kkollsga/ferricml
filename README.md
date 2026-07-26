# FerricML

FerricML is a lean, pure-Rust toolkit for classical machine learning. Its
initial scope is linear and logistic regression, random-forest classification
and regression, reproducible evaluation, and stable model artifacts.

The implementation is domain-agnostic. Dataset acquisition, labeling, and
application-specific evaluation belong in downstream projects rather than the
FerricML crate.

FerricML is MIT licensed. The crate is prepared for crates.io publication;
releases are built from version-matching Git tags after the repository's full
validation gate passes.

## Documentation

FerricML has two documentation surfaces, and they do different jobs.

- **The narrative guide** is built from [`docs/`](docs/index.md) with MkDocs and
  is configured for Read the Docs. Start at [`docs/guide/quickstart.md`](docs/guide/quickstart.md),
  then the guided tour, then the contract documents. Build it locally with
  `make docs-build`, which runs the same `--strict` build the hosted one does.
- **The API reference** is rustdoc, published per release at
  [docs.rs/ferricml](https://docs.rs/ferricml). The guide links to it and never
  reproduces a signature list, so there is no second copy of the API to rot.

Every Rust sample in the guide is a doctest: the markdown under `docs/` is
compiled into the test suite, so a sample that stops compiling or stops
producing the value it claims fails `make gate`.

## Current foundation

The core crate provides validated dense data and sample weights, weighted
logistic regression with raw decision scores, minimum-norm linear and ridge regression,
weighted standard scaling and typed serialized scaler/model pipelines,
pairwise linear ranking and tie-aware rank metrics,
validated classification and regression metrics, deterministic holdout and
fold splitters, direct estimator scoring, serial typed cross-validation,
deterministic random-forest classification and regression, compact histogram
gradient-boosted regression, private packed
trees, bounded parallel training, allocation-free batch prediction, and a
generic fitted pipeline seam.
Its public estimator vocabulary and observable semantics follow FerricML's
locked reference contract; typed Rust parameters and caller-owned `_into`
methods preserve validation and performance. See
[API and growth](docs/api-and-growth.md) and
[the frozen reference contract](docs/reference-semantics.md). Evaluation
semantics and examples are collected in
[evaluation and model selection](docs/evaluation-and-model-selection.md).

Logistic, linear, ridge, histogram-gradient-boosting, standard-scaler, and
supported typed pipeline models support bounded, versioned, checksummed binary
artifacts with strict schema verification; logistic retains legacy-v1
decoding. Boosted ensembles persist backend-neutral logical trees while their
compact runtime layout stays private. See
[artifact-envelope.md](docs/artifact-envelope.md). What reproducing a fitted
artifact promises across operating systems and architectures, versus only per
runner, is stated in [determinism.md](docs/determinism.md).

The pairwise ranker consumes checked pairs over one item matrix and exposes raw
item scores and score differences, not classifier probabilities. Its mirrored
objective and tie normalization are documented in
[pairwise-ranking.md](docs/pairwise-ranking.md).

The histogram regressor is a finite-dense, serial squared-error implementation
with bounded bins and compact private prediction trees. Its deliberately small
parameter subset is documented in
[histogram-gradient-boosting.md](docs/histogram-gradient-boosting.md).

## Verification and benchmarks

The ordinary local gate checks formatting, lint, default tests, root dependency
isolation, and an external consumer compiled from the extracted `.crate`:

```console
make gate
```

Use `make gate-full` for all-target tests and warning-denied rustdoc.
`make api-check` compares the exact public Rust API snapshot; intentional
public changes are captured with `make api-refresh` and reviewed as a normal
diff. `make semver-check` is an informational comparison
with the latest crates.io release and cleanly reports when no first-release
baseline exists.

The frozen behavior, shape, validation, and quality contract has a dedicated
Rust gate:

```console
make reference-check
```

`make bench-self` runs only FerricML's Criterion workloads, while
`make bench-history` records a named first-party local baseline. Neither timing
target is part of shared CI.

The registered FerricML linear, ranking, preprocessing, and compact boosting
workloads and their compatible named-history protocol are documented
in [model-performance.md](docs/model-performance.md). `make bench-diagnostic`
captures dated local evidence without consuming an immutable release-history
slot.

Use `--enforce --runner-id apple-m4-local` only on the registered stable
machine.

## Contributing and releases

GitHub Actions checks formatting, clippy, default and all-feature tests,
documentation, package assembly, and the frozen reference contract. See
[RELEASING.md](RELEASING.md) for the token-gated crates.io and GitHub release
process.
