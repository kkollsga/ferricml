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

## Current foundation

The core crate provides validated dense data, logistic regression,
deterministic random-forest classification and regression, private packed
trees, bounded parallel training, allocation-free batch prediction, and a
generic fitted pipeline seam.
Its public estimator vocabulary and observable semantics follow a locked subset
of scikit-learn; typed Rust parameters and caller-owned `_into` methods preserve
validation and performance. See [API and growth](docs/api-and-growth.md) and
[the conformance contract](docs/sklearn-conformance.md).

Logistic models support a versioned, checksummed binary artifact with strict
feature-schema verification. The backend-neutral envelope proposed in
[artifact-envelope.md](docs/artifact-envelope.md) remains deliberately
unfrozen for private tree layouts.

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

The pinned scikit-learn script is a separate correctness/API verifier:

```console
dev-docs/temp/sklearn-1.9/bin/python scripts/sklearn_conformance.py
```

Third-party Rust performance comparisons are never part of root tests or CI.
They live in an independent package and can be run on demand:

```console
make bench-rafor
```

`make bench-self` runs only FerricML's Criterion workloads, while
`make bench-history` records a named first-party local baseline. Neither timing
target is part of shared CI.

Use `--enforce --runner-id apple-m4-local` only on the registered stable
machine. The matched Rafor protocol and thresholds are documented in
[forest-head-to-head.md](docs/forest-head-to-head.md).

## Contributing and releases

GitHub Actions checks formatting, clippy, default and all-feature tests,
documentation, package assembly, and the pinned scikit-learn conformance
fixture. Competitor performance benchmarks remain manual-only and are not part
of CI. See [RELEASING.md](RELEASING.md) for the token-gated crates.io and GitHub
release process.
