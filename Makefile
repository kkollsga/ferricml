# FerricML development gates.

SHELL := /bin/bash
PYTHON ?= python3

.PHONY: gate gate-full api-check api-refresh reference-check package-check semver-check bench-self bench-history bench-diagnostic

## Ordinary pre-push gate: formatting, default lint/tests, dependency isolation,
## and the extracted-package external-consumer contract.
gate:
	cargo fmt --all -- --check
	cargo clippy --locked --all-targets -- -D warnings
	cargo test --locked
	$(PYTHON) scripts/check_root_dependency_isolation.py
	$(PYTHON) scripts/check_reference_isolation.py
	$(PYTHON) scripts/check_source_layout.py
	$(PYTHON) scripts/check_source_layout.py --self-test
	$(PYTHON) scripts/check_release_workflow.py
	$(PYTHON) scripts/rust_api_profiles.py self-test
	$(PYTHON) scripts/performance_history.py self-test
	$(MAKE) package-check

## Complete Rust gate. Public API, reference, and performance checks remain
## separately named because they require pinned tools or reference environments.
gate-full: gate
	cargo clippy --locked --all-features --all-targets -- -D warnings
	cargo test --locked --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features

## Compare the public Rust surface with its exact snapshot. The profile records
## derived impls, so dropping a derive from a public type fails here; the
## capability snapshot is a separate mechanism because cargo-public-api cannot
## see const values. Both are part of the same public contract.
api-check:
	$(PYTHON) scripts/rust_api_profiles.py check
	cargo test --locked --test capability_snapshot

## Refresh exact API snapshots only when their complete content input changed.
api-refresh:
	$(PYTHON) scripts/rust_api_profiles.py refresh --skip-if-unchanged
	FERRICML_REFRESH_CAPABILITY_SNAPSHOT=1 cargo test --locked --test capability_snapshot

## Verify FerricML's frozen behavior, shape, validation, and quality contract.
reference-check:
	cargo test --locked --test reference_semantics

## Build the crates.io archive and run a public-API consumer against its extract.
package-check:
	bash scripts/check_packaged_crate.sh

## Informational comparison with the latest published crate; the first release
## reports that no baseline exists and succeeds.
semver-check:
	$(PYTHON) scripts/check_semver.py

## Run only FerricML's root Criterion suite.
bench-self:
	cargo bench --locked --bench forest --bench models --bench boosting -- --noplot

## Capture an immutable versioned FerricML-only summary and compare it with the
## previous release and, when available, the three-release anchor. Set the
## runner identity in dev-docs/bench/runner.json or FERRICML_PERF_RUNNER_ID.
bench-history:
	$(PYTHON) scripts/performance_history.py capture $(PERF_HISTORY_ARGS)

## Capture dated registered-runner evidence without occupying a release slot.
bench-diagnostic:
	$(PYTHON) scripts/performance_history.py capture --diagnostic $(PERF_HISTORY_ARGS)
