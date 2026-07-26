# FerricML development gates.

SHELL := /bin/bash
PYTHON ?= python3

# Documentation-site toolchain. DOCS_PYTHON matches the interpreter line pinned
# in .readthedocs.yaml so the local build and the hosted build agree.
DOCS_PYTHON ?= python3.12
DOCS_VENV ?= .venv-docs

.PHONY: gate gate-full api-check api-refresh reference-check package-check semver-check bench-self bench-history bench-diagnostic docs-env docs-build docs-serve

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
	$(PYTHON) scripts/performance_history.py self-test
	$(MAKE) package-check

## Complete Rust gate. Public API, reference, and performance checks remain
## separately named because they require pinned tools or reference environments.
gate-full: gate
	cargo clippy --locked --all-features --all-targets -- -D warnings
	cargo test --locked --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features

## Compare the public Rust surface with its exact snapshot. The capability
## snapshot is a separate mechanism because cargo-public-api cannot see const
## values; both are part of the same public contract.
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

## Install the exactly pinned documentation toolchain into a local virtualenv.
## Nothing here enters the crate's dependency graph.
docs-env:
	$(DOCS_PYTHON) -m venv $(DOCS_VENV)
	$(DOCS_VENV)/bin/pip install --quiet --disable-pip-version-check \
		--requirement requirements/docs.txt

## Build the narrative site the way Read the Docs does. `--strict` is the local
## equivalent of that platform's fail_on_warning, so a dead internal link or a
## page missing from the nav fails here too. Deliberately not a gate member: it
## needs a Python toolchain the Rust gates must not require.
docs-build: | $(DOCS_VENV)
	$(DOCS_VENV)/bin/mkdocs build --strict

## Serve the site locally with live reload.
docs-serve: | $(DOCS_VENV)
	$(DOCS_VENV)/bin/mkdocs serve

$(DOCS_VENV):
	$(MAKE) docs-env
