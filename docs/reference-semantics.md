# Frozen reference semantics

FerricML owns a frozen, implementation-independent contract for the observable
behavior of its supported estimators. The contract covers retained parameters,
validation order, output shapes, deterministic predictions, and quality floors.
It is expressed as ordinary Rust tests and committed fixtures so users and CI
do not depend on another machine-learning runtime.

Run the contract with:

```console
make reference-check
```

Exact public API snapshots, behavioral tests, and frozen reference fixtures are
separate contracts. A matching symbol list does not establish behavior, and a
matching prediction fixture does not establish the public Rust surface.

Third-party provenance and regeneration tools are local development materials
under the gitignored `dev-docs/references/` workspace. They may inform fixture
updates, but are not packaged, shipped, or required by CI. Any intentional
fixture change must be reviewed together with the Rust test that states its
meaning.
