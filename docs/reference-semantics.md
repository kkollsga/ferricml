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

Classification probabilities are **not renormalized**. A row sums to one only
within `n_classes` `f32` ulps, and that bound is part of the contract rather
than a tolerance to be tightened later: a normalizing pass would move mass
without measuring anything, and it would cost a second pass over every
prediction row that the allocation-free inference contract exists to avoid.
Tests assert the bound, and assert that the deviation is real rather than
hypothetical.

Third-party provenance and regeneration tools are local development materials
under the gitignored `dev-docs/references/` workspace. They may inform fixture
updates, but are not packaged, shipped, or required by CI. Any intentional
fixture change must be reviewed together with the Rust test that states its
meaning.
