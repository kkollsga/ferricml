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

**A tree's minimum split and leaf sizes bound weight, not rows.** The reference
bounds `min_samples_split` and `min_samples_leaf` by the number of rows in a
node and lets a sample weight move nothing else; FerricML bounds them by the
node's total weight. This is a deliberate divergence, taken so that an integer
sample weight is the same fitted model as repeating that row *unconditionally* —
under row counting the equivalence holds only while the constraint does not
bind, because duplicating a row changes the row count the bound is compared
against. Unweighted fitting is unaffected: a node's weight is its row count when
every weight is one, so every frozen unweighted fixture is unchanged. The
weighted tree fixtures pin the bound at one, where it cannot bind, so they
compare the weighted impurity and leaf arithmetic — where the two agree exactly.

**Class weighting is a caller-side transformation, not a parameter.** No
FerricML estimator takes a `class_weight`. A per-class weight is a function of
the label and therefore already a per-row weight, so it is expressed by building
`data::SampleWeights` and calling a `fit_weighted` entry point. This is a
deliberate divergence from the reference's estimator-level parameter, decided
once for the whole crate: one weighting concept means one thing for every
estimator to implement, one capability flag to declare, one validation order to
freeze, and no question of how a class weight and a sample weight compose. The
balanced rule — inverse class frequency scaled so the total weight stays the row
count — is documented and tested as a recipe on `SampleWeights` rather than
hidden behind a parameter value, and a caller wanting a different rule writes a
different closure instead of waiting for another accepted string.

Third-party provenance and regeneration tools are local development materials
under the gitignored `dev-docs/references/` workspace. They may inform fixture
updates, but are not packaged, shipped, or required by CI. Any intentional
fixture change must be reviewed together with the Rust test that states its
meaning.
