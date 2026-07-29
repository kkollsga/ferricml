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
every weight is one, so every frozen unweighted fixture is unchanged.

The divergence is frozen in two pieces, because one fixture cannot state both
halves. Most weighted tree fixtures hold the bound at one with every weight at
least one, where the two rules are provably the same function, and so compare
the weighted impurity and leaf arithmetic — where the two must agree exactly.
One further fixture holds weights that straddle one against a split bound of
three, which is the region where the rules genuinely differ, and it records the
reference's outputs *and* FerricML's differing ones side by side. It separates
them in both directions at once: a two-row child weighing less than one is a
leaf the reference admits and FerricML refuses, and a two-row node weighing four
is a leaf the reference refuses to split and FerricML splits. Adopting row
counting would make that fixture fail, which is the property it exists for.

**A penalized linear model's `alpha` is measured against a mean, and its zeros
are positive.** `Lasso` and `ElasticNet` minimize the weighted squared error
divided by twice the total sample weight, plus `alpha * l1_ratio * ||b||_1` and
`0.5 * alpha * (1 - l1_ratio) * ||b||_2^2`. Two consequences are contract, not
implementation detail. First, that `alpha` is a different quantity from
`Ridge`'s, which accompanies an *undivided* squared-error term: the two agree at
`ridge_alpha = alpha * total_weight`. Second, the penalty applies to raw-scale
coefficients — fitting centers the design when an intercept is requested and
never rescales the columns — so a feature's units decide how strongly it is
penalized, and scale-free selection is obtained by composing a
`StandardScaler` in front where the transformation is explicit and persists.

A coefficient the fit removes is exactly `0.0` and **positively signed**. The
reference stores a negatively signed zero for a coefficient shrunk from below;
FerricML deliberately does not, because a coefficient the fit removed has no
sign to carry and a signed zero is a different byte pattern for a
mathematically identical model. Convergence is the largest absolute coefficient
change across one full sweep, which is FerricML's own criterion rather than the
reference's duality gap; the two agree on the minimizer and not on the
iteration at which they stop, so a fixture comparison runs both far past their
default tolerances. Exhausting `max_iter` is
`ModelError::SolverDidNotConverge`, never a fit that stopped part way and looks
like one that arrived.

**A logistic solver is selectable, and the default is the matrix-free one.**
`LogisticSolver::Lbfgs` minimizes the same objective without forming a
second-order system, which is what lets a joint multinomial fit exceed the exact
path's parameter ceiling. The ceiling is a property of the selected solver's
storage rather than of the model, so a shape the exact path accepts still takes
the exact path and produces the identical fit. `LogisticSolver::Newton` was the
default until a cross-library sweep measured 2.4x to 50.6x against it over 24
measurements for a truth-distance identical to five decimal places in 23 of
them. `tol` is the largest coefficient update under `Newton` and the mean
objective's gradient norm under `Lbfgs` — different quantities, documented
rather than conflated, so the default meaning of `tol` moved with the default
solver even though its value did not. Both solvers persist: payload versions 1
and 2 record no solver and therefore name a `Newton` fit, and versions 3 and 4
carry one extra word naming it.

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

**One degeneracy rule for every scaler.** A column with a spread of exactly
zero keeps a divisor of one, so a constant feature survives as a constant rather
than producing a non-finite value. The test is exact equality with zero, and it
is the same test for the standard, min-max, max-abs, and robust scalers. A
column whose spread is merely *small* is real data: it is scaled normally, and
if that overflows `f32` the batch is refused with the offending row and column
before anything is written. FerricML deliberately does not use a magnitude
threshold, which would silently decline to scale a legitimately tiny-scaled
column, and would give the crate two degeneracy rules where one will do.

**Robust scaling quantiles use linear interpolation between the two bracketing
order statistics** — Hyndman–Fan type 7 — applied uniformly, including at the
median. Small samples do not contain the value a percentile asks for, so the
interpolation rule is a documented semantic choice rather than an
implementation detail, and it is carried as a typed parameter at every internal
call site so a second rule can be added without silently repointing the first
consumer at it.

**A missing inverse is refused rather than silently applied as the identity.**
`preprocessing::FunctionTransformer` takes an optional inverse function, and
asking a transformer that was not given one to invert is
`api::ModelError::NoInverseFunction`. The reference keeps `inverse_transform`
present on such a transformer — it answers to both `hasattr` and `dir()` — and
returns its input unchanged; its `check_inverse` option warns about a *wrong*
inverse and says nothing about a missing one. Returning the input is
indistinguishable from a successful recovery of the originals, which is a worse
failure than a refusal, so this is a deliberate divergence.

Invertibility is deliberately **not** a declared capability, and that is not an
oversight left over from the probability split. A `Capabilities` field is an
associated constant, so a declaration is a statement about a fitted *type*.
Whether a `FunctionTransformer` inverts is a property of the fitted *instance* —
it is decided by whether `with_inverse_func` was called — so neither `true` nor
`false` would be true of the type, and a bit would have to be wrong for half of
them in whichever direction it was set. The instance-level question is already
answerable exactly, through `get_params().inverse_func()`, which is an `Option`
rather than a claim. Nor is there anywhere the answer gets erased: the
probability tag exists for runtime dispatch, where the concrete type is gone by
construction, and FerricML has no runtime-dispatch transformer type. A consumer
that *requires* an inverse states that as a bound on a trait, which is a proof
rather than a tag; the trait is owed to the first such consumer and to nothing
before it.

**Parameters FerricML does not claim** are recorded as non-claims rather than
left as gaps, because an unclaimed parameter and a divergent one are different
things. `RobustScaler` does not offer scaling the quantile spread to the spread
of a standard normal distribution: that needs an inverse-normal-CDF primitive
with its own accuracy contract, and one optional flag does not justify it.
`FunctionTransformer` applies an **elementwise** `fn(f32) -> f32` and does not
accept a map that reads a whole row or column; the elementwise case covers the
common transformations, and anything else is expressed by implementing
`api::Transformer` directly, which is the honest way to say the transformation
is the caller's rather than FerricML's.

Third-party provenance and the tooling that regenerates these fixtures are
maintainer-side development materials, kept in a workspace outside the crate.
They may inform a fixture update, but they are not packaged, shipped, or
required by CI: what a consumer receives is the frozen fixture values and the
Rust tests that state their meaning. Any intentional fixture change must be
reviewed together with the test that states what it means.
