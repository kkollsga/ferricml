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

**A logistic solver is selectable, and the default is unchanged.**
`LogisticSolver::Newton` is the default and produced every fitted artifact
FerricML has ever written. `LogisticSolver::Lbfgs` minimizes the same objective
without forming a second-order system, which is what lets a joint multinomial
fit exceed the exact path's parameter ceiling. The ceiling is a property of the
selected solver's storage rather than of the model, so a shape the exact path
accepts still takes the exact path and produces the identical fit. `tol` is the
largest coefficient update under `Newton` and the mean objective's gradient norm
under `Lbfgs` — different quantities, documented rather than conflated. Neither
logistic payload schema records a solver, so a model fitted under a non-default
one has no artifact representation rather than bytes that would decode as a
`Newton`-provenance model.

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

Third-party provenance and regeneration tools are local development materials
under the gitignored `dev-docs/references/` workspace. They may inform fixture
updates, but are not packaged, shipped, or required by CI. Any intentional
fixture change must be reviewed together with the Rust test that states its
meaning.
