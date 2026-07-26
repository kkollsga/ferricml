# Determinism policy

FerricML promises an identical fitted artifact for identical data, parameters,
seed, and thread count. That sentence has always been silent about the thing
callers most often need to know: whether it still holds when the *machine*
changes. This document states what FerricML promises across operating systems
and architectures, what it promises only per runner, and — for each claim —
what actually establishes it.

Nothing here is aspirational. Where a claim is verified, the verifying test is
named. Where it is not verifiable from a single machine, it says so rather than
asserting.

## The three tiers

**Tier 1 — identical on every target.** The fitted artifact is byte-identical
on any target with IEEE-754 `binary32` and `binary64`, for identical data,
parameters, seed, and thread count. This covers:

| Estimator | Artifact kind |
|---|---|
| `LinearRegression` | 2 |
| `Ridge` | 3 |
| `Lasso`, `ElasticNet` | — |
| `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler` | 4, 14, 15 |
| `RandomForestRegressor`, `RandomForestClassifier` | 10, — |
| `HistGradientBoostingRegressor` | 9 |
| `StagedPipeline` and the scaler pipelines over the above | 5–7, 16 |
| `DummyClassifier`, `DummyRegressor` | — |

**Tier 2 — identical on the targets FerricML tests.** The fitted artifact is
byte-identical on `x86_64-unknown-linux-gnu` and `aarch64-apple-darwin`, and
FerricML makes no promise beyond those two. This covers every model whose
fitting evaluates a transcendental function:

| Estimator | Artifact kind | Transcendental |
|---|---|---|
| `LogisticRegression`, binary | 1 | `exp` through the sigmoid, and `exp` and `ln` through the loss value the damped step compares |
| `LogisticRegression`, multiclass | — | `exp` through the row softmax, and `exp` and `ln` through the loss value the damped step compares |
| `LogisticRegression`, `LogisticSolver::Lbfgs` | — | `exp` and `ln`, through the loss value and the softmax |
| `PairwiseLinearRanker` | 8 | `exp`, through the logistic model it fits |
| `Pipeline<StandardScaler, LogisticRegression>` | 5 | as above |
| `HistGradientBoostingClassifier` | 20 | `ln` for the baseline, `exp` once per row per iteration through the sigmoid |
| `PlattCalibrator` | — | `exp`, through the sigmoid |

The two Newton rows gained `ln` on 2026-07-26, when the exact step was damped by
an Armijo backtracking search. Deciding how far to step means comparing the
objective at two points, and the binary log-loss value is
`log_sum_exp(0, raw) - target * raw`. Their tier is unchanged — both already
evaluated `exp` — but the entry is listed because the set of libm functions a
fitting path calls is what the tier rests on, and a reader checking the claim
should find every one of them named.

`PlattCalibrator` was added to the table on the same date. It was always tier 2
by this document's own rule and had simply never been listed, which is the kind
of omission the rule at the bottom of this file exists to prevent. Its fit has no
artifact kind — nothing in the crate persists a calibrator, and
`CalibratedClassifier` says so in its own capability declaration — so the tier
governs its fitted `slope`, `intercept` and centre rather than any bytes.

The boosted classifier is tier 2 for the same reason logistic is, and for the
same *kind* of reason it is unavoidable: its gradient is `y - sigmoid(raw)`, so a
one-ulp libm difference at any iteration changes the next tree's leaf values and
can change a split. Its squared-error sibling stays tier 1 — the two share a
grower, and only the objective evaluates a transcendental. Its fitted bytes are
**no longer frozen** — see "What establishes each claim" below for what that
does and does not still guarantee.

**Tier 3 — per runner only.** Wall-clock timings, throughput, and every
benchmark number FerricML records. These are a property of one registered
machine and are never compared across runners; the project's performance
protocol owns them. Nothing in tier 3 is part of the artifact contract.

## Why tier 1 holds

An algorithm produces identical bits on every IEEE-754 target when three things
are fixed: the arithmetic, the evaluation order, and the integer behaviour.

**The arithmetic.** IEEE-754 requires `+`, `-`, `*`, `/`, and `sqrt` to be
correctly rounded, so each returns the same bits everywhere. Rust performs no
fast-math reassociation and does not contract multiply-add into a fused FMA, so
the operations a source line names are the operations that execute. Comparisons
and `f32`/`f64` conversions are likewise exact.

Tier 1's fitting paths use only those operations. That is a checkable claim,
not an assurance: searching `src/` for `exp`, `ln`, `log*`, `powf`, `powi`,
`tanh`, `cbrt`, `mul_add`, and the trigonometric functions returns, outside the
tier 2 estimators and outside `#[cfg(test)]` code, only `sqrt` — in the scaler, in the least-squares weight
transform, and in the forest's feature-subset size. The one previous exception
was the forest's population-variance impurity, which used `powi(2)`; `powi` is
not an IEEE-mandated operation and carries no correct-rounding guarantee, so it
is now written as an explicit multiplication instead. Re-run that search when
adding an estimator: a new transcendental on a fitting path moves that
estimator from tier 1 to tier 2, and this table with it.

**The evaluation order.** The accumulation policy in `src/numeric/mod.rs` is
the authority, and it is binding on every module. Reductions run in ascending
row order and ascending column order within a row; `sum_in_order` is that rule
written as code, and it seeds from `-0.0` because that is IEEE addition's true
identity — seeding from `+0.0` would turn a sum of negative zeros positive and
change the fitted bytes. No path may reorder a reduction for speed.

**Iterative solvers are covered by the same reasoning, not exempted from it.**
Coordinate descent and limited-memory BFGS both converge to a point that depends
on the *sequence* of iterates, so their determinism is the determinism of every
reduction inside them. Both reduce through `sum_in_order` in a fixed index
order; coordinate descent sweeps columns in ascending order; and the L-BFGS line
search bisects rather than interpolating, so its next trial is a function of the
current bracket alone rather than of a fitted polynomial's rounding. Neither
solver returns an iterate it did not converge — an exhausted budget is
`ModelError::SolverDidNotConverge` — so there is no path on which a stopping
condition that fired differently on two machines produces two "fitted" models
instead of one model and one error. Coordinate descent evaluates no
transcendental at all, which is why `Lasso` and `ElasticNet` are tier 1 despite
being iterative.

**A step-length search is part of that sequence, and the damped Newton path's is
narrower than the L-BFGS one's.** `optimize::damping` tries `1, 1/2, 1/4, ...`:
exact powers of two, each exactly representable, and a trial point is one
multiplication and one subtraction per coordinate. It does not interpolate, and
unlike bisection its next trial does not even depend on a bracket — only on the
halving index — so the property the paragraph above rests on is preserved and
tightened rather than weakened. The acceptance test is a pair of `f64`
comparisons, which IEEE-754 evaluates exactly; a comparison that fired
differently on two machines would need the *objective value* to differ, which is
the `exp`/`ln` scope tier 2 already states, not a new source of divergence. So
the number of halvings, and therefore the iterate sequence, is a function of the
data, the parameters, the seed and the thread count alone. `src/optimize/damping.rs`
asserts the reproducibility of the accepted factor directly, and asserts that a
zero-length step is refused rather than accepted — without which a search could
report progress it did not make and spend a whole budget standing still.

**Thread count.** Forest training is the only parallel fitting path. It derives
tree `i`'s seed from `i` alone and sorts the finished trees back into index
order before packing, so the fitted forest does not depend on `n_jobs` at all —
a stronger property than the crate-wide promise, and one
`src/ensemble/random_forest/tests.rs` asserts directly by comparing the packed
bytes of a serial fit against a four-worker fit. Every other estimator is
serial.

**Integer and index behaviour.** The shared generator (`src/numeric/rng.rs`) is
SplitMix64 over `u64` with rejection-sampled bounds, so its stream does not
depend on pointer width; a frozen-stream test pins the exact values. "Shared"
is singular and checkable: every seeded path in the crate — forest bootstrap
sampling, feature subsetting, permutation inspection, and every shuffled
dataset split — draws from that one generator, and the `rng-single-source`
layout rule fails the gate if a second definition appears outside
`src/numeric/`. Every
count the artifact format admits — feature widths, node counts, tree counts —
is bounded far below `2^32`, so no `as usize` conversion behaves differently on
a 32-bit target.

**The one target class this excludes.** A target without IEEE `binary32` and
`binary64`, or one evaluating `f32`/`f64` in x87 excess precision (32-bit x86
without SSE2), is outside tier 1. Rust's mainstream x86 targets require SSE2,
so this is a statement about exotic targets rather than a practical caveat.

## Why tier 2 is only tier 2

`exp` and `ln` are not IEEE-mandated operations. Rust's `f32::exp` and
`f64::exp` call the platform's math library, and different libm
implementations — glibc's on Linux, Apple's on macOS — are free to differ in
the last unit in the last place for the same argument. A logistic solver
iterates on sigmoid values, so a one-ulp difference at any iteration can change
the converged coefficients, which changes the artifact bytes.

That is why logistic, the logistic pipeline, and the pairwise ranker are scoped
to the platforms actually tested rather than to all targets. It is a statement
about the guarantee, not an observed divergence: on the two tested targets the
bytes agree today.

## What establishes each claim

**Artifact fingerprints are unfrozen while the library shape settles** (user
decision, 2026-07-25). `tests/artifact_fingerprints.rs` previously pinned an
exact length and SHA-256 digest per estimator; it now asserts only that
encoding a fitted model twice yields identical bytes. FerricML is pre-1.0 with
no users, and a byte-stability promise to nobody constrains the format while
biasing the design toward whatever is cheap not to change.

**What that changed, stated plainly.** A pinned digest was doing two jobs, and
only one of them was the promise being retired. It also served as the
*cross-platform* channel: the same constant was checked on
`x86_64-unknown-linux-gnu` (CI, `make gate-full` on every push) and on
`aarch64-apple-darwin` (the registered local runner, every `make gate`), so a
green `main` was itself the evidence that both platforms produced identical
bytes. Nothing else compares the two, because no single test run sees both.

So today:

- **Still established:** determinism on one machine — same data, parameters,
  seed and thread count produce identical bytes, asserted directly. Canonicity
  and round-tripping, in `tests/artifact_hardening.rs`. Numerical agreement
  with the reference, in `tests/reference_semantics.rs`, which is
  platform-sensitive in its own right and would catch gross divergence.
- **No longer established:** that a tier 2 estimator's fitted bytes are
  *byte-identical across platforms*. Tier 1's argument is unaffected — it rests
  on IEEE-754 arithmetic and fixed evaluation order, not on a test. Tier 2's
  rested on the fingerprint, and now rests on reasoning alone until the
  fingerprints return.

Two gaps predating this decision, still open:

1. **No third platform is tested.** `x86_64-apple-darwin`,
   `aarch64-unknown-linux-gnu`, Windows, and every 32-bit target are unverified.
2. **A standalone `LogisticRegression` artifact was never fingerprinted
   directly**, only the two nested ones.

Re-freezing is the intended end state, not an abandoned idea: when the API and
feature set settle, restoring the digests is one test run per estimator, and it
restores the cross-platform evidence with them.

## What a caller may rely on

- **Refitting reproduces the model.** Same data, parameters, seed, and thread
  count give the same artifact bytes — on one machine always, and across
  machines per the tier above.
- **Artifact bytes identify a model.** Decoding is canonical: a fitted model has
  exactly one valid encoding, and a reader rejects any other byte string that
  would describe it, so hashing an artifact is a sound way to name a model.
  `docs/artifact-envelope.md` describes what the reader validates.
- **Decoding is not fitting.** Loading an artifact reconstructs stored values
  and never re-evaluates a transcendental, so a model fitted on one platform
  and loaded on another *is* the same model. Tier 2 is about where the fit
  happens, never about where it is used.
- **Inference is bit-stable on one platform** for a given model and input.
  Across platforms, tier 1 models predict identically; a logistic model's
  probabilities go through `exp` at inference too, so they carry tier 2's scope.

## Adding an estimator

Answer three questions and record them here in the same commit:

1. Does any fitting path evaluate something outside `+ - * /` and `sqrt`? If
   yes, the estimator is tier 2 and the operation is named above.
2. Is any reduction order left to the compiler or the scheduler? If yes, fix it
   through `sum_in_order` before proceeding.
3. Is fitting parallel? If yes, each partition's result must depend only on its
   own index, and results must be combined in index order.

Then add the estimator to `tests/artifact_fingerprints.rs`. An estimator with
no frozen fingerprint has no cross-platform evidence, whatever tier the
reasoning puts it in.
