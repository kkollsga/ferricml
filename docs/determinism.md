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
| `LogisticRegression`, binary | 1 | `exp`, through the sigmoid |
| `LogisticRegression`, multiclass | — | `exp`, through the row softmax |
| `PairwiseLinearRanker` | 8 | `exp`, through the logistic model it fits |
| `Pipeline<StandardScaler, LogisticRegression>` | 5 | as above |
| `HistGradientBoostingClassifier` | 20 | `ln` for the baseline, `exp` once per row per iteration through the sigmoid |

The boosted classifier is tier 2 for the same reason logistic is, and for the
same *kind* of reason it is unavoidable: its gradient is `y - sigmoid(raw)`, so a
one-ulp libm difference at any iteration changes the next tree's leaf values and
can change a split. Its squared-error sibling stays tier 1 — the two share a
grower, and only the objective evaluates a transcendental. Its fitted bytes are
frozen in `tests/artifact_fingerprints.rs`, so the same green-`main` argument
covers it.

**Tier 3 — per runner only.** Wall-clock timings, throughput, and every
`dev-docs/bench/` number. These are a property of one registered machine and
are never compared across runners; the performance protocol in `CLAUDE.md` owns
them. Nothing in tier 3 is part of the artifact contract.

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
tier 2 estimators, only `sqrt` — in the scaler, in the least-squares weight
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

**Thread count.** Forest training is the only parallel fitting path. It derives
tree `i`'s seed from `i` alone and sorts the finished trees back into index
order before packing, so the fitted forest does not depend on `n_jobs` at all —
a stronger property than the crate-wide promise, and one
`src/ensemble/random_forest/tests.rs` asserts directly by comparing the packed
bytes of a serial fit against a four-worker fit. Every other estimator is
serial.

**Integer and index behaviour.** The shared generator (`src/numeric/rng.rs`) is
SplitMix64 over `u64` with rejection-sampled bounds, so its stream does not
depend on pointer width; a frozen-stream test pins the exact values. Every
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

`tests/artifact_fingerprints.rs` freezes the exact length and SHA-256 digest of
one fitted artifact per estimator. It runs inside `cargo test`, so it executes
in two places:

- **`x86_64-unknown-linux-gnu`**, where CI runs `make gate-full` on
  `ubuntu-latest` for every push and pull request; and
- **`aarch64-apple-darwin`**, the registered local runner, on every `make gate`.

A green `main` therefore *is* the cross-platform evidence, for every estimator
that test covers: linear, ridge, standard scaler, all three scaler pipelines,
the pairwise ranker, both histogram-boosted estimators, the forest regressor and
classifier, and the staged pipeline. Every tier 2 estimator is covered — the
pairwise ranker and the logistic pipeline each contain a fitted logistic model,
and the boosted classifier is fingerprinted directly, so libm differences would
show up as a digest mismatch.

Two gaps are worth naming rather than glossing:

1. **No third platform is tested.** `x86_64-apple-darwin`,
   `aarch64-unknown-linux-gnu`, Windows, and every 32-bit target are unverified.
   Tier 1's reasoning covers them; tier 2's does not, and no promise is made.
2. **A standalone `LogisticRegression` artifact has no frozen fingerprint**,
   only the two nested ones. The fitting path is identical, so the coverage is
   real, but the direct assertion is missing.

Neither gap is closed by this document. Closing the first means adding a
platform to the CI matrix; closing the second means one more entry in the
fingerprint test.

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
