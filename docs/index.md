# FerricML

FerricML is a lean, pure-Rust toolkit for classical machine learning: linear and
logistic regression, penalized linear models, decision trees, random forests,
extra-trees, histogram gradient boosting, preprocessing, fitted pipelines,
reproducible evaluation, and stable binary model artifacts.

It is one crate with two dependencies, no default features, and no `unsafe` in
its own source. Fitting a model, scoring it, saving it, and loading it are all
safe, typed Rust.

## What FerricML is

- **A library of fitted estimators with a frozen observable contract.** Retained
  parameters, validation order, output shapes, predictions and quality floors
  are pinned by committed tests and fixtures, not by whatever the code happens
  to do this week. See [Frozen reference semantics](reference-semantics.md).
- **Deterministic by construction.** Identical data, parameters, seed and thread
  count give an identical fitted artifact — and for most estimators, on any
  IEEE-754 target. The promise is tiered and each tier states what establishes
  it: see [Determinism](determinism.md).
- **Allocation-conscious at inference.** Every prediction path has a
  caller-owned `_into` primitive that writes into a buffer you supply. The
  allocating convenience method delegates to it rather than the other way
  round, so the fast path is the real path.
- **Typed rather than stringly-typed.** Parameters are builder methods on a real
  parameter type, so a misspelled parameter is a compile error and a parameter
  search grid crosses a `usize` axis with an `f32` axis without erasing either.
  There are no magic strings.
- **Domain-agnostic.** It computes; it does not fetch, label, or interpret.

## What FerricML is not

- **Not a deep-learning framework.** No autodiff, no GPU, no neural networks.
- **Not a data-loading or feature-engineering platform.** Dataset acquisition,
  labeling and application-specific evaluation belong in downstream projects.
- **Not a Python library, and not bindings to one.** There is no Python in the
  crate's dependency graph. (The documentation *site* is built with Python
  tooling that lives outside the crate entirely.)
- **Not sparse-, missing-, or categorical-aware yet.** Inputs are validated,
  finite, dense, row-major `f32`. These representations are planned work, not
  quiet omissions.
- **Not a place where semantics change silently.** Deliberate differences from
  the reference implementations FerricML is checked against are written down and
  tested as differences. Where FerricML declines to claim a parameter, it says
  so rather than leaving a gap.

## This site, and docs.rs

FerricML has two documentation surfaces and they do different jobs.

**docs.rs is the API reference.** Every released version's complete rustdoc —
every type, every method, every signature — is published automatically at
[docs.rs/ferricml](https://docs.rs/ferricml), regenerated from the code on each
release. It cannot drift from the crate, because it *is* the crate.

**This site is the narrative guide**: concepts, contracts, and how to accomplish
a task. The boundary between the two is a stated rule, not a habit:

!!! note "The docs.rs boundary"

    A page here may **name** a type or function and link it to docs.rs. A page
    here may **not** reproduce its signature list, its parameter table, or its
    method inventory. Where you need the exact surface, this site sends you to
    docs.rs, which regenerates from the code and cannot rot.

The reason is specific rather than stylistic. FerricML's public API is held by
an exact snapshot contract *because* the surface moves; a hand-maintained second
copy on a documentation site would diverge from that snapshot without anything
noticing. Duplicated API listings rot, so this site does not keep any.

The one thing this site does duplicate is **code**, and only under a guarantee.
Every Rust sample on every page of this site is a doctest: the markdown files
are compiled into the crate's test suite, so a sample that stops compiling — or
stops producing the result it claims — fails FerricML's ordinary `make gate`
alongside everything else. A sample that has silently rotted is worse than no
sample, so none of them can.

## The contracts

These are the deep documents. They are reference material rather than an
on-ramp, and they are the authority where anything else disagrees.

| Document | What it settles |
| --- | --- |
| [API and growth](api-and-growth.md) | The estimator vocabulary, what each module owns, and how the library is allowed to grow |
| [Frozen reference semantics](reference-semantics.md) | The behavioral contract, and every deliberate divergence from the reference |
| [Determinism](determinism.md) | What reproduces across machines, what reproduces only per runner, and what proves each |
| [Artifact envelope](artifact-envelope.md) | The on-disk model format, and what a reader validates before trusting a byte |
| [Histogram gradient boosting](histogram-gradient-boosting.md) | The boosted regressor's deliberately small parameter subset |
| [Evaluation and model selection](evaluation-and-model-selection.md) | Metrics, splitters, scoring, cross-validation, and typed parameter search |
| [Pairwise ranking](pairwise-ranking.md) | The pairwise objective, tie handling, and why a ranking score is not a probability |
| [Performance](model-performance.md) | The first-party benchmark workloads and the named-history protocol |
| [Dependency security](security.md) | The `cargo audit` policy and the reviewed maintenance warnings |

## Licence

MIT. FerricML is pre-1.0: the API is expected to move, and where a contract is
deliberately unfrozen while the shape settles, the document that owns it says so
in as many words.
