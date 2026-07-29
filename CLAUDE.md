# CLAUDE.md — FerricML working rules

FerricML is one pure-Rust crate for lean classical machine learning.
Real-dataset acquisition and application-specific evaluation stay in downstream
projects; synthetic dataset generation is owned by `src/datasets/` behind the
non-default `datasets` feature. This file is the tracked standing policy. The
gitignored working-folder and inbox maps live in `dev-docs/README.md` and
`inbox/README.md`; `dev-docs/learn-from-us.md` is the outward-facing adaptation
guide.

## Working style

- **Root cause before fixes.** Reproduce a defect and capture evidence before
  changing behavior. Scan for the defect class and land the smallest regression
  test that would have caught it.
- **Evidence before optimization.** Establish a release-build baseline for the
  affected workload, make one attributable change, and measure again. A
  plausible optimization without a repeatable improvement is not a fix.
- **Offload long output.** Put logs and dumps in `dev-docs/temp/`, or benchmark
  output in `dev-docs/bench/out/`, and report the path. Keep user-facing reports
  concise.
- Preserve unrelated work in a dirty worktree. Stage explicit paths; never use
  `git add .` or `git add -A` when other work is present.
- Do not change versions, promote the changelog, tag, publish, or push `main`
  outside the `release` skill.

## Architecture and feature boundaries

- `src/data/` owns validated dense inputs and target types.
- `src/datasets/` owns synthetic dataset recipes, deterministic sources, and
  ground truth, feature-gated off the default surface.
- `src/api/` owns estimator vocabulary, typed errors, model swapping, and
  allocation-free batch contracts.
- `src/ensemble/` owns public ensemble facades and their private estimator
  families. Callers must not depend on forest tree layout.
- `src/pipeline/` owns fitted preprocessing/model composition without
  per-row dynamic dispatch.
- `src/artifact/` owns fitted-model persistence and compatibility checks.
- `src/lib.rs` is the only Rust source at crate root; capability facades own
  directories and keep estimator-family child modules private.
- `default = []` is a product boundary. Comparison dependencies must not enter
  the default dependency graph.
- The public model API follows FerricML's locked semantic contract while
  retaining Rust-native typed parameters and caller-owned `_into` fast paths.
  Exact public API snapshots, behavioral tests, and reference fixtures are
  separate contracts; one cannot substitute for another.

## Tiered gates

The routine local gate is:

```bash
make gate
```

The heavier named entry points are `gate-full`, `api-check`, `api-refresh`,
`reference-check`, `exchange-check`, `package-check`, `semver-check`,
`mutants`, `bench-self`, `bench-history`, `bench-diagnostic`, `docs-env`,
`docs-build`, and `docs-serve`. Do not invent substitutes for these contracts.

Run heavier checks only when their surface is touched:

- Feature-boundary changes: `cargo test --locked --all-features` and
  all-feature rustdoc with warnings denied.
- Public Rust API changes: exact API snapshot checks; review an intentional
  snapshot delta in the same commit.
- Packaging changes: build the crate archive and exercise it from an external
  consumer using only extracted contents.
- Reference-semantic changes: regenerate and review the frozen fixtures through
  the local `dev-docs` reference workspace, then run `reference-check`.
- Exchange-surface changes — the generator, the container format, or its Python
  reader: `exchange-check`; it is also release evidence.
- Performance-sensitive changes: follow the performance protocol below.

CI remains authoritative for its parallel matrix. Competitor crates and timing
thresholds never belong in ordinary CI. Release performance evidence is a
first-party-only local gate on the registered stable runner.

## Performance protocol

1. Benchmark an otherwise-idle machine with release builds and fixed workload,
   seed, thread count, and runner identity.
2. Record first-party measurements in `dev-docs/bench/results/`; raw Criterion
   trees and profiles go in `dev-docs/bench/out/`.
3. Re-measure a suspect result. Treat disagreement as noise until reproduced.
4. Preserve immutable per-release FerricML results and compare both the prior
   release and an approximately three-release anchor so slow drift cannot
   ratchet into the baseline.
5. Gate FerricML against its own registered-runner history. Use third-party
   results for diagnosis and design decisions, never as a shared-runner test.

## Code health

- Keep validation at public boundaries and make invalid shapes fail before
  allocation or training work begins.
- Preserve deterministic fitted artifacts for identical data, parameters,
  seed, and thread count.
- Prefer batch dispatch and caller-owned buffers on hot paths; avoid per-row
  trait-object dispatch or hidden allocation.
- Keep model internals private so storage can be compacted without public API
  churn.
- A user-visible behavior or API change needs tests and an `[Unreleased]`
  changelog entry. Internal workflow-only changes do not.
- If an in-scope bug surfaces, reproduce and fix it in a bisectable commit. If
  genuinely out of scope, capture it through `add-todo`; never step over it
  silently.

## Workflow skills

The six workflows under `.claude/skills/` operate local state; their Codex
adapters live under `.agents/skills/`.

- Large feature or non-trivial refactor: `phased-plan`.
- Capture work: `add-todo`.
- Tidy local working state: `dev-docs-cleanup`.
- Incoming coordination: `read-inbox`; outgoing coordination: `notify`.
- Shipping: `release`. Never reproduce its steps ad hoc.

`dev-docs-cleanup` is recommended before a new phased plan and runs at release
completion. `add-todo` is the single authority on todo-entry shape.
After changing either adapter, run `scripts/check_doctrine_adapters.sh` and its
`--self-test`; only convention-file reference wording may differ.

## Inbox hygiene

`inbox/unread/` contains only notes still requiring action. Triage through
`read-inbox`: preserve durable detail in `dev-docs/`, append a FerricML status
footer when actioned, and move the message to `inbox/read/`. Route a message
through `notify` only when the receiving project has a concrete action.

Send and receive share `YYYY-MM-DD-from-<sender>-<topic>.md`. Resolve projects
below the configured development-projects parent; `mcp-servers` is one project
with one inbox, not a collection of component targets.

## Public posts and repository writes

Publishing under the user's identity is prohibited unless the exact final text
is shown and approved immediately beforehand. This includes issues, comments,
reviews, reactions, discussions, email, registry metadata, and state changes
on external repositories. Approval is one publication event and does not carry
forward. Subagents never publish.

Routine work on FerricML's own repository is governed separately: commits are
local by default, and every push requires explicit in-the-moment approval for
the exact ref update. A plan approval is not push approval. The narrow
exception is an approved push's same-scope CI repair loop; it ends when checks
are green, scope changes, progress stalls, or the conversation pivots.

Commit messages are public. Use `type: short description` (`feat`, `fix`,
`docs`, `refactor`, `test`, `chore`, `ci`) and describe the mechanical change,
not sensitive strategy.

## Release safety

FerricML has one crate, one Cargo version, one changelog, and version-matching
`vX.Y.Z` tags. Unless the user explicitly requests another level, every
release increments only the patch component. Semver analysis remains required
evidence but does not override that patch-default policy.

The one exception is a **breaking** diff. `make semver-check` fails when the
diff against the published crate carries changes Cargo would call breaking and
the release step would not be a breaking one — below `1.0` the minor is the
breaking component, so `0.1.2 → 0.1.3` cannot carry one. That failure stops the
release for an explicit decision on the level; it is **never** resolved by
editing a version to satisfy the gate. Breaking there means *observable by a
consumer*: `semver-check` exempts one lint, under conditions it reads out of the
source rather than assumes. A variant discriminant that shifts on an enum which
is `#[non_exhaustive]`, carries no `#[repr(...)]`, pins no explicit
discriminant, and carries data in at least one variant can be reached through no
`as` cast and no `mem::discriminant` comparison, so no consumer can write an
expression that detects it. Every other lint still fails, as does that lint on
an enum failing any of those conditions. An exemption is reported with its
evidence in the release record, never applied silently, and is not a way to
release an observable break as a patch. Recorded decision (2026-07-25): the
release after published `0.1.2` was a minor bump to `0.2.0`. That decision has
been taken and `0.2.0` is the published baseline, so a `semver-check` failure is
a live finding again rather than an expected one. The tag is the
crates.io publication boundary. The `release` skill must goal-check the active
plan, run contracts and package checks, obtain explicit approval before remote
ref updates, wait for green `main`, and only then push the matching tag. Never
force-update branches during cleanup; use worktree-aware checks and safe
deletion, or leave cleanup deferred.
