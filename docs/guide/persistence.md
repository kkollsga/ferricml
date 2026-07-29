# Saving and loading models

A fitted FerricML model serializes to a bounded, versioned, checksummed binary
artifact, and loads back from it. There is no pickle, no `serde` derive on a
private struct, and no format that changes when an internal layout does.

Persistence lives on two traits, so `to_artifact` and `from_artifact` need one
of them in scope — the same way `predict` needs `api::Estimator`. Estimators
implement `artifact::ModelArtifact` and are bound to the one feature schema they
were fitted on; transformers and pipelines implement `artifact::StageArtifact`
and are bound to both the schema they consume and the one they produce. If a
type has these methods at all, it implements the trait: there is no separate
opt-in, so anything you can save you can also compose into a pipeline and save
as part of that.

## The round trip

```rust
use ferricml::artifact::ModelArtifact;
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;

let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())?;
let expected = model.predict(&data.as_view())?;

// The second argument identifies the input feature schema.
let schema = [7_u8; 32];
let bytes = model.to_artifact(schema)?;

// ... write `bytes` to disk, send them over a network, store them anywhere ...

let restored = Ridge::from_artifact(&bytes, schema)?;
assert_eq!(restored.predict(&data.as_view())?, expected);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## The schema hash

That `[u8; 32]` is not decoration. It is a caller-chosen identity for the
feature schema the model expects — the ordered feature names, or whatever
canonical description of your columns you hash.

Decoding **requires** it to match. That is the mechanism that stops a model
being loaded against a differently-shaped or differently-ordered feature set,
which is the failure mode that produces plausible numbers rather than an error.
FerricML cannot know what your columns mean, so it makes you name them and then
holds you to it.

## What the reader checks before it trusts a byte

Loading is not the inverse of a `write` call; it is a validating parser. Before
any model state is constructed, `from_artifact` checks the size limit, verifies
the SHA-256 footer, confirms the magic and envelope version, confirms the
estimator kind and payload version, requires the expected schema identity,
validates every declared length before borrowing a component, bounds every
allocation, rejects non-finite or inconsistent values, and rejects trailing
bytes.

Unknown required flags, payload versions, schema roles, nonzero reserved fields
and component versions are **rejected rather than guessed**. The hard encoded
size limit is 32 MiB.

Two consequences follow:

**Decoding is canonical.** A fitted model has exactly one valid encoding, and a
reader refuses any other byte string that would describe the same model. That is
what makes hashing an artifact a sound way to name a model.

**SHA-256 here is corruption detection, not authentication.** Artifacts are not
signed. Anyone who can modify the bytes can recompute the checksum. Treat an
artifact from an untrusted source the way you would treat any untrusted input.

## Decoding is not fitting

Loading an artifact reconstructs stored values and never re-evaluates a
transcendental function. A model fitted on one platform and loaded on another
*is* the same model.

This matters because FerricML's determinism promise is tiered, and the tiering
is about where the **fit** happens, never about where the model is used. See
[determinism](../determinism.md) for the full statement of which estimators
reproduce byte-identically on any IEEE-754 target and which are scoped to the
platforms FerricML tests.

## Which models persist

Logistic, linear, ridge, both histogram-boosting estimators, both random
forests, both extra-trees, standard scaler, min-max scaler, max-abs scaler,
robust scaler, the pairwise ranker, the three concrete scaler pipelines and any
schema-bound `StagedPipeline` all persist.

Some deliberately do not:

- **The stateless transformers** — `Normalizer`, `Binarizer`,
  `FunctionTransformer` — have no artifact, because there would be nothing in it
  but a feature width the pipeline already validates.
- **The dummy estimators** carry no artifact kind. They are a quality floor, not
  something to ship.
That is the general pattern: where the format cannot express something
faithfully, the answer is an error, not an approximation — and where the state
matters enough to keep, the answer is a new payload version rather than a
reinterpretation of the current one. Logistic regression is the worked example
of the second half: its two original schemas record no solver and therefore name
a `Newton` fit, and a fit under the other solver is written at a *new* payload
version carrying one extra word. Every artifact written before that version
existed keeps its exact bytes and its exact reader.

Whether a given estimator persists is declared in its `Capabilities`, which is
public, semver-relevant surface with its own change-detecting snapshot.

For a `StagedPipeline`, persistence is bound at the type level: asking one that
cannot persist is a **compile error** rather than a runtime failure.

## Runtime model choice

When the model type is a runtime decision, `AnyRegressor` and `AnyClassifier`
are the owned dispatch layer. They match once per batch, not once per row.

```rust
use ferricml::api::AnyRegressor;
use ferricml::artifact::ModelArtifact;
use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::linear_model::{Ridge, RidgeParams};

let data = DenseMatrix::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1)?;
let targets = RegressionTargets::new(vec![1.0, 3.0, 5.0, 7.0])?;

let model = Ridge::fit(&data.as_view(), &targets, RidgeParams::default())?;
let expected = model.predict(&data.as_view())?;

let any = AnyRegressor::from(model);
let bytes = any.to_artifact([7; 32])?;

// Restoring recovers the runtime variant and the payload schema it chose.
let restored = AnyRegressor::from_artifact(&bytes, [7; 32])?;
assert_eq!(restored.predict(&data.as_view())?, expected);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A dispatch artifact is an envelope rather than a format: it carries a dispatch
version and a variant tag, then the selected estimator's own complete artifact
nested whole. The nested bytes are checksummed, schema-bound and validated
exactly as they would be standalone, so a variant tag that disagrees with its
payload fails the nested kind check instead of being reinterpreted.

Generic estimators and pipelines remain the primary zero-overhead layer. Use
dispatch where you genuinely need the runtime choice.

## What is not frozen

FerricML is pre-1.0, and one part of the persistence story is deliberately
unfrozen while the library's shape settles: exact artifact **fingerprints** —
the byte length and digest of a given fitted model — are no longer pinned. What
remains asserted is that encoding a fitted model twice yields identical bytes,
that decoding is canonical, and that round trips preserve the model.

[Determinism](../determinism.md) states precisely what that changed, including
what it means for cross-platform evidence, rather than leaving it implied.

The private compact forest-node layout is also explicitly unfrozen. Forests and
boosted models persist backend-neutral **logical trees**, never their compact
runtime representation, so the fast in-memory layout can be changed without
touching the format.

For the full envelope layout, component framing and per-estimator payload
schemas, see [the artifact envelope](../artifact-envelope.md).
