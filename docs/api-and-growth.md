# API and growth structure

FerricML uses a stable estimator vocabulary expressed as safe, typed Rust:
fitted estimators expose `n_features_in`, exact retained
parameters through `get_params`, and the operations `fit`, `predict`,
`predict_proba`, and `transform`. Allocating convenience methods delegate to
caller-owned `_into` primitives. Classification probabilities are row-major,
with one column per sorted entry in `classes`.

The supported contract and deliberate differences are frozen in
[`reference-semantics.md`](reference-semantics.md). FerricML avoids dynamic
object mutation, implicit numeric coercion, backend-native persistence, and
magic parameter strings. Typed parameter builders and validated dense `f32`
containers are intentional Rust interfaces, while their names and observable
estimator meaning follow the reference contract.

## Extension structure

- `api` owns backend-independent estimator categories, errors, retained
  parameter access, batch-level runtime model enums, and the compile-time
  capability descriptor. `Capabilities` records only what varies between
  estimator types and is not already guaranteed by the type system — weighted
  fitting, artifact persistence, and multiclass fitting — so it never
  becomes a second parameter system; it is carried by `HasCapabilities`, a
  generic trait rather than an associated constant on the object-safe
  categories, which must stay dyn-compatible. `decision_function` records
  whether a fitted classifier exposes a raw, unsquashed score. Producing
  probabilities is a separate declared capability, carried by
  `ProbabilisticClassifier` rather than required of every `Classifier`, and is
  not what `decision_function` records. The tag exists because Rust has no
  runtime attribute lookup, so a meta-estimator generic over a classifier
  cannot otherwise discover that the type it holds has one. Note what a tag can
  and cannot do — it makes the capability discoverable, not callable, because a
  decision function is an inherent method rather than part of the object-safe
  contract, so a consumer that must *call* one still needs a bound naming a
  trait that carries it.
- `data` owns validated row-major inputs, targets, and sample weights. Sample
  weights are the crate's only weighting concept: no estimator takes a
  `class_weight`, because a per-class weight is a function of the label and so
  already a per-row weight. The balanced rule is a documented, tested recipe for
  building `SampleWeights`, which keeps one weighting notion for every estimator,
  one capability flag, and one validation order to freeze.
- `ensemble` owns public ensemble estimators and parameter types; each private
  estimator family owns its validation, training, persistence conversion, and
  compact representation below the public facade. A forest's weighted entry
  points treat a sample weight as a fractional row count: it multiplies the
  bootstrap replication count, and the product is what every impurity, split
  threshold test, leaf mean, and leaf distribution accumulates. The minimum
  split and leaf sizes therefore count weight rather than rows, which is what
  makes an integer weight the same fit as repeating the row. A row of weight
  zero is not in the training sample at all, so it is also excluded from the
  bootstrap draw rather than consuming one of it.
- `pipeline` composes fitted transformers and an estimator generically. Its
  `with_transformed` path uses caller-owned workspace and static dispatch.
  `Pipeline` holds one transformer; concrete standard-scaler pipelines provide
  allocation-free prediction and explicit persistence for logistic, linear, and
  ridge estimators. `StagedPipeline` holds two or more stages as a
  `TransformerStack` tuple and can fit the whole composition in one pass, each
  stage on the previous stage's output. Every handoff is validated before the
  composition exists, and one caller-owned workspace is split into a disjoint
  segment per stage, so multi-stage inference allocates nothing. Prediction
  stays on the generic callback rather than per-category convenience methods,
  which cannot coexist as inherent methods of one name.
- `linear_model` separates estimator facades from private numerical seams, and
  now holds two solver families rather than one. The closed-form fits — ordinary
  least squares and ridge — stay closed form. `LogisticRegression` selects
  between exact Newton steps and a matrix-free limited-memory quasi-Newton path
  through `LogisticSolver`, defaulting to Newton and keeping every fitted
  artifact it has ever produced; the matrix-free path exists because a joint
  multinomial system is `classes * parameters` square, so the exact one refuses
  shapes it cannot allocate. `Lasso` and `ElasticNet` are fitted by cyclic
  coordinate descent, which is the solver an L1 penalty requires: the penalty is
  not differentiable at zero, and that is precisely what makes it produce
  coefficients that are exactly zero rather than small.
- `metrics` owns deterministic classification and regression measures with
  explicit errors for invalid or undefined inputs.
- `model_selection` owns validated index partitions, deterministic holdout and
  fold iterators, batch estimator scoring, serial typed cross-validation, and
  typed parameter search. Splitters remain independent of estimator internals,
  while fitting stays in caller-provided closures. A search grid is built from
  the parameter type's own builder methods rather than string keys, and search
  evaluates candidates through cross-validation and the shared scorer contract
  rather than carrying an evaluation path of its own.
- `preprocessing` owns fitted transformer implementations and their state.
  `StandardScaler` uses deterministic two-pass population statistics and
  accepts sample weights; `MinMaxScaler` and `MaxAbsScaler` fit order
  statistics, which no per-sample weight can move, so they declare no weighted
  entry point. Each carries a degenerate column explicitly — a constant column
  scales by one and a zero-magnitude column passes through — rather than
  dividing by an empty range. The shared non-finite preflight is stated once
  for the family, so a finite input that scales to a non-finite `f32` is
  reported at its first row-major location before anything is written.
- `ranking` owns pair construction, the pairwise linear estimator, and
  denominator-safe rank metrics. It remains distinct from `Classifier`: raw
  ranking scores and pair margins are not probabilities.
- `artifact` owns stable envelope identity, bounded decoding primitives, and
  artifact errors. It does not expose backend tree layouts.
- `dummy` owns baseline estimators that ignore their features: the
  majority-class classifier, whose probabilities are the observed class
  frequencies, and the mean regressor. They are the quality floor a real
  estimator has to beat and the reference implementation of the estimator
  contract, so they carry no tunable behavior, no weighted entry point, and no
  artifact kind.
- `calibration` owns post-hoc probability calibration. A calibrator is a fitted
  monotone map of one model score onto a probability, so it changes how
  confident a prediction is without changing which way round two rows are
  ordered. `IsotonicRegression` is the non-parametric one and is also a
  standalone monotone regressor over a single-column matrix. It states its tie
  convention rather than inheriting one: observations sharing an input value are
  pooled into their mean *before* pool-adjacent-violators runs, which is forced
  rather than chosen — a function of one input takes one value at one input —
  and it makes the fit independent of observation order. Prediction interpolates
  linearly between fitted points and holds the end values outside the fitted
  range rather than extrapolating a trend the fit never observed.
  `PlattCalibrator` is the parametric one, a two-parameter logistic fit reached
  through the shared objective contract rather than a third logistic solver. It
  regresses on Platt's prior-corrected targets rather than on the raw labels,
  which is what keeps the fit finite when the score separates the classes
  perfectly — with raw labels that problem has no finite maximum-likelihood
  solution, and the resulting map would assert exactly the certainty
  calibration exists to remove.
  `CalibratedClassifier<C, K>` composes an already-fitted classifier with an
  already-fitted calibrator and is itself an ordinary `Classifier`, so it
  reaches the scorer, cross-validation, and permutation-importance paths without
  any of them learning that calibration exists. The score it calibrates is the
  wrapped model's positive-class probability, which is the one score the
  `ProbabilisticClassifier` contract requires; that is what lets the wrapper be
  generic over that public contract rather than over the estimators FerricML
  ships. The
  calibration rows are always a caller-supplied parameter, never the wrapped
  model's own training rows taken implicitly. Predicted labels are the argmax of
  the *calibrated* probabilities, so a row whose probability crosses the
  decision point does change label; because the map is monotone, the ranking of
  any two rows — and therefore any threshold-sweeping score such as ROC AUC — is
  unchanged. The composition declares its capabilities per calibrator rather
  than by intersecting the wrapped model's: it owns fitted parts, so weighted
  fitting, persistence, and multiclass fitting are declared away structurally.
  Both calibrators declare probabilities, which is what the wrapper exists to
  produce, and a Platt composition additionally gains a `decision_function` the
  wrapped model never had.
- `inspection` owns model-agnostic attribution. Permutation importance works
  through the public batch prediction and scoring contracts only, so it needs
  no estimator cooperation and exposes no model internals. It holds no scoring
  logic of its own: it calls the same caller-owned-buffer scoring entry point
  cross-validation does, and takes the orientation of the result from the
  score's own declaration. Its per-feature
  values are quality losses, oriented so a larger number always means a more
  important feature whichever direction the underlying metric improves in.

Classification covers an arbitrary observed class set. `ClassTargets` carries
the sorted, deduplicated labels a fit observed, and that set is the probability
column order; nothing assumes the labels are contiguous or zero-based.
`LogisticRegression::fit_multiclass` is one joint multinomial optimization
whose probabilities are the softmax of a centred score vector with no pinned
reference class, and `RandomForestClassifier::fit_multiclass` averages per-tree
probability vectors rather than voting on per-tree labels. Both keep their
original binary fit unchanged beside the new one, including its asymmetric
single-row decision score, because the two parametrizations are different
models rather than two spellings of one. Probability rows are never
renormalized: they sum to one only within `n_classes` `f32` ulps, which is a
frozen part of the contract rather than a tolerance to tighten later.

`AnyClassifier` and `AnyRegressor` remain the owned runtime-swap layer. They
match once per batch; the regressor variants cover forests, linear regression,
ridge, and histogram gradient boosting. Both persist through a dispatch envelope
that nests the selected estimator's own artifact whole, so restoring one
restores the runtime variant and the payload schema that variant chose. Generic estimators and pipelines
remain the primary zero-overhead layer.

Meta-layers compose capabilities rather than restating them. A dispatch enum
and a fitted pipeline declare the intersection of their variants' or parts'
capabilities, so an undispatched caller is never promised more than it gets,
while `capabilities` on a dispatch value reports the variant actually held.
Capabilities a composition cannot have at all — weighted fitting, when the
composition owns only already-fitted parts — are declared away structurally
instead of being inherited.

A declared capability is public, semver-relevant surface, and it is frozen by
**two** snapshots rather than one. `cargo-public-api` records that a type
declares capabilities but never which — the API profile contains
`pub const Ridge::CAPABILITIES: Capabilities` with no value — so a flipped
declaration would otherwise be invisible to the exact API check. The declared
*values* therefore live in their own generated companion snapshot beside the
API profile, and `api-check` compares both. The two are closed against each
other: every `HasCapabilities` impl the API profile records must have a value
row, and every row must correspond to such an impl, so a new estimator cannot
declare a capability that nothing change-detects. Behavioral agreement between
a declaration and its estimator is a third, separate contract, proven
generically by the conformance battery; a snapshot cannot substitute for it,
and it cannot substitute for a snapshot.

A const value is the profile's one remaining blind spot, and only because it is
one. The profile used to have a second, larger one: it omitted *auto-derived*
impls, so none of the crate's 159 `derive(…)` attributes reached the baseline
and removing `#[derive(Clone)]` from a public parameter type was a breaking
change the exact API check reported as clean. The capture now omits only
blanket and auto-trait impls, so `impl core::clone::Clone for
ferricml::linear_model::RidgeParams` is a baseline row like any other and
losing it is a reviewable diff. Because a detector that has never been shown to
fail proves only that today's tree is clean, `make gate` runs
`scripts/rust_api_profiles.py self-test`, which strips every `Clone`, `Debug`
and `PartialEq` impl row out of the frozen baseline in turn and asserts each
removal is reported — and asserts the capture command has not quietly gone back
to omitting them.

Artifact support is a trait rather than a list. A fitted type persists exactly
when it implements `artifact::ModelArtifact` or `artifact::StageArtifact`, and
implementing one *is* writing the encoder — the estimator kind and the identity
a composed payload records come with it. There is no second registration to
remember, which is what previously let seven estimators ship a working encoder
that the composition layer could not see.

A `StagedPipeline` then *computes* its persistence from its parts rather than
being gated on them: every stage's declaration and the estimator's, intersected.
Asking a composition that cannot persist for its bytes is still a compile error,
because `to_artifact` keeps the bound — but declaring is not, so a composition
that honestly persists nothing still has a capability vocabulary and is still
checked by the conformance battery. Gating the *declaration* on the bound had
conflated "declares no artifact" with "cannot declare anything".

That a composition needs no kind of its own is possible because a staged
composition uses a single artifact kind and records which concrete parts it
holds inside the payload: order, estimator type, and stage count are all checked
on decode, so one composition never decodes as another. `Pipeline`'s three
concrete compositions predate that scheme and keep their own artifact kinds.

## What the documentation checker does not check

Documentation that contradicts behaviour has been this crate's most active
defect class, so `make gate` runs `scripts/check_documentation_truth.py`. It
reads prose — rustdoc comments in `src/` and the narrative pages in `docs/`,
which rustdoc never sees at all — and reports four kinds of claim that stopped
being true:

- a capability declaration whose doc comment does not name a capability the
  declaration turns on;
- a `Type::member` reference to a type this crate declares, where the member
  does not exist;
- a generic bound written in prose that the documented item does not carry;
- a repository path cited in prose that is not there.

Each rule is proven twice by `--self-test`: once against a synthetic violation,
and once against a tree with the rule's input removed, because a check that
passes by not looking is the failure mode this crate keeps rediscovering. Two
historical defects — the "declares nothing" comment above a probability
declaration, and a documented `CrossValidationError` variant named `Scoring`
that never existed — are reconstructed there as inputs. Prose about an entity
that is *supposed* to be absent has to be written that way round, because the
checker cannot tell a cautionary example from a live reference, and a rule with
an escape hatch is a rule with a hole in it.

**The blind spots are the point of this section.** The checker verifies that a
*named entity* exists; it cannot verify a *claim about behaviour*. The five
`calibration` sentences that told a reader probabilities were required of every
`Classifier` were plain English, and only the one that spelled the bound as
`C: Classifier` was mechanically detectable. Completeness claims are worse:
"the only", "every", "exactly one" and their relatives appear on some 600 prose
lines here, and no rule can separate a true one from a false one. Requiring
each to cite its enforcement was considered and rejected on evidence — the two
completeness defects this crate has actually had, `determinism.md`'s
cross-platform argument and this page's own claim about the API profile's
single blind spot, **both cited their evidence correctly** and were wrong about
what that evidence established. A rule demanding pointers would have passed
both while flagging hundreds of sound sentences, and a checker that cries wolf
gets disabled. Those claims are caught by review, and the honest position is
that they are written down as unchecked rather than presumed safe.
