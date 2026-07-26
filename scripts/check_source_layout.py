#!/usr/bin/env python3
"""Enforce FerricML's domain-owned private source layout.

Each boundary is a named rule over a source tree, so `--self-test` can assert
that every rule still fires against a synthetic tree that violates it. A rule
that silently stopped matching would otherwise pass both the check and its own
self-test.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"

# Modules that own fitted estimators or their families. The shared numeric
# kernels sit below all of them and must never depend on one.
ESTIMATOR_MODULES = (
    "ensemble",
    "linear_model",
    "pipeline",
    "preprocessing",
    "ranking",
    "tree",
)

# The observable signature of a quantile definition: the rule vocabulary and the
# evaluator that consumes it. Both belong to the shared numeric kernels alone.
QUANTILE_DEFINITION_MARKERS = ("enum QuantileRule", "fn quantile_sorted")

# The observable signature of a seeded generator: SplitMix64's mixing function
# and the golden-ratio increment its state advances by. A module that carries
# either is deriving pseudo-random values of its own.
RNG_DEFINITION_MARKERS = ("fn mix64", "0x9e37_79b9_7f4a_7c15")


def read_if_present(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def tree_text(directory: Path) -> str:
    """Every source file under `directory`, including its child modules.

    This is the *only* reader a dependency rule may use. A non-recursive twin
    exists — [`shallow_text`] — but exclusively as the self-test's foil, and
    [`rules_reading_a_module_directory`] counts a rule that calls it as still
    owing its child-module proof, so a rule cannot escape the obligation by
    switching readers. A facade that grew into a directory of
    per-family child modules would silently narrow every rule reading it: the
    check would keep passing while the dependency it forbids sat one level down.
    That is not hypothetical — `preprocessing`, `pipeline`, `metrics`,
    `model_selection`, `linear_model`, `ensemble` and `ranking` each grew from
    one file into a directory, and nine rules were still reading facades only
    when the 2026-07-26 independent review found them. Every rule below is
    proven against a synthetic violation placed in a *child* module, because a
    violation in the facade cannot tell the two readers apart.
    """
    return "\n".join(path.read_text() for path in sorted(directory.rglob("*.rs")))


def shallow_text(directory: Path) -> str:
    """The facade-only reader, kept solely so `--self-test` can fail against it.

    No rule calls this, and a rule that started to would keep — not lose — its
    child-module obligation, which is then unsatisfiable. `self_test`
    substitutes it for [`tree_text`] and asserts that every child-module
    violation goes *unreported* — which is what makes each of those violations
    a proof that its rule reads the tree rather than the facade. Without this,
    a synthetic violation demonstrates only that the rule matches a string
    somewhere.
    """
    return "\n".join(path.read_text() for path in sorted(directory.glob("*.rs")))


def crate_root_is_lib_only(root: Path) -> list[str]:
    root_sources = sorted(path.name for path in (root / "src").glob("*.rs"))
    if root_sources != ["lib.rs"]:
        return [f"crate-root Rust sources must be only lib.rs, found {root_sources}"]
    return []


def obsolete_root_implementations_are_gone(root: Path) -> list[str]:
    source = root / "src"
    return [
        f"obsolete root implementation exists: {removed.relative_to(root)}"
        for removed in (source / "forest.rs", source / "boosting")
        if removed.exists()
    ]


def artifact_is_runtime_neutral(root: Path) -> list[str]:
    if "crate::ensemble" in tree_text(root / "src" / "artifact"):
        return ["artifact foundation depends on a concrete ensemble runtime"]
    return []


def ensemble_families_stay_private(root: Path) -> list[str]:
    facade = read_if_present(root / "src" / "ensemble" / "mod.rs")
    if "pub mod random_forest" in facade or "pub mod hist_gradient_boosting" in facade:
        return ["private ensemble estimator families are exposed as child modules"]
    return []


def numeric_depends_on_no_estimator(root: Path) -> list[str]:
    text = tree_text(root / "src" / "numeric")
    return [
        f"numeric kernels depend on estimator module {module}"
        for module in ESTIMATOR_MODULES
        if f"crate::{module}" in text
    ]


def quantile_definition_lives_only_in_numeric(root: Path) -> list[str]:
    """One quantile definition, in the shared kernels, named at every call site.

    A quantile is not one function: the defensible interpolation rules disagree
    on small samples, so a second definition growing beside the first would let
    two transformers silently freeze fitted values against different meanings of
    the same word. The rule is a documented semantic choice carried as a typed
    parameter, which only works while there is exactly one place that defines
    it. The primitive has to exist for this rule to mean anything, so its
    absence is itself a finding rather than a silently vacuous pass.
    """
    source = root / "src"
    numeric = tree_text(source / "numeric")
    if not any(marker in numeric for marker in QUANTILE_DEFINITION_MARKERS):
        return ["quantile primitive is missing from the shared numeric kernels"]
    findings = []
    for path in sorted(source.rglob("*.rs")):
        if "numeric" in path.relative_to(source).parts:
            continue
        text = path.read_text()
        findings.extend(
            f"quantile definition re-derived outside numeric: "
            f"{path.relative_to(root)} defines {marker!r}"
            for marker in QUANTILE_DEFINITION_MARKERS
            if marker in text
        )
    return findings


def rng_definition_lives_only_in_numeric(root: Path) -> list[str]:
    """One seeded generator, in the shared kernels, serving the whole crate.

    `src/numeric/mod.rs` rule 6 states this and calls it binding on every
    module: a seed has to mean the same thing in every estimator, in inspection
    and in every shuffled split, which only holds while one definition exists.
    A second copy does not announce itself by producing a wrong number — the
    one this rule was written for was character-identical to the shared stream
    and emitted the same values for the same seed, while its doc comment
    claimed to be deliberately independent of it. Nothing detected that for as
    long as it existed, so the marker is textual and the boundary is the
    directory: the mixing function and the increment belong to `src/numeric/`
    and nowhere else. Modules that need a derived seed call
    `derive_tree_seed` or `derive_repetition_seed` rather than re-deriving one.

    Both markers have to be present for this rule to mean anything, so their
    absence is itself a finding rather than a silently vacuous pass.

    Two boundaries are deliberate. The scan covers `src/` and not `tests/`:
    the integration crates cannot see a `pub(crate)` generator, and their
    SplitMix64 copies drive fixture generation and artifact fuzzing rather than
    a fitted model, so a seed there means one thing to one test. The markers are
    textual, so a duplicate written as some *other* generator would pass — this
    rule closes the copy-the-shared-one path, which is the one that produces two
    streams claiming to be one, not every conceivable source of randomness.
    """
    source = root / "src"
    numeric = tree_text(source / "numeric")
    if not all(marker in numeric for marker in RNG_DEFINITION_MARKERS):
        return ["seeded generator is missing from the shared numeric kernels"]
    findings = []
    for path in sorted(source.rglob("*.rs")):
        if "numeric" in path.relative_to(source).parts:
            continue
        text = path.read_text()
        findings.extend(
            f"generator definition re-derived outside numeric: "
            f"{path.relative_to(root)} defines {marker!r}"
            for marker in RNG_DEFINITION_MARKERS
            if marker in text
        )
    return findings


def preprocessing_sits_below_composition(root: Path) -> list[str]:
    """Transformers are consumed by composition, never the other way round.

    A fitted transformer is a self-contained map from one dense batch to
    another. Pipelines, model selection, and the estimator families are its
    *consumers*; naming one inside `preprocessing` would invert that dependency
    and make a transformer's behaviour depend on what it happens to be composed
    into. Keeping the arrow pointing one way is what lets the same scaler be
    used standalone, as a pipeline stage, and inside a cross-validated search
    without three variants of it existing. The module has to exist for the rule
    to mean anything, so its absence is itself a finding rather than a silently
    vacuous pass.
    """
    text = tree_text(root / "src" / "preprocessing")
    if not text:
        return ["preprocessing module is missing"]
    return [
        f"preprocessing depends on its own consumer {module}"
        for module in ("pipeline", "model_selection", "ensemble", "linear_model")
        if f"crate::{module}" in text
    ]


def inspection_uses_only_public_surfaces(root: Path) -> list[str]:
    """Inspection must name no estimator family and no persistence internals.

    Model-agnostic inspection is defined by working through the public
    prediction and scoring contracts alone. Naming a concrete estimator or the
    artifact layer is the observable symptom of it reaching past them.
    """
    text = tree_text(root / "src" / "inspection")
    return [
        f"inspection depends on non-public-surface module {module}"
        for module in (*ESTIMATOR_MODULES, "artifact")
        if f"crate::{module}" in text
    ]


def calibration_uses_only_public_surfaces(root: Path) -> list[str]:
    """Calibration wraps a fitted model through its public contract alone.

    A calibrator is defined by being addable *around* any fitted classifier, so
    it reaches the wrapped model only through the public prediction and scoring
    surfaces. Naming a concrete estimator family — or the persistence layer
    below one — is the observable symptom of a wrapper that works for the models
    FerricML happens to ship rather than for the contract, which is exactly the
    generality the module exists to prove. The module has to exist for the rule
    to mean anything, so its absence is itself a finding rather than a silently
    vacuous pass.
    """
    text = tree_text(root / "src" / "calibration")
    if not text:
        return ["calibration module is missing"]
    return [
        f"calibration depends on non-public-surface module {module}"
        for module in (*ESTIMATOR_MODULES, "artifact")
        if f"crate::{module}" in text
    ]


def loss_depends_on_no_estimator(root: Path) -> list[str]:
    """The objective contract sits below every solver that consumes it.

    A loss exists so that linear and ensemble solvers share one definition of
    what they minimize. Naming a concrete estimator module inside it would
    invert that dependency and reintroduce the per-estimator fusion of
    objective and solver the contract was built to remove.
    """
    text = tree_text(root / "src" / "loss")
    return [
        f"loss contract depends on estimator module {module}"
        for module in ESTIMATOR_MODULES
        if f"crate::{module}" in text
    ]


def optimize_depends_only_on_loss_and_numeric(root: Path) -> list[str]:
    """A solver minimizes an objective, never a named model.

    The optimizer seam exists so that a new objective reaches a matrix-free
    solver without shipping a solver of its own. Naming a concrete estimator
    module inside it would invert that dependency and rebuild the
    per-estimator fusion of objective and update rule the seam removes, and it
    would make the solver's own proofs depend on whichever models happen to
    exist. Only `loss` and `numeric` sit below it. The module has to exist for
    the rule to mean anything, so its absence is itself a finding rather than a
    silently vacuous pass.
    """
    text = tree_text(root / "src" / "optimize")
    if not text:
        return ["optimize module is missing"]
    permitted = ("loss", "numeric")
    return [
        f"optimize depends on module {module} outside {permitted}"
        for module in (*ESTIMATOR_MODULES, "api", "artifact", "data")
        if f"crate::{module}" in text
    ]


def capability_descriptor_names_no_estimator(root: Path) -> list[str]:
    """The capability descriptor is vocabulary, not a registry of estimators.

    Each estimator declares its own capabilities next to its implementation.
    Naming a concrete estimator module inside the descriptor is the observable
    symptom of the reverse arrangement — a central table that every new
    estimator has to be added to, which is the combinatorics problem the
    descriptor exists to remove.
    """
    text = read_if_present(root / "src" / "api" / "capabilities.rs")
    return [
        f"capability descriptor names estimator module {module}"
        for module in ESTIMATOR_MODULES
        if f"crate::{module}" in text
    ]


def baselines_depend_on_no_estimator(root: Path) -> list[str]:
    """A baseline must be an independent quality floor.

    Reusing a real estimator's machinery would make the floor move with the
    thing it is supposed to measure, so the baselines name no estimator module.
    """
    text = tree_text(root / "src" / "dummy")
    return [
        f"baseline estimators depend on estimator module {module}"
        for module in ESTIMATOR_MODULES
        if f"crate::{module}" in text
    ]


def composition_families_stay_private(root: Path) -> list[str]:
    """Transformer families and pipeline internals stay behind their facades.

    `preprocessing` and `pipeline` each grew from one file into a directory of
    per-family and per-concern child modules. Exposing one as `pub mod` would
    make its internal layout — fitted state, stage order, artifact component
    encoding — part of the public API, which is exactly what the facades exist
    to prevent. Only re-exported types cross the boundary.
    """
    findings = []
    for facade in ("preprocessing", "pipeline"):
        text = read_if_present(root / "src" / facade / "mod.rs")
        findings.extend(
            f"{facade} facade exposes child module: {line.strip()}"
            for line in text.splitlines()
            if line.strip().startswith("pub mod ")
        )
    return findings


def metrics_depend_on_no_estimator(root: Path) -> list[str]:
    """Metrics score values, never estimators.

    A metric takes expected and predicted values and returns a number. Naming an
    estimator module inside it is the observable symptom of a metric that only
    works for one kind of model, which is what keeps the evaluation vocabulary
    reusable by callers whose model FerricML does not ship.
    """
    text = tree_text(root / "src" / "metrics")
    return [
        f"metrics depend on estimator module {module}"
        for module in ESTIMATOR_MODULES
        if f"crate::{module}" in text
    ]


def evaluation_families_stay_private(root: Path) -> list[str]:
    """Metric and model-selection internals stay behind their facades.

    Both grew from single files into directories of per-concern child modules —
    averaging, confusion counts, curve sweeps, one file per splitter family.
    Exposing one as `pub mod` would make that arrangement public API, so a later
    regrouping would be a breaking change. Only re-exported types cross.
    """
    findings = []
    for facade in ("metrics", "model_selection"):
        text = read_if_present(root / "src" / facade / "mod.rs")
        findings.extend(
            f"{facade} facade exposes child module: {line.strip()}"
            for line in text.splitlines()
            if line.strip().startswith("pub mod ")
        )
    return findings


def split_families_stay_private(root: Path) -> list[str]:
    """One file per splitter family stays behind the split facade.

    `model_selection/split/` is a directory of per-family child modules —
    grouped, group-shuffle, repeated, time-ordered. The parent facade re-exports
    the splitter types, so which file a splitter lives in is an internal
    arrangement. Exposing one as `pub mod` would publish that arrangement and
    make a later regrouping a breaking change.
    """
    text = read_if_present(root / "src" / "model_selection" / "split" / "mod.rs")
    return [
        f"split facade exposes child module: {line.strip()}"
        for line in text.splitlines()
        if line.strip().startswith("pub mod ")
    ]


def search_consumes_the_scorer_seam(root: Path) -> list[str]:
    """Hyperparameter search reaches metrics only through the scorer contract.

    Search evaluates candidates by calling cross-validation, which calls the one
    caller-owned scoring entry point. Naming `crate::metrics` inside search is
    the observable symptom of a second scorer dispatch growing beside that one —
    the duplication that already had to be removed from permutation importance
    once. The module has to exist for the rule to mean anything, so its absence
    is itself a finding rather than a silently vacuous pass.
    """
    module = root / "src" / "model_selection" / "search.rs"
    text = read_if_present(module) + tree_text(
        root / "src" / "model_selection" / "search"
    )
    if not text:
        return ["hyperparameter search module is missing"]
    if "crate::metrics" in text:
        return ["search re-derives scoring instead of consuming the scorer contract"]
    return []


def tree_sits_below_the_estimators_that_consume_it(root: Path) -> list[str]:
    """The shared grower sits below every estimator family that grows a tree.

    A standalone tree and one member of a forest are the same tree, and the
    only way that stays a fact about the code rather than a claim about two
    implementations is if one grower serves both. That requires the dependency
    to run one way: `ensemble` consumes `tree`, never the reverse. Naming a
    consumer inside `tree` is the observable symptom of the inversion, and it
    would let the grower specialise for the forest — which is precisely the
    coupling that would make a standalone tree diverge from a forest member
    again. The module has to exist for the rule to mean anything, so its
    absence is itself a finding rather than a silently vacuous pass.
    """
    text = tree_text(root / "src" / "tree")
    if not text:
        return ["tree module is missing"]
    return [
        f"tree grower depends on estimator module {module}"
        for module in ESTIMATOR_MODULES
        if module != "tree" and f"crate::{module}" in text
    ]


def tree_family_stays_private(root: Path) -> list[str]:
    """The packed layout and the split search stay behind the tree facade.

    `tree` publishes estimators and parameter types; it does not publish how a
    node is stored, how leaves are packed into their parent's child slots, or
    how candidates are swept. Exposing a child module as `pub mod` would make
    that arrangement public API and turn a later compaction of the layout into
    a breaking change — the same rule the ensemble, preprocessing, pipeline,
    metrics, and model-selection facades already carry, and the one that keeps
    "callers must not depend on tree layout" enforceable rather than merely
    intended.
    """
    text = read_if_present(root / "src" / "tree" / "mod.rs")
    return [
        f"tree facade exposes child module: {line.strip()}"
        for line in text.splitlines()
        if line.strip().startswith("pub mod ")
    ]


def forest_core_sits_below_the_ensembles_that_consume_it(root: Path) -> list[str]:
    """The shared ensemble core sits below every bagged ensemble facade.

    A random forest and an extremely randomized ensemble are one implementation
    with two split searches and two artifact kinds. That stays true only while
    the dependency runs one way: each facade consumes `ensemble::forest`, and
    the core names none of them. Naming a facade inside the core is the
    observable symptom of the inversion, and it is what would let the shared
    averaging, seeding, or metadata layout specialise for whichever ensemble
    was written first — the same failure the standalone tree already has a rule
    against, at the level above it. The module has to exist for the rule to
    mean anything, so its absence is itself a finding.
    """
    text = tree_text(root / "src" / "ensemble" / "forest")
    if not text:
        return ["ensemble forest core is missing"]
    return [
        f"forest core depends on ensemble facade {facade}"
        for facade in ("random_forest", "extra_trees", "hist_gradient_boosting")
        if f"ensemble::{facade}" in text or f"super::{facade}" in text
    ]


# The two traits that *are* the persistence contract. Both must exist for the
# rule below to mean anything, so their absence is itself a finding.
PERSISTENCE_TRAIT_MARKERS = ("pub trait ModelArtifact", "pub trait StageArtifact")


def persistence_is_declared_only_through_the_trait(root: Path) -> list[str]:
    """Persisting is implementing the trait, never writing an inherent method.

    A fitted type used to declare persistence twice: once by writing an inherent
    `to_artifact`, and again by being listed as composable. The second
    declaration was a separate act of remembering, and seven estimators shipped
    a working encoder without it — invisible to the type system, and dropping
    every composition that ended in one out of the conformance battery.

    Implementing `ModelArtifact` or `StageArtifact` now *is* writing the
    encoder, so there is no second place to omit a type from. That only stays
    true while the inherent form cannot come back, and it is exactly the form
    that would come back: a new estimator's author writes the encoder where the
    other methods are. A trait implementation spells its methods `fn`; only an
    inherent one spells them `pub fn`, which makes the difference a rule can
    see.
    """
    source = root / "src"
    contract = "\n".join(
        path.read_text() for path in sorted((source / "artifact").rglob("*.rs"))
    )
    missing = [
        marker for marker in PERSISTENCE_TRAIT_MARKERS if marker not in contract
    ]
    if missing:
        return [
            f"persistence contract is missing from src/artifact: {sorted(missing)}"
        ]
    findings = []
    for path in sorted(source.rglob("*.rs")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            for method in ("to_artifact", "from_artifact"):
                if stripped.startswith(f"pub fn {method}("):
                    findings.append(
                        f"persistence declared as an inherent method rather than "
                        f"through the trait: {path.relative_to(root)}:{number} "
                        f"defines `pub fn {method}`"
                    )
    return findings


RULES: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("crate-root-lib-only", crate_root_is_lib_only),
    ("obsolete-root-implementations", obsolete_root_implementations_are_gone),
    ("artifact-runtime-neutral", artifact_is_runtime_neutral),
    ("ensemble-families-private", ensemble_families_stay_private),
    ("numeric-below-estimators", numeric_depends_on_no_estimator),
    ("quantile-single-source", quantile_definition_lives_only_in_numeric),
    ("rng-single-source", rng_definition_lives_only_in_numeric),
    ("preprocessing-below-composition", preprocessing_sits_below_composition),
    ("inspection-public-surfaces-only", inspection_uses_only_public_surfaces),
    ("loss-below-estimators", loss_depends_on_no_estimator),
    ("optimize-below-estimators", optimize_depends_only_on_loss_and_numeric),
    ("capability-descriptor-neutral", capability_descriptor_names_no_estimator),
    ("baselines-independent", baselines_depend_on_no_estimator),
    ("composition-families-private", composition_families_stay_private),
    ("metrics-below-estimators", metrics_depend_on_no_estimator),
    ("evaluation-families-private", evaluation_families_stay_private),
    ("split-families-private", split_families_stay_private),
    ("search-consumes-the-scorer-seam", search_consumes_the_scorer_seam),
    ("calibration-public-surfaces-only", calibration_uses_only_public_surfaces),
    ("tree-below-estimators", tree_sits_below_the_estimators_that_consume_it),
    ("tree-family-private", tree_family_stays_private),
    ("forest-core-below-facades", forest_core_sits_below_the_ensembles_that_consume_it),
    ("persistence-through-the-trait", persistence_is_declared_only_through_the_trait),
)


def violations(root: Path = ROOT) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(root))
    return found


def write_clean_tree(root: Path) -> Path:
    """Write the smallest source tree that satisfies every rule.

    Every directory a dependency rule reads carries a **child module**, because
    that is the shape the crate's modules keep growing into and the shape a
    facade-only reader stops seeing. The child modules exist so each rule's
    synthetic violation can be written below the facade rather than into it;
    a violation in the facade passes under either reader and proves nothing
    about which one the rule uses.

    The quantile primitive lives in `numeric/quantile/mod.rs`, and the
    generator in `numeric/rng/mod.rs`, rather than in flat files beside the
    facade, for the same reason: `quantile-single-source` and
    `rng-single-source` are the rules whose recursion protects a *non-firing*
    property — the primitive being found — so the discriminating assertion is
    that this clean tree passes at all. Under a facade-only reader each reports
    its primitive missing, which `self_test` asserts directly through
    [`CLEAN_TREE_PROVEN_RECURSION`].
    """
    source = root / "src"
    for relative, text in {
        "lib.rs": "pub mod artifact;\npub mod ensemble;\nmod numeric;\n",
        "artifact/mod.rs": "//! artifact\npub(crate) use self::inner::Thing;\nmod contract;\nmod component;\n",
        "artifact/contract.rs": (
            "//! contract\npub trait ModelArtifact { fn to_artifact(&self); }\n"
            "pub trait StageArtifact { fn to_artifact(&self); }\n"
        ),
        "artifact/component/mod.rs": "//! component encoding\n",
        "ensemble/mod.rs": "//! ensemble\nmod random_forest;\npub use random_forest::Forest;\n",
        "ensemble/random_forest/mod.rs": "//! forest\nuse crate::tree::grow_tree;\n",
        "ensemble/forest/mod.rs": "//! forest core\nmod training;\nmod seeding;\n",
        "ensemble/forest/training.rs": "//! training\nuse crate::tree::grow_tree;\n",
        "ensemble/forest/seeding/mod.rs": "//! seeding\nuse crate::numeric::kernel;\n",
        "tree/mod.rs": "//! tree\nmod grower;\nmod split;\npub use grower::DecisionTreeRegressor;\n",
        "tree/grower.rs": "//! grower\nuse crate::numeric::kernel;\npub struct DecisionTreeRegressor;\n",
        "tree/split/mod.rs": "//! split search\nuse crate::numeric::kernel;\n",
        "numeric/mod.rs": "//! numeric\npub(crate) fn kernel() {}\n",
        "numeric/rng/mod.rs": (
            "//! rng\npub(crate) struct OwnedRng { state: u64 }\n"
            "const INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;\n"
            "fn mix64(value: u64) -> u64 { value }\n"
        ),
        "numeric/stream/mod.rs": "//! stream\n",
        "numeric/quantile/mod.rs": (
            "//! quantile\npub(crate) enum QuantileRule { Linear }\n"
            "pub(crate) fn quantile_sorted() {}\n"
        ),
        "loss/mod.rs": "//! loss\nmod objective;\nmod boosting;\n",
        "loss/objective.rs": "//! objective\nuse crate::numeric::kernel;\n",
        "loss/boosting/mod.rs": "//! boosting objective\nuse crate::numeric::kernel;\n",
        "optimize/mod.rs": "//! optimize\nmod lbfgs;\nmod newton;\n",
        "optimize/lbfgs.rs": "//! lbfgs\nuse crate::numeric::kernel;\nuse crate::loss::Objective;\n",
        "optimize/newton/mod.rs": "//! newton\nuse crate::loss::Objective;\n",
        "inspection/mod.rs": "//! inspection\nmod permutation;\nmod scoring;\n",
        "inspection/permutation.rs": "//! permutation\nuse crate::api::Regressor;\n",
        "inspection/scoring/mod.rs": "//! scoring\nuse crate::api::Regressor;\n",
        "api/mod.rs": "//! api\nmod capabilities;\n",
        "api/capabilities.rs": "//! capabilities\npub struct Capabilities;\n",
        "dummy/mod.rs": "//! dummy\nmod classifier;\nmod strategy;\n",
        "dummy/classifier.rs": "//! baseline\nuse crate::api::Classifier;\n",
        "dummy/strategy/mod.rs": "//! strategy\nuse crate::api::Classifier;\n",
        "preprocessing/mod.rs": "//! preprocessing\nmod standard_scaler;\n",
        "preprocessing/standard_scaler/mod.rs": "//! scaler\n",
        "pipeline/mod.rs": "//! pipeline\nmod staged;\npub use staged::StagedPipeline;\n",
        "pipeline/staged.rs": "//! staged\npub struct StagedPipeline;\n",
        "metrics/mod.rs": "//! metrics\nmod confusion;\nmod curves;\npub use confusion::ConfusionMatrix;\n",
        "metrics/confusion.rs": "//! confusion\npub struct ConfusionMatrix;\n",
        "metrics/curves/mod.rs": "//! curves\npub struct RocCurve;\n",
        "model_selection/mod.rs": "//! model selection\nmod split;\npub use split::Split;\n",
        "model_selection/search.rs": "//! search\nuse crate::api::Regressor;\n",
        "model_selection/search/grid/mod.rs": "//! grid search\nuse crate::api::Regressor;\n",
        "model_selection/split/mod.rs": "//! split\nmod grouped;\npub use grouped::GroupKFold;\n",
        "model_selection/split/grouped.rs": "//! grouped\npub struct GroupKFold;\n",
        "calibration/mod.rs": "//! calibration\nmod isotonic;\nmod platt;\npub use isotonic::IsotonicRegression;\n",
        "calibration/isotonic.rs": "//! isotonic\nuse crate::api::Regressor;\n",
        "calibration/platt/mod.rs": "//! platt\nuse crate::api::Regressor;\n",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def append(path: Path, text: str) -> None:
    path.write_text(path.read_text() + text)


SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "crate-root-lib-only",
        lambda root: (root / "src" / "helpers.rs").write_text("//! stray\n"),
        "crate-root Rust sources must be only lib.rs",
    ),
    (
        "obsolete-root-implementations",
        lambda root: (root / "src" / "boosting").mkdir(),
        "obsolete root implementation exists",
    ),
    (
        "artifact-runtime-neutral",
        lambda root: append(
            root / "src" / "artifact" / "mod.rs", "use crate::ensemble::Forest;\n"
        ),
        "artifact foundation depends on a concrete ensemble runtime",
    ),
    (
        "ensemble-families-private",
        lambda root: append(root / "src" / "ensemble" / "mod.rs", "pub mod random_forest;\n"),
        "private ensemble estimator families are exposed as child modules",
    ),
    (
        "numeric-below-estimators",
        lambda root: append(
            root / "src" / "numeric" / "mod.rs", "use crate::linear_model::Ridge;\n"
        ),
        "numeric kernels depend on estimator module linear_model",
    ),
    (
        "quantile-single-source",
        lambda root: append(
            root / "src" / "preprocessing" / "standard_scaler" / "mod.rs",
            "pub(crate) enum QuantileRule { Linear }\n",
        ),
        "quantile definition re-derived outside numeric",
    ),
    (
        "rng-single-source",
        lambda root: append(
            root / "src" / "model_selection" / "split" / "mod.rs",
            "fn mix64(value: u64) -> u64 { value }\n",
        ),
        "generator definition re-derived outside numeric",
    ),
    (
        "preprocessing-below-composition",
        lambda root: append(
            root / "src" / "preprocessing" / "standard_scaler" / "mod.rs",
            "use crate::pipeline::Pipeline;\n",
        ),
        "preprocessing depends on its own consumer pipeline",
    ),
    (
        "inspection-public-surfaces-only",
        lambda root: append(
            root / "src" / "inspection" / "permutation.rs",
            "use crate::ensemble::RandomForestRegressor;\n",
        ),
        "inspection depends on non-public-surface module ensemble",
    ),
    (
        "loss-below-estimators",
        lambda root: append(
            root / "src" / "loss" / "objective.rs",
            "use crate::ensemble::HistGradientBoostingRegressor;\n",
        ),
        "loss contract depends on estimator module ensemble",
    ),
    (
        "optimize-below-estimators",
        lambda root: append(
            root / "src" / "optimize" / "lbfgs.rs",
            "use crate::linear_model::LogisticRegression;\n",
        ),
        "optimize depends on module linear_model",
    ),
    (
        "capability-descriptor-neutral",
        lambda root: append(
            root / "src" / "api" / "capabilities.rs",
            "use crate::preprocessing::StandardScaler;\n",
        ),
        "capability descriptor names estimator module preprocessing",
    ),
    (
        "baselines-independent",
        lambda root: append(
            root / "src" / "dummy" / "classifier.rs",
            "use crate::ensemble::RandomForestClassifier;\n",
        ),
        "baseline estimators depend on estimator module ensemble",
    ),
    (
        "composition-families-private",
        lambda root: append(
            root / "src" / "preprocessing" / "mod.rs", "pub mod standard_scaler;\n"
        ),
        "preprocessing facade exposes child module",
    ),
    (
        "metrics-below-estimators",
        lambda root: append(
            root / "src" / "metrics" / "confusion.rs",
            "use crate::ensemble::RandomForestClassifier;\n",
        ),
        "metrics depend on estimator module ensemble",
    ),
    (
        "evaluation-families-private",
        lambda root: append(
            root / "src" / "model_selection" / "mod.rs", "pub mod split;\n"
        ),
        "model_selection facade exposes child module",
    ),
    (
        "split-families-private",
        lambda root: append(
            root / "src" / "model_selection" / "split" / "mod.rs", "pub mod grouped;\n"
        ),
        "split facade exposes child module",
    ),
    (
        "search-consumes-the-scorer-seam",
        lambda root: append(
            root / "src" / "model_selection" / "search.rs",
            "use crate::metrics::accuracy_score;\n",
        ),
        "search re-derives scoring instead of consuming the scorer contract",
    ),
    (
        "calibration-public-surfaces-only",
        lambda root: append(
            root / "src" / "calibration" / "isotonic.rs",
            "use crate::ensemble::RandomForestClassifier;\n",
        ),
        "calibration depends on non-public-surface module ensemble",
    ),
    (
        "tree-below-estimators",
        lambda root: append(
            root / "src" / "tree" / "grower.rs",
            "use crate::ensemble::RandomForestRegressor;\n",
        ),
        "tree grower depends on estimator module ensemble",
    ),
    (
        "tree-family-private",
        lambda root: append(root / "src" / "tree" / "mod.rs", "pub mod grower;\n"),
        "tree facade exposes child module",
    ),
    (
        "persistence-through-the-trait",
        lambda root: append(
            root / "src" / "preprocessing" / "standard_scaler" / "mod.rs",
            "impl StandardScaler {\n    pub fn to_artifact(&self) -> Vec<u8> { Vec::new() }\n}\n",
        ),
        "persistence declared as an inherent method rather than through the trait",
    ),
    (
        "forest-core-below-facades",
        lambda root: append(
            root / "src" / "ensemble" / "forest" / "training.rs",
            "use crate::ensemble::random_forest::RandomForestRegressor;\n",
        ),
        "forest core depends on ensemble facade random_forest",
    ),
)


# One violation per dependency rule, written into a **child module** of the
# directory the rule reads rather than into its facade.
#
# A violation in the facade is reported by a recursive reader and by a
# facade-only one alike, so it cannot establish which reader the rule uses.
# These can: `self_test` asserts each one fires under [`tree_text`] and stays
# silent under [`shallow_text`]. That second half is the assertion the crate was
# missing on 2026-07-26, when nine dependency rules read facades only and passed
# their own self-test because no module they cover had grown a child yet.
CHILD_MODULE_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "artifact-runtime-neutral",
        lambda root: append(
            root / "src" / "artifact" / "component" / "mod.rs",
            "use crate::ensemble::Forest;\n",
        ),
        "artifact foundation depends on a concrete ensemble runtime",
    ),
    (
        "numeric-below-estimators",
        lambda root: append(
            root / "src" / "numeric" / "stream" / "mod.rs",
            "use crate::linear_model::Ridge;\n",
        ),
        "numeric kernels depend on estimator module linear_model",
    ),
    (
        "quantile-single-source",
        lambda root: append(
            root / "src" / "metrics" / "curves" / "mod.rs",
            "pub(crate) fn quantile_sorted() {}\n",
        ),
        "quantile definition re-derived outside numeric",
    ),
    (
        # The other marker, one level below the facade, in the module the real
        # duplicate lived in: a private generator hides in a child module of a
        # splitter as easily as in the splitter itself.
        "rng-single-source",
        lambda root: append(
            root / "src" / "model_selection" / "split" / "grouped.rs",
            "const INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;\n",
        ),
        "generator definition re-derived outside numeric",
    ),
    (
        "preprocessing-below-composition",
        lambda root: append(
            root / "src" / "preprocessing" / "standard_scaler" / "mod.rs",
            "use crate::model_selection::GridSearch;\n",
        ),
        "preprocessing depends on its own consumer model_selection",
    ),
    (
        "inspection-public-surfaces-only",
        lambda root: append(
            root / "src" / "inspection" / "scoring" / "mod.rs",
            "use crate::artifact::ModelArtifact;\n",
        ),
        "inspection depends on non-public-surface module artifact",
    ),
    (
        "calibration-public-surfaces-only",
        lambda root: append(
            root / "src" / "calibration" / "platt" / "mod.rs",
            "use crate::linear_model::LogisticRegression;\n",
        ),
        "calibration depends on non-public-surface module linear_model",
    ),
    (
        "loss-below-estimators",
        lambda root: append(
            root / "src" / "loss" / "boosting" / "mod.rs",
            "use crate::ensemble::HistGradientBoostingRegressor;\n",
        ),
        "loss contract depends on estimator module ensemble",
    ),
    (
        "optimize-below-estimators",
        lambda root: append(
            root / "src" / "optimize" / "newton" / "mod.rs",
            "use crate::linear_model::LogisticRegression;\n",
        ),
        "optimize depends on module linear_model",
    ),
    (
        "baselines-independent",
        lambda root: append(
            root / "src" / "dummy" / "strategy" / "mod.rs",
            "use crate::ensemble::RandomForestClassifier;\n",
        ),
        "baseline estimators depend on estimator module ensemble",
    ),
    (
        "metrics-below-estimators",
        lambda root: append(
            root / "src" / "metrics" / "curves" / "mod.rs",
            "use crate::ensemble::RandomForestClassifier;\n",
        ),
        "metrics depend on estimator module ensemble",
    ),
    (
        "search-consumes-the-scorer-seam",
        lambda root: append(
            root / "src" / "model_selection" / "search" / "grid" / "mod.rs",
            "use crate::metrics::accuracy_score;\n",
        ),
        "search re-derives scoring instead of consuming the scorer contract",
    ),
    (
        "tree-below-estimators",
        lambda root: append(
            root / "src" / "tree" / "split" / "mod.rs",
            "use crate::ensemble::RandomForestRegressor;\n",
        ),
        "tree grower depends on estimator module ensemble",
    ),
    (
        "forest-core-below-facades",
        lambda root: append(
            root / "src" / "ensemble" / "forest" / "seeding" / "mod.rs",
            "use crate::ensemble::random_forest::RandomForestRegressor;\n",
        ),
        "forest core depends on ensemble facade random_forest",
    ),
)

# The single-source rules read `src/numeric` recursively to decide whether their
# primitive exists at all, so that recursion protects a *non-firing* property:
# the primitive being found one level down. No violation can demonstrate that by
# firing. It is proven instead by the clean tree, which places each primitive in
# a child module and is asserted to report it missing under [`shallow_text`].
# Each rule's `CHILD_MODULE_VIOLATIONS` entry covers its other half.
#
# Keyed by the absence finding the facade-only reader must produce, so the
# exemption cannot be claimed by a rule that has no absence case.
CLEAN_TREE_PROVEN_RECURSION: dict[str, str] = {
    "quantile-single-source": "quantile primitive is missing",
    "rng-single-source": "seeded generator is missing",
}

# Floor on the child-module proofs, so the count cannot shrink by attrition.
#
# `rules_reading_a_module_directory()` derives *who owes* a proof from the live
# source, which is the right closure for a rule being added or downgraded but
# not for one being deleted: removing a rule, its synthetic violation and its
# child-module violation together satisfies every derived assertion while the
# self-test's summary line quietly counts one fewer. A floor makes that removal
# an explicit edit to this number with a reason attached, which is the same
# treatment the reach floors in `tests/artifact_hardening.rs` get. Raise it when
# proofs are added; lower it only alongside the rule being retired.
MINIMUM_CHILD_MODULE_PROOFS = 14


def rules_reading_a_module_directory() -> set[str]:
    """Rule names whose implementation reads a module directory.

    Derived from the source rather than listed, so a new dependency rule cannot
    be added without also owing the child-module proof below.

    [`shallow_text`] counts as well as [`tree_text`], which is what makes the
    closure run in both directions. Deriving the owed set from the recursive
    reader alone let a rule *stop* owing its proof by being downgraded to the
    facade-only reader in the same edit that deleted the proof — the two halves
    cancelling, the self-test passing, the count silently dropping by one. A
    rule that reads a directory owes the proof however it reads it, so the
    downgrade now fails against the child-module violation it still owes.
    """
    return {
        name
        for name, rule in RULES
        if "tree_text(" in inspect.getsource(rule)
        or "shallow_text(" in inspect.getsource(rule)
    }


@contextmanager
def facade_only_reader() -> Iterator[None]:
    """Run the rules against [`shallow_text`], the reader they must not use."""
    global tree_text  # noqa: PLW0603 - deliberate, and restored on exit
    original = tree_text
    tree_text = shallow_text
    try:
        yield
    finally:
        tree_text = original


def self_test() -> None:
    live = violations()
    assert live == [], f"live tree violates its own layout rules: {live}"

    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    declared = {name for name, _ in RULES}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )

    recursive = rules_reading_a_module_directory()
    child_covered = {name for name, _, _ in CHILD_MODULE_VIOLATIONS}
    assert child_covered <= declared, (
        f"stale child-module violations: {sorted(child_covered - declared)}"
    )
    assert recursive <= child_covered, (
        "every rule reading a module directory needs a child-module violation: "
        f"missing={sorted(recursive - child_covered)}"
    )
    assert set(CLEAN_TREE_PROVEN_RECURSION) <= recursive, (
        "stale clean-tree recursion exemption: "
        f"{sorted(set(CLEAN_TREE_PROVEN_RECURSION) - recursive)}"
    )
    assert len(CHILD_MODULE_VIOLATIONS) >= MINIMUM_CHILD_MODULE_PROOFS, (
        f"child-module proofs fell to {len(CHILD_MODULE_VIOLATIONS)}, below the "
        f"floor of {MINIMUM_CHILD_MODULE_PROOFS}; lower the floor deliberately "
        "or restore the proof"
    )

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)
        clean = write_clean_tree(base / "clean")
        found = violations(clean)
        assert found == [], f"synthetic clean tree reported violations: {found}"

        # The clean tree keeps each single-source primitive in a child module,
        # so a facade-only reader cannot find it. These are the assertions that
        # prove those rules read the tree.
        with facade_only_reader():
            shallow_found = violations(clean)
        for name, absence in CLEAN_TREE_PROVEN_RECURSION.items():
            assert any(absence in item for item in shallow_found), (
                "the clean tree no longer distinguishes a recursive reader from "
                f"a facade-only one for {name}; reported {shallow_found}"
            )

        for index, (name, mutate, expected) in enumerate(SYNTHETIC_VIOLATIONS):
            tree = write_clean_tree(base / f"facade-{index}-{name}")
            mutate(tree)
            found = violations(tree)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against its synthetic violation; "
                f"reported {found}"
            )

        for index, (name, mutate, expected) in enumerate(CHILD_MODULE_VIOLATIONS):
            tree = write_clean_tree(base / f"child-{index}-{name}")
            mutate(tree)
            found = violations(tree)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against a violation in a child "
                f"module; reported {found}"
            )
            with facade_only_reader():
                shallow_found = violations(tree)
            assert not any(expected in item for item in shallow_found), (
                f"rule {name}'s child-module violation is also reported by a "
                "facade-only reader, so it does not prove the rule reads the "
                f"tree; reported {shallow_found}"
            )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "source layout verifier self-test passed "
            f"({len(RULES)} rules, each proven against a synthetic violation; "
            f"{len(CHILD_MODULE_VIOLATIONS)} of them proven again against a "
            "violation in a child module that a facade-only reader misses)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_source_layout.py [--self-test]", file=sys.stderr)
        return 2
    found = violations()
    if found:
        print("source layout check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("source layout: domain ownership boundaries pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
