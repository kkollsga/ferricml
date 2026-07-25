#!/usr/bin/env python3
"""Enforce FerricML's domain-owned private source layout.

Each boundary is a named rule over a source tree, so `--self-test` can assert
that every rule still fires against a synthetic tree that violates it. A rule
that silently stopped matching would otherwise pass both the check and its own
self-test.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable


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


def read_if_present(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def directory_text(directory: Path) -> str:
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
    if "crate::ensemble" in directory_text(root / "src" / "artifact"):
        return ["artifact foundation depends on a concrete ensemble runtime"]
    return []


def ensemble_families_stay_private(root: Path) -> list[str]:
    facade = read_if_present(root / "src" / "ensemble" / "mod.rs")
    if "pub mod random_forest" in facade or "pub mod hist_gradient_boosting" in facade:
        return ["private ensemble estimator families are exposed as child modules"]
    return []


def numeric_depends_on_no_estimator(root: Path) -> list[str]:
    text = directory_text(root / "src" / "numeric")
    return [
        f"numeric kernels depend on estimator module {module}"
        for module in ESTIMATOR_MODULES
        if f"crate::{module}" in text
    ]


def inspection_uses_only_public_surfaces(root: Path) -> list[str]:
    """Inspection must name no estimator family and no persistence internals.

    Model-agnostic inspection is defined by working through the public
    prediction and scoring contracts alone. Naming a concrete estimator or the
    artifact layer is the observable symptom of it reaching past them.
    """
    text = directory_text(root / "src" / "inspection")
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
    text = directory_text(root / "src" / "calibration")
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
    text = directory_text(root / "src" / "loss")
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
    text = directory_text(root / "src" / "optimize")
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
    text = directory_text(root / "src" / "dummy")
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
    text = directory_text(root / "src" / "metrics")
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
    text = read_if_present(module) + directory_text(
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
    text = directory_text(root / "src" / "tree")
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
    text = directory_text(root / "src" / "ensemble" / "forest")
    if not text:
        return ["ensemble forest core is missing"]
    return [
        f"forest core depends on ensemble facade {facade}"
        for facade in ("random_forest", "extra_trees", "hist_gradient_boosting")
        if f"ensemble::{facade}" in text or f"super::{facade}" in text
    ]


RULES: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("crate-root-lib-only", crate_root_is_lib_only),
    ("obsolete-root-implementations", obsolete_root_implementations_are_gone),
    ("artifact-runtime-neutral", artifact_is_runtime_neutral),
    ("ensemble-families-private", ensemble_families_stay_private),
    ("numeric-below-estimators", numeric_depends_on_no_estimator),
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
)


def violations(root: Path = ROOT) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(root))
    return found


def write_clean_tree(root: Path) -> Path:
    """Write the smallest source tree that satisfies every rule."""
    source = root / "src"
    for relative, text in {
        "lib.rs": "pub mod artifact;\npub mod ensemble;\nmod numeric;\n",
        "artifact/mod.rs": "//! artifact\npub(crate) use self::inner::Thing;\n",
        "ensemble/mod.rs": "//! ensemble\nmod random_forest;\npub use random_forest::Forest;\n",
        "ensemble/random_forest/mod.rs": "//! forest\nuse crate::tree::grow_tree;\n",
        "ensemble/forest/mod.rs": "//! forest core\nmod training;\n",
        "ensemble/forest/training.rs": "//! training\nuse crate::tree::grow_tree;\n",
        "tree/mod.rs": "//! tree\nmod grower;\npub use grower::DecisionTreeRegressor;\n",
        "tree/grower.rs": "//! grower\nuse crate::numeric::kernel;\npub struct DecisionTreeRegressor;\n",
        "numeric/mod.rs": "//! numeric\npub(crate) fn kernel() {}\n",
        "numeric/rng.rs": "//! rng\n",
        "loss/mod.rs": "//! loss\nmod objective;\n",
        "loss/objective.rs": "//! objective\nuse crate::numeric::kernel;\n",
        "optimize/mod.rs": "//! optimize\nmod lbfgs;\n",
        "optimize/lbfgs.rs": "//! lbfgs\nuse crate::numeric::kernel;\nuse crate::loss::Objective;\n",
        "inspection/mod.rs": "//! inspection\nmod permutation;\n",
        "inspection/permutation.rs": "//! permutation\nuse crate::api::Regressor;\n",
        "api/mod.rs": "//! api\nmod capabilities;\n",
        "api/capabilities.rs": "//! capabilities\npub struct Capabilities;\n",
        "dummy/mod.rs": "//! dummy\nmod classifier;\n",
        "dummy/classifier.rs": "//! baseline\nuse crate::api::Classifier;\n",
        "preprocessing/mod.rs": "//! preprocessing\nmod standard_scaler;\n",
        "preprocessing/standard_scaler/mod.rs": "//! scaler\n",
        "pipeline/mod.rs": "//! pipeline\nmod staged;\npub use staged::StagedPipeline;\n",
        "pipeline/staged.rs": "//! staged\npub struct StagedPipeline;\n",
        "metrics/mod.rs": "//! metrics\nmod confusion;\npub use confusion::ConfusionMatrix;\n",
        "metrics/confusion.rs": "//! confusion\npub struct ConfusionMatrix;\n",
        "model_selection/mod.rs": "//! model selection\nmod split;\npub use split::Split;\n",
        "model_selection/search.rs": "//! search\nuse crate::api::Regressor;\n",
        "model_selection/split/mod.rs": "//! split\nmod grouped;\npub use grouped::GroupKFold;\n",
        "model_selection/split/grouped.rs": "//! grouped\npub struct GroupKFold;\n",
        "calibration/mod.rs": "//! calibration\nmod isotonic;\npub use isotonic::IsotonicRegression;\n",
        "calibration/isotonic.rs": "//! isotonic\nuse crate::api::Regressor;\n",
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
            root / "src" / "numeric" / "rng.rs", "use crate::linear_model::Ridge;\n"
        ),
        "numeric kernels depend on estimator module linear_model",
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
        "forest-core-below-facades",
        lambda root: append(
            root / "src" / "ensemble" / "forest" / "training.rs",
            "use crate::ensemble::random_forest::RandomForestRegressor;\n",
        ),
        "forest core depends on ensemble facade random_forest",
    ),
)


def self_test() -> None:
    live = violations()
    assert live == [], f"live tree violates its own layout rules: {live}"

    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    declared = {name for name, _ in RULES}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)
        clean = write_clean_tree(base / "clean")
        found = violations(clean)
        assert found == [], f"synthetic clean tree reported violations: {found}"

        for name, mutate, expected in SYNTHETIC_VIOLATIONS:
            tree = write_clean_tree(base / name)
            mutate(tree)
            found = violations(tree)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against its synthetic violation; "
                f"reported {found}"
            )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "source layout verifier self-test passed "
            f"({len(RULES)} rules, each proven against a synthetic violation)"
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
