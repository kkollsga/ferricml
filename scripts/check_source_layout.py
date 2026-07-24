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


RULES: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("crate-root-lib-only", crate_root_is_lib_only),
    ("obsolete-root-implementations", obsolete_root_implementations_are_gone),
    ("artifact-runtime-neutral", artifact_is_runtime_neutral),
    ("ensemble-families-private", ensemble_families_stay_private),
    ("numeric-below-estimators", numeric_depends_on_no_estimator),
    ("inspection-public-surfaces-only", inspection_uses_only_public_surfaces),
    ("capability-descriptor-neutral", capability_descriptor_names_no_estimator),
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
        "ensemble/random_forest/mod.rs": "//! forest\n",
        "numeric/mod.rs": "//! numeric\npub(crate) fn kernel() {}\n",
        "numeric/rng.rs": "//! rng\n",
        "inspection/mod.rs": "//! inspection\nmod permutation;\n",
        "inspection/permutation.rs": "//! permutation\nuse crate::api::Regressor;\n",
        "api/mod.rs": "//! api\nmod capabilities;\n",
        "api/capabilities.rs": "//! capabilities\npub struct Capabilities;\n",
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
        "capability-descriptor-neutral",
        lambda root: append(
            root / "src" / "api" / "capabilities.rs",
            "use crate::preprocessing::StandardScaler;\n",
        ),
        "capability descriptor names estimator module preprocessing",
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
