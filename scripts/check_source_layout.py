#!/usr/bin/env python3
"""Enforce FerricML's domain-owned private source layout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def violations(root: Path = ROOT) -> list[str]:
    source = root / "src"
    found: list[str] = []
    root_sources = sorted(path.name for path in source.glob("*.rs"))
    if root_sources != ["lib.rs"]:
        found.append(f"crate-root Rust sources must be only lib.rs, found {root_sources}")

    for removed in (source / "forest.rs", source / "boosting"):
        if removed.exists():
            found.append(f"obsolete root implementation exists: {removed.relative_to(root)}")

    artifact_text = "\n".join(
        path.read_text() for path in sorted((source / "artifact").glob("*.rs"))
    )
    if "crate::ensemble" in artifact_text:
        found.append("artifact foundation depends on a concrete ensemble runtime")

    ensemble_facade = (source / "ensemble" / "mod.rs").read_text()
    if "pub mod random_forest" in ensemble_facade or "pub mod hist_gradient_boosting" in ensemble_facade:
        found.append("private ensemble estimator families are exposed as child modules")
    return found


def self_test() -> None:
    assert violations() == []


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("source layout verifier self-test passed")
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
