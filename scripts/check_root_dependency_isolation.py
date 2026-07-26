#!/usr/bin/env python3
"""Keep comparison workspaces out of the crate dependency and gate surface."""

from __future__ import annotations

import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The targets a contributor or CI runs as a matter of course. A comparison
# workspace reached from one of these is a comparison dependency inside the
# crate's routine surface, which `default = []` exists to prevent.
PROTECTED: tuple[str, ...] = ("gate", "gate-full", "package-check", "bench-self", "bench-history")

COMPARISON = "benchmarks/"

# A target header: a line at column zero naming one or more targets, then a
# colon, then its prerequisites. The negative lookahead keeps `SHELL := ...`
# out, the leading character class keeps comments and variable assignments out,
# and the prerequisites stay on the line because a comparison workspace named as
# a prerequisite of a protected target is exactly as much of a leak as one named
# in its recipe.
HEADER = re.compile(r"^(?P<names>[^\s:#=][^:=]*):(?!=)(?P<rest>.*)$")


def target_lines(makefile: str) -> dict[str, list[str]]:
    """Map each declared target to the lines make attributes to it.

    Attribution is the whole point of this reader. A rule that searched the
    document instead would fire on the sanctioned comparison targets that live
    outside the protected set, which is why `self_test` runs the clean Makefile
    through [`every_line`] as well.
    """
    lines: dict[str, list[str]] = {}
    active: list[str] = []
    for line in makefile.splitlines():
        if line.startswith("\t"):
            for name in active:
                lines[name].append(line)
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = HEADER.match(line)
        if match is None:
            active = []
            continue
        active = match.group("names").split()
        for name in active:
            lines.setdefault(name, []).append(line)
    return lines


def every_line(makefile: str) -> dict[str, list[str]]:
    """Attribute every line of the file to every target.

    No rule calls this. `self_test` substitutes it for [`target_lines`] and
    asserts the clean Makefile then reports a leak, which is what proves the
    real rule reads one target's lines rather than the document.
    """
    all_lines = makefile.splitlines()
    names = {
        name
        for line in all_lines
        if (match := HEADER.match(line))
        for name in match.group("names").split()
    }
    return {name: list(all_lines) for name in names}


def read_makefile(root: Path) -> str:
    return (root / "Makefile").read_text().lower()


def protected_targets_are_declared(root: Path) -> list[str]:
    """Absence of the input is the finding.

    Every rule below scans the lines of a protected target. Rename or drop one
    and there is nothing left to scan, and this file would report success while
    checking a target that no longer exists.
    """
    declared = set(target_lines(read_makefile(root)))
    return [
        f"Makefile target {name}: protected target is not declared, so nothing is scanned for it"
        for name in PROTECTED
        if name not in declared
    ]


def protected_targets_avoid_comparison_workspaces(root: Path) -> list[str]:
    lines = target_lines(read_makefile(root))
    return [
        f"Makefile target {name}: comparison workspace invocation"
        for name in PROTECTED
        for line in lines.get(name, ())
        if COMPARISON in line
    ]


def comparison_workspace_is_outside_the_crate(root: Path) -> list[str]:
    if (root / "benchmarks" / "alternatives" / "Cargo.toml").exists():
        return ["benchmarks/alternatives: comparison workspace exists inside the crate tree"]
    return []


RULES: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("protected-targets-declared", protected_targets_are_declared),
    ("protected-targets-avoid-comparison", protected_targets_avoid_comparison_workspaces),
    ("comparison-workspace-outside-the-crate", comparison_workspace_is_outside_the_crate),
)


def violations(root: Path = ROOT) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(root))
    return found


CLEAN_MAKEFILE = """\
SHELL := /bin/bash
PYTHON ?= python3

.PHONY: gate gate-full package-check bench-self bench-history bench-compare

## Ordinary pre-push gate: lint, tests, and the isolation checks.
gate:
\tcargo test --locked
\t$(MAKE) package-check

gate-full: gate
\tcargo test --locked --all-features

package-check:
\tbash scripts/check_packaged_crate.sh

bench-self:
\tcargo bench --locked --bench forest

bench-history:
\t$(PYTHON) scripts/performance_history.py capture

## Sanctioned: comparison measurement lives in its own target, outside the
## routine surface, and names the comparison workspace openly.
bench-compare:
\tcargo bench --manifest-path benchmarks/alternatives/Cargo.toml
"""


def write_clean_root(root: Path) -> Path:
    """Write the smallest crate root that satisfies every rule.

    Its Makefile declares every protected target *and* an unprotected target
    that invokes a comparison workspace. The second one is the discriminating
    part: a reader that searched the document rather than one target's lines
    would report it, so this Makefile passing at all is what proves the
    attribution is real. It also carries the two shapes that have to stay out
    of target detection — a `:=` assignment and a comment containing a colon.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "Makefile").write_text(CLEAN_MAKEFILE)
    return root


def append_to_gate(root: Path, line: str) -> None:
    makefile = (root / "Makefile").read_text()
    marker = "gate:\n"
    index = makefile.index(marker) + len(marker)
    (root / "Makefile").write_text(makefile[:index] + line + makefile[index:])


def rename_gate(root: Path) -> None:
    makefile = (root / "Makefile").read_text()
    (root / "Makefile").write_text(makefile.replace("\ngate:\n", "\ncheck:\n"))


def add_comparison_workspace(root: Path) -> None:
    manifest = root / "benchmarks" / "alternatives" / "Cargo.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('[package]\nname = "alternatives"\n')


SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "protected-targets-declared",
        rename_gate,
        "Makefile target gate: protected target is not declared",
    ),
    (
        "protected-targets-avoid-comparison",
        lambda root: append_to_gate(
            root, "\tcargo bench --manifest-path benchmarks/alternatives/Cargo.toml\n"
        ),
        "Makefile target gate: comparison workspace invocation",
    ),
    (
        "comparison-workspace-outside-the-crate",
        add_comparison_workspace,
        "benchmarks/alternatives: comparison workspace exists inside the crate tree",
    ),
)

# The comparison rule again, reached through a protected target's
# *prerequisites* rather than its recipe. `gate: benchmarks/...` builds the
# comparison workspace before the gate body ever runs, so the header line is
# part of the target's surface.
PREREQUISITE_VIOLATION = (
    "protected-targets-avoid-comparison",
    lambda root: (root / "Makefile").write_text(
        (root / "Makefile").read_text().replace(
            "\ngate:\n", "\ngate: benchmarks/alternatives/target/release/compare\n"
        )
    ),
    "Makefile target gate: comparison workspace invocation",
)

# `protected-targets-declared` and `comparison-workspace-outside-the-crate` do
# not read a target's lines at all, so no violation of theirs can distinguish
# the two readers. The attribution they do not exercise is proven by the clean
# Makefile instead, which carries a sanctioned comparison target that only a
# document-wide reader reports.
CLEAN_MAKEFILE_PROVEN_ATTRIBUTION: tuple[str, ...] = ("protected-targets-avoid-comparison",)


@contextmanager
def target_blind_reader() -> Iterator[None]:
    """Run the rules against [`every_line`], the reader they must not use."""
    global target_lines  # noqa: PLW0603 - deliberate, and restored on exit
    original = target_lines
    target_lines = every_line
    try:
        yield
    finally:
        target_lines = original


def self_test() -> None:
    live = violations()
    assert live == [], f"live tree leaks comparison tooling into the crate: {live}"

    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    declared = {name for name, _ in RULES}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )
    assert set(CLEAN_MAKEFILE_PROVEN_ATTRIBUTION) <= declared, (
        "stale clean-Makefile attribution proof: "
        f"{sorted(set(CLEAN_MAKEFILE_PROVEN_ATTRIBUTION) - declared)}"
    )

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)

        clean = write_clean_root(base / "clean")
        found = violations(clean)
        assert found == [], f"synthetic clean root reported violations: {found}"

        # The clean Makefile's `bench-compare` invokes a comparison workspace
        # from an unprotected target. A reader that searched the document would
        # report it against every protected target; the real reader must not.
        with target_blind_reader():
            blind = violations(clean)
        assert any("comparison workspace invocation" in item for item in blind), (
            "the clean Makefile no longer distinguishes a target-scoped reader "
            f"from a document-wide one; reported {blind}"
        )

        cases = [*SYNTHETIC_VIOLATIONS, PREREQUISITE_VIOLATION]
        for index, (name, mutate, expected) in enumerate(cases):
            root = write_clean_root(base / f"violation-{index}-{name}")
            mutate(root)
            found = violations(root)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against its synthetic violation; reported {found}"
            )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "root dependency isolation verifier self-test passed "
            f"({len(RULES)} rules, each proven against a synthetic violation; "
            "target attribution proven against a sanctioned comparison target "
            "only a document-wide reader reports)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_root_dependency_isolation.py [--self-test]", file=sys.stderr)
        return 2

    found = violations()
    if found:
        print("comparison tooling leaked into the crate or gate surface:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("root dependency isolation: comparison workspaces are outside the crate and gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
