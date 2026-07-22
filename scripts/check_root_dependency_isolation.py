#!/usr/bin/env python3
"""Reject competitor crates from root dependencies and automated gates."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"linfa", "linfa-ensemble", "linfa-trees", "rafor", "smartcore"}


def check_automation() -> list[str]:
    leaked: list[str] = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        contents = path.read_text().lower()
        matches = sorted(name for name in FORBIDDEN if name in contents)
        if matches:
            leaked.append(f"{path.relative_to(ROOT)}: {', '.join(matches)}")

    makefile = (ROOT / "Makefile").read_text()
    active_target = ""
    protected = {"gate", "gate-full", "package-check", "bench-self", "bench-history"}
    for line in makefile.splitlines():
        if line and not line[0].isspace() and ":" in line:
            active_target = line.split(":", 1)[0]
        if active_target in protected and (
            "bench-rafor" in line or "benchmarks/alternatives" in line or "run_forest_performance" in line
        ):
            leaked.append(f"Makefile target {active_target}: competitor invocation")
    return leaked


def main() -> int:
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    names = {package["name"] for package in metadata["packages"]}
    leaked = sorted(names & FORBIDDEN)
    if leaked:
        print(f"root dependency graph contains competitor crates: {', '.join(leaked)}", file=sys.stderr)
        return 1
    automation_leaks = check_automation()
    if automation_leaks:
        print("competitor tooling leaked into CI/release gates:", file=sys.stderr)
        for leak in automation_leaks:
            print(f"- {leak}", file=sys.stderr)
        return 1
    print("root dependency isolation: no competitor crates in dependencies, CI, or release gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
