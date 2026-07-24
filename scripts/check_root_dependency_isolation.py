#!/usr/bin/env python3
"""Keep comparison workspaces out of the crate dependency and gate surface."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
def check_isolation() -> list[str]:
    leaked: list[str] = []
    makefile = (ROOT / "Makefile").read_text().lower()
    active_target = ""
    protected = {"gate", "gate-full", "package-check", "bench-self", "bench-history"}
    for line in makefile.splitlines():
        if line and not line[0].isspace() and ":" in line:
            active_target = line.split(":", 1)[0]
        if active_target in protected and "benchmarks/" in line:
            leaked.append(f"Makefile target {active_target}: comparison workspace invocation")

    alternatives = ROOT / "benchmarks" / "alternatives"
    if (alternatives / "Cargo.toml").exists():
        leaked.append("benchmarks/alternatives: comparison workspace is tracked in the crate tree")
    return leaked


def main() -> int:
    isolation_leaks = check_isolation()
    if isolation_leaks:
        print("comparison tooling leaked into the crate or gate surface:", file=sys.stderr)
        for leak in isolation_leaks:
            print(f"- {leak}", file=sys.stderr)
        return 1
    print("root dependency isolation: comparison workspaces are outside the crate and gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
