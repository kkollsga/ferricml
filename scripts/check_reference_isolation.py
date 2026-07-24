#!/usr/bin/env python3
"""Keep external reference identities out of FerricML's tracked tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (b"sci" + b"kit", b"sk" + b"learn")


def violations(entries: list[tuple[str, bytes]]) -> list[str]:
    found = []
    for name, content in entries:
        lowered_name = name.encode().lower()
        lowered_content = content.lower()
        if any(term in lowered_name for term in FORBIDDEN):
            found.append(f"forbidden external identity in tracked path: {name}")
        if any(term in lowered_content for term in FORBIDDEN):
            found.append(f"forbidden external identity in tracked content: {name}")
    return found


def tracked_entries() -> list[tuple[str, bytes]]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    names = [name.decode() for name in result.stdout.split(b"\0") if name]
    return [(name, (ROOT / name).read_bytes()) for name in names]


def self_test() -> None:
    assert violations([("src/lib.rs", b"FerricML owns this contract")]) == []

    first, second = (term.decode() for term in FORBIDDEN)
    found = violations(
        [
            (f"docs/{first}.md", b"generic content"),
            ("README.md", f"mentions {second}".encode()),
        ]
    )
    assert len(found) == 2
    assert "tracked path" in found[0]
    assert "tracked content" in found[1]


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("reference isolation verifier self-test passed")
        return 0
    if sys.argv[1:]:
        print("usage: check_reference_isolation.py [--self-test]", file=sys.stderr)
        return 2

    found = violations(tracked_entries())
    if found:
        print("reference isolation check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("reference isolation: tracked paths and content are generic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
