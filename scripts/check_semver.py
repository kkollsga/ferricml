#!/usr/bin/env python3
"""Report cargo-semver-checks evidence without making it a release gate."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CRATE_API = "https://crates.io/api/v1/crates/ferricml"


def published_baseline() -> str | None:
    request = urllib.request.Request(CRATE_API, headers={"User-Agent": "ferricml-semver-check"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    crate = payload.get("crate", {})
    return crate.get("max_stable_version") or crate.get("max_version")


def main() -> int:
    try:
        baseline = published_baseline()
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"semver: unable to query crates.io: {error}", file=sys.stderr)
        return 1

    if baseline is None:
        print("semver: no published baseline for ferricml; first release has nothing to compare")
        return 0

    if shutil.which("cargo-semver-checks") is None:
        print(
            f"semver: latest published baseline is {baseline}, but cargo-semver-checks is not installed",
            file=sys.stderr,
        )
        return 1

    command = [
        "cargo",
        "semver-checks",
        "check-release",
        "--baseline-version",
        baseline,
    ]
    result = subprocess.run(command, cwd=ROOT)
    verdict = "compatible" if result.returncode == 0 else "potential incompatibilities reported"
    print(f"semver: informational comparison against ferricml {baseline}: {verdict}")
    # The report informs the release decision; it is deliberately not a gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
