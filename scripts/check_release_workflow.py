#!/usr/bin/env python3
"""Audit the security and validation boundaries of the release workflow."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def fail(message: str) -> None:
    print(f"release workflow audit failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    text = WORKFLOW.read_text()
    starts = list(re.finditer(r"(?m)^  ([a-z0-9-]+):\n", text[text.index("jobs:\n") + 6 :]))
    jobs_text = text[text.index("jobs:\n") + 6 :]
    jobs = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(jobs_text)
        jobs[match.group(1)] = jobs_text[match.start() : end]

    expected = {
        "validate-tag",
        "rust-and-package",
        "public-api",
        "sklearn-conformance",
        "publish-crate",
        "github-release",
    }
    if set(jobs) != expected:
        fail(f"expected jobs {sorted(expected)}, found {sorted(jobs)}")

    prefix = text[: text.index("jobs:\n")]
    if "permissions:\n  contents: read" not in prefix:
        fail("workflow-wide permissions must be contents: read")
    if text.count("contents: write") != 1 or "contents: write" not in jobs["github-release"]:
        fail("only the GitHub-release job may have contents: write")
    secret = "secrets.CARGO_REGISTRY_TOKEN"
    if text.count(secret) != 1 or secret not in jobs["publish-crate"]:
        fail("the crates.io token must appear only in the publish job")
    validation = jobs["validate-tag"]
    for required in (
        '${GITHUB_REF_NAME}" = "v${version}',
        'git rev-list -n 1 "refs/tags/${GITHUB_REF_NAME}"',
        '"$(git rev-parse origin/main)"',
        'grep --fixed-strings "## [${version}]" CHANGELOG.md',
    ):
        if required not in validation:
            fail(f"tag validation is missing: {required}")
    if "make gate-full" not in jobs["rust-and-package"]:
        fail("release validation must run gate-full")
    if "make package-check" not in jobs["rust-and-package"]:
        fail("release validation must run package-check explicitly")
    if "make api-check" not in jobs["public-api"]:
        fail("release validation must run the exact API check")
    if "scripts/sklearn_conformance.py" not in jobs["sklearn-conformance"]:
        fail("release validation must run pinned scikit conformance")
    if "cargo publish --locked" not in jobs["publish-crate"]:
        fail("publish job is missing cargo publish --locked")

    print("release workflow audit: validation and least-privilege boundaries pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
