#!/usr/bin/env python3
"""Audit the security and validation boundaries of the release workflow."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"

EXPECTED_JOBS: tuple[str, ...] = (
    "validate-tag",
    "rust-and-package",
    "public-api",
    "reference-contract",
    "publish-crate",
    "github-release",
)

JOBS_MARKER = "jobs:\n"
JOB_HEADER = re.compile(r"(?m)^  ([a-z0-9-]+):\n")

# Every string the release workflow must contain, and the one job that may
# contain it. Ownership is half the contract: a token in the right job proves
# the step runs, and its absence from every other job is what keeps the
# least-privilege boundaries meaningful. A crates.io token reachable from a job
# that has not passed the gates is a publish path around them.
REQUIRED_IN_JOB: tuple[tuple[str, str, str], ...] = (
    (
        '${GITHUB_REF_NAME}" = "v${version}',
        "validate-tag",
        "the tag must be checked against the crate version",
    ),
    (
        'git rev-list -n 1 "refs/tags/${GITHUB_REF_NAME}"',
        "validate-tag",
        "the tag's commit must be resolved",
    ),
    (
        '"$(git rev-parse origin/main)"',
        "validate-tag",
        "the tag must be checked against the exact main tip",
    ),
    (
        'grep --fixed-strings "## [${version}]" CHANGELOG.md',
        "validate-tag",
        "the changelog must carry the released version",
    ),
    (
        "make gate-full",
        "rust-and-package",
        "release validation must run gate-full",
    ),
    (
        "make package-check",
        "rust-and-package",
        "release validation must run package-check explicitly",
    ),
    (
        "make api-check",
        "public-api",
        "release validation must run the exact API check",
    ),
    (
        "make reference-check",
        "reference-contract",
        "release validation must run the frozen reference contract",
    ),
    (
        "cargo publish --locked",
        "publish-crate",
        "publish must be the locked publish",
    ),
    (
        "secrets.CARGO_REGISTRY_TOKEN",
        "publish-crate",
        "the crates.io token must reach only the publish job",
    ),
    (
        "contents: write",
        "github-release",
        "only the GitHub-release job may hold write permission",
    ),
)


def split_jobs(text: str) -> dict[str, str]:
    """Map each declared job to its own body.

    Returning an empty mapping for an unparsable file is deliberate: it makes
    `job-set-is-exact` report the absence rather than raising, so a workflow
    this reader cannot understand is a finding instead of a traceback.
    """
    if JOBS_MARKER not in text:
        return {}
    block = text[text.index(JOBS_MARKER) + len(JOBS_MARKER) :]
    starts = list(JOB_HEADER.finditer(block))
    return {
        match.group(1): block[
            match.start() : (starts[index + 1].start() if index + 1 < len(starts) else len(block))
        ]
        for index, match in enumerate(starts)
    }


def whole_document_jobs(text: str) -> dict[str, str]:
    """Give every job the whole file as its body.

    No rule calls this. `self_test` substitutes it for [`split_jobs`] and
    asserts the clean workflow then reports every required token as misplaced,
    which is what proves the ownership rule reads one job's body rather than
    the document.
    """
    if JOBS_MARKER not in text:
        return {}
    block = text[text.index(JOBS_MARKER) + len(JOBS_MARKER) :]
    return {match.group(1): text for match in JOB_HEADER.finditer(block)}


def job_set_is_exact(text: str) -> list[str]:
    jobs = split_jobs(text)
    if set(jobs) != set(EXPECTED_JOBS):
        return [f"expected jobs {sorted(EXPECTED_JOBS)}, found {sorted(jobs)}"]
    return []


def workflow_permissions_are_read_only(text: str) -> list[str]:
    prefix = text[: text.index(JOBS_MARKER)] if JOBS_MARKER in text else text
    if "permissions:\n  contents: read" not in prefix:
        return ["workflow-wide permissions must be contents: read"]
    return []


HEADER_SCOPE = "the workflow header"


def required_tokens_live_in_their_job(text: str) -> list[str]:
    jobs = split_jobs(text)
    # The text before `jobs:` is a scope of its own: a write permission or a
    # secret declared there reaches every job at once, which is the widest
    # possible misplacement rather than the absence of one.
    prefix = text[: text.index(JOBS_MARKER)] if JOBS_MARKER in text else ""
    scopes = {HEADER_SCOPE: prefix, **{f"job {name}": body for name, body in jobs.items()}}
    found: list[str] = []
    for token, owner, reason in REQUIRED_IN_JOB:
        if token not in jobs.get(owner, ""):
            found.append(f"{reason}: {token!r} is absent from job {owner}")
        for name, body in sorted(scopes.items()):
            if name != f"job {owner}" and token in body:
                found.append(f"{reason}: {token!r} also appears in {name}")
    return found


RULES: tuple[tuple[str, Callable[[str], list[str]]], ...] = (
    ("job-set-is-exact", job_set_is_exact),
    ("workflow-permissions-read-only", workflow_permissions_are_read_only),
    ("required-tokens-in-their-job", required_tokens_live_in_their_job),
)


def violations(text: str) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(text))
    return found


CLEAN_WORKFLOW = """\
name: Release

on:
  push:
    tags:
      - "v*.*.*"

permissions:
  contents: read

jobs:
  validate-tag:
    name: Validate release tag and main tip
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - name: Validate tag, exact main tip, and changelog
        run: |
          version="$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')"
          test "${GITHUB_REF_NAME}" = "v${version}"
          tag_commit="$(git rev-list -n 1 "refs/tags/${GITHUB_REF_NAME}")"
          test "${tag_commit}" = "$(git rev-parse origin/main)"
          grep --fixed-strings "## [${version}]" CHANGELOG.md

  rust-and-package:
    name: Full Rust and packaged-crate gates
    needs: validate-tag
    runs-on: ubuntu-latest
    steps:
      - name: Full Rust gate
        run: make gate-full
      - name: Explicit packaged-crate gate
        run: make package-check

  public-api:
    name: Exact Rust public API
    needs: validate-tag
    runs-on: ubuntu-latest
    steps:
      - name: Diff exact public API surface
        run: make api-check

  reference-contract:
    name: Frozen reference semantics
    needs: validate-tag
    runs-on: ubuntu-latest
    steps:
      - name: Verify frozen behavior and quality contract
        run: make reference-check

  publish-crate:
    name: Publish crate
    needs: [rust-and-package, public-api, reference-contract]
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Publish to crates.io
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
        run: cargo publish --locked

  github-release:
    name: Create GitHub release
    needs: publish-crate
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Create GitHub release
        run: gh release create "${GITHUB_REF_NAME}" --verify-tag --generate-notes
"""


def remove(token: str) -> Callable[[str], str]:
    return lambda text: text.replace(token, "# removed by the self-test", 1)


def copy_into_another_job(token: str, owner: str) -> Callable[[str], str]:
    """Repeat a required token as a comment inside a job that must not hold it."""
    other = next(name for name in EXPECTED_JOBS if name != owner)
    return lambda text: text.replace(f"  {other}:\n", f"  {other}:\n    # {token}\n", 1)


def copy_into_the_header(token: str) -> Callable[[str], str]:
    """Repeat a required token in the text before `jobs:`, where it reaches all of them."""
    return lambda text: text.replace("name: Release\n", f"name: Release\n# {token}\n", 1)


def move_permissions_into_a_job(text: str) -> str:
    """Demote the workflow-wide default to a single job's declaration."""
    return text.replace("permissions:\n  contents: read\n\n", "").replace(
        "  validate-tag:\n", "  validate-tag:\n    permissions:\n      contents: read\n", 1
    )


# One synthetic violation per rule, plus — derived from `REQUIRED_IN_JOB` rather
# than listed, so a new required token cannot be added without owing all three
# proofs — a removal and two misplacements for every token the ownership rule
# covers: one into a job that must not hold it, one into the workflow header
# where it would reach every job at once.
def synthetic_violations() -> tuple[tuple[str, Callable[[str], str], str], ...]:
    cases: list[tuple[str, Callable[[str], str], str]] = [
        (
            "job-set-is-exact",
            lambda text: text.replace("  public-api:\n", "  public-surface:\n", 1),
            "expected jobs",
        ),
        (
            "workflow-permissions-read-only",
            remove("permissions:\n  contents: read"),
            "workflow-wide permissions must be contents: read",
        ),
        (
            "workflow-permissions-read-only",
            move_permissions_into_a_job,
            "workflow-wide permissions must be contents: read",
        ),
    ]
    for token, owner, reason in REQUIRED_IN_JOB:
        cases.append(
            (
                "required-tokens-in-their-job",
                remove(token),
                f"{reason}: {token!r} is absent from job {owner}",
            )
        )
        cases.append(
            (
                "required-tokens-in-their-job",
                copy_into_another_job(token, owner),
                f"{reason}: {token!r} also appears in job",
            )
        )
        cases.append(
            (
                "required-tokens-in-their-job",
                copy_into_the_header(token),
                f"{reason}: {token!r} also appears in {HEADER_SCOPE}",
            )
        )
    return tuple(cases)


@contextmanager
def document_wide_reader() -> Iterator[None]:
    """Run the rules against [`whole_document_jobs`], the reader they must not use."""
    global split_jobs  # noqa: PLW0603 - deliberate, and restored on exit
    original = split_jobs
    split_jobs = whole_document_jobs
    try:
        yield
    finally:
        split_jobs = original


def self_test() -> None:
    live = violations(WORKFLOW.read_text())
    assert live == [], f"live release workflow fails its own audit: {live}"

    cases = synthetic_violations()
    covered = {name for name, _, _ in cases}
    declared = {name for name, _ in RULES}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )

    clean = violations(CLEAN_WORKFLOW)
    assert clean == [], f"synthetic clean workflow reported violations: {clean}"

    # Every required token sits in exactly one job of the clean workflow, so a
    # reader that handed each job the whole document would report all of them as
    # misplaced. This is the assertion that proves the ownership rule is scoped
    # to a job body, and with it that the crates.io token and the write
    # permission are confined rather than merely present.
    with document_wide_reader():
        document_wide = violations(CLEAN_WORKFLOW)
    misplaced = {
        token for token, _, _ in REQUIRED_IN_JOB if any(repr(token) in item for item in document_wide)
    }
    assert misplaced == {token for token, _, _ in REQUIRED_IN_JOB}, (
        "the clean workflow no longer distinguishes a job-scoped reader from a "
        f"document-wide one; unreported={sorted({t for t, _, _ in REQUIRED_IN_JOB} - misplaced)}"
    )

    for name, mutate, expected in cases:
        mutated = mutate(CLEAN_WORKFLOW)
        assert mutated != CLEAN_WORKFLOW, (
            f"rule {name}'s synthetic violation did not change the workflow, so it "
            "proves nothing"
        )
        found = violations(mutated)
        assert any(expected in item for item in found), (
            f"rule {name} did not fire against its synthetic violation "
            f"({expected!r}); reported {found}"
        )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        cases = synthetic_violations()
        print(
            "release workflow verifier self-test passed "
            f"({len(RULES)} rules, {len(cases)} synthetic violations; each of the "
            f"{len(REQUIRED_IN_JOB)} required tokens proven absent, proven misplaced "
            "into another job and into the workflow header, and proven confined to "
            "its own job by a reader a document-wide one cannot imitate)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_release_workflow.py [--self-test]", file=sys.stderr)
        return 2

    found = violations(WORKFLOW.read_text())
    if found:
        print("release workflow audit failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("release workflow audit: validation and least-privilege boundaries pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
