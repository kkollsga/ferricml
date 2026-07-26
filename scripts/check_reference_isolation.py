#!/usr/bin/env python3
"""Keep external reference identities out of FerricML's working tree."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (b"sci" + b"kit", b"sk" + b"learn")

# What a developer is about to commit, not only what is already committed.
#
# The scope is the index *plus* everything untracked that git would not ignore,
# because that is the set `git add -A` would stage and therefore the set the
# next commit can contain. Reading `git ls-files` alone — the reader this file
# used until 2026-07-26 — meant a new file passed `make gate` and failed only
# after `git add`, which was reported three times independently and hit for real
# by `requirements/docs.txt`. Ignored paths stay out: `--exclude-standard`
# applies `.gitignore`, `.git/info/exclude` and the user's `core.excludesFile`,
# so `target/`, `research/`, `site/`, `.venv-docs/` and the local `dev-docs/`
# working folder are neither scanned nor scannable. They cannot reach a commit,
# and a local reference workspace under an ignored path is a supported way to
# work in this crate rather than a leak.
SCAN: tuple[str, ...] = (
    "git",
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
    "-z",
)

# The reader this file used before the scope above, kept so `self_test` can
# prove the difference is observable rather than asserted.
INDEX_ONLY: tuple[str, ...] = ("git", "ls-files", "-z")

# The scope one flag too wide: untracked *and* ignored. Kept for the same
# reason, in the other direction.
UNFILTERED: tuple[str, ...] = ("git", "ls-files", "--cached", "--others", "-z")

# Paths every scan of this repository must reach. A checker that reports success
# because it found nothing to look at is the defect class this crate keeps
# hitting, and `git ls-files` exits zero with empty output for several ways of
# getting the invocation wrong — wrong directory, a pathspec that matches
# nothing, a flag combination that filters everything away.
ANCHORS: tuple[str, ...] = ("Cargo.toml", "README.md", "src/lib.rs", "CHANGELOG.md")

Entry = tuple[str, bytes]


def forbidden_identity_in_path(entries: list[Entry], anchors: tuple[str, ...]) -> list[str]:
    del anchors
    return [
        f"forbidden external identity in path: {name}"
        for name, _ in entries
        if any(term in name.encode().lower() for term in FORBIDDEN)
    ]


def forbidden_identity_in_content(entries: list[Entry], anchors: tuple[str, ...]) -> list[str]:
    del anchors
    return [
        f"forbidden external identity in content: {name}"
        for name, content in entries
        if any(term in content.lower() for term in FORBIDDEN)
    ]


def scan_covers_anchors(entries: list[Entry], anchors: tuple[str, ...]) -> list[str]:
    seen = {name for name, _ in entries}
    return [
        f"scan did not reach {anchor}: the check would report success without reading it"
        for anchor in anchors
        if anchor not in seen
    ]


RULES: tuple[tuple[str, Callable[[list[Entry], tuple[str, ...]], list[str]]], ...] = (
    ("forbidden-identity-in-path", forbidden_identity_in_path),
    ("forbidden-identity-in-content", forbidden_identity_in_content),
    ("scan-covers-anchors", scan_covers_anchors),
)


def violations(entries: list[Entry], anchors: tuple[str, ...] = ANCHORS) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(entries, anchors))
    return found


def scanned_names(root: Path) -> list[str]:
    result = subprocess.run(list(SCAN), cwd=root, check=True, capture_output=True)
    names: list[str] = []
    seen: set[str] = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        name = raw.decode()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def scanned_entries(root: Path = ROOT) -> list[Entry]:
    """Every scanned path, with its bytes where it has any.

    An index entry whose worktree file is gone (a staged deletion) and a
    symlink have no content to scan, but their *names* are still part of the
    commit and are still checked.
    """
    entries: list[Entry] = []
    for name in scanned_names(root):
        path = root / name
        readable = path.is_file() and not path.is_symlink()
        entries.append((name, path.read_bytes() if readable else b""))
    return entries


@contextmanager
def scan_scope(args: tuple[str, ...]) -> Iterator[None]:
    """Run the scan under a different git file-listing scope."""
    global SCAN  # noqa: PLW0603 - deliberate, and restored on exit
    original = SCAN
    SCAN = args
    try:
        yield
    finally:
        SCAN = original


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


FIRST, SECOND = (term.decode() for term in FORBIDDEN)

# The untracked files of the synthetic repository below. They are anchors, so
# a global `core.excludesFile` that happened to hide one would fail the
# self-test loudly instead of quietly removing the case it exists to prove.
CLEAN_ANCHORS: tuple[str, ...] = (
    "Cargo.toml",
    "README.md",
    "src/lib.rs",
    "requirements/docs.txt",
    "docs/draft.md",
)


def write_clean_repo(root: Path) -> Path:
    """Write the smallest repository that must scan clean.

    It carries all three kinds of path the scope decision is about: tracked
    files, untracked files git would not ignore, and ignored files. The ignored
    ones carry a forbidden identity **in the clean repository**, so the
    assertion that this repository reports nothing is itself the proof that
    ignored paths are outside the scan; a reader one flag too wide reports them.
    """
    for relative, text in {
        ".gitignore": "/target\n/research\n",
        "Cargo.toml": '[package]\nname = "ferricml"\n',
        "README.md": "FerricML is a lean pure-Rust toolkit.\n",
        "src/lib.rs": "//! FerricML owns this contract\n",
        # Untracked and not ignored: the shape that passed the gate before
        # staging and failed after it.
        "requirements/docs.txt": "mkdocs==1.6.1\n",
        "docs/draft.md": "A page still being written.\n",
        # Ignored: a local comparison workspace and its build output.
        "target/leak.md": f"generated notes mentioning {FIRST}\n",
        "research/leak.md": f"local experiment against {SECOND}\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".gitignore", "Cargo.toml", "README.md", "src/lib.rs")
    return root


def stage(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(root, "add", "--", relative)


def write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def drop_anchor(root: Path) -> None:
    git(root, "rm", "-q", "--cached", "--", "README.md")
    (root / "README.md").unlink()


# One synthetic violation per rule. The identity rules are violated in an
# **untracked** path, so each doubles as proof that the scan is the working
# tree rather than the index: `self_test` asserts each fires under [`SCAN`] and
# stays silent under [`INDEX_ONLY`], the reader this file used until the scope
# was widened.
SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "forbidden-identity-in-path",
        lambda root: write(root, f"docs/{FIRST}-comparison.md", "notes\n"),
        "forbidden external identity in path",
    ),
    (
        "forbidden-identity-in-content",
        lambda root: write(root, "requirements/docs.txt", f"# ported from {SECOND}\n"),
        "forbidden external identity in content",
    ),
    (
        "scan-covers-anchors",
        drop_anchor,
        "scan did not reach README.md",
    ),
)

# The same two identity rules again, this time in a **tracked** path. They fire
# under either scope, and are here so the rules themselves stay proven
# independently of the scope decision above.
TRACKED_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "forbidden-identity-in-path",
        lambda root: stage(root, f"src/{SECOND}_port.rs", "//! port\n"),
        "forbidden external identity in path",
    ),
    (
        "forbidden-identity-in-content",
        lambda root: stage(root, "README.md", f"FerricML, unlike {FIRST}.\n"),
        "forbidden external identity in content",
    ),
)

# `scan-covers-anchors` protects a *non-firing* property — the scan reaching a
# path — so no violation of it can demonstrate the scope by firing: widening the
# scope can only ever remove one of its reports. It is proven instead by the
# clean repository, whose anchors include two untracked files, and which
# `self_test` asserts reports nothing under [`SCAN`] and reports them missing
# under [`INDEX_ONLY`]. Its entry in `SYNTHETIC_VIOLATIONS` covers the rule's
# other half: that it fires at all.
CLEAN_REPO_PROVEN_SCOPE: tuple[str, ...] = ("scan-covers-anchors",)


def self_test() -> None:
    live = scanned_entries()
    found = violations(live)
    assert found == [], f"live working tree violates reference isolation: {found}"

    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    declared = {name for name, _ in RULES}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )
    stale = {name for name, _, _ in TRACKED_VIOLATIONS} - declared
    assert not stale, f"stale tracked violations: {sorted(stale)}"
    stale = set(CLEAN_REPO_PROVEN_SCOPE) - declared
    assert not stale, f"stale clean-repository scope exemption: {sorted(stale)}"

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)

        clean = write_clean_repo(base / "clean")
        found = violations(scanned_entries(clean), CLEAN_ANCHORS)
        assert found == [], f"synthetic clean repository reported violations: {found}"

        # The clean repository's ignored paths carry a forbidden identity, so a
        # scan that dropped `--exclude-standard` reports them. This is the
        # assertion that proves `target/` and `research/` are outside the scan.
        with scan_scope(UNFILTERED):
            unfiltered = violations(scanned_entries(clean), CLEAN_ANCHORS)
        assert any("target/leak.md" in item for item in unfiltered) and any(
            "research/leak.md" in item for item in unfiltered
        ), f"ignored paths are no longer excluded by a flag the scan controls: {unfiltered}"

        # Its untracked files are anchors, so the index-only reader cannot
        # reach them. This is the assertion that proves the scan is the working
        # tree rather than the index.
        with scan_scope(INDEX_ONLY):
            index_only = violations(scanned_entries(clean), CLEAN_ANCHORS)
        assert any("requirements/docs.txt" in item for item in index_only), (
            "the clean repository no longer distinguishes a working-tree scan "
            f"from an index-only one; reported {index_only}"
        )

        for index, (name, mutate, expected) in enumerate(SYNTHETIC_VIOLATIONS):
            repo = write_clean_repo(base / f"untracked-{index}-{name}")
            mutate(repo)
            found = violations(scanned_entries(repo), CLEAN_ANCHORS)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against its synthetic violation; "
                f"reported {found}"
            )
            if name in CLEAN_REPO_PROVEN_SCOPE:
                continue
            with scan_scope(INDEX_ONLY):
                index_only = violations(scanned_entries(repo), CLEAN_ANCHORS)
            assert not any(expected in item for item in index_only), (
                f"rule {name}'s violation is also reported by an index-only "
                "reader, so it does not prove the scan covers unstaged work; "
                f"reported {index_only}"
            )

        for index, (name, mutate, expected) in enumerate(TRACKED_VIOLATIONS):
            repo = write_clean_repo(base / f"tracked-{index}-{name}")
            mutate(repo)
            found = violations(scanned_entries(repo), CLEAN_ANCHORS)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against a tracked violation; reported {found}"
            )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "reference isolation verifier self-test passed "
            f"({len(RULES)} rules, each proven against a synthetic violation; "
            f"{len(SYNTHETIC_VIOLATIONS) - len(CLEAN_REPO_PROVEN_SCOPE)} of them "
            "proven again against a violation in an untracked path an index-only "
            "reader misses, and ignored paths proven outside the scan)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_reference_isolation.py [--self-test]", file=sys.stderr)
        return 2

    found = violations(scanned_entries())
    if found:
        print("reference isolation check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("reference isolation: committable paths and content are generic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
