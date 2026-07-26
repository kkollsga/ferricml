#!/usr/bin/env python3
"""Gate the release level against the compatibility of the analysed diff.

This used to print `cargo-semver-checks` evidence and exit `0` whether the diff
was clean or carried breaking changes. Evidence nobody can fail is not a gate:
at `4e4e2f6` the report read "semver requires new major version: 8 major and 0
minor checks failed" and the command still succeeded, while three documents
independently instructed the next release to increment only the patch. For a
`0.x` crate that is the worst available direction, because Cargo treats the
minor as the breaking component below `1.0` and both `^0.1.2` and `"0.1"`
accept `0.1.3`.

So the report is now a gate. It compares two things it derives independently:

* the **required** level, read from what `cargo-semver-checks` found in the
  diff, and
* the **permitted** level, derived from the version step the release would
  actually take — which, while `Cargo.toml` still sits on the published
  baseline, is the version the patch-default policy would produce.

A required level the permitted step cannot carry fails. This does not decide
the release level and must never be silenced by editing a version to suit it:
choosing to release a breaking change as a minor bump is the user's decision,
and this gate exists so that decision is made rather than defaulted into.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Cargo.toml"
CRATE_API = "https://crates.io/api/v1/crates/ferricml"

Version = tuple[int, int, int]

# `cargo-semver-checks` classifies a diff by the level of release it requires.
# The names are its own; the ordering is what lets a permitted step be compared
# with a required one.
NONE, MINOR, MAJOR = "none", "minor", "major"
RANK: dict[str, int] = {NONE: 0, MINOR: 1, MAJOR: 2}

SUMMARY = re.compile(r"(\d+) major and (\d+) minor checks failed")
PACKAGE_VERSION = re.compile(r'(?m)^version\s*=\s*"([^"]+)"')
RELEASE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> Version:
    """Read the `major.minor.patch` prefix of a version string.

    Pre-release and build metadata are deliberately dropped: they do not change
    which component a compatible step may move, and FerricML releases only
    plain `X.Y.Z` versions anyway.
    """
    match = RELEASE.match(text.strip())
    if match is None:
        raise ValueError(f"unreadable version {text!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def render(version: Version) -> str:
    return ".".join(str(part) for part in version)


def declared_version() -> Version:
    """Read the version out of the manifest's `[package]` table.

    Scoped to that table so a version pinned on a dependency further down the
    file can never be mistaken for the crate's own.
    """
    text = MANIFEST.read_text()
    start = text.index("[package]")
    end = text.find("\n[", start + 1)
    package = text[start : end if end != -1 else len(text)]
    match = PACKAGE_VERSION.search(package)
    if match is None:
        raise ValueError("Cargo.toml [package] declares no version")
    return parse_version(match.group(1))


def breaking_component(baseline: Version) -> int:
    """The index of the component Cargo treats as breaking for this baseline.

    Below `1.0` the leading zeros shift that leftwards: `^0.1.2` admits
    `<0.2.0`, so the minor is breaking; `^0.0.3` admits only `0.0.3`, so every
    component is. This is the whole reason a patch-default policy is dangerous
    for a `0.x` crate and safe for a `1.x` one.
    """
    for index, part in enumerate(baseline):
        if part != 0:
            return index
    return len(baseline) - 1


def permitted_level(baseline: Version, planned: Version) -> str:
    """The strongest class of change the step `baseline -> planned` may carry.

    `MAJOR` means the step is a breaking release under Cargo's rules for this
    baseline; `MINOR` means it may add but not break; `NONE` means consumers
    resolving the baseline get the new version and expect no surface change at
    all.
    """
    if planned < baseline:
        raise ValueError(f"planned {render(planned)} is behind published {render(baseline)}")
    if planned == baseline:
        return NONE
    breaking = breaking_component(baseline)
    first = next(index for index in range(len(baseline)) if planned[index] != baseline[index])
    if first <= breaking:
        return MAJOR
    if first == breaking + 1:
        return MINOR
    return NONE


def patch_default(baseline: Version) -> Version:
    """The release the standing patch-default policy would produce."""
    return (baseline[0], baseline[1], baseline[2] + 1)


def planned_release(baseline: Version, declared: Version) -> tuple[Version, str]:
    """The version this release would carry, and where that number came from.

    While the manifest still sits on the published baseline no level has been
    chosen yet, so the honest thing to gate is the one the standing policy
    would produce by default. Once the manifest is ahead, a level *has* been
    chosen explicitly and that choice is what gets checked.
    """
    if declared > baseline:
        return declared, "the version already declared in Cargo.toml"
    return patch_default(baseline), "the patch-default policy applied to the published baseline"


def required_level(exit_code: int, output: str) -> tuple[str, str]:
    """Read the level `cargo-semver-checks` says the diff requires.

    A non-zero exit whose summary this reader cannot parse is reported as
    `MAJOR`, because an unreadable incompatibility report is not evidence of
    compatibility.
    """
    match = SUMMARY.search(output)
    if match is not None:
        major, minor = int(match.group(1)), int(match.group(2))
        detail = f"{major} major and {minor} minor checks failed"
        if major:
            return MAJOR, detail
        if minor:
            return MINOR, detail
        return (NONE, detail) if exit_code == 0 else (MAJOR, f"{detail}, but the run failed")
    if exit_code == 0:
        return NONE, "no incompatibility reported"
    return MAJOR, "the run failed and its summary could not be read"


def verdict(baseline: Version, declared: Version, required: str, detail: str) -> list[str]:
    """Every reason this release level cannot carry this diff.

    The gate is compatibility, not level equality. A diff that only *adds* is
    compatible with a patch release — no consumer resolving the baseline
    breaks — so the standing patch-default policy keeps working for the
    ordinary case, which is the point: this gate is meant to stop a breaking
    diff, not to overrule the policy on every release that grows the surface.
    A diff that **breaks** needs a step Cargo treats as breaking.
    """
    if declared < baseline:
        return [
            f"Cargo.toml is on {render(declared)}, behind the published "
            f"{render(baseline)}; nothing can be released from here"
        ]
    planned, origin = planned_release(baseline, declared)
    permitted = permitted_level(baseline, planned)
    if RANK[required] < RANK[MAJOR] or permitted == MAJOR:
        return []
    breaking = breaking_component(baseline)
    component = ("major", "minor", "patch")[breaking]
    needed = list(baseline)
    needed[breaking] += 1
    for index in range(breaking + 1, len(needed)):
        needed[index] = 0
    return [
        f"the diff requires a {required} release ({detail}), but releasing "
        f"{render(planned)} over published {render(baseline)} is a {permitted} "
        f"step that Cargo treats as compatible",
        f"{render(planned)} comes from {origin}",
        f"for a baseline of {render(baseline)} the breaking component is the "
        f"{component}, so a breaking diff needs at least {render(tuple(needed))}",
        "the patch default does not apply to this diff; stop and take an "
        "explicit decision on the release level rather than editing the "
        "version to satisfy this gate",
    ]


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


def run_semver_checks(baseline: str) -> tuple[int, str]:
    """Run the comparison, streaming its report while capturing it for the gate.

    The full report is the evidence a reader needs, so it is printed verbatim;
    the capture exists only so the summary line can be classified.
    """
    result = subprocess.run(
        ["cargo", "semver-checks", "check-release", "--baseline-version", baseline],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, end="")
    return result.returncode, result.stdout


# baseline, declared, required -> whether the gate must pass. The table is the
# contract: the top block is the state that shipped this finding, and every
# other row exists so a later edit cannot quietly turn one of them around.
SELF_TEST_CASES: tuple[tuple[str, str, str, bool, str], ...] = (
    # The finding, at 4e4e2f6: eight breaking changes and a patch default.
    ("0.1.2", "0.1.2", MAJOR, False, "0.x patch default cannot carry a breaking diff"),
    # The same tree once the recorded minor decision is in the manifest.
    ("0.1.2", "0.2.0", MAJOR, True, "0.x minor bump is the breaking step"),
    ("0.1.2", "1.0.0", MAJOR, True, "a major bump carries anything"),
    # An additive diff is compatible with any forward step, so the standing
    # patch default survives it. This gate stops breakage; it does not overrule
    # the policy on every release that grows the surface.
    ("0.1.2", "0.1.2", MINOR, True, "0.x patch default may add"),
    ("0.1.2", "0.1.2", NONE, True, "a clean diff needs nothing"),
    # Above 1.0 the same policy is safe for additions and unsafe for breaks.
    ("1.4.2", "1.4.2", MINOR, True, "1.x patch default may add"),
    ("1.4.2", "1.4.2", MAJOR, False, "1.x patch default cannot carry a breaking diff"),
    ("1.4.2", "1.4.3", MAJOR, False, "an explicit 1.x patch cannot carry one either"),
    ("1.4.2", "1.5.0", MAJOR, False, "1.x minor bump is not the breaking step"),
    ("1.4.2", "2.0.0", MAJOR, True, "1.x major bump is the breaking step"),
    # Below 0.1 every increment is breaking, so the default is always enough.
    ("0.0.3", "0.0.3", MAJOR, True, "0.0.x patch default is itself a breaking step"),
    # A manifest behind the registry cannot release at all.
    ("0.1.2", "0.1.1", NONE, False, "the tree is behind the published crate"),
)


def self_test() -> None:
    for baseline, declared, required, expected, reason in SELF_TEST_CASES:
        found = verdict(parse_version(baseline), parse_version(declared), required, "self-test")
        passed = not found
        assert passed == expected, (
            f"{reason}: baseline {baseline}, declared {declared}, required "
            f"{required} was expected to {'pass' if expected else 'fail'}; "
            f"reported {found}"
        )

    # The table above is a fixture; this is the live tree. Reading the real
    # manifest is what stops the fixture from drifting into a description of a
    # crate FerricML no longer is. Below `0.1` the patch is itself the breaking
    # component, so the default would be sufficient and there is nothing to
    # assert — that exemption is stated rather than silently skipped.
    live = declared_version()
    assert render(live) in MANIFEST.read_text(), "the manifest version did not round-trip"
    if breaking_component(live) != 2:
        assert verdict(live, live, MAJOR, "self-test"), (
            f"the patch default over {render(live)} was allowed to carry a breaking diff"
        )
        assert not verdict(live, live, NONE, "self-test"), (
            f"the patch default over {render(live)} was refused a clean diff"
        )

    # Reading the report: the summary decides the level, and an unreadable
    # failure is a breaking diff rather than a silent pass.
    assert required_level(1, "Summary semver requires new major version: 8 major and 0 minor checks failed")[0] == MAJOR
    assert required_level(1, "Summary semver requires new minor version: 0 major and 3 minor checks failed")[0] == MINOR
    assert required_level(0, "Checked 100 items: 0 major and 0 minor checks failed")[0] == NONE
    assert required_level(0, "no summary here")[0] == NONE
    assert required_level(101, "the run died before it printed a summary")[0] == MAJOR
    assert required_level(1, "0 major and 0 minor checks failed")[0] == MAJOR

    # The level a step may carry is a property of the baseline's leading zeros,
    # not of which component moved.
    assert permitted_level((0, 1, 2), (0, 1, 3)) == MINOR
    assert permitted_level((0, 1, 2), (0, 2, 0)) == MAJOR
    assert permitted_level((1, 4, 2), (1, 4, 3)) == NONE
    assert permitted_level((1, 4, 2), (1, 5, 0)) == MINOR
    assert permitted_level((1, 4, 2), (2, 0, 0)) == MAJOR
    assert permitted_level((0, 0, 3), (0, 0, 4)) == MAJOR


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "semver gate self-test passed "
            f"({len(SELF_TEST_CASES)} release-level cases, both 0.x breaking "
            "components, and an unreadable report classified as breaking)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_semver.py [--self-test]", file=sys.stderr)
        return 2

    try:
        baseline_text = published_baseline()
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"semver: unable to query crates.io: {error}", file=sys.stderr)
        return 1

    if baseline_text is None:
        print("semver: no published baseline for ferricml; first release has nothing to compare")
        return 0

    if shutil.which("cargo-semver-checks") is None:
        print(
            f"semver: latest published baseline is {baseline_text}, but cargo-semver-checks "
            "is not installed",
            file=sys.stderr,
        )
        return 1

    baseline = parse_version(baseline_text)
    declared = declared_version()
    exit_code, output = run_semver_checks(baseline_text)
    required, detail = required_level(exit_code, output)

    found = verdict(baseline, declared, required, detail)
    if found:
        print(f"semver: release level FAILS against published {baseline_text}:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1

    planned, _ = planned_release(baseline, declared)
    print(
        f"semver: releasing {render(planned)} over published {baseline_text} is a "
        f"{permitted_level(baseline, planned)} step and the diff requires "
        f"{required} ({detail})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
