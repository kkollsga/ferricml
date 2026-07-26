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

# The one finding that is breakage nobody can observe

`cargo-semver-checks` reports `enum_no_repr_variant_discriminant_changed` when
adding a variant shifts a later variant's discriminant, and states its own
reason for calling that breaking: "This breaks downstream code that used its
value via a numeric cast like `as isize`."

That sentence names the entire observation channel, and for some enums the
channel does not exist. An `as` cast is only accepted on a *unit-only* enum —
one where every variant is fieldless — so an enum carrying data in any variant
cannot be cast at all, and there is no expression a consumer can write that
yields the number that moved. `mem::discriminant` is the only other route and
it hands back an opaque value that is neither ordered nor convertible to an
integer, and is not comparable across compilations. With no `#[repr(...)]` no
value is pinned into the public surface either, and with `#[non_exhaustive]`
the variant that caused the shift cannot break a consumer's `match`.

So this gate distinguishes an *observable* break from an unobservable one, for
that single lint, under conditions it reads out of the source tree rather than
assumes. A discriminant shift on a fieldless enum, on one carrying `#[repr]`,
on one that is not `#[non_exhaustive]`, or on one pinning explicit discriminant
values is observable and still fails, as does every other lint the report can
produce. An exempted finding is *reported* — the lint, the item, and each
condition with its evidence — never swallowed, so a reader of the release
evidence sees that a lint fired and why it was judged unobservable.

A derived `Ord` is not a fifth condition, and the reason is worth stating rather
than leaving to be rediscovered: derived comparison follows *declaration order*,
not discriminant value, and inserting a variant anywhere leaves the relative
order of every pre-existing pair unchanged. There is no comparison a consumer
can write whose answer moves. `Hash` is not one either, because a derived hash
is not stable across compilations to begin with.

The conditions are read from the **current** tree, and that is sound because
every way the baseline could have been observable while the current tree is not
is itself a separate breaking lint in the same report: removing a `#[repr]`,
adding `#[non_exhaustive]`, and turning a unit variant into a data-carrying one
are all reported on their own, are not exemptible, and keep the required level
at major.
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
from typing import Callable, NamedTuple, Sequence

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


def required_level(exit_code: int, output: str, exempted: Sequence[str] = ()) -> tuple[str, str]:
    """Read the level `cargo-semver-checks` says the diff requires.

    A non-zero exit whose summary this reader cannot parse is reported as
    `MAJOR`, because an unreadable incompatibility report is not evidence of
    compatibility.

    `exempted` names the failed checks [`unobservable_exemptions`] proved no
    consumer can observe. They are subtracted from the major count, and only
    from it — an exemption cannot be granted to a check the summary counted as
    minor, which [`unobservable_exemptions`] refuses outright. An exemption
    never takes the required level below `MINOR`: what was proved is that the
    diff breaks nobody, not that it changed nothing, and a patch step carries a
    minor requirement anyway, so nothing is gained by claiming more.
    """
    match = SUMMARY.search(output)
    if match is not None:
        major, minor = int(match.group(1)), int(match.group(2))
        detail = f"{major} major and {minor} minor checks failed"
        if exempted:
            detail = (
                f"{detail}, {len(exempted)} major exempted as unobservable "
                f"({', '.join(sorted(exempted))})"
            )
            return (MAJOR, detail) if major - len(exempted) > 0 else (MINOR, detail)
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


# ------------------------------------------------- breakage nobody can observe

# The single lint this file will consider exempting, and the only one. Its own
# description names the whole observation channel it is worried about — a
# numeric cast — which is what makes it the one lint whose premise can be
# checked and found absent. Nothing else in the report is exemptible here: this
# is one lint under stated conditions, not a switch that ignores breakage.
EXEMPTIBLE_LINT = "enum_no_repr_variant_discriminant_changed"

FAILURE_HEADER = re.compile(r"^--- failure ([a-z0-9_]+):")
FAILED_IN = "Failed in:"

# `variant ModelError::MulticlassSystemTooLarge 44 -> 46 in /abs/path.rs:228`
DISCRIMINANT_ENTRY = re.compile(
    r"^variant\s+([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(-?\d+)\s*->\s*(-?\d+)\s+in\s+(.+):(\d+)$"
)

ENUM_DECL = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b")
VARIANT_DECL = re.compile(r"^([A-Z][A-Za-z0-9_]*)\s*(.*)$")


class Variant(NamedTuple):
    name: str
    carries_data: bool
    explicit_discriminant: bool
    line: int


class EnumItem(NamedTuple):
    """One `enum` declaration as it stands in the tree being released."""

    name: str
    path: Path
    line: int
    end: int
    attributes: tuple[str, ...]
    variants: tuple[Variant, ...]

    def where(self, root: Path) -> str:
        try:
            return f"{self.path.relative_to(root)}:{self.line}"
        except ValueError:
            return f"{self.path}:{self.line}"


def _code(line: str) -> str:
    """The line with any `//` comment removed, for brace counting.

    An enum body holds no string literals, so there is nothing a `//` inside
    one could be mistaken for. Doc comments collapse to nothing, which is what
    keeps a `{` inside prose from opening a block.
    """
    index = line.find("//")
    return line if index == -1 else line[:index]


def _read_enum(
    path: Path, lines: list[str], start: int, attributes: tuple[str, ...], name: str
) -> tuple[EnumItem | None, int]:
    """Read the enum declared at `lines[start]`, and the index after its body.

    Variants are the lines at brace depth one: a struct or tuple variant opens
    a deeper block whose fields are therefore never mistaken for variants of
    their own. An unterminated body yields `None`, which refuses the exemption
    rather than guessing at a shape this reader could not follow.
    """
    depth = 0
    opened = False
    variants: list[Variant] = []
    pending = 0
    index = start
    while index < len(lines):
        code = _code(lines[index])
        stripped = code.strip()
        if pending:
            pending += stripped.count("[") - stripped.count("]")
            index += 1
            continue
        if stripped.startswith("#["):
            pending = stripped.count("[") - stripped.count("]")
            index += 1
            continue
        if opened and depth == 1 and stripped:
            match = VARIANT_DECL.match(stripped)
            if match is not None:
                remainder = match.group(2).strip()
                variants.append(
                    Variant(
                        name=match.group(1),
                        carries_data=remainder.startswith(("{", "(")),
                        explicit_discriminant="=" in remainder,
                        line=index + 1,
                    )
                )
        depth += code.count("{") - code.count("}")
        if not opened and depth > 0:
            opened = True
        if opened and depth == 0:
            item = EnumItem(
                name=name,
                path=path,
                line=start + 1,
                end=index + 1,
                attributes=attributes,
                variants=tuple(variants),
            )
            return item, index + 1
        index += 1
    return None, index


def parse_enums(path: Path) -> list[EnumItem]:
    """Every `enum` declared in one file, with the attributes attached to it.

    The attribute buffer survives blank lines and comments — both are legal
    between an attribute and the item it decorates — and is discarded at any
    other item, so an attribute cannot be attributed to a declaration further
    down the file. A `#[repr(u8)]` written inside a doc comment is never
    collected, because a comment line is not an attribute line; a doc comment
    *claiming* the enum has no `#[repr]` therefore cannot be read as one
    either.
    """
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    items: list[EnumItem] = []
    attributes: list[str] = []
    pending = 0
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if pending:
            attributes.append(stripped)
            pending += stripped.count("[") - stripped.count("]")
            index += 1
            continue
        if stripped.startswith("#["):
            attributes.append(stripped)
            pending = stripped.count("[") - stripped.count("]")
            index += 1
            continue
        if not stripped or stripped.startswith("//"):
            index += 1
            continue
        match = ENUM_DECL.match(stripped)
        if match is None:
            attributes = []
            index += 1
            continue
        item, index = _read_enum(path, lines, index, tuple(attributes), match.group(1))
        if item is not None:
            items.append(item)
        attributes = []
    return items


def find_enum(root: Path, name: str) -> tuple[EnumItem | None, str | None]:
    """The single `enum name` under `root/src`, or the reason there is not one.

    Ambiguity is a refusal rather than a choice: two declarations of the name
    mean the report cannot be tied to the declaration whose attributes were
    read, and an exemption resting on the wrong declaration is exactly the
    failure this whole section has to not commit.
    """
    source = root / "src"
    if not source.is_dir():
        return None, f"there is no `src/` under {root} to check `enum {name}` against"
    found = [
        item
        for path in sorted(source.rglob("*.rs"))
        for item in parse_enums(path)
        if item.name == name
    ]
    if not found:
        return None, f"no `enum {name}` is declared under {source}, so nothing could be checked"
    if len(found) > 1:
        places = ", ".join(item.where(root) for item in found)
        return None, (
            f"`enum {name}` is declared {len(found)} times ({places}); the report "
            f"cannot be tied to one declaration"
        )
    return found[0], None


def _is_non_exhaustive(item: EnumItem, root: Path) -> tuple[bool, str]:
    if "#[non_exhaustive]" in item.attributes:
        return True, (
            f"`{item.name}` is `#[non_exhaustive]` ({item.where(root)}), so every "
            f"consumer `match` already carries a wildcard arm and the added variants "
            f"cannot break one"
        )
    return False, (
        f"`{item.name}` is not `#[non_exhaustive]` ({item.where(root)}), so a "
        f"consumer's exhaustive `match` is part of its contract"
    )


def _has_no_repr(item: EnumItem, root: Path) -> tuple[bool, str]:
    carried = [text for text in item.attributes if text.startswith("#[repr")]
    if not carried:
        return True, (
            f"`{item.name}` carries no `#[repr(...)]`, so no discriminant value is "
            f"part of its public contract and the layout is the compiler's to choose"
        )
    return False, (
        f"`{item.name}` carries {' '.join(carried)}, which pins its discriminants "
        f"into the public contract"
    )


def _carries_data(item: EnumItem, root: Path) -> tuple[bool, str]:
    with_data = [variant.name for variant in item.variants if variant.carries_data]
    if not item.variants:
        return False, f"no variant of `{item.name}` could be read, so nothing was established"
    if with_data:
        return True, (
            f"{len(with_data)} of `{item.name}`'s {len(item.variants)} variants carry "
            f"data (first `{with_data[0]}`), so it is not a unit-only enum and "
            f"`{item.name} as isize` does not compile — the cast the lint names cannot "
            f"be written"
        )
    return False, (
        f"every one of `{item.name}`'s {len(item.variants)} variants is fieldless, so "
        f"it is a unit-only enum and `{item.name} as isize` compiles"
    )


def _has_no_explicit_discriminants(item: EnumItem, root: Path) -> tuple[bool, str]:
    explicit = [variant.name for variant in item.variants if variant.explicit_discriminant]
    if not explicit:
        return True, (
            f"no variant of `{item.name}` declares an explicit discriminant, so no "
            f"number was ever written into the surface for a reader to depend on"
        )
    return False, (
        f"`{item.name}` declares explicit discriminants on {', '.join(explicit)}, "
        f"which states those numbers as part of the surface"
    )


# Each condition is read out of the tree, in the order a reader would want to
# check them. All four must hold; the first that does not is the refusal.
CONDITIONS: tuple[tuple[str, Callable[[EnumItem, Path], tuple[bool, str]]], ...] = (
    ("non-exhaustive", _is_non_exhaustive),
    ("no-repr", _has_no_repr),
    ("carries-data", _carries_data),
    ("no-explicit-discriminants", _has_no_explicit_discriminants),
)

# Not a property of any one enum, so not a condition: it is why the conditions
# above close the last route rather than most of them.
OPAQUE_DISCRIMINANT = (
    "`mem::discriminant` yields an opaque value that is neither ordered nor "
    "convertible to an integer and is not comparable across compilations, so the "
    "shift is unobservable through it as well"
)


def unobservable_shift(entry: str, root: Path) -> tuple[bool, list[str]]:
    """Whether one `Failed in:` entry is a discriminant shift nobody can see.

    Returns the verdict and the lines that justify it, satisfied or refused
    alike. Every way of failing to establish something — an entry this reader
    cannot parse, a path that is not the declaring file, a variant the enum
    does not declare — is a refusal, so the exemption is granted only by
    evidence and never by the absence of it.
    """
    match = DISCRIMINANT_ENTRY.match(entry.strip())
    if match is None:
        return False, [
            f"`{entry.strip()}` is not in the `variant Enum::Variant old -> new in "
            f"path:line` form this reader understands, so nothing about it was checked"
        ]
    name, variant, before, after, reported_path, reported_line = match.groups()
    item, reason = find_enum(root, name)
    if item is None:
        return False, [str(reason)]
    where = Path(reported_path)
    if not where.is_absolute():
        where = root / where
    if where.resolve() != item.path.resolve():
        return False, [
            f"the report places the shift in {reported_path}, but `enum {name}` is "
            f"declared in {item.where(root)}; the two are not the same item"
        ]
    if not item.line <= int(reported_line) <= item.end:
        return False, [
            f"the report places the shift at line {reported_line}, outside `enum "
            f"{name}`'s body (lines {item.line}-{item.end})"
        ]
    if variant not in {declared.name for declared in item.variants}:
        return False, [
            f"`{name}` declares no variant `{variant}` in the tree being released, so "
            f"the report and the source disagree about what exists"
        ]
    evidence: list[str] = []
    for _, condition in CONDITIONS:
        satisfied, sentence = condition(item, root)
        if not satisfied:
            return False, [sentence]
        evidence.append(sentence)
    evidence.append(OPAQUE_DISCRIMINANT)
    evidence.append(
        f"so the move of `{name}::{variant}` from {before} to {after} is a change no "
        f"consumer can write an expression to detect"
    )
    return True, evidence


def failure_blocks(output: str) -> list[tuple[str, list[str]]]:
    """`(lint, failing items)` for every `--- failure ... ---` block reported.

    The items are the lines under `Failed in:`, which end at the first blank
    line — the report puts its summary after one.
    """
    blocks: list[tuple[str, list[str]]] = []
    collecting = False
    for raw in output.splitlines():
        stripped = raw.strip()
        header = FAILURE_HEADER.match(stripped)
        if header is not None:
            blocks.append((header.group(1), []))
            collecting = False
            continue
        if not blocks:
            continue
        if stripped == FAILED_IN:
            collecting = True
            continue
        if collecting:
            if not stripped or stripped.startswith("---"):
                collecting = False
                continue
            blocks[-1][1].append(stripped)
    return blocks


def unobservable_exemptions(output: str, root: Path) -> tuple[list[str], list[str], list[str]]:
    """`(exempted lints, evidence lines, refusal lines)` for one report.

    A block is exempted only when it is the one exemptible lint, names at least
    one failing item, and **every** one of them is proved unobservable — one
    entry this reader cannot clear keeps the whole check breaking, so an
    exemption can never cover a second finding that happened to share a block.

    Before any of that the report has to be the shape this reader assumes: a
    summary it can read, one `--- failure` block per counted check, and at
    least as many major checks as blocks it would exempt. A report that fails
    those is not evidence of anything, and every exemption is withdrawn rather
    than applied to a report that is no longer being understood.
    """
    blocks = failure_blocks(output)
    exemptible = [block for block in blocks if block[0] == EXEMPTIBLE_LINT]
    if not exemptible:
        return [], [], []

    summary = SUMMARY.search(output)
    if summary is None:
        return [], [], [
            f"{EXEMPTIBLE_LINT} was reported but the run printed no summary this "
            f"reader could count against; no exemption is applied"
        ]
    major, minor = int(summary.group(1)), int(summary.group(2))
    if len(blocks) != major + minor:
        return [], [], [
            f"the run reported {len(blocks)} failure blocks but a summary of "
            f"{major} major and {minor} minor checks; the report is not the shape "
            f"this reader parses, so no exemption is applied"
        ]
    if len(exemptible) > major:
        return [], [], [
            f"{len(exemptible)} {EXEMPTIBLE_LINT} blocks were reported against only "
            f"{major} major checks, so at least one was counted as minor; no "
            f"exemption is applied"
        ]

    exempted: list[str] = []
    evidence: list[str] = []
    refusals: list[str] = []
    for lint, entries in exemptible:
        if not entries:
            refusals.append(
                f"{lint} named no failing item, so there was nothing to check; it "
                f"still requires a breaking release"
            )
            continue
        cleared: list[str] = []
        refused: list[str] = []
        for entry in entries:
            satisfied, lines = unobservable_shift(entry, root)
            rendered = [f"  {line}" for line in lines]
            if satisfied:
                cleared.append(entry)
                cleared.extend(rendered)
            else:
                refused.append(f"{entry}: {lines[0]}")
                refused.extend(f"  {line}" for line in lines[1:])
        if refused:
            refusals.append(
                f"{lint} was considered for the unobservable-discriminant exemption "
                f"and refused; it still requires a breaking release"
            )
            refusals.extend(f"  {line}" for line in refused)
            continue
        exempted.append(lint)
        evidence.append(f"{lint}, exempted because no consumer can observe it:")
        evidence.extend(f"  {line}" for line in cleared)
    return exempted, evidence, refusals


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


# ------------------------------------------- fixtures for the exemption proofs

# Prose that names both attributes and carries a brace. None of it is an
# attribute and none of it opens a block, and the fixtures below rely on that:
# an enum whose `#[non_exhaustive]` is only claimed in its documentation must be
# refused.
DOC_PROSE = (
    "/// Prose naming #[repr(u8)] and #[non_exhaustive] while being neither, and\n"
    "/// carrying a brace `{` that opens nothing.\n"
)

CLEAN_ATTRIBUTES = "#[derive(Clone, Debug, PartialEq, Eq)]\n#[non_exhaustive]\n"

DATA_VARIANTS = """\
    /// A fieldless variant.
    EmptyData,
    /// Refused when `{rows}` differs from the sample count.
    TargetLength {
        rows: usize,
        targets: usize,
    },
    /// The variant the report says moved.
    MulticlassSystemTooLarge {
        classes: usize,
        limit: usize,
    },
"""

FIELDLESS_VARIANTS = """\
    /// A fieldless variant.
    EmptyData,
    /// Another fieldless variant.
    TargetLength,
    /// The variant the report says moved.
    MulticlassSystemTooLarge,
"""

# The last struct variant left open, so the body never closes.
UNTERMINATED_VARIANTS = DATA_VARIANTS[: DATA_VARIANTS.index("        classes: usize,")]

SHIFTED_VARIANT = "MulticlassSystemTooLarge"


def enum_source(attributes: str, variants: str, name: str = "ModelError") -> str:
    """One synthetic file declaring `name`, with a trailing impl block.

    The impl exists so the attribute buffer has something other than an enum to
    be discarded at, which is what stops an attribute from being read as
    belonging to a declaration further down the file.
    """
    return (
        "use std::fmt;\n\n"
        + DOC_PROSE
        + attributes
        + f"pub enum {name} {{\n"
        + variants
        + "}\n\n"
        + f"impl fmt::Display for {name} {{\n"
        "    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n"
        '        f.write_str("x")\n'
        "    }\n"
        "}\n"
    )


ENUM_FILE = Path("src") / "api" / "error.rs"


def write_enum_tree(root: Path, text: str, extra: str | None = None) -> Path:
    """A tree holding one enum file, and optionally a second declaration."""
    target = root / ENUM_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    if extra is not None:
        (root / "src" / "api" / "duplicate.rs").write_text(extra)
    return target


def variant_line(path: Path, *variants: str) -> int:
    """The 1-based line the first present of `variants` is declared on.

    The alternatives exist for the fixtures that delete the shifted variant: the
    report still has to name a line inside the enum body, or the case would be
    refused for the wrong reason.
    """
    lines = path.read_text().splitlines()
    for variant in variants:
        for index, line in enumerate(lines):
            if line.strip().startswith(variant):
                return index + 1
    raise AssertionError(f"{path} declares none of {variants}")


def discriminant_entry(path: Path, line: int, owner: str = "ModelError") -> str:
    return f"variant {owner}::{SHIFTED_VARIANT} 44 -> 46 in {path}:{line}"


def synthetic_report(
    blocks: Sequence[tuple[str, Sequence[str]]],
    major: int,
    minor: int = 0,
    summary: bool = True,
) -> str:
    """A report in `cargo-semver-checks` shape, carrying the given blocks.

    Shaped from the run captured in `dev-docs/temp/semver-before.txt`: a header
    line per failed check, a description, a `Failed in:` list, and one summary
    line the gate counts against.
    """
    parts: list[str] = []
    for lint, entries in blocks:
        parts.append(f"--- failure {lint}: a described change ---")
        parts.append("")
        parts.append("Description:")
        parts.append("Something about the change and the code it breaks.")
        parts.append("        ref: https://doc.rust-lang.org/reference/")
        parts.append("")
        parts.append("Failed in:")
        parts.extend(f"  {entry}" for entry in entries)
        parts.append("")
    if summary:
        parts.append(
            f"     Summary semver requires new major version: {major} major and "
            f"{minor} minor checks failed"
        )
    return "\n".join(parts) + "\n"


# Every way an enum can make its discriminant shift observable, or make the
# question unanswerable. Each is the clean fixture with one thing changed, and
# each change must turn the exemption into a refusal that still fails the gate.
#
# The first four cases are named for the condition they violate, and
# `exemption_self_test` requires that naming to cover `CONDITIONS` exactly: a
# condition without a synthetic violation is a condition nothing has shown to
# fire.
#
# `(case, attributes, variants, declared name, second declaration, refusal)`
SOURCE_REFUSALS: tuple[tuple[str, str, str, str, str | None, str], ...] = (
    (
        # The attributes are gone but the doc comment still names them, which is
        # what proves prose is not read as an attribute.
        "non-exhaustive",
        "#[derive(Clone, Debug)]\n",
        DATA_VARIANTS,
        "ModelError",
        None,
        "is not `#[non_exhaustive]`",
    ),
    (
        "no-repr",
        CLEAN_ATTRIBUTES + "#[repr(u8)]\n",
        DATA_VARIANTS,
        "ModelError",
        None,
        "carries #[repr(u8)]",
    ),
    (
        # The case the brief names: a unit-only enum can be written `as isize`,
        # so the number that moved is readable and the shift is a real break.
        "carries-data",
        CLEAN_ATTRIBUTES,
        FIELDLESS_VARIANTS,
        "ModelError",
        None,
        "every one of `ModelError`'s 3 variants is fieldless",
    ),
    (
        "no-explicit-discriminants",
        CLEAN_ATTRIBUTES,
        DATA_VARIANTS.replace("EmptyData,", "EmptyData = 10,"),
        "ModelError",
        None,
        "declares explicit discriminants on EmptyData",
    ),
    (
        "enum-renamed",
        CLEAN_ATTRIBUTES,
        DATA_VARIANTS,
        "OtherError",
        None,
        "no `enum ModelError` is declared",
    ),
    (
        "declared-twice",
        CLEAN_ATTRIBUTES,
        DATA_VARIANTS,
        "ModelError",
        enum_source(CLEAN_ATTRIBUTES, DATA_VARIANTS),
        "is declared 2 times",
    ),
    (
        "variant-absent",
        CLEAN_ATTRIBUTES,
        DATA_VARIANTS.replace(SHIFTED_VARIANT, "SomethingElse"),
        "ModelError",
        None,
        f"declares no variant `{SHIFTED_VARIANT}`",
    ),
    (
        # The body never closes, so nothing was parsed. A reader that guessed
        # here would be guessing at exactly the attributes it must not guess at.
        "unterminated-body",
        CLEAN_ATTRIBUTES,
        UNTERMINATED_VARIANTS,
        "ModelError",
        None,
        "no `enum ModelError` is declared",
    ),
)

# Ways the report itself stops being evidence. Each must be reported and must
# leave the required level at major, because a report this reader cannot follow
# is not a report of compatibility.
REPORT_REFUSALS: tuple[tuple[str, str], ...] = (
    ("entry-unparsable", "is not in the `variant Enum::Variant"),
    ("path-mismatch", "are not the same item"),
    ("line-outside-body", "outside `enum ModelError`'s body"),
    ("no-failing-item", "named no failing item"),
    ("summary-missing", "printed no summary"),
    ("block-count-mismatch", "not the shape this reader parses"),
    ("counted-as-minor", "counted as minor"),
    ("second-entry-not-exempt", "no `enum Other` is declared"),
)

# Breaking categories that are not this lint. None of them is exemptible, and
# the first is the one the brief names: a removed variant is observable by
# anything that constructed or matched it.
NON_EXEMPTIBLE_LINTS: tuple[str, ...] = (
    "enum_variant_missing",
    "enum_missing",
    "enum_repr_int_removed",
    "enum_marked_non_exhaustive",
    "inherent_method_missing",
)


def mirror_live_enum(root: Path, transform: Callable[[str], str] | None = None) -> Path:
    """Copy the live `ModelError` declaration into a synthetic tree.

    The point of mirroring rather than re-typing is that the text under test is
    the crate's own. A transform that does not change it is an assertion
    failure: a case that mutated nothing would prove nothing.
    """
    text = (ROOT / ENUM_FILE).read_text()
    if transform is not None:
        changed = transform(text)
        assert changed != text, "a live-tree mutation left the source unchanged"
        text = changed
    target = root / ENUM_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


LIVE_DECLARATION = "pub enum ModelError {"


def _drop_lines(text: str, exact: str) -> str:
    kept = [line for line in text.splitlines() if line.strip() != exact]
    return "\n".join(kept) + "\n"


# Mutations of the crate's own source, each the edit that would make the shift
# observable. They are written structurally rather than against one exact
# attribute ordering, so the proofs survive `ModelError` gaining an attribute.
LIVE_ENUM_MUTATIONS: tuple[tuple[str, Callable[[str], str], str], ...] = (
    (
        "repr-added-to-the-live-enum",
        lambda text: text.replace(LIVE_DECLARATION, f"#[repr(u8)]\n{LIVE_DECLARATION}", 1),
        "carries #[repr(u8)]",
    ),
    (
        "non-exhaustive-removed-from-the-live-enum",
        lambda text: _drop_lines(text, "#[non_exhaustive]"),
        "is not `#[non_exhaustive]`",
    ),
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

    exemption_self_test()


def _exempted_release_passes(output: str, exempted: Sequence[str]) -> bool:
    """Whether a `0.2.0 -> 0.2.1` patch release survives this report.

    The whole exemption exists to change this answer, so every case below is
    stated as the gate's own outcome rather than as an intermediate value.
    """
    required, detail = required_level(1, output, exempted)
    return not verdict((0, 2, 0), (0, 2, 0), required, detail)


def exemption_self_test() -> None:
    """Prove the unobservable-discriminant exemption fires only where stated.

    Every condition is proven against a synthetic violation that must still
    fail the gate; every way of losing the input — an unparsable entry, a
    missing source tree, a report shape this reader does not recognise — is
    proven to be reported rather than passed over; and the live `ModelError` is
    checked against the real file, twice mutated, so the exemption cannot
    survive someone adding `#[repr(u8)]` to it.
    """
    import tempfile

    assert EXEMPTIBLE_LINT not in NON_EXEMPTIBLE_LINTS
    covered = {case for case, *_ in SOURCE_REFUSALS}
    conditions = {name for name, _ in CONDITIONS}
    assert conditions <= covered, (
        f"conditions with no synthetic violation: {sorted(conditions - covered)}"
    )

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace).resolve()

        # The clean fixture: `#[non_exhaustive]`, no `#[repr]`, data-carrying
        # variants, no explicit discriminants. This is the only shape that is
        # allowed to pass, and every case below is it with one thing changed.
        clean = base / "clean"
        path = write_enum_tree(clean, enum_source(CLEAN_ATTRIBUTES, DATA_VARIANTS))
        entry = discriminant_entry(path, variant_line(path, SHIFTED_VARIANT))
        report = synthetic_report([(EXEMPTIBLE_LINT, [entry])], major=1)
        exempted, evidence, refusals = unobservable_exemptions(report, clean)
        assert exempted == [EXEMPTIBLE_LINT], f"the clean fixture was not exempted: {refusals}"
        assert not refusals, refusals
        assert any("cannot be written" in line for line in evidence), evidence
        assert required_level(1, report, exempted)[0] == MINOR, (
            "an exempted breaking check must leave a minor requirement, never none"
        )
        assert _exempted_release_passes(report, exempted), (
            "the exemption did not let a patch release carry an unobservable shift"
        )
        # Loud, not silent: the exempted lint and every condition it rested on
        # have to appear in what the gate prints.
        rendered = "\n".join(evidence)
        assert EXEMPTIBLE_LINT in rendered and SHIFTED_VARIANT in rendered, rendered
        for name, condition in CONDITIONS:
            item, reason = find_enum(clean, "ModelError")
            assert item is not None, reason
            satisfied, sentence = condition(item, clean)
            assert satisfied, f"condition {name} failed on the clean fixture: {sentence}"
            assert sentence in rendered, f"condition {name} was not reported: {rendered}"

        # Each condition, refused against a source that violates exactly it.
        for case, attributes, variants, name, extra, expected in SOURCE_REFUSALS:
            root = base / f"source-{case}"
            path = write_enum_tree(root, enum_source(attributes, variants, name), extra)
            line = variant_line(path, SHIFTED_VARIANT, "EmptyData")
            report = synthetic_report(
                [(EXEMPTIBLE_LINT, [discriminant_entry(path, line)])], major=1
            )
            exempted, evidence, refusals = unobservable_exemptions(report, root)
            assert exempted == [] and not evidence, (
                f"{case} was exempted; the gate would pass an observable break: {evidence}"
            )
            assert any(expected in line for line in refusals), (
                f"{case} was refused without saying why: {refusals}"
            )
            assert required_level(1, report, exempted)[0] == MAJOR, case
            assert not _exempted_release_passes(report, exempted), (
                f"{case} let a patch release carry an observable break"
            )

        # The source tree disappearing underneath the check is a finding, not a
        # pass: without `src/` there is nothing the conditions were read from.
        root = base / "source-missing"
        root.mkdir(parents=True)
        report = synthetic_report(
            [(EXEMPTIBLE_LINT, [discriminant_entry(root / ENUM_FILE, 30)])], major=1
        )
        exempted, _, refusals = unobservable_exemptions(report, root)
        assert exempted == [] and any("there is no `src/`" in line for line in refusals), refusals
        assert not _exempted_release_passes(report, exempted)

        # Ways the report stops being evidence. The tree is clean throughout, so
        # what is refused is the report rather than the enum.
        shapes = base / "shapes"
        path = write_enum_tree(shapes, enum_source(CLEAN_ATTRIBUTES, DATA_VARIANTS))
        line = variant_line(path, SHIFTED_VARIANT)
        good = discriminant_entry(path, line)
        elsewhere = shapes / "src" / "api" / "other.rs"
        reports: dict[str, str] = {
            "entry-unparsable": synthetic_report(
                [(EXEMPTIBLE_LINT, ["something this reader has never seen"])], major=1
            ),
            "path-mismatch": synthetic_report(
                [(EXEMPTIBLE_LINT, [discriminant_entry(elsewhere, line)])], major=1
            ),
            "line-outside-body": synthetic_report(
                [(EXEMPTIBLE_LINT, [discriminant_entry(path, 9999)])], major=1
            ),
            "no-failing-item": synthetic_report([(EXEMPTIBLE_LINT, [])], major=1),
            "summary-missing": synthetic_report(
                [(EXEMPTIBLE_LINT, [good])], major=1, summary=False
            ),
            "block-count-mismatch": synthetic_report(
                [(EXEMPTIBLE_LINT, [good]), ("enum_variant_missing", ["variant X::Y"])], major=1
            ),
            "counted-as-minor": synthetic_report(
                [(EXEMPTIBLE_LINT, [good])], major=0, minor=1
            ),
            "second-entry-not-exempt": synthetic_report(
                [
                    (
                        EXEMPTIBLE_LINT,
                        [good, f"variant Other::{SHIFTED_VARIANT} 1 -> 2 in {path}:{line}"],
                    )
                ],
                major=1,
            ),
        }
        assert set(reports) == {case for case, _ in REPORT_REFUSALS}, sorted(reports)
        for case, expected in REPORT_REFUSALS:
            report = reports[case]
            exempted, evidence, refusals = unobservable_exemptions(report, shapes)
            assert exempted == [] and not evidence, f"{case} was exempted: {evidence}"
            assert any(expected in line for line in refusals), f"{case}: {refusals}"
            # The level is whatever the report itself says, unchanged: a refused
            # exemption must be indistinguishable from no exemption at all.
            assert required_level(1, report, exempted) == required_level(1, report), case
            counted = SUMMARY.search(report)
            if counted is None or int(counted.group(1)) > 0:
                assert required_level(1, report, exempted)[0] == MAJOR, case
                assert not _exempted_release_passes(report, exempted), case

        # Every other breaking category, including a removed variant, is
        # untouched by this exemption — alone and alongside an exemptible one.
        for lint in NON_EXEMPTIBLE_LINTS:
            report = synthetic_report([(lint, [f"variant ModelError::Gone in {path}:{line}"])], 1)
            exempted, evidence, refusals = unobservable_exemptions(report, shapes)
            assert (exempted, evidence, refusals) == ([], [], []), lint
            assert required_level(1, report, exempted)[0] == MAJOR, lint
            assert not _exempted_release_passes(report, exempted), (
                f"{lint} was allowed onto a patch release"
            )

            mixed = synthetic_report(
                [
                    (EXEMPTIBLE_LINT, [good]),
                    (lint, [f"variant ModelError::Gone in {path}:{line}"]),
                ],
                major=2,
            )
            exempted, _, _ = unobservable_exemptions(mixed, shapes)
            assert exempted == [EXEMPTIBLE_LINT], lint
            assert required_level(1, mixed, exempted)[0] == MAJOR, (
                f"the exemption swallowed {lint} sharing a report with it"
            )
            assert not _exempted_release_passes(mixed, exempted), lint

        # The live declaration, read out of `src/api/error.rs` itself. The
        # unmutated mirror is exempted, and the two mutations that would make
        # the shift observable are refused — which is what makes this a check of
        # the code rather than a hardcoded belief about `ModelError`.
        mirror = base / "live"
        path = mirror_live_enum(mirror)
        line = variant_line(path, SHIFTED_VARIANT)
        report = synthetic_report([(EXEMPTIBLE_LINT, [discriminant_entry(path, line)])], major=1)
        exempted, evidence, refusals = unobservable_exemptions(report, mirror)
        assert exempted == [EXEMPTIBLE_LINT], (
            f"the live `ModelError` no longer satisfies the exemption: {refusals}"
        )
        assert _exempted_release_passes(report, exempted)

        for case, mutate, expected in LIVE_ENUM_MUTATIONS:
            root = base / f"live-{case}"
            path = mirror_live_enum(root, mutate)
            line = variant_line(path, SHIFTED_VARIANT)
            report = synthetic_report(
                [(EXEMPTIBLE_LINT, [discriminant_entry(path, line)])], major=1
            )
            exempted, _, refusals = unobservable_exemptions(report, root)
            assert exempted == [], f"{case} was still exempted"
            assert any(expected in line for line in refusals), f"{case}: {refusals}"
            assert not _exempted_release_passes(report, exempted), case

    # The live tree itself, not a mirror of it. The fixtures above are a
    # description of a crate; this is the crate. `ModelError` is the enum the
    # 2026-07-26 finding named, and if it stops satisfying a condition the
    # exemption stops applying — which is a decision to take here, deliberately,
    # rather than to discover at release time from a gate that has quietly
    # started failing again.
    item, reason = find_enum(ROOT, "ModelError")
    assert item is not None, f"the live tree no longer declares one `ModelError`: {reason}"
    assert len(item.variants) >= 20, (
        f"only {len(item.variants)} variants of `ModelError` were parsed out of "
        f"{item.where(ROOT)}; the reader is no longer reading the declaration"
    )
    for name, condition in CONDITIONS:
        satisfied, sentence = condition(item, ROOT)
        assert satisfied, (
            f"live `ModelError` no longer satisfies the {name} condition ({sentence}), "
            f"so a discriminant shift on it is now observable; either revert that or "
            f"drop the exemption from this file and take the release as breaking"
        )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "semver gate self-test passed "
            f"({len(SELF_TEST_CASES)} release-level cases, both 0.x breaking "
            "components, and an unreadable report classified as breaking; "
            f"{len(CONDITIONS)} unobservability conditions each proven to refuse "
            f"a synthetic violation, {len(SOURCE_REFUSALS)} source refusals and "
            f"{len(REPORT_REFUSALS)} report refusals proven to keep the gate "
            f"failing, {len(NON_EXEMPTIBLE_LINTS)} other breaking categories "
            f"proven untouched alone and alongside an exempted one, and the live "
            f"`ModelError` read out of {ENUM_FILE} under "
            f"{len(LIVE_ENUM_MUTATIONS)} mutations)"
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

    # Reported before the verdict, and whether or not the gate goes on to fail:
    # an exempted finding is a lint that fired, and a reader of the release
    # evidence has to see both that it fired and the grounds on which it was
    # judged unobservable.
    exempted, evidence, refusals = unobservable_exemptions(output, ROOT)
    if evidence:
        plural = "" if len(exempted) == 1 else "s"
        print(
            f"semver: {len(exempted)} reported breaking check{plural} cannot be "
            f"observed by any consumer and therefore do{'es' if not plural else ''} "
            f"not raise the required release level:"
        )
        for line in evidence:
            print(f"  {line}")
    for line in refusals:
        print(line if line.startswith(" ") else f"semver: {line}")

    required, detail = required_level(exit_code, output, exempted)

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
