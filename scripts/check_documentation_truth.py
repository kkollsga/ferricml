#!/usr/bin/env python3
"""Detect documentation that contradicts the code it documents.

Documentation drifting away from behavior is FerricML's most active defect
class and was, until this file, the only one with nothing watching it. Rustdoc
already checks *intra-doc links* under `-D warnings`, so this checker
deliberately does not re-do that work. It targets the gap rustdoc cannot see:
prose that makes a **checkable claim about the code** without being a link — a
member name, a trait bound, a capability enumeration, a repository path.

Each rule is named, and `--self-test` asserts that every one of them still
fires against a synthetic violation *and* reports a finding when its input
disappears. A rule that quietly stopped matching would otherwise pass both the
check and its own self-test, which is the failure mode this crate has been
bitten by before.

Two historical defects are reconstructed as synthetic inputs and asserted to
fire; see `HISTORICAL_REGRESSIONS`.

What this deliberately does **not** check is recorded in `docs/api-and-growth.md`
under "What the documentation checker does not check", so the blind spots are
written down rather than discovered later.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]

# Directories whose prose is checked. `src` is rustdoc; `docs` is the narrative
# site, which rustdoc never sees at all and which therefore has no other
# mechanical reader than this one.
RUST_SOURCE = "src"
NARRATIVE = "docs"

# Prefixes that make a backticked token a repository path rather than a module
# path or a shell fragment. `dev-docs/` and `inbox/` are deliberately absent:
# both are gitignored local working state, so a fresh clone — CI's, and a
# packaged consumer's — has neither, and checking them would fail everywhere
# except the machine that happens to own them.
REPOSITORY_PATH_ROOTS = (
    "src/",
    "tests/",
    "scripts/",
    "benches/",
    "docs/",
)

# The capability vocabulary, and the words a doc comment may use to name each
# one. A declaration that turns a capability on is a claim the prose beside it
# has to carry; the alternatives exist because the docs say "persistence" and
# "weighted fitting" rather than the field spellings.
CAPABILITY_WORDS: dict[str, tuple[str, ...]] = {
    "sample_weights": ("sample_weights", "sample weight", "weighted", "weighting"),
    "artifact": ("artifact", "persist"),
    "multiclass": ("multiclass",),
    "decision_function": ("decision_function", "decision function", "decision score"),
    "probability": ("probabilit",),
}

# Members every derive contributes. Without these, a doc naming
# `ParamsType::default()` reads as a dangling reference.
DERIVE_MEMBERS: dict[str, tuple[str, ...]] = {
    "Default": ("default",),
    "Clone": ("clone", "clone_from"),
    "Copy": (),
    "Debug": ("fmt",),
    "PartialEq": ("eq", "ne"),
    "Eq": (),
    "PartialOrd": ("partial_cmp", "lt", "le", "gt", "ge"),
    "Ord": ("cmp", "max", "min", "clamp"),
    "Hash": ("hash",),
}

DECLARATION = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(struct|enum|trait|union)\s+"
    r"([A-Z][A-Za-z0-9_]*)"
)
IMPLEMENTATION = re.compile(
    r"^\s*impl(?:<[^>]*>)?\s+(?:(?P<trait>[A-Za-z_$][\w:$]*)(?:<[^>]*>)?\s+for\s+)?"
    r"(?P<target>[A-Za-z_$][\w$]*)"
)
ASSOCIATED_ITEM = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+fn|async\s+fn|unsafe\s+fn"
    r"|extern\s+fn|fn|const|type)\s+([A-Za-z_][\w]*)"
)
FIELD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z_][\w]*)\s*:")
VARIANT = re.compile(r"^\s*([A-Z][\w]*)\s*(?:\{|\(|,|=|$)")
DERIVE = re.compile(r"^\s*#\[derive\(([^)]*)\)\]")

CAPABILITY_IMPL = re.compile(
    r"^\s*impl(?:<[^>]*>)?\s+(?:\$crate::api::)?HasCapabilities\s+for\s+(.+?)\s*\{"
)
CAPABILITY_SETTER = re.compile(r"\.with_([a-z_]+)\(\s*(true|false)\s*\)")
CAPABILITY_CONST = re.compile(r"^\s*const CAPABILITIES\s*:")

# A path mention in prose. Only the last two segments matter, because that is
# what has to resolve: `crate::api::ModelError::EmptyData` and `ModelError::EmptyData`
# make the same claim.
PATH_MENTION = re.compile(r"\b([A-Z][A-Za-z0-9_]*)::([a-zA-Z_][A-Za-z0-9_]*)\b")
BOUND_MENTION = re.compile(r"`([A-Z][A-Za-z0-9]{0,2}):\s*([A-Za-z_][\w]*)`")
REPOSITORY_PATH_MENTION = re.compile(r"`([A-Za-z][A-Za-z0-9_-]*/[A-Za-z0-9_./*-]+)`")

GENERIC_PARAMETERS = re.compile(r"^[^<]*<([^>]*)>")

# The crate manifest, which is the authority the install snippet is checked
# against. Only the `[package]` table is read, because `[dependencies]` carries
# `version = "..."` keys of its own that say nothing about this crate.
MANIFEST = "Cargo.toml"
TABLE_HEADER = re.compile(r"^\s*\[\s*([^\]]+?)\s*\]\s*$")
MANIFEST_FIELD = re.compile(r'^\s*(name|version)\s*=\s*"([^"]*)"')
PLAIN_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# One comparator of a Cargo version requirement: an optional operator and one
# to three components, each a number or a `*`.
COMPARATOR = re.compile(
    r"^\s*(\^|~|=|>=|<=|>|<)?\s*(\d+|\*)(?:\.(\d+|\*))?(?:\.(\d+|\*))?\s*$"
)


def rust_sources(root: Path) -> list[Path]:
    return sorted((root / RUST_SOURCE).rglob("*.rs"))


def narrative_sources(root: Path) -> list[Path]:
    return sorted((root / NARRATIVE).rglob("*.md"))


def install_sources(root: Path) -> list[Path]:
    """Every page that may carry an install snippet.

    This is the prose the other rules read, plus the root `README.md`. The
    readme is outside `docs/` and outside rustdoc, so no rule here reads it —
    but it is the crates.io landing page and the most likely second home for an
    install snippet, which makes it exactly the place a stale requirement would
    survive unread.
    """
    sources = [*rust_sources(root), *narrative_sources(root)]
    readme = root / "README.md"
    if readme.exists():
        sources.append(readme)
    return sources


def documentation_lines(path: Path) -> Iterable[tuple[int, str]]:
    """Every prose line of a file, as `(line number, text)`.

    Rust contributes its doc comments only; a Markdown page is prose
    throughout. Both are checked, because the narrative site is the half of the
    documentation rustdoc never reads.
    """
    text = path.read_text()
    if path.suffix == ".md":
        yield from enumerate(text.splitlines(), 1)
        return
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("///"):
            yield number, stripped[3:].strip()
        elif stripped.startswith("//!"):
            yield number, stripped[3:].strip()


def block_start(lines: list[str], start: int) -> int | None:
    """Index of the line opening the block of the item declared at `start`.

    A `where` clause routinely pushes the brace several lines below the `impl`,
    so the opening line has to be found rather than assumed. A declaration that
    reaches `;` first — a unit or tuple struct — has no block at all.
    """
    for index in range(start, min(start + 20, len(lines))):
        if "{" in lines[index]:
            return index
        if lines[index].rstrip().endswith(";"):
            return None
    return None


def block_end(lines: list[str], start: int) -> int:
    """Index of the line closing the brace-delimited block opened at `start`."""
    depth = 0
    opened = False
    for index in range(start, len(lines)):
        depth += lines[index].count("{") - lines[index].count("}")
        opened = opened or "{" in lines[index]
        if opened and depth <= 0:
            return index
    return len(lines) - 1


def doc_block_above(lines: list[str], index: int) -> list[str]:
    """The contiguous `///` block documenting the item at `index`.

    Attribute lines between the prose and the item are skipped, because
    `#[derive(...)]` and `#[cfg(...)]` sit there routinely and do not end a doc
    comment.
    """
    cursor = index - 1
    while cursor >= 0 and lines[cursor].strip().startswith("#["):
        cursor -= 1
    block: list[str] = []
    while cursor >= 0 and lines[cursor].strip().startswith("///"):
        block.insert(0, lines[cursor].strip()[3:].strip())
        cursor -= 1
    return block


class RustIndex:
    """What the crate's own types actually declare.

    Built by line scanning rather than by parsing, which is sound here because
    the tree is `cargo fmt --check` clean. The index is deliberately
    conservative: a type it could not see the declaration of is absent, and an
    absent type is never reported against, so macro-generated estimators cost
    coverage rather than producing false findings.
    """

    def __init__(self, root: Path) -> None:
        self.kinds: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}
        self.implemented_traits: dict[str, set[str]] = {}
        self.generic_bounds: dict[str, dict[str, set[str]]] = {}
        for path in rust_sources(root):
            self._scan(path.read_text().splitlines())
        self._inherit_trait_members()

    def _member_set(self, name: str) -> set[str]:
        return self.members.setdefault(name, set())

    def _scan(self, lines: list[str]) -> None:
        index = 0
        pending_derives: set[str] = set()
        while index < len(lines):
            line = lines[index]
            derive = DERIVE.match(line)
            if derive:
                pending_derives.update(
                    token.strip() for token in derive.group(1).split(",")
                )
                index += 1
                continue
            declaration = DECLARATION.match(line)
            if declaration:
                kind, name = declaration.group(1), declaration.group(2)
                self.kinds.setdefault(name, kind)
                members = self._member_set(name)
                for derived in pending_derives:
                    members.update(DERIVE_MEMBERS.get(derived, ()))
                opening = block_start(lines, index)
                pending_derives = set()
                if opening is None:
                    index += 1
                    continue
                self._record_bounds(name, lines, index, opening)
                end = block_end(lines, opening)
                for body_line in lines[opening + 1 : end]:
                    self._record_member(kind, members, body_line)
                index = end + 1
                continue
            implementation = IMPLEMENTATION.match(line)
            opening = block_start(lines, index) if implementation else None
            if implementation and opening is not None:
                target = implementation.group("target")
                implemented = implementation.group("trait")
                if implemented:
                    self.implemented_traits.setdefault(target, set()).add(
                        implemented.rsplit("::", 1)[-1]
                    )
                self._record_bounds(target, lines, index, opening)
                end = block_end(lines, opening)
                members = self._member_set(target)
                for body_line in lines[opening + 1 : end]:
                    associated = ASSOCIATED_ITEM.match(body_line)
                    if associated:
                        members.add(associated.group(1))
                index = end + 1
                pending_derives = set()
                continue
            pending_derives = set()
            index += 1

    @staticmethod
    def _record_member(kind: str, members: set[str], body_line: str) -> None:
        associated = ASSOCIATED_ITEM.match(body_line)
        if associated:
            members.add(associated.group(1))
            return
        if kind == "enum":
            variant = VARIANT.match(body_line)
            if variant:
                members.add(variant.group(1))
                return
        field = FIELD.match(body_line)
        if field:
            members.add(field.group(1))

    def _record_bounds(
        self, target: str, lines: list[str], index: int, opening: int
    ) -> None:
        """Every trait bound applied to a generic parameter of `target`.

        Bounds reach a type from three places — its own generics, its `where`
        clause, and the `impl` blocks written for it — and a doc comment naming
        a bound is making a claim about their union, not about any one of them.
        Only the header is read, so a bound on some nested helper cannot be
        mistaken for a bound on the item itself.
        """
        bounds = self.generic_bounds.setdefault(target, {})
        header = "\n".join(lines[index : opening + 1])
        parameters = GENERIC_PARAMETERS.match(lines[index])
        sources = [parameters.group(1)] if parameters else []
        sources.extend(re.findall(r"where\s+(.*)", header, flags=re.DOTALL))
        for source in sources:
            for parameter, listed in re.findall(
                r"\b([A-Z][A-Za-z0-9]{0,2})\s*:\s*([^,>\n]+)", source
            ):
                traits = bounds.setdefault(parameter, set())
                traits.update(
                    token.strip().split("<")[0].rsplit("::", 1)[-1]
                    for token in listed.split("+")
                    if token.strip()
                )

    def _inherit_trait_members(self) -> None:
        """A type answers for the traits it implements.

        `Ridge::predict` is a real reference even though `predict` has a
        default body on the trait and no line inside `impl Regressor for Ridge`.
        """
        for target, traits in self.implemented_traits.items():
            members = self._member_set(target)
            for trait in traits:
                members.update(self.members.get(trait, set()))

    def resolves(self, type_name: str, member: str) -> bool:
        return member in self.members.get(type_name, set())

    def bounds_for(self, target: str, parameter: str) -> set[str]:
        return self.generic_bounds.get(target, {}).get(parameter, set())


def capability_declarations(root: Path) -> Iterable[tuple[Path, int, str, list[str], str]]:
    """Every `HasCapabilities` impl, with its prose and its declaration body.

    Prose is read from **both** positions Rust makes available, because both are
    in use and both render: above the `impl` line, where rustdoc attaches it to
    the impl block, and above the `const CAPABILITIES` line inside it, where
    rustdoc attaches it to the associated const. Reading only the first is what
    made this rule look like it covered the crate when it did not — 20 of 29
    declarations sat outside it, and half of those carried a written explanation
    the rule simply never looked at, including one saying "Nothing" above a
    declaration that turns probabilities on.
    """
    for path in rust_sources(root):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            match = CAPABILITY_IMPL.match(line)
            if not match:
                continue
            end = block_end(lines, index)
            block = list(doc_block_above(lines, index))
            for cursor in range(index + 1, end + 1):
                if CAPABILITY_CONST.match(lines[cursor]):
                    block.extend(doc_block_above(lines, cursor))
                    break
            body = "\n".join(lines[index + 1 : end + 1])
            yield path, index + 1, match.group(1), block, body


def capability_declarations_are_documented(root: Path) -> list[str]:
    """Every capability declaration carries prose, so the completeness rule can read it.

    This is the coverage floor under the rule below, and it exists because that
    rule can only check a declaration a human has written a sentence about. With
    nothing enforcing the sentence, 21 of the crate's 29 declarations were
    outside it while it reported clean — which is the same failure this codebase
    keeps producing: a check that runs and proves less than it appears to.

    What the sentence has to say is not enforced here, because it cannot be. The
    rule below enforces that it names every capability turned *on*; the value of
    the rest is in explaining why a capability is deliberately absent, which no
    regex can grade.
    """
    findings = [
        f"capability declaration carries no doc comment: "
        f"{path.relative_to(root)}:{line} declares for {target} and nothing "
        f"explains what it claims"
        for path, line, target, block, _ in capability_declarations(root)
        if not block
    ]
    if not any(True for _ in capability_declarations(root)):
        findings.append(
            "no capability declaration was found at all; the capability "
            "documentation rules can no longer prove anything"
        )
    return findings


def capability_documentation_matches_declaration(root: Path) -> list[str]:
    """A documented capability declaration must name every capability it turns on.

    `Capabilities` is the crate's machine-readable claim about what an estimator
    can do, and the sentence above it is the human-readable one. When the two
    disagree the human one is what a reader believes, and it is the one nothing
    checks — which is exactly how `DummyClassifier` came to read "Declares
    nothing" directly above `Capabilities::NONE.with_probability(true)`, and how
    two ensembles came to enumerate three of their four capabilities after the
    probability split added a fourth.

    Only the positive direction is enforced. A doc may say a capability is
    *absent* and explain why — several usefully do — so a mention with no
    corresponding `true` is not a finding; a `true` with no mention is. The
    declarations have to exist for the rule to mean anything, so their absence
    is itself a finding rather than a silently vacuous pass; that a declaration
    is documented at all is the separate rule above.
    """
    findings: list[str] = []
    documented = 0
    for path, line, target, block, body in capability_declarations(root):
        if not block:
            continue
        documented += 1
        prose = " ".join(block).lower()
        declared = {
            capability
            for capability, value in CAPABILITY_SETTER.findall(body)
            if value == "true"
        }
        findings.extend(
            f"capability documentation omits a declared capability: "
            f"{path.relative_to(root)}:{line} declares {capability!r} "
            f"for {target} and its doc comment never names it"
            for capability in sorted(declared)
            if not any(word in prose for word in CAPABILITY_WORDS[capability])
        )
    if not documented:
        findings.append(
            "no documented capability declaration was found; the "
            "capability-documentation rule can no longer prove anything"
        )
    return findings


def documented_paths_resolve(root: Path) -> list[str]:
    """A `Type::member` written in prose must name a member that type has.

    This is the rule that turns an unlinked mention into a checkable reference.
    Rustdoc already resolves intra-doc *links*, which is how a
    `CrossValidationError::Scoring` variant that never existed was eventually
    found — but only after a day of red `gate-full`, and only because someone
    had written it as a link. The same claim in plain backticks has no reader at
    all, and the narrative pages under `docs/` are outside rustdoc's reach
    entirely.

    The check is scoped to prefixes this crate declares, so `Vec::push` and
    `f32::NAN` are never candidates and a bad reference is the only thing that
    can be reported. The prose has to contain such references for the rule to
    mean anything, so their absence is itself a finding rather than a silently
    vacuous pass.
    """
    index = RustIndex(root)
    findings: list[str] = []
    checked = 0
    for path in (*rust_sources(root), *narrative_sources(root)):
        for number, line in documentation_lines(path):
            for type_name, member in PATH_MENTION.findall(line):
                if type_name not in index.kinds:
                    continue
                checked += 1
                if not index.resolves(type_name, member):
                    findings.append(
                        f"documentation names a member that does not exist: "
                        f"{path.relative_to(root)}:{number} refers to "
                        f"{type_name}::{member}, and {index.kinds[type_name]} "
                        f"{type_name} has no such member"
                    )
    if not checked:
        findings.append(
            "no documentation reference to a crate type's member was found; the "
            "path-resolution rule can no longer prove anything"
        )
    return findings


def documented_bounds_are_real_bounds(root: Path) -> list[str]:
    """A bound written in prose must be a bound the documented item carries.

    This is the shape that bit the calibration module twice: its wrapper is
    documented as "generic over `C: Classifier`" while every impl in the file
    is written for `C: ProbabilisticClassifier`. A reader who takes the sentence
    at face value copies the bound into their own generic code and then cannot
    compile it, which makes this the most expensive form of stale prose the
    crate has produced.

    A bound is looked for across the union of the item's own generics, its
    `where` clause, and the `impl` blocks written for it, because that union is
    what the sentence is really claiming. The crate has to document at least one
    bound for the rule to mean anything, so the absence of any documented bound
    is itself a finding rather than a silently vacuous pass.
    """
    findings: list[str] = []
    index = RustIndex(root)
    checked = 0
    for path in rust_sources(root):
        lines = path.read_text().splitlines()
        for line_index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("///") or stripped.startswith("//!"):
                continue
            documented_item = DECLARATION.match(line) or IMPLEMENTATION.match(line)
            if not documented_item:
                continue
            target = documented_item.group(2 if DECLARATION.match(line) else "target")
            for prose in doc_block_above(lines, line_index):
                for parameter, trait in BOUND_MENTION.findall(prose):
                    checked += 1
                    if trait not in index.bounds_for(target, parameter):
                        findings.append(
                            f"documentation states a bound the item does not carry: "
                            f"{path.relative_to(root)}:{line_index + 1} documents "
                            f"{target} as `{parameter}: {trait}`, and {parameter} is "
                            f"bounded by "
                            f"{sorted(index.bounds_for(target, parameter)) or 'nothing'}"
                        )
    if not checked:
        findings.append(
            "no documented generic bound was found; the bound-agreement rule can "
            "no longer prove anything"
        )
    return findings


def documented_repository_paths_exist(root: Path) -> list[str]:
    """A repository path named in prose must exist.

    Documentation that points at the evidence for its own claim is the house
    style, and the pointer is worth exactly as much as its being real. A
    renamed test file or a moved script turns every sentence citing it into a
    dangling claim that nothing else reads. Paths have to be cited for the rule
    to mean anything, so the absence of any citation is itself a finding rather
    than a silently vacuous pass.
    """
    findings: list[str] = []
    checked = 0
    for path in (*rust_sources(root), *narrative_sources(root)):
        for number, line in documentation_lines(path):
            for mention in REPOSITORY_PATH_MENTION.findall(line):
                if not mention.startswith(REPOSITORY_PATH_ROOTS):
                    continue
                checked += 1
                concrete = mention.split("*")[0].rstrip("/")
                if not (root / concrete).exists():
                    findings.append(
                        f"documentation names a repository path that does not exist: "
                        f"{path.relative_to(root)}:{number} cites {mention}"
                    )
    if not checked:
        findings.append(
            "no repository path citation was found; the path-existence rule can "
            "no longer prove anything"
        )
    return findings


def manifest_package(root: Path) -> tuple[str, tuple[int, int, int] | None, list[str]]:
    """The crate's own name and version, read from the `[package]` table.

    Returns the findings alongside them rather than raising, because a manifest
    this cannot read is a rule that has lost its input, and the house rule is
    that such a rule reports rather than passes.
    """
    path = root / MANIFEST
    if not path.exists():
        return "", None, [f"the crate manifest {MANIFEST} is missing; the "
                          "dependency-requirement rule can no longer prove anything"]
    name = ""
    version = ""
    table = ""
    for line in path.read_text().splitlines():
        header = TABLE_HEADER.match(line)
        if header:
            table = header.group(1)
            continue
        if table != "package":
            continue
        field = MANIFEST_FIELD.match(line)
        if not field:
            continue
        if field.group(1) == "name" and not name:
            name = field.group(2)
        elif field.group(1) == "version" and not version:
            version = field.group(2)
    if not name or not version:
        return "", None, [
            f"the crate manifest {MANIFEST} declares no [package] name and version "
            "pair; the dependency-requirement rule can no longer prove anything"
        ]
    parsed = PLAIN_VERSION.match(version)
    if not parsed:
        return name, None, [
            f"the crate manifest version {version!r} is not a plain "
            "major.minor.patch version, so no documented requirement can be "
            "evaluated against it"
        ]
    major, minor, patch = (int(part) for part in parsed.groups())
    return name, (major, minor, patch), []


def _comparator_bounds(
    operator: str, given: list[int], wildcard: bool
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, bool] | None:
    """`(lower, upper, lower_inclusive)` for one comparator, upper exclusive.

    `None` means a form this evaluator does not model, which is reported rather
    than assumed to pass — a requirement nobody can evaluate is not a
    requirement anybody has checked.
    """
    if wildcard:
        # `*`, `1.*`, `1.2.*`. A wildcard carries no operator in Cargo.
        if operator not in ("", "^"):
            return None
        if not given:
            return None, None, True
        if len(given) == 1:
            return (given[0], 0, 0), (given[0] + 1, 0, 0), True
        return (given[0], given[1], 0), (given[0], given[1] + 1, 0), True
    padded = (given + [0, 0, 0])[:3]
    lower = (padded[0], padded[1], padded[2])
    if operator in ("", "^"):
        # Caret, which is what a bare requirement means. The leftmost non-zero
        # component of the components actually written is the one that may not
        # move; below 1.0 that is the minor, which is the whole reason `"0.1"`
        # cannot reach a 0.2.0 crate. An omitted component is a wider bound,
        # not a zero: `^0.0` admits every 0.0.x, while `^0.0.0` admits only
        # 0.0.0 itself.
        if len(given) == 1:
            return lower, (padded[0] + 1, 0, 0), True
        if len(given) == 2:
            if padded[0] != 0:
                return lower, (padded[0] + 1, 0, 0), True
            return lower, (0, padded[1] + 1, 0), True
        if padded[0] != 0:
            return lower, (padded[0] + 1, 0, 0), True
        if padded[1] != 0:
            return lower, (0, padded[1] + 1, 0), True
        return lower, (0, 0, padded[2] + 1), True
    if operator == "~":
        if len(given) == 1:
            return lower, (padded[0] + 1, 0, 0), True
        return lower, (padded[0], padded[1] + 1, 0), True
    if operator == "=":
        if len(given) == 3:
            return lower, (padded[0], padded[1], padded[2] + 1), True
        if len(given) == 2:
            return lower, (padded[0], padded[1] + 1, 0), True
        return lower, (padded[0] + 1, 0, 0), True
    if operator == ">=":
        return lower, None, True
    if operator == "<":
        return None, lower, True
    # `>` and `<=` against a partial version have subtleties this rule does not
    # model, so they are reported as unevaluable unless fully spelled out.
    if len(given) < 3:
        return None
    if operator == ">":
        return lower, None, False
    return None, (lower[0], lower[1], lower[2] + 1), True


def requirement_admits(version: tuple[int, int, int], requirement: str) -> bool | None:
    """Whether `version` satisfies a Cargo version requirement.

    `None` means the requirement is not a form this evaluator models.
    """
    if not requirement.strip():
        return None
    for part in requirement.split(","):
        if part.strip() == "*":
            continue
        match = COMPARATOR.match(part)
        if not match:
            return None
        operator = match.group(1) or ""
        components = [group for group in match.groups()[1:] if group is not None]
        wildcard = "*" in components
        if wildcard and components[-1] != "*":
            return None
        numbers = [int(component) for component in components if component != "*"]
        bounds = _comparator_bounds(operator, numbers, wildcard)
        if bounds is None:
            return None
        lower, upper, inclusive = bounds
        if lower is not None and (version < lower or (not inclusive and version == lower)):
            return False
        if upper is not None and version >= upper:
            return False
    return True


def documented_requirements(root: Path, name: str) -> Iterable[tuple[Path, int, str]]:
    """Every documented requirement on this crate, as `(path, line, requirement)`.

    Three spellings are read, because all three are things a reader copies: a
    manifest line in a TOML block, a dependency table with a `version` key, and
    a `cargo add` command line.

    Exempt, and therefore not yielded at all:

    * a `path` or `git` dependency, which names no registry version and so can
      contradict no manifest — the packaged-consumer fixture is written that
      way deliberately;
    * `cargo add <crate>` with no version, which resolves to whatever is newest
      and is therefore correct by construction.

    A manifest line is required to be the *whole* line, because that is what
    distinguishes a snippet a reader copies from a sentence that quotes one.
    Prose about a requirement that is deliberately wrong — this page's own
    explanation of the defect, for one — has to be written inline, the same way
    round the other rules already require of prose about an absent entity.
    """
    escaped = re.escape(name)
    inline = re.compile(rf'^\s*{escaped}\s*=\s*"([^"]*)"\s*(?:#.*)?$')
    table = re.compile(rf"^\s*{escaped}\s*=\s*\{{(.*)\}}\s*(?:#.*)?$")
    add = re.compile(rf"\bcargo\s+add\s+{escaped}(?P<rest>\S*[^\n]*)")
    version_key = re.compile(r'\bversion\s*=\s*"([^"]*)"')
    source_key = re.compile(r"\b(path|git)\s*=")
    add_version = re.compile(r'(?:@|--vers(?:ion)?[= ]\s*)"?([^"\s]+)"?')
    for path in install_sources(root):
        for number, line in documentation_lines(path):
            simple = inline.match(line)
            if simple:
                yield path, number, simple.group(1)
                continue
            structured = table.match(line)
            if structured:
                body = structured.group(1)
                pinned = version_key.search(body)
                if pinned:
                    yield path, number, pinned.group(1)
                elif not source_key.search(body):
                    yield path, number, ""
                continue
            command = add.search(line)
            if command:
                pinned = add_version.search(command.group("rest"))
                if pinned:
                    yield path, number, pinned.group(1)


def documented_dependency_requirements_resolve(root: Path) -> list[str]:
    """A documented `crate = "<req>"` must be a requirement the manifest satisfies.

    The install snippet is the first thing a reader runs, and it is the one
    claim in the documentation whose being wrong produces no error. Below 1.0
    Cargo treats the minor as the breaking component, so `ferricml = "0.1"`
    against a 0.2.0 crate is `>=0.1.0, <0.2.0`: it resolves, quietly, to the
    last 0.1.x release, and the reader then works through a page describing an
    API their build does not have. That was live in `docs/guide/quickstart.md`,
    and the published 0.2.0 carried breaking changes, so the mismatch was not
    cosmetic.

    The requirement is *evaluated* the way Cargo evaluates it rather than
    string-matched against the current version, so `"0.2"`, `"0.2.0"`,
    `"^0.2"`, `"~0.2"`, `"0.2.*"` and `">=0.2, <0.3"` all pass against 0.2.0
    and only a genuinely unsatisfiable requirement is reported. A requirement
    that excludes the manifest in the *forward* direction is a finding too:
    `"0.2.1"` against a 0.2.0 manifest names a release that does not exist yet,
    so a reader following it resolves nothing at all.

    A form the evaluator does not model is reported rather than passed, and so
    is a manifest it cannot read. The exempt spellings — `path`, `git`, and a
    bare `cargo add` — are not counted towards the rule having proved anything,
    because a page whose only snippet is a path dependency has given this rule
    nothing to check.
    """
    name, version, findings = manifest_package(root)
    if version is None:
        return findings
    checked = 0
    for path, number, requirement in documented_requirements(root, name):
        checked += 1
        admits = requirement_admits(version, requirement)
        if admits is None:
            findings.append(
                f"documentation states a dependency requirement that cannot be "
                f"evaluated: {path.relative_to(root)}:{number} writes "
                f'{name} = "{requirement}", which is not a Cargo requirement form '
                f"this rule models"
            )
        elif not admits:
            findings.append(
                f"documentation states a dependency requirement the manifest "
                f"contradicts: {path.relative_to(root)}:{number} tells a reader to "
                f'depend on {name} = "{requirement}", which does not admit the '
                f"manifest's {'.'.join(str(part) for part in version)}"
            )
    if not checked:
        findings.append(
            "no documented dependency requirement on the crate itself was found; "
            "the dependency-requirement rule can no longer prove anything"
        )
    return findings


RULES: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("capability-declarations-documented", capability_declarations_are_documented),
    ("capability-documentation-complete", capability_documentation_matches_declaration),
    ("documented-paths-resolve", documented_paths_resolve),
    ("documented-bounds-are-real", documented_bounds_are_real_bounds),
    ("documented-repository-paths-exist", documented_repository_paths_exist),
    ("documented-dependency-requirement-resolves", documented_dependency_requirements_resolve),
)


def violations(root: Path = ROOT) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(root))
    return found


CLEAN_TREE: dict[str, str] = {
    "src/lib.rs": "//! crate\npub mod api;\npub mod linear_model;\n",
    "src/api/mod.rs": (
        "//! api\n"
        "mod capabilities;\n"
        "mod error;\n"
        "mod traits;\n"
    ),
    "src/api/capabilities.rs": (
        "//! capabilities\n"
        "/// What a fitted estimator can do.\n"
        "pub struct Capabilities {\n"
        "    sample_weights: bool,\n"
        "    artifact: bool,\n"
        "    probability: bool,\n"
        "}\n"
        "impl Capabilities {\n"
        "    /// Nothing declared.\n"
        "    pub const NONE: Capabilities = Capabilities {\n"
        "        sample_weights: false,\n"
        "        artifact: false,\n"
        "        probability: false,\n"
        "    };\n"
        "    /// Declares weighted fitting.\n"
        "    pub const fn with_sample_weights(self, _value: bool) -> Capabilities {\n"
        "        self\n"
        "    }\n"
        "    /// Declares probability production.\n"
        "    pub const fn with_probability(self, _value: bool) -> Capabilities {\n"
        "        self\n"
        "    }\n"
        "}\n"
        "/// Carries a capability declaration.\n"
        "pub trait HasCapabilities {\n"
        "    /// The declaration.\n"
        "    const CAPABILITIES: Capabilities;\n"
        "}\n"
    ),
    "src/api/error.rs": (
        "//! errors\n"
        "/// Why a fit or a prediction failed.\n"
        "pub enum ModelError {\n"
        "    /// No rows.\n"
        "    EmptyData,\n"
        "    /// Wrong width.\n"
        "    FeatureDimension,\n"
        "}\n"
    ),
    "src/api/traits.rs": (
        "//! traits\n"
        "/// Predicts labels. See [`ModelError::EmptyData`] for the empty case.\n"
        "pub trait Classifier {\n"
        "    /// Predicts one batch.\n"
        "    fn predict(&self);\n"
        "}\n"
        "/// Predicts probabilities.\n"
        "pub trait ProbabilisticClassifier: Classifier {\n"
        "    /// Predicts one probability matrix.\n"
        "    fn predict_proba(&self);\n"
        "}\n"
    ),
    "src/linear_model/mod.rs": "//! linear models\nmod ridge;\n",
    "src/linear_model/ridge.rs": (
        "//! ridge\n"
        "use crate::api::{Capabilities, HasCapabilities};\n"
        "/// A fitted penalized linear model.\n"
        "///\n"
        "/// The wrapper is generic over `C: Classifier`, and its behavior is\n"
        "/// frozen in `tests/reference_semantics.rs`.\n"
        "pub struct Ridge<C> {\n"
        "    coefficients: Vec<f32>,\n"
        "    inner: C,\n"
        "}\n"
        "impl<C: Classifier> Ridge<C> {\n"
        "    /// Fits the model.\n"
        "    pub fn fit() {}\n"
        "}\n"
        "/// Declares weighted fitting and genuine probabilities.\n"
        "impl<C> HasCapabilities for Ridge<C> {\n"
        "    const CAPABILITIES: Capabilities = Capabilities::NONE\n"
        "        .with_sample_weights(true)\n"
        "        .with_probability(true);\n"
        "}\n"
        "/// A fitted baseline.\n"
        "pub struct Baseline;\n"
        "impl HasCapabilities for Baseline {\n"
        "    /// Declares probabilities and nothing else: nothing is persisted.\n"
        "    const CAPABILITIES: Capabilities = Capabilities::NONE\n"
        "        .with_probability(true);\n"
        "}\n"
    ),
    "tests/reference_semantics.rs": "// frozen behavior\n",
    "docs/index.md": (
        "# Guide\n\n"
        "An empty batch surfaces `ModelError::EmptyData`.\n"
    ),
    # The manifest the install snippet is measured against, and a page carrying
    # the snippet in both spellings a reader copies.
    "Cargo.toml": (
        "[package]\n"
        'name = "ferricml"\n'
        'version = "0.2.0"\n'
        "\n"
        "[dependencies]\n"
        'nalgebra = "0.34"\n'
    ),
    "docs/guide/quickstart.md": (
        "# Install\n\n"
        "```toml\n"
        "[dependencies]\n"
        'ferricml = "0.2"\n'
        "```\n\n"
        "Or from the command line:\n\n"
        "```console\n"
        "cargo add ferricml@0.2\n"
        "```\n"
    ),
}

CLEAN_REQUIREMENT = 'ferricml = "0.2"'
CLEAN_ADD = "cargo add ferricml@0.2"


def write_clean_tree(root: Path) -> Path:
    """Write the smallest tree that satisfies every rule.

    It has to exercise every rule positively, not merely avoid violating them:
    a tree with no capability declaration, no path reference, no documented
    bound and no cited file would pass four vacuous rules.
    """
    for relative, text in CLEAN_TREE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def rewrite(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"synthetic mutation target missing in {path}: {old!r}"
    path.write_text(text.replace(old, new, 1))


def drop_files(root: Path, *relatives: str) -> None:
    for relative in relatives:
        (root / relative).unlink()


def blank_files(root: Path, *relatives: str) -> None:
    for relative in relatives:
        (root / relative).write_text("")


def requirement_spelling(spelling: str) -> Callable[[Path], None]:
    """A mutation rewriting the clean tree's install snippet to `spelling`."""
    return lambda root: rewrite(
        root / "docs" / "guide" / "quickstart.md",
        CLEAN_REQUIREMENT,
        f"ferricml = {spelling}",
    )


# Requirement spellings the rule must **accept** against a 0.2.0 manifest, each
# proven at the tree level so the regex, the evaluator, and the exemptions are
# all on the path being tested. A rule that rejects everything catches the
# defect too, which makes the accept side the half that says the rule is a rule
# rather than a tripwire. The exempt spellings are checked with the `cargo add`
# line left in place, because on its own an exempt snippet leaves the rule with
# nothing to check and must therefore report vacuity rather than pass.
ACCEPTED_REQUIREMENTS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("bare-minor", requirement_spelling('"0.2"')),
    ("bare-patch", requirement_spelling('"0.2.0"')),
    ("caret-minor", requirement_spelling('"^0.2"')),
    ("caret-patch", requirement_spelling('"^0.2.0"')),
    ("tilde-minor", requirement_spelling('"~0.2"')),
    ("wildcard-patch", requirement_spelling('"0.2.*"')),
    ("wildcard-any", requirement_spelling('"*"')),
    ("explicit-range", requirement_spelling('">=0.2, <0.3"')),
    ("dependency-table", requirement_spelling('{ version = "0.2", default-features = false }')),
    ("path-dependency", requirement_spelling('{ path = "../ferricml" }')),
    ("git-dependency", requirement_spelling('{ git = "https://example.invalid/ferricml.git" }')),
    (
        "unversioned-cargo-add",
        lambda root: rewrite(
            root / "docs" / "guide" / "quickstart.md",
            CLEAN_ADD,
            "cargo add ferricml",
        ),
    ),
)


SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "capability-declarations-documented",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "    /// Declares probabilities and nothing else: nothing is persisted.\n",
            "",
        ),
        "capability declaration carries no doc comment",
    ),
    (
        "capability-documentation-complete",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "/// Declares weighted fitting and genuine probabilities.",
            "/// Declares weighted fitting.",
        ),
        "capability documentation omits a declared capability",
    ),
    (
        # The same rule, against prose in the *other* legitimate position. A
        # doc comment on the associated const renders exactly as one on the
        # impl block does, and reading only the latter is what left two thirds
        # of the crate's declarations outside this rule.
        "capability-documentation-complete",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "    /// Declares probabilities and nothing else: nothing is persisted.",
            "    /// Declares nothing at all.",
        ),
        "capability documentation omits a declared capability",
    ),
    (
        "documented-paths-resolve",
        lambda root: rewrite(
            root / "src" / "api" / "traits.rs",
            "[`ModelError::EmptyData`]",
            "[`ModelError::NoRows`]",
        ),
        "documentation names a member that does not exist",
    ),
    (
        "documented-bounds-are-real",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "`C: Classifier`",
            "`C: ProbabilisticClassifier`",
        ),
        "documentation states a bound the item does not carry",
    ),
    (
        "documented-repository-paths-exist",
        lambda root: drop_files(root, "tests/reference_semantics.rs"),
        "documentation names a repository path that does not exist",
    ),
    (
        # The live defect: below 1.0 the minor is the breaking component, so
        # this resolves to the newest 0.1.x and never reaches the 0.2.0 crate.
        "documented-dependency-requirement-resolves",
        requirement_spelling('"0.1"'),
        "documentation states a dependency requirement the manifest contradicts",
    ),
    (
        # The forward direction, which is a finding for the same reason: a
        # requirement naming a release that does not exist resolves to nothing.
        "documented-dependency-requirement-resolves",
        requirement_spelling('"0.2.1"'),
        "documentation states a dependency requirement the manifest contradicts",
    ),
    (
        # A form the evaluator does not model is reported, not waved through.
        "documented-dependency-requirement-resolves",
        requirement_spelling('"latest"'),
        "documentation states a dependency requirement that cannot be evaluated",
    ),
)

# Removing a rule's input must be a finding too. Every one of these mutations
# leaves a tree that violates nothing — and a rule that reported a clean pass
# against it would be a check that had stopped looking.
SYNTHETIC_VACUITIES: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "capability-declarations-documented",
        lambda root: (
            rewrite(
                root / "src" / "linear_model" / "ridge.rs",
                "impl<C> HasCapabilities for Ridge<C> {",
                "impl<C> Unrelated for Ridge<C> {",
            ),
            rewrite(
                root / "src" / "linear_model" / "ridge.rs",
                "impl HasCapabilities for Baseline {",
                "impl Unrelated for Baseline {",
            ),
        )
        and None,
        "no capability declaration was found at all",
    ),
    (
        "capability-documentation-complete",
        lambda root: (
            rewrite(
                root / "src" / "linear_model" / "ridge.rs",
                "/// Declares weighted fitting and genuine probabilities.\n",
                "",
            ),
            rewrite(
                root / "src" / "linear_model" / "ridge.rs",
                "    /// Declares probabilities and nothing else: nothing is persisted.\n",
                "",
            ),
        )
        and None,
        "no documented capability declaration was found",
    ),
    (
        "documented-paths-resolve",
        lambda root: (
            rewrite(
                root / "src" / "api" / "traits.rs",
                " See [`ModelError::EmptyData`] for the empty case.",
                "",
            ),
            rewrite(
                root / "docs" / "index.md",
                "An empty batch surfaces `ModelError::EmptyData`.",
                "An empty batch is refused.",
            ),
        )
        and None,
        "no documentation reference to a crate type's member was found",
    ),
    (
        "documented-bounds-are-real",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "The wrapper is generic over `C: Classifier`, and its",
            "The wrapper's",
        ),
        "no documented generic bound was found",
    ),
    (
        "documented-repository-paths-exist",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "frozen in `tests/reference_semantics.rs`",
            "frozen",
        ),
        "no repository path citation was found",
    ),
    (
        # The page still exists and still reads as an install page; it simply
        # no longer states a requirement. Nothing is violated, and a pass here
        # would mean the rule had stopped looking.
        "documented-dependency-requirement-resolves",
        lambda root: (
            rewrite(
                root / "docs" / "guide" / "quickstart.md",
                CLEAN_REQUIREMENT,
                "# add the crate",
            ),
            rewrite(
                root / "docs" / "guide" / "quickstart.md",
                CLEAN_ADD,
                "cargo build",
            ),
        )
        and None,
        "no documented dependency requirement on the crate itself was found",
    ),
    (
        # The page the rule reads is empty.
        "documented-dependency-requirement-resolves",
        lambda root: blank_files(root, "docs/guide/quickstart.md"),
        "no documented dependency requirement on the crate itself was found",
    ),
    (
        # The manifest the rule measures against is gone, so there is no
        # version any requirement could be compared with.
        "documented-dependency-requirement-resolves",
        lambda root: drop_files(root, "Cargo.toml"),
        f"the crate manifest {MANIFEST} is missing",
    ),
    (
        # The manifest is present but says nothing the rule can use.
        "documented-dependency-requirement-resolves",
        lambda root: blank_files(root, "Cargo.toml"),
        "declares no [package] name and version pair",
    ),
)

# The two documentation defects this checker was built from, reconstructed as
# inputs. A detector that cannot catch the defects that motivated it is not
# finished, so they are asserted here rather than described in a commit message.
#
# The other two known instances are deliberately out of reach and are recorded
# in `docs/api-and-growth.md`: both were completeness claims that named their
# own evidence correctly and were wrong about what that evidence established.
HISTORICAL_REGRESSIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        # 2026-07-25, the probability trait split: the declaration gained
        # `.with_probability(true)` and the sentence above it was left saying
        # the estimator declared nothing at all.
        "capability-documentation-complete",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "/// Declares weighted fitting and genuine probabilities.",
            "/// Declares nothing: a baseline is refitted rather than persisted,\n"
            "/// and has no weighted entry point.",
        ),
        "capability documentation omits a declared capability",
    ),
    (
        # 2026-07-25, repaired in 31fdb17: a doc comment named a
        # `CrossValidationError::Scoring` variant that has never existed.
        "documented-paths-resolve",
        lambda root: rewrite(
            root / "src" / "api" / "traits.rs",
            "See [`ModelError::EmptyData`] for the empty case.",
            "A probability metric returns [`ModelError::Scoring`] rather than a "
            "substituted value.",
        ),
        "documentation names a member that does not exist",
    ),
    (
        # 2026-07-27: the quickstart's install snippet still said
        # `ferricml = "0.1"` after the 0.2.0 release. Cargo resolved it to
        # 0.1.2 without a warning, so a reader worked through a page describing
        # an API their build did not contain, and the release it silently
        # pinned away from was a breaking one.
        "documented-dependency-requirement-resolves",
        requirement_spelling('"0.1"'),
        "documentation states a dependency requirement the manifest contradicts",
    ),
)


def assert_accepts(
    workspace: Path,
    label: str,
    cases: tuple[tuple[str, Callable[[Path], None]], ...],
) -> None:
    for name, mutate in cases:
        tree = write_clean_tree(workspace / f"{label}-{name}")
        mutate(tree)
        found = violations(tree)
        assert found == [], f"{label} case {name} was rejected; reported {found}"


def assert_fires(
    workspace: Path,
    label: str,
    cases: tuple[tuple[str, Callable[[Path], None], str], ...],
) -> None:
    for name, mutate, expected in cases:
        tree = write_clean_tree(workspace / f"{label}-{name}")
        mutate(tree)
        found = violations(tree)
        assert any(expected in item for item in found), (
            f"{label} case {name} did not fire; reported {found}"
        )


def self_test() -> None:
    live = violations()
    assert live == [], f"live tree violates its own documentation rules: {live}"

    declared = {name for name, _ in RULES}
    for label, cases in (
        ("violation", SYNTHETIC_VIOLATIONS),
        ("vacuity", SYNTHETIC_VACUITIES),
    ):
        covered = {name for name, _, _ in cases}
        assert covered == declared, (
            f"every rule needs a {label} case: missing={sorted(declared - covered)}, "
            f"stale={sorted(covered - declared)}"
        )

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)
        clean = write_clean_tree(base / "clean")
        found = violations(clean)
        assert found == [], f"synthetic clean tree reported violations: {found}"

        assert_accepts(base, "accepted", ACCEPTED_REQUIREMENTS)
        assert_fires(base, "violation", SYNTHETIC_VIOLATIONS)
        assert_fires(base, "vacuity", SYNTHETIC_VACUITIES)
        assert_fires(base, "historical", HISTORICAL_REGRESSIONS)


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "documentation truth verifier self-test passed "
            f"({len(RULES)} rules, each proven to fire against a synthetic "
            "violation and against the loss of its own input; "
            f"{len(ACCEPTED_REQUIREMENTS)} dependency requirement spellings "
            "proven to be accepted rather than merely unreported; "
            f"{len(HISTORICAL_REGRESSIONS)} historical defects reconstructed)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_documentation_truth.py [--self-test]", file=sys.stderr)
        return 2
    found = violations()
    if found:
        print("documentation truth check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("documentation truth: prose agrees with the code it documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
