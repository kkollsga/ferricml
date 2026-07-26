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

# A path mention in prose. Only the last two segments matter, because that is
# what has to resolve: `crate::api::ModelError::EmptyData` and `ModelError::EmptyData`
# make the same claim.
PATH_MENTION = re.compile(r"\b([A-Z][A-Za-z0-9_]*)::([a-zA-Z_][A-Za-z0-9_]*)\b")
BOUND_MENTION = re.compile(r"`([A-Z][A-Za-z0-9]{0,2}):\s*([A-Za-z_][\w]*)`")
REPOSITORY_PATH_MENTION = re.compile(r"`([A-Za-z][A-Za-z0-9_-]*/[A-Za-z0-9_./*-]+)`")

GENERIC_PARAMETERS = re.compile(r"^[^<]*<([^>]*)>")


def rust_sources(root: Path) -> list[Path]:
    return sorted((root / RUST_SOURCE).rglob("*.rs"))


def narrative_sources(root: Path) -> list[Path]:
    return sorted((root / NARRATIVE).rglob("*.md"))


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
    is itself a finding rather than a silently vacuous pass.
    """
    findings: list[str] = []
    documented = 0
    for path in rust_sources(root):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            match = CAPABILITY_IMPL.match(line)
            if not match:
                continue
            block = doc_block_above(lines, index)
            if not block:
                continue
            documented += 1
            prose = " ".join(block).lower()
            body = "\n".join(lines[index + 1 : block_end(lines, index) + 1])
            declared = {
                capability
                for capability, value in CAPABILITY_SETTER.findall(body)
                if value == "true"
            }
            findings.extend(
                f"capability documentation omits a declared capability: "
                f"{path.relative_to(root)}:{index + 1} declares {capability!r} "
                f"for {match.group(1)} and its doc comment never names it"
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


RULES: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("capability-documentation-complete", capability_documentation_matches_declaration),
    ("documented-paths-resolve", documented_paths_resolve),
    ("documented-bounds-are-real", documented_bounds_are_real_bounds),
    ("documented-repository-paths-exist", documented_repository_paths_exist),
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
    ),
    "tests/reference_semantics.rs": "// frozen behavior\n",
    "docs/index.md": (
        "# Guide\n\n"
        "An empty batch surfaces `ModelError::EmptyData`.\n"
    ),
}


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


SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
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
)

# Removing a rule's input must be a finding too. Every one of these mutations
# leaves a tree that violates nothing — and a rule that reported a clean pass
# against it would be a check that had stopped looking.
SYNTHETIC_VACUITIES: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "capability-documentation-complete",
        lambda root: rewrite(
            root / "src" / "linear_model" / "ridge.rs",
            "/// Declares weighted fitting and genuine probabilities.\n",
            "",
        ),
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
)


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
