#!/usr/bin/env python3
"""Enforce that an `X_into` method and its allocating `X` partner are a pair.

FerricML's naming contract — restated in `docs/api-and-growth.md:5-7` and in
CLAUDE.md — is that allocating convenience methods delegate to caller-owned
`_into` primitives, and that a single-row entry point carries `_one`. Nothing
enforced the *shape* half of that until this file, and the gap produced a defect
that then propagated: `predict_positive_proba` took one row while
`predict_positive_proba_into` took a matrix, so the caller-owned batch method
had no allocating partner and a single-row method held the batch name. It was
copied verbatim into a new estimator family a sprint after it was introduced.

The input is the frozen public-API baselines rather than `src/`, because the
baselines are `cargo-public-api`'s normalized rendering of the *whole* public
surface: macro-generated estimator facades, trait impls and monomorphised
compositions all appear there in one spelling, and a regex over Rust source
would see none of them the same way. The cost is that a surface change reaches
this checker only after `make api-refresh`, which the same commit owes anyway.

Each rule is named, and `--self-test` proves that every one of them fires
against a synthetic violation *and* reports a finding when its input disappears,
because a rule that quietly stopped matching would otherwise pass both the check
and its own self-test. The four defects this was written for are reconstructed
from the pre-fix baseline rows in `HISTORICAL_REGRESSIONS`.

What this deliberately does **not** check is recorded in
`docs/api-and-growth.md` under "What the accessor-pairing checker does not
check", so the blind spots are written down rather than discovered later.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rust_api_profiles import frozen_profiles, load_manifest  # noqa: E402


# Caller-owned `_into` methods whose allocating partner is genuinely absent
# today, with the reason. This list is closed in **both** directions: an entry
# that no longer describes the baseline is reported as stale, so a gap cannot be
# fixed and left recorded, and a new one cannot be added without landing here
# where a reviewer sees it.
EXPECTED_UNPAIRED: dict[str, str] = {
    "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, "
    "ferricml::linear_model::LinearRegression>::predict_into": (
        "2026-07-26: the fitted-pipeline forms carry a transform workspace as "
        "well as an output buffer, so their allocating partner is a design "
        "question about pipeline shape rather than a missing forwarder; "
        "tracked with the pipeline arity work, not with the accessor sweep"
    ),
    "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, "
    "ferricml::linear_model::Ridge>::predict_into": (
        "2026-07-26: as above, the same method on a second composition"
    ),
    "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, "
    "ferricml::linear_model::LogisticRegression>::predict_into": (
        "2026-07-26: as above, the same method on a third composition"
    ),
    "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, "
    "ferricml::linear_model::LogisticRegression>::predict_proba_into": (
        "2026-07-26: as above, the probability form of the same shape"
    ),
    "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, "
    "ferricml::linear_model::LogisticRegression>::predict_class_proba_into": (
        "2026-07-26: as above, the single-column probability form"
    ),
    "ferricml::pipeline::Pipeline<ferricml::preprocessing::StandardScaler, "
    "ferricml::linear_model::LogisticRegression>::decision_function_into": (
        "2026-07-26: as above, the raw-score form"
    ),
}

# Below this many observed `_into`/allocating pairs the baseline is not being
# read: the parse broke, the capture changed spelling, or the file is a stub.
# Reporting that is the difference between a passing check and a check that
# passed by not looking. The live surface carries well over a hundred.
MINIMUM_OBSERVED_PAIRS = 40

INTO = "_into"


def _is_output_argument(argument: str) -> bool:
    """Whether an argument is a caller-owned buffer the callee writes into.

    `&mut [f32]` and `&'workspace mut [f32]` are both spellings the capture
    produces. A `&mut` receiver is not one of these: `&mut self` is excluded by
    name, because a method mutating its own model is not an output parameter.
    """
    if argument in ("&mut self", "&self", "self"):
        return False
    text = argument.strip()
    if not text.startswith("&"):
        return False
    rest = text[1:].lstrip()
    if rest.startswith("'"):
        rest = rest.split(None, 1)[1] if " " in rest else ""
    return rest.startswith("mut ")


def _split_arguments(text: str) -> list[str]:
    """Top-level comma-separated arguments of a rendered signature.

    Depth is tracked across `<>`, `()` and `[]` because argument types nest all
    three — `impl FnOnce(&MatrixView<'_>) -> Result<R, ModelError>` is one
    argument. The `>` of an arrow is skipped so a return type inside a closure
    bound does not unbalance the angle count.
    """
    arguments: list[str] = []
    depth = 0
    current = ""
    previous = ""
    for character in text:
        if character in "<([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == ">" and previous != "-":
            depth -= 1
        if character == "," and depth == 0:
            arguments.append(current.strip())
            current = ""
        else:
            current += character
        previous = character
    if current.strip():
        arguments.append(current.strip())
    return arguments


def _balanced_suffix_start(text: str) -> int | None:
    """Index of the `<` opening a balanced generic group that ends `text`."""
    if not text.endswith(">") or text.endswith("->"):
        return None
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        character = text[index]
        if character == ">" and not text[index - 1 : index] == "-":
            depth += 1
        elif character == "<":
            depth -= 1
            if depth == 0:
                return index
    return None


def _strip_generic_parameters(text: str) -> str:
    """`impl<A, B> Rest` and `Type<A>::method<'a>` shorn of their parameters."""
    start = _balanced_suffix_start(text)
    return text[:start] if start is not None else text


def _impl_scope(line: str) -> str | None:
    """`inherent` or `trait-impl` for an `impl …` line, or `None`."""
    if not line.startswith("impl"):
        return None
    body = line[len("impl") :]
    if body.startswith("<"):
        depth = 0
        for index, character in enumerate(body):
            if character == "<":
                depth += 1
            elif character == ">":
                depth -= 1
                if depth == 0:
                    body = body[index + 1 :]
                    break
    body = body.strip()
    if " where " in body:
        body = body.split(" where ", 1)[0].strip()
    return "trait-impl" if " for " in body else "inherent"


def _method(line: str) -> tuple[str, str, list[str]] | None:
    """`(owner, method, arguments)` for a rendered `pub fn` row, or `None`."""
    for prefix in ("pub const unsafe fn ", "pub unsafe fn ", "pub const fn ", "pub fn "):
        if line.startswith(prefix):
            signature = line[len(prefix) :]
            break
    else:
        return None

    depth = 0
    open_index = None
    previous = ""
    for index, character in enumerate(signature):
        if character == "<":
            depth += 1
        elif character == ">" and previous != "-":
            depth -= 1
        elif character == "(" and depth == 0:
            open_index = index
            break
        previous = character
    if open_index is None:
        return None

    depth = 0
    close_index = None
    for index in range(open_index, len(signature)):
        if signature[index] == "(":
            depth += 1
        elif signature[index] == ")":
            depth -= 1
            if depth == 0:
                close_index = index
                break
    if close_index is None:
        return None

    head = _strip_generic_parameters(signature[:open_index])
    if "::" not in head:
        return None
    owner, _, method = head.rpartition("::")
    if not owner or not method:
        return None
    return owner, method, _split_arguments(signature[open_index + 1 : close_index])


class Surface:
    """Every public method the frozen baselines render, by declaring scope.

    A method is keyed by `(scope, owner, name)` so an inherent forwarder and the
    trait implementation behind it stay distinguishable — the difference is
    exactly what finding 9 was about. Free functions are keyed under their
    module and marked `free`, because `permutation_importance_regressor` and its
    `_into` form are a pair on the same terms as a method is.
    """

    def __init__(self, profiles: Iterable[tuple[str, str]]) -> None:
        self.arguments: dict[tuple[str, str, str], list[str]] = {}
        self.scopes_by_owner: dict[str, dict[str, set[str]]] = {}
        for _, text in profiles:
            self._scan(text)

    def _scan(self, text: str) -> None:
        scope = "free"
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("pub trait "):
                scope = "trait-decl"
                continue
            impl = _impl_scope(line)
            if impl is not None:
                scope = impl
                continue
            if line.startswith(("pub mod ", "pub struct ", "pub enum ", "pub union ")):
                scope = "free"
                continue
            parsed = _method(line)
            if parsed is None:
                continue
            owner, name, arguments = parsed
            # A lowercase final owner segment is a module path, so the row is a
            # free function rather than a method, whatever block it follows.
            final = owner.rpartition("::")[2]
            where = "free" if final[:1].islower() else scope
            self.arguments.setdefault((where, owner, name), arguments)
            self.scopes_by_owner.setdefault(owner, {}).setdefault(name, set()).add(where)

    def methods(self) -> Iterable[tuple[str, str, str, list[str]]]:
        for (scope, owner, name), arguments in sorted(self.arguments.items()):
            yield scope, owner, name, arguments

    def has(self, scope: str, owner: str, name: str) -> bool:
        return (scope, owner, name) in self.arguments

    def scopes(self, owner: str, name: str) -> set[str]:
        return self.scopes_by_owner.get(owner, {}).get(name, set())


def into_has_an_allocating_partner(
    surface: Surface, expected_unpaired: dict[str, str]
) -> list[str]:
    """Every inherent `X_into` needs an inherent `X` on the same type.

    Restricted to inherent methods on purpose. A *trait declaration* may carry
    `X_into` alone — `pipeline::TransformerStack::transform_into` does, and
    correctly, because a stack is an internal composition contract reached only
    through `StagedPipeline`, which does provide `transform`. A *trait impl* row
    lists only the methods the impl defines, so a provided allocating default is
    absent from the baseline while being perfectly callable.
    """
    findings: list[str] = []
    observed: set[str] = set()
    for scope, owner, name, _ in surface.methods():
        if scope != "inherent" or not name.endswith(INTO):
            continue
        partner = name[: -len(INTO)]
        if surface.has("inherent", owner, partner):
            continue
        key = f"{owner}::{name}"
        observed.add(key)
        if key in expected_unpaired:
            continue
        findings.append(
            f"caller-owned `{key}` has no allocating `{partner}` partner; add the "
            f"allocating form or record the gap in EXPECTED_UNPAIRED with a reason"
        )
    findings.extend(
        f"recorded unpaired `{key}` no longer describes the surface; remove it "
        f"from EXPECTED_UNPAIRED"
        for key in sorted(set(expected_unpaired) - observed)
    )
    return findings


def into_and_allocating_agree_in_shape(
    surface: Surface, _expected_unpaired: dict[str, str]
) -> list[str]:
    """`X_into` must be `X` plus caller-owned output buffers, and nothing else.

    Comparing whole argument lists rather than first arguments is the whole
    point: the audit's own sweep compared only the first, which is why
    `PairwiseLinearRanker::compare` — `&MatrixView` first on both halves, one
    `PairIndex` against a slice of them second — was recorded as correctly
    paired when it was the same defect.
    """
    findings: list[str] = []
    for scope, owner, name, into_arguments in surface.methods():
        if not name.endswith(INTO):
            continue
        partner = name[: -len(INTO)]
        allocating = surface.arguments.get((scope, owner, partner))
        if allocating is None:
            continue
        shared = into_arguments[: len(allocating)]
        extra = into_arguments[len(allocating) :]
        if shared == allocating and all(_is_output_argument(item) for item in extra) and extra:
            continue
        findings.append(
            f"`{owner}::{name}` is not `{partner}` plus caller-owned output "
            f"buffers: `{partner}` takes ({', '.join(allocating)}) and "
            f"`{name}` takes ({', '.join(into_arguments)})"
        )
    return findings


def inherent_forwarders_come_in_pairs(
    surface: Surface, _expected_unpaired: dict[str, str]
) -> list[str]:
    """A type forwarding `X` inherently must forward `X_into` inherently too.

    Reaching the allocating form without a trait import while the
    allocation-free form needs one inverts the crate's stated preference on hot
    paths. The rule only applies where the caller-owned form exists at all, so
    an estimator that simply has no `_into` is untouched: that is a gap, not an
    asymmetry, and this file does not have an opinion about gaps.
    """
    findings: list[str] = []
    for scope, owner, name, _ in surface.methods():
        if scope != "inherent" or name.endswith(INTO) or name.endswith("_one"):
            continue
        into = name + INTO
        if surface.has("inherent", owner, into):
            continue
        elsewhere = surface.scopes(owner, into)
        if elsewhere:
            findings.append(
                f"`{owner}` forwards `{name}` inherently but reaches `{into}` only "
                f"through {sorted(elsewhere)}; add the inherent forwarder so the "
                f"allocation-free path needs no more imports than the allocating one"
            )
    return findings


def pairs_are_actually_observed(
    surface: Surface, _expected_unpaired: dict[str, str]
) -> list[str]:
    """The input has to contain pairs, or the check passed by not looking."""
    pairs = sum(
        1
        for scope, owner, name, _ in surface.methods()
        if name.endswith(INTO) and surface.has(scope, owner, name[: -len(INTO)])
    )
    if pairs < MINIMUM_OBSERVED_PAIRS:
        return [
            f"only {pairs} allocating/`_into` pairs were found in the API "
            f"baselines, below the floor of {MINIMUM_OBSERVED_PAIRS}; the input "
            f"is missing, truncated, or no longer parses"
        ]
    return []


RULES: tuple[tuple[str, Callable[[Surface, dict[str, str]], list[str]]], ...] = (
    ("into-has-an-allocating-partner", into_has_an_allocating_partner),
    ("into-and-allocating-agree-in-shape", into_and_allocating_agree_in_shape),
    ("inherent-forwarders-come-in-pairs", inherent_forwarders_come_in_pairs),
    ("pairs-are-actually-observed", pairs_are_actually_observed),
)


def violations(
    profiles: Iterable[tuple[str, str]],
    expected_unpaired: dict[str, str] | None = None,
) -> list[str]:
    surface = Surface(profiles)
    recorded = EXPECTED_UNPAIRED if expected_unpaired is None else expected_unpaired
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(surface, recorded))
    return found


# --------------------------------------------------------------- self-test

CLEAN_BASELINE = """pub mod ferricml
pub mod ferricml::estimators
pub struct ferricml::estimators::Model
impl ferricml::estimators::Model
pub fn ferricml::estimators::Model::predict(&self, &ferricml::data::MatrixView<'_>) -> \
core::result::Result<alloc::vec::Vec<f32>, ferricml::api::ModelError>
pub fn ferricml::estimators::Model::predict_into(&self, &ferricml::data::MatrixView<'_>, \
&mut [f32]) -> core::result::Result<(), ferricml::api::ModelError>
pub fn ferricml::estimators::Model::predict_one(&self, &[f32]) -> \
core::result::Result<f32, ferricml::api::ModelError>
pub fn ferricml::estimators::Model::predict_class_proba(&self, \
&ferricml::data::MatrixView<'_>, u8) -> \
core::result::Result<alloc::vec::Vec<f32>, ferricml::api::ModelError>
pub fn ferricml::estimators::Model::predict_class_proba_into(&self, \
&ferricml::data::MatrixView<'_>, u8, &mut [f32]) -> \
core::result::Result<(), ferricml::api::ModelError>
pub fn ferricml::estimators::Model::transform(&self, &ferricml::data::MatrixView<'_>) -> \
core::result::Result<ferricml::data::DenseMatrix, ferricml::api::ModelError>
pub fn ferricml::estimators::Model::transform_into<'output>(&self, \
&ferricml::data::MatrixView<'_>, &'output mut [f32]) -> \
core::result::Result<ferricml::data::MatrixView<'output>, ferricml::api::ModelError>
impl ferricml::api::Estimator for ferricml::estimators::Model
pub fn ferricml::estimators::Model::n_features_in(&self) -> usize
pub trait ferricml::api::Stack
pub fn ferricml::api::Stack::transform_into<'workspace>(&self, \
&ferricml::data::MatrixView<'_>, &'workspace mut [f32]) -> \
core::result::Result<ferricml::data::MatrixView<'workspace>, ferricml::api::ModelError>
"""


def _padded(baseline: str) -> str:
    """The clean baseline plus enough honest pairs to clear the vacuity floor.

    Every rule other than the floor has to be exercised on input the floor
    accepts, otherwise a synthetic case would report two findings and the one
    being proven would be indistinguishable from the padding.
    """
    filler = []
    for index in range(MINIMUM_OBSERVED_PAIRS):
        owner = f"ferricml::estimators::Filler{index}"
        filler.append(f"impl {owner}")
        filler.append(
            f"pub fn {owner}::predict(&self, &ferricml::data::MatrixView<'_>) -> "
            f"core::result::Result<alloc::vec::Vec<f32>, ferricml::api::ModelError>"
        )
        filler.append(
            f"pub fn {owner}::predict_into(&self, &ferricml::data::MatrixView<'_>, "
            f"&mut [f32]) -> core::result::Result<(), ferricml::api::ModelError>"
        )
    return baseline + "\n".join(filler) + "\n"


def _replace(baseline: str, old: str, new: str) -> str:
    assert old in baseline, f"synthetic edit no longer applies: {old!r}"
    return baseline.replace(old, new, 1)


SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[str], str], str], ...] = (
    (
        "into-has-an-allocating-partner",
        lambda baseline: _replace(
            baseline,
            "pub fn ferricml::estimators::Model::transform(&self, "
            "&ferricml::data::MatrixView<'_>) -> "
            "core::result::Result<ferricml::data::DenseMatrix, ferricml::api::ModelError>\n",
            "",
        ),
        "has no allocating `transform` partner",
    ),
    (
        "into-and-allocating-agree-in-shape",
        lambda baseline: _replace(
            baseline,
            "pub fn ferricml::estimators::Model::predict(&self, "
            "&ferricml::data::MatrixView<'_>)",
            "pub fn ferricml::estimators::Model::predict(&self, &[f32])",
        ),
        "is not `predict` plus caller-owned output buffers",
    ),
    (
        "inherent-forwarders-come-in-pairs",
        lambda baseline: _replace(
            baseline,
            "pub fn ferricml::estimators::Model::predict_class_proba_into(&self, "
            "&ferricml::data::MatrixView<'_>, u8, &mut [f32]) -> "
            "core::result::Result<(), ferricml::api::ModelError>\n",
            "impl ferricml::api::ProbabilisticClassifier for ferricml::estimators::Model\n"
            "pub fn ferricml::estimators::Model::predict_class_proba_into(&self, "
            "&ferricml::data::MatrixView<'_>, u8, &mut [f32]) -> "
            "core::result::Result<(), ferricml::api::ModelError>\n"
            "impl ferricml::estimators::Model\n",
        ),
        "only through ['trait-impl']",
    ),
    (
        "pairs-are-actually-observed",
        lambda baseline: baseline,
        "below the floor",
    ),
)

# All four rules read one input — the frozen API baselines — so they cannot go
# vacuous independently, and pretending otherwise with four identical cases
# would be theatre. `pairs-are-actually-observed` is the shared non-vacuity
# guard, and these are the ways the input can go away underneath it: deleted,
# emptied of methods, or renamed out from under the parser by a capture-tool
# change. Each must be reported rather than passing quietly. The fourth way —
# the baseline file itself missing — is caught before any rule runs, by
# `frozen_profiles`, and is asserted separately in `self_test`.
INPUT_LOSS_CASES: tuple[tuple[str, Callable[[str], str], str], ...] = (
    ("empty-baseline", lambda _baseline: "", "below the floor"),
    (
        "no-method-rows",
        lambda baseline: "\n".join(
            line for line in baseline.splitlines() if not line.startswith("pub fn ")
        )
        + "\n",
        "below the floor",
    ),
    (
        "rows-no-longer-parse",
        lambda baseline: baseline.replace("pub fn ", "pub function "),
        "below the floor",
    ),
)

# The four real defects, reconstructed from the rows the baseline carried before
# they were fixed. A rule that stops catching the thing it was written for is
# the failure this crate keeps rediscovering.
HISTORICAL_REGRESSIONS: tuple[tuple[str, str, str], ...] = (
    (
        # Audit finding 1: the single-row method held the batch name, so the
        # caller-owned batch form had no allocating partner at all.
        "predict_positive_proba",
        "impl ferricml::tree::DecisionTreeClassifier\n"
        "pub fn ferricml::tree::DecisionTreeClassifier::predict_positive_proba(&self, &[f32]) "
        "-> core::result::Result<f32, ferricml::api::ModelError>\n"
        "pub fn ferricml::tree::DecisionTreeClassifier::predict_positive_proba_into(&self, "
        "&ferricml::data::MatrixView<'_>, &mut [f32]) -> "
        "core::result::Result<(), ferricml::api::ModelError>\n",
        "is not `predict_positive_proba` plus caller-owned output buffers",
    ),
    (
        # The same defect on the ranker, which the audit's first-argument sweep
        # recorded as correctly paired.
        "compare",
        "impl ferricml::ranking::PairwiseLinearRanker\n"
        "pub fn ferricml::ranking::PairwiseLinearRanker::compare(&self, "
        "&ferricml::data::MatrixView<'_>, ferricml::ranking::PairIndex) -> "
        "core::result::Result<ferricml::ranking::PairOutcome, ferricml::ranking::PairwiseError>\n"
        "pub fn ferricml::ranking::PairwiseLinearRanker::compare_into(&self, "
        "&ferricml::data::MatrixView<'_>, &[ferricml::ranking::PairIndex], "
        "&mut [ferricml::ranking::PairOutcome]) -> "
        "core::result::Result<(), ferricml::ranking::PairwiseError>\n",
        "is not `compare` plus caller-owned output buffers",
    ),
    (
        # Audit finding 11: the only caller-owned method in the crate with no
        # allocating partner of any kind.
        "pair_margins",
        "impl ferricml::ranking::PairwiseLinearRanker\n"
        "pub fn ferricml::ranking::PairwiseLinearRanker::pair_margins_into(&self, "
        "&ferricml::data::MatrixView<'_>, &[ferricml::ranking::PairIndex], &mut [f32]) -> "
        "core::result::Result<(), ferricml::ranking::PairwiseError>\n",
        "has no allocating `pair_margins` partner",
    ),
    (
        # Audit finding 9: the allocating forwarder was inherent and the
        # caller-owned one reachable only through the trait.
        "predict_class_proba_forwarder",
        "impl ferricml::linear_model::LogisticRegression\n"
        "pub fn ferricml::linear_model::LogisticRegression::predict_class_proba(&self, "
        "&ferricml::data::MatrixView<'_>, u8) -> "
        "core::result::Result<alloc::vec::Vec<f32>, ferricml::api::ModelError>\n"
        "impl ferricml::api::ProbabilisticClassifier for "
        "ferricml::linear_model::LogisticRegression\n"
        "pub fn ferricml::linear_model::LogisticRegression::predict_class_proba_into(&self, "
        "&ferricml::data::MatrixView<'_>, u8, &mut [f32]) -> "
        "core::result::Result<(), ferricml::api::ModelError>\n",
        "only through ['trait-impl']",
    ),
)


def self_test() -> None:
    live = violations(frozen_profiles(load_manifest()))
    assert live == [], f"live surface violates its own pairing rules: {live}"

    declared = {name for name, _ in RULES}
    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )

    clean = _padded(CLEAN_BASELINE)
    found = violations([("clean", clean)], {})
    assert found == [], f"synthetic clean baseline reported violations: {found}"

    for name, mutate, expected in SYNTHETIC_VIOLATIONS:
        source = CLEAN_BASELINE if name == "pairs-are-actually-observed" else clean
        found = violations([(name, mutate(source))], {})
        assert any(expected in item for item in found), (
            f"violation case {name} did not fire; reported {found}"
        )

    for name, mutate, expected in INPUT_LOSS_CASES:
        found = violations([(name, mutate(clean))], {})
        assert any(expected in item for item in found), (
            f"input-loss case {name} passed vacuously; reported {found}"
        )

    absent = {"profiles": [{"baseline": "tests/api-baselines/rust/does-not-exist.txt"}]}
    try:
        frozen_profiles(absent)
    except RuntimeError:
        pass
    else:  # pragma: no cover - the guard exists precisely so this cannot happen
        raise AssertionError("a missing API baseline was not reported as a finding")

    for name, rows, expected in HISTORICAL_REGRESSIONS:
        found = violations([(name, _padded(rows))], {})
        assert any(expected in item for item in found), (
            f"historical defect {name} is no longer detected; reported {found}"
        )

    # A recorded gap that has been closed must be reported as stale, so the
    # exemption list cannot outlive the thing it excuses.
    stale = violations([("clean", clean)], {"ferricml::gone::Type::predict_into": "reason"})
    assert any("no longer describes the surface" in item for item in stale), (
        f"a stale recorded gap was not reported; got {stale}"
    )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "accessor pairing verifier self-test passed "
            f"({len(RULES)} rules, each proven to fire against a synthetic "
            f"violation; {len(INPUT_LOSS_CASES) + 1} ways of losing the input "
            f"proven to be reported; {len(HISTORICAL_REGRESSIONS)} historical "
            "defects reconstructed)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_accessor_pairing.py [--self-test]", file=sys.stderr)
        return 2
    try:
        profiles = frozen_profiles(load_manifest())
    except (RuntimeError, OSError, ValueError) as error:
        print(f"accessor pairing check failed: {error}", file=sys.stderr)
        return 1
    found = violations(profiles)
    if found:
        print("accessor pairing check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("accessor pairing: every `_into` method is its allocating partner plus a buffer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
