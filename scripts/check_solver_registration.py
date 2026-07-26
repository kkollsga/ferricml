#!/usr/bin/env python3
"""Enforce that every budgeted solver is registered in the convergence battery.

`tests/solver_convergence_contract.rs` starves every iterative solver the public
API budgets and requires `ModelError::SolverDidNotConverge` rather than the last
iterate. It is the mechanism behind a claim the changelog makes about *every*
iterative solver, and it works: a listed solver that regressed to returning an
unconverged iterate fails on the row it already has, and its own
`every_budgeted_entry_point_in_the_public_api_is_enumerated` catches a row being
**removed**.

What it cannot catch is a row never being **added**. A new estimator with an
iteration budget and no row leaves the file passing, the general claim
unenforced for that estimator, and nothing anywhere reporting the omission —
which is the precise shape of the defect the battery was built to end, one level
up. `ElasticNet` landed with its row; the next one has nothing but the author's
memory standing behind it.

# Why this rule is syntactic, and why that is the point

This file claims nothing about numerics. It asks one textual question — *does a
params type that exposes both `with_max_iter` and `with_tol` appear in the
battery?* — and a rule that shallow cannot be satisfied by the wrong fix. There
is no way to quiet it by weakening a tolerance, widening an acceptance test, or
making a solver return its last iterate; the only thing that satisfies it is a
row, and the row then has to pass the battery's own assertions on its own merits.
A semantic rule here would be a second, worse copy of the battery.

# Why *both* builders

`with_max_iter` alone is an iteration count, not a budget with a convergence
test: `HistGradientBoostingRegressorParams` and its classifier sibling run a
fixed number of boosting rounds and have no tolerance, so exhausting the count is
the normal successful outcome rather than a failure to converge.
`with_tol` alone is a numerical threshold inside a direct solve:
`LinearRegressionParams::with_tol` is a rank tolerance, and there is no iterate
to refuse. `SolverDidNotConverge` is only meaningful where a budget and a
convergence test both exist, which is exactly where both builders do. The
predicate is therefore the conjunction, and the three params types that carry one
builder without the other are correctly outside this file's scope rather than
recorded as exemptions.

`--self-test` proves every rule fires against a synthetic violation, proves the
absence of either input is reported rather than passed over, and proves the
source scan reads child modules rather than facades — the failure that let nine
rules in `scripts/check_source_layout.py` pass vacuously until 2026-07-26. Two of
the five live budgeted params types sit in a child module (`linear_model/lasso/`,
`linear_model/elastic_net/`), so a facade-only reader would already miss them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]

# The two builders whose co-presence *is* the definition of a budgeted solver.
BUDGET_BUILDER = "pub fn with_max_iter("
TOLERANCE_BUILDER = "pub fn with_tol("

BATTERY = "tests/solver_convergence_contract.rs"

# Budgeted params types that genuinely cannot carry a battery row, with the
# reason. Closed in **both** directions: an entry that no longer describes a
# budgeted type is reported as stale, so an exemption cannot outlive the thing it
# excuses, and a new one cannot be added without landing here where a reviewer
# sees it. Empty is the correct state — every budgeted solver in the crate is
# registered — and it staying empty is the point.
EXPECTED_UNREGISTERED: dict[str, str] = {}

# Below this many budgeted params types the source scan is not reading `src/`:
# the tree moved, the builders were renamed, or the impl-header parse broke.
# Reporting that is the difference between a passing check and a check that
# passed by not looking. Five exist today — logistic, Lasso, elastic net, Platt,
# pairwise ranker — and the floor sits below that so adding one is not an edit
# here, while losing the parse still is.
MINIMUM_BUDGETED_PARAMS_TYPES = 5


def source_files(root: Path, recursive: bool = True) -> list[Path]:
    """Every Rust source file under `src/`, child modules included.

    The recursion is load-bearing rather than incidental. Two of the five live
    budgeted params types — `LassoParams` and `ElasticNetParams` — live in
    `linear_model/lasso/mod.rs` and `linear_model/elastic_net/mod.rs`, one level
    below the facade, and the next estimator family will land the same way
    because that is the layout `check_source_layout.py` requires. `recursive` is
    the self-test's foil, never a caller's option: `self_test` sets it False and
    asserts that a child-module violation goes *unreported*, which is what makes
    each violation a proof that this scanner reads the tree.
    """
    source = root / "src"
    if not source.is_dir():
        return []
    return sorted(source.rglob("*.rs") if recursive else source.glob("*.rs"))


def _impl_owner(line: str) -> str | None:
    """The type an inherent `impl Name {` line opens, or `None`.

    Only inherent impls at column zero are considered, which is what rustfmt
    produces for every one of them in this crate. A trait implementation —
    `impl Default for LassoParams {` — is not a place a builder can be declared
    `pub fn`, so skipping it costs nothing and keeps the owner unambiguous.
    """
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
    if not body.endswith("{"):
        return None
    body = body[:-1].strip()
    if " where " in body or " for " in body:
        return None
    name = body.split("<", 1)[0].strip()
    return name or None


def budgeted_params_types(files: Iterable[Path], root: Path) -> dict[str, str]:
    """`{type name: declaring path}` for every type carrying both builders.

    Both builders have to be found for this file to mean anything, so finding
    none at all is a finding rather than a silently vacuous pass — see
    [`budgeted_params_are_observed`].
    """
    budget: dict[str, str] = {}
    tolerance: dict[str, str] = {}
    for path in files:
        owner: str | None = None
        for raw in path.read_text().splitlines():
            found = _impl_owner(raw)
            if found is not None:
                owner = found
                continue
            if raw and not raw[0].isspace():
                # Any other item at column zero closes the impl block.
                owner = None
                continue
            if owner is None:
                continue
            stripped = raw.strip()
            where = str(path.relative_to(root))
            if stripped.startswith(BUDGET_BUILDER):
                budget.setdefault(owner, where)
            elif stripped.startswith(TOLERANCE_BUILDER):
                tolerance.setdefault(owner, where)
    return {name: budget[name] for name in sorted(budget.keys() & tolerance.keys())}


def battery_registrations(root: Path) -> str | None:
    """The battery's text with its `use` declarations removed, or `None`.

    Imports are stripped so that importing a params type is not by itself a
    registration. A row is a use of the type in test code; an import with no row
    is the half-finished edit this rule exists to report, and it would otherwise
    satisfy a plain substring search.
    """
    path = root / BATTERY
    if not path.is_file():
        return None
    lines: list[str] = []
    inside_use = False
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if inside_use:
            if stripped.endswith("};") or stripped.endswith("};,"):
                inside_use = False
            continue
        if stripped.startswith("use "):
            inside_use = not stripped.endswith(";")
            continue
        lines.append(raw)
    return "\n".join(lines)


def budgeted_params_are_registered(
    budgeted: dict[str, str], battery: str | None, exempt: dict[str, str]
) -> list[str]:
    """Every budgeted params type is named by the convergence battery."""
    if battery is None:
        return []
    findings = []
    for name, where in budgeted.items():
        if name in exempt or name in battery:
            continue
        findings.append(
            f"budgeted params type `{name}` ({where}) exposes both "
            f"`with_max_iter` and `with_tol` but is never named in {BATTERY}; "
            f"add a starved-and-generous row for its fitting entry points, or "
            f"record the omission in EXPECTED_UNREGISTERED with a reason"
        )
    return findings


def recorded_exemptions_are_live(
    budgeted: dict[str, str], _battery: str | None, exempt: dict[str, str]
) -> list[str]:
    """An exemption for a type that is no longer budgeted is stale."""
    return [
        f"recorded exemption `{name}` no longer names a params type carrying "
        f"both `with_max_iter` and `with_tol`; remove it from "
        f"EXPECTED_UNREGISTERED"
        for name in sorted(set(exempt) - set(budgeted))
    ]


def budgeted_params_are_observed(
    budgeted: dict[str, str], _battery: str | None, _exempt: dict[str, str]
) -> list[str]:
    """The source scan has to find budgeted types, or it checked nothing."""
    if len(budgeted) < MINIMUM_BUDGETED_PARAMS_TYPES:
        return [
            f"only {len(budgeted)} params types carrying both `with_max_iter` "
            f"and `with_tol` were found under src/, below the floor of "
            f"{MINIMUM_BUDGETED_PARAMS_TYPES}; the tree moved, a builder was "
            f"renamed, or the impl-header parse broke"
        ]
    return []


def the_battery_is_readable(
    _budgeted: dict[str, str], battery: str | None, _exempt: dict[str, str]
) -> list[str]:
    """The battery must exist and carry rows, or there is nothing to register in.

    Reported as its own finding rather than left to cascade through
    [`budgeted_params_are_registered`], so a deleted or emptied battery names its
    own cause instead of arriving as one complaint per estimator.
    """
    if battery is None:
        return [f"the convergence battery is missing: {BATTERY}"]
    if "Row {" not in battery:
        return [
            f"{BATTERY} declares no `Row {{`; the battery is a stub, so nothing "
            f"below is a registration"
        ]
    return []


RULES: tuple[
    tuple[str, Callable[[dict[str, str], str | None, dict[str, str]], list[str]]], ...
] = (
    ("budgeted-params-are-registered", budgeted_params_are_registered),
    ("recorded-exemptions-are-live", recorded_exemptions_are_live),
    ("budgeted-params-are-observed", budgeted_params_are_observed),
    ("battery-is-readable", the_battery_is_readable),
)


def violations(
    root: Path = ROOT,
    exempt: dict[str, str] | None = None,
    recursive: bool = True,
) -> list[str]:
    budgeted = budgeted_params_types(source_files(root, recursive), root)
    battery = battery_registrations(root)
    recorded = EXPECTED_UNREGISTERED if exempt is None else exempt
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(budgeted, battery, recorded))
    return found


# --------------------------------------------------------------- self-test

PARAMS_IMPL = """impl {name} {{
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {{
        self.max_iter = max_iter;
        self
    }}

    pub fn with_tol(mut self, tol: f32) -> Self {{
        self.tol = tol;
        self
    }}
}}
"""

# One params type per file, so a synthetic violation can delete or move exactly
# one of them. `SecondParams` and `ThirdParams` sit in **child modules**, which
# is where the real ones live and where a facade-only reader stops seeing them.
CLEAN_SOURCES: dict[str, str] = {
    "lib.rs": "pub mod first;\npub mod second;\npub mod third;\n",
    "first.rs": "//! first\npub struct FirstParams;\n" + PARAMS_IMPL.format(name="FirstParams"),
    "second/mod.rs": (
        "//! second\npub struct SecondParams;\n" + PARAMS_IMPL.format(name="SecondParams")
    ),
    "third/nested/mod.rs": (
        "//! third\npub struct ThirdParams;\n" + PARAMS_IMPL.format(name="ThirdParams")
    ),
    # A budget with no convergence test, and a tolerance with no budget. Both
    # must stay outside the budgeted set, or the conjunction this file is built
    # on is not the predicate it claims to be.
    "boosting.rs": (
        "//! boosting\npub struct BoostingParams;\nimpl BoostingParams {\n"
        "    pub fn with_max_iter(mut self, max_iter: usize) -> Self { self }\n}\n"
    ),
    "direct.rs": (
        "//! direct\npub struct DirectParams;\nimpl DirectParams {\n"
        "    pub fn with_tol(mut self, tol: f32) -> Self { self }\n}\n"
    ),
    # A trait implementation is not a place a builder is declared `pub fn`, so
    # one carrying both names must not create a budgeted type out of thin air.
    "trait_impl.rs": (
        "//! trait impl\nimpl Default for DecoyParams {\n"
        "    pub fn with_max_iter(mut self, max_iter: usize) -> Self { self }\n"
        "    pub fn with_tol(mut self, tol: f32) -> Self { self }\n}\n"
    ),
}

CLEAN_BATTERY = """//! Every iterative solver refuses an exhausted budget.

use ferricml::api::ModelError;
use ferricml::first::{First, FirstParams};
use ferricml::second::{
    Second, SecondParams,
};
use ferricml::third::{Third, ThirdParams};

struct Row {
    name: &'static str,
    starved: fn() -> Result<(), ModelError>,
    generous: fn() -> Result<(), ModelError>,
}

fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "First::fit",
            starved: || First::fit(FirstParams::default().with_max_iter(1)).map(drop),
            generous: || First::fit(FirstParams::default()).map(drop),
        },
        Row {
            name: "Second::fit",
            starved: || Second::fit(SecondParams::default().with_max_iter(1)).map(drop),
            generous: || Second::fit(SecondParams::default()).map(drop),
        },
        Row {
            name: "Third::fit",
            starved: || Third::fit(ThirdParams::default().with_max_iter(1)).map(drop),
            generous: || Third::fit(ThirdParams::default()).map(drop),
        },
    ]
}
"""


def write_clean_tree(root: Path, padded: bool = True) -> Path:
    """The smallest tree that satisfies every rule.

    `padded` adds registered filler params types so the synthetic tree clears
    [`MINIMUM_BUDGETED_PARAMS_TYPES`]. Without it every other rule's case would
    report the floor finding as well, and the one being proven would be
    indistinguishable from the padding — so the floor rule is the single case
    that runs unpadded, which is also the only honest way to make a
    non-vacuity guard fire.
    """
    for relative, text in CLEAN_SOURCES.items():
        path = root / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    battery = root / BATTERY
    battery.parent.mkdir(parents=True, exist_ok=True)
    text = CLEAN_BATTERY
    if padded:
        filler_rows = []
        for index in range(MINIMUM_BUDGETED_PARAMS_TYPES):
            name = f"Filler{index}Params"
            path = root / "src" / "filler" / f"{index}.rs"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"//! filler\npub struct {name};\n" + PARAMS_IMPL.format(name=name))
            filler_rows.append(
                f"        Row {{\n"
                f'            name: "Filler{index}::fit",\n'
                f"            starved: || Filler{index}::fit({name}::default()).map(drop),\n"
                f"            generous: || Filler{index}::fit({name}::default()).map(drop),\n"
                f"        }},"
            )
        text = text.replace("    ]\n}", "\n".join(filler_rows) + "\n    ]\n}")
    battery.write_text(text)
    return root


def _drop_rows_for(root: Path, name: str) -> None:
    """Remove every mention of `name` from the battery except its import.

    This is the real defect's shape: the params type exists, the battery imports
    it, and no row uses it. A checker reading the whole file would pass.
    """
    battery = root / BATTERY
    text = battery.read_text()
    kept = []
    skipping = False
    for line in text.splitlines():
        if line.strip() == "Row {":
            block: list[str] = [line]
            skipping = True
            kept.append(block)
            continue
        if skipping:
            kept[-1].append(line)
            if line.strip() == "},":
                skipping = False
            continue
        kept.append(line)
    rebuilt = []
    for item in kept:
        if isinstance(item, list):
            if any(name in line for line in item):
                continue
            rebuilt.extend(item)
        else:
            rebuilt.append(item)
    battery.write_text("\n".join(rebuilt) + "\n")


SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "budgeted-params-are-registered",
        lambda root: _drop_rows_for(root, "FirstParams"),
        "budgeted params type `FirstParams`",
    ),
    (
        "recorded-exemptions-are-live",
        lambda root: None,
        "no longer names a params type carrying",
    ),
    (
        # The one case that runs against the unpadded tree: a non-vacuity floor
        # is proven by giving it too little to look at, not by breaking
        # something else.
        "budgeted-params-are-observed",
        lambda root: None,
        "below the floor of",
    ),
    (
        "battery-is-readable",
        lambda root: (root / BATTERY).unlink(),
        "the convergence battery is missing",
    ),
)

# The registration failure, written into a **child module** of the tree rather
# than into a file beside the facade.
#
# A params type declared next to `lib.rs` is found by a recursive reader and by a
# facade-only one alike, so an omission there cannot establish which reader this
# scanner uses. These can: `self_test` asserts each fires under the recursive
# scan and stays silent under the facade-only one. Both live budgeted params
# types that a facade-only reader would miss — `LassoParams` and
# `ElasticNetParams` — sit exactly one and one level down respectively, and
# `ThirdParams` covers two levels because `model_selection/split/` shows that
# depth is not bounded at one.
CHILD_MODULE_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "budgeted-params-are-registered",
        lambda root: _drop_rows_for(root, "SecondParams"),
        "budgeted params type `SecondParams`",
    ),
    (
        "budgeted-params-are-registered",
        lambda root: _drop_rows_for(root, "ThirdParams"),
        "budgeted params type `ThirdParams`",
    ),
)

# Ways the input can disappear underneath the rules. Each must be reported rather
# than passing quietly, because every one of them makes the check vacuous while
# leaving it green.
INPUT_LOSS_CASES: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "battery-deleted",
        lambda root: (root / BATTERY).unlink(),
        "the convergence battery is missing",
    ),
    (
        "battery-emptied",
        lambda root: (root / BATTERY).write_text(""),
        "declares no `Row {`",
    ),
    (
        "battery-stubbed-to-imports",
        lambda root: (root / BATTERY).write_text(
            "use ferricml::first::FirstParams;\nuse ferricml::second::SecondParams;\n"
        ),
        "declares no `Row {`",
    ),
    (
        "source-tree-gone",
        lambda root: _remove_tree(root / "src"),
        "below the floor of",
    ),
    (
        "builders-renamed",
        lambda root: _rename_builders(root),
        "below the floor of",
    ),
)


def _remove_tree(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory)


def _rename_builders(root: Path) -> None:
    for path in (root / "src").rglob("*.rs"):
        text = path.read_text()
        path.write_text(text.replace("pub fn with_tol(", "pub fn with_tolerance("))


# The defect class this file was written for, reconstructed as the tree that
# existed the moment before a row was added. `ElasticNet` was the crate's most
# recent budgeted estimator; had its battery row been forgotten, nothing in the
# repository would have said so. This is that repository.
HISTORICAL_REGRESSIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    (
        "elastic-net-landed-without-a-row",
        lambda root: _drop_rows_for(root, "ThirdParams"),
        "budgeted params type `ThirdParams`",
    ),
)


def self_test() -> None:
    live = violations()
    assert live == [], f"live tree violates its own registration rules: {live}"

    declared = {name for name, _ in RULES}
    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    assert covered == declared, (
        f"every rule needs a synthetic violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )
    child_covered = {name for name, _, _ in CHILD_MODULE_VIOLATIONS}
    assert child_covered <= declared, (
        f"stale child-module violations: {sorted(child_covered - declared)}"
    )

    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)

        clean = write_clean_tree(base / "clean")
        found = violations(clean, {})
        assert found == [], f"synthetic clean tree reported violations: {found}"

        # The predicate is the conjunction: neither half alone makes a type
        # budgeted, and a trait impl does not declare a builder at all. Asserted
        # on the unpadded tree, which carries `BoostingParams` (budget only),
        # `DirectParams` (tolerance only) and a trait impl naming both.
        bare = write_clean_tree(base / "bare", padded=False)
        budgeted = budgeted_params_types(source_files(bare), bare)
        assert set(budgeted) == {"FirstParams", "SecondParams", "ThirdParams"}, (
            f"the budgeted set is not the conjunction of both builders: {sorted(budgeted)}"
        )

        for index, (name, mutate, expected) in enumerate(SYNTHETIC_VIOLATIONS):
            tree = write_clean_tree(
                base / f"violation-{index}-{name}",
                padded=name != "budgeted-params-are-observed",
            )
            mutate(tree)
            stale = {"GoneParams": "reason"} if name == "recorded-exemptions-are-live" else {}
            found = violations(tree, stale)
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against its synthetic violation; reported {found}"
            )

        for index, (name, mutate, expected) in enumerate(CHILD_MODULE_VIOLATIONS):
            tree = write_clean_tree(base / f"child-{index}-{name}")
            mutate(tree)
            found = violations(tree, {})
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against a violation in a child module; "
                f"reported {found}"
            )
            shallow = violations(tree, {}, recursive=False)
            assert not any(expected in item for item in shallow), (
                f"rule {name}'s child-module violation is also reported by a "
                f"facade-only reader, so it does not prove the scan reads the "
                f"tree; reported {shallow}"
            )

        for index, (name, mutate, expected) in enumerate(INPUT_LOSS_CASES):
            tree = write_clean_tree(base / f"input-loss-{index}-{name}")
            mutate(tree)
            found = violations(tree, {})
            assert any(expected in item for item in found), (
                f"input-loss case {name} passed vacuously; reported {found}"
            )

        for index, (name, mutate, expected) in enumerate(HISTORICAL_REGRESSIONS):
            tree = write_clean_tree(base / f"historical-{index}-{name}")
            mutate(tree)
            found = violations(tree, {})
            assert any(expected in item for item in found), (
                f"historical defect {name} is no longer detected; reported {found}"
            )

        # An exemption suppresses the finding it names and nothing else, so a
        # recorded gap cannot silently cover a second omission.
        tree = write_clean_tree(base / "exempt")
        _drop_rows_for(tree, "FirstParams")
        _drop_rows_for(tree, "SecondParams")
        found = violations(tree, {"FirstParams": "reason"})
        assert not any("`FirstParams`" in item for item in found), (
            f"an exemption did not suppress its own finding; reported {found}"
        )
        assert any("`SecondParams`" in item for item in found), (
            f"an exemption suppressed a second omission as well; reported {found}"
        )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "solver registration verifier self-test passed "
            f"({len(RULES)} rules, each proven to fire against a synthetic "
            f"violation; {len(CHILD_MODULE_VIOLATIONS)} proven again against a "
            f"violation in a child module a facade-only reader misses; "
            f"{len(INPUT_LOSS_CASES)} ways of losing either input proven to be "
            f"reported; {len(HISTORICAL_REGRESSIONS)} historical defect "
            "reconstructed)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_solver_registration.py [--self-test]", file=sys.stderr)
        return 2
    found = violations()
    if found:
        print("solver registration check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1
    budgeted = budgeted_params_types(source_files(ROOT), ROOT)
    print(
        f"solver registration: all {len(budgeted)} budgeted params types are "
        f"named in {BATTERY}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
