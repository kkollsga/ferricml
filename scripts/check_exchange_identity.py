#!/usr/bin/env python3
"""Prove FerricML's exchange containers mean the same thing in both languages.

The generator is Rust and there is deliberately no second implementation of it,
so the whole cross-language contract rests on one loop: the crate materializes a
container, `python/ferricml_datasets` reads it back, and what the reader hands a
consumer is what the crate's own pinned literals say the stream produces. That
loop was proven once by hand and then went untracked, which means a reader-side
regression — a dtype read as native-endian, an offset applied to the wrong
array, a `memmap` shape transposed — would have been caught by nobody: the Rust
tests never run the reader, and the reader's own structural checks pass happily
on bytes it has decoded wrongly.

This is that loop as a named gate. It is deliberately **not** part of `make
gate`: it needs an interpreter with NumPy, exactly as `reference-check` needs a
reference environment and `docs-build` needs a pinned documentation toolchain.
Being separately named is also what lets it fail loudly. A cross-language test
that skipped itself when the interpreter or NumPy was absent would read as a
green gate forever, which is the failure mode this file is designed against:
every reason to give up is an error, never a skip.

Each claim is a named rule over a loaded catalogue, so `--self-test` can plant a
violation of each one in a synthetic catalogue and assert the rule still fires.
The plant that matters most is the one no digest can catch: a value changed in
the array file *and the recorded digest recomputed over it*, so the container is
internally consistent and disagrees only with the Rust literal. A gate that
caught only corruption would prove the file arrived intact, not that the two
languages agree about what it says.

    make exchange-check
    make exchange-check EXCHANGE_PYTHON=/path/to/an/interpreter/with/numpy
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - the loud half of the design
    raise SystemExit(
        "exchange identity check: this interpreter has no NumPy, which the "
        "container reader requires.\n"
        f"  interpreter: {sys.executable}\n"
        "Point the gate at one that has it — `make exchange-check "
        "EXCHANGE_PYTHON=<interpreter>` — rather than skipping the check: a "
        "cross-language gate that skips is a green light nobody earned."
    ) from None

from ferricml_datasets import (  # noqa: E402 - after the NumPy diagnosis above
    Container,
    ContainerError,
    NotRegenerable,
    default_directory,
    load,
    materialize,
)

# The suite whose containers carry the frozen reference lanes. Its entries are
# the ones drawn from the raw stream states the crate's fixtures were recorded
# against, which is what makes them comparable with a pinned literal at all.
SUITE = "reference"

# Where the pinned stream literals live, and the block that holds them.
#
# The literals are *read* rather than copied, because a second copy of them in
# this file would be a second thing to keep byte-identical by hand — which is
# precisely the arrangement the exchange format exists to abolish. Reading them
# also means this gate cannot be satisfied by editing the Python side: the
# comparison moves only when the Rust pin moves.
PIN_SOURCE = Path("src") / "datasets" / "tests.rs"
PIN_BLOCK = re.compile(
    r"Recipe::new\(\s*(?P<rows>\d+)\s*,\s*(?P<columns>\d+)\s*,\s*"
    r"Source::Sampled\s*\{\s*state:\s*(?P<state>\d+)\s*\}\s*\)\s*"
    r"\.unwrap\(\)\s*\.design\(\);\s*"
    r"assert_eq!\(\s*\w+\.as_slice\(\),\s*\[(?P<values>[^\]]*)\]",
    re.S,
)

# The split whose first row is the design's first row. A lane's test half is the
# same stream continued past the training rows, so only the training half can be
# compared with the literals a two-row recipe pins.
FIRST_SPLIT = "train"


@dataclass(frozen=True)
class Pins:
    """The Rust literals a materialized container has to reproduce.

    `state` is the stream this design was drawn from, `rows`/`columns` the shape
    the literals were read at, and `values` the design in row-major order — the
    fill order `a_generated_design_is_filled_row_by_row` pins, which is what
    makes the first `len(values)` draws of a *wider* design the same numbers.
    """

    state: int
    rows: int
    columns: int
    values: tuple[float, ...]

    @property
    def array(self) -> np.ndarray:
        return np.array(self.values, dtype=np.dtype("<f4"))


@dataclass
class Catalogue:
    """One materialized suite, loaded, beside the pins it must agree with.

    `listed` is what the crate said it would write, taken from the generator's
    own `--list` rather than from a roster repeated here: a lane or a seed added
    to the crate has to appear in this gate without anyone remembering to add
    it, and a container that silently stopped being written has to be a failure
    rather than one fewer comparison.
    """

    directory: Path
    listed: tuple[str, ...]
    pins: Pins | None
    containers: dict[str, Container] = field(default_factory=dict)
    unloadable: list[tuple[str, str]] = field(default_factory=list)


def read_pins(root: Path) -> Pins | None:
    """The pinned literals, parsed out of the crate's own test source."""
    source = root / PIN_SOURCE
    if not source.is_file():
        return None
    matches = PIN_BLOCK.findall(source.read_text())
    if len(matches) != 1:
        return None
    rows, columns, state, values = matches[0]
    parsed = []
    for literal in values.split(","):
        text = literal.strip().replace("_", "")
        if not text:
            continue
        for suffix in ("f32", "f64"):
            text = text.removesuffix(suffix)
        parsed.append(float(text))
    if not parsed:
        return None
    return Pins(int(state), int(rows), int(columns), tuple(parsed))


def decimal(values: np.ndarray) -> str:
    """A row of `f32` values at nine significant digits, which round-trips them.

    Enough to distinguish any two `f32` values, so a reported disagreement can
    be read as a disagreement rather than as a printing artefact.
    """
    return ", ".join(f"{value:.9g}" for value in values)


def parse_name(name: str) -> tuple[str, int, str] | None:
    """A container's lane, seed and split, as its name records them."""
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != SUITE or not parts[2].isdigit():
        return None
    return parts[1], int(parts[2]), parts[3]


# -- the rules ---------------------------------------------------------------


def pinned_literals_are_readable(catalogue: Catalogue) -> list[str]:
    """The comparison has something to compare against.

    Absence is the finding rather than a silently vacuous pass, which is the
    same treatment every single-source rule in `check_source_layout.py` gets:
    a renamed test, a reshaped literal block, or a regex that stopped matching
    would otherwise leave this file reporting success while comparing nothing.
    """
    if catalogue.pins is None:
        return [
            f"pinned source literals are missing: {PIN_SOURCE} carries no single "
            "frozen `Source::Sampled` design block for this check to read"
        ]
    return []


def the_catalogue_is_complete(catalogue: Catalogue) -> list[str]:
    """Every container the generator listed loads, digest and table verified.

    Loading is the check: `ferricml_datasets.load` verifies the array file's
    length and digest and walks the array table against the file it describes,
    so a container that arrives truncated, re-hashed, or with a table that no
    longer covers its bytes fails here rather than being partly believed.
    """
    if not catalogue.listed:
        return [
            "the generator listed no container, so every comparison below would "
            "pass without reading anything"
        ]
    return [
        f"listed container could not be loaded: {name}: {reason}"
        for name, reason in catalogue.unloadable
    ]


def the_stream_literals_agree(catalogue: Catalogue) -> list[str]:
    """What Python decodes is what Rust pinned, value for value.

    This is the whole cross-language claim. The comparison is exact rather than
    tolerant for the same reason the Rust assertion is: every operation between
    the stream and the stored `f32` is exact, so any difference at all is a
    changed contract rather than a rounding disagreement.
    """
    pins = catalogue.pins
    if pins is None:
        return []
    width = len(pins.values)
    findings: list[str] = []
    compared = 0
    for name, container in sorted(catalogue.containers.items()):
        parsed = parse_name(name)
        if parsed is None:
            findings.append(f"container name is not a lane, seed and split: {name}")
            continue
        _, seed, split = parsed
        if seed != pins.state or split != FIRST_SPLIT:
            continue
        features = container.features
        if features.dtype != np.dtype("<f4"):
            findings.append(
                f"{name}: features decoded as {features.dtype}, not little-endian f32"
            )
            continue
        if features.shape[1] < width:
            findings.append(
                f"{name}: {features.shape[1]} columns cannot carry the "
                f"{width} pinned values"
            )
            continue
        observed = np.asarray(features[0, :width])
        compared += 1
        if not np.array_equal(observed.view(np.uint32), pins.array.view(np.uint32)):
            findings.append(
                f"{name}: the Python reader disagrees with the pinned literal "
                f"at stream state {pins.state}: read [{decimal(observed)}], "
                f"pinned [{decimal(pins.array)}]"
            )
    if compared == 0 and not findings:
        findings.append(
            f"no materialized container carries stream state {pins.state} at its "
            f"{FIRST_SPLIT} split, so the pinned comparison never ran"
        )
    return findings


def the_payload_block_is_honest(catalogue: Catalogue) -> list[str]:
    """A derived container says so, and refuses to be called regenerable.

    Both halves of a reference lane record the digest of the single design they
    were cut out of, so the digest cannot tell them apart from that design. The
    `payload` block is what does, and a reader that reported it wrongly — or a
    writer that stopped writing it — would hand a harness a recipe it would
    regenerate into 1152 untargeted rows while every digest it checked agreed.
    """
    findings: list[str] = []
    for name, container in sorted(catalogue.containers.items()):
        parsed = parse_name(name)
        if parsed is None:
            continue
        lane, seed, split = parsed
        if container.payload != "derived":
            findings.append(
                f"{name}: payload reads {container.payload!r}, and a reference "
                "lane's split is not its recipe's own output"
            )
            continue
        derivation = container.derivation
        if derivation is None:
            findings.append(f"{name}: payload is derived but records no derivation")
            continue
        if (derivation.lane, derivation.seed, derivation.split) != (lane, seed, split):
            findings.append(
                f"{name}: derivation records {derivation.lane}/{derivation.seed}/"
                f"{derivation.split}, which is not what the container is named"
            )
        try:
            container.regenerable_recipe()
        except NotRegenerable:
            continue
        findings.append(
            f"{name}: a derived container offered its recipe as regenerable"
        )
    return findings


def the_splits_are_halves_of_one_design(catalogue: Catalogue) -> list[str]:
    """Two containers, one design: shapes that sum, one digest, no shared row.

    The reader has no way to know that a lane's two halves came from one
    generated matrix, so this is the property that catches an offset applied to
    the wrong array or a split boundary read from the wrong field: both halves
    decode, both pass their digests, and the rows they hand back overlap.
    """
    pairs: dict[tuple[str, int], dict[str, Container]] = {}
    findings: list[str] = []
    for name, container in sorted(catalogue.containers.items()):
        parsed = parse_name(name)
        if parsed is None:
            continue
        lane, seed, split = parsed
        pairs.setdefault((lane, seed), {})[split] = container
    for (lane, seed), halves in sorted(pairs.items()):
        if len(halves) != 2:
            findings.append(
                f"{SUITE}_{lane}_{seed}: {sorted(halves)} is not both halves of a lane"
            )
            continue
        first, second = (halves[key] for key in sorted(halves))
        if first.spec_digest != second.spec_digest:
            findings.append(
                f"{SUITE}_{lane}_{seed}: the two halves record different recipes"
            )
            continue
        recipe = first.recipe
        rows = first.features.shape[0] + second.features.shape[0]
        if rows != int(recipe["rows"]):
            findings.append(
                f"{SUITE}_{lane}_{seed}: the halves decode to {rows} rows, and "
                f"the recipe they share is {recipe['rows']} rows tall"
            )
        for half, container in sorted(halves.items()):
            if container.features.shape[1] != int(recipe["columns"]):
                findings.append(
                    f"{SUITE}_{lane}_{seed}_{half}: decoded "
                    f"{container.features.shape[1]} columns against a recipe of "
                    f"{recipe['columns']}"
                )
        if np.array_equal(first.features[0], second.features[0]):
            findings.append(
                f"{SUITE}_{lane}_{seed}: both halves decode the same first row, "
                "so one of them is not the stream continued past the other"
            )
    return findings


RULES: tuple[tuple[str, Callable[[Catalogue], list[str]]], ...] = (
    ("pinned-literals-readable", pinned_literals_are_readable),
    ("catalogue-complete", the_catalogue_is_complete),
    ("stream-literals-agree", the_stream_literals_agree),
    ("payload-block-honest", the_payload_block_is_honest),
    ("splits-are-one-design", the_splits_are_halves_of_one_design),
)


def violations(catalogue: Catalogue) -> list[str]:
    found: list[str] = []
    for _, rule in RULES:
        found.extend(rule(catalogue))
    return found


# -- building the catalogue ---------------------------------------------------


def listed_entries(root: Path) -> tuple[str, ...]:
    """The catalogue the generator says this suite has, from the generator."""
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--release",
            "--features",
            "datasets",
            "--bin",
            "ferricml-datagen",
            "--",
            "--list",
            "--suite",
            SUITE,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "exchange identity check: the generator would not list its "
            f"catalogue (exit {completed.returncode})\n{completed.stderr.strip()}"
        )
    return tuple(
        line.split("\t")[0] for line in completed.stdout.splitlines() if line.strip()
    )


def build(directory: Path, listed: tuple[str, ...], root: Path) -> Catalogue:
    catalogue = Catalogue(directory=directory, listed=listed, pins=read_pins(root))
    for name in listed:
        try:
            catalogue.containers[name] = load(directory / name)
        except (OSError, ContainerError, json.JSONDecodeError, KeyError) as error:
            catalogue.unloadable.append((name, f"{type(error).__name__}: {error}"))
    return catalogue


def materialized(root: Path) -> Catalogue:
    """Regenerate the suite and read it back.

    `force=True` rather than the cache, deliberately. The generator's cache is
    keyed on the recipe, not on the code that draws it, so a container written
    by an older crate whose recipe has not changed is reused — which is exactly
    the container a regression would leave behind, and reusing it would let this
    gate compare last week's bytes against this week's literals and agree.
    """
    listed = listed_entries(root)
    directory = default_directory(root)
    materialize(suite=SUITE, out=directory, root=root, force=True)
    return build(directory, listed, root)


# -- the synthetic catalogue and its planted violations -----------------------

# A clean catalogue's shape. Small enough to write by hand and wide enough to
# carry the pinned values, with a second seed present so a rule that compared
# *every* seed against the literals would fail the clean tree rather than pass
# it.
CLEAN_COLUMNS = 12
CLEAN_TRAIN_ROWS = 4
CLEAN_TEST_ROWS = 2
CLEAN_SEEDS = (11, 22)
CLEAN_LANES = ("nonlinear", "regression")

# The pin block, as the synthetic crate source carries it. Written in the live
# file's shape so `read_pins` is exercised against the arrangement it has to
# parse rather than against a convenience spelling.
CLEAN_PIN_VALUES = (
    -0.36751127,
    -0.4752698,
    0.27608466,
    0.009227991,
    -0.6696149,
    0.10387528,
)
CLEAN_PIN_SOURCE = (
    "#[test]\n"
    "fn the_first_design_values_of_every_source_are_frozen() {\n"
    "    let sampled = Recipe::new(2, 3, Source::Sampled { state: 11 })\n"
    "        .unwrap()\n"
    "        .design();\n"
    "    assert_eq!(\n"
    "        sampled.as_slice(),\n"
    "        [\n"
    + "".join(f"            {value},\n" for value in CLEAN_PIN_VALUES)
    + "        ]\n"
    "    );\n"
    "}\n"
)


def clean_features(seed: int, split: str) -> np.ndarray:
    """One synthetic design half, carrying the pins where the real one does."""
    rows = CLEAN_TRAIN_ROWS if split == FIRST_SPLIT else CLEAN_TEST_ROWS
    offset = 0.0 if split == FIRST_SPLIT else 100.0
    values = np.arange(rows * CLEAN_COLUMNS, dtype=np.dtype("<f4"))
    values = (values / 64.0 + offset + seed).astype(np.dtype("<f4"))
    features = values.reshape(rows, CLEAN_COLUMNS)
    if seed == CLEAN_SEEDS[0] and split == FIRST_SPLIT:
        features[0, : len(CLEAN_PIN_VALUES)] = np.array(
            CLEAN_PIN_VALUES, dtype=np.dtype("<f4")
        )
    return features


def write_container(
    directory: Path,
    name: str,
    features: np.ndarray,
    *,
    lane: str,
    seed: int,
    split: str,
    recipe_rows: int,
) -> None:
    """Write one synthetic container, manifest and array file, self-consistent."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = features.tobytes()
    (directory / f"{name}.bin").write_bytes(payload)
    manifest = {
        "format": 2,
        "spec_digest": f"{seed:064d}",
        "portability": "bit-exact",
        "payload": {
            "kind": "derived",
            "derivation": "reference-split",
            "lane": lane,
            "seed": seed,
            "split": split,
        },
        "recipe": {
            "rows": recipe_rows,
            "columns": CLEAN_COLUMNS,
            "source": {"kind": "sampled", "state": seed},
        },
        "data": {
            "file": f"{name}.bin",
            "bytes": len(payload),
            "digest": hashlib.sha256(payload).hexdigest(),
        },
        "arrays": [
            {
                "name": "features",
                "dtype": "f32",
                "rows": int(features.shape[0]),
                "columns": int(features.shape[1]),
                "byte_offset": 0,
                "len": int(features.size),
            }
        ],
    }
    (directory / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))


def write_clean_catalogue(base: Path) -> tuple[Path, tuple[str, ...]]:
    """The smallest catalogue and crate source that satisfy every rule."""
    root = base / "crate"
    (root / PIN_SOURCE).parent.mkdir(parents=True, exist_ok=True)
    (root / PIN_SOURCE).write_text(CLEAN_PIN_SOURCE)
    directory = root / "containers"
    listed = []
    for lane in CLEAN_LANES:
        for seed in CLEAN_SEEDS:
            for split in (FIRST_SPLIT, "test"):
                name = f"{SUITE}_{lane}_{seed}_{split}"
                listed.append(name)
                write_container(
                    directory,
                    name,
                    clean_features(seed, split),
                    lane=lane,
                    seed=seed,
                    split=split,
                    recipe_rows=CLEAN_TRAIN_ROWS + CLEAN_TEST_ROWS,
                )
    return root, tuple(listed)


def edit_manifest(root: Path, name: str, mutate: Callable[[dict], None]) -> None:
    path = root / "containers" / f"{name}.manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2))


def rewrite_values(root: Path, name: str, mutate: Callable[[np.ndarray], None]) -> None:
    """Change a decoded value and re-record the digest over the result.

    The re-hash is the point. A container whose digest no longer matches is
    caught by the reader before any rule here runs, which proves the transport
    intact and nothing about the two languages agreeing; re-hashing produces a
    container that is internally perfect and semantically wrong, which is the
    only shape a real cross-language regression takes.
    """
    directory = root / "containers"
    manifest = json.loads((directory / f"{name}.manifest.json").read_text())
    record = manifest["arrays"][0]
    values = np.fromfile(directory / f"{name}.bin", dtype=np.dtype("<f4"))
    values = values.reshape(record["rows"], record["columns"])
    mutate(values)
    payload = values.tobytes()
    (directory / f"{name}.bin").write_bytes(payload)
    manifest["data"]["digest"] = hashlib.sha256(payload).hexdigest()
    manifest["data"]["bytes"] = len(payload)
    (directory / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))


def drop_pin_block(root: Path) -> None:
    (root / PIN_SOURCE).write_text("//! the pin block, gone\n")


def one_ulp(root: Path) -> None:
    def mutate(values: np.ndarray) -> None:
        value = values[0, 0].view(np.uint32)
        values[0, 0] = (value + np.uint32(1)).view(np.float32)

    rewrite_values(root, f"{SUITE}_{CLEAN_LANES[0]}_{CLEAN_SEEDS[0]}_{FIRST_SPLIT}", mutate)


def transpose_first_row(root: Path) -> None:
    """Decode-order damage: the pinned prefix reversed, every byte still present."""

    def mutate(values: np.ndarray) -> None:
        width = len(CLEAN_PIN_VALUES)
        values[0, :width] = values[0, :width][::-1]

    rewrite_values(root, f"{SUITE}_{CLEAN_LANES[0]}_{CLEAN_SEEDS[0]}_{FIRST_SPLIT}", mutate)


def truncate_array_file(root: Path) -> None:
    path = root / "containers" / f"{SUITE}_{CLEAN_LANES[0]}_{CLEAN_SEEDS[0]}_test.bin"
    path.write_bytes(path.read_bytes()[:-4])


def remove_a_container(root: Path) -> None:
    name = f"{SUITE}_{CLEAN_LANES[1]}_{CLEAN_SEEDS[1]}_test"
    (root / "containers" / f"{name}.manifest.json").unlink()


def claim_generated(root: Path) -> None:
    edit_manifest(
        root,
        f"{SUITE}_{CLEAN_LANES[0]}_{CLEAN_SEEDS[0]}_{FIRST_SPLIT}",
        lambda manifest: manifest.__setitem__("payload", {"kind": "generated"}),
    )


def mislabel_derivation(root: Path) -> None:
    edit_manifest(
        root,
        f"{SUITE}_{CLEAN_LANES[0]}_{CLEAN_SEEDS[0]}_test",
        lambda manifest: manifest["payload"].__setitem__("split", FIRST_SPLIT),
    )


def split_the_recipe(root: Path) -> None:
    edit_manifest(
        root,
        f"{SUITE}_{CLEAN_LANES[1]}_{CLEAN_SEEDS[0]}_test",
        lambda manifest: manifest.__setitem__("spec_digest", "f" * 64),
    )


def duplicate_the_first_row(root: Path) -> None:
    """The two halves decode to the same opening row: one design read twice."""
    directory = root / "containers"
    lane, seed = CLEAN_LANES[1], CLEAN_SEEDS[1]
    train = np.fromfile(directory / f"{SUITE}_{lane}_{seed}_{FIRST_SPLIT}.bin", dtype=np.dtype("<f4"))

    def mutate(values: np.ndarray) -> None:
        values[0, :] = train[:CLEAN_COLUMNS]

    rewrite_values(root, f"{SUITE}_{lane}_{seed}_test", mutate)


# One planted violation per rule, and a second for the rules whose failure has
# two distinguishable shapes: a container that does not arrive, and a container
# that arrives and lies.
SYNTHETIC_VIOLATIONS: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    ("pinned-literals-readable", drop_pin_block, "pinned source literals are missing"),
    ("catalogue-complete", remove_a_container, "listed container could not be loaded"),
    ("catalogue-complete", truncate_array_file, "listed container could not be loaded"),
    ("stream-literals-agree", one_ulp, "disagrees with the pinned literal"),
    ("stream-literals-agree", transpose_first_row, "disagrees with the pinned literal"),
    ("payload-block-honest", claim_generated, "payload reads 'generated'"),
    ("payload-block-honest", mislabel_derivation, "is not what the container is named"),
    ("splits-are-one-design", split_the_recipe, "the two halves record different recipes"),
    (
        "splits-are-one-design",
        duplicate_the_first_row,
        "both halves decode the same first row",
    ),
)


def self_test() -> None:
    covered = {name for name, _, _ in SYNTHETIC_VIOLATIONS}
    declared = {name for name, _ in RULES}
    assert covered == declared, (
        f"every rule needs a planted violation: missing={sorted(declared - covered)}, "
        f"stale={sorted(covered - declared)}"
    )

    # The live pins have to be readable for the real run to mean anything, and
    # reading them here rather than only in the run itself is what makes a
    # renamed test or a reshaped literal block fail before any container is
    # generated.
    assert read_pins(ROOT) is not None, (
        f"the live {PIN_SOURCE} carries no single frozen `Source::Sampled` "
        "design block, so the exchange gate would compare against nothing"
    )

    with tempfile.TemporaryDirectory() as workspace:
        base = Path(workspace)
        root, listed = write_clean_catalogue(base / "clean")
        found = violations(build(root / "containers", listed, root))
        assert found == [], f"synthetic clean catalogue reported violations: {found}"

        for index, (name, mutate, expected) in enumerate(SYNTHETIC_VIOLATIONS):
            planted = base / f"planted-{index}-{name}"
            planted.mkdir()
            root, listed = write_clean_catalogue(planted)
            mutate(root)
            found = violations(build(root / "containers", listed, root))
            assert any(expected in item for item in found), (
                f"rule {name} did not fire against its planted violation "
                f"{mutate.__name__}; reported {found}"
            )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print(
            "exchange identity verifier self-test passed "
            f"({len(RULES)} rules, each proven against a planted violation; "
            f"{len(SYNTHETIC_VIOLATIONS)} plants in all, including a value "
            "changed under a recomputed digest — the mismatch no integrity "
            "check can see)"
        )
        return 0
    if sys.argv[1:]:
        print("usage: check_exchange_identity.py [--self-test]", file=sys.stderr)
        return 2

    if shutil.which("cargo") is None:
        print(
            "exchange identity check: cargo is not on PATH, and the generator is "
            "the crate. Nothing here can be checked without it.",
            file=sys.stderr,
        )
        return 1

    catalogue = materialized(ROOT)
    found = violations(catalogue)
    if found:
        print("exchange identity check failed:", file=sys.stderr)
        for item in found:
            print(f"- {item}", file=sys.stderr)
        return 1

    pins = catalogue.pins
    assert pins is not None, "a readable pin block is a rule above"
    print(
        f"exchange identity: {len(catalogue.containers)} regenerated containers "
        f"read back through {Path(sys.executable).name} agree with the "
        f"{len(pins.values)} literals {PIN_SOURCE} pins at stream state "
        f"{pins.state}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
