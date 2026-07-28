"""Read FerricML's synthetic dataset containers from Python.

FerricML generates and Python reads. That direction is the whole point: the
conformance suite used to carry a hand-mirrored copy of the crate's SplitMix64
generator and of every quality lane's label expression, kept byte-identical to
the Rust original by inspection, and a mirror that drifts by one rounding step
changes every design matrix while every aggregate check still passes. There is
now one generator, and this module reads what it wrote.

A container is two files sharing a stem: ``<name>.manifest.json`` is text — the
recipe in full, its digests, the determinism envelope, what the arrays *are*,
and a table saying where each of them sits — and ``<name>.bin`` is those arrays
concatenated little-endian with no header of its own. That split is why this
module needs NumPy and nothing else: the manifest opens with ``json.load`` and
every array is a ``numpy.memmap`` over a byte range of the second file.

Nothing is copied. :func:`load` maps each array at its recorded offset, so
opening a container costs the manifest text and a handful of ``mmap`` calls
whatever the arrays weigh, and an array a caller never touches is never read.

Two refusals mirror the Rust reader exactly, because the reasons for them do not
change with the language:

* a container whose array file does not hash to its recorded digest, or whose
  table does not describe that file exactly, is refused rather than partly
  believed; and
* a **derived** container — one whose arrays are not its recipe's own output —
  refuses to be treated as regenerable from that recipe. Both halves of a frozen
  reference lane record the digest of the single design they were cut out of, so
  the digest cannot tell them apart from that design and the ``payload`` block
  is what does.

Typical use::

    from ferricml_datasets import generate, load

    lane = generate("reference_nonlinear_11_train", suite="reference")
    model.fit(lane.features, lane.target)

    grid = load("target/ferricml-datasets/accuracy_linear-regression")
    beta = grid.truth["coefficients"]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

__all__ = [
    "Container",
    "ContainerError",
    "Derivation",
    "NotRegenerable",
    "FORMAT_VERSION",
    "default_directory",
    "generate",
    "load",
    "materialize",
    "repository_root",
]

#: Container format this reader accepts, and the one the crate's
#: ``manifest.rs`` writes. Version 1 had no ``payload`` block, so a reader
#: meeting one would have to *assume* its arrays were its recipe's output —
#: which is the assumption the block exists to prevent — and refusing is the
#: only honest answer.
FORMAT_VERSION = 2

#: How the manifest's ``dtype`` strings map onto NumPy. Little-endian is
#: explicit rather than native: the array file is written little-endian on every
#: platform, so a big-endian reader must byte-swap rather than agree by accident.
_DTYPES = {"f32": np.dtype("<f4"), "u8": np.dtype("u1"), "u64": np.dtype("<u8")}

_ENV_ROOT = "FERRICML_ROOT"
_ENV_DIRECTORY = "FERRICML_DATASETS"


class ContainerError(Exception):
    """A container is not one this reader will believe."""


class NotRegenerable(ContainerError):
    """A derived container was asked for as though its recipe produced it.

    Raised by :meth:`Container.regenerable_recipe`. The container still loads
    and its arrays are still readable — what is refused is the *claim* that
    running its recipe through the generator would reproduce them.
    """


@dataclass(frozen=True)
class Derivation:
    """Which recorded dataset a derived container holds.

    ``kind`` is always ``"reference-split"`` today, and is carried rather than
    assumed for the same reason the crate carries it: a second kind of
    derivation must be distinguishable from the first without redefining what
    the surrounding block means.
    """

    kind: str
    lane: str
    seed: int
    split: str


class Container:
    """One materialized dataset, mapped rather than read.

    Attribute access is by array name — the names are the contract, not the
    order — with the three shapes every consumer actually wants promoted to
    properties. Every array is two-dimensional, including the vectors: a target
    is ``rows x 1`` and a coefficient vector is ``1 x columns``, so a consumer
    can reshape without knowing which family produced the container.
    """

    def __init__(self, path: Path, manifest: Mapping[str, Any], arrays: dict[str, np.ndarray]):
        self.path = path
        self.name = path.name
        self._manifest = manifest
        self._arrays = arrays

    # -- what the container is ------------------------------------------------

    @property
    def recipe(self) -> Mapping[str, Any]:
        """The recipe recorded in the manifest, as plain nested dictionaries.

        Provenance for a derived container and a reproduction instruction for a
        generated one. :meth:`regenerable_recipe` is the accessor that
        distinguishes the two.
        """
        return self._manifest["recipe"]

    @property
    def spec_digest(self) -> str:
        """SHA-256 of the recipe, as the crate computed it."""
        return self._manifest["spec_digest"]

    @property
    def data_digest(self) -> str:
        """SHA-256 of the array file, verified when the container was loaded."""
        return self._manifest["data"]["digest"]

    @property
    def portability(self) -> str:
        """``"bit-exact"`` or ``"per-runner"``.

        A per-runner container is no less usable — the bytes are the bytes —
        but regenerating its recipe elsewhere may not reproduce them, which is
        exactly when a materialized file is worth having rather than a recipe.
        """
        return self._manifest["portability"]

    @property
    def payload(self) -> str:
        """``"generated"`` or ``"derived"``."""
        return self._manifest["payload"]["kind"]

    @property
    def derivation(self) -> Derivation | None:
        """What a derived container holds, or ``None`` when it is generated."""
        block = self._manifest["payload"]
        if block["kind"] != "derived":
            return None
        return Derivation(
            kind=block["derivation"],
            lane=block["lane"],
            seed=int(block["seed"]),
            split=block["split"],
        )

    def regenerable_recipe(self) -> Mapping[str, Any]:
        """The recipe, when running it really would reproduce these arrays.

        Raises :class:`NotRegenerable` otherwise. This is the refusal the
        ``payload`` block exists for, and it has teeth precisely because the
        digest does not: a reference lane's training split records the digest of
        the 1152-row design it was cut from, so a caller keying on
        :attr:`spec_digest` alone would regenerate that design, get 1152
        untargeted rows back, and find every digest it checked agreeing.
        """
        derivation = self.derivation
        if derivation is not None:
            raise NotRegenerable(
                f"{self.name} holds the {derivation.split} split of the "
                f"{derivation.lane} reference lane at seed {derivation.seed}, "
                "which its recipe does not produce"
            )
        return self.recipe

    # -- the arrays -----------------------------------------------------------

    def array(self, name: str) -> np.ndarray:
        """The named array, two-dimensional and read-only."""
        try:
            return self._arrays[name]
        except KeyError:
            available = ", ".join(sorted(self._arrays)) or "none"
            raise KeyError(f"{self.name} has no array {name!r}; it has: {available}") from None

    def vector(self, name: str) -> np.ndarray:
        """The named array flattened, refusing anything wider than one column.

        A view rather than a copy — a ``rows x 1`` array is contiguous — and a
        refusal rather than a silent ravel, because flattening a genuinely
        two-dimensional array would hand a caller a shape it did not ask for.
        """
        values = self.array(name)
        if values.shape[1] != 1:
            raise ContainerError(f"{name} is {values.shape[0]}x{values.shape[1]}, not a vector")
        return values.reshape(-1)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.array(name)

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        """Every array, keyed by name, in the order the file lays them out."""
        return dict(self._arrays)

    @property
    def features(self) -> np.ndarray:
        """The design matrix, ``rows x columns``."""
        return self.array("features")

    @property
    def target(self) -> np.ndarray:
        """The target as a one-dimensional array.

        ``uint8`` for a classification family and ``float32`` for a regression
        one, which is the same distinction the crate's ``Target`` draws. A
        family with no target — a clustered design — has no such array, and
        asking for it raises.
        """
        return self.vector("target")

    @property
    def weights(self) -> np.ndarray:
        """Per-row sample weights, when the recipe asked for a pattern."""
        return self.vector("weights")

    @property
    def groups(self) -> np.ndarray:
        """Per-row group labels, when the recipe or the task assigned them."""
        return self.vector("groups")

    @property
    def truth(self) -> Mapping[str, np.ndarray]:
        """What the family recorded about the right answer, by short name.

        The ``truth_`` prefix is stripped, so ``truth_coefficients`` is
        ``truth["coefficients"]``. Scalars the family knows — the intercept, the
        rank, the class count — are one-value arrays rather than manifest
        fields, so this mapping has exactly one kind of value and a caller has
        exactly one code path.
        """
        return {
            name[len("truth_") :]: values
            for name, values in self._arrays.items()
            if name.startswith("truth_")
        }

    def __repr__(self) -> str:
        return (
            f"<Container {self.name!r} {self.payload} "
            f"{len(self._arrays)} arrays spec={self.spec_digest[:12]}>"
        )


def load(path: str | os.PathLike[str], *, verify: bool = True) -> Container:
    """Opens a container, mapping its arrays without copying them.

    ``path`` names the container's stem, with or without the
    ``.manifest.json`` suffix.

    Every claim the container makes is checked before any of it is believed:
    the format version, the array file's length and digest, and the array
    table against the file it describes — contiguous, in order, uniquely named,
    ending on the last byte. That is the same set the crate's own reader
    enforces, and for the same reason: a container is a file on disk, and a
    file on disk is untrusted input.

    ``verify=False`` skips only the digest, which is the one check that has to
    read the whole array file. Everything structural still runs. Use it when
    the container was just written by a trusted local run and the read is in a
    loop.
    """
    manifest_path = _manifest_path(Path(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise ContainerError(
            f"{manifest_path} declares container format {manifest.get('format')!r}, "
            f"and this reader accepts {FORMAT_VERSION}"
        )

    data_path = manifest_path.parent / manifest["data"]["file"]
    declared = int(manifest["data"]["bytes"])
    actual = data_path.stat().st_size
    if actual != declared:
        raise ContainerError(f"{data_path} is {actual} bytes; the manifest declares {declared}")
    if verify:
        with data_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest["data"]["digest"]:
            raise ContainerError(f"{data_path} does not hash to the digest recorded for it")

    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for record in manifest["arrays"]:
        name, rows, columns = record["name"], int(record["rows"]), int(record["columns"])
        length, start = int(record["len"]), int(record["byte_offset"])
        if name in arrays:
            raise ContainerError(f"{manifest_path} names the array {name!r} twice")
        if rows * columns != length:
            raise ContainerError(f"{name} declares {rows}x{columns} values but a length of {length}")
        if start != offset:
            raise ContainerError(f"{name} starts at byte {start}; the previous array ends at {offset}")
        dtype = _DTYPES.get(record["dtype"])
        if dtype is None:
            raise ContainerError(f"{name} declares the unknown dtype {record['dtype']!r}")
        # One mapping per array, at its own offset. `np.memmap` maps from the
        # enclosing page and indexes into it, so an array that does not begin
        # on a page — or on its own alignment — still costs no copy.
        values = np.memmap(data_path, dtype=dtype, mode="r", offset=start, shape=(rows, columns))
        arrays[name] = values
        offset += length * dtype.itemsize
    if offset != declared:
        raise ContainerError(
            f"the array table covers {offset} of {declared} bytes of {data_path}"
        )
    stem = manifest_path.with_name(manifest_path.name.removesuffix(".manifest.json"))
    return Container(stem, manifest, arrays)


def materialize(
    *,
    suite: str = "all",
    name: str | None = None,
    out: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> Path:
    """Runs ``ferricml-datagen`` and returns the directory it wrote into.

    The generator is the crate, invoked through ``cargo run --release``, because
    there is deliberately no second implementation of it here. The release
    profile is not a preference: a debug build of the performance grid is slow
    enough to be mistaken for a hang.

    The binary is itself a cache — an entry whose container already records the
    same recipe is read back rather than regenerated — so calling this
    repeatedly costs one ``cargo`` invocation and a directory scan.
    """
    root = repository_root(root)
    directory = Path(out) if out is not None else default_directory(root)
    command = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "--features",
        "datasets",
        "--bin",
        "ferricml-datagen",
        "--",
        "--out",
        str(directory),
        "--suite",
        suite,
    ]
    if name is not None:
        command += ["--name", name]
    if force:
        command.append("--force")
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ContainerError(
            f"ferricml-datagen exited {completed.returncode}\n{completed.stderr.strip()}"
        )
    return directory


def generate(
    name: str,
    *,
    suite: str = "all",
    out: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
    verify: bool = True,
) -> Container:
    """Returns one catalogue entry's container, materializing it if absent.

    The cache is the container itself: a readable one under this name is
    returned without starting a subprocess at all. A miss materializes the whole
    *suite* rather than the one entry, because the cost of a run is dominated by
    ``cargo``'s own startup and the crate's cache then skips every entry already
    current — so a harness reading fifty lanes pays for one invocation instead
    of fifty.
    """
    root = repository_root(root)
    directory = Path(out) if out is not None else default_directory(root)
    if not force:
        try:
            return load(directory / name, verify=verify)
        except (OSError, ContainerError, json.JSONDecodeError, KeyError):
            pass
    materialize(suite=suite, out=directory, root=root, force=force)
    return load(directory / name, verify=verify)


def repository_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The FerricML checkout to generate from.

    ``FERRICML_ROOT`` wins, then an explicit argument, then the nearest ancestor
    of this file that holds a ``Cargo.toml``. The environment variable is first
    on purpose and the path is left **unresolved**: ``dev-docs/`` is a symlink
    into the primary checkout from every parallel worktree, so resolving a
    script's own path lands in the primary checkout no matter who invoked it —
    which once wrote a regenerated fixture into the wrong tree.
    """
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).expanduser()
    if explicit is not None:
        return Path(explicit)
    for candidate in Path(__file__).parents:
        if (candidate / "Cargo.toml").is_file():
            return candidate
    raise ContainerError(
        f"no Cargo.toml above {__file__}; set {_ENV_ROOT} or pass root=..."
    )


def default_directory(root: str | os.PathLike[str]) -> Path:
    """Where containers live when a caller names no directory.

    Under ``target/`` so generated data is already ignored by the repository and
    is removed by ``cargo clean`` along with everything else that was built.
    """
    override = os.environ.get(_ENV_DIRECTORY)
    if override:
        return Path(override).expanduser()
    return Path(root) / "target" / "ferricml-datasets"


def _manifest_path(path: Path) -> Path:
    if path.name.endswith(".manifest.json"):
        return path
    return path.with_name(path.name + ".manifest.json")
