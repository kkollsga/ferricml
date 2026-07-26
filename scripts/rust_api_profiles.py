#!/usr/bin/env python3
"""Validate, capture, and exactly compare FerricML Rust API profiles."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

SELF_PATH = Path(__file__).resolve()
ROOT = SELF_PATH.parents[1]
MANIFEST_PATH = ROOT / "tests" / "api-baselines" / "rust-api-profiles.json"
STAMP_PATH = ROOT / "target" / "rust-api-profiles.stamp"
VALID_CLASSIFICATIONS = {"profile-root", "public-api", "implementation-only"}

# Noise `cargo public-api` may omit. Blanket impls (`impl<T> Any for T`) and
# auto-trait impls (`impl Send for …`) say nothing a reviewer can act on.
#
# `auto-derived-impls` is deliberately **not** among them. Omitting it (the
# third `-s`, which this profile used until the derive blind spot was found)
# hides every `#[derive(Clone)]`, `#[derive(Debug)]` and `#[derive(PartialEq)]`
# in the crate, so dropping a derive from a public type is a semver-breaking
# change the exact API check reports as clean.
OMITTED_NOISE = ("blanket-impls", "auto-trait-impls")

# Derived impls the profile must be able to see, as `cargo public-api` spells
# them. Every public `Params` type in the crate is `Clone + Debug + PartialEq`
# by derive, so a capture showing none of these is not a clean surface — it is
# a blind one, and the exact comparison below would then pass vacuously.
REQUIRED_DERIVED_TRAITS = (
    "core::clone::Clone",
    "core::cmp::PartialEq",
    "core::fmt::Debug",
)


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Rust API profile schema")
    return manifest


def cargo_features(package: str) -> set[str]:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    crate = next((item for item in metadata["packages"] if item["name"] == package), None)
    if crate is None:
        raise ValueError(f"package {package!r} does not exist")
    return set(crate["features"])


def validate_manifest(manifest: dict[str, Any]) -> None:
    for key in ("package", "nightly", "cargo_public_api_version"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest {key} must be a non-empty string")

    classifications = manifest.get("feature_classifications")
    if not isinstance(classifications, dict):
        raise ValueError("feature_classifications must be an object")
    unknown = set(classifications.values()) - VALID_CLASSIFICATIONS
    if unknown:
        raise ValueError(f"unknown feature classifications: {sorted(unknown)}")
    declared = cargo_features(manifest["package"])
    classified = set(classifications)
    if declared != classified:
        raise ValueError(
            f"Cargo feature classification drift: missing={sorted(declared - classified)}, "
            f"stale={sorted(classified - declared)}"
        )
    if classifications.get("default") != "profile-root":
        raise ValueError("default must be classified as profile-root")

    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError("profiles must contain exactly the default profile")
    by_name = {profile.get("name"): profile for profile in profiles}
    if set(by_name) != {"default"}:
        raise ValueError("profile must be named default")
    if by_name["default"].get("features") != [] or by_name["default"].get("all_features", False):
        raise ValueError("default profile must disable every optional feature")

    baselines: set[str] = set()
    for profile in profiles:
        baseline = profile.get("baseline")
        if (
            not isinstance(baseline, str)
            or not baseline.startswith("tests/api-baselines/rust/")
            or baseline in baselines
        ):
            raise ValueError(f"invalid or duplicate baseline: {baseline!r}")
        baselines.add(baseline)


def source_digest(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    # This script is a content input too: it decides what the capture contains,
    # so a changed capture command must invalidate the stamp exactly as a
    # changed source file does.
    for path in [SELF_PATH, MANIFEST_PATH, ROOT / "Cargo.toml"]:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    for path in sorted((ROOT / "src").rglob("*.rs")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    for profile in sorted(manifest["profiles"], key=lambda item: item["baseline"]):
        path = ROOT / profile["baseline"]
        digest.update(profile["baseline"].encode())
        digest.update(path.read_bytes() if path.exists() else b"<absent>")
    return digest.hexdigest()


def public_api_command(manifest: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    command = [
        "cargo",
        "public-api",
        "-p",
        manifest["package"],
        "--no-default-features",
    ]
    for omitted in OMITTED_NOISE:
        command += ["--omit", omitted]
    if profile.get("all_features"):
        command.append("--all-features")
    return command


def impl_traits(profile: str) -> set[str]:
    """Every trait the profile records an `impl … for …` row for."""
    traits: set[str] = set()
    for line in profile.splitlines():
        if not line.startswith("impl"):
            continue
        head, separator, _ = line.partition(" for ")
        if not separator:
            continue
        traits.add(head.rsplit(" ", 1)[-1])
    return traits


def missing_derived_traits(profile: str) -> list[str]:
    """Which required derived impls the capture failed to record.

    A non-empty result means the profile went blind rather than that the crate
    changed, so it is reported as a tool failure instead of an API diff.
    """
    recorded = impl_traits(profile)
    return [trait for trait in REQUIRED_DERIVED_TRAITS if trait not in recorded]


def differences(expected: str, current: str, *, fromfile: str, tofile: str) -> list[str]:
    """The unified diff between a baseline and a capture; empty when equal."""
    if expected == current:
        return []
    return list(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def capture(manifest: dict[str, Any], profile: dict[str, Any]) -> str:
    result = subprocess.run(
        public_api_command(manifest, profile),
        cwd=ROOT,
        env={**os.environ, "RUSTUP_TOOLCHAIN": manifest["nightly"]},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"{profile['name']} API capture failed:\n{details}")
    missing = missing_derived_traits(result.stdout)
    if missing:
        raise RuntimeError(
            f"{profile['name']} API capture records no impls for {missing}, so derived "
            "impls are invisible to it and removing a derive from a public type would "
            "compare clean. Restore derived impls to the capture before trusting it."
        )
    return result.stdout


def verify_tool(manifest: dict[str, Any]) -> None:
    if shutil.which("cargo-public-api") is None:
        raise RuntimeError(
            "cargo-public-api is not installed; install the manifest-pinned version with "
            "`cargo +stable install cargo-public-api --locked --version "
            f"{manifest['cargo_public_api_version']}`"
        )
    version = subprocess.run(
        ["cargo", "public-api", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = f"cargo-public-api {manifest['cargo_public_api_version']}"
    if version != expected:
        raise RuntimeError(f"expected {expected!r}, found {version!r}")


def run_profiles(manifest: dict[str, Any], *, check: bool) -> int:
    verify_tool(manifest)
    failed = False
    for profile in manifest["profiles"]:
        name = profile["name"]
        print(f"{name}: capturing", flush=True)
        current = capture(manifest, profile)
        baseline = ROOT / profile["baseline"]
        if check:
            expected = baseline.read_text(encoding="utf-8") if baseline.exists() else ""
            diff = differences(
                expected,
                current,
                fromfile=profile["baseline"],
                tofile=f"current:{name}",
            )
            if diff:
                failed = True
                sys.stdout.writelines(diff)
            else:
                print(f"{name}: exact API match")
        else:
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(current, encoding="utf-8")
            print(f"{name}: wrote {profile['baseline']}")
    return 1 if failed else 0


def frozen_profiles(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """Every checked-in baseline, as `(relative path, text)`."""
    frozen = []
    for profile in manifest["profiles"]:
        path = ROOT / profile["baseline"]
        if not path.exists():
            raise RuntimeError(f"{profile['baseline']} is absent; run `make api-refresh`")
        frozen.append((profile["baseline"], path.read_text(encoding="utf-8")))
    return frozen


def strip_derived_rows(profile: str, trait: str) -> str:
    """The profile with every `impl <trait> for …` row and its members removed.

    `cargo public-api` prints an impl's members directly beneath it, so a
    trailing `pub fn` run belongs to the impl above it. Anything else ends the
    run; keeping a stray member row would only make the synthetic violation
    smaller, never make it pass.
    """
    kept: list[str] = []
    inside = False
    for line in profile.splitlines(keepends=True):
        if line.startswith("impl"):
            head, separator, _ = line.partition(" for ")
            inside = bool(separator) and head.rsplit(" ", 1)[-1] == trait
        elif inside and not line.startswith("pub fn "):
            inside = False
        if not inside:
            kept.append(line)
    return "".join(kept)


def self_test(manifest: dict[str, Any]) -> int:
    """Prove the derive detector can fail, against synthetic violations.

    An exact-comparison contract that has never been shown to fire proves only
    that the tree is currently clean. These checks are cheap, need no pinned
    tool, and run in the ordinary gate, so a capture command that silently went
    blind again fails here rather than years later.
    """
    command = public_api_command(manifest, manifest["profiles"][0])
    omitted = {value for flag, value in zip(command, command[1:]) if flag == "--omit"}
    assert "auto-derived-impls" not in omitted, (
        "the capture omits auto-derived impls, so removing a derive from a public "
        "type would compare clean"
    )
    assert not any(
        argument.startswith("-s") and set(argument[1:]) == {"s"} for argument in command
    ), "`-sss` omits auto-derived impls; spell the omissions out instead"

    parsed = impl_traits(
        "impl core::clone::Clone for ferricml::linear_model::Ridge\n"
        "impl<'a> core::marker::Copy for ferricml::api::AnyRegressorParams<'a>\n"
        "impl<S, E> ferricml::api::HasCapabilities for ferricml::pipeline::Pipeline<S, E>\n"
        "pub fn ferricml::linear_model::Ridge::clone(&self) -> ferricml::linear_model::Ridge\n"
    )
    assert parsed == {
        "core::clone::Clone",
        "core::marker::Copy",
        "ferricml::api::HasCapabilities",
    }, f"impl rows parsed as {sorted(parsed)}"

    assert missing_derived_traits("") == list(REQUIRED_DERIVED_TRAITS), (
        "an empty capture must report every required derived impl as missing"
    )

    for name, frozen in frozen_profiles(manifest):
        missing = missing_derived_traits(frozen)
        assert not missing, (
            f"{name} records no impls for {missing}; the frozen baseline is blind to "
            "derived impls, so nothing change-detects them"
        )

        assert not differences(frozen, frozen, fromfile=name, tofile=name), (
            "identical profiles must compare equal, or the removal proof below is "
            "true of everything"
        )

        for trait in REQUIRED_DERIVED_TRAITS:
            stripped = strip_derived_rows(frozen, trait)
            assert stripped != frozen, f"{name} has no `impl {trait} for …` row to remove"
            assert missing_derived_traits(stripped) == [trait], (
                f"stripping {trait} from {name} must leave exactly it missing"
            )
            # The synthetic violation this whole mechanism exists for: a public
            # type that lost its derive. It must surface as an API diff.
            diff = differences(stripped, frozen, fromfile=name, tofile="current")
            assert diff, (
                f"removing every `impl {trait} for …` row from {name} produced no "
                "API diff, so dropping a derive from a public type is invisible"
            )

    print(
        "Rust API profile self-test: derived impls are captured, and removing one "
        f"fails the exact comparison ({len(REQUIRED_DERIVED_TRAITS)} traits proven)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("self-test")
    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--skip-if-unchanged", action="store_true")
    subparsers.add_parser("check")
    value = subparsers.add_parser("value")
    value.add_argument("key", choices=["nightly", "cargo_public_api_version"])
    args = parser.parse_args()

    try:
        manifest = load_manifest()
        validate_manifest(manifest)
        if args.command == "value":
            print(manifest[args.key])
            return 0
        if args.command == "validate":
            print("Rust API profiles valid: default; 1 classified feature")
            return 0
        if args.command == "self-test":
            return self_test(manifest)
        if args.command == "refresh" and args.skip_if_unchanged:
            digest = source_digest(manifest)
            if STAMP_PATH.exists() and STAMP_PATH.read_text(encoding="utf-8").strip() == digest:
                print("Rust API snapshots: complete content hash unchanged; refresh skipped")
                return 0
        result = run_profiles(manifest, check=args.command == "check")
        if args.command == "refresh" and result == 0:
            STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
            STAMP_PATH.write_text(source_digest(manifest) + "\n", encoding="utf-8")
        return result
    except AssertionError as error:
        print(f"Rust API profile self-test failed: {error}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Rust API profile error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
