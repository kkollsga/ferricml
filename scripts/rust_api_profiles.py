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

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "api-baselines" / "rust-api-profiles.json"
STAMP_PATH = ROOT / "target" / "rust-api-profiles.stamp"
VALID_CLASSIFICATIONS = {"profile-root", "public-api", "implementation-only"}


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
    for path in [MANIFEST_PATH, ROOT / "Cargo.toml"]:
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
        "-sss",
        "--no-default-features",
    ]
    if profile.get("all_features"):
        command.append("--all-features")
    return command


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
            if expected != current:
                failed = True
                sys.stdout.writelines(
                    difflib.unified_diff(
                        expected.splitlines(keepends=True),
                        current.splitlines(keepends=True),
                        fromfile=profile["baseline"],
                        tofile=f"current:{name}",
                    )
                )
            else:
                print(f"{name}: exact API match")
        else:
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(current, encoding="utf-8")
            print(f"{name}: wrote {profile['baseline']}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
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
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Rust API profile error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
