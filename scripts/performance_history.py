#!/usr/bin/env python3
"""Capture and compare FerricML-only Criterion release history.

The immutable summaries live in the gitignored development workspace.  They
are intentionally independent of the opt-in third-party benchmark package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / "dev-docs" / "bench" / "results" / "releases"
DEFAULT_OUT = ROOT / "dev-docs" / "bench" / "out"
RUNNER_CONFIG = ROOT / "dev-docs" / "bench" / "runner.json"
PROTOCOL = "forest-history-v1"
FIT_LIMIT = 1.15
INFERENCE_LIMIT = 1.10

FIT = "forest_historical_fit_2048x64_20t/ferricml"
INFERENCE = tuple(
    f"forest_historical_into_{rows}x64_100t/{operation}"
    for rows in (1, 32, 1024)
    for operation in ("labels", "full_proba", "class_proba")
)


def output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def cargo_version() -> str:
    metadata = json.loads(output(["cargo", "metadata", "--no-deps", "--format-version", "1"]))
    matches = [item["version"] for item in metadata["packages"] if item["name"] == "ferricml"]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one FerricML package")
    return matches[0]


def version_key(version: str) -> tuple[int, int, int]:
    core = version.split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"unsupported release version: {version}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def runner_id(argument: str | None) -> str:
    if argument:
        return argument
    environment = os.environ.get("FERRICML_PERF_RUNNER_ID")
    if environment:
        return environment
    if RUNNER_CONFIG.exists():
        configured = json.loads(RUNNER_CONFIG.read_text()).get("runner_id")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
    raise RuntimeError(
        "stable runner identity is required; pass --runner-id, set "
        "FERRICML_PERF_RUNNER_ID, or create dev-docs/bench/runner.json"
    )


def active_build_processes() -> list[str]:
    listing = subprocess.run(
        ["ps", "-Ao", "pid=,comm="], check=True, text=True, capture_output=True
    ).stdout
    active = []
    for line in listing.splitlines():
        executable = Path(line.strip().split(maxsplit=1)[-1]).name.lower()
        if executable in {"cargo", "rustc", "criterion"} or executable.startswith("forest-"):
            active.append(line.strip())
    return active


def cpu_idle_percent() -> float:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["top", "-l", "2", "-n", "0", "-s", "1"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        lines = [line for line in result.splitlines() if line.startswith("CPU usage:")]
        match = re.search(r"([0-9.]+)% idle", lines[-1] if lines else "")
        if not match:
            raise RuntimeError("could not read macOS CPU idle percentage")
        return float(match.group(1))
    if platform.system() == "Linux":
        def counters() -> tuple[int, int]:
            fields = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
            return sum(fields), fields[3] + (fields[4] if len(fields) > 4 else 0)

        total_before, idle_before = counters()
        time.sleep(1)
        total_after, idle_after = counters()
        return 100.0 * (idle_after - idle_before) / (total_after - total_before)
    raise RuntimeError(f"idle verification is unsupported on {platform.system()}")


def idle_evidence(samples: int, minimum: float) -> list[float]:
    active = active_build_processes()
    if active:
        raise RuntimeError(f"active build or benchmark processes: {active}")
    measured = [cpu_idle_percent() for _ in range(samples)]
    if min(measured) < minimum:
        raise RuntimeError(
            f"runner is not idle enough: samples={measured}, required each >= {minimum:.1f}%"
        )
    return measured


def estimate(criterion: Path, benchmark: str) -> float:
    path = criterion.joinpath(*benchmark.split("/"), "new", "estimates.json")
    try:
        payload = json.loads(path.read_text())
        return float(payload["median"]["point_estimate"])
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"missing Criterion median for {benchmark}: {path}") from error


def load_history(history_dir: Path, current: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for path in history_dir.glob("v*.json"):
        try:
            candidate = json.loads(path.read_text())
            version_key(candidate["version"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            candidate.get("protocol") == current["protocol"]
            and candidate.get("runner", {}).get("id") == current["runner"]["id"]
            and version_key(candidate["version"]) < version_key(current["version"])
        ):
            records.append(candidate)
    return sorted(records, key=lambda item: version_key(item["version"]))


def compare_one(current: dict[str, Any], reference: dict[str, Any], kind: str) -> dict[str, Any]:
    current_metrics = current["metrics"]
    reference_metrics = reference["metrics"]
    fit_ratio = current_metrics[FIT] / reference_metrics[FIT]
    inference_ratios = {
        name: current_metrics[name] / reference_metrics[name] for name in INFERENCE
    }
    failures = [
        {"benchmark": name, "ratio": ratio, "limit": INFERENCE_LIMIT}
        for name, ratio in inference_ratios.items()
        if ratio > INFERENCE_LIMIT
    ]
    if fit_ratio > FIT_LIMIT:
        failures.append({"benchmark": FIT, "ratio": fit_ratio, "limit": FIT_LIMIT})
    return {
        "kind": kind,
        "status": "pass" if not failures else "regression",
        "reference_version": reference["version"],
        "fit_ratio": fit_ratio,
        "maximum_inference_ratio": max(inference_ratios.values()),
        "inference_ratios": inference_ratios,
        "failures": failures,
    }


def comparisons(current: dict[str, Any], history_dir: Path) -> dict[str, Any]:
    history = load_history(history_dir, current)
    if history:
        previous = compare_one(current, history[-1], "previous_release")
    else:
        previous = {
            "kind": "previous_release",
            "status": "insufficient_history",
            "reason": "no earlier release exists for this runner and protocol",
        }
    if len(history) >= 3:
        anchor = compare_one(current, history[-3], "approximately_three_release_anchor")
    else:
        anchor = {
            "kind": "approximately_three_release_anchor",
            "status": "insufficient_history",
            "reason": f"need three earlier releases; found {len(history)}",
        }
    statuses = (previous["status"], anchor["status"])
    if "regression" in statuses:
        verdict = "regression"
    elif "insufficient_history" in statuses:
        verdict = "insufficient_history"
    else:
        verdict = "pass"
    return {
        "thresholds": {"fit_ratio": FIT_LIMIT, "inference_ratio": INFERENCE_LIMIT},
        "previous_release": previous,
        "anchor": anchor,
        "verdict": verdict,
    }


def capture(args: argparse.Namespace) -> int:
    history_dir = Path(args.history_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    version = args.version or cargo_version()
    identity = runner_id(args.runner_id)
    destination = history_dir / f"v{version}.json"
    if destination.exists():
        raise RuntimeError(
            f"immutable release summary already exists: {destination}; "
            "use a new version, never overwrite release history"
        )

    idle_before = idle_evidence(args.idle_samples, args.minimum_idle)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = out_root / f"{stamp}-ferricml-history-v{version}"
    evidence.mkdir(parents=True, exist_ok=False)
    before_load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    command = [
        "cargo",
        "bench",
        "--locked",
        "--bench",
        "forest",
        "--",
        "--noplot",
        "--sample-size",
        str(args.sample_size),
        "--warm-up-time",
        str(args.warm_up_time),
        "--measurement-time",
        str(args.measurement_time),
    ]
    run = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (evidence / "cargo-bench.txt").write_text(run.stdout)
    print(run.stdout, end="")
    if run.returncode:
        return run.returncode

    criterion = ROOT / "target" / "criterion"
    metrics = {FIT: estimate(criterion, FIT)}
    metrics.update({name: estimate(criterion, name) for name in INFERENCE})
    logical_cpus = os.cpu_count()
    after_load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    record: dict[str, Any] = {
        "schema_version": 1,
        "version": version,
        "protocol": PROTOCOL,
        "runner": {
            "id": identity,
            "system": platform.system(),
            "system_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": logical_cpus,
        },
        "toolchain": {
            "rustc": output(["rustc", "--version", "--verbose"]),
            "cargo": output(["cargo", "--version"]),
        },
        "run": {
            "timestamp_utc": stamp,
            "git_commit": output(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(output(["git", "status", "--short", "--untracked-files=no"])),
            "load_average_before": before_load,
            "load_average_after": after_load,
            "cpu_idle_percent_before": idle_before,
            "minimum_idle_percent": args.minimum_idle,
            "sample_size": args.sample_size,
            "warm_up_seconds": args.warm_up_time,
            "measurement_seconds": args.measurement_time,
            "note": args.note,
            "raw_evidence": str(evidence.relative_to(ROOT)),
        },
        "metrics": metrics,
    }
    record["comparison"] = comparisons(record, history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")

    for name in ("benchmark.json", "estimates.json", "sample.json", "tukey.json"):
        for source in criterion.rglob(name):
            if "forest_historical_" not in source.as_posix():
                continue
            copied = evidence / "criterion" / source.relative_to(criterion)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copied)

    verdict = record["comparison"]["verdict"]
    print(f"immutable summary: {destination}")
    print(f"history verdict: {verdict}")
    if verdict == "insufficient_history":
        print("insufficient history is expected until prior and three-release anchors exist")
    return 1 if verdict == "regression" and args.enforce else 0


def check(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text())
    result = comparisons(summary, Path(args.history_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["verdict"] == "regression" and args.enforce else 0


def fixture(version: str, fit: float = 100.0, inference: float = 100.0) -> dict[str, Any]:
    metrics = {FIT: fit}
    metrics.update({name: inference for name in INFERENCE})
    return {
        "schema_version": 1,
        "version": version,
        "protocol": PROTOCOL,
        "runner": {"id": "self-test"},
        "metrics": metrics,
    }


def self_test() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        history = Path(temporary)
        first = fixture("0.1.0")
        result = comparisons(first, history)
        assert result["verdict"] == "insufficient_history"
        assert result["previous_release"]["status"] == "insufficient_history"
        assert result["anchor"]["status"] == "insufficient_history"

        for version in ("0.1.0", "0.2.0", "0.3.0"):
            (history / f"v{version}.json").write_text(json.dumps(fixture(version)))
        fourth = fixture("0.4.0", fit=114.9, inference=109.9)
        result = comparisons(fourth, history)
        assert result["verdict"] == "pass"
        assert result["anchor"]["reference_version"] == "0.1.0"

        regressed = fixture("0.4.0", fit=115.1, inference=110.1)
        result = comparisons(regressed, history)
        assert result["verdict"] == "regression"
        assert result["previous_release"]["status"] == "regression"
        assert result["anchor"]["status"] == "regression"
    print("performance history self-test passed (insufficient, prior, anchor, regression)")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--runner-id")
    capture_parser.add_argument("--version")
    capture_parser.add_argument("--history-dir", default=DEFAULT_HISTORY)
    capture_parser.add_argument("--out-dir", default=DEFAULT_OUT)
    capture_parser.add_argument("--sample-size", type=int, default=20)
    capture_parser.add_argument("--warm-up-time", type=float, default=1.0)
    capture_parser.add_argument("--measurement-time", type=float, default=2.0)
    capture_parser.add_argument("--note", default="")
    capture_parser.add_argument("--idle-samples", type=int, default=3)
    capture_parser.add_argument("--minimum-idle", type=float, default=90.0)
    capture_parser.add_argument("--enforce", action="store_true")
    capture_parser.set_defaults(function=capture)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("summary")
    check_parser.add_argument("--history-dir", default=DEFAULT_HISTORY)
    check_parser.add_argument("--enforce", action="store_true")
    check_parser.set_defaults(function=check)
    subparsers.add_parser("self-test").set_defaults(function=lambda _args: self_test())
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        return args.function(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"performance history error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
