#!/usr/bin/env python3
"""Run the opt-in forest contract and retain raw evidence outside git."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce", action="store_true", help="enforce limits on the reference runner")
    parser.add_argument("--runner-id")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--warm-up-time", type=float, default=1.0)
    parser.add_argument("--measurement-time", type=float, default=2.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = root / "dev-docs" / "bench" / "out" / f"{stamp}-forest-contract"
    evidence.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "timestamp_utc": stamp,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "git_commit": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "logical_cpus": os.cpu_count(),
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }
    (evidence / "machine.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    bench = [
        "cargo", "bench", "--manifest-path", str(root / "benchmarks/alternatives/Cargo.toml"),
        "--bench", "forest_contract", "--", "--noplot", "--sample-size", str(args.sample_size),
        "--warm-up-time", str(args.warm_up_time), "--measurement-time", str(args.measurement_time),
    ]
    run = subprocess.run(bench, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (evidence / "cargo-bench.txt").write_text(run.stdout)
    print(run.stdout, end="")
    if run.returncode:
        return run.returncode
    criterion_dir = root / "benchmarks" / "alternatives" / "target" / "criterion"
    # Retain Criterion's machine-readable samples/estimates with the console
    # log. The directory is gitignored evidence, not a committed baseline.
    for source in criterion_dir.rglob("*"):
        if source.is_file() and source.name in {
            "benchmark.json",
            "estimates.json",
            "sample.json",
            "tukey.json",
        }:
            destination = evidence / "criterion" / source.relative_to(criterion_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    evaluate = [
        sys.executable, str(root / "scripts/evaluate_forest_performance.py"),
        "--criterion-dir", str(criterion_dir),
        "--contract", str(root / "benchmarks/forest-performance-contract.json"),
        "--output", str(evidence / "evaluation.json"),
    ]
    if args.enforce:
        evaluate.extend(["--enforce", "--runner-id", args.runner_id or ""])
    result = subprocess.run(evaluate, cwd=root)
    print(f"evidence: {evidence}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
