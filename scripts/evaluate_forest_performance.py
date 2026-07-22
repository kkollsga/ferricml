#!/usr/bin/env python3
"""Evaluate Criterion medians against FerricML's frozen forest contract."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path


def median_ns(criterion_dir: Path, benchmark: str) -> float:
    path = criterion_dir.joinpath(*benchmark.split("/"), "new", "estimates.json")
    try:
        payload = json.loads(path.read_text())
        value = float(payload["median"]["point_estimate"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Criterion median for {benchmark} at {path}: {error}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid median for {benchmark}: {value}")
    return value


def evaluate(contract: dict, criterion_dir: Path) -> dict:
    limits = contract["limits"]
    fit = contract["fit"]
    fit_ferric = median_ns(criterion_dir, fit["ferricml"])
    fit_rafor = median_ns(criterion_dir, fit["rafor"])
    fit_ratio = fit_ferric / fit_rafor
    fit_baseline = float(fit["historical_baseline_ns"])
    if fit_baseline <= 0:
        raise ValueError("fit historical_baseline_ns must be recorded and positive")

    suite_results = []
    all_inference_ratios = []
    for suite in contract["inference_suites"]:
        lanes = []
        for lane in suite["lanes"]:
            ferric = median_ns(criterion_dir, lane["ferricml"])
            rafor = median_ns(criterion_dir, lane["rafor"])
            ratio = ferric / rafor
            all_inference_ratios.append(ratio)
            lanes.append({**lane, "ferricml_ns": ferric, "rafor_ns": rafor, "ratio": ratio})
        suite_results.append({
            "operation": suite["operation"],
            "allocations": suite["allocations"],
            "geomean_ratio": math.prod(lane["ratio"] for lane in lanes) ** (1 / len(lanes)),
            "lanes": lanes,
        })
    inference_geomean = math.prod(all_inference_ratios) ** (1 / len(all_inference_ratios))

    historical = []
    for lane in contract["historical_inference"]:
        baseline = float(lane["baseline_ns"])
        if baseline <= 0:
            raise ValueError(f"historical baseline for {lane['name']} must be positive")
        current = median_ns(criterion_dir, lane["benchmark"])
        historical.append({**lane, "current_ns": current, "ratio": current / baseline})

    checks = {
        "rafor_fit": fit_ratio <= float(limits["rafor_fit_ratio"]),
        "rafor_inference_lanes": all(
            lane["ratio"] <= float(limits["rafor_inference_lane_ratio"])
            for suite in suite_results
            for lane in suite["lanes"]
        ),
        "historical_fit": fit_ferric / fit_baseline <= float(limits["historical_fit_ratio"]),
        "historical_inference": all(
            lane["ratio"] <= float(limits["historical_inference_ratio"])
            for lane in historical
        ),
    }
    return {
        "schema_version": 1,
        "fit": {
            "ferricml_ns": fit_ferric,
            "rafor_ns": fit_rafor,
            "rafor_ratio": fit_ratio,
            "historical_ratio": fit_ferric / fit_baseline,
        },
        "inference_geomean_ratio": inference_geomean,
        "inference_suites": suite_results,
        "historical_inference": historical,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("benchmarks/forest-performance-contract.json"))
    parser.add_argument("--criterion-dir", type=Path, default=Path("benchmarks/alternatives/target/criterion"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true", help="fail on limits; report-only by default")
    parser.add_argument("--runner-id", help="required with --enforce and checked against the contract")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    if contract.get("schema_version") != 1:
        raise SystemExit("unsupported contract schema_version")
    if args.enforce:
        reference = contract["reference_runner"]
        actual = {"system": platform.system(), "machine": platform.machine()}
        if args.runner_id != reference["id"]:
            raise SystemExit("--enforce requires the contract reference runner id")
        for key in ("system", "machine"):
            if actual[key] != reference[key]:
                raise SystemExit(f"runner fingerprint mismatch for {key}: {actual[key]} != {reference[key]}")

    try:
        result = evaluate(contract, args.criterion_dir)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid performance inputs: {error}") from error
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return int(args.enforce and not result["passed"])


if __name__ == "__main__":
    sys.exit(main())
