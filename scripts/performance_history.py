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
MULTI_SUITE_PROTOCOL = "ferricml-history-v2"
FOREST_PROTOCOL = "forest-history-v1"
FERRICML_MODELS_PROTOCOL = "ferricml-models-v2"
FERRICML_MODELS_SUITE = "ferricml-models-v2"
FIT_LIMIT = 1.15
INFERENCE_LIMIT = 1.10

FIT = "forest_historical_fit_2048x64_20t/ferricml"
INFERENCE = tuple(
    f"forest_historical_into_{rows}x64_100t/{operation}"
    for rows in (1, 32, 1024)
    for operation in ("labels", "full_proba", "class_proba")
)

MODEL_FIT = (
    "ferricml_models_v1_fit_2048x48/linear",
    "ferricml_models_v1_fit_2048x48/ridge",
    "ferricml_models_v1_fit_2048x48/ranker_1024_pairs",
    "ferricml_models_v1_fit_2048x48/scaler_ridge_pipeline",
    "ferricml_boosting_v1_fit_2048x48_64t7l/ferricml",
    "ferricml_models_v2_logistic_fit_2048x48/ferricml",
    "ferricml_models_v2_scaler_fit_2048x48/ferricml",
)
MODEL_INFERENCE = (
    "ferricml_models_v1_into_1024x48/linear",
    "ferricml_models_v1_into_1024x48/ridge",
    "ferricml_models_v1_into_1024x48/ranker_scores",
    "ferricml_models_v1_into_1024x48/scaler_ridge_pipeline",
    "ferricml_boosting_v2_predict_one_256x_32t7l/predict",
    "ferricml_boosting_v2_predict_one_256x_64t7l/predict",
    "ferricml_boosting_v2_predict_one_256x_64t15l/predict",
    "ferricml_boosting_v2_predict_one_256x_128t15l/predict",
    "ferricml_boosting_v1_into_32x48_64t7l/predict",
    "ferricml_boosting_v1_into_1024x48_64t7l/predict",
    "ferricml_models_v2_logistic_into_1024x48/proba",
    "ferricml_models_v2_scaler_into_1024x48/transform",
    "ferricml_model_selection_v2_holdout_1000000/ordinary_shuffled_20pct",
    "ferricml_model_selection_v2_holdout_1000000/ordinary_shuffled_80pct",
    "ferricml_model_selection_v2_holdout_1000000/ordinary_unshuffled_20pct",
    "ferricml_model_selection_v2_holdout_1000000/stratified_4_class_20pct",
    "ferricml_model_selection_v2_stratified_262144/256_class_50pct",
)
BENCH_TARGETS = ("forest", "models", "boosting")


def limits(fit: tuple[str, ...], inference: tuple[str, ...]) -> dict[str, float]:
    result = {name: FIT_LIMIT for name in fit}
    result.update({name: INFERENCE_LIMIT for name in inference})
    return result


SUITE_SPECS = {
    "forest-v1": {
        "protocol": FOREST_PROTOCOL,
        "benchmarks": (FIT, *INFERENCE),
        "limits": limits((FIT,), INFERENCE),
    },
    FERRICML_MODELS_SUITE: {
        "protocol": FERRICML_MODELS_PROTOCOL,
        "benchmarks": (*MODEL_FIT, *MODEL_INFERENCE),
        "limits": limits(MODEL_FIT, MODEL_INFERENCE),
    },
}


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
        if executable in {"cargo", "rustc", "criterion"} or executable.startswith(
            ("forest-", "models-", "boosting-")
        ):
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


def record_suites(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    suites = record.get("suites")
    if isinstance(suites, dict):
        return suites
    if record.get("protocol") == FOREST_PROTOCOL and isinstance(record.get("metrics"), dict):
        return {
            "forest-v1": {
                "protocol": FOREST_PROTOCOL,
                "limits": SUITE_SPECS["forest-v1"]["limits"],
                "metrics": record["metrics"],
            }
        }
    return {}


def load_history(
    history_dir: Path, current: dict[str, Any], include_equal_version: bool = False
) -> list[dict[str, Any]]:
    records = []
    for path in history_dir.glob("v*.json"):
        try:
            candidate = json.loads(path.read_text())
            version_key(candidate["version"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidate_version = version_key(candidate["version"])
        current_version = version_key(current["version"])
        version_matches = candidate_version < current_version or (
            include_equal_version and candidate_version == current_version
        )
        if candidate.get("runner", {}).get("id") == current["runner"]["id"] and version_matches:
            records.append(candidate)
    return sorted(records, key=lambda item: version_key(item["version"]))


def compare_one(
    current: dict[str, Any], reference: dict[str, Any], reference_version: str, kind: str
) -> dict[str, Any]:
    current_metrics = current["metrics"]
    reference_metrics = reference["metrics"]
    shared = sorted(set(current_metrics) & set(reference_metrics))
    new_lanes = sorted(set(current_metrics) - set(reference_metrics))
    missing_lanes = sorted(set(reference_metrics) - set(current_metrics))
    ratios = {name: current_metrics[name] / reference_metrics[name] for name in shared}
    failures = [
        {"benchmark": name, "ratio": ratio, "limit": current["limits"][name]}
        for name, ratio in ratios.items()
        if ratio > current["limits"][name]
    ]
    if failures:
        status = "regression"
    elif not shared or new_lanes or missing_lanes:
        status = "insufficient_history"
    else:
        status = "pass"
    return {
        "kind": kind,
        "status": status,
        "reference_version": reference_version,
        "ratios": ratios,
        "new_lanes": new_lanes,
        "missing_lanes": missing_lanes,
        "failures": failures,
    }


def comparisons(
    current: dict[str, Any], history_dir: Path, include_equal_version: bool = False
) -> dict[str, Any]:
    history = load_history(history_dir, current, include_equal_version)
    suite_results = {}
    for suite_name, current_suite in record_suites(current).items():
        compatible = []
        for candidate in history:
            candidate_suite = record_suites(candidate).get(suite_name)
            if candidate_suite and candidate_suite.get("protocol") == current_suite.get("protocol"):
                compatible.append((candidate["version"], candidate_suite))
        if compatible:
            reference_version, reference_suite = compatible[-1]
            previous = compare_one(
                current_suite, reference_suite, reference_version, "previous_release"
            )
        else:
            previous = {
                "kind": "previous_release",
                "status": "insufficient_history",
                "reason": "no earlier compatible release exists for this runner and suite",
            }
        if len(compatible) >= 3:
            reference_version, reference_suite = compatible[-3]
            anchor = compare_one(
                current_suite,
                reference_suite,
                reference_version,
                "approximately_three_release_anchor",
            )
        else:
            anchor = {
                "kind": "approximately_three_release_anchor",
                "status": "insufficient_history",
                "reason": f"need three earlier compatible releases; found {len(compatible)}",
            }
        suite_results[suite_name] = {
            "protocol": current_suite["protocol"],
            "previous_release": previous,
            "anchor": anchor,
        }
    statuses = [
        comparison["status"]
        for suite in suite_results.values()
        for comparison in (suite["previous_release"], suite["anchor"])
    ]
    if "regression" in statuses:
        verdict = "regression"
    elif "insufficient_history" in statuses:
        verdict = "insufficient_history"
    else:
        verdict = "pass"
    return {
        "thresholds": {"fit_ratio": FIT_LIMIT, "inference_ratio": INFERENCE_LIMIT},
        "suites": suite_results,
        "verdict": verdict,
    }


def parse_model_metadata(console: str) -> dict[str, dict[str, Any]]:
    prefix = "FERRICML_BENCH_METADATA "
    result = {}
    for line in console.splitlines():
        if prefix not in line:
            continue
        payload = json.loads(line.split(prefix, 1)[1])
        if "model" not in payload:
            continue
        name = payload.pop("model")
        if not isinstance(name, str) or name in result:
            raise RuntimeError(f"invalid or repeated benchmark model metadata: {name!r}")
        if any(not isinstance(payload.get(field), int) or payload[field] <= 0 for field in (
            "trees",
            "max_leaf_nodes",
            "logical_nodes",
            "artifact_bytes",
        )):
            raise RuntimeError(f"invalid benchmark model metadata for {name}: {payload}")
        result[name] = payload
    expected = {"32t7l", "64t7l", "64t15l", "128t15l"}
    if set(result) != expected:
        raise RuntimeError(
            f"benchmark model metadata mismatch: expected {sorted(expected)}, got {sorted(result)}"
        )
    return result


def capture(args: argparse.Namespace) -> int:
    history_dir = Path(args.history_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    version = args.version or cargo_version()
    identity = runner_id(args.runner_id)
    release_destination = history_dir / f"v{version}.json"
    if not args.diagnostic and release_destination.exists():
        raise RuntimeError(
            f"immutable release summary already exists: {release_destination}; "
            "use a new version, never overwrite release history"
        )

    idle_before = idle_evidence(args.idle_samples, args.minimum_idle)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_kind = "ferricml-model-diagnostic" if args.diagnostic else f"ferricml-history-v{version}"
    evidence = out_root / f"{stamp}-{run_kind}"
    evidence.mkdir(parents=True, exist_ok=False)
    before_load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    command = [
        "cargo",
        "bench",
        "--locked",
    ]
    for target in BENCH_TARGETS:
        command.extend(("--bench", target))
    command.extend(
        (
            "--",
            "--noplot",
            "--sample-size",
            str(args.sample_size),
            "--warm-up-time",
            str(args.warm_up_time),
            "--measurement-time",
            str(args.measurement_time),
        )
    )
    run = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (evidence / "cargo-bench.txt").write_text(run.stdout)
    print(run.stdout, end="")
    if run.returncode:
        return run.returncode

    criterion = ROOT / "target" / "criterion"
    suites = {
        name: {
            "protocol": spec["protocol"],
            "limits": spec["limits"],
            "metrics": {
                benchmark: estimate(criterion, benchmark)
                for benchmark in spec["benchmarks"]
            },
        }
        for name, spec in SUITE_SPECS.items()
    }
    model_metadata = parse_model_metadata(run.stdout)
    logical_cpus = os.cpu_count()
    after_load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    record: dict[str, Any] = {
        "schema_version": 2,
        "version": version,
        "protocol": MULTI_SUITE_PROTOCOL,
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
            "history_includes_equal_version": args.diagnostic,
        },
        "suites": suites,
        "model_metadata": model_metadata,
    }
    record["comparison"] = comparisons(
        record, history_dir, include_equal_version=args.diagnostic
    )
    if args.diagnostic:
        destination = history_dir.parent / f"{stamp}-ferricml-model-workloads.json"
    else:
        destination = release_destination
        history_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")

    for name in ("benchmark.json", "estimates.json", "sample.json", "tukey.json"):
        for source in criterion.rglob(name):
            if not any(
                prefix in source.as_posix()
                for prefix in (
                    "forest_historical_",
                    "ferricml_models_v1_",
                    "ferricml_models_v2_",
                    "ferricml_boosting_v1_",
                    "ferricml_boosting_v2_",
                )
            ):
                continue
            copied = evidence / "criterion" / source.relative_to(criterion)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copied)

    verdict = record["comparison"]["verdict"]
    label = "dated diagnostic summary" if args.diagnostic else "immutable release summary"
    print(f"{label}: {destination}")
    print(f"history verdict: {verdict}")
    if verdict == "insufficient_history":
        print("insufficient history is expected until prior and three-release anchors exist")
    return 1 if verdict == "regression" and args.enforce else 0


def check(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text())
    include_equal = bool(summary.get("run", {}).get("history_includes_equal_version", False))
    result = comparisons(
        summary, Path(args.history_dir).resolve(), include_equal_version=include_equal
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["verdict"] == "regression" and args.enforce else 0


def fixture(
    version: str,
    fit: float = 100.0,
    inference: float = 100.0,
    include_models: bool = False,
) -> dict[str, Any]:
    metrics = {FIT: fit}
    metrics.update({name: inference for name in INFERENCE})
    record = {
        "schema_version": 2,
        "version": version,
        "protocol": MULTI_SUITE_PROTOCOL,
        "runner": {"id": "self-test"},
        "suites": {
            "forest-v1": {
                "protocol": FOREST_PROTOCOL,
                "limits": dict(SUITE_SPECS["forest-v1"]["limits"]),
                "metrics": metrics,
            }
        },
    }
    if include_models:
        model_metrics = {name: fit for name in MODEL_FIT}
        model_metrics.update({name: inference for name in MODEL_INFERENCE})
        record["suites"][FERRICML_MODELS_SUITE] = {
            "protocol": FERRICML_MODELS_PROTOCOL,
            "limits": dict(SUITE_SPECS[FERRICML_MODELS_SUITE]["limits"]),
            "metrics": model_metrics,
        }
    return record


def legacy_forest_fixture(version: str) -> dict[str, Any]:
    current = fixture(version)
    return {
        "schema_version": 1,
        "version": version,
        "protocol": FOREST_PROTOCOL,
        "runner": {"id": "self-test"},
        "metrics": current["suites"]["forest-v1"]["metrics"],
    }


def legacy_models_fixture(version: str) -> dict[str, Any]:
    record = fixture(version, include_models=True)
    suite = record["suites"].pop(FERRICML_MODELS_SUITE)
    suite["protocol"] = "ferricml-models-v1"
    record["suites"]["ferricml-models-v1"] = suite
    return record


def self_test() -> int:
    metadata = "\n".join(
        (
            'FERRICML_BENCH_METADATA {"suite":"models-v2","rows":2048}',
            *(
                f'FERRICML_BENCH_METADATA {{"model":"{name}","trees":1,'
                '"max_leaf_nodes":2,"logical_nodes":3,"artifact_bytes":4}'
                for name in ("32t7l", "64t7l", "64t15l", "128t15l")
            ),
        )
    )
    assert set(parse_model_metadata(metadata)) == {
        "32t7l",
        "64t7l",
        "64t15l",
        "128t15l",
    }
    with tempfile.TemporaryDirectory() as temporary:
        history = Path(temporary)
        first = fixture("0.1.0")
        result = comparisons(first, history)
        assert result["verdict"] == "insufficient_history"
        forest = result["suites"]["forest-v1"]
        assert forest["previous_release"]["status"] == "insufficient_history"
        assert forest["anchor"]["status"] == "insufficient_history"

        (history / "v0.1.0.json").write_text(json.dumps(legacy_forest_fixture("0.1.0")))
        (history / "v0.1.1.json").write_text(json.dumps(legacy_models_fixture("0.1.1")))
        mixed = fixture("0.2.0", include_models=True)
        result = comparisons(mixed, history)
        assert result["suites"]["forest-v1"]["previous_release"]["status"] == "pass"
        assert (
            result["suites"][FERRICML_MODELS_SUITE]["previous_release"]["status"]
            == "insufficient_history"
        )

        new_lane = fixture("0.2.0")
        forest_suite = new_lane["suites"]["forest-v1"]
        forest_suite["metrics"]["forest_new_lane"] = 100.0
        forest_suite["limits"]["forest_new_lane"] = INFERENCE_LIMIT
        result = comparisons(new_lane, history)
        previous = result["suites"]["forest-v1"]["previous_release"]
        assert previous["status"] == "insufficient_history"
        assert previous["new_lanes"] == ["forest_new_lane"]

        missing_lane = fixture("0.2.0")
        del missing_lane["suites"]["forest-v1"]["metrics"][INFERENCE[0]]
        result = comparisons(missing_lane, history)
        previous = result["suites"]["forest-v1"]["previous_release"]
        assert previous["status"] == "insufficient_history"
        assert previous["missing_lanes"] == [INFERENCE[0]]

        for version in ("0.2.0", "0.3.0"):
            (history / f"v{version}.json").write_text(json.dumps(fixture(version)))
        fourth = fixture("0.4.0", fit=114.9, inference=109.9)
        result = comparisons(fourth, history)
        assert result["verdict"] == "pass"
        assert (
            result["suites"]["forest-v1"]["anchor"]["reference_version"] == "0.1.1"
        )

        regressed = fixture("0.4.0", fit=115.1, inference=110.1)
        result = comparisons(regressed, history)
        assert result["verdict"] == "regression"
        forest = result["suites"]["forest-v1"]
        assert forest["previous_release"]["status"] == "regression"
        assert forest["anchor"]["status"] == "regression"

        incompatible = fixture("0.5.0")
        incompatible["suites"]["forest-v1"]["protocol"] = "forest-history-v2"
        result = comparisons(incompatible, history)
        assert (
            result["suites"]["forest-v1"]["previous_release"]["status"]
            == "insufficient_history"
        )

        for version in ("1.0.0", "1.1.0", "1.2.0"):
            (history / f"v{version}.json").write_text(
                json.dumps(fixture(version, include_models=True))
            )
        model_current = fixture("1.3.0", include_models=True)
        result = comparisons(model_current, history)
        models = result["suites"][FERRICML_MODELS_SUITE]
        assert models["previous_release"]["status"] == "pass"
        assert models["anchor"]["status"] == "pass"
        assert models["anchor"]["reference_version"] == "1.0.0"

        model_regressed = fixture(
            "1.3.0", fit=115.1, inference=110.1, include_models=True
        )
        result = comparisons(model_regressed, history)
        models = result["suites"][FERRICML_MODELS_SUITE]
        assert models["previous_release"]["status"] == "regression"
        assert models["anchor"]["status"] == "regression"
    print(
        "performance history self-test passed "
        "(mixed protocols, missing/new lanes, prior, anchor, regression)"
    )
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
    capture_parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="write dated evidence without occupying an immutable release version",
    )
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
