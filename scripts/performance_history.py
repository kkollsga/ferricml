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

# Idle-aware comparability.  Two runs are judged against a 1.10/1.15 ratio
# limit, so the machine states they were captured under have to be closer than
# that limit or the comparison launders background load into a code verdict.
# The recorded evidence is in dev-docs/plans/performance-gate-reliability.md
# (G0 Finding 1): every recorded comparison at a gap of at most 0.47 points
# stayed inside its limits, while both recorded `regression` verdicts came from
# the two runs with the largest negative gap.  The reproduced false regression
# sits at 5.57 points, and this runner's structural idle band is 86-90% — four
# points wide — so the rule is: compare within the machine's idle regime and
# refuse across it.
IDLE_GAP_COMPARABLE = 1.0
IDLE_GAP_MAX = 4.0
IDLE_RATIO_PER_POINT = 0.01
IDLE_BANDS = ("comparable", "widened", "favored", "not_comparable", "unknown")

# Capture measurement configuration.  Criterion's median interval narrows as
# 1/sqrt(n), and at the former sample-size 20 the worst gated lane carried a
# 13.52% mean interval — wider than the 10% margin the 1.10 limit allows, so
# the gate was judging lanes it could not resolve.  Requiring the ratio's own
# uncertainty to stay inside half that margin needs an interval of at most
# 7.07%, which 75 samples just reaches and 100 clears at 6.05%.  100 is also
# Criterion's default, so `bench-self` and `bench-history` finally measure
# identically and the tripwire can corroborate the gate; it is additionally
# the only candidate with independent live-tree corroboration (2.34-6.16%
# measured for exactly that lane).  See performance-gate-reliability.md,
# G0 Finding 5.
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_WARM_UP_SECONDS = 3.0
DEFAULT_MEASUREMENT_SECONDS = 5.0
MEASUREMENT_FIELDS = ("sample_size", "warm_up_seconds", "measurement_seconds")

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
    "ferricml_forest_v2_weighted_fit_2048x64_20t/classifier_unweighted",
    "ferricml_forest_v2_weighted_fit_2048x64_20t/classifier_weighted",
    "ferricml_forest_v2_weighted_fit_2048x64_20t/regressor_weighted",
    "ferricml_boosting_v2_weighted_fit_2048x48_64t7l/unweighted",
    "ferricml_boosting_v2_weighted_fit_2048x48_64t7l/weighted",
    "ferricml_boosting_v3_classifier_fit_2048x48_64t7l/ferricml",
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
    "ferricml_boosting_v3_classifier_predict_one_256x_64t7l/predict",
    "ferricml_boosting_v3_classifier_proba_into_32x48_64t7l/predict_proba",
    "ferricml_boosting_v3_classifier_proba_into_1024x48_64t7l/predict_proba",
    "ferricml_models_v2_logistic_into_1024x48/proba",
    "ferricml_models_v2_scaler_into_1024x48/transform",
    "ferricml_model_selection_v2_holdout_1000000/ordinary_shuffled_20pct",
    "ferricml_model_selection_v2_holdout_1000000/ordinary_shuffled_80pct",
    "ferricml_model_selection_v2_holdout_1000000/stratified_4_class_20pct",
    "ferricml_model_selection_v2_stratified_262144/256_class_50pct",
    "ferricml_forest_v1_regressor_into_32x64_100t/predict",
    "ferricml_forest_v1_regressor_into_1024x64_100t/predict",
    "ferricml_artifact_v1_forest_regressor_512x16_32t/encode",
    "ferricml_artifact_v1_forest_regressor_512x16_32t/decode",
    "ferricml_artifact_v1_forest_classifier_512x16_32t/encode",
    "ferricml_artifact_v1_forest_classifier_512x16_32t/decode",
    "ferricml_artifact_v1_forest_classifier_512x16_32t/multiclass_encode",
    "ferricml_artifact_v1_forest_classifier_512x16_32t/multiclass_decode",
    "ferricml_inspection_v1_permutation_256x8_3r/forest_mse",
    "ferricml_inspection_v1_permutation_256x8_3r/ridge_r2",
)
# Diagnostic lanes are captured, recorded, and reported exactly like gated
# lanes — so the workload stays registered and visible to `bench-history` — but
# carry no limit, so they can never produce a pass/fail verdict.
#
# `ordinary_unshuffled_20pct` is the unshuffled branch of `train_test_split`:
# two `Vec<usize>` allocations totalling 8 MB, sequentially filled, containing
# no algorithm.  What it measures is the allocator's page-supply state, and it
# does not settle.  Its own within-run median confidence interval is 8.46 /
# 13.65 / 24.56% across the three archived captures (sample-size 20) at
# 11k-17k iterations, while every sibling in its group stays under 2.58% at
# 630-1260 iterations; raising the sample count does not help it either
# (2.23 / 21.19 / 20.24% across three live trees at sample-size 100, siblings
# under 1.14%, with a 33% spread in the point estimate on identical code).  A
# lane whose single-run interval is +/-10-21% cannot be judged against a 1.10
# ratio limit.  See performance-gate-reliability.md, G0 Findings 2 and 5.
MODEL_DIAGNOSTIC = (
    "ferricml_model_selection_v2_holdout_1000000/ordinary_unshuffled_20pct",
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
        "diagnostic": (),
    },
    FERRICML_MODELS_SUITE: {
        "protocol": FERRICML_MODELS_PROTOCOL,
        "benchmarks": (*MODEL_FIT, *MODEL_INFERENCE, *MODEL_DIAGNOSTIC),
        "limits": limits(MODEL_FIT, MODEL_INFERENCE),
        "diagnostic": MODEL_DIAGNOSTIC,
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


def idle_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    """Summarise a run's recorded idle evidence, or None when it has none."""
    samples = (record.get("run") or {}).get("cpu_idle_percent_before")
    if not isinstance(samples, list) or not samples:
        return None
    values = []
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            return None
        values.append(float(sample))
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "samples": values,
    }


def measurement_config(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a run's Criterion measurement configuration, or None."""
    run = record.get("run") or {}
    values = {field: run.get(field) for field in MEASUREMENT_FIELDS}
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    return values


def comparability(
    current_idle: dict[str, Any] | None,
    reference_idle: dict[str, Any] | None,
    current_config: dict[str, Any] | None = None,
    reference_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify how comparable two runs were.

    Two things can make a pair incomparable.  A measurement-configuration
    difference is absolute: the sample count sets how tightly a lane's median
    can be resolved at all, so a 20-sample run and a 100-sample run are not
    measuring the same quantity.  An idle difference is graded: a negative gap
    means the current run was the noisier one, so its ratios are inflated and
    the envelope is widened, while a positive gap means it was the quieter one
    and the limits are *not* relaxed — that direction produces an optimistic
    pass rather than a false regression, and an optimistic pass has to stay
    legible instead of being rewarded.
    """
    evidence = {
        "current_idle": current_idle,
        "reference_idle": reference_idle,
        "current_measurement": current_config,
        "reference_measurement": reference_config,
    }
    if (
        current_config is not None
        and reference_config is not None
        and current_config != reference_config
    ):
        return {
            "band": "not_comparable",
            "reason": "measurement_configuration",
            "gap": None,
            "limit_scale": 1.0,
            **evidence,
        }
    if current_idle is None or reference_idle is None:
        return {
            "band": "unknown",
            "reason": None,
            "gap": None,
            "limit_scale": 1.0,
            **evidence,
        }
    gap = current_idle["mean"] - reference_idle["mean"]
    distance = abs(gap)
    scale = 1.0
    reason = None
    if distance > IDLE_GAP_MAX:
        band, reason = "not_comparable", "idle_gap"
    elif distance > IDLE_GAP_COMPARABLE:
        if gap < 0:
            band = "widened"
            scale = 1.0 + IDLE_RATIO_PER_POINT * (distance - IDLE_GAP_COMPARABLE)
        else:
            band = "favored"
    else:
        band = "comparable"
    return {
        "band": band,
        "reason": reason,
        "gap": gap,
        "limit_scale": scale,
        **evidence,
    }


def compare_one(
    current: dict[str, Any],
    reference: dict[str, Any],
    reference_version: str,
    kind: str,
    comparable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparable = comparable or comparability(None, None)
    current_metrics = current["metrics"]
    reference_metrics = reference["metrics"]
    shared = sorted(set(current_metrics) & set(reference_metrics))
    new_lanes = sorted(set(current_metrics) - set(reference_metrics))
    missing_lanes = sorted(set(reference_metrics) - set(current_metrics))
    ratios = {name: current_metrics[name] / reference_metrics[name] for name in shared}
    scale = comparable["limit_scale"]
    limits = {name: limit * scale for name, limit in current["limits"].items()}
    exceeded = [
        {"benchmark": name, "ratio": ratio, "limit": limits[name]}
        for name, ratio in ratios.items()
        if name in limits and ratio > limits[name]
    ]
    failures: list[dict[str, Any]] = exceeded
    unexplained: list[dict[str, Any]] = []
    if comparable["band"] == "not_comparable":
        # The two machine states differ by more than the ratio limit they would
        # be judged against, so neither a pass nor a regression is knowable.
        # Say exactly that rather than picking whichever wrong answer the
        # numbers happen to point at.
        failures, unexplained = [], exceeded
        status = "not_comparable"
    elif failures:
        status = "regression"
    elif not shared or new_lanes or missing_lanes:
        status = "insufficient_history"
    else:
        status = "pass"
    return {
        "kind": kind,
        "status": status,
        "reference_version": reference_version,
        "comparability": comparable,
        "ratios": ratios,
        "diagnostic_ratios": {
            name: ratio for name, ratio in ratios.items() if name not in limits
        },
        "new_lanes": new_lanes,
        "missing_lanes": missing_lanes,
        "failures": failures,
        "unexplained_failures": unexplained,
    }


def comparisons(
    current: dict[str, Any], history_dir: Path, include_equal_version: bool = False
) -> dict[str, Any]:
    history = load_history(history_dir, current, include_equal_version)
    current_idle = idle_summary(current)
    current_config = measurement_config(current)
    suite_results = {}
    for suite_name, current_suite in record_suites(current).items():
        compatible = []
        for candidate in history:
            candidate_suite = record_suites(candidate).get(suite_name)
            if candidate_suite and candidate_suite.get("protocol") == current_suite.get("protocol"):
                compatible.append(
                    (
                        candidate["version"],
                        candidate_suite,
                        idle_summary(candidate),
                        measurement_config(candidate),
                    )
                )
        if compatible:
            reference_version, reference_suite, reference_idle, reference_config = compatible[-1]
            previous = compare_one(
                current_suite,
                reference_suite,
                reference_version,
                "previous_release",
                comparability(
                    current_idle, reference_idle, current_config, reference_config
                ),
            )
        else:
            previous = {
                "kind": "previous_release",
                "status": "insufficient_history",
                "reason": "no earlier compatible release exists for this runner and suite",
            }
        if len(compatible) >= 3:
            reference_version, reference_suite, reference_idle, reference_config = compatible[-3]
            anchor = compare_one(
                current_suite,
                reference_suite,
                reference_version,
                "approximately_three_release_anchor",
                comparability(
                    current_idle, reference_idle, current_config, reference_config
                ),
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
    elif "not_comparable" in statuses:
        verdict = "not_comparable"
    elif "insufficient_history" in statuses:
        verdict = "insufficient_history"
    else:
        verdict = "pass"
    return {
        "thresholds": {"fit_ratio": FIT_LIMIT, "inference_ratio": INFERENCE_LIMIT},
        "suites": suite_results,
        "verdict": verdict,
    }


def verdict_report(comparison: dict[str, Any]) -> list[str]:
    """Explain a non-comparable verdict, naming both runs' idle evidence.

    A `not_comparable` verdict means the tripwire did not run.  It is never
    evidence that the tripwire passed, so the wording has to leave no room to
    bank it as one.
    """
    lines = []
    for suite_name, suite in comparison.get("suites", {}).items():
        for key in ("previous_release", "anchor"):
            result = suite.get(key, {})
            if result.get("status") != "not_comparable":
                continue
            evidence = result["comparability"]
            lines.append(
                f"not comparable: {suite_name} vs v{result['reference_version']} "
                f"({result['kind']}); reason: {evidence['reason']}"
            )
            current, reference = evidence["current_idle"], evidence["reference_idle"]
            if current and reference:
                lines.append(
                    f"  idle before this run: {current['samples']} (mean "
                    f"{current['mean']:.2f}%); reference: {reference['samples']} "
                    f"(mean {reference['mean']:.2f}%); gap {evidence['gap']:+.2f} points, "
                    f"limit {IDLE_GAP_MAX:.1f}"
                )
            if evidence["reason"] == "measurement_configuration":
                lines.append(
                    f"  measurement configuration differs: this run "
                    f"{evidence['current_measurement']}; reference "
                    f"{evidence['reference_measurement']}"
                )
            for failure in result.get("unexplained_failures", []):
                lines.append(
                    f"  over limit but not attributable: {failure['benchmark']} "
                    f"ratio {failure['ratio']:.4f} vs {failure['limit']:.4f}"
                )
            lines.append(
                "  this is NOT a pass: the two runs differ by more than the ratio "
                "limit they would be judged against, so neither a pass nor a "
                "regression is knowable. Re-measure both points under one idle "
                "regime and one measurement configuration."
            )
    return lines


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
                    "ferricml_forest_v2_",
                    "ferricml_artifact_v1_",
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
    for line in verdict_report(record["comparison"]):
        print(line)
    return 1 if verdict == "regression" and args.enforce else 0


def check(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text())
    include_equal = bool(summary.get("run", {}).get("history_includes_equal_version", False))
    result = comparisons(
        summary, Path(args.history_dir).resolve(), include_equal_version=include_equal
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    for line in verdict_report(result):
        print(line)
    return 1 if result["verdict"] == "regression" and args.enforce else 0


def fixture(
    version: str,
    fit: float = 100.0,
    inference: float = 100.0,
    include_models: bool = False,
    idle: list[float] | None = None,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {FIT: fit}
    metrics.update({name: inference for name in INFERENCE})
    record: dict[str, Any] = {
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
    run: dict[str, Any] = {}
    if idle is not None:
        run["cpu_idle_percent_before"] = list(idle)
    if measurement is not None:
        run.update(measurement)
    if run:
        record["run"] = run
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


# Every idle-comparability rule is proven to *fire* against a synthetic input,
# not merely asserted to leave the live tree clean.  Reference metrics are
# always 100.0, so `inference` is directly the ratio in percent.  Layout:
# (label, expected band, current idle samples, reference idle samples,
#  current inference metric, expected previous-release status).
IDLE_CASES: tuple[tuple[str, str, list[float] | None, list[float] | None, float, str], ...] = (
    # Matched machine states behave exactly as before this rule existed.
    ("comparable_pass", "comparable", [92.0] * 3, [92.0] * 3, 109.9, "pass"),
    ("comparable_regression", "comparable", [92.0] * 3, [92.0] * 3, 110.1, "regression"),
    # The control for `widened_fires`: identical numbers, matched idle, fails.
    ("widened_control", "comparable", [92.0] * 3, [92.0] * 3, 110.5, "regression"),
    # Gap -2.0 widens the 1.10 limit to 1.111, so 1.105 now passes ...
    ("widened_fires", "widened", [90.0] * 3, [92.0] * 3, 110.5, "pass"),
    # ... but widening is bounded: 1.115 still exceeds it.
    ("widened_bounded", "widened", [90.0] * 3, [92.0] * 3, 111.5, "regression"),
    # A quieter current run is favoured, never relaxed: 1.105 still fails.
    ("favored_never_relaxes", "favored", [94.0] * 3, [92.0] * 3, 110.5, "regression"),
    # Beyond the maximum gap no verdict is knowable, in either direction.
    ("not_comparable_downgrade", "not_comparable", [87.0] * 3, [92.0] * 3, 130.0, "not_comparable"),
    ("not_comparable_symmetry", "not_comparable", [97.0] * 3, [92.0] * 3, 100.0, "not_comparable"),
    # Missing evidence on either side keeps the pre-existing behaviour.
    ("unknown_unchanged", "unknown", None, None, 110.1, "regression"),
    ("unknown_one_sided", "unknown", [92.0] * 3, None, 110.1, "regression"),
    # Band boundaries are pinned exactly: 1.0 still comparable, 4.0 still
    # widened (gap -4.0 widens 1.10 to 1.133, so 1.130 passes).
    ("boundary_comparable", "comparable", [91.0] * 3, [92.0] * 3, 110.1, "regression"),
    ("boundary_widened", "widened", [88.0] * 3, [92.0] * 3, 113.0, "pass"),
)


def idle_case_result(
    root: Path,
    label: str,
    current_idle: list[float] | None,
    reference_idle: list[float] | None,
    inference: float,
) -> dict[str, Any]:
    history = root / label
    history.mkdir()
    (history / "v0.1.0.json").write_text(json.dumps(fixture("0.1.0", idle=reference_idle)))
    current = fixture("0.2.0", inference=inference, idle=current_idle)
    return comparisons(current, history)


def idle_self_test(root: Path) -> None:
    covered = {band for _, band, _, _, _, _ in IDLE_CASES}
    assert covered == set(IDLE_BANDS), (
        f"every comparability band needs a synthetic case: "
        f"missing={sorted(set(IDLE_BANDS) - covered)}, stale={sorted(covered - set(IDLE_BANDS))}"
    )
    for label, band, current_idle, reference_idle, inference, expected in IDLE_CASES:
        result = idle_case_result(root, label, current_idle, reference_idle, inference)
        previous = result["suites"]["forest-v1"]["previous_release"]
        assert previous["comparability"]["band"] == band, (
            f"{label}: expected band {band}, got {previous['comparability']['band']}"
        )
        assert previous["status"] == expected, (
            f"{label}: expected status {expected}, got {previous['status']}"
        )
        if expected == "not_comparable":
            # A non-comparable pair never reports a regression, and never
            # silently swallows the lanes that were over limit either.
            assert previous["failures"] == [], f"{label}: reported failures anyway"
            assert result["verdict"] == "not_comparable", (
                f"{label}: verdict {result['verdict']} outranked not_comparable"
            )
            assert verdict_report(result), f"{label}: produced no explanation"
        if label == "not_comparable_downgrade":
            assert previous["unexplained_failures"], (
                "an over-limit lane must still be reported, just not as a regression"
            )
        if label == "not_comparable_symmetry":
            assert not previous["unexplained_failures"], (
                "a clean-looking comparison across the idle band is still refused"
            )

    # Gap sign convention: negative means the current run was the noisier one.
    quiet, noisy = {"mean": 96.0, "min": 96.0, "samples": [96.0]}, {
        "mean": 90.0,
        "min": 90.0,
        "samples": [90.0],
    }
    assert comparability(noisy, quiet)["gap"] < 0
    assert comparability(quiet, noisy)["gap"] > 0
    assert comparability(noisy, quiet)["limit_scale"] == 1.0, (
        "beyond the maximum gap the widening model is out of range and must not apply"
    )
    assert idle_summary({"run": {"cpu_idle_percent_before": [90.0, 92.0]}})["mean"] == 91.0
    assert idle_summary({}) is None
    assert idle_summary({"run": {"cpu_idle_percent_before": []}}) is None
    assert idle_summary({"run": {"cpu_idle_percent_before": ["90"]}}) is None


def measurement_self_test(root: Path) -> None:
    # The parser must keep serving the configuration the derivation chose;
    # a default drifting away from the constant would silently restore the
    # under-sampled captures the constants exist to prevent.
    defaults = parser().parse_args(["capture"])
    assert defaults.sample_size == DEFAULT_SAMPLE_SIZE
    assert defaults.warm_up_time == DEFAULT_WARM_UP_SECONDS
    assert defaults.measurement_time == DEFAULT_MEASUREMENT_SECONDS

    settled = {
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "warm_up_seconds": DEFAULT_WARM_UP_SECONDS,
        "measurement_seconds": DEFAULT_MEASUREMENT_SECONDS,
    }
    idle = [92.0] * 3

    def previous(label: str, reference_measurement: dict[str, Any] | None) -> dict[str, Any]:
        history = root / label
        history.mkdir()
        (history / "v0.1.0.json").write_text(
            json.dumps(fixture("0.1.0", idle=idle, measurement=reference_measurement))
        )
        current = fixture("0.2.0", idle=idle, measurement=settled)
        result = comparisons(current, history)
        return result["suites"]["forest-v1"]["previous_release"]

    matched = previous("matched", dict(settled))
    assert matched["status"] == "pass", matched
    assert matched["comparability"]["band"] == "comparable"

    # Each field independently makes the pair incomparable: the sample count
    # sets how tightly a median can be resolved, and the time budgets set how
    # many iterations back each sample.
    for field, changed in (
        ("sample_size", 20),
        ("warm_up_seconds", 1.0),
        ("measurement_seconds", 2.0),
    ):
        reference = dict(settled)
        reference[field] = changed
        result = previous(f"differs_{field}", reference)
        assert result["status"] == "not_comparable", f"{field}: {result['status']}"
        assert result["comparability"]["reason"] == "measurement_configuration"

    # History predating the fields falls back to the idle bands rather than
    # failing, so older records stay readable.
    legacy = previous("legacy", None)
    assert legacy["comparability"]["band"] == "comparable", legacy["comparability"]
    assert legacy["status"] == "pass"

    assert measurement_config({"run": dict(settled)}) == settled
    assert measurement_config({"run": {"sample_size": 100}}) is None
    assert measurement_config({}) is None


def diagnostic_self_test(root: Path) -> None:
    for suite_name, spec in SUITE_SPECS.items():
        gated, diagnostic = set(spec["limits"]), set(spec["diagnostic"])
        # A lane must be exactly one of gated or diagnostic.  Falling out of
        # both is how a workload silently stops being measured at all.
        assert gated | diagnostic == set(spec["benchmarks"]), (
            f"{suite_name}: every benchmark must be gated or diagnostic; "
            f"unclassified={sorted(set(spec['benchmarks']) - gated - diagnostic)}"
        )
        assert not gated & diagnostic, (
            f"{suite_name}: lanes in both sets: {sorted(gated & diagnostic)}"
        )
    for name in MODEL_DIAGNOSTIC:
        assert name in SUITE_SPECS[FERRICML_MODELS_SUITE]["benchmarks"], (
            f"{name} must still be captured and recorded"
        )
        assert name not in SUITE_SPECS[FERRICML_MODELS_SUITE]["limits"], (
            f"{name} must not carry a limit"
        )

    lane = "forest_diagnostic_lane"
    history = root / "history"
    history.mkdir()
    reference = fixture("0.1.0")
    reference["suites"]["forest-v1"]["metrics"][lane] = 100.0
    (history / "v0.1.0.json").write_text(json.dumps(reference))

    # A limitless lane five times slower produces no failure ...
    current = fixture("0.2.0")
    current["suites"]["forest-v1"]["metrics"][lane] = 500.0
    previous = comparisons(current, history)["suites"]["forest-v1"]["previous_release"]
    assert previous["status"] == "pass", f"diagnostic lane gated the verdict: {previous}"
    assert previous["failures"] == []
    assert previous["diagnostic_ratios"] == {lane: 5.0}, (
        "a diagnostic lane must still be reported, just not judged"
    )

    # ... while the same ratio on a gated lane still fails.
    regressed = fixture("0.2.0")
    regressed["suites"]["forest-v1"]["metrics"][lane] = 100.0
    regressed["suites"]["forest-v1"]["metrics"][INFERENCE[0]] = 500.0
    previous = comparisons(regressed, history)["suites"]["forest-v1"]["previous_release"]
    assert previous["status"] == "regression", "the gated control did not fire"

    # Dropping a diagnostic lane is still noticed.
    dropped = comparisons(fixture("0.2.0"), history)
    previous = dropped["suites"]["forest-v1"]["previous_release"]
    assert previous["status"] == "insufficient_history"
    assert previous["missing_lanes"] == [lane]


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

        idle_root = Path(temporary) / "idle"
        idle_root.mkdir()
        idle_self_test(idle_root)

        diagnostic_root = Path(temporary) / "diagnostic"
        diagnostic_root.mkdir()
        diagnostic_self_test(diagnostic_root)

        measurement_root = Path(temporary) / "measurement"
        measurement_root.mkdir()
        measurement_self_test(measurement_root)
    print(
        "performance history self-test passed "
        "(mixed protocols, missing/new lanes, prior, anchor, regression, "
        f"{len(IDLE_CASES)} idle-comparability cases over {len(IDLE_BANDS)} bands, "
        f"{len(MODEL_DIAGNOSTIC)} diagnostic-only lanes, "
        f"{len(MEASUREMENT_FIELDS)} measurement-configuration fields)"
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
    capture_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    capture_parser.add_argument(
        "--warm-up-time", type=float, default=DEFAULT_WARM_UP_SECONDS
    )
    capture_parser.add_argument(
        "--measurement-time", type=float, default=DEFAULT_MEASUREMENT_SECONDS
    )
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
