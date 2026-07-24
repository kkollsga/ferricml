#!/usr/bin/env python3
"""Generate or verify FerricML's scikit-learn 1.9 black-box fixture.

Only public estimator APIs are used. By default this script is read-only and
fails with a diff if the frozen fixture differs. Pass ``--update`` explicitly
to replace the project-owned fixture after reviewing an intentional change.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import sys

import numpy as np
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.preprocessing import StandardScaler

REFERENCE_SKLEARN = "1.9.0"
REFERENCE_NUMPY = "2.4.1"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "reference_semantics_v1.rs"
MASK = (1 << 64) - 1
QUALITY_SEEDS = (11, 22, 33, 44, 55)
QUALITY_MARKER = "\n#[derive(Clone, Copy, Debug)]\npub struct QualityReference"
QUALITY_PATTERN = re.compile(
    r'QualityReference \{ lane: "([^"]+)", seed: (\d+), '
    r"accuracy: ([^,]+), brier: ([^,]+), nrmse: ([^ }]+) \},"
)
ARRAY_PATTERN = re.compile(
    r"^pub const ([A-Z0-9_]+): &\[(f32|f64|u8)\] = &\[(.*)\];$",
    re.MULTILINE,
)
EXACT_F32_PORTABILITY = 2.0e-5
PORTABLE_F32_ARRAYS = {
    "LOGISTIC_NO_INTERCEPT_COEFFICIENTS",
    "LOGISTIC_NO_INTERCEPT_DECISIONS",
    "LOGISTIC_NO_INTERCEPT_PROBABILITIES",
    "LOGISTIC_WEIGHTED_COEFFICIENTS",
    "LOGISTIC_WEIGHTED_INTERCEPT",
    "LOGISTIC_WEIGHTED_DECISIONS",
    "LOGISTIC_WEIGHTED_PROBABILITIES",
    "LINEAR_FULL_COEFFICIENTS",
    "LINEAR_FULL_INTERCEPT",
    "LINEAR_FULL_PREDICTIONS",
    "LINEAR_RANK_DEFICIENT_COEFFICIENTS",
    "LINEAR_WEIGHTED_COEFFICIENTS",
    "LINEAR_WEIGHTED_INTERCEPT",
    "RIDGE_FULL_COEFFICIENTS",
    "RIDGE_FULL_INTERCEPT",
    "RIDGE_FULL_PREDICTIONS",
    "RIDGE_ALPHA_ZERO_COEFFICIENTS",
    "RIDGE_WEIGHTED_COEFFICIENTS",
    "RIDGE_WEIGHTED_INTERCEPT",
}

# Exact trees are portable because randomness is disabled. Randomized forest
# quality can vary slightly across supported platforms even with a fixed public
# random_state, so CI validates it against a narrow envelope that is much
# tighter than FerricML's approved quality deltas.
QUALITY_ACCURACY_PORTABILITY = 0.01
QUALITY_BRIER_PORTABILITY = 0.002
QUALITY_NRMSE_PORTABILITY = 0.002


class SplitMix64:
    """Small cross-language generator owned by the conformance protocol."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        return value ^ (value >> 31)

    def signed_unit(self) -> np.float32:
        fraction = np.float32(self.next() >> 40) / np.float32(1 << 24)
        return np.float32(fraction * np.float32(2.0) - np.float32(1.0))


def matrix(rng: SplitMix64, rows: int, columns: int) -> np.ndarray:
    values = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            values[row, column] = rng.signed_unit()
    return values


def classification_data(lane: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = SplitMix64(seed)
    train_x = matrix(rng, 768, 12)
    test_x = matrix(rng, 384, 12)

    def labels(values: np.ndarray) -> np.ndarray:
        output = np.empty(values.shape[0], dtype=np.uint8)
        for index, row in enumerate(values):
            if lane == "nonlinear":
                score = np.float32(
                    row[0] * row[1]
                    + np.float32(0.7) * row[2] * row[2]
                    - np.float32(0.45) * row[3]
                    + np.float32(0.2) * row[4] * row[5]
                )
                output[index] = score > np.float32(0.15)
            elif lane == "separable":
                score = np.float32(
                    np.float32(1.2) * row[0]
                    - np.float32(0.9) * row[1]
                    + np.float32(0.5) * row[2]
                )
                output[index] = score > np.float32(0.0)
            elif lane == "imbalanced":
                score = np.float32(
                    np.float32(1.3) * row[0]
                    + np.float32(0.8) * row[1]
                    - np.float32(0.35) * row[2] * row[2]
                )
                output[index] = score > np.float32(1.25)
            elif lane == "noise":
                # A deterministic nuisance-feature problem: the target depends
                # on weak signal plus a feature-independent pseudo-noise term.
                noise = np.float32(((index * 1103515245 + seed) & 0xFFFF) / 32768.0 - 1.0)
                score = np.float32(np.float32(0.25) * row[0] + noise)
                output[index] = score > np.float32(0.0)
            else:
                raise ValueError(f"unknown classification lane: {lane}")
        return output

    return train_x, labels(train_x), test_x, labels(test_x)


def regression_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = SplitMix64(seed)
    train_x = matrix(rng, 768, 12)
    test_x = matrix(rng, 384, 12)

    def targets(values: np.ndarray) -> np.ndarray:
        output = np.empty(values.shape[0], dtype=np.float32)
        for index, row in enumerate(values):
            noise = np.float32(((index * 214013 + seed * 2531011) & 0xFFFF) / 32768.0 - 1.0)
            output[index] = np.float32(
                np.float32(1.7) * row[0]
                - np.float32(0.8) * row[1] * row[1]
                + np.float32(0.6) * row[2] * row[3]
                + np.float32(0.3) * row[4]
                + np.float32(0.1) * noise
            )
        return output

    return train_x, targets(train_x), test_x, targets(test_x)


def classifier_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "n_estimators": 64,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 0,
        "n_jobs": 1,
    }
    params.update(overrides)
    return params


def regressor_params(**overrides: object) -> dict[str, object]:
    params = classifier_params(max_features=1.0)
    params.update(overrides)
    return params


def rust_float(value: float) -> str:
    result = format(float(value), ".17g")
    if "." not in result and "e" not in result:
        result += ".0"
    return result


def rust_array(name: str, values: np.ndarray, rust_type: str) -> str:
    flat = values.reshape(-1)
    if rust_type == "f32":
        encoded = ", ".join(rust_float(np.float32(value)) for value in flat)
    elif rust_type == "f64":
        encoded = ", ".join(rust_float(float(value)) for value in flat)
    else:
        encoded = ", ".join(str(int(value)) for value in flat)
    return f"pub const {name}: &[{rust_type}] = &[{encoded}];\n"


def parsed_arrays(text: str) -> tuple[str, dict[str, tuple[str, list[float | int]]]]:
    arrays: dict[str, tuple[str, list[float | int]]] = {}
    for match in ARRAY_PATTERN.finditer(text):
        name, rust_type, encoded = match.groups()
        if name in arrays:
            raise RuntimeError(f"duplicate exact fixture array: {name}")
        parts = [part.strip() for part in encoded.split(",") if part.strip()]
        values: list[float | int]
        if rust_type == "u8":
            values = [int(part) for part in parts]
        else:
            values = [float(part) for part in parts]
        arrays[name] = (rust_type, values)
    skeleton = ARRAY_PATTERN.sub(
        lambda match: (
            f"pub const {match.group(1)}: &[{match.group(2)}] = &[<values>];"
        ),
        text,
    )
    return skeleton, arrays


def validate_exact_portability(frozen: str, generated: str) -> None:
    frozen_skeleton, frozen_arrays = parsed_arrays(frozen)
    generated_skeleton, generated_arrays = parsed_arrays(generated)
    if frozen_skeleton != generated_skeleton:
        raise RuntimeError("exact fixture declarations or metadata drifted")
    if frozen_arrays.keys() != generated_arrays.keys():
        raise RuntimeError("exact fixture array names or order drifted")

    failures: list[str] = []
    for name, (frozen_type, frozen_values) in frozen_arrays.items():
        generated_type, generated_values = generated_arrays[name]
        if frozen_type != generated_type:
            failures.append(f"{name}: type changed from {frozen_type} to {generated_type}")
            continue
        if len(frozen_values) != len(generated_values):
            failures.append(
                f"{name}: length changed from {len(frozen_values)} to "
                f"{len(generated_values)}"
            )
            continue
        tolerance = (
            EXACT_F32_PORTABILITY
            if frozen_type == "f32" and name in PORTABLE_F32_ARRAYS
            else 0.0
        )
        for index, (expected, actual) in enumerate(
            zip(frozen_values, generated_values, strict=True)
        ):
            delta = abs(float(expected) - float(actual))
            if delta > tolerance:
                failures.append(
                    f"{name}[{index}]: drift {delta:.6g} exceeds {tolerance:.6g}"
                )
                break
    if failures:
        raise RuntimeError("exact fixture drifted:\n" + "\n".join(failures))


def self_test_exact_portability() -> None:
    frozen = (
        'pub const SKLEARN_VERSION: &str = "1.9.0";\n'
        "pub const LINEAR_FULL_COEFFICIENTS: &[f32] = &[1.0];\n"
        "pub const LINEAR_FULL_X: &[f32] = &[2.0];\n"
        "pub const EXACT_LABELS: &[u8] = &[0, 1];\n"
    )
    within_tolerance = frozen.replace("&[1.0];", "&[1.00001];")
    validate_exact_portability(frozen, within_tolerance)
    for drifted in (
        frozen.replace("&[1.0];", "&[1.001];"),
        frozen.replace("&[2.0];", "&[2.000001];"),
        frozen.replace("&[0, 1];", "&[1, 0];"),
    ):
        try:
            validate_exact_portability(frozen, drifted)
        except RuntimeError:
            pass
        else:
            raise AssertionError("exact portability self-test accepted fixture drift")


def validate_reference_api() -> None:
    if sklearn.__version__ != REFERENCE_SKLEARN:
        raise RuntimeError(
            f"expected scikit-learn {REFERENCE_SKLEARN}, got {sklearn.__version__}"
        )
    if np.__version__ != REFERENCE_NUMPY:
        raise RuntimeError(f"expected NumPy {REFERENCE_NUMPY}, got {np.__version__}")

    classifier = RandomForestClassifier()
    regressor = RandomForestRegressor()
    boosting = HistGradientBoostingRegressor()
    supported = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "random_state",
        "n_jobs",
    }
    classifier_defaults = classifier.get_params(deep=False)
    regressor_defaults = regressor.get_params(deep=False)
    assert supported <= classifier_defaults.keys()
    assert supported <= regressor_defaults.keys()
    assert {key: classifier_defaults[key] for key in supported} == {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": None,
        "n_jobs": None,
    }
    assert {key: regressor_defaults[key] for key in supported} == {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True,
        "random_state": None,
        "n_jobs": None,
    }
    boosting_defaults = boosting.get_params(deep=False)
    assert {
        key: boosting_defaults[key]
        for key in (
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "max_bins",
        )
    } == {
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
    }

    invalid_x = np.asarray([[0.0], [1.0]], dtype=np.float32)
    invalid_y = np.asarray([0, 1], dtype=np.uint8)
    for kwargs in (
        {"n_estimators": 0},
        {"max_depth": 0},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"max_features": 0},
        {"n_jobs": 0},
    ):
        try:
            RandomForestClassifier(**kwargs).fit(invalid_x, invalid_y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"scikit accepted invalid parameters: {kwargs}")


def exact_fixture() -> str:
    train_x = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]],
        dtype=np.float32,
    )
    classifier_y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    regression_y = np.asarray([-1, 0, 1, 2, 4, 6, 7, 9], dtype=np.float32)
    test_x = np.asarray(
        [[-1, 0.5], [0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [4, 0.5]],
        dtype=np.float32,
    )
    exact_common = {
        "n_estimators": 1,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "bootstrap": False,
        "random_state": 0,
        "n_jobs": 1,
    }
    classifier = RandomForestClassifier(**exact_common).fit(train_x, classifier_y)
    regressor = RandomForestRegressor(**exact_common).fit(train_x, regression_y)

    tie_x = np.asarray([[0], [1], [2], [3]], dtype=np.float32)
    tie_y = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    tie_test_x = np.asarray([[-1], [1.5], [4]], dtype=np.float32)
    tie = RandomForestClassifier(
        **classifier_params(
            n_estimators=1,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features=None,
            bootstrap=False,
        )
    ).fit(tie_x, tie_y)

    single_y = np.ones(4, dtype=np.uint8)
    single = RandomForestClassifier(
        **classifier_params(n_estimators=1, max_features=None, bootstrap=False)
    ).fit(tie_x, single_y)

    logistic_no_intercept_x = np.asarray(
        [[10, 0], [11, 1], [12, 2], [13, 3], [14, 4], [15, 5]],
        dtype=np.float32,
    )
    logistic_no_intercept_y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    logistic_no_intercept_test_x = np.asarray(
        [[9, -1], [11.5, 1.5], [12.5, 2.5], [14, 4], [16, 6]],
        dtype=np.float32,
    )
    logistic_no_intercept = LogisticRegression(
        C=1.0,
        fit_intercept=False,
        max_iter=100,
        tol=1e-8,
        solver="lbfgs",
    ).fit(logistic_no_intercept_x, logistic_no_intercept_y)
    logistic_weights = np.asarray([1, 2, 1, 3, 1, 2], dtype=np.float32)
    logistic_weighted = LogisticRegression(
        C=0.75,
        fit_intercept=True,
        max_iter=100,
        tol=1e-8,
        solver="lbfgs",
    ).fit(
        logistic_no_intercept_x,
        logistic_no_intercept_y,
        sample_weight=logistic_weights,
    )

    linear_full_x = np.asarray([[0, 0], [1, 0], [0, 1], [2, 3]], dtype=np.float32)
    linear_full_y = np.asarray([3, 4, 5, 11], dtype=np.float32)
    linear_test_x = np.asarray([[-1, 2], [3, -2], [4, 5]], dtype=np.float32)
    linear_full = SklearnLinearRegression(tol=1e-6).fit(linear_full_x, linear_full_y)

    linear_rank_deficient_x = np.asarray([[1, 2], [2, 4], [3, 6]], dtype=np.float32)
    linear_rank_deficient_y = np.asarray([1, 2, 3], dtype=np.float32)
    linear_rank_deficient = SklearnLinearRegression(
        fit_intercept=False,
        tol=0.0,
    ).fit(linear_rank_deficient_x, linear_rank_deficient_y)

    linear_weighted_x = np.asarray([[0], [1], [2], [4]], dtype=np.float32)
    linear_weighted_y = np.asarray([1, 2, 2, 5], dtype=np.float32)
    linear_weights = np.asarray([1, 2, 1, 2], dtype=np.float32)
    linear_weighted = SklearnLinearRegression(tol=1e-6).fit(
        linear_weighted_x,
        linear_weighted_y,
        sample_weight=linear_weights,
    )
    ridge_full = SklearnRidge(alpha=1.0, fit_intercept=True).fit(
        linear_full_x, linear_full_y
    )
    ridge_alpha_zero = SklearnRidge(alpha=0.0, fit_intercept=False, solver="svd").fit(
        linear_rank_deficient_x, linear_rank_deficient_y
    )
    ridge_weighted = SklearnRidge(alpha=1.0, fit_intercept=True).fit(
        linear_weighted_x,
        linear_weighted_y,
        sample_weight=linear_weights,
    )
    scaler_x = np.asarray(
        [[1, 2, 5], [3, 4, 5], [5, 8, 5], [7, 10, 5]], dtype=np.float32
    )
    scaler_weights = np.asarray([1, 2, 1, 4], dtype=np.float32)
    scaler_default = StandardScaler().fit(scaler_x)
    scaler_no_mean = StandardScaler(with_mean=False).fit(scaler_x)
    scaler_no_std = StandardScaler(with_std=False).fit(scaler_x)
    scaler_weighted = StandardScaler().fit(scaler_x, sample_weight=scaler_weights)
    hgb_train_x = np.arange(8, dtype=np.float32).reshape(-1, 1)
    hgb_train_y = np.asarray([0, 0, 0, 0, 4, 4, 4, 4], dtype=np.float32)
    hgb_test_x = np.asarray([[-1], [3.5], [3.5001], [8]], dtype=np.float32)
    hgb = HistGradientBoostingRegressor(
        learning_rate=1.0,
        max_iter=1,
        max_leaf_nodes=2,
        max_depth=None,
        min_samples_leaf=1,
        l2_regularization=0.0,
        max_bins=255,
        early_stopping=False,
    ).fit(hgb_train_x, hgb_train_y)
    hgb_quality_nrmse = []
    for seed in (11, 22, 33):
        quality_train_x, quality_train_y, quality_test_x, quality_test_y = regression_data(seed)
        quality_model = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=32,
            max_leaf_nodes=7,
            max_depth=None,
            min_samples_leaf=10,
            l2_regularization=0.0,
            max_bins=64,
            early_stopping=False,
        ).fit(quality_train_x, quality_train_y)
        predictions = quality_model.predict(quality_test_x)
        rmse = float(np.sqrt(np.mean((predictions - quality_test_y) ** 2)))
        scale = float(np.std(quality_test_y))
        hgb_quality_nrmse.append(rmse / scale)

    assert classifier.n_features_in_ == 2
    assert classifier.classes_.tolist() == [0, 1]
    assert classifier.predict_proba(test_x).shape == (5, 2)
    assert regressor.predict(test_x).shape == (5,)
    assert tie.predict_proba(tie_test_x).shape == (3, 2)
    assert tie.predict(tie_test_x).tolist() == [0, 0, 0]
    assert single.classes_.tolist() == [1]
    assert single.predict_proba(tie_test_x).shape == (3, 1)
    assert logistic_no_intercept.intercept_.tolist() == [0.0]
    assert logistic_no_intercept.predict_proba(logistic_no_intercept_test_x).shape == (
        5,
        2,
    )
    assert logistic_weighted.predict_proba(logistic_no_intercept_test_x).shape == (5, 2)
    assert linear_full.rank_ == 2
    assert linear_rank_deficient.rank_ == 1
    assert scaler_default.n_features_in_ == 3
    assert scaler_default.var_[2] == 0.0
    assert scaler_default.scale_[2] == 1.0
    assert hgb.n_iter_ == 1

    result = "// @generated by local reference tooling\n"
    result += "// FerricML-owned frozen black-box outputs.\n"
    result += rust_array("EXACT_TRAIN_X", train_x, "f32")
    result += rust_array("EXACT_CLASSIFIER_Y", classifier_y, "u8")
    result += rust_array("EXACT_REGRESSION_Y", regression_y, "f32")
    result += rust_array("EXACT_TEST_X", test_x, "f32")
    result += rust_array("EXACT_CLASSES", classifier.classes_, "u8")
    result += rust_array("EXACT_LABELS", classifier.predict(test_x), "u8")
    result += rust_array("EXACT_PROBABILITIES", classifier.predict_proba(test_x), "f32")
    result += rust_array("EXACT_REGRESSION", regressor.predict(test_x), "f32")
    result += rust_array("TIE_TRAIN_X", tie_x, "f32")
    result += rust_array("TIE_Y", tie_y, "u8")
    result += rust_array("TIE_TEST_X", tie_test_x, "f32")
    result += rust_array("TIE_CLASSES", tie.classes_, "u8")
    result += rust_array("TIE_LABELS", tie.predict(tie_test_x), "u8")
    result += rust_array("TIE_PROBABILITIES", tie.predict_proba(tie_test_x), "f32")
    result += rust_array("SINGLE_CLASSES", single.classes_, "u8")
    result += rust_array("SINGLE_LABELS", single.predict(tie_test_x), "u8")
    result += rust_array("SINGLE_PROBABILITIES", single.predict_proba(tie_test_x), "f32")
    result += rust_array("LOGISTIC_NO_INTERCEPT_TRAIN_X", logistic_no_intercept_x, "f32")
    result += rust_array("LOGISTIC_NO_INTERCEPT_Y", logistic_no_intercept_y, "u8")
    result += rust_array(
        "LOGISTIC_NO_INTERCEPT_TEST_X", logistic_no_intercept_test_x, "f32"
    )
    result += rust_array(
        "LOGISTIC_NO_INTERCEPT_COEFFICIENTS", logistic_no_intercept.coef_, "f32"
    )
    result += rust_array(
        "LOGISTIC_NO_INTERCEPT_DECISIONS",
        logistic_no_intercept.decision_function(logistic_no_intercept_test_x),
        "f32",
    )
    result += rust_array(
        "LOGISTIC_NO_INTERCEPT_PROBABILITIES",
        logistic_no_intercept.predict_proba(logistic_no_intercept_test_x),
        "f32",
    )
    result += rust_array("LOGISTIC_WEIGHTS", logistic_weights, "f32")
    result += rust_array(
        "LOGISTIC_WEIGHTED_COEFFICIENTS", logistic_weighted.coef_, "f32"
    )
    result += rust_array(
        "LOGISTIC_WEIGHTED_INTERCEPT", logistic_weighted.intercept_, "f32"
    )
    result += rust_array(
        "LOGISTIC_WEIGHTED_DECISIONS",
        logistic_weighted.decision_function(logistic_no_intercept_test_x),
        "f32",
    )
    result += rust_array(
        "LOGISTIC_WEIGHTED_PROBABILITIES",
        logistic_weighted.predict_proba(logistic_no_intercept_test_x),
        "f32",
    )
    result += rust_array("LINEAR_FULL_X", linear_full_x, "f32")
    result += rust_array("LINEAR_FULL_Y", linear_full_y, "f32")
    result += rust_array("LINEAR_TEST_X", linear_test_x, "f32")
    result += rust_array("LINEAR_FULL_COEFFICIENTS", linear_full.coef_, "f32")
    result += rust_array("LINEAR_FULL_INTERCEPT", np.atleast_1d(linear_full.intercept_), "f32")
    result += rust_array("LINEAR_FULL_PREDICTIONS", linear_full.predict(linear_test_x), "f32")
    result += rust_array("LINEAR_RANK_DEFICIENT_X", linear_rank_deficient_x, "f32")
    result += rust_array("LINEAR_RANK_DEFICIENT_Y", linear_rank_deficient_y, "f32")
    result += rust_array(
        "LINEAR_RANK_DEFICIENT_COEFFICIENTS", linear_rank_deficient.coef_, "f32"
    )
    result += rust_array("LINEAR_WEIGHTED_X", linear_weighted_x, "f32")
    result += rust_array("LINEAR_WEIGHTED_Y", linear_weighted_y, "f32")
    result += rust_array("LINEAR_WEIGHTS", linear_weights, "f32")
    result += rust_array("LINEAR_WEIGHTED_COEFFICIENTS", linear_weighted.coef_, "f32")
    result += rust_array(
        "LINEAR_WEIGHTED_INTERCEPT", np.atleast_1d(linear_weighted.intercept_), "f32"
    )
    result += rust_array("RIDGE_FULL_COEFFICIENTS", ridge_full.coef_, "f32")
    result += rust_array("RIDGE_FULL_INTERCEPT", np.atleast_1d(ridge_full.intercept_), "f32")
    result += rust_array("RIDGE_FULL_PREDICTIONS", ridge_full.predict(linear_test_x), "f32")
    result += rust_array("RIDGE_ALPHA_ZERO_COEFFICIENTS", ridge_alpha_zero.coef_, "f32")
    result += rust_array("RIDGE_WEIGHTED_COEFFICIENTS", ridge_weighted.coef_, "f32")
    result += rust_array(
        "RIDGE_WEIGHTED_INTERCEPT", np.atleast_1d(ridge_weighted.intercept_), "f32"
    )
    result += rust_array("SCALER_TRAIN_X", scaler_x, "f32")
    result += rust_array("SCALER_WEIGHTS", scaler_weights, "f32")
    result += rust_array("SCALER_DEFAULT_MEAN", scaler_default.mean_, "f64")
    result += rust_array("SCALER_DEFAULT_VARIANCE", scaler_default.var_, "f64")
    result += rust_array("SCALER_DEFAULT_SCALE", scaler_default.scale_, "f64")
    result += rust_array(
        "SCALER_DEFAULT_TRANSFORMED", scaler_default.transform(scaler_x), "f32"
    )
    result += rust_array(
        "SCALER_NO_MEAN_TRANSFORMED", scaler_no_mean.transform(scaler_x), "f32"
    )
    result += rust_array(
        "SCALER_NO_STD_TRANSFORMED", scaler_no_std.transform(scaler_x), "f32"
    )
    result += rust_array("SCALER_WEIGHTED_MEAN", scaler_weighted.mean_, "f64")
    result += rust_array("SCALER_WEIGHTED_VARIANCE", scaler_weighted.var_, "f64")
    result += rust_array("SCALER_WEIGHTED_SCALE", scaler_weighted.scale_, "f64")
    result += rust_array(
        "SCALER_WEIGHTED_TRANSFORMED", scaler_weighted.transform(scaler_x), "f32"
    )
    result += rust_array("HGB_TRAIN_X", hgb_train_x, "f32")
    result += rust_array("HGB_TRAIN_Y", hgb_train_y, "f32")
    result += rust_array("HGB_TEST_X", hgb_test_x, "f32")
    result += rust_array("HGB_PREDICTIONS", hgb.predict(hgb_test_x), "f32")
    result += rust_array(
        "HGB_QUALITY_NRMSE", np.asarray(hgb_quality_nrmse, dtype=np.float64), "f64"
    )
    return result


def quality_fixture() -> str:
    rows: list[str] = []
    for lane in ("nonlinear", "separable", "imbalanced", "noise"):
        for seed in QUALITY_SEEDS:
            train_x, train_y, test_x, test_y = classification_data(lane, seed)
            model = RandomForestClassifier(**classifier_params(random_state=seed)).fit(train_x, train_y)
            labels = model.predict(test_x)
            probabilities = model.predict_proba(test_x)[:, 1]
            accuracy = float(np.mean(labels == test_y))
            brier = float(np.mean((probabilities - test_y) ** 2))
            rows.append(
                f'    QualityReference {{ lane: "{lane}", seed: {seed}, accuracy: {rust_float(accuracy)}, brier: {rust_float(brier)}, nrmse: 0.0 }},'
            )

    for seed in QUALITY_SEEDS:
        train_x, train_y, test_x, test_y = regression_data(seed)
        model = RandomForestRegressor(**regressor_params(random_state=seed)).fit(train_x, train_y)
        predictions = model.predict(test_x)
        rmse = float(np.sqrt(np.mean((predictions - test_y) ** 2)))
        nrmse = rmse / float(np.std(test_y))
        rows.append(
            f'    QualityReference {{ lane: "regression", seed: {seed}, accuracy: 0.0, brier: 0.0, nrmse: {rust_float(nrmse)} }},'
        )

    return """

#[derive(Clone, Copy, Debug)]
pub struct QualityReference {
    pub lane: &'static str,
    pub seed: u64,
    pub accuracy: f64,
    pub brier: f64,
    pub nrmse: f64,
}

pub const QUALITY_REFERENCES: &[QualityReference] = &[
""" + "\n".join(rows) + "\n];\n"


def generated_fixture() -> str:
    validate_reference_api()
    return exact_fixture() + quality_fixture()


def quality_references(fixture: str) -> dict[tuple[str, int], tuple[float, float, float]]:
    references = {
        (lane, int(seed)): (float(accuracy), float(brier), float(nrmse))
        for lane, seed, accuracy, brier, nrmse in QUALITY_PATTERN.findall(fixture)
    }
    expected_count = len(QUALITY_SEEDS) * 5
    if len(references) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} quality references, found {len(references)}"
        )
    return references


def validate_quality_portability(frozen: str, generated: str) -> None:
    frozen_references = quality_references(frozen)
    generated_references = quality_references(generated)
    if frozen_references.keys() != generated_references.keys():
        raise RuntimeError("generated quality lanes or seeds differ from the fixture")

    failures: list[str] = []
    for key in sorted(frozen_references):
        frozen_accuracy, frozen_brier, frozen_nrmse = frozen_references[key]
        accuracy, brier, nrmse = generated_references[key]
        if key[0] == "regression":
            delta = abs(nrmse - frozen_nrmse)
            if delta > QUALITY_NRMSE_PORTABILITY:
                failures.append(
                    f"{key}: nRMSE drift {delta:.6g} exceeds "
                    f"{QUALITY_NRMSE_PORTABILITY}"
                )
        else:
            accuracy_delta = abs(accuracy - frozen_accuracy)
            brier_delta = abs(brier - frozen_brier)
            if accuracy_delta > QUALITY_ACCURACY_PORTABILITY:
                failures.append(
                    f"{key}: accuracy drift {accuracy_delta:.6g} exceeds "
                    f"{QUALITY_ACCURACY_PORTABILITY}"
                )
            if brier_delta > QUALITY_BRIER_PORTABILITY:
                failures.append(
                    f"{key}: Brier drift {brier_delta:.6g} exceeds "
                    f"{QUALITY_BRIER_PORTABILITY}"
                )
    if failures:
        raise RuntimeError(
            "quality fixture left its portability envelope:\n" + "\n".join(failures)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="explicitly replace the frozen project-owned fixture",
    )
    args = parser.parse_args()
    self_test_exact_portability()
    generated = generated_fixture()
    if args.update:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(generated, encoding="utf-8")
        print(f"updated {FIXTURE.relative_to(ROOT)}")
        return 0

    if not FIXTURE.exists():
        print(f"missing fixture: {FIXTURE}; review and run with --update", file=sys.stderr)
        return 1
    frozen = FIXTURE.read_text(encoding="utf-8")
    frozen_exact, marker, _ = frozen.partition(QUALITY_MARKER)
    generated_exact, generated_marker, _ = generated.partition(QUALITY_MARKER)
    if not marker or not generated_marker:
        print("fixture is missing the quality-reference boundary", file=sys.stderr)
        return 1
    try:
        validate_exact_portability(frozen_exact, generated_exact)
    except RuntimeError as error:
        diff = difflib.unified_diff(
            frozen_exact.splitlines(),
            generated_exact.splitlines(),
            fromfile="frozen",
            tofile="generated",
            lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
        print(error, file=sys.stderr)
        print("review before using --update", file=sys.stderr)
        return 1
    try:
        validate_quality_portability(frozen, generated)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print(
        f"verified {FIXTURE.relative_to(ROOT)} with "
        f"scikit-learn {sklearn.__version__} / NumPy {np.__version__}; "
        "exact contracts and portable fitted outputs match; quality is within "
        "the portability envelope"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
