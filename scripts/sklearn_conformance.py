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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

REFERENCE_SKLEARN = "1.9.0"
REFERENCE_NUMPY = "2.4.1"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sklearn_1_9.rs"
MASK = (1 << 64) - 1
QUALITY_SEEDS = (11, 22, 33, 44, 55)
QUALITY_MARKER = "\n#[derive(Clone, Copy, Debug)]\npub struct QualityReference"
QUALITY_PATTERN = re.compile(
    r'QualityReference \{ lane: "([^"]+)", seed: (\d+), '
    r"accuracy: ([^,]+), brier: ([^,]+), nrmse: ([^ }]+) \},"
)

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
    else:
        encoded = ", ".join(str(int(value)) for value in flat)
    return f"pub const {name}: &[{rust_type}] = &[{encoded}];\n"


def validate_reference_api() -> None:
    if sklearn.__version__ != REFERENCE_SKLEARN:
        raise RuntimeError(
            f"expected scikit-learn {REFERENCE_SKLEARN}, got {sklearn.__version__}"
        )
    if np.__version__ != REFERENCE_NUMPY:
        raise RuntimeError(f"expected NumPy {REFERENCE_NUMPY}, got {np.__version__}")

    classifier = RandomForestClassifier()
    regressor = RandomForestRegressor()
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

    result = "// @generated by scripts/sklearn_conformance.py --update\n"
    result += "// Public scikit-learn APIs only; project-owned black-box outputs.\n\n"
    result += f'pub const SKLEARN_VERSION: &str = "{sklearn.__version__}";\n'
    result += f'pub const NUMPY_VERSION: &str = "{np.__version__}";\n'
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
        "LOGISTIC_NO_INTERCEPT_PROBABILITIES",
        logistic_no_intercept.predict_proba(logistic_no_intercept_test_x),
        "f32",
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
    if frozen_exact != generated_exact:
        diff = difflib.unified_diff(
            frozen_exact.splitlines(),
            generated_exact.splitlines(),
            fromfile="frozen",
            tofile="generated",
            lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
        print("exact fixture drifted; review before using --update", file=sys.stderr)
        return 1
    try:
        validate_quality_portability(frozen, generated)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print(
        f"verified {FIXTURE.relative_to(ROOT)} with "
        f"scikit-learn {sklearn.__version__} / NumPy {np.__version__}; "
        "exact outputs match and quality is within the portability envelope"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
