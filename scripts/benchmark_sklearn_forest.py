#!/usr/bin/env python3
"""Matched scikit-learn side of FerricML's forest comparison gate."""

import json
import pickle
import statistics
import time

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

MASK = (1 << 64) - 1
SEED = 42


class SplitMix64:
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


def fixture(rows: int, columns: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = SplitMix64(seed)
    values = np.empty((rows, columns), dtype=np.float32)
    labels = np.empty(rows, dtype=np.uint8)
    for row_index in range(rows):
        for column in range(columns):
            values[row_index, column] = rng.signed_unit()
        row = values[row_index]
        score = np.float32(
            np.float32(1.4) * row[0] * row[1]
            + np.float32(0.9) * row[2]
            - np.float32(0.8) * row[3] * row[3]
            + np.float32(0.5) * np.sin(np.float32(3.0) * row[4])
            + np.float32(0.25) * row[5] * row[6]
        )
        labels[row_index] = score > np.float32(0.0)
    return values, labels


def model(trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        criterion="gini",
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=SEED,
        n_jobs=1,
    )


def median_seconds(action, samples: int) -> float:
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        action()
        timings.append((time.perf_counter_ns() - started) / 1_000_000_000)
    return statistics.median(timings)


def main() -> None:
    quality_x, quality_y = fixture(4096, 32, 0x12345678)
    quality_test_x, quality_test_y = fixture(2048, 32, 0x87654321)
    quality_model = model(100).fit(quality_x, quality_y)
    quality_predictions = quality_model.predict(quality_test_x)
    quality_probabilities = quality_model.predict_proba(quality_test_x)[:, 1]

    train_x, train_y = fixture(2048, 64, 0xFEEDBEEF)
    fit_seconds = median_seconds(lambda: model(20).fit(train_x, train_y), 30)

    prediction_model = model(100).fit(train_x, train_y)
    test_x, _ = fixture(1024, 64, 0xDECAFBAD)
    prediction_model.predict(test_x)
    predict_seconds = median_seconds(lambda: prediction_model.predict(test_x), 200)
    prediction_model.predict_proba(test_x)
    proba_seconds = median_seconds(lambda: prediction_model.predict_proba(test_x), 200)

    print(
        json.dumps(
            {
                "implementation": f"scikit-learn-{sklearn.__version__}",
                "protocol": {
                    "threads": 1,
                    "max_depth": 12,
                    "max_features": "sqrt",
                    "bootstrap": True,
                },
                "quality": {
                    "train_rows": 4096,
                    "test_rows": 2048,
                    "features": 32,
                    "trees": 100,
                    "accuracy": float(np.mean(quality_predictions == quality_test_y)),
                    "brier": float(
                        np.mean((quality_probabilities - quality_test_y) ** 2)
                    ),
                    "pickle_bytes": len(pickle.dumps(quality_model, protocol=5)),
                },
                "fit": {
                    "train_rows": 2048,
                    "features": 64,
                    "trees": 20,
                    "median_ms": fit_seconds * 1000.0,
                },
                "predict": {
                    "rows": 1024,
                    "features": 64,
                    "trees": 100,
                    "median_us": predict_seconds * 1_000_000.0,
                    "rows_per_second": 1024.0 / predict_seconds,
                },
                "predict_proba": {
                    "rows": 1024,
                    "features": 64,
                    "trees": 100,
                    "median_us": proba_seconds * 1_000_000.0,
                    "rows_per_second": 1024.0 / proba_seconds,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
