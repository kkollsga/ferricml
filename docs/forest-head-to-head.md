# Random-forest performance contract

FerricML locks matched public operations against Rafor 0.3 and its own
historical medians. The external harness is a standalone, opt-in Cargo package
under `benchmarks/alternatives`; competitor crates are absent from FerricML's
features and therefore from root `--all-features` and `--all-targets` gates.
The package is not invoked by normal CI; it is a quick manual diagnostic when
forest code, compiler output, or the target workload changes.

## Locked protocol

Both implementations fit binary classifiers with one thread, 64 finite `f32`
features, Gini splits, depth 12, minimum split 2, minimum leaf 1, bootstrap
sampling, square-root feature selection, 100 prediction trees, and seed 42.
Fit uses 2,048 rows and 20 trees. Inference locks 1, 32, and 1,024 rows.
Their owned RNGs and split implementations produce different tree topologies,
so this is an end-to-end public-operation contract, not a per-branch kernel
comparison. Quality remains locked separately against scikit-learn.

Operations are compared by returned meaning and shape:

- labels: allocated label vector against allocated label vector;
- full probabilities: allocated `rows x 2` matrix against the same;
- class probability: allocated positive-class column against the same. Rafor
  must first allocate its full matrix because it has no column API;
- FerricML's caller-owned label/full/column `_into` methods are recorded as
  separate, zero-allocation historical lanes. Rafor 0.3 has no equivalent.

Criterion uses medians from 20 samples after a one-second warmup and a two-
second measurement target. Every matched inference lane must be at most 1.5x
Rafor; fit must also be at most 1.5x. Every FerricML inference median must be at
most 1.10x its recorded reference median and fit at most 1.15x. Geometric means
are informational and never hide a failing lane.

Timing enforcement requires both `--enforce` and the matching stable runner
ID/fingerprint. CI neither builds nor runs the third-party package. Manual runs
without `--enforce` report only. Investigate a failing stable run under the
same idle/power conditions, then repeat it once. A reproducible second miss is
a regression; thresholds and baselines are not adjusted to pass it.

## Reference result

Recorded 2026-07-21 on Apple M4 arm64, Rust 1.97.0. Raw Criterion output,
machine metadata, and the evaluation are retained under the gitignored
`dev-docs/bench/out/20260721T112620Z-forest-contract/`.

| Operation | Rows | FerricML | Rafor | Ratio |
|---|---:|---:|---:|---:|
| labels | 1 | 0.877 us | 2.891 us | 0.303x |
| labels | 32 | 23.873 us | 33.951 us | 0.703x |
| labels | 1,024 | 3.598 ms | 3.175 ms | 1.133x |
| full probability | 1 | 0.854 us | 2.908 us | 0.294x |
| full probability | 32 | 24.248 us | 33.929 us | 0.715x |
| full probability | 1,024 | 3.512 ms | 3.202 ms | 1.097x |
| class probability | 1 | 0.869 us | 2.932 us | 0.296x |
| class probability | 32 | 23.993 us | 34.431 us | 0.697x |
| class probability | 1,024 | 3.644 ms | 3.187 ms | 1.143x |
| fit, 20 trees | 2,048 | 36.175 ms | 27.722 ms | 1.305x |

All external and historical gates passed. The maximum matched inference ratio
was 1.143x and the informational inference geometric mean was 0.618x.

The repaired harness initially exposed a 32-row FerricML miss of roughly 2.3x.
Tree-major batch accumulation reduced repeated model-cache churn, but the lane
still exceeded 1.5x. Profiling the representation showed that fetching an
explicit leaf node/array entry dominated the small batch. Fitted branches now
store leaf `f32` bits directly in child slots with two private flag bits. This
keeps a branch at 16 bytes, avoids the leaf fetch, preserves exact public
predictions, and moved the 32-row ratios below 0.72x.

## Reproduction

Report-only, with raw evidence retained automatically:

```console
python3 scripts/run_forest_performance.py
```

Enforce on the registered stable runner:

```console
python3 scripts/run_forest_performance.py --enforce --runner-id apple-m4-local
```

The contract and baselines live in
`benchmarks/forest-performance-contract.json`; the evaluator can also inspect
an existing Criterion directory directly with
`scripts/evaluate_forest_performance.py`.
