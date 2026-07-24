# First-party model performance

FerricML measures its own release builds with fixed synthetic fixtures and
Criterion. These lanes are diagnostics and longitudinal regression evidence,
not shared-CI timing gates or comparisons with third-party crates.

## Registered workloads

The `models` target measures 1,024-row caller-owned inference and 2,048x48 fit
for linear regression, ridge, pairwise linear ranking, and a fitted
standard-scaler/ridge pipeline. It also measures MSE, tie-aware ROC AUC, and a
seeded holdout over 4,096 rows plus five-fold ridge cross-validation over a
256x12 fixture. The `boosting` target measures single-row
prediction for 32x7, 64x7, 64x15, and 128x15 tree/leaf limits; 32-row and
1,024-row caller-owned prediction for 64x7; and 2,048x48 fitting for 64x7.
All data, targets, pair construction, parameters, and model schemas are fixed
by the benchmark source.

The boosted fixtures produce these actual persisted model sizes:

| Trees x leaf limit | Logical nodes | Artifact bytes |
| ---: | ---: | ---: |
| 32x7 | 416 | 9,108 |
| 64x7 | 832 | 18,068 |
| 64x15 | 1,856 | 38,548 |
| 128x15 | 3,712 | 76,948 |

Logical-node counts describe the stable artifact records, not the private
compact runtime layout.

## 2026-07-23 diagnostic

On the registered Apple M4 runner with Rust 1.97.0, three pre-run CPU-idle
samples were 96.36%, 97.14%, and 96.99%. With 20 samples, a one-second warmup,
and two-second measurement target, median 64x7 single-row prediction was 180
ns. The other single-row medians were 92 ns (32x7), 328 ns (64x15), and 683 ns
(128x15). The approximately 1,000-node diagnostic objective of 2 microseconds
was therefore met without a performance-specific implementation change.

The same run measured 64x7 caller-owned prediction at 6.27 microseconds for 32
rows and 279 microseconds for 1,024 rows, and fitting at 77.7 milliseconds.
Machine-readable results and raw Criterion output remain in the local
`dev-docs/bench/results/` and `dev-docs/bench/out/` lifecycle tiers.

## Compatible history

`scripts/performance_history.py` records named suites. `forest-v1` retains
compatibility with the release-0.1 `forest-history-v1` record, while
`ferricml-models-v1` begins a separate history. Comparisons operate on matching
suite protocols and metric intersections. A new or missing lane is reported as
`insufficient_history`, never as a pass; shared lanes still expose any real
regression. Each suite independently checks the prior release and, once three
compatible earlier records exist, an approximately three-release anchor.

`make bench-history` creates an immutable versioned release record.
`make bench-diagnostic` writes dated evidence without occupying that version.
Both require the configured registered-runner identity, reject active build or
benchmark processes, and require three CPU-idle samples of at least 90%.
