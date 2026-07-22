# Random forest green-room record

FerricML's forest implementation is independently written. This record keeps
the behavior research separate from the implementation work.

## Specification-room input

On 2026-07-21, the specification room inspected scikit-learn at commit
`bf12ea1dab8b6343076cac6fbfabd2b8095c13dc` through the read-only open-source
MCP code graph. scikit-learn is BSD-3-Clause licensed. No source file was copied
into the FerricML workspace.

The externally observable behavior used for the FerricML specification is:

- random forests train independent decision trees on bootstrap samples;
- a bootstrap sample normally contains as many draws as the source dataset;
- duplicate bootstrap draws act as integer row weights during tree growth;
- dense training input is represented as `f32`;
- binary classification trees use class proportions at each leaf;
- forest class probabilities are the arithmetic mean of tree probabilities;
- regression trees use a mean target at each leaf;
- forest regression predictions are the arithmetic mean of tree predictions;
- a reference split oracle enumerates candidate feature partitions and selects
  the partition with the smallest weighted child impurity;
- the relevant impurities are Gini for classification and squared error for
  regression;
- classifier defaults conventionally use `sqrt(feature_count)` candidates per
  node, while regressor defaults consider every feature.

## Independent implementation boundary

The implementation room received only the behavior above plus FerricML's own
API, determinism, safety, and performance requirements. It did not receive or
inspect scikit-learn source. FerricML does not reproduce third-party names,
comments, private data structures, RNG streams, serialization, or binary
artifacts.

FerricML deliberately defines its own behavior where compatibility is not
required:

- a documented cross-platform RNG and per-tree seed derivation;
- byte-identical tree order across worker counts;
- finite dense row-major `f32` input only in version 1;
- packed FerricML-owned private nodes and a future backend-neutral artifact;
- allocation-free hot inference;
- typed errors instead of assertion-driven failure;
- no missing/categorical values, pruning, OOB scoring, monotonic constraints,
  multi-output targets, or class/sample weights in version 1.

Compatibility tests cover independently constructed partitions,
probabilities, predictions, bootstrap semantics, and stopping rules. They do
not embed third-party fixtures or serialized models.

As a black-box confirmation after implementation, the independently derived
four-row classification and regression stump oracle was also run against the
public scikit-learn 1.9.0 API. Both implementations selected threshold `1.5`;
the classifier produced pure `0.0`/`1.0` leaves and the regressor produced
means `1.5`/`3.5`. This check supplied no implementation details or reusable
artifact to the implementation room.
