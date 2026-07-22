# Model artifact envelope

FerricML artifact version 1 supports fitted logistic-regression models. The
private forest representation remains outside the persistence contract; no
byte sequence produced from packed trees is a compatibility promise.

`LogisticRegression::to_artifact` writes, in little-endian order, the FerricML
magic, envelope version, estimator kind, caller-supplied 32-byte feature-schema
identity, fitted dimensions and parameters, intercept, and ordered `f32`
coefficients. A SHA-256 footer covers every preceding byte.
`LogisticRegression::from_artifact` verifies the checksum before parsing,
requires the expected feature-schema identity, bounds feature allocation,
rejects non-finite or inconsistent values, and rejects trailing bytes.

A future multi-backend envelope must additionally carry:

- an envelope format version and estimator kind;
- the estimator/model payload version, independent of the envelope version;
- ordered feature names plus a canonical feature-schema hash;
- every effective fitted parameter, including deterministic seed and training
  parallelism metadata where relevant;
- sorted classifier classes, when applicable;
- ordered preprocessing steps and their fitted state;
- optional calibration method/state and decision-threshold metadata;
- explicit scalar representation and byte order;
- payload length and a checksum covering header metadata and payload.

Every envelope must use fixed-width encodings, reject unknown required fields,
validate dimensions and numeric finiteness, and verify its checksum before any
model allocation is trusted. Optional metadata needs length-delimited fields so
readers can skip additions they understand to be non-semantic.

An estimator payload is backend-neutral: it describes mathematical model
state rather than serializing a Rust enum, Python pickle, or third-party type.
Field IDs and migration rules for additional estimator kinds remain open until
their first reader/writer is implemented.

The private compact forest-node layout is explicitly unfrozen. Node compaction
and traversal layout can change without an artifact migration until a logical
tree payload and round-trip tests are defined.
