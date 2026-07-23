# Model artifact envelope

FerricML writes artifact envelope version 2 for fitted logistic, linear, ridge,
standard-scaler, and supported typed pipeline models. It continues to read the
legacy version-1 logistic format. The private forest representation remains
outside the persistence contract; no byte sequence produced from packed trees
is a compatibility promise.

`LogisticRegression::to_artifact` writes, in little-endian order:

- the eight-byte `FERRICML` magic and envelope version `2`;
- the never-reused estimator kind and independent payload version;
- required flags, declared payload length, schema-record count, and a zero
  reserved field;
- role-tagged, 32-byte feature-schema identities;
- length-delimited typed components containing fitted dimensions, parameters,
  intercept, and ordered `f32` coefficients;
- a SHA-256 integrity footer covering every preceding byte.

The current writer uses no required flags, one input-schema record, and one
logistic-state component. Unknown required flags, payload versions, schema
roles, nonzero reserved fields, and component versions are rejected rather
than guessed. The hard encoded-size limit is 32 MiB.

`LogisticRegression::from_artifact` checks the size and checksum before parsing
counts or model state, requires the expected input-schema identity, validates
all declared lengths before borrowing component payloads, bounds feature
allocation, rejects non-finite or inconsistent values, and rejects trailing
bytes. SHA-256 provides corruption detection only; artifacts are not signed or
authenticated.

Standalone `StandardScaler` artifacts carry ordered input and transformed
schema records plus fitted `f64` mean, population variance, and scale values.
The three supported typed pipeline artifacts retain those same schema roles
and contain length-delimited complete scaler and estimator artifacts. Nested
checks make every component independently typed and schema-bound; decode also
validates the fitted feature-width handoff before constructing the pipeline.
This is intentionally not a generic serialization trait: unsupported pipeline
shapes have no persistence API.

`PairwiseLinearRanker` has its own never-reused estimator kind. Its metadata
component freezes the pairwise logistic objective and mirrored-normalization
versions, ranker parameters, and feature width. A separate nested logistic
artifact carries the no-intercept fitted coefficients. Decode requires exact
agreement between both components, including an exact positive-zero
intercept, before exposing item scoring.

Additional estimator payloads must carry:

- a never-reused estimator kind and independent payload version;
- ordered feature names plus a canonical feature-schema hash;
- every effective fitted parameter, including deterministic seed and training
  parallelism metadata where relevant;
- sorted classifier classes, when applicable;
- ordered preprocessing steps and their fitted state;
- optional calibration method/state and decision-threshold metadata;
- explicit scalar representation and byte order;
- declared component and payload lengths covered by the outer checksum.

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
