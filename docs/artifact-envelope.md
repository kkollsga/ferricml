# Model artifact envelope

FerricML writes artifact envelope version 2 for fitted logistic, linear, ridge,
histogram-gradient-boosting, random-forest regression, standard-scaler, and
supported typed pipeline models. It continues to read the legacy version-1
logistic format. The private packed forest representation remains outside the
persistence contract: forests persist as backend-neutral logical trees, and no
byte sequence produced from packed forest nodes is a compatibility promise.

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

`HistGradientBoostingRegressor` artifacts freeze the squared-error objective,
fitted baseline, effective parameters, feature width, tree count, and total
logical-node count. Each tree is a length-delimited preorder sequence of
backend-neutral branch and leaf records. Branches encode feature, threshold,
and logical child indices; leaves encode only their prediction value. The
reader bounds tree and aggregate node counts before allocation, requires a
canonical full binary topology, validates finite values and feature indices,
and rejects unreachable, repeated, cyclic, over-depth, truncated, or trailing
records before constructing compact runtime trees. The private compact node
representation and traversal layout remain free to change.

`RandomForestRegressor` artifacts use the same logical-tree records. Their
metadata component freezes the averaging objective, feature width, every
retained parameter — estimator count, depth and sample-size limits, the
feature-selection policy, bootstrap flag, deterministic seed, and requested
training parallelism — plus the tree count and total logical-node count. The
packed layout stores leaves inline in their parent's flag bits, so encoding
synthesizes the leaf records and decoding rebuilds builder nodes that re-enter
the same topology validator fitting uses. The reader bounds feature width, tree
count, and aggregate node count before allocating, rejects parameter tags and
counts it does not recognize, and refuses a forest whose averaged prediction
could not stay finite. Classifier persistence remains unsupported until the
multiclass leaf and probability semantics are frozen.

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

The private compact forest-node layout remains explicitly unfrozen. Histogram
boosting and random-forest regression both persist the logical tree contract
described above, never their compact runtime representations.
