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

A joint multinomial fit is a different model from a binary one, so it is a
second payload schema under the same estimator kind rather than a widening of
the first. Payload version `1` stores one coefficient row and one intercept;
payload version `2` stores the observed class list, one intercept per class, and
one coefficient row per class, and requires at least two classes. Every binary
artifact therefore keeps its exact bytes and its exact reader. Decoding reads
the recorded payload version from the header to select which reader runs — a
two-byte selection made before anything is hashed, so a hostile buffer is never
hashed twice — and the selected reader then re-validates that same field along
with the size limit, checksum, magic, kind, and schema identity. Class labels
are stored as fixed-width words in sorted, deduplicated order, which is the only
order decode accepts, so a model has exactly one encoding and its probability
columns cannot be permuted by a rewrite.

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
records before constructing compact runtime trees. Canonical means the pre-order
the writer produces: a branch's left child is the next record and its right
child is the record after that branch's whole left subtree, so a tree has
exactly one accepted record order and an artifact is a canonical name for the
model it holds. The private compact node
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
could not stay finite.

`RandomForestClassifier` artifacts have their own never-reused kind and carry
both fitted leaf representations under one payload version. Their metadata adds
a leaf-arithmetic tag and the sorted class list to the regressor's fields, and
the tag is read before any tree is, so neither flavour's trees ever reach the
other's builder. A binary fit reuses the scalar logical-tree records unchanged,
and its leaves must be the `0..=1` a fitted class probability is; its class list
must be a sorted subset of `{0, 1}`, because the scalar leaf is the probability
of class `1` and nothing else can be read out of it. A multiclass fit writes the
same topology records with a **reserved zero** where a scalar leaf carries its
value, followed by one length-delimited probability block per tree holding that
tree's leaf distributions. Those distributions are ordered by the leaf's
pre-order rank rather than by the runtime leaf ordinal: rank is determined by
the topology alone, so an artifact cannot be rewritten by permuting the ordinals
and the block together. Every declared count is checked against the bytes
present before anything is reserved, every distribution entry must be a finite
`0..=1`, and each decoded tree re-enters the same topology and class-topology
validators fitting uses.

`AnyRegressor` artifacts are dispatch envelopes rather than model formats. They
carry a dispatch version and a variant tag, then the selected estimator's own
complete artifact nested whole and length-delimited. Decoding hands the nested
bytes back to that estimator, so it is checksummed, schema-bound, and validated
exactly as it would be standalone; a variant tag that disagrees with the nested
payload fails the nested kind check instead of being reinterpreted. Adding a
runtime variant therefore never perturbs an existing estimator's payload.

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
