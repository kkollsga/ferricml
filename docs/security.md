# Dependency security review

Run `cargo audit` as an informational local gate. Vulnerability findings block
release work until resolved or explicitly reviewed. Maintenance-only warnings
may be accepted temporarily when the dependency is transitive, no supported
upgrade removes it, and the warning does not describe a known vulnerability.

## The numerical dependency

FerricML has two runtime dependencies: `faer` for the dense decompositions and
`sha2` for artifact digests. `faer` replaced `nalgebra 0.34.2` on 2026-07-27,
and the reason was **correctness, not performance or supply chain** — the
previous backend returned an SVD that does not reconstruct its own input on
exactly-rank-deficient tall designs. The changelog carries the measurement. That
matters here because it inverts an argument this document used to make.

Measured on the registered `aarch64-apple-darwin` runner, the shipping graph
(normal plus build dependencies, development dependencies excluded) is 56
crates. Every one of them can be taken under MIT, with a single build-time
exception: `unicode-ident` is `(MIT OR Apache-2.0) AND Unicode-3.0`, so the
Unicode licence applies alongside whichever of the pair is chosen. Nothing in
the graph is copyleft.

## Reviewed maintenance warning

- `RUSTSEC-2024-0436` applies to `paste 1.0.15`, a build-time macro crate
  FerricML never names directly. The advisory reports that the crate is
  unmaintained; it reports no vulnerability.
- **How it is reached, read off the dependency graph rather than assumed.**
  Before the backend change it came in through `nalgebra 0.34.2` and
  `simba 0.9.1`. It now comes in through `faer 0.24.4` and `gemm 0.19.0`, and
  through the four `gemm-*` kernel crates beside it. The path changed; the
  presence did not.
- **The earlier assessment is withdrawn in its premise and kept in its
  conclusion.** This review used to reason that replacing the numerical backend
  solely to drop this macro "would carry disproportionate numerical and
  compatibility risk". That reasoning is now known to have been backwards: the
  backend was replaced, and it was replaced because keeping it was the numerical
  risk. The conclusion survives on a plainer fact than the one it used to rest
  on — the new backend depends on `paste` as well, so no backend choice
  available to FerricML removes it, and a swap made for this advisory alone
  would buy nothing.
- **The general lesson, since this review got it wrong once.** "Changing this
  dependency is riskier than living with the warning" is a claim about the
  dependency's quality, and it needs evidence of that quality rather than the
  inertia of an incumbent. Nothing had measured the incumbent when that sentence
  was written.
- Review again by 2026-10-24, or earlier when the numerical dependency updates.
