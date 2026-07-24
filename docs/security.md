# Dependency security review

Run `cargo audit` as an informational local gate. Vulnerability findings block
release work until resolved or explicitly reviewed. Maintenance-only warnings
may be accepted temporarily when the dependency is transitive, no supported
upgrade removes it, and the warning does not describe a known vulnerability.

## Reviewed maintenance warning

- `RUSTSEC-2024-0436` applies to `paste 1.0.15`, reached transitively through
  `nalgebra 0.34.2` and `simba 0.9.1`.
- The advisory reports that the crate is unmaintained; it does not report a
  vulnerability. FerricML does not depend on it directly.
- Replacing the numerical backend solely to remove this transitive build-time
  macro would carry disproportionate numerical and compatibility risk.
- Review again by 2026-10-24, or earlier when the numerical dependency updates.
