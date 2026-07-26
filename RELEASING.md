# Releasing FerricML

FerricML is a Rust-only crate. It publishes to crates.io only from a
`v<version>` tag whose version matches `Cargo.toml`. The tagged commit must be
the exact current `main` tip, and the changelog must contain a matching release
heading. Use the repository's `release` skill; these notes document its gates,
not an alternate manual path.

## One-time setup

Store a crates.io API token as the repository Actions secret
`CARGO_REGISTRY_TOKEN`:

```console
gh secret set CARGO_REGISTRY_TOKEN --repo kkollsga/ferricml
```

Paste the token at the prompt. Never put the token in a tracked file, command
argument, workflow log, or GitHub release description.

## Cut a release

1. Goal-check the active phased plan and preserve any explicit deferral through
   the backlog. Fetch remote state, inspect the worktree, and confirm the
   release branch descends from `origin/main`.
2. Run pre-version evidence:

   ```console
   make gate-full
   make api-check
   make reference-check
   make package-check
   make semver-check
   ```

   For performance-sensitive changes, run `make bench-history` on the
   registered, otherwise-idle runner. The immutable summary compares the prior
   release at 10% inference/15% fit thresholds and the approximately
   three-release anchor. Before enough versions exist it must explicitly report
   `insufficient_history`; that is evidence, not a fabricated pass.
3. Increment only the patch component unless the user explicitly requests a
   minor or major release. Semver evidence remains required and visible but
   does not override this patch-default policy.

   The one condition under which the patch default is wrong is a **breaking**
   diff, and `make semver-check` is the gate that says so: it fails when
   `cargo-semver-checks` finds changes Cargo would call breaking and the
   release step would not be a breaking one. Below `1.0` the minor is the
   breaking component, so `0.1.2 → 0.1.3` is a *compatible* update and cannot
   carry one. When that gate fails, stop and take an explicit decision on the
   release level; never edit the version to silence it. An additive-only diff
   is compatible with a patch release and the default still applies.

   **Recorded decision (2026-07-25):** the next release is a minor bump to
   `0.2.0`, because the unshipped diff against published `0.1.2` carries eight
   breaking changes. See `dev-docs/plans/parity-sprints.md` §Program status.

   Update only `Cargo.toml`,
   `Cargo.lock`, and the matching dated `CHANGELOG.md` heading and links.
   Re-run every applicable gate above, plus:

   ```console
   cargo publish --locked --dry-run
   ```

4. Commit the explicit release files. Show the exact feature-branch push,
   `HEAD:main` update, and tag to the user and obtain immediate approval. Push
   the feature branch, wait for required checks, and then fast-forward `main`
   without checking it out:

   ```console
   git push origin <release-branch>
   git push origin HEAD:main
   git fetch origin main
   test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
   git tag -a v<version> -m "FerricML <version>"
   git push origin v<version>
   ```

   Never force-push or move an existing tag. The tag push is the publication
   boundary.
5. The release workflow independently requires the tag commit to equal the
   current remote `main` tip, then runs `gate-full`, `package-check`, exact API,
   and frozen reference checks. The crates token is available only to the publish
   job; only the final GitHub-release job receives `contents: write`.
6. Poll the workflow and verify the exact version on crates.io, the GitHub
   release, and its source commit. Then run worktree-aware cleanup and doctrine
   synchronization. Delete branches only when safe; never reset or force-delete
   another worktree's state.

FerricML's release performance record is first-party only.
