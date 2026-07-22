#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$(cargo metadata --no-deps --format-version 1 | python3 -c \
  'import json, sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "ferricml"))')"
archive="target/package/ferricml-${version}.crate"
package_root="$(mktemp -d)"
trap 'rm -rf "$package_root"' EXIT

cargo package --locked --allow-dirty --no-verify
tar -xzf "$archive" -C "$package_root"
mv "$package_root/ferricml-${version}" "$package_root/ferricml"

# These paths are development-only. Checking the extract, rather than
# `cargo package --list`, makes the assertion cover the exact archive consumed
# below.
for source_only in .github CLAUDE.md Makefile RELEASING.md benches benchmarks dev-docs research requirements scripts tests; do
  if [[ -e "$package_root/ferricml/$source_only" ]]; then
    echo "packaged crate unexpectedly contains source-only path: $source_only" >&2
    exit 1
  fi
done

# The fixture is copied beside the extracted archive. Its only FerricML path is
# `../ferricml`, so neither Cargo nor the program can fall back to this checkout.
cp -R tests/fixtures/package-consumer "$package_root/consumer"
cargo generate-lockfile --manifest-path "$package_root/consumer/Cargo.toml"
CARGO_TARGET_DIR="$package_root/target" cargo run \
  --manifest-path "$package_root/consumer/Cargo.toml" \
  --locked \
  --quiet

echo "packaged crate: external fit/predict consumer passed for ferricml ${version}"
