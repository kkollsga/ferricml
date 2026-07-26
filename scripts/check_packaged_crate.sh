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
for source_only in .cargo .github .readthedocs.yaml .venv-docs CLAUDE.md Makefile mkdocs.yml RELEASING.md benches benchmarks dev-docs research requirements scripts site tests; do
  if [[ -e "$package_root/ferricml/$source_only" ]]; then
    echo "packaged crate unexpectedly contains source-only path: $source_only" >&2
    exit 1
  fi
done

# The narrative documentation ships, because it is the crate's only offline
# documentation and because `src/lib.rs` compiles those pages as doctests. What
# must never ship with it is documentation-*site* machinery: build config, the
# pinned Python toolchain, theme assets, generated HTML. That rule is
# "everything under docs/ is hand-written markdown", and this is where it stops
# being a convention: the assertion reads the extracted archive, so adding a
# site file under docs/ fails `make package-check` rather than quietly adding
# weight to every download.
if [[ ! -d "$package_root/ferricml/docs" ]]; then
  echo "packaged crate is missing docs/; the narrative markdown is expected to ship" >&2
  exit 1
fi
while IFS= read -r packaged_doc; do
  if [[ "$packaged_doc" != *.md ]]; then
    echo "packaged crate contains a non-markdown docs path: ${packaged_doc#"$package_root/ferricml/"}" >&2
    echo "docs/ ships hand-written markdown only; site machinery belongs outside it" >&2
    exit 1
  fi
done < <(find "$package_root/ferricml/docs" -type f)

# The fixture is copied beside the extracted archive. Its only FerricML path is
# `../ferricml`, so neither Cargo nor the program can fall back to this checkout.
cp -R tests/fixtures/package-consumer "$package_root/consumer"
cargo generate-lockfile --manifest-path "$package_root/consumer/Cargo.toml"
CARGO_TARGET_DIR="$package_root/target" cargo run \
  --manifest-path "$package_root/consumer/Cargo.toml" \
  --locked \
  --quiet

echo "packaged crate: external fit/predict consumer passed for ferricml ${version}"
