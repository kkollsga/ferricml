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

# The development-only paths are `package.exclude` in Cargo.toml, read from
# there rather than repeated here: a second copy is a second place to remember,
# and nothing would have failed the day `.cargo` was added to one list and not
# the other. Checking the extract, rather than `cargo package --list`, makes the
# assertion cover the exact archive consumed below.
#
# An exclude entry naming a path this checkout does not have proves nothing by
# being absent from the archive, so the two counters below separate the entries
# from the assertions that are actually live here. Zero of either means this
# loop checked nothing, which is a failure rather than a pass.
excluded=0
asserted=0
while IFS= read -r source_only; do
  excluded=$((excluded + 1))
  if [[ -e "$source_only" ]]; then
    asserted=$((asserted + 1))
  fi
  if [[ -e "$package_root/ferricml/$source_only" ]]; then
    echo "packaged crate unexpectedly contains source-only path: $source_only" >&2
    exit 1
  fi
done < <(python3 -c '
import pathlib, tomllib
manifest = tomllib.loads(pathlib.Path("Cargo.toml").read_text())
for entry in manifest["package"].get("exclude", []):
    print(entry.lstrip("/"))
')

if [[ "$excluded" -eq 0 ]]; then
  echo "Cargo.toml declares no package.exclude paths; the source-only assertion checked nothing" >&2
  exit 1
fi
if [[ "$asserted" -eq 0 ]]; then
  echo "no excluded path exists in this checkout; the source-only assertion checked nothing" >&2
  exit 1
fi

# The exclusion list above says what must not ship, which is a list somebody
# has to remember to extend: a development directory nobody excluded ships
# silently, and no assertion above can fail because of it. This is the other
# direction, and it fails by default. The archive's top level is a small,
# stable set, so it is written down here and every entry of the extract has to
# be in it. The exclusions stay as the convenience that keeps the archive
# small; this list is the contract.
#
# Both directions are checked. An entry in the archive that is not allowlisted
# is weight every consumer downloads for no reason — or a leak. An allowlisted
# entry that no archive contains is a stale line that would let the real thing
# stop shipping unnoticed, which is how `docs/` would go missing from a crate
# whose doctests compile it.
allowed_top_level=(
  .cargo_vcs_info.json
  .gitignore
  Cargo.lock
  Cargo.toml
  Cargo.toml.orig
  CHANGELOG.md
  LICENSE
  README.md
  docs
  src
)

shipped=0
while IFS= read -r entry; do
  shipped=$((shipped + 1))
  permitted=0
  for candidate in "${allowed_top_level[@]}"; do
    if [[ "$entry" == "$candidate" ]]; then
      permitted=1
      break
    fi
  done
  if [[ "$permitted" -eq 0 ]]; then
    echo "packaged crate ships an unexpected top-level entry: $entry" >&2
    echo "Every path in the archive is downloaded by everyone who depends on ferricml." >&2
    echo "If '$entry' belongs in the published crate, add it to allowed_top_level in" >&2
    echo "scripts/check_packaged_crate.sh and say why in the same commit. If it does" >&2
    echo "not — and a development directory does not — add it to package.exclude in" >&2
    echo "Cargo.toml, which is where the archive is kept small." >&2
    exit 1
  fi
done < <(ls -A "$package_root/ferricml")

for candidate in "${allowed_top_level[@]}"; do
  if [[ ! -e "$package_root/ferricml/$candidate" ]]; then
    echo "allowlisted top-level entry is absent from the archive: $candidate" >&2
    echo "The allowlist is the contract, so an entry no archive contains is either a" >&2
    echo "stale line to delete or something that stopped shipping and should not have." >&2
    exit 1
  fi
done

if [[ "$shipped" -ne "${#allowed_top_level[@]}" ]]; then
  echo "the archive has $shipped top-level entries and the allowlist has ${#allowed_top_level[@]}" >&2
  exit 1
fi

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

echo "packaged crate: ${asserted} of ${excluded} excluded paths exist here and are absent from the archive"
echo "packaged crate: all ${shipped} top-level entries are allowlisted, and every allowlisted entry shipped"
echo "packaged crate: external fit/predict consumer passed for ferricml ${version}"
