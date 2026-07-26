#!/usr/bin/env bash
#
# Build the crates.io archive and run an external consumer against its extract.
#
# Most of this file is an end-to-end contract: `cargo package` really runs, the
# archive is really extracted, and a program whose only FerricML path is the
# extract really compiles and predicts. That part cannot be self-tested cheaply
# and does not need to be — it fails by not working.
#
# Three of its assertions are not end-to-end, though. "No excluded path ships",
# "every top-level entry is allowlisted", and "everything under docs/ is
# markdown" are ordinary rules over a directory tree, and this crate's house
# standard for a rule is that something proves it can still fire. The counters
# below were offered as making a separate self-test redundant, and they do not:
# they prove the *input* is non-empty, which is a different property from the
# *assertion* being falsifiable. Two holes the counters could not see, both found
# by writing the self-test rather than by reading the script:
#
#   * `package.exclude` accepts gitignore-style globs. A literal `-e` test on
#     `dev-docs/**` is false in the checkout and false in the archive, so such an
#     entry is checked by nothing while `asserted` stays healthy on the entries
#     beside it. It is reported as a finding now instead of passing over.
#   * the docs/ markdown loop iterates whatever `find` returns. A mis-invoked
#     `find` returns nothing, the loop body never runs, and the rule passes. It
#     now has a floor on how many files it saw.
#
# A third assertion turned out to be unfalsifiable as written. Comparing the
# shipped count with the allowlist length can only disagree once both directions
# of the allowlist already hold — a duplicated allowlist line — so that is what
# its message now names and what its self-test case exercises.
#
# So the three rules are named functions over an extract directory, and
# `--self-test` builds synthetic extracts that violate each one. The prior
# argument that the allowlist running in both directions cross-checks the archive
# root *was* sound, and it is why a wrong extract prefix fails loudly rather than
# quietly — but it covers the prefix, never the rule.
#
# Usage:
#   check_packaged_crate.sh              build the archive and check it
#   check_packaged_crate.sh --self-test  prove every rule fires, no packaging

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The archive's top level is a small, stable set, so it is written down here and
# every entry of the extract has to be in it.
#
# The exclusion list says what must not ship, which is a list somebody has to
# remember to extend: a development directory nobody excluded ships silently,
# and no assertion over that list can fail because of it. This is the other
# direction, and it fails by default. The exclusions stay as the convenience that
# keeps the archive small; this list is the contract.
#
# Both directions are checked. An entry in the archive that is not allowlisted
# is weight every consumer downloads for no reason — or a leak. An allowlisted
# entry that no archive contains is a stale line that would let the real thing
# stop shipping unnoticed, which is how `docs/` would go missing from a crate
# whose doctests compile it.
ALLOWED_TOP_LEVEL='.cargo_vcs_info.json
.gitignore
Cargo.lock
Cargo.toml
Cargo.toml.orig
CHANGELOG.md
LICENSE
README.md
docs
src'

# Below this many files under the archive's docs/ the markdown rule is not
# reading anything: `find` was mis-invoked, the directory is a stub, or the
# extract path is wrong. The crate ships well over a dozen pages.
MINIMUM_PACKAGED_DOCS=8

# ---------------------------------------------------------------------------
# Rules. Each takes an extract root and prints one line per finding.
# ---------------------------------------------------------------------------

# The development-only paths are `package.exclude` in Cargo.toml, read from
# there rather than repeated here: a second copy is a second place to remember,
# and nothing would have failed the day `.cargo` was added to one list and not
# the other. Checking the extract, rather than `cargo package --list`, makes the
# assertion cover the exact archive consumed below.
#
# An exclude entry naming a path this checkout does not have proves nothing by
# being absent from the archive, so the two counters separate the entries from
# the assertions that are actually live here. Zero of either means this loop
# checked nothing, which is a failure rather than a pass.
#
# $1 extract root, $2 checkout root, $3 newline-separated exclude entries.
source_only_findings() {
  local extract="$1" checkout="$2" entries="$3"
  local excluded=0 asserted=0 entry

  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    excluded=$((excluded + 1))
    case "$entry" in
      *'*'* | *'?'* | *'['*)
        echo "package.exclude entry '$entry' is a glob, and this assertion compares literal paths, so the entry is checked by nothing; spell it as a path or teach this rule to expand it"
        continue
        ;;
    esac
    if [ -e "$checkout/$entry" ]; then
      asserted=$((asserted + 1))
    fi
    if [ -e "$extract/$entry" ]; then
      echo "packaged crate unexpectedly contains source-only path: $entry"
    fi
  done <<EOF
$entries
EOF

  if [ "$excluded" -eq 0 ]; then
    echo "Cargo.toml declares no package.exclude paths; the source-only assertion checked nothing"
  elif [ "$asserted" -eq 0 ]; then
    echo "no excluded path exists in this checkout; the source-only assertion checked nothing"
  fi
  SOURCE_ONLY_EXCLUDED="$excluded"
  SOURCE_ONLY_ASSERTED="$asserted"
}

# $1 extract root, $2 newline-separated allowlist.
top_level_findings() {
  local extract="$1" allowed="$2"
  local shipped=0 declared=0 entry candidate permitted

  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    shipped=$((shipped + 1))
    permitted=0
    while IFS= read -r candidate; do
      if [ "$entry" = "$candidate" ]; then
        permitted=1
        break
      fi
    done <<EOF
$allowed
EOF
    if [ "$permitted" -eq 0 ]; then
      echo "packaged crate ships an unexpected top-level entry: $entry"
      echo "Every path in the archive is downloaded by everyone who depends on ferricml."
      echo "If '$entry' belongs in the published crate, add it to ALLOWED_TOP_LEVEL in"
      echo "scripts/check_packaged_crate.sh and say why in the same commit. If it does"
      echo "not — and a development directory does not — add it to package.exclude in"
      echo "Cargo.toml, which is where the archive is kept small."
    fi
  done <<EOF
$(ls -A "$extract" 2>/dev/null)
EOF

  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    declared=$((declared + 1))
    if [ ! -e "$extract/$candidate" ]; then
      echo "allowlisted top-level entry is absent from the archive: $candidate"
      echo "The allowlist is the contract, so an entry no archive contains is either a"
      echo "stale line to delete or something that stopped shipping and should not have."
    fi
  done <<EOF
$allowed
EOF

  # With both directions above satisfied, the counts can only disagree when the
  # allowlist repeats a name — the one thing neither direction sees, because a
  # duplicated line is matched by the entry it duplicates and found present in
  # the archive twice over. It makes the contract look larger than it is.
  if [ "$shipped" -ne "$declared" ]; then
    echo "the archive has $shipped top-level entries and the allowlist has $declared; the allowlist repeats a name, or the extract is not the archive root"
  fi
  TOP_LEVEL_SHIPPED="$shipped"
}

# The narrative documentation ships, because it is the crate's only offline
# documentation and because `src/lib.rs` compiles those pages as doctests. What
# must never ship with it is documentation-*site* machinery: build config, the
# pinned Python toolchain, theme assets, generated HTML. That rule is
# "everything under docs/ is hand-written markdown", and this is where it stops
# being a convention: the assertion reads the extracted archive, so adding a
# site file under docs/ fails `make package-check` rather than quietly adding
# weight to every download.
#
# $1 extract root.
docs_findings() {
  local extract="$1"
  local seen=0 packaged_doc

  if [ ! -d "$extract/docs" ]; then
    echo "packaged crate is missing docs/; the narrative markdown is expected to ship"
    DOCS_SEEN=0
    return
  fi
  while IFS= read -r packaged_doc; do
    [ -n "$packaged_doc" ] || continue
    seen=$((seen + 1))
    case "$packaged_doc" in
      *.md) ;;
      *)
        echo "packaged crate contains a non-markdown docs path: ${packaged_doc#"$extract/"}"
        echo "docs/ ships hand-written markdown only; site machinery belongs outside it"
        ;;
    esac
  done <<EOF
$(find "$extract/docs" -type f)
EOF
  if [ "$seen" -lt "$MINIMUM_PACKAGED_DOCS" ]; then
    echo "only $seen files were found under the archive's docs/, below the floor of $MINIMUM_PACKAGED_DOCS; the directory is a stub or the scan is not reading it"
  fi
  DOCS_SEEN="$seen"
}

exclude_entries() {
  python3 -c '
import pathlib, tomllib
manifest = tomllib.loads(pathlib.Path("Cargo.toml").read_text())
for entry in manifest["package"].get("exclude", []):
    print(entry.lstrip("/"))
'
}

all_findings() {
  local extract="$1" checkout="$2" entries="$3" allowed="$4"
  source_only_findings "$extract" "$checkout" "$entries"
  top_level_findings "$extract" "$allowed"
  docs_findings "$extract"
}

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

# A synthetic extract that satisfies every rule, beside a synthetic checkout that
# owns the excluded paths. Small on purpose: the rules read a directory tree, so
# a directory tree is the whole input, and nothing here needs `cargo package`.
write_clean_extract() {
  local base="$1" entry index
  mkdir -p "$base/extract" "$base/checkout/scripts" "$base/checkout/dev-docs"
  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    case "$entry" in
      docs | src) mkdir -p "$base/extract/$entry" ;;
      *) : > "$base/extract/$entry" ;;
    esac
  done <<EOF
$ALLOWED_TOP_LEVEL
EOF
  : > "$base/extract/src/lib.rs"
  index=0
  while [ "$index" -lt "$MINIMUM_PACKAGED_DOCS" ]; do
    : > "$base/extract/docs/page-$index.md"
    index=$((index + 1))
  done
}

CLEAN_EXCLUDES='scripts
dev-docs'

# One synthetic violation: build the clean pair, apply `mutate`, and require the
# expected finding. The clean pair is asserted to report nothing first, in
# `self_test`, so each case here proves its own rule rather than proving that
# something is broken.
self_test_case() {
  local name="$1" expected="$2" mutate="$3"
  local base findings excludes allowed
  base="$(mktemp -d)"
  write_clean_extract "$base"
  excludes="$CLEAN_EXCLUDES"
  allowed="$ALLOWED_TOP_LEVEL"
  eval "$mutate"
  findings="$(all_findings "$base/extract" "$base/checkout" "$excludes" "$allowed")"
  rm -rf "$base"
  case "$findings" in
    *"$expected"*) ;;
    *)
      echo "self-test case '$name' did not fire; reported: ${findings:-<nothing>}" >&2
      exit 1
      ;;
  esac
  SELF_TEST_CASES=$((SELF_TEST_CASES + 1))
}

self_test() {
  local base findings
  SELF_TEST_CASES=0

  base="$(mktemp -d)"
  write_clean_extract "$base"
  findings="$(all_findings "$base/extract" "$base/checkout" "$CLEAN_EXCLUDES" "$ALLOWED_TOP_LEVEL")"
  rm -rf "$base"
  if [ -n "$findings" ]; then
    echo "the synthetic clean extract reported findings: $findings" >&2
    exit 1
  fi

  # Rule 1: no source-only path ships, and the loop has live input.
  self_test_case "source-only-path-ships" \
    "unexpectedly contains source-only path: scripts" \
    'mkdir -p "$base/extract/scripts"'
  self_test_case "exclude-list-is-empty" \
    "declares no package.exclude paths" \
    'excludes=""'
  self_test_case "no-excluded-path-exists-here" \
    "no excluded path exists in this checkout" \
    'rm -rf "$base/checkout/scripts" "$base/checkout/dev-docs"'
  self_test_case "exclude-entry-is-a-glob" \
    "is a glob, and this assertion compares literal paths" \
    'excludes="dev-docs/**"'

  # Rule 2: the allowlist is the contract, in both directions and by count.
  self_test_case "unexpected-top-level-entry" \
    "ships an unexpected top-level entry: benches" \
    'mkdir -p "$base/extract/benches"'
  self_test_case "allowlisted-entry-absent" \
    "absent from the archive: LICENSE" \
    'rm -f "$base/extract/LICENSE"'
  self_test_case "allowlist-repeats-a-name" \
    "the allowlist repeats a name" \
    'allowed="$ALLOWED_TOP_LEVEL
LICENSE"'

  # Rule 3: docs/ ships markdown, and the scan is reading it.
  self_test_case "docs-directory-missing" \
    "is missing docs/" \
    'rm -rf "$base/extract/docs"'
  self_test_case "non-markdown-docs-path" \
    "contains a non-markdown docs path: docs/theme.css" \
    ': > "$base/extract/docs/theme.css"'
  self_test_case "docs-scan-reads-nothing" \
    "below the floor of" \
    'rm -f "$base/extract/docs/page-0.md"'

  echo "packaged crate verifier self-test passed (${SELF_TEST_CASES} cases across 3 archive rules, each proven to fire against a synthetic extract that violates it, with the clean extract proven to report nothing; the external-consumer step is end-to-end and is exercised by the real run)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit 0
fi
if [ -n "${1:-}" ]; then
  echo "usage: check_packaged_crate.sh [--self-test]" >&2
  exit 2
fi

cd "$repo_root"

version="$(cargo metadata --no-deps --format-version 1 | python3 -c \
  'import json, sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "ferricml"))')"
archive="target/package/ferricml-${version}.crate"
package_root="$(mktemp -d)"
trap 'rm -rf "$package_root"' EXIT

cargo package --locked --allow-dirty --no-verify
tar -xzf "$archive" -C "$package_root"
mv "$package_root/ferricml-${version}" "$package_root/ferricml"

excludes="$(exclude_entries)"
findings="$(all_findings "$package_root/ferricml" "$repo_root" "$excludes" "$ALLOWED_TOP_LEVEL")"
if [ -n "$findings" ]; then
  echo "packaged crate check failed:" >&2
  echo "$findings" >&2
  exit 1
fi
# Re-run in this shell so the counters below are the ones the assertions saw; the
# command substitution above ran them in a subshell. They are the run's report
# rather than its assertions, and any finding has already failed the build.
all_findings "$package_root/ferricml" "$repo_root" "$excludes" "$ALLOWED_TOP_LEVEL" >/dev/null

# The fixture is copied beside the extracted archive. Its only FerricML path is
# `../ferricml`, so neither Cargo nor the program can fall back to this checkout.
cp -R tests/fixtures/package-consumer "$package_root/consumer"
cargo generate-lockfile --manifest-path "$package_root/consumer/Cargo.toml"
CARGO_TARGET_DIR="$package_root/target" cargo run \
  --manifest-path "$package_root/consumer/Cargo.toml" \
  --locked \
  --quiet

echo "packaged crate: ${SOURCE_ONLY_ASSERTED} of ${SOURCE_ONLY_EXCLUDED} excluded paths exist here and are absent from the archive"
echo "packaged crate: all ${TOP_LEVEL_SHIPPED} top-level entries are allowlisted, and every allowlisted entry shipped"
echo "packaged crate: ${DOCS_SEEN} packaged docs pages are all markdown"
echo "packaged crate: external fit/predict consumer passed for ferricml ${version}"
