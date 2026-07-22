#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

normalize() {
  sed -e 's/CLAUDE\.md/AGENTS.md/g' "$1"
}

check_pair() {
  local claude_root="$1"
  local agents_root="$2"
  local expected="$3"
  local claude_list agents_list rel

  claude_list="$(find "$claude_root" -mindepth 2 -maxdepth 2 -type f -name SKILL.md -print \
    | sed "s#^${claude_root}/##" | LC_ALL=C sort)"
  agents_list="$(find "$agents_root" -mindepth 2 -maxdepth 2 -type f -name SKILL.md -print \
    | sed "s#^${agents_root}/##" | LC_ALL=C sort)"

  [[ "$claude_list" == "$agents_list" ]] || {
    echo "adapter file sets differ" >&2
    return 1
  }
  [[ "$(printf '%s\n' "$claude_list" | sed '/^$/d' | wc -l | tr -d ' ')" == "$expected" ]] || {
    echo "expected ${expected} skills in each adapter" >&2
    return 1
  }

  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    diff -u <(normalize "$claude_root/$rel") "$agents_root/$rel" >/dev/null || {
      echo "semantic adapter drift: $rel" >&2
      return 1
    }
  done <<< "$claude_list"
}

self_test() {
  local fixture
  fixture="$(mktemp -d)"
  trap 'rm -rf "$fixture"' RETURN
  mkdir -p "$fixture/claude/demo" "$fixture/agents/demo"
  printf 'See CLAUDE.md\n' > "$fixture/claude/demo/SKILL.md"
  printf 'See AGENTS.md\n' > "$fixture/agents/demo/SKILL.md"
  check_pair "$fixture/claude" "$fixture/agents" 1
  printf 'unexpected drift\n' >> "$fixture/agents/demo/SKILL.md"
  if check_pair "$fixture/claude" "$fixture/agents" 1 2>/dev/null; then
    echo "self-test failed to detect drift" >&2
    return 1
  fi
  echo "doctrine adapter verifier self-test passed"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

required=(add-todo dev-docs-cleanup notify phased-plan read-inbox release)
for skill in "${required[@]}"; do
  [[ -f "$repo_root/.claude/skills/$skill/SKILL.md" ]] || { echo "missing Claude skill: $skill" >&2; exit 1; }
  [[ -f "$repo_root/.agents/skills/$skill/SKILL.md" ]] || { echo "missing Codex skill: $skill" >&2; exit 1; }
done

diff -u <(normalize "$repo_root/CLAUDE.md") "$repo_root/AGENTS.md" >/dev/null || {
  echo "semantic adapter drift: CLAUDE.md vs AGENTS.md" >&2
  exit 1
}
check_pair "$repo_root/.claude/skills" "$repo_root/.agents/skills" 6
echo "doctrine adapters are synchronized"
