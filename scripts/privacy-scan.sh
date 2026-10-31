#!/usr/bin/env bash
# Fail if a public checkout carries something that must stay private.
#
#   scripts/privacy-scan.sh            # scan the whole tree
#   scripts/privacy-scan.sh <dir>      # scan a subdirectory
#
# Two classes of rule:
#   1. infrastructure traces  -- local paths, host names, credential file names,
#      GPL-adjacent project names. Checked everywhere, no exceptions.
#   2. design values          -- tuned millimetre parameters written as literals.
#      Checked outside tests/, because tests legitimately carry made-up numbers.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT}"
cd "$TARGET" || exit 2

FAIL=0
report() {  # report <label> <hits>
  if [ -n "$2" ]; then
    echo "::error::privacy: $1"
    echo "$2" | sed 's/^/    /'
    FAIL=1
  fi
}

scan() {  # scan <label> <regex> <path filter>
  local label="$1" pattern="$2" filter="${3:-.}"
  local hits
  hits=$(grep -rInE "$pattern" --include='*.py' --include='*.md' --include='*.json' \
                      --include='*.sh' --include='*.yml' --include='*.txt' "$filter" 2>/dev/null \
         | grep -v '^\./\.git/' || true)
  report "$label" "$hits"
}

# ---- 1. infrastructure traces (everywhere, tests included)
INFRA='/data/agent|/mnt/|192\.168\.|10\.88\.|elysia|SER8|ser8|kigcraft|KigCraft|plumAlpha|plum_rd|KIG_ZIP_PASS|ARK_API_KEY|\.credentials|kigerspeak'

HITS=$(grep -rInE "$INFRA" --include='*.py' --include='*.md' --include='*.json' \
                        --include='*.sh' --include='*.yml' --include='*.txt' . 2>/dev/null \
       | grep -v '^\./\.git/' | grep -v 'scripts/privacy-scan.sh' || true)
report "infrastructure trace (path / host / credential / third-party GPL name)" "$HITS"

# ---- 2. design values (outside tests/)
for f in core cli pipeline adapters blender tools scripts params; do
  [ -d "$f" ] || continue
  scan "tuned IPD value"        'ipd_mm[^0-9]{0,12}(5[5-9]|6[0-9]|7[0-5])(\.[0-9]+)?' "$f"
  scan "tuned wall thickness"   'wall_mm[^0-9]{0,12}(2\.[0-9]|3\.[0-9]|4\.[0-9])'       "$f"
  scan "tuned vent diameter"    'vent_diameter_mm[^0-9]{0,12}(5\.[0-9]|6\.[0-9]|7\.[0-9])' "$f"
  scan "tuned vent spacing"     'vent_spacing_mm[^0-9]{0,12}(1[5-9]|2[0-2])'             "$f"
  scan "tuned split offset"     'split_(offset|y)_mm[^0-9]{0,12}-(1[0-9]|2[0-9])'       "$f"
  scan "tuned eye placement"    'eye_(y|z)_mm[^0-9]{0,12}([5-9][0-9]|1[0-9][0-9])'      "$f"
done

if [ "$FAIL" -eq 0 ]; then
  echo "privacy scan: clean"
fi
exit "$FAIL"
