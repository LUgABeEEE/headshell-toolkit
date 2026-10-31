#!/usr/bin/env bash
# Memory-safe, serialized Blender launcher for the kig head-shell project.
# - one Blender at a time (flock)
# - hard address-space cap so a pathological boolean aborts in-process
#   instead of triggering the kernel global OOM killer / dsh-web restart
# - no geometry/render setting is changed: quality is unaffected
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Toolchain lives at <workspace>/.tools (sibling of this repo); override with BLENDER_BIN.
BLENDER="${BLENDER_BIN:-$(dirname "$ROOT")/.tools/blender-4.2.9-linux-x64/blender}"
LOCK="${BLENDER_LOCK:-$ROOT/.blender.lock}"
MEM_KB="${BLENDER_MEM_KB:-8388608}"      # 8 GiB address space per Blender (production-safe)
WAIT="${BLENDER_LOCK_WAIT:-3600}"        # seconds to wait for the lock

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

exec 9>"$LOCK"
flock -w "$WAIT" 9

# ulimit -v is inherited by Blender and its children.
ulimit -v "$MEM_KB"

exec "$BLENDER" "$@"
