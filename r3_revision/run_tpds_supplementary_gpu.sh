#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${1:-$SCRIPT_DIR/results/tpds_supplementary_gpu}"

export PYTHONUNBUFFERED=1
export HADMC_PLATFORM_TAG="gpu"
export HADMC_TPDS_RESULTS_DIR="$RESULTS_DIR"

if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR/.." rev-parse HEAD >/dev/null 2>&1; then
  export HADMC_GIT_COMMIT="$(git -C "$SCRIPT_DIR/.." rev-parse HEAD)"
fi

mkdir -p "$RESULTS_DIR"

python3 -u "$SCRIPT_DIR/code/tpds_supplementary_experiments.py" \
  --platform-tag gpu \
  --results-dir "$RESULTS_DIR"