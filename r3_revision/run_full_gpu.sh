#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-$HOME/HAD-MC/experiments_r3/results_gpu}"

mkdir -p "$RESULTS_DIR"
export PYTHONUNBUFFERED=1

python3 -u "$SCRIPT_DIR/code/hadmc_experiments_complete.py" \
  --platform-tag gpu \
  --results-dir "$RESULTS_DIR" \
  --allow-missing-financial