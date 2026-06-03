#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$HOME/HAD-MC/experiments_r3/probes_gpu}"

python3 "$SCRIPT_DIR/code/platform_probe.py" \
  --platform-tag gpu \
  --output-dir "$OUTPUT_DIR"