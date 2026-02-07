#!/bin/bash
# =============================================================================
# HAD-MC 2.0: One-Click Reproduction Script
# =============================================================================
# This script runs ALL experiments described in the paper.
# Hardware Requirements: NVIDIA GPU (A100 recommended), CUDA 12.x, Python 3.10+
# Expected Runtime: ~30-60 minutes on A100
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Output:
#   - COMPLETE_EXPERIMENT_RESULTS.json  (all raw data)
#   - figures/                          (all paper figures)
#   - Console output with summary
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  HAD-MC 2.0 - Full Experiment Reproduction"
echo "=============================================="
echo ""

# --- Step 0: Environment Check ---
echo "[Step 0/4] Checking environment..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# --- Step 1: Install Dependencies ---
echo "[Step 1/4] Installing dependencies..."
pip install torch torchvision numpy matplotlib seaborn --quiet 2>/dev/null || true
echo "Dependencies ready."
echo ""

# --- Step 2: Run All Experiments ---
echo "[Step 2/4] Running all experiments (this may take 30-60 minutes)..."
echo "  Experiments include:"
echo "    1. NEU-DET Baseline + HAD-MC 2.0 Compression"
echo "    2. SOTA Comparison (AMC, HAQ, DECORE)"
echo "    3. Ablation Study"
echo "    4. PPO vs DQN Controller Comparison"
echo "    5. Cross-Dataset Validation"
echo "    6. Cross-Platform Latency Analysis"
echo "    7. Latency LUT Validation"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 hadmc_experiments_complete.py

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment script failed!"
    exit 1
fi

echo ""
echo "All experiments completed successfully!"
echo ""

# --- Step 3: Generate Figures ---
echo "[Step 3/4] Generating paper figures..."
python3 generate_figures.py

if [ $? -ne 0 ]; then
    echo "WARNING: Figure generation had issues, but experiment data is valid."
fi

echo ""
echo "Figures generated in ./figures/"
echo ""

# --- Step 4: Summary ---
echo "[Step 4/4] Experiment Summary"
echo "=============================================="
echo ""

python3 -c "
import json
with open('COMPLETE_EXPERIMENT_RESULTS.json') as f:
    data = json.load(f)

print('=== KEY RESULTS ===')
print()

# Baseline
bl = data['experiment_1_neudet_baseline_and_compression']
baseline = bl['baseline']
hadmc = bl['hadmc2_compressed']
print(f'Baseline:    Acc={baseline[\"accuracy\"]:.2f}%, Params={baseline[\"params_m\"]:.2f}M, Latency={baseline[\"latency_ms\"]:.2f}ms')
print(f'HAD-MC 2.0:  Acc={hadmc[\"accuracy\"]:.2f}%, Params={hadmc[\"params_m\"]:.2f}M, Latency={hadmc[\"latency_ms\"]:.2f}ms, Speedup={hadmc[\"speedup\"]:.2f}x')
print()

# SOTA
print('=== SOTA Comparison ===')
for method, result in data['experiment_2_sota_comparison'].items():
    print(f'  {method}: Acc={result[\"accuracy\"]:.2f}%, Compression={result[\"compression_ratio\"]:.1f}%, Speedup={result[\"speedup\"]:.2f}x')
print()

# Ablation
print('=== Ablation Study ===')
for config, result in data['experiment_3_ablation_study'].items():
    print(f'  {config}: Acc={result[\"accuracy\"]:.2f}%, Latency={result[\"latency_ms\"]:.2f}ms')
print()

print('All results saved to COMPLETE_EXPERIMENT_RESULTS.json')
print('All figures saved to ./figures/')
print()
print('Reproduction complete!')
"

echo ""
echo "=============================================="
echo "  HAD-MC 2.0 Reproduction Complete!"
echo "=============================================="
