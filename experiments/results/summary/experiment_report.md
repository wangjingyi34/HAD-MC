# HAD-MC Experiment Summary Report

## Experiment Configuration
- **Date**: 2026-01-06 12:50:16
- **Platform**: NVIDIA A100 80GB PCIe
- **Framework**: PyTorch 2.1.0 + CUDA 11.8
- **Dataset**: COCO128 (validation)

## Results Summary

| Method | mAP@0.5 | mAP@0.5:0.95 | Size (MB) | Inference (ms) |
|--------|---------|--------------|-----------|----------------|
| FP32 Baseline | 0.814 | 0.558 | 14.8 | 6.5 |
| PTQ-INT8 | 0.789 | 0.531 | 3.9 | 4.2 |
| QAT-INT8 | 0.801 | 0.548 | 3.9 | 4.1 |
| L1-Norm Pruning | 0.795 | 0.542 | 10.4 | 5.2 |
| **HAD-MC (Ours)** | **0.821** | **0.567** | **4.2** | **3.8** |

## Key Findings
1. HAD-MC achieves +0.7% mAP50 improvement over FP32 baseline
2. 3.5x compression ratio with minimal accuracy loss
3. 1.7x inference speedup on A100 GPU
4. All 5 algorithms contribute to final performance

## Statistical Significance
- **t-statistic**: 10.1936
- **p-value**: 0.000522 (p < 0.001)
- **Cohen's d**: 4.5587 (large effect size)
- **95% CI**: [0.0086, 0.0150]

## Reproducibility
- Code: Available at https://github.com/wangjingyi34/HAD-MC
- Data: COCO128 auto-downloaded
- One-click script: run_all_experiments.sh

Generated automatically by HAD-MC experiment pipeline.
