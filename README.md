# HAD-MC 2.0: Hardware-Aware Deep Model Compression via Synergistic RL Co-Design

<p align="center">
  <img src="r3_revision/figures/fig_framework_architecture.png" alt="HAD-MC 2.0 Framework" width="800"/>
</p>

## Overview

**HAD-MC 2.0** is a novel framework for hardware-aware model compression that formulates the compression task as a **synergistic co-design problem** solved by a **Proximal Policy Optimization (PPO)**-based reinforcement learning agent. Unlike existing methods that treat pruning, quantization, and knowledge distillation as isolated, sequential steps, HAD-MC 2.0 jointly optimizes all three techniques simultaneously, guided by a hardware-in-the-loop feedback mechanism using an empirically constructed **Latency Look-Up Table (LUT)**.

### Key Features

- **Synergistic Co-Design**: Jointly optimizes structural pruning, mixed-precision quantization, and knowledge distillation in a single RL search process.
- **PPO-Based Controller**: Uses Proximal Policy Optimization for stable and sample-efficient policy search, outperforming DQN-based alternatives.
- **Hardware-Aware Optimization**: Incorporates real-world latency measurements via a Latency LUT, ensuring that compression decisions translate to actual speedups on the target hardware.
- **Multi-Objective Reward**: Balances accuracy, compression ratio, and inference latency in a unified reward function.

### Key Results (NVIDIA A100 GPU, ResNet18, NEU-DET Dataset)

| Method | Accuracy (%) | Params (M) | Speedup (×) | Compression (%) |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (FP32) | 100.00 | 11.17 | 1.00 | 0.0 |
| AMC [He et al., 2018] | 100.00 | 2.80 | 1.01 | 75.0 |
| HAQ [Wang et al., 2019] | 100.00 | 4.37 | 1.00 | 60.9 |
| DECORE [Alwani et al., 2022] | 99.72 | 2.80 | 1.02 | 75.0 |
| **HAD-MC 2.0 (Ours)** | **100.00** | **2.79** | **1.37** | **75.0** |

> HAD-MC 2.0 achieves **75% compression** and **1.37x speedup** with **zero accuracy loss**, significantly outperforming all SOTA methods.

---

## Repository Structure

```
HAD-MC/
├── README.md                          # This file
├── r3_revision/                       # R3 Revision Materials
│   ├── manuscript_r3.md               # Revised manuscript (R3)
│   ├── response_to_reviewers.md       # Response to reviewer comments
│   ├── COMPLETE_EXPERIMENT_RESULTS.json  # All raw experimental data
│   ├── hadmc_experiments_complete.py   # Complete experiment script
│   ├── generate_figures.py            # Figure generation script
│   ├── run_all.sh                     # One-click reproduction script
│   ├── requirements.txt               # Python dependencies
│   ├── REPRODUCIBILITY_REPORT.md      # Two-run reproducibility verification
│   ├── code_review_issues.md          # Code review and audit report
│   ├── deep_review_report_r3.md       # Deep review against reviewer comments
│   ├── expert_review_synthesis.md     # 12-expert review synthesis
│   └── figures/                       # All paper figures (PNG + PDF)
│       ├── fig_sota_comparison.png
│       ├── fig_ablation_study.png
│       ├── fig_ppo_vs_dqn.png
│       ├── fig_radar_comparison.png
│       ├── fig_cross_dataset.png
│       ├── fig_cross_platform.png
│       ├── fig_latency_lut.png
│       ├── fig_pareto_front.png
│       └── fig_training_convergence.png
├── hadmc2/                            # Core framework code
│   ├── models/                        # Model definitions
│   ├── compression/                   # Compression algorithms
│   ├── rl_controller/                 # PPO controller
│   └── utils/                         # Utility functions
└── data/                              # Dataset preparation scripts
```

---

## Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU (A100 recommended, any CUDA-capable GPU supported)
- **Software**: Python 3.10+, PyTorch 2.0+, CUDA 12.x

### Installation

```bash
# Clone the repository
git clone https://github.com/wangjingyi34/HAD-MC.git
cd HAD-MC

# Install dependencies
pip install -r r3_revision/requirements.txt
```

### One-Click Reproduction

To reproduce **all experiments** described in the paper:

```bash
cd r3_revision
chmod +x run_all.sh
./run_all.sh
```

This script will:
1. Check your environment (GPU, CUDA, PyTorch)
2. Run all 7 experiments (~30-60 minutes on A100)
3. Generate all paper figures
4. Print a summary of key results

### Individual Experiments

You can also run experiments individually:

```bash
cd r3_revision

# Run all experiments and save results
python3 hadmc_experiments_complete.py

# Generate figures from saved results
python3 generate_figures.py
```

---

## Experiments

### Experiment 1: NEU-DET Baseline & HAD-MC 2.0 Compression
Trains a ResNet18 baseline on the NEU-DET steel defect detection dataset (6 classes), then applies the full HAD-MC 2.0 compression pipeline (structural pruning + INT8 quantization + knowledge distillation + Conv-BN fusion).

### Experiment 2: SOTA Comparison
Compares HAD-MC 2.0 against three state-of-the-art methods:
- **AMC** (AutoML for Model Compression) — RL-based pruning
- **HAQ** (Hardware-Aware Quantization) — RL-based mixed-precision quantization
- **DECORE** — Decoupled compression with evolutionary search

### Experiment 3: Ablation Study
Evaluates the contribution of each component:
- Pruning Only
- Quantization Only
- Distillation Only
- Pruning + Quantization
- Pruning + Distillation
- Full HAD-MC 2.0 (all components + Conv-BN fusion)

### Experiment 4: PPO vs. DQN Controller
Compares the PPO-based controller against a DQN-based alternative, demonstrating PPO's superior stability and sample efficiency.

### Experiment 5: Cross-Dataset Validation
Validates generalizability on:
- NEU-DET (steel defect detection)
- Fire-Smoke Detection (FS-DS)
- Financial Fraud Detection

### Experiment 6: Cross-Platform Latency Analysis
Projects performance across multiple hardware platforms using Latency LUTs:
- NVIDIA A100 (Cloud GPU)
- NVIDIA Jetson Orin (Edge GPU)
- Huawei Ascend 310 (Edge NPU)
- Hygon DCU (Domestic Accelerator)

### Experiment 7: Latency LUT Validation
Validates the accuracy of the Latency Look-Up Table by measuring real-world latency for various layer configurations on the A100 GPU.

---

## Results Visualization

<p align="center">
  <img src="r3_revision/figures/fig_sota_comparison.png" alt="SOTA Comparison" width="800"/>
  <br><em>Figure 2: SOTA comparison showing HAD-MC 2.0's superior speedup</em>
</p>

<p align="center">
  <img src="r3_revision/figures/fig_radar_comparison.png" alt="Radar Chart" width="500"/>
  <br><em>Figure 3: Multi-objective performance comparison</em>
</p>

<p align="center">
  <img src="r3_revision/figures/fig_ablation_study.png" alt="Ablation Study" width="800"/>
  <br><em>Figure 4: Ablation study demonstrating the importance of synergistic optimization</em>
</p>

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2025hadmc,
  title={HAD-MC 2.0: Hardware-Aware Deep Model Compression via Synergistic Reinforcement Learning Co-Design},
  author={Wang, Jingyi and others},
  journal={[Journal Name]},
  year={2025}
}
```

---

## License

This project is released under the MIT License.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
