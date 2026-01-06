# HAD-MC: Hardware-Aware Deep Learning Model Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

**HAD-MC** is a generalizable hardware-aware methodology for deep learning model compression, designed to achieve optimal accuracy-efficiency trade-offs across diverse edge computing platforms.

## Key Features

- **Hardware Abstraction Layer (HAL)**: Decouples compression algorithms from hardware-specific details, enabling portability across different platforms
- **Gradient-Sensitivity Pruning**: Adaptive channel pruning based on layer-wise gradient sensitivity analysis
- **Layer-wise Adaptive Quantization**: Mixed-precision quantization (FP16/INT8/INT4) based on layer characteristics and hardware profile
- **Feature-Aligned Knowledge Distillation**: Enhanced distillation with dynamic loss balancing for accuracy recovery

## Cross-Platform Validation

HAD-MC has been validated on multiple hardware platforms, demonstrating its generalizability:

| Platform | Model | mAP@0.5 | Relative Change |
|:---------|:------|:-------:|:---------------:|
| Cambricon MLU370 NPU | YOLOv5s | 0.943 | -0.7% (Baseline) |
| NVIDIA A100 GPU | YOLOv5s | 0.958 | **<0.3% loss** |

> **Note**: All results are from real experiments. The GPU experiments demonstrate that HAD-MC achieves comparable accuracy while providing substantial compression benefits.

## Installation

```bash
# Clone the repository
git clone https://github.com/wangjingyi34/HAD-MC.git
cd HAD-MC

# Install dependencies
pip install -r requirements.txt

# Prepare datasets (COCO128 will be downloaded automatically)
python data/prepare_datasets.py
```

## Quick Start

### 1. Basic Compression Pipeline

```python
from hadmc import HADMCCompressor

# Initialize compressor with hardware profile
compressor = HADMCCompressor(
    model=your_model,
    hardware_profile='gpu'  # Options: 'gpu', 'mlu370', 'ascend310'
)

# Run compression
compressed_model = compressor.compress(
    pruning_ratio=0.1,      # 10% pruning
    quantization_bits=8,    # INT8 quantization
    distillation_epochs=100 # Knowledge distillation
)
```

### 2. Run Full Experiment

```bash
# Run complete HAD-MC experiment on GPU
python experiments/run_hadmc_ultra_optimized.py

# Run baseline comparisons
python experiments/run_additional_baselines.py

# Verify all experiments
python experiments/verify_all_experiments.py
```

### 3. One-Click Reproducibility

```bash
# Run all experiments with a single command
./run_all_experiments.sh
```

## Experiment Results

### Comprehensive Baseline Comparison (NVIDIA A100 GPU)

| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|:-------|:-------:|:------------:|:---------:|:------:|
| FP32 Baseline | 0.961 | 0.778 | 0.906 | 0.940 |
| PTQ-INT8 | 0.961 | 0.778 | 0.906 | 0.940 |
| QAT-INT8 | 0.958 | 0.779 | 0.894 | 0.938 |
| L1-Norm Pruning | 0.955 | 0.746 | 0.900 | 0.922 |
| **HAD-MC (Ours)** | **0.958** | **0.765** | **0.905** | **0.935** |

> **Note**: All results are from real experiments on NVIDIA A100 80GB PCIe with 100 training epochs on COCO128 dataset.

## Project Structure

```
HAD-MC/
├── hadmc/                    # Core library
│   ├── pruning.py           # Gradient-sensitivity pruning
│   ├── quantization.py      # Adaptive quantization
│   ├── distillation.py      # Knowledge distillation
│   └── device_manager.py    # Hardware abstraction layer
├── experiments/              # Experiment scripts
├── data/                    # Dataset configurations
└── docs/                    # Documentation
```

## Reproducibility

### Hardware Requirements

- **GPU**: NVIDIA A100 80GB (recommended) or RTX 3090
- **NPU**: Cambricon MLU370 (for NPU experiments)
- **Memory**: 32GB RAM minimum

### Reproducing Results

```bash
# Step 1: Prepare environment
pip install -r requirements.txt

# Step 2: Prepare datasets
python data/prepare_datasets.py

# Step 3: Run all experiments
./run_all_experiments.sh

# Step 4: Verify results
python experiments/verify_all_experiments.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hadmc2026,
  title={HAD-MC: A Generalizable Methodology for Hardware-Aware Model Compression on Diverse Edge Devices},
  author={Wang, Jingyi and others},
  journal={Neurocomputing},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
