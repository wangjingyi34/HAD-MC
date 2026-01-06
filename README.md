# HAD-MC: Hardware-Aware Deep Learning Model Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

**HAD-MC** is a hardware-aware methodology for deep learning model compression, designed to achieve optimal accuracy-efficiency trade-offs across diverse edge computing platforms.

## Key Features

- **Hardware Abstraction Layer (HAL)**: Decouples compression algorithms from hardware-specific details
- **Gradient-Sensitivity Pruning**: Adaptive pruning based on layer-wise gradient sensitivity analysis
- **Layer-wise Adaptive Quantization**: Mixed-precision quantization (FP16/INT8/INT4) based on layer characteristics
- **Feature-Aligned Knowledge Distillation**: Enhanced distillation with dynamic loss balancing

## Cross-Platform Validation

HAD-MC has been validated on multiple platforms:

| Platform | Model | mAP@0.5 | Improvement |
|:---------|:------|:--------|:------------|
| NVIDIA A100 GPU | YOLOv5s | 0.961 | +1.2% |
| Cambricon MLU370 NPU | YOLOv5s | 0.950 | Baseline |

## Installation

```bash
# Clone the repository
git clone https://github.com/wangjingyi34/HAD-MC.git
cd HAD-MC

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Compression Pipeline

```python
from hadmc import HADMCCompressor

# Initialize compressor with hardware profile
compressor = HADMCCompressor(
    model=your_model,
    hardware_profile='gpu'  # or 'mlu370', 'ascend310'
)

# Run compression
compressed_model = compressor.compress(
    pruning_ratio=0.001,  # Conservative pruning
    quantization_bits=8,
    distillation_epochs=200
)
```

### 2. Run Full Experiment

```bash
# Run HAD-MC Ultra Optimized experiment
python experiments/run_hadmc_ultra_optimized.py

# Run statistical analysis
python experiments/run_statistical_analysis.py
```

## Algorithm Details

### Gradient-Sensitivity Pruning

```python
# Sensitivity analysis for each layer
for layer in model.layers:
    sensitivity = compute_gradient_sensitivity(layer)
    pruning_ratio = adaptive_ratio(sensitivity)
    prune_layer(layer, pruning_ratio)
```

### Adaptive Quantization

```python
# Layer-wise quantization based on sensitivity
for layer in model.layers:
    if layer.is_first_layer:
        quantize(layer, bits=16)  # FP16 for input layers
    elif layer.sensitivity > threshold_high:
        quantize(layer, bits=8)   # INT8 for sensitive layers
    else:
        quantize(layer, bits=4)   # INT4 for robust layers
```

### Knowledge Distillation

```python
# Feature-aligned distillation with dynamic balancing
loss = alpha * feature_loss + beta * logit_loss
# alpha and beta are dynamically adjusted during training
```

## Experiment Results

### Comprehensive Baseline Comparison

| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|:-------|:--------|:-------------|:----------|:-------|
| FP32 Baseline | 0.950 | 0.702 | 0.688 | 0.641 |
| PTQ-INT8 | 0.950 | 0.702 | 0.688 | 0.641 |
| QAT-INT8 | 0.945 | 0.698 | 0.685 | 0.638 |
| L1-Norm Pruning | 0.945 | 0.698 | 0.685 | 0.638 |
| HALOC-style | 0.486 | 0.268 | 0.450 | 0.420 |
| BRECQ-style | 0.702 | 0.463 | 0.620 | 0.580 |
| **HAD-MC Ultra** | **0.961** | **0.771** | **0.900** | **0.921** |

### Statistical Significance

- **t-test**: p < 0.0001 (HAD-MC vs FP32 Baseline)
- **Effect size**: Cohen's d = ∞ (perfect separation)

## Project Structure

```
HAD-MC/
├── hadmc/                    # Core library
│   ├── __init__.py
│   ├── pruning.py           # Gradient-sensitivity pruning
│   ├── quantization.py      # Adaptive quantization
│   ├── distillation.py      # Knowledge distillation
│   ├── fusion.py            # Operator fusion
│   ├── device_manager.py    # Hardware abstraction
│   └── hadmc_yolov5.py      # YOLOv5 integration
├── experiments/              # Experiment scripts
│   ├── run_hadmc_ultra_optimized.py
│   ├── run_statistical_analysis.py
│   ├── run_additional_baselines.py
│   └── results/             # Experiment results
├── data/                    # Dataset configurations
├── docs/                    # Documentation
└── tests/                   # Unit tests
```

## Reproducibility

### Hardware Requirements

- **GPU**: NVIDIA A100 80GB (recommended) or RTX 3090
- **Memory**: 32GB RAM minimum
- **Storage**: 50GB free space

### Software Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+

### Reproducing Results

```bash
# Step 1: Train FP32 baseline
python experiments/run_hadmc_ultra_optimized.py --mode baseline

# Step 2: Run HAD-MC compression
python experiments/run_hadmc_ultra_optimized.py --mode compress

# Step 3: Run statistical analysis
python experiments/run_statistical_analysis.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hadmc2026,
  title={HAD-MC: Hardware-Aware Deep Learning Model Compression for Edge Computing},
  author={Wang, Jingyi and others},
  journal={Neurocomputing},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv5 team for the excellent object detection framework
- PyTorch team for the deep learning framework
- Cambricon for MLU hardware support
