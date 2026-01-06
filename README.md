# HAD-MC: Hardware-Aware Deep Model Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

A hardware-aware deep model compression framework that achieves synergistic optimization through gradient-guided pruning, adaptive quantization, and feature-aligned knowledge distillation.

## 🔑 Key Features

- **Hardware Abstraction Layer (HAL)**: Unified interface for diverse hardware platforms (NPU, GPU, CPU)
- **Synergistic Compression Pipeline**: Joint optimization of pruning, quantization, and distillation
- **Cross-Platform Portability**: Validated on Cambricon MLU370, NVIDIA GPU, Huawei Ascend, and x86 CPU
- **Real-World Deployment**: Tested on financial security and industrial defect detection scenarios

## 📊 Performance Highlights

### Main Results on FS-DS Dataset (Cambricon MLU370)

| Method | mAP@0.5 (%) | ΔmAP (%) | Latency (ms) | Speedup | Model Size (MB) | Compression |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| FP32 Baseline | 92.5 | - | 38.4 | 1.0× | 28.4 | 1.0× |
| PTQ-INT8 | 88.1 | -4.4 | 15.1 | 2.5× | 7.3 | 3.9× |
| QAT-INT8 | 90.3 | -2.2 | 15.5 | 2.4× | 7.3 | 3.9× |
| AWQ | 89.5 | -3.0 | 16.2 | 2.3× | 8.1 | 3.5× |
| SmoothQuant | 89.8 | -2.7 | 15.9 | 2.4× | 7.9 | 3.6× |
| Neuware (Vendor) | 90.1 | -2.4 | 13.8 | 2.8× | 7.5 | 3.8× |
| HALOC | 88.9 | -3.6 | 17.5 | 2.2× | 10.2 | 2.8× |
| **HAD-MC (Ours)** | **91.8** | **-0.7** | **12.1** | **3.2×** | **4.9** | **5.8×** |

### Results on NEU-DET Dataset (Cambricon MLU370)

| Method | Accuracy (%) | Size (MB) | FLOPs (G) | Latency (ms) | Compression |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline (ResNet-18) | 90.2 | 44.6 | 1.82 | 15.2 | 1.0× |
| Pruning (L1) | 88.5 | 22.3 | 0.91 | 10.1 | 2.0× |
| QAT | 89.1 | 11.2 | 1.82 | 7.5 | 4.0× |
| AMC | 88.1 | 15.6 | 0.64 | 8.2 | 2.8× |
| HAQ | 88.9 | 9.8 | 1.82 | 6.8 | 4.6× |
| **HAD-MC (Ours)** | **88.7** | **7.6** | **0.91** | **5.5** | **5.82×** |

### Cross-Platform Validation (GPU)

To demonstrate the generalizability of HAD-MC methodology, we conducted additional experiments on NVIDIA A100 GPU:

| Configuration | mAP@0.5 | mAP@0.5:0.95 | Note |
|:---|:---:|:---:|:---|
| FP32 Baseline | 0.961 | 0.778 | 100 epochs training |
| PTQ-INT8 | 0.961 | 0.778 | Post-training quantization |
| QAT-INT8 | 0.958 | 0.779 | Quantization-aware training |
| L1-Norm Pruning | 0.955 | 0.746 | 30% pruning ratio |

*Data source: Real experiments on NVIDIA A100 80GB PCIe with COCO128 dataset*

### Ablation Study on FS-DS Dataset

| Configuration | mAP@0.5 (%) | Latency (ms) | Model Size (MB) |
|:---|:---:|:---:|:---:|
| Baseline (PTQ-INT8) | 88.1 | 15.1 | 7.3 |
| + Layer-wise Precision Quant. (LPQ) | 90.5 | 14.8 | 6.8 |
| + Gradient Sensitivity Pruning (GSP) | 88.7 | 13.5 | 4.9 |
| + Knowledge Distillation (KD) | 91.8 | 13.6 | 4.9 |
| Full HAD-MC | **91.8** | **12.1** | **4.9** |

### Multi-Channel Video Processing

HAD-MC enables processing of **20 concurrent 1080p video streams** on MLU370, compared to:
- FP32 Baseline: 4 channels max
- Neuware (Vendor): 12 channels max
- **HAD-MC: 20 channels (5× improvement)**

## 🏗️ Framework Architecture

![HAD-MC Framework](docs/figures/hadmc_framework_architecture.png)

The HAD-MC framework consists of three main components:

1. **Synergistic Offline Compression Pipeline**
   - Gradient-Guided Pruning: Removes redundant weights based on gradient sensitivity
   - Adaptive Quantization: Layer-wise precision allocation based on hardware constraints
   - Feature-Aligned Distillation: Knowledge transfer from teacher to compressed student

2. **Hardware Abstraction Layer (HAL)**
   - Unified hardware profile interface
   - Automatic backend selection for different platforms
   - Hardware-aware optimization constraints

3. **Target Hardware Platforms**
   - Cambricon MLU370 (primary validation)
   - NVIDIA GPU (cross-platform validation)
   - Huawei Ascend (extended support)

## 📁 Project Structure

```
HAD-MC/
├── hadmc/                    # Core framework code
│   ├── __init__.py
│   ├── pruning.py           # Gradient-guided pruning (Algorithm 1)
│   ├── quantization.py      # Adaptive quantization (Algorithm 2)
│   ├── distillation.py      # Feature-aligned distillation (Algorithm 3)
│   ├── fusion.py            # Operator fusion (Algorithm 4)
│   ├── hal.py               # Hardware Abstraction Layer
│   ├── inference_engine.py  # Dedicated inference engine
│   ├── memory_manager.py    # Tile-based memory management
│   ├── cloud_edge.py        # Cloud-edge collaboration (Engineering Extension)
│   └── utils.py             # Utility functions
├── experiments/             # Experiment scripts
│   ├── neudet_experiment.py # NEU-DET dataset experiments
│   ├── financial_experiment.py # FS-DS dataset experiments
│   ├── cross_platform_validation.py # GPU validation
│   ├── ablation_study.py    # Ablation experiments
│   └── verify_all_experiments.py # Verification script
├── data/                    # Dataset configurations
│   ├── neudet/              # NEU-DET dataset
│   ├── financial/           # FS-DS dataset
│   └── prepare_datasets.py  # Dataset preparation script
├── docs/                    # Documentation and figures
│   └── figures/             # Academic figures
├── tests/                   # Unit tests
├── run_all_experiments.sh   # One-click experiment script
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wangjingyi34/HAD-MC.git
cd HAD-MC

# Install dependencies
pip install -r requirements.txt

# For MLU370 support, install Neuware SDK
# For GPU support, install PyTorch with CUDA
```

### Prepare Datasets

```bash
# Download and prepare datasets
python data/prepare_datasets.py

# NEU-DET dataset will be automatically downloaded
# FS-DS dataset requires manual request (proprietary)
```

### Run Experiments

```bash
# Run all experiments (requires appropriate hardware)
bash run_all_experiments.sh

# Or run individual experiments
python experiments/neudet_experiment.py
python experiments/financial_experiment.py
python experiments/cross_platform_validation.py
```

### Verify Results

```bash
# Verify all experiment results
python experiments/verify_all_experiments.py
```

## 📖 Core Algorithms

### Algorithm 1: Gradient-Guided Pruning

```python
from hadmc import GradientGuidedPruning

pruner = GradientGuidedPruning(
    model=model,
    target_sparsity=0.5,
    sensitivity_threshold=0.8
)
pruned_model = pruner.prune()
```

### Algorithm 2: Adaptive Quantization

```python
from hadmc import AdaptiveQuantization

quantizer = AdaptiveQuantization(
    model=model,
    target_bitwidth=8,
    hardware_profile=hal.get_profile()
)
quantized_model = quantizer.quantize()
```

### Algorithm 3: Feature-Aligned Distillation

```python
from hadmc import FeatureAlignedDistillation

distiller = FeatureAlignedDistillation(
    teacher=teacher_model,
    student=student_model,
    temperature=4.0,
    alpha=0.7
)
distilled_model = distiller.distill(train_loader, epochs=10)
```

### Algorithm 4: Operator Fusion

```python
from hadmc import OperatorFusion

fuser = OperatorFusion(
    model=model,
    hardware_profile=hal.get_profile()
)
fused_model = fuser.fuse()
```

### Algorithm 5: Hash-based Incremental Update (Cloud-Edge)

```python
from hadmc import CloudEdgeCollaboration

collab = CloudEdgeCollaboration(
    edge_model=edge_model,
    cloud_model=cloud_model
)
updated_model = collab.incremental_update()
```

## 🔧 Hardware Support

| Platform | Status | Notes |
|:---|:---:|:---|
| Cambricon MLU370 | ✅ Primary | Full support with Neuware SDK |
| NVIDIA GPU | ✅ Validated | PyTorch + TensorRT |
| Huawei Ascend 310 | ✅ Extended | CANN toolkit required |
| x86 CPU (Hygon 7280) | ✅ Extended | OpenVINO support |

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{hadmc2024,
  title={HAD-MC: Hardware-Aware Deep Model Compression for Edge AI Deployment},
  author={Wang, Jingyi and others},
  journal={Expert Systems with Applications},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NEU-DET dataset from Northeastern University
- Cambricon Technologies for MLU370 hardware support
- NVIDIA for GPU computing resources

---

**Note**: The FS-DS (Financial Security Dataset) is proprietary and not publicly available. Please contact the authors for access.
