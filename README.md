# HAD-MC: Hardware-Aware Dynamic Model Compression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of **"HAD-MC: Domestic Edge Computing Model Compression and Deployment Based on Hardware Perception"** (submitted to Neurocomputing).

## 🎯 Overview

HAD-MC is a comprehensive model compression framework specifically designed for domestic edge computing platforms (MLU370, Ascend 310P). It combines five core algorithms to achieve significant compression while maintaining accuracy:

- **Algorithm 1**: Layer-wise Precision Allocation (gradient-based quantization)
- **Algorithm 2**: Gradient Sensitivity-Guided Pruning
- **Algorithm 3**: Feature-Aligned Knowledge Distillation
- **Algorithm 4**: Operator Fusion for NPU optimization
- **Algorithm 5**: Hash-based Incremental Update (SHA256)

### Key Features

✅ **70%+ latency reduction** on edge devices  
✅ **93%+ mAP preservation** for object detection  
✅ **20+ concurrent streams** support  
✅ **One-click deployment** with complete reproducibility  
✅ **Real experimental data** from production scenarios

## 📊 Performance

| Metric | FP32 Baseline | HAD-MC | Improvement |
|--------|---------------|--------|-------------|
| Latency (ms) | 45.2 | 13.5 | **70.1% ↓** |
| Model Size (MB) | 89.4 | 22.3 | **75.1% ↓** |
| mAP@0.5 (%) | 95.8 | 93.4 | **-2.4%** |
| Concurrent Streams | 8 | 24 | **3× ↑** |

*Results on NEU-DET surface defect detection dataset with MLU370*

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU experiments)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/HAD-MC.git
cd HAD-MC

# Install dependencies
pip install -r requirements.txt

# Prepare datasets
python data/prepare_datasets.py
```

### Run Experiments

#### 1. Full Pipeline (All 5 Algorithms)

```bash
python experiments/full_pipeline.py
```

This runs the complete HAD-MC pipeline on a sample model:
- Layer-wise precision allocation
- Gradient sensitivity-guided pruning
- Feature-aligned knowledge distillation
- Operator fusion
- Hash-based incremental update

**Expected output:**
```
Model Size Reduction: 75.1%
Latency Reduction: 70.1%
Accuracy Change: -2.4%
```

#### 2. NEU-DET Surface Defect Detection

```bash
python experiments/neudet_experiment.py
```

Reproduces the NEU-DET experiments from the paper with:
- ResNet-18 backbone
- 6 defect classes
- Full HAD-MC compression pipeline

**Expected runtime:** ~5 minutes on CPU, ~2 minutes on GPU

#### 3. Financial Fraud Detection

```bash
python experiments/financial_experiment.py
```

Demonstrates HAD-MC on tabular data:
- 32-feature financial dataset
- Binary classification (fraud/normal)
- Real-time inference optimization

#### 4. Cloud-Edge Collaboration

```bash
python experiments/cloud_edge_experiment.py
```

Simulates cloud-edge deployment scenario:
- Model partitioning
- Incremental updates
- Bandwidth optimization

## 📁 Project Structure

```
HAD-MC/
├── hadmc/                      # Core algorithms
│   ├── quantization.py        # Algorithm 1: Precision allocation
│   ├── pruning.py             # Algorithm 2: Gradient pruning
│   ├── distillation.py        # Algorithm 3: Knowledge distillation
│   ├── fusion.py              # Algorithm 4: Operator fusion
│   ├── incremental_update.py  # Algorithm 5: Hash-based update
│   └── utils.py               # Utility functions
├── experiments/               # Experimental scripts
│   ├── full_pipeline.py       # Complete pipeline demo
│   ├── neudet_experiment.py   # NEU-DET experiments
│   ├── financial_experiment.py # Financial fraud detection
│   └── cloud_edge_experiment.py # Cloud-edge collaboration
├── data/                      # Datasets
│   ├── prepare_datasets.py    # Dataset preparation script
│   ├── financial/             # Financial fraud dataset
│   └── neudet/                # NEU-DET dataset
├── models/                    # Pre-trained models
├── results/                   # Experimental results
├── docs/                      # Documentation
│   ├── ALGORITHMS.md          # Detailed algorithm descriptions
│   ├── EXPERIMENTS.md         # Experimental setup guide
│   └── API.md                 # API reference
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔬 Algorithms

### Algorithm 1: Layer-wise Precision Allocation

Allocates mixed precision (FP32/INT8/INT4) based on gradient sensitivity:

```python
from hadmc.quantization import LayerwisePrecisionAllocator

allocator = LayerwisePrecisionAllocator(model, calibration_loader)
quantized_model = allocator.run(target_bits=6)
```

**Key parameters:**
- `tau_h`: High sensitivity threshold (default: 1e-3)
- `tau_l`: Low sensitivity threshold (default: 1e-5)
- `target_bits`: Average bit-width target (default: 6)

### Algorithm 2: Gradient Sensitivity-Guided Pruning

Prunes channels based on gradient importance:

```python
from hadmc.pruning import GradientSensitivityPruner

pruner = GradientSensitivityPruner(model, train_loader, flops_target=0.5)
pruned_model = pruner.run()
```

**Key parameters:**
- `flops_target`: Target FLOPs ratio (0-1)
- `min_channels`: Minimum channels per layer (default: 8)

### Algorithm 3: Feature-Aligned Knowledge Distillation

Transfers knowledge from teacher to student with feature alignment:

```python
from hadmc.distillation import FeatureAlignedDistiller

distiller = FeatureAlignedDistiller(teacher_model, student_model)
distilled_model = distiller.run(train_loader, epochs=5)
```

**Key parameters:**
- `temperature`: Softmax temperature (default: 4.0)
- `alpha`: Task loss weight (default: 0.3)
- `beta`: Soft loss weight (default: 0.3)

### Algorithm 4: Operator Fusion

Fuses Conv+BN+ReLU patterns for NPU optimization:

```python
from hadmc.fusion import OperatorFuser

fuser = OperatorFuser(model)
fused_model = fuser.run()
```

**Supported patterns:**
- Conv2d + BatchNorm2d + ReLU
- Conv2d + ReLU
- Linear + ReLU

### Algorithm 5: Hash-based Incremental Update

Minimizes bandwidth for model updates using SHA256 hashing:

```python
from hadmc.incremental_update import IncrementalUpdater

updater = IncrementalUpdater(block_size=4096)
changed_blocks = updater.compute_delta(old_model, new_model)
bandwidth_saved = updater.get_bandwidth_reduction()
```

**Key parameters:**
- `block_size`: Granularity for change detection (bytes)
- `hash_algorithm`: 'sha256' or 'md5'

## 📈 Experimental Results

### NEU-DET Surface Defect Detection

| Method | mAP@0.5 | Latency (ms) | Size (MB) |
|--------|---------|--------------|-----------|
| FP32 Baseline | 95.8% | 45.2 | 89.4 |
| TensorRT INT8 | 94.2% | 18.7 | 23.1 |
| ONNX Runtime | 93.8% | 22.4 | 25.6 |
| **HAD-MC (Ours)** | **93.4%** | **13.5** | **22.3** |

### Financial Fraud Detection

| Method | F1-Score | Latency (μs) | Size (MB) |
|--------|----------|--------------|-----------|
| FP32 Baseline | 87.3% | 124 | 12.4 |
| Standard INT8 | 85.1% | 67 | 3.2 |
| **HAD-MC (Ours)** | **86.8%** | **42** | **2.9** |

### Cloud-Edge Collaboration

| Metric | Without HAD-MC | With HAD-MC | Improvement |
|--------|----------------|-------------|-------------|
| Update Bandwidth (MB) | 89.4 | 18.7 | **79.1% ↓** |
| Update Time (s) | 12.3 | 2.6 | **78.9% ↓** |
| Concurrent Streams | 8 | 24 | **3× ↑** |

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific algorithm tests:

```bash
pytest tests/test_quantization.py -v
pytest tests/test_pruning.py -v
pytest tests/test_distillation.py -v
```

## 📚 Documentation

- **[Algorithms Guide](docs/ALGORITHMS.md)**: Detailed algorithm descriptions and mathematical formulations
- **[Experiments Guide](docs/EXPERIMENTS.md)**: How to reproduce all paper experiments
- **[API Reference](docs/API.md)**: Complete API documentation
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Deploy HAD-MC on domestic NPUs

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you use HAD-MC in your research, please cite:

```bibtex
@article{hadmc2024,
  title={HAD-MC: Domestic Edge Computing Model Compression and Deployment Based on Hardware Perception},
  author={[Authors]},
  journal={Neurocomputing},
  year={2024},
  note={Under review}
}
```

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/wangjingyi34/HAD-MC/issues)
- **Email**: langkexiaoyi@gmail.com

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- NEU-DET dataset from Northeastern University
- Financial fraud dataset from [source]
- Domestic NPU support from Cambricon and Huawei

---

**Note**: This is research code for reproducibility. For production deployment, please contact the authors for optimized implementations.
