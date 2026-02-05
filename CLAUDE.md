# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HAD-MC (Hardware-Aware Deep Model Compression) is a framework for compressing deep learning models using synergistic optimization through gradient-guided pruning, adaptive quantization, and feature-aligned knowledge distillation. The framework is designed for edge AI deployment with cross-platform support for Cambricon MLU370, NVIDIA GPU, Huawei Ascend, and x86 CPU.

## Core Architecture

The framework consists of 5 core algorithms located in `hadmc/`:

1. **Algorithm 1: Adaptive Quantization** (`hadmc/quantization.py`)
   - `LayerwisePrecisionAllocator`: Assigns precision (FP32/INT8/INT4) based on gradient sensitivity thresholds
   - `AdaptiveQuantizer`: Simplified wrapper for post-training (PTQ) and quantization-aware (QAT) quantization

2. **Algorithm 2: Gradient-Guided Pruning** (`hadmc/pruning.py`)
   - `GradientSensitivityPruner`: Removes channels based on gradient magnitude
   - Supports both classification and detection tasks with auto-detection

3. **Algorithm 3: Feature-Aligned Distillation** (`hadmc/distillation.py`)
   - `FeatureAlignedDistiller`: Knowledge distillation from teacher to student models
   - Uses soft loss (KL divergence) and task loss weighting

4. **Algorithm 4: Operator Fusion** (`hadmc/fusion.py`)
   - `OperatorFuser`: Fuses common patterns like Conv+BN+ReLU
   - Pattern-based matching for optimization

5. **Algorithm 5: Incremental Update** (`hadmc/incremental_update.py`)
   - `HashBasedUpdater`: Creates delta updates using SHA256 hashing of parameter blocks
   - Enables efficient cloud-edge model synchronization

### Device Management

`hadmc/device_manager.py` provides automatic hardware detection and device selection:
- Priority: MLU370 (Cambricon) → Ascend (Huawei) → CPU (fallback)
- Global singleton via `get_device()` function
- Use `device_manager.get_device()` to get the appropriate device string

### Task Type Support

All algorithms support automatic task type detection (`classification` or `detection`):
- Classification returns single tensor output
- Detection (YOLOv5) returns list of tensors
- Auto-detection runs a dummy forward pass on model initialization

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_all_algorithms.py -v

# Run with coverage
pytest tests/ --cov=hadmc --cov-report=html
```

### Running Experiments

**One-click full experiment pipeline:**
```bash
bash run_all_experiments.sh                # Full pipeline (100 epochs)
bash run_all_experiments.sh --quick        # Quick validation (10 epochs)
```

**Individual experiments:**
```bash
# NEU-DET dataset
python experiments/neudet_experiment.py

# GPU cross-platform validation
python experiments/gpu_cross_platform_validation.py

# Full real GPU experiment
python experiments/full_real_gpu_experiment.py

# Run additional baselines comparison
python experiments/run_additional_baselines.py
```

**Verification:**
```bash
python experiments/verify_all_experiments.py
```

### Dataset Preparation
```bash
# NEU-DET dataset
python data/prepare_datasets.py

# COCO128 (for YOLOv5 experiments) is auto-downloaded by run_all_experiments.sh
```

## Important Implementation Notes

### NumPy Version Compatibility
The project requires NumPy < 2.0 due to PyTorch compatibility issues. The `run_all_experiments.sh` script includes automatic version checking and downgrading if needed.

### Quantization Limitations
Current implementation uses PyTorch's dynamic quantization (`torch.quantization.quantize_dynamic`) which:
- Only supports Linear layers reliably for INT8
- Conv2d quantization requires custom implementation
- For real mixed-precision (FP32/INT8/INT4), need custom quantization aware training

### Model Structure Assumptions
The framework assumes models are standard PyTorch `nn.Module` with:
- `Conv2d` and `Linear` layers for pruning/quantization
- `BatchNorm2d` for fusion patterns
- Forward pass returns either single tensor (classification) or list of tensors (detection)

### YOLOv5 Integration
The project integrates with Ultralytics YOLOv5:
- YOLOv5 is cloned automatically to `./yolov5/` by `run_all_experiments.sh`
- HAD-MC YOLOv5 wrapper: `hadmc/hadmc_yolov5.py`
- Model checkpoints are in YOLOv5 format with `model` and `ema` keys

### Experiment Results Storage
All experiment outputs are stored in `experiments/results/`:
- Trained models: `experiments/results/{experiment_name}/weights/best.pt`
- JSON reports: `experiments/results/{experiment_name}.json`
- Logs: `experiments/logs/{experiment_name}.log`

## Datasets

- **FS-DS (Financial Security Dataset)**: Proprietary dataset for financial security detection. Not publicly available - contact authors for access.
- **NEU-DET**: Steel surface defect detection dataset. Download from: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
- **COCO128**: Small subset of COCO dataset used for GPU validation. Auto-downloaded by experiment scripts.

## Hardware Platform Notes

### Cambricon MLU370 (Primary)
- Requires Neuware SDK and `cambricon-pytorch` package
- Device string: `'mlu:0'`
- Use DeviceManager for automatic detection

### NVIDIA GPU (Cross-platform validation)
- Tested on A100 80GB PCIe
- PyTorch with CUDA required
- Device string: `'cuda'` or `'cuda:0'`

### Huawei Ascend (Extended support)
- Requires CANN toolkit and `torch-npu` package
- Device string: `'npu:0'`
- Ascend 310, 310P, 910, 910B supported

## File Structure Context

- `hadmc/`: Core framework algorithms
- `experiments/`: Experiment scripts and results
- `data/`: Dataset configurations and preparation scripts
- `tests/`: Unit tests for all algorithms
- `docs/`: Documentation and academic figures
- `run_all_experiments.sh`: Master experiment pipeline script
- `run_all_experiments_complete.sh`: Extended version with 12-step pipeline

## Ablation Study Configurations

The ablation study tests these configurations (from `run_all_experiments.sh`):
1. Baseline (No Compression)
2. Pruning Only
3. Quantization Only
4. Pruning + Quantization
5. HAD-MC Full (Pruning + Quantization + Distillation)
