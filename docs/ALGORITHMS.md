# HAD-MC Algorithms Documentation

This document provides detailed descriptions of the five core algorithms in HAD-MC.

## Table of Contents

1. [Algorithm 1: Layer-wise Precision Allocation](#algorithm-1-layer-wise-precision-allocation)
2. [Algorithm 2: Gradient Sensitivity-Guided Pruning](#algorithm-2-gradient-sensitivity-guided-pruning)
3. [Algorithm 3: Feature-Aligned Knowledge Distillation](#algorithm-3-feature-aligned-knowledge-distillation)
4. [Algorithm 4: Operator Fusion](#algorithm-4-operator-fusion)
5. [Algorithm 5: Hash-based Incremental Update](#algorithm-5-hash-based-incremental-update)

---

## Algorithm 1: Layer-wise Precision Allocation

### Overview

Layer-wise precision allocation assigns different quantization bit-widths (FP32, INT8, INT4) to different layers based on their gradient sensitivity. Sensitive layers retain higher precision while less critical layers use lower precision.

### Mathematical Formulation

**Gradient Sensitivity:**

For each layer $l$ with parameters $\theta_l$:

$$S_l = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial L_i}{\partial \theta_l} \right|$$

where:
- $N$ is the number of calibration samples
- $L_i$ is the loss for sample $i$
- $S_l$ is the average gradient magnitude

**Precision Assignment:**

$$
b_l = \begin{cases}
32 & \text{if } S_l > \tau_h \\
8 & \text{if } \tau_l \leq S_l \leq \tau_h \\
4 & \text{if } S_l < \tau_l
\end{cases}
$$

where:
- $b_l$ is the bit-width for layer $l$
- $\tau_h$ is the high sensitivity threshold (default: 1e-3)
- $\tau_l$ is the low sensitivity threshold (default: 1e-5)

**Constraint:**

The average bit-width must satisfy:

$$\frac{1}{L} \sum_{l=1}^{L} b_l \leq B_{target}$$

where $B_{target}$ is the target average bit-width (default: 6).

### Implementation

```python
from hadmc.quantization import LayerwisePrecisionAllocator

# Initialize
allocator = LayerwisePrecisionAllocator(
    model=model,
    calibration_loader=calib_loader,
    tau_h=1e-3,
    tau_l=1e-5,
    device='cuda'
)

# Run allocation
quantized_model = allocator.run(target_bits=6)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | Required | PyTorch model to quantize |
| `calibration_loader` | DataLoader | Required | Calibration dataset |
| `tau_h` | float | 1e-3 | High sensitivity threshold |
| `tau_l` | float | 1e-5 | Low sensitivity threshold |
| `target_bits` | int | 6 | Target average bit-width |
| `device` | str | 'cpu' | Device for computation |

### Expected Results

- **Model size reduction**: 60-75%
- **Latency reduction**: 40-60%
- **Accuracy drop**: <1%

---

## Algorithm 2: Gradient Sensitivity-Guided Pruning

### Overview

Gradient sensitivity-guided pruning removes less important channels from convolutional layers based on their gradient magnitudes during training.

### Mathematical Formulation

**Channel Importance:**

For each convolutional layer $l$ with $C_l$ output channels:

$$I_l^c = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial L_i}{\partial W_l^c} \right|$$

where:
- $I_l^c$ is the importance of channel $c$ in layer $l$
- $W_l^c$ is the weight tensor for channel $c$

**Pruning Decision:**

Channels are ranked by importance and pruned to meet FLOPs target:

$$\text{Prune}(l, c) = \begin{cases}
1 & \text{if } I_l^c < \text{threshold}_l \\
0 & \text{otherwise}
\end{cases}$$

**FLOPs Constraint:**

$$\frac{\text{FLOPs}_{\text{pruned}}}{\text{FLOPs}_{\text{original}}} \leq R_{target}$$

where $R_{target}$ is the target FLOPs ratio (e.g., 0.5 for 50% reduction).

### Implementation

```python
from hadmc.pruning import GradientSensitivityPruner

# Initialize
pruner = GradientSensitivityPruner(
    model=model,
    train_loader=train_loader,
    flops_target=0.5,
    device='cuda'
)

# Run pruning
pruned_model = pruner.run()
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | Required | PyTorch model to prune |
| `train_loader` | DataLoader | Required | Training dataset |
| `flops_target` | float | 0.5 | Target FLOPs ratio (0-1) |
| `min_channels` | int | 8 | Minimum channels per layer |
| `device` | str | 'cpu' | Device for computation |

### Expected Results

- **FLOPs reduction**: 40-60%
- **Model size reduction**: 30-50%
- **Accuracy drop**: 1-3% (before fine-tuning)

---

## Algorithm 3: Feature-Aligned Knowledge Distillation

### Overview

Feature-aligned knowledge distillation transfers knowledge from a large teacher model to a compact student model by matching both output logits and intermediate features.

### Mathematical Formulation

**Total Loss:**

$$L_{total} = \alpha L_{task} + \beta L_{soft} + \gamma L_{feature}$$

**Task Loss (Hard Labels):**

$$L_{task} = \text{CrossEntropy}(y_{student}, y_{true})$$

**Soft Loss (Knowledge Distillation):**

$$L_{soft} = \tau^2 \cdot \text{KL}\left(\frac{y_{student}}{\tau}, \frac{y_{teacher}}{\tau}\right)$$

where $\tau$ is the temperature (default: 4.0).

**Feature Loss (Feature Matching):**

$$L_{feature} = \sum_{k=1}^{K} \left\| A_k(F_k^{student}) - F_k^{teacher} \right\|_2^2$$

where:
- $F_k$ is the feature map from layer $k$
- $A_k$ is an adaptation layer (1×1 conv) to match dimensions
- $K$ is the number of feature matching layers

### Implementation

```python
from hadmc.distillation import FeatureAlignedDistiller

# Initialize
distiller = FeatureAlignedDistiller(
    teacher_model=teacher,
    student_model=student,
    device='cuda'
)

# Run distillation
distilled_model = distiller.run(
    train_loader=train_loader,
    epochs=5,
    lr=0.001
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_model` | nn.Module | Required | Pre-trained teacher model |
| `student_model` | nn.Module | Required | Student model to train |
| `temperature` | float | 4.0 | Distillation temperature |
| `alpha` | float | 0.3 | Task loss weight |
| `beta` | float | 0.3 | Soft loss weight |
| `gamma` | float | 0.4 | Feature loss weight |
| `device` | str | 'cpu' | Device for computation |

### Expected Results

- **Accuracy recovery**: +2-5% compared to direct training
- **Convergence speed**: 2-3× faster
- **Final accuracy**: Within 2% of teacher

---

## Algorithm 4: Operator Fusion

### Overview

Operator fusion combines consecutive operations (Conv+BN+ReLU) into single fused operators to reduce memory access and improve NPU utilization.

### Mathematical Formulation

**Conv-BN Fusion:**

Original operations:
$$y = \text{ReLU}(\text{BN}(\text{Conv}(x)))$$

Fused operation:
$$y = \text{ReLU}(W_{fused} \cdot x + b_{fused})$$

where:
$$W_{fused} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W_{conv}$$
$$b_{fused} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b_{conv} - \mu) + \beta$$

Parameters:
- $\gamma, \beta$: BN scale and shift
- $\mu, \sigma^2$: BN running mean and variance
- $\epsilon$: BN epsilon
- $W_{conv}, b_{conv}$: Conv weights and bias

### Implementation

```python
from hadmc.fusion import OperatorFuser

# Initialize
fuser = OperatorFuser(model=model)

# Run fusion
fused_model = fuser.run()
```

### Supported Patterns

1. **Conv2d + BatchNorm2d + ReLU**
2. **Conv2d + BatchNorm2d**
3. **Conv2d + ReLU**
4. **Linear + ReLU**

### Expected Results

- **Latency reduction**: 15-25%
- **Memory access reduction**: 30-40%
- **No accuracy loss**: Mathematically equivalent

---

## Algorithm 5: Hash-based Incremental Update

### Overview

Hash-based incremental update minimizes bandwidth for model updates by only transmitting changed blocks, identified using SHA256 hashing.

### Mathematical Formulation

**Block Division:**

Model parameters $\theta$ are divided into $B$ blocks:
$$\theta = \{\theta_1, \theta_2, ..., \theta_B\}$$

where each block size is $s$ bytes (default: 4096).

**Hash Computation:**

For each block $\theta_b$:
$$h_b = \text{SHA256}(\theta_b)$$

**Change Detection:**

$$\Delta_b = \begin{cases}
1 & \text{if } h_b^{new} \neq h_b^{old} \\
0 & \text{otherwise}
\end{cases}$$

**Bandwidth Reduction:**

$$R_{bandwidth} = 1 - \frac{\sum_{b=1}^{B} \Delta_b \cdot s}{|\theta|}$$

where $|\theta|$ is the total model size.

### Implementation

```python
from hadmc.incremental_update import IncrementalUpdater

# Initialize
updater = IncrementalUpdater(block_size=4096)

# Compute delta
changed_blocks = updater.compute_delta(old_model, new_model)

# Get bandwidth reduction
bandwidth_saved = updater.get_bandwidth_reduction()
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | int | 4096 | Block size in bytes |
| `hash_algorithm` | str | 'sha256' | Hash algorithm ('sha256' or 'md5') |

### Expected Results

- **Bandwidth reduction**: 60-80% for incremental updates
- **Update time reduction**: 70-85%
- **Hash overhead**: <0.1% of model size

---

## Algorithm Integration

### Sequential Pipeline

The five algorithms are typically applied in sequence:

```python
# 1. Quantization
from hadmc.quantization import LayerwisePrecisionAllocator
allocator = LayerwisePrecisionAllocator(model, calib_loader)
model = allocator.run(target_bits=6)

# 2. Pruning
from hadmc.pruning import GradientSensitivityPruner
pruner = GradientSensitivityPruner(model, train_loader, flops_target=0.5)
model = pruner.run()

# 3. Knowledge Distillation
from hadmc.distillation import FeatureAlignedDistiller
distiller = FeatureAlignedDistiller(teacher, model)
model = distiller.run(train_loader, epochs=5)

# 4. Operator Fusion
from hadmc.fusion import OperatorFuser
fuser = OperatorFuser(model)
model = fuser.run()

# 5. Incremental Update (for deployment)
from hadmc.incremental_update import IncrementalUpdater
updater = IncrementalUpdater()
delta = updater.compute_delta(old_model, model)
```

### Combined Results

When all algorithms are applied together:

- **Model size**: 70-80% reduction
- **Latency**: 65-75% reduction
- **Accuracy**: <3% drop
- **Update bandwidth**: 70-85% reduction

---

## References

1. **Quantization**: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018
2. **Pruning**: He et al., "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration", CVPR 2019
3. **Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2014
4. **Fusion**: Rotem et al., "Glow: Graph Lowering Compiler Techniques for Neural Networks", arXiv 2018
5. **Incremental Update**: Merkle, "A Digital Signature Based on a Conventional Encryption Function", CRYPTO 1987

---

## FAQ

**Q: Can I apply only some of the algorithms?**  
A: Yes, each algorithm is independent and can be used separately.

**Q: What order should I apply the algorithms?**  
A: The recommended order is: Quantization → Pruning → Distillation → Fusion → Incremental Update

**Q: Do I need to fine-tune after each algorithm?**  
A: Fine-tuning is recommended after pruning and distillation for best results.

**Q: Can I use custom thresholds?**  
A: Yes, all thresholds and hyperparameters can be customized via constructor arguments.

**Q: How do I choose the right hyperparameters?**  
A: Start with defaults, then tune based on your accuracy/efficiency trade-off requirements.
