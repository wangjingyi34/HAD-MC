# HAD-MC 2.0 Agents Documentation

## Overview

HAD-MC 2.0 employs five specialized agents, each responsible for one aspect of model compression. These agents operate in parallel under the coordination of a PPO controller to achieve optimal compression configurations.

## Agent Architecture

```
+---------------------+     +----------------------+     +----------------------+
|  PPO Controller      |<---->|  State              |<---->|  Reward Function    |
|  (Coordination)     |     +----------------------+     +----------------------+
+----------+------------+
           |
           v
+----------+------------+     +------------+------------+     +------------+
| Pruning  | Quantization | Distillation | Fusion     | Update     |
| Agent    | Agent        | Agent        | Agent      | Agent      |
+----------+------------+     +------------+------------+     +------------+
           |                 |             |            |
           +-----------------+-------------+------------+
                            |
                            v
                    +----------------+
                    |    Model        |
                    +----------------+
```

## 1. Pruning Agent

### Algorithm Details

The Pruning Agent performs gradient-based structured channel pruning using Taylor expansion approximation.

**Importance Metric:**

For each output channel j in a convolutional layer:
```
I(j) = |Σ_i Σ_k Σ_l W(j,i,k,l) · ∂L/∂O(j,i,k,l)|
```

Where:
- W(j,i,k,l) is the weight at position (j,i,k,l)
- ∂L/∂O(j,i,k,l) is the gradient of loss with respect to output
- The sum is over all input positions (i,k,l)

**Pruning Decision:**

Given a pruning ratio r, the agent removes the (r × 100)% of channels with lowest importance I(j).

**Channel Removal:**

1. Identify channels to remove: `J_remove = argsort(I, k=N_remove, largest=False)`
2. Create mask: `M(j) = 1 if j ∉ J_remove else 0`
3. Apply mask: `W'(j,i,k,l) = W(j,i,k,l) · M(j)`

**Impact on Model:**
- Reduces parameters by factor: (1 - r)
- May require re-initialization of next layer if input channels change
- Structured pruning maintains efficient memory access patterns

### Implementation Details

**File:** `hadmc2/agents/pruning_agent.py`

**Class:** `PruningAgent`

**Key Methods:**

```python
class PruningAgent:
    def __init__(self, model, train_loader, device='cpu')
        # Initialize with model and training data

    def compute_importance(self) -> Dict[str, torch.Tensor]:
        # Compute I(j) for each channel using Taylor expansion

    def prune(self, pruning_config: Dict, model=None) -> nn.Module:
        # Remove channels based on importance scores

    def get_action_space(self) -> Dict:
        # Returns {type: 'discrete', layer_idx: [...], pruning_ratio: [...]}

    def get_action(self, state: torch.Tensor) -> tuple:
        # Sample action (for MARL)
```

### Mathematical Foundation

**Taylor Expansion Justification:**

The first-order Taylor expansion of the loss function around current weights W:

```
L(W - ΔW) ≈ L(W) - Σ_i Σ_j Σ_k Σ_l (∂L/∂W(i,j,k,l)) · ΔW(i,j,k,l)
```

The importance metric approximates the impact of removing a channel on the loss.

**Relationship to Optimal Brain Surgeon (OBS):**

The importance metric relates to OBS pruning:
- OBS removes neurons with lowest activation average
- Gradient pruning removes neurons with lowest gradient sensitivity
- Gradient pruning is theoretically sounder for near-optimal weights

### Hyperparameters

| Parameter | Range | Default | Description |
|-----------|-------|----------|-------------|
| Pruning Ratios | [0.0, 0.9] | [0.0, 0.1, ..., 0.9] | Discrete options |
| Gradient Hook | All layers | Yes | Full gradient capture |
| Pruning Strategy | Structured | L1-norm | Per-channel removal |

### Expected Behavior

| Pruning Ratio | Expected Accuracy Drop | Expected Size Reduction |
|---------------|---------------------|----------------------|
| 0.1 | ~1-2% | ~10% |
| 0.3 | ~3-5% | ~30% |
| 0.5 | ~5-8% | ~50% |
| 0.7 | ~10-15% | ~70% |

## 2. Quantization Agent

### Algorithm Details

The Quantization Agent performs layer-wise mixed-precision quantization using calibration data.

**Calibration Process:**

For each layer, compute activation statistics:
```
μ = mean(x)        # Running mean over calibration batch
σ = std(x)         # Running standard deviation
x_min = min(x)      # Minimum activation
x_max = max(x)      # Maximum activation
```

**Affine Quantization:**

For a given bit-width b:
```
scale = (x_max - x_min) / (2^b - 1)
zero_point = -x_min / scale

x_q = clamp(round(x / scale + zero_point), 0, 2^b - 1)
x̃ = (x_q - zero_point) · scale
```

**Quantization Error:**

Mean Squared Error (MSE) between original and quantized:
```
MSE = E[(x - x̃)²]
```

The agent favors higher bit-widths for layers with higher quantization error.

### Implementation Details

**File:** `hadmc2/agents/quantization_agent.py`

**Class:** `QuantizationAgent`

**Key Methods:**

```python
class QuantizationAgent:
    def __init__(self, model, calibration_loader, device='cpu')
        # Initialize with model and calibration data

    def calibrate(self, num_batches=100):
        # Collect statistics for each layer

    def quantize(self, quantization_config, model=None) -> nn.Module:
        # Apply layer-wise quantization

    def _quantize_int8(self, module, name):
        # INT8 quantization (8-bit)
```

**Bit-width Trade-offs:**

| Bit-width | Dynamic Range | Quantization Error | Speedup |
|-----------|---------------|------------------|---------|
| FP32 | Full | 0% | 1× |
| FP16 | ±65504 | ~0.1% | 2× |
| INT8 | ±128 | ~0.5-1% | 4× |
| INT4 | ±8 | ~1-3% | 8× |

### Mathematical Foundation

**Quantization as Optimization Problem:**

Find quantization parameters (scale, zero_point) that minimize:
```
minimize: E[|clamp(round(x/s + z), 0, 2^b-1) · s - z|²]
subject to: scale > 0, 0 ≤ zero_point < 2^b - 1
```

Where s = scale, z = zero_point.

**Optimal Solution (for symmetric quantization):**
```
scale* = (x_max - x_min) / (2^b - 2)
zero_point* = (x_max + x_min) / 2
```

**Per-layer Quantization:**

Different layers have different sensitivity to quantization:
- Early layers: More sensitive to quantization (higher precision needed)
- Middle layers: Moderate sensitivity
- Final layers: Less sensitive (lower precision acceptable)

## 3. Distillation Agent

### Algorithm Details

The Distillation Agent transfers knowledge from a larger teacher model to a smaller student model using both soft and hard labels.

**Knowledge Distillation Loss:**

```
L = α · L_soft + (1 - α) · L_hard + λ · L_feature

L_soft = T² · KL(π_s(z/T) || π_t(z/T))      # Soft label loss
L_hard = CE(y, ŷ)                           # Hard label loss
L_feature = Σ_k ||F_s^k(x) - F_t^k(x)||²        # Feature alignment
```

**Components Explained:**

1. **Soft Label Loss (L_soft):**
   - π_t: Teacher's softmax output with temperature T
   - π_s: Student's softmax output with temperature T
   - KL: Kullback-Leibler divergence
   - Temperature T softens the output distribution

2. **Hard Label Loss (L_hard):**
   - Standard cross-entropy with ground truth labels
   - Ensures student still learns the task

3. **Feature Alignment Loss (L_feature):**
   - Matches intermediate layer features
   - Uses adaptive pooling for dimension mismatch

**Temperature T:**
- T → 0: Distribution approaches one-hot (teacher overconfident)
- T → ∞: Distribution becomes uniform (teacher underconfident)
- Typical range: [1.0, 20.0]

**Alpha α:**
- α = 0: Only hard labels (no distillation)
- α = 1: Only soft labels (pure distillation)
- Typical range: [0.0, 1.0]

### Implementation Details

**File:** `hadmc2/agents/distillation_agent.py`

**Class:** `DistillationAgent`

**Key Methods:**

```python
class DistillationAgent:
    def __init__(self, teacher_model, student_model, device='cpu')
        # Teacher provides knowledge, student receives it

    def distill(self, train_loader, val_loader,
               temperature=4.0, alpha=0.5, epochs=10):
        # Main distillation loop

    def _compute_feature_loss(self, inputs, layer_indices=None):
        # Align intermediate representations

    def _extract_features(self, model, inputs, layer_indices=None):
        # Extract features from intermediate layers
```

### Mathematical Foundation

**Upper Bound on Student Capacity:**

Given teacher parameters N_t and student parameters N_s:
```
L_s ≥ L_t - (N_t - N_s) · log(N_t / N_s) - 2 · N_s · log(1 + N_t/N_s)
```

This indicates the student cannot be better than the teacher if it has enough capacity.

**Information Bottleneck:**

For distillation with compression ratio r:
```
I = r · L_t  # Information bottleneck
```

Student must encode the information in r · N_t parameters.

### Hyperparameters

| Parameter | Range | Default | Description |
|-----------|-------|----------|-------------|
| Temperature T | [1.0, 20.0] | 4.0 | Softness of soft labels |
| Alpha α | [0.0, 1.0] | 0.5 | Balance soft/hard loss |
| Feature Weight λ | [0.0, 1.0] | 0.1 | Feature loss importance |
| Epochs | [1, 100] | 10 | Training duration |
| Learning Rate | [1e-5, 1e-3] | 1e-4 | Optimizer LR |

## 4. Fusion Agent

### Algorithm Details

The Fusion Agent identifies and fuses common operator patterns to improve inference efficiency.

**Supported Patterns:**

1. **Conv + BN:**
   ```
   Output = Conv(x) ⊙ γ + β  →  Conv_fused(x)
   ```

2. **Conv + BN + ReLU:**
   ```
   Output = ReLU(Conv(x) ⊙ γ + β)  →  Conv_fused(x)
   ```

3. **Conv + ReLU:**
   ```
   Output = ReLU(Conv(x))  → Conv_ReLU(x)
   ```

**BatchNorm Fusion Mathematics:**

For Conv with weight W, bias b, BatchNorm with γ, β, mean μ, var σ:
```
std = √(var + ε)
γ_norm = γ / std
β_norm = (b - μ) · γ_norm + β

W_fused = W · γ_norm.reshape(-1, 1, 1, 1)
b_fused = β_norm
```

**Performance Gains:**

| Pattern | FLOPs Reduction | Memory Access Reduction | Speedup |
|---------|----------------|----------------------|--------|
| Conv + BN | ~10% | ~25% | 1.2-1.5× |
| Conv + BN + ReLU | ~10% | ~25% | 1.2-1.5× |
| Conv + ReLU | ~2% | ~5% | 1.05-1.1× |

### Implementation Details

**File:** `hadmc2/agents/fusion_agent.py`

**Class:** `FusionAgent`

**Key Methods:**

```python
class FusionAgent:
    def __init__(self, model, device='cpu')
        # Analyze model for fusable patterns

    def analyze(self) -> List[Dict]:
        # Find all fusable patterns in model

    def fuse(self, fusion_config, model=None) -> nn.Module:
        # Apply fusion operations

    def _fuse_conv_bn(self, model, conv_name, bn_name):
        # Fuse Conv and BatchNorm

    def _fuse_conv_bn_relu(self, model, conv_name, bn_name, relu_name):
        # Fuse Conv, BN, and ReLU
```

**Pattern Detection:**

The agent traverses the model graph and identifies sequences:
```
Module sequence: [Conv2d, BatchNorm2d, ReLU]
                ↓
Pattern: conv_bn_relu
Fused to: Conv_fused
```

### Mathematical Foundation

**BatchNorm Correctness:**

After fusion, for input x:
```
Original: y = ReLU(W · x + b) ⊙ γ + β
Fused:  y' = W' · x + b'

We need: y' ≈ y
```

The fusion ensures mathematical equivalence in eval mode.

**Gradient Flow in Training:**

During training, fused layers require special handling:
1. Unfuse for backpropagation
2. Compute gradients through original graph
3. Refuse for weight update
4. OR: Use autograd-friendly fusion

HAD-MC 2.0 applies fusion after training (eval mode), avoiding gradient complications.

## 5. Update Agent

### Algorithm Details

The Update Agent enables efficient model updates for edge devices through three strategies.

**Strategies Explained:**

1. **Full Update:**
   - Download complete model
   - Simple but high bandwidth
   - Size: ~100% of model

2. **Incremental Update:**
   - Download and update only r fraction of layers
   - Reduces bandwidth by factor r
   - Accuracy may degrade temporarily

3. **Hash-based Update:**
   - Identify changed weight clusters
   - Download only delta (changed clusters)
   - Maximizes bandwidth efficiency

**Hash-based Update Algorithm:**

```
# Training phase:
For each layer:
    cluster_weights ← KMeans(weights, k=256)
    hash_table ← centroids

# Update phase:
For each layer:
    new_hash ← SHA256(new_weights)
    if new_hash != baseline_hash[layer]:
        # Determine changed clusters
        changed_clusters ← find_changed_clusters(new_weights, baseline_weights)
        # Download only delta
        delta ← centroids[changed_clusters]
        update_model(delta)
```

**Clustering:**

Uses K-means on flattened weight tensors:
```
Input: W ∈ R^(n)  # Flattened weights
Objective: minimize Σ ||W_j - μ_c(j)||²
where: c(j) ∈ {1, ..., k} is cluster assignment
       μ_c(j) is centroid of cluster c(j)
```

### Implementation Details

**File:** `hadmc2/agents/update_agent.py`

**Class:** `UpdateAgent`

**Key Methods:**

```python
class UpdateAgent:
    def __init__(self, model, device='cpu')
        # Model to be updated

    def update(self, new_data_loader,
             update_strategy='full', update_ratio=0.5):
        # Apply update strategy

    def build_hash_tables(self, num_clusters=256):
        # Create weight clusters

    def compute_delta(self, old_model) -> Dict[str, torch.Tensor]:
        # Compute parameter delta

    def apply_delta(self, delta: Dict[str, torch.Tensor]):
        # Apply delta to model

    def get_update_size(self, delta=None) -> float:
        # Calculate update size in MB
```

**Efficiency Metrics:**

| Strategy | Bandwidth Usage | Update Time | Accuracy Recovery |
|-----------|----------------|-------------|------------------|
| Full | 100% | Fast | N/A |
| Incremental (r=0.5) | 50% | Medium | Yes |
| Incremental (r=0.1) | 10% | Slow | Yes |
| Hash-based | ~20-40% | Medium | Yes |

## Agent Coordination

### MARL Framework

All five agents operate simultaneously under PPO coordination:

```
State S_t → PPO → Actions A_t → Agents → Model M_{t+1} → Metrics → Reward R_t
       ↑____________________|____________________↓________|_________|
```

**Coordination Challenges:**
1. Order independence: Agents shouldn't interfere
2. Trade-off balance: Pruning vs. accuracy, quantization vs. speed
3. Hardware constraints: Platform-specific limitations
4. Convergence: All agents must converge together

**Action Encoding:**

PPO outputs joint action space:
```
A_t = [a_pruing, a_quantization, a_distillation, a_fusion, a_update]

where:
a_pruing ∈ {0, ..., 10 × n_layers - 1}      # 10 pruning ratios × n layers
a_quantization ∈ {0, ..., 4 × n_layers - 1}  # 4 bit widths × n layers
a_distillation ∈ R²                               # temperature, alpha (continuous)
a_fusion ∈ {0, ..., 6 × n_points - 1}        # 6 patterns × n points
a_update ∈ {0, ..., 3 × 10 - 1}               # 3 strategies × 10 ratios
```

### Training Dynamics

**Early Training:**
- High exploration
- Large action space exploration
- Pareto frontier grows rapidly

**Mid Training:**
- Exploitation of good configurations
- Pareto front stabilizes
- Agents learn preferences

**Late Training:**
- Fine-tuning of trade-offs
- Minimal policy changes
- Convergence to local optima

## References

1. Louizos, C., et al. (2020). "DropBlock: A structured dropout method for convolutional neural networks." ICLR 2020.

2. Zhou, A., et al. (2017). "Incremental network quantization: Towards lossless bitwidth reduction." ICCV 2017.

3. Hinton, G., et al. (2015). "Distilling the knowledge in a neural network." arXiv:1503.02531.

4. Brock, A., et al. (2021). "Highly Efficient 4-bit Neural Network Inference on GPUs for Image Classification." arXiv:2109.03380.

5. Chen, T., et al. (2020). "Fusing Convolution and Batch Normalization for Faster Inference." arXiv:2008.04688.
