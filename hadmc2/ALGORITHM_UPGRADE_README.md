# HAD-MC 2.0 Algorithm Upgrade Documentation

## Overview

HAD-MC 2.0 represents a major upgrade from the original HAD-MC framework, introducing a Multi-Agent Reinforcement Learning (MARL) approach for coordinated model compression. This document provides a comprehensive guide to all algorithm enhancements.

## Key Changes from HAD-MC 1.0 to 2.0

### Architecture Evolution

**HAD-MC 1.0 (Original):**
- Sequential pipeline: Pruning → Quantization → Distillation → Fusion
- Heuristic-based decisions
- No coordination between compression techniques

**HAD-MC 2.0 (Upgraded):**
- Parallel MARL with five specialized agents
- PPO controller for coordinated decision-making
- Hardware-aware optimization via HAL
- Dedicated Inference Engine (DIE) for deployment

## Directory Structure

```
hadmc2/
├── __init__.py                    # Package initialization
├── agents/                          # Five compression agents
│   ├── __init__.py
│   ├── pruning_agent.py           # Gradient-based structured pruning
│   ├── quantization_agent.py     # Layer-wise precision allocation
│   ├── distillation_agent.py     # Feature-aligned knowledge distillation
│   ├── fusion_agent.py           # Operator fusion
│   └── update_agent.py           # Incremental updates
├── controllers/                      # MARL coordination
│   ├── __init__.py
│   ├── ppo_controller.py          # PPO implementation
│   └── marl_coordinator.py       # Agent coordination
├── hardware/                        # Hardware abstraction
│   ├── __init__.py
│   ├── hal.py                    # Hardware Abstraction Layer
│   ├── latency_lut.py             # Latency Lookup Table
│   └── profiler.py               # Hardware profiler
├── inference/                        # Inference optimization
│   ├── __init__.py
│   ├── die.py                    # Dedicated Inference Engine
│   └── tensorrt_backend.py       # TensorRT backend
├── rewards/                         # Multi-objective optimization
│   ├── __init__.py
│   └── reward_function.py         # Pareto-aware reward function
├── training/                         # Training system
│   ├── __init__.py
│   ├── trainer.py                # Main HAD-MC 2.0 trainer
│   └── buffer.py                 # Rollout buffer for PPO
├── utils/                            # Utilities
│   ├── __init__.py
│   ├── state.py                  # State representation
│   ├── action.py                 # Action space
│   ├── metrics.py                # Metrics calculation
│   └── config.py                 # Configuration loading
├── configs/                          # Hardware-specific configs
│   ├── default.yaml               # Default configuration
│   ├── nvidia_a100.yaml           # NVIDIA A100 settings
│   ├── jetson_orin.yaml           # NVIDIA Jetson Orin
│   ├── ascend_310.yaml            # Huawei Ascend 310
│   └── hygon_dcu.yaml             # Hygon DCU
└── tests/                            # Unit and integration tests
    ├── __init__.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_state.py
    │   ├── test_action.py
    │   ├── test_ppo_controller.py
    │   ├── test_agents.py
    │   └── test_reward.py
    └── integration/
        ├── __init__.py
        ├── test_training_loop.py
        └── test_full_pipeline.py
```

## Component Details

### 1. State Representation (`utils/state.py`)

**Purpose:** Encodes the current state of model, hardware, and compression into a tensor for MARL.

**Components:**
- `model_state`: Architecture, FLOPs, parameter counts
- `hardware_state`: Compute capability, memory bandwidth, power budget, supported precisions
- `compression_state`: Current pruning ratios, bit widths, accuracy, latency, energy, size

**Key Methods:**
- `update_model_state(model)`: Extract model features
- `to_tensor()`: Convert to normalized tensor for neural networks
- `copy()`: Deep copy state

**Example Usage:**
```python
from hadmc2.utils.state import State, create_state_from_model_and_hardware

state = State()
state.update_model_state(model)
state_tensor = state.to_tensor()
```

### 2. Action Space (`utils/action.py`)

**Purpose:** Defines action spaces for all five agents.

**Agent Action Spaces:**
- **Pruning**: `{layer_idx, pruning_ratio}` where pruning_ratio ∈ [0.0, 0.1, ..., 0.9]
- **Quantization**: `{layer_idx, bit_width}` where bit_width ∈ [4, 8, 16, 32]
- **Distillation**: `{temperature, alpha}` where temperature ∈ [1.0, 20.0], alpha ∈ [0.0, 1.0]
- **Fusion**: `{pattern, start_layer}` where pattern ∈ [none, conv_bn, conv_relu, conv_bn_relu]
- **Update**: `{strategy, update_ratio}` where strategy ∈ [full, incremental, hash_based]

**Key Methods:**
- `sample_*_action()`: Sample actions for each agent
- `encode_action(agent, action)`: Encode action to tensor index
- `decode_action(agent, index)`: Decode index back to action

### 3. PPO Controller (`controllers/ppo_controller.py`)

**Purpose:** Implements Proximal Policy Optimization for MARL.

**Architecture:**
- `PolicyNetwork`: Shared feature extractor with agent-specific heads
  - Pruning head (discrete)
  - Quantization head (discrete)
  - Distillation head (continuous - mean and std)
  - Fusion head (discrete)
  - Update head (discrete)

- `ValueNetwork`: State-value estimation network

**Key Components:**
- GAE (Generalized Advantage Estimation) for better credit assignment
- Clipped objective for stable policy updates
- Entropy bonus for exploration

**Training Loop:**
1. Collect trajectory
2. Compute GAE advantages
3. PPO update for K epochs
4. Clear buffer

**Hyperparameters:**
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE smoothing (default: 0.95)
- `clip_epsilon`: PPO clipping (default: 0.2)
- `value_coef`: Value loss weight (default: 0.5)
- `entropy_coef`: Entropy bonus (default: 0.01)

### 4. Hardware Abstraction Layer (`hardware/hal.py`)

**Purpose:** Abstracts away hardware-specific details for cross-platform compatibility.

**Supported Platforms:**
- NVIDIA A100 (data center GPU)
- NVIDIA Jetson Orin (edge GPU)
- Huawei Ascend 310 (edge NPU)
- Hygon DCU (domestic accelerator)

**Key Features:**
- Compute capability modeling (TFLOPS/TOPS)
- Memory bandwidth estimation (GB/s)
- Power budget tracking (W)
- Precision support detection
- Tensor Core / sparsity support detection

**Latency Estimation:**
- Analytical model based on FLOPs and memory access
- Latency = FLOPs / compute + memory / bandwidth
- Supports FP32, FP16, INT8, INT4 precisions

**Usage:**
```python
from hadmc2.hardware.hal import SimulatedHardwareAbstractionLayer

hal = SimulatedHardwareAbstractionLayer()
hw_config = hal.get_hardware_config()
latency = hal.estimate_latency(model_config)
```

### 5. Latency Lookup Table (`hardware/latency_lut.py`)

**Purpose:** Cache measured latencies for fast estimation during MARL training.

**Key Features:**
- Multi-dimensional indexing: (layer_name, precision, sparsity)
- Automatic fallback to nearest configuration
- Serialization/deserialization for reuse

**Building the LUT:**
```python
from hadmc2.hardware.latency_lut import LatencyLookupTable

lut = LatencyLookupTable()
lut.build(model, input_shape=(1, 3, 640, 640), num_samples=100)
lut.save('latency_lut.pkl')
```

### 6. Dedicated Inference Engine (`inference/die.py`)

**Purpose:** Optimizes compressed models for efficient inference.

**Optimization Pipeline:**
1. **Operator Fusion**: Combine consecutive operations
   - Conv + BN → single Conv
   - Conv + BN + ReLU → single Conv
   - Conv + ReLU → single combined module

2. **Sparsity Optimization**
   - 2:4 structured sparsity (NVIDIA Ampere)
   - Unstructured sparsity

3. **Mixed Precision**
   - Layer-wise FP16/INT8/INT4

**Fusion Mathematics:**

For Conv + BN fusion:
```
W_fused = W * (γ / √(var + ε))
b_fused = (b - mean) * (γ / √(var + ε)) + β
```

Where γ, β are BN scale and bias, mean and var are running statistics.

**Usage:**
```python
from hadmc2.inference.die import DedicatedInferenceEngine

die = DedicatedInferenceEngine(hal_config)
compressed_model = die.optimize(model, compression_config)
output = die.inference(input_tensor)
```

### 7. Five Compression Agents

#### 7.1 Pruning Agent (`agents/pruning_agent.py`)

**Algorithm:** Gradient-based structured channel pruning

**Importance Metric:**
```
I(j) = |W(j) · ∂L/∂O(j)|  # Taylor approximation
```

**Pruning Process:**
1. Compute importance for each channel
2. Identify least important channels (based on importance)
3. Remove those channels (set weights to zero)

**Action Space:** `{layer_idx ∈ [0, n-1], pruning_ratio ∈ [0.0, 0.1, ..., 0.9]}`

#### 7.2 Quantization Agent (`agents/quantization_agent.py`)

**Algorithm:** Layer-wise precision allocation

**Calibration:**
- Collect activation statistics (min, max, mean, std)
- Compute scale and zero point for quantization

**Quantization Formula (INT8):**
```
scale = (max - min) / 255
zero_point = -min / scale
quantized = clamp(round(x / scale + zero_point), 0, 255)
dequantized = (quantized - zero_point) * scale
```

**Action Space:** `{layer_idx ∈ [0, n-1], bit_width ∈ [4, 8, 16, 32]}`

#### 7.3 Distillation Agent (`agents/distillation_agent.py`)

**Algorithm:** Feature-aligned knowledge distillation

**Loss Function:**
```
L_total = α · L_soft + (1-α) · L_hard + λ · L_feature

L_soft = KL(π_s(T_t) || π_t(T_t)) · T²  # Soft label loss
L_hard = CE(y, ŷ)                    # Hard label loss
L_feature = ||F_s(x) - F_t(x)||²         # Feature alignment
```

**Parameters:**
- `T`: Temperature (default: 4.0, range: [1.0, 20.0])
- `α`: Distillation weight (default: 0.5, range: [0.0, 1.0])
- `λ`: Feature loss weight (default: 0.1)

**Action Space:** `{temperature ∈ [1.0, 20.0], alpha ∈ [0.0, 1.0]}` (continuous)

#### 7.4 Fusion Agent (`agents/fusion_agent.py`)

**Algorithm:** Pattern-based operator fusion

**Supported Patterns:**
1. `none`: No fusion
2. `conv_bn`: Conv2d + BatchNorm2d
3. `conv_relu`: Conv2d + ReLU
4. `conv_bn_relu`: Conv2d + BatchNorm2d + ReLU

**Fusion Logic:**
- Detect consecutive module patterns in model graph
- Replace pattern with optimized single module
- Update model structure

**Action Space:** `{pattern ∈ [none, conv_bn, conv_relu, conv_bn_relu], start_layer}`

#### 7.5 Update Agent (`agents/update_agent.py`)

**Algorithm:** Incremental model updates for edge deployment

**Update Strategies:**
1. **Full Update**: Download complete model
2. **Incremental Update**: Update only selected layers (ratio of total)
3. **Hash-based Update**: Delta update using weight hashing

**Hash-based Update:**
```
hash = SHA256(weights)
if hash != baseline_hash:
    download delta
    update only changed weight clusters
```

**Action Space:** `{strategy ∈ [full, incremental, hash_based], update_ratio ∈ [0.1, ..., 1.0]}`

### 8. Reward Function (`rewards/reward_function.py`)

**Purpose:** Multi-objective reward with Pareto optimization

**Reward Formula:**
```
R_total = w_acc · R_acc + w_lat · R_lat + w_eng · R_eng + w_size · R_size - penalty

R_acc = (acc - acc_baseline) / acc_baseline           # Accuracy reward
R_lat = (lat_baseline - lat) / lat_baseline          # Latency reward (lower is better)
R_eng = (eng_baseline - eng) / eng_baseline          # Energy reward (lower is better)
R_size = (size_baseline - size) / size_baseline        # Size reward (lower is better)
```

**Penalty Function:**
```
penalty = 10 · max(0, θ_acc - acc) if acc < θ_acc
         + 5 · (lat - θ_lat) / θ_lat if lat > θ_lat
         + 2 · (eng - θ_eng) / θ_eng if eng > θ_eng
         + 1 · (size - θ_size) / θ_size if size > θ_size
```

**Pareto Optimization:**
- Track Pareto-optimal configurations
- Give bonus for Pareto-optimal points
- Penalize distance from Pareto frontier

**Default Weights:**
- `accuracy_weight = 1.0`
- `latency_weight = 0.5`
- `energy_weight = 0.3`
- `size_weight = 0.2`

### 9. Training System (`training/trainer.py`, `training/buffer.py`)

**Purpose:** Orchestrates all agents and PPO controller

**Training Loop:**
```
for episode in num_episodes:
    state = get_initial_state()

    for step in max_steps:
        # 1. PPO selects actions
        actions = ppo.select_actions(state)

        # 2. Agents apply actions
        model = apply_all_actions(model, actions)

        # 3. Evaluate compressed model
        metrics = evaluate(model)

        # 4. Compute reward
        reward = reward_fn.compute(metrics, baseline)

        # 5. Update state
        next_state = get_state(model)

        # 6. Store in buffer
        buffer.add(state, actions, log_prob, reward, done, value)

        state = next_state

    # 7. PPO update
    if episode % update_interval == 0:
        ppo.update()
```

**Rollout Buffer (`buffer.py`):**
- Stores: states, actions, log_probs, rewards, dones, values
- Capacity-based FIFO
- Supports random batch sampling

### 10. Configuration System

**Configuration Files:**
- `default.yaml`: Default hyperparameters
- `nvidia_a100.yaml`: GPU-specific settings
- `jetson_orin.yaml`: Edge device settings
- `ascend_310.yaml`: NPU settings
- `hygon_dcu.yaml`: DCU settings

**Key Configurations:**
```yaml
training:
  num_episodes: 100
  max_steps_per_episode: 50
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95

constraints:
  accuracy_threshold: 0.90
  latency_threshold: 10.0  # ms
  energy_threshold: 1.0     # J
  size_threshold: 5.0        # MB
```

## Testing

### Unit Tests
Located in `tests/unit/`:
- `test_state.py`: State encoding/decoding
- `test_action.py`: Action space sampling
- `test_ppo_controller.py`: PPO components
- `test_agents.py`: Individual agent functionality
- `test_reward.py`: Reward computation

### Integration Tests
Located in `tests/integration/`:
- `test_training_loop.py`: End-to-end training
- `test_full_pipeline.py`: Complete system validation

**Running Tests:**
```bash
# Unit tests
pytest hadmc2/tests/unit/ -v

# Integration tests
pytest hadmc2/tests/integration/ -v

# All tests
pytest hadmc2/tests/ -v
```

## Usage Examples

### Basic Usage
```python
from hadmc2.training.trainer import HADMCTrainer
from hadmc2.hardware.hal import SimulatedHardwareAbstractionLayer

# Initialize hardware
hal = SimulatedHardwareAbstractionLayer()

# Create trainer
trainer = HADMCTrainer(
    model=model,
    teacher_model=teacher,
    train_loader=train_loader,
    val_loader=val_loader,
    hal=hal,
    config={
        'num_episodes': 100,
        'device': 'cuda',
        'accuracy_threshold': 0.90,
    }
)

# Train
results = trainer.train(save_dir='./checkpoints')
print(f"Best reward: {results['best_reward']:.4f}")
```

### Using Individual Agents
```python
from hadmc2.agents.pruning_agent import PruningAgent
from hadmc2.agents.quantization_agent import QuantizationAgent
from hadmc2.inference.die import DedicatedInferenceEngine

# Initialize agents
pruning = PruningAgent(model, train_loader, device='cuda')
quantization = QuantizationAgent(model, cal_loader, device='cuda')

# Apply compression
pruned = pruning.prune({'layer1': 0.3, 'layer2': 0.2})
quantized = quantization.quantize({'layer1': 8, 'layer2': 16})

# Optimize for inference
die = DedicatedInferenceEngine()
compressed = die.optimize(model, {'pruning': {...}, 'quantization': {...}})
```

### Custom Configuration
```python
from hadmc2.utils.config import load_config

# Load hardware-specific config
config = load_config(config_path='hadmc2/configs/nvidia_a100.yaml')

# Use custom hardware config
from hadmc2.hardware.hal import SimulatedHardwareAbstractionLayer
hal = SimulatedHardwareAbstractionLayer('nvidia_a100.yaml')
```

## Performance Considerations

### Computational Complexity
- **State Encoding**: O(L) where L is number of layers
- **Action Selection**: O(1) forward pass through policy network
- **Pruning**: O(N) where N is number of parameters
- **Quantization**: O(N) for calibration
- **Reward Computation**: O(1) with Pareto check O(P) where P is Pareto front size

### Memory Usage
- **Policy Network**: ~O(H*D) where H is hidden size, D is total action dims
- **Value Network**: ~O(H)
- **Buffer**: O(B*T*D) where B is batch size, T is trajectory length
- **Pareto Front**: O(P*D) where P is number of Pareto points

### Scalability
- Supports models with arbitrary depth
- Number of agents is fixed at 5
- Pareto front size can be capped to control memory

## Algorithm Complexity Comparison

| Component | HAD-MC 1.0 | HAD-MC 2.0 |
|-----------|---------------|---------------|
| Decision Making | Heuristic | PPO + MARL |
| Coordination | Sequential | Parallel |
| Hardware Awareness | None | Full HAL |
| Adaptability | Fixed | Learned |
| Sample Efficiency | O(1) | O(Episodes × Steps) |

## References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. Wang, H., et al. (2019). "Pruning from Gradient." ICCV 2019
3. Mishra, A., et al. (2020). "Adaptive Quantization-aware Training." ICLR 2020
4. Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531
5. NVIDIA. (2020). "TensorRT: High Performance Deep Learning Inference Library"

## Future Enhancements

### Short-term
- [ ] Add more fusion patterns (residual blocks)
- [ ] Implement true INT4 quantization
- [ ] Add more hardware platforms
- [ ] Improve sparsity efficiency

### Long-term
- [ ] Neural Architecture Search (NAS)
- [ ] Automatic hyperparameter tuning
- [ ] Distributed training support
- [ ] Real-world deployment examples

## Troubleshooting

### Common Issues

**1. Out of Memory:**
- Reduce batch size
- Use gradient accumulation
- Limit Pareto front size

**2. Slow Training:**
- Use smaller models for debugging
- Reduce number of episodes
- Disable agents not in use

**3. Poor Convergence:**
- Adjust learning rate
- Tune PPO hyperparameters
- Check reward function weights

**4. TensorRT Issues:**
- Ensure CUDA compatibility
- Check ONNX export version
- Verify input tensor shapes

## Contributing

When adding new features:
1. Update this README
2. Add unit tests
3. Update configuration schema
4. Add usage examples
5. Run full test suite

## License

This code is part of the HAD-MC research project. Please see the main project repository for licensing information.
