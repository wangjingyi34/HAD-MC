# HAD-MC Deployment Guide

This guide covers deploying HAD-MC compressed models on various platforms, with focus on domestic edge computing devices.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Domestic NPU Deployment](#domestic-npu-deployment)
3. [Cloud-Edge Deployment](#cloud-edge-deployment)
4. [Production Optimization](#production-optimization)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### One-Click Deployment

```bash
# Clone repository
git clone https://github.com/your-username/HAD-MC.git
cd HAD-MC

# Install dependencies
pip install -r requirements.txt

# Prepare datasets
python data/prepare_datasets.py

# Run full pipeline
python experiments/full_pipeline.py

# Results will be saved to results/pipeline_results.json
```

### Docker Deployment

```bash
# Build Docker image
docker build -t hadmc:latest .

# Run container
docker run --gpus all -v $(pwd)/results:/app/results hadmc:latest

# Results will be in ./results/
```

---

## Domestic NPU Deployment

### Cambricon MLU370

#### Prerequisites

- Cambricon MLU370 card
- Cambricon Neuware SDK 3.0+
- PyTorch-MLU 1.13+

#### Installation

```bash
# Install Cambricon PyTorch
pip install torch-mlu==1.13.0

# Verify installation
python -c "import torch_mlu; print(torch_mlu.is_available())"
```

#### Deployment

```python
import torch
import torch_mlu

# Load compressed model
model = torch.load('models/hadmc_compressed.pth')

# Move to MLU
model = model.to('mlu')

# Inference
input_data = torch.randn(1, 3, 224, 224).to('mlu')
output = model(input_data)
```

#### Performance Tuning

```python
# Enable operator fusion
torch_mlu.core.set_fusion_enabled(True)

# Set batch size for optimal throughput
batch_size = 32  # Adjust based on model size

# Enable mixed precision
from torch_mlu.core.amp import autocast
with autocast():
    output = model(input_data)
```

#### Expected Performance

| Metric | Value |
|--------|-------|
| Latency (batch=1) | 13.5 ms |
| Throughput (batch=32) | 2370 images/s |
| Power Consumption | 75W |
| Concurrent Streams | 24 |

### Huawei Ascend 310P

#### Prerequisites

- Huawei Ascend 310P card
- CANN 6.0+
- PyTorch-NPU 2.0+

#### Installation

```bash
# Install Ascend PyTorch
pip install torch-npu==2.0.0

# Verify installation
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

#### Deployment

```python
import torch
import torch_npu

# Load compressed model
model = torch.load('models/hadmc_compressed.pth')

# Move to NPU
model = model.to('npu:0')

# Inference
input_data = torch.randn(1, 3, 224, 224).to('npu:0')
output = model(input_data)
```

#### Performance Tuning

```python
# Enable ACL optimization
import torch_npu.npu.acl as acl
acl.init()

# Set precision mode
torch_npu.npu.set_option({
    'ACL_PRECISION_MODE': 'allow_mix_precision'
})

# Enable dynamic shape
torch_npu.npu.set_compile_mode(jit_compile=True)
```

#### Expected Performance

| Metric | Value |
|--------|-------|
| Latency (batch=1) | 14.2 ms |
| Throughput (batch=32) | 2250 images/s |
| Power Consumption | 65W |
| Concurrent Streams | 20 |

---

## Cloud-Edge Deployment

### Architecture

```
┌─────────────────┐
│  Cloud Server   │
│  (Model Update) │
└────────┬────────┘
         │ Incremental Update (Algorithm 5)
         │ Bandwidth: 18.7 MB (79% reduction)
         ▼
┌─────────────────┐
│  Edge Device    │
│  (MLU370/310P)  │
│  - Inference    │
│  - Local Cache  │
└─────────────────┘
```

### Cloud-Side Setup

```python
from hadmc.incremental_update import IncrementalUpdater

# Train new model
new_model = train_model(data)

# Compress with HAD-MC
compressed_model = apply_hadmc_pipeline(new_model)

# Compute delta
updater = IncrementalUpdater(block_size=4096)
delta = updater.compute_delta(old_model, compressed_model)

# Send delta to edge devices
send_to_edge_devices(delta)
```

### Edge-Side Setup

```python
from hadmc.incremental_update import IncrementalUpdater

# Receive delta from cloud
delta = receive_from_cloud()

# Apply delta to local model
updater = IncrementalUpdater()
updated_model = updater.apply_delta(current_model, delta)

# Verify integrity
if updater.verify_checksum(updated_model):
    current_model = updated_model
    save_model(current_model)
```

### Bandwidth Optimization

| Update Type | Full Model | HAD-MC Delta | Reduction |
|-------------|------------|--------------|-----------|
| Initial Deployment | 89.4 MB | 22.3 MB | 75.1% |
| Incremental Update | 89.4 MB | 18.7 MB | 79.1% |
| Fine-tune Update | 89.4 MB | 12.4 MB | 86.1% |

---

## Production Optimization

### Multi-Stream Inference

```python
import torch
import torch_mlu

# Load model
model = torch.load('models/hadmc_compressed.pth').to('mlu')
model.eval()

# Create multiple streams
num_streams = 24
streams = [torch_mlu.Stream() for _ in range(num_streams)]

# Inference with streams
def multi_stream_inference(inputs):
    results = []
    for i, input_data in enumerate(inputs):
        stream_id = i % num_streams
        with torch_mlu.stream(streams[stream_id]):
            output = model(input_data.to('mlu'))
            results.append(output)
    torch_mlu.synchronize()
    return results
```

### Batch Processing

```python
# Dynamic batching for optimal throughput
def dynamic_batch_inference(inputs, max_batch_size=32):
    results = []
    for i in range(0, len(inputs), max_batch_size):
        batch = inputs[i:i+max_batch_size]
        batch_tensor = torch.stack(batch).to('mlu')
        output = model(batch_tensor)
        results.extend(output.cpu().split(1))
    return results
```

### Model Caching

```python
# Cache frequently used models
from functools import lru_cache

@lru_cache(maxsize=10)
def load_model(model_path):
    model = torch.load(model_path)
    model = model.to('mlu')
    model.eval()
    return model

# Use cached model
model = load_model('models/hadmc_compressed.pth')
```

---

## Performance Benchmarks

### Latency Comparison

| Platform | FP32 Baseline | HAD-MC | Reduction |
|----------|---------------|--------|-----------|
| MLU370 | 45.2 ms | 13.5 ms | 70.1% |
| Ascend 310P | 48.7 ms | 14.2 ms | 70.8% |
| NVIDIA T4 | 42.3 ms | 12.8 ms | 69.7% |

### Throughput Comparison

| Platform | Batch Size | FP32 (img/s) | HAD-MC (img/s) | Speedup |
|----------|------------|--------------|----------------|---------|
| MLU370 | 1 | 22 | 74 | 3.4× |
| MLU370 | 32 | 710 | 2370 | 3.3× |
| Ascend 310P | 1 | 21 | 70 | 3.3× |
| Ascend 310P | 32 | 680 | 2250 | 3.3× |

### Power Efficiency

| Platform | Power (W) | Throughput (img/s) | Efficiency (img/s/W) |
|----------|-----------|-------------------|---------------------|
| MLU370 (FP32) | 150 | 710 | 4.7 |
| MLU370 (HAD-MC) | 75 | 2370 | **31.6** |
| Ascend 310P (FP32) | 130 | 680 | 5.2 |
| Ascend 310P (HAD-MC) | 65 | 2250 | **34.6** |

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptom:** RuntimeError: out of memory

**Solution:**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch_mlu.empty_cache()
```

#### 2. Low Throughput

**Symptom:** Throughput lower than expected

**Solution:**
```python
# Enable operator fusion
torch_mlu.core.set_fusion_enabled(True)

# Use multiple streams
num_streams = 24

# Optimize batch size
# Run benchmark to find optimal batch size
for bs in [1, 2, 4, 8, 16, 32]:
    throughput = benchmark(model, batch_size=bs)
    print(f"Batch {bs}: {throughput} img/s")
```

#### 3. Accuracy Drop

**Symptom:** Accuracy lower than expected after compression

**Solution:**
```python
# Fine-tune with more epochs
train_model(compressed_model, train_loader, epochs=10)

# Use higher precision for sensitive layers
allocator = LayerwisePrecisionAllocator(
    model, calib_loader,
    tau_h=1e-2,  # More layers in FP32
    tau_l=1e-4
)

# Reduce pruning ratio
pruner = GradientSensitivityPruner(
    model, train_loader,
    flops_target=0.7  # Less aggressive pruning
)
```

#### 4. Incremental Update Failure

**Symptom:** Model update fails or produces incorrect results

**Solution:**
```python
# Verify checksum before applying
if not updater.verify_checksum(delta):
    print("Checksum mismatch! Re-downloading...")
    delta = re_download_delta()

# Use smaller block size for more granular updates
updater = IncrementalUpdater(block_size=2048)

# Enable compression for delta transmission
import gzip
compressed_delta = gzip.compress(delta)
```

---

## Monitoring and Logging

### Performance Monitoring

```python
import time
import logging

logging.basicConfig(level=logging.INFO)

def monitor_inference(model, input_data):
    start_time = time.time()
    
    # Inference
    output = model(input_data)
    
    # Log metrics
    latency = (time.time() - start_time) * 1000
    logging.info(f"Latency: {latency:.2f} ms")
    
    # Check memory usage
    memory_used = torch_mlu.memory_allocated() / 1024**2
    logging.info(f"Memory: {memory_used:.2f} MB")
    
    return output
```

### Production Logging

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename=f'logs/hadmc_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log deployment events
logging.info("Model deployed successfully")
logging.info(f"Platform: MLU370")
logging.info(f"Model size: 22.3 MB")
logging.info(f"Expected latency: 13.5 ms")
```

---

## Best Practices

### 1. Model Versioning

```python
# Use semantic versioning
model_version = "1.2.3"  # major.minor.patch

# Save with version info
torch.save({
    'model_state_dict': model.state_dict(),
    'version': model_version,
    'compression_config': config,
    'timestamp': datetime.now()
}, f'models/hadmc_v{model_version}.pth')
```

### 2. A/B Testing

```python
# Deploy new model alongside old model
model_a = load_model('models/hadmc_v1.2.3.pth')  # Current
model_b = load_model('models/hadmc_v1.3.0.pth')  # New

# Route traffic
if user_id % 10 < 2:  # 20% traffic to new model
    output = model_b(input_data)
else:
    output = model_a(input_data)
```

### 3. Gradual Rollout

```python
# Rollout schedule
rollout_schedule = {
    'day_1': 0.1,   # 10% of devices
    'day_3': 0.3,   # 30% of devices
    'day_7': 0.7,   # 70% of devices
    'day_14': 1.0   # 100% of devices
}

# Update devices gradually
def should_update(device_id, day):
    rollout_pct = rollout_schedule.get(f'day_{day}', 0)
    return hash(device_id) % 100 < rollout_pct * 100
```

---

## Support

For deployment issues:
- **GitHub Issues**: https://github.com/your-username/HAD-MC/issues
- **Email**: support@hadmc.example.com
- **Documentation**: https://hadmc.readthedocs.io

For commercial deployment:
- **Enterprise Support**: enterprise@hadmc.example.com
- **Custom Integration**: consulting@hadmc.example.com
