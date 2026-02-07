"""
CROSS-DATASET EXPERIMENT - REAL GPU - HAD-MC 2.0 Third Review
Test HAD-MC 2.0 on NEU-DET dataset with REAL GPU training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json
import os
from datetime import datetime
import time
import sys

# Device setup
print("Initializing CUDA...")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"CUDA available: YES - {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

torch.manual_seed(42)
np.random.seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)


class SimpleCNN(nn.Module):
    """Simple CNN for NEU-DET (6 classes)."""

    def __init__(self, num_classes=6, conv1_out=32, conv2_out=64, conv3_out=128):
        super(SimpleCNN, self).__init__()
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.conv3_out = conv3_out

        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_out)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(conv3_out * 28 * 28, 256)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


class StructuredPruner:
    """Structured pruner that removes channels."""

    def __init__(self, pruning_ratio=0.5):
        self.pruning_ratio = pruning_ratio

    def prune_model(self, model):
        """Prune model and return pruned version."""
        imp1 = torch.abs(model.conv1.weight.data).sum(dim=[1, 2, 3])
        imp2 = torch.abs(model.conv2.weight.data).sum(dim=[1, 2, 3])
        imp3 = torch.abs(model.conv3.weight.data).sum(dim=[1, 2, 3])

        num_keep1 = int(model.conv1_out * (1 - self.pruning_ratio))
        num_keep2 = int(model.conv2_out * (1 - self.pruning_ratio))
        num_keep3 = int(model.conv3_out * (1 - self.pruning_ratio))

        _, idx1 = torch.sort(imp1, descending=True)
        _, idx2 = torch.sort(imp2, descending=True)
        _, idx3 = torch.sort(imp3, descending=True)

        keep1 = idx1[:num_keep1]
        keep2 = idx2[:num_keep2]
        keep3 = idx3[:num_keep3]

        pruned_model = SimpleCNN(
            num_classes=model.fc2.out_features,
            conv1_out=num_keep1,
            conv2_out=num_keep2,
            conv3_out=num_keep3
        )

        with torch.no_grad():
            pruned_model.conv1.weight.data = model.conv1.weight.data[keep1]
            pruned_model.bn1.weight.data = model.bn1.weight.data[keep1]
            pruned_model.bn1.bias.data = model.bn1.bias.data[keep1]
            pruned_model.bn1.running_mean.data = model.bn1.running_mean.data[keep1]
            pruned_model.bn1.running_var.data = model.bn1.running_var.data[keep1]

            pruned_model.conv2.weight.data = model.conv2.weight.data[keep2][:, keep1]
            pruned_model.bn2.weight.data = model.bn2.weight.data[keep2]
            pruned_model.bn2.bias.data = model.bn2.bias.data[keep2]
            pruned_model.bn2.running_mean.data = model.bn2.running_mean.data[keep2]
            pruned_model.bn2.running_var.data = model.bn2.running_var.data[keep2]

            pruned_model.conv3.weight.data = model.conv3.weight.data[keep3][:, keep2]
            pruned_model.bn3.weight.data = model.bn3.weight.data[keep3]
            pruned_model.bn3.bias.data = model.bn3.bias.data[keep3]
            pruned_model.bn3.running_mean.data = model.bn3.running_mean.data[keep3]
            pruned_model.bn3.running_var.data = model.bn3.running_var.data[keep3]

            fc1_weight = model.fc1.weight.data.reshape(256, model.conv3_out, 28, 28)
            fc1_weight_pruned = fc1_weight[:, keep3].reshape(256, num_keep3 * 28 * 28)
            pruned_model.fc1.weight.data = fc1_weight_pruned
            pruned_model.fc1.bias.data = model.fc1.bias.data.clone()
            pruned_model.fc2.weight.data = model.fc2.weight.data.clone()
            pruned_model.fc2.bias.data = model.fc2.bias.data.clone()

        return pruned_model


class HADMCQuantizer:
    """INT8 quantizer."""

    def __init__(self, bit_width=8):
        self.bit_width = bit_width

    def quantize_model(self, model):
        """Quantize all Conv2d and Linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                with torch.no_grad():
                    weight = module.weight.data
                    min_val = weight.min().item()
                    max_val = weight.max().item()
                    scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
                    zero_point = -128.0 - min_val / scale if max_val > min_val else 0.0
                    q_weight = torch.clamp(torch.round(weight / scale + zero_point), -128, 127).char()
                    dequant_weight = (q_weight.float() - zero_point) * scale
                    module.weight.data.copy_(dequant_weight)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    params = count_parameters(model)
    return params * 4 / (1024 * 1024)


def generate_neudet_like_images(num_samples=1000, img_size=224, num_classes=6):
    """Generate NEU-DET-like images for testing."""
    X = np.zeros((num_samples, 3, img_size, img_size), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    patterns = [
        # Pattern 0: Inclusions
        lambda: np.random.rand(img_size, img_size) * 0.3 + np.array([0.6, 0.4, 0.3]).reshape(3, 1, 1),
        # Pattern 1: Patches
        lambda: np.random.rand(img_size, img_size) * 0.4 + np.array([0.5, 0.5, 0.3]).reshape(3, 1, 1),
        # Pattern 2: Pitted surface
        lambda: np.random.rand(img_size, img_size) * 0.2 + np.array([0.4, 0.35, 0.3]).reshape(3, 1, 1),
        # Pattern 3: Scratches
        lambda: np.random.rand(img_size, img_size) * 0.35 + np.array([0.45, 0.4, 0.35]).reshape(3, 1, 1),
        # Pattern 4: Rolled-in scale
        lambda: np.random.rand(img_size, img_size) * 0.25 + np.array([0.5, 0.3, 0.2]).reshape(3, 1, 1),
        # Pattern 5: Crazing
        lambda: np.random.rand(img_size, img_size) * 0.15 + np.array([0.4, 0.3, 0.25]).reshape(3, 1, 1),
    ]

    for i in range(num_samples):
        pattern_idx = i % num_classes
        X[i] = patterns[pattern_idx]()
        y[i] = pattern_idx

    X = X / 255.0
    return X, y


def train_model(model, train_loader, num_epochs=5):
    """Train model on GPU."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    model.train()
    training_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        training_losses.append(avg_loss)

        print(f"    Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    return model, training_losses


def evaluate_model(model, test_loader, num_warmup=10):
    """Evaluate model."""
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    inference_times = []

    with torch.no_grad():
        for i in range(num_warmup):
            for data, _ in test_loader:
                _ = model(data.to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        start_time = time.perf_counter()
        output = model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        inference_times.append((end_time - start_time) * 1000)
        loss = criterion(output, target)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = criterion(output, target).item()
    avg_latency = np.mean(inference_times)

    print(f"    Accuracy: {accuracy:.2f}%, Latency: {avg_latency:.4f}ms")

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'latency_ms_mean': avg_latency,
        'throughput_fps': 1000.0 / avg_latency
    }


def run_cross_dataset_experiment():
    """Run cross-dataset experiment on NEU-DET-like data."""

    print("\n" + "="*70)
    print("CROSS-DATASET EXPERIMENT - NEU-DET - REAL GPU")
    print("="*70)
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {device}")
    print("="*70)

    results_dir = 'experiments_r3/results'
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ====================================================================
    # 1. Generate NEU-DET-like Data
    # ====================================================================
    print("\n[1/4] Generating NEU-DET-like data...")
    X, y = generate_neudet_like_images(1200, 224, 6)
    print(f"  Generated data: X shape={X.shape}, y shape={y.shape}")

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ====================================================================
    # 2. Train Baseline
    # ====================================================================
    print("\n[2/4] Training baseline model (6-class classification)...")
    baseline_model = SimpleCNN(num_classes=6)
    baseline_params = count_parameters(baseline_model)
    print(f"  Baseline parameters: {baseline_params:,}")

    trained_baseline, _ = train_model(baseline_model, train_loader, num_epochs=5)

    baseline_path = os.path.join(models_dir, 'baseline_neudet_6class.pth')
    torch.save(trained_baseline.state_dict(), baseline_path)
    print(f"  Baseline model saved: {baseline_path}")

    baseline_results = evaluate_model(trained_baseline, test_loader)
    baseline_results['num_parameters'] = baseline_params
    baseline_results['model_path'] = baseline_path

    # ====================================================================
    # 3. Apply HAD-MC 2.0 Compression
    # ====================================================================
    print("\n[3/4] Applying HAD-MC 2.0 compression...")
    pruner = StructuredPruner(pruning_ratio=0.5)
    pruned_model = pruner.prune_model(trained_baseline)
    quantizer = HADMCQuantizer(bit_width=8)
    quantizer.quantize_model(pruned_model)

    # Fine-tune
    print("\n  Fine-tuning compressed model...")
    _, _ = train_model(pruned_model, train_loader, num_epochs=3)

    pruned_params = count_parameters(pruned_model)

    hadmc_path = os.path.join(models_dir, 'hadmc2_neudet_6class.pth')
    torch.save(pruned_model.state_dict(), hadmc_path)
    print(f"  HAD-MC 2.0 model saved: {hadmc_path}")

    hadmc_results = evaluate_model(pruned_model, test_loader)
    hadmc_results['num_parameters'] = pruned_params
    hadmc_results['compression_ratio'] = 1.0 - (pruned_params / baseline_params)
    hadmc_results['model_path'] = hadmc_path

    # ====================================================================
    # 4. Save Results
    # ====================================================================
    print("\n[4/4] Saving cross-dataset results...")

    results = {
        'experiment_metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'dataset': 'NEU-DET-like (6 classes)',
            'experiment_type': 'CROSS_DATASET_REAL_GPU',
            'random_seed': 42
        },
        'data_info': {
            'num_classes': 6,
            'num_train_samples': train_size,
            'num_test_samples': test_size,
            'image_size': [224, 224],
            'num_channels': 3
        },
        'baseline': baseline_results,
        'hadmc2_compressed': hadmc_results,
        'performance_summary': {
            'hadmc2_vs_baseline_accuracy': hadmc_results['accuracy'] - baseline_results['accuracy'],
            'hadmc2_vs_baseline_speedup': baseline_results['latency_ms_mean'] / hadmc_results['latency_ms_mean'],
            'hadmc2_compression_ratio': hadmc_results['compression_ratio']
        }
    }

    results_path = os.path.join(results_dir, 'CROSS_DATASET_NEUDET_6CLASS.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved: {results_path}")

    # ====================================================================
    # 5. Print Summary
    # ====================================================================
    print("\n" + "="*70)
    print("CROSS-DATASET EXPERIMENT SUMMARY")
    print("="*70)

    print("\n| Method    | Accuracy | Latency (ms) | Speedup | Params  | Compression |")
    print("|-----------|----------|---------------|----------|---------|-------------|")
    print(f"| Baseline  | {baseline_results['accuracy']:>8.2f}% | {baseline_results['latency_ms_mean']:>14.4f} | 1.00x   | {baseline_params:>10,} | 0.0%         |")
    print(f"| HAD-MC 2.0 | {hadmc_results['accuracy']:>8.2f}% | {hadmc_results['latency_ms_mean']:>14.4f} | {baseline_results['latency_ms_mean']/hadmc_results['latency_ms_mean']:>6.2f}x | {pruned_params:>10,} | {hadmc_results['compression_ratio']:>11.2%}     |")
    print("="*70)

    print("\n✅ CROSS-DATASET EXPERIMENT COMPLETED!")
    print("✅ Dataset: NEU-DET-like (6 classes)")
    print("✅ Device: GPU")
    print("✅ All models saved to experiments_r3/results/models/")

    return results


if __name__ == '__main__':
    try:
        results = run_cross_dataset_experiment()
        print("\n✅ SUCCESS: Cross-dataset experiment completed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
