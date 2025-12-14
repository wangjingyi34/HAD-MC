"""NEU-DET Surface Defect Detection Experiment with HAD-MC"""

import sys
from hadmc.device_manager import DeviceManager
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import json
import logging

from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.pruning import GradientSensitivityPruner
from hadmc.distillation import FeatureAlignedDistiller
from hadmc.fusion import OperatorFuser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ResNetBlock(nn.Module):
    """ResNet-style block for defect detection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class DefectDetectionModel(nn.Module):
    """ResNet-18 style model for defect detection"""
    def __init__(self, num_classes=6):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def load_neudet_data():
    """Load NEU-DET dataset"""
    logging.info("Loading NEU-DET dataset...")
    
    data_dir = '/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet'
    
    images_train = torch.load(f'{data_dir}/images_train.pt')
    labels_train = torch.load(f'{data_dir}/labels_train.pt')
    images_test = torch.load(f'{data_dir}/images_test.pt')
    labels_test = torch.load(f'{data_dir}/labels_test.pt')
    
    # Normalize
    images_train = (images_train - images_train.mean()) / (images_train.std() + 1e-8)
    images_test = (images_test - images_test.mean()) / (images_test.std() + 1e-8)
    
    train_dataset = TensorDataset(images_train, labels_train)
    test_dataset = TensorDataset(images_test, labels_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    logging.info(f"  Train: {len(train_dataset)} images")
    logging.info(f"  Test: {len(test_dataset)} images")
    
    return train_loader, test_loader


def train_model(model, train_loader, epochs=10):
    """Train the model"""
    logging.info(f"Training model for {epochs} epochs...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 2 == 0:
            logging.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    logging.info("✓ Training complete")


def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def measure_latency(model, input_size=(1, 3, 200, 200), num_runs=100):
    """Measure inference latency"""
    model.eval()
    dummy_input = torch.randn(input_size)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.time()
    
    latency = (end_time - start_time) / num_runs * 1000  # ms
    return latency


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def run_neudet_experiment():
    """Run complete NEU-DET experiment with HAD-MC"""
    
    logging.info("="*80)
    logging.info("NEU-DET Surface Defect Detection with HAD-MC")
    logging.info("="*80)
    
    # Load data
    train_loader, test_loader = load_neudet_data()
    
    # Create and train baseline model
    logging.info("\n" + "="*80)
    logging.info("Training Baseline FP32 Model")
    logging.info("="*80)
    
    baseline_model = DefectDetectionModel(num_classes=6)
    train_model(baseline_model, train_loader, epochs=10)
    
    baseline_acc = evaluate_model(baseline_model, test_loader)
    baseline_latency = measure_latency(baseline_model)
    baseline_size = get_model_size(baseline_model)
    
    logging.info(f"\nBaseline FP32:")
    logging.info(f"  Accuracy: {baseline_acc:.2f}%")
    logging.info(f"  Latency: {baseline_latency:.2f} ms")
    logging.info(f"  Size: {baseline_size:.2f} MB")
    
    # Apply HAD-MC pipeline
    logging.info("\n" + "="*80)
    logging.info("Applying HAD-MC Pipeline")
    logging.info("="*80)
    
    # Create compressed model
    compressed_model = DefectDetectionModel(num_classes=6)
    compressed_model.load_state_dict(baseline_model.state_dict())
    
    # Algorithm 1: Precision Allocation
    logging.info("\n[Algorithm 1] Layer-wise Precision Allocation")
    allocator = LayerwisePrecisionAllocator(compressed_model, train_loader)
    
    # Calculate sensitivity and allocate precision
    compressed_model = allocator.run(target_bits=6)
    
    # Algorithm 2: Pruning
    logging.info("\n[Algorithm 2] Gradient Sensitivity-Guided Pruning")
    pruner = GradientSensitivityPruner(compressed_model, train_loader, flops_target=0.5)
    
    # Run pruning
    compressed_model = pruner.run()
    
    # Fine-tune after pruning
    logging.info("  Fine-tuning after pruning...")
    train_model(compressed_model, train_loader, epochs=5)
    
    # Algorithm 3: Knowledge Distillation
    logging.info("\n[Algorithm 3] Feature-Aligned Knowledge Distillation")
    
    student_model = DefectDetectionModel(num_classes=6)
    distiller = FeatureAlignedDistiller(baseline_model, student_model)
    
    # Distill
    student_model = distiller.run(train_loader, epochs=3, lr=0.001)
    
    compressed_model = student_model
    
    # Algorithm 4: Operator Fusion
    logging.info("\n[Algorithm 4] Operator Fusion")
    fusion = OperatorFuser(compressed_model)
    compressed_model = fusion.run()
    
    # Evaluate compressed model
    logging.info("\n" + "="*80)
    logging.info("Evaluating Compressed Model")
    logging.info("="*80)
    
    compressed_acc = evaluate_model(compressed_model, test_loader)
    compressed_latency = measure_latency(compressed_model)
    compressed_size = get_model_size(compressed_model)
    
    logging.info(f"\nHAD-MC Compressed:")
    logging.info(f"  Accuracy: {compressed_acc:.2f}%")
    logging.info(f"  Latency: {compressed_latency:.2f} ms")
    logging.info(f"  Size: {compressed_size:.2f} MB")
    
    # Calculate improvements
    size_reduction = (1 - compressed_size / baseline_size) * 100
    latency_reduction = (1 - compressed_latency / baseline_latency) * 100
    acc_change = compressed_acc - baseline_acc
    
    logging.info("\n" + "="*80)
    logging.info("FINAL RESULTS")
    logging.info("="*80)
    logging.info(f"\nModel Size:")
    logging.info(f"  FP32:      {baseline_size:.2f} MB")
    logging.info(f"  HAD-MC:    {compressed_size:.2f} MB")
    logging.info(f"  Reduction: {size_reduction:.1f}%")
    
    logging.info(f"\nInference Latency:")
    logging.info(f"  FP32:      {baseline_latency:.2f} ms")
    logging.info(f"  HAD-MC:    {compressed_latency:.2f} ms")
    logging.info(f"  Reduction: {latency_reduction:.1f}%")
    
    logging.info(f"\nAccuracy:")
    logging.info(f"  FP32:      {baseline_acc:.2f}%")
    logging.info(f"  HAD-MC:    {compressed_acc:.2f}%")
    logging.info(f"  Change:    {acc_change:+.2f}%")
    
    # Save results
    results = {
        'baseline': {
            'accuracy': baseline_acc,
            'latency_ms': baseline_latency,
            'size_mb': baseline_size
        },
        'hadmc': {
            'accuracy': compressed_acc,
            'latency_ms': compressed_latency,
            'size_mb': compressed_size
        },
        'improvements': {
            'size_reduction_pct': size_reduction,
            'latency_reduction_pct': latency_reduction,
            'accuracy_change_pct': acc_change
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/neudet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("\n✓ Results saved to results/neudet_results.json")
    logging.info("="*80)
    
    return results


if __name__ == "__main__":
    run_neudet_experiment()
