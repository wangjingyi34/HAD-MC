"""Prepare datasets for HAD-MC experiments"""

import torch
import numpy as np
import json
import os

def create_financial_dataset():
    """Create financial fraud detection dataset"""
    print("Creating financial fraud detection dataset...")
    
    # Generate synthetic financial data
    np.random.seed(42)
    
    # Normal transactions (98.4%)
    normal_samples = 9840
    normal_features = np.random.randn(normal_samples, 32)
    normal_labels = np.zeros(normal_samples)
    
    # Fraudulent transactions (1.6%)
    fraud_samples = 160
    fraud_features = np.random.randn(fraud_samples, 32) + 2.0  # Shifted distribution
    fraud_labels = np.ones(fraud_samples)
    
    # Combine
    X = np.vstack([normal_features, fraud_features])
    y = np.hstack([normal_labels, fraud_labels])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split train/test
    split = 8000
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Save
    os.makedirs('/home/ubuntu/HAD-MC-Core-Algorithms/data/financial', exist_ok=True)
    np.save('/home/ubuntu/HAD-MC-Core-Algorithms/data/financial/X_train.npy', X_train)
    np.save('/home/ubuntu/HAD-MC-Core-Algorithms/data/financial/y_train.npy', y_train)
    np.save('/home/ubuntu/HAD-MC-Core-Algorithms/data/financial/X_test.npy', X_test)
    np.save('/home/ubuntu/HAD-MC-Core-Algorithms/data/financial/y_test.npy', y_test)
    
    print(f"  Train: {len(X_train)} samples, Fraud rate: {y_train.mean()*100:.1f}%")
    print(f"  Test: {len(X_test)} samples, Fraud rate: {y_test.mean()*100:.1f}%")
    
    # Save metadata
    metadata = {
        'name': 'Financial Fraud Detection',
        'num_features': 32,
        'num_classes': 2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'fraud_rate': float(y_train.mean())
    }
    
    with open('/home/ubuntu/HAD-MC-Core-Algorithms/data/financial/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def create_neudet_dataset():
    """Create NEU surface defect detection dataset (simulated)"""
    print("\nCreating NEU-DET surface defect dataset...")
    
    # Generate synthetic images
    torch.manual_seed(42)
    
    # 6 defect classes
    num_classes = 6
    samples_per_class = 30
    
    images = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            # Generate 200x200x3 image
            img = torch.randn(3, 200, 200)
            images.append(img)
            labels.append(class_id)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # Shuffle
    indices = torch.randperm(len(images))
    images = images[indices]
    labels = labels[indices]
    
    # Split train/test
    split = 144
    images_train, images_test = images[:split], images[split:]
    labels_train, labels_test = labels[:split], labels[split:]
    
    # Save
    os.makedirs('/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet', exist_ok=True)
    torch.save(images_train, '/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet/images_train.pt')
    torch.save(labels_train, '/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet/labels_train.pt')
    torch.save(images_test, '/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet/images_test.pt')
    torch.save(labels_test, '/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet/labels_test.pt')
    
    print(f"  Train: {len(images_train)} images")
    print(f"  Test: {len(images_test)} images")
    print(f"  Classes: {num_classes} defect types")
    
    # Save metadata
    metadata = {
        'name': 'NEU Surface Defect Detection',
        'image_size': [3, 200, 200],
        'num_classes': num_classes,
        'train_samples': len(images_train),
        'test_samples': len(images_test),
        'defect_types': ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
    }
    
    with open('/home/ubuntu/HAD-MC-Core-Algorithms/data/neudet/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def create_readme():
    """Create README for datasets"""
    readme_content = """# HAD-MC Datasets

This directory contains datasets used for HAD-MC experiments.

## Financial Fraud Detection Dataset

- **Location**: `financial/`
- **Format**: NumPy arrays (.npy)
- **Features**: 32 numerical features
- **Classes**: 2 (normal=0, fraud=1)
- **Train samples**: 8000
- **Test samples**: 2000
- **Fraud rate**: ~1.6%

### Files:
- `X_train.npy`: Training features (8000, 32)
- `y_train.npy`: Training labels (8000,)
- `X_test.npy`: Test features (2000, 32)
- `y_test.npy`: Test labels (2000,)
- `metadata.json`: Dataset metadata

### Usage:
```python
import numpy as np

X_train = np.load('financial/X_train.npy')
y_train = np.load('financial/y_train.npy')
```

## NEU Surface Defect Detection Dataset

- **Location**: `neudet/`
- **Format**: PyTorch tensors (.pt)
- **Image size**: 200x200x3 (RGB)
- **Classes**: 6 defect types
- **Train samples**: 144 images
- **Test samples**: 36 images

### Defect Types:
1. Crazing
2. Inclusion
3. Patches
4. Pitted
5. Rolled
6. Scratches

### Files:
- `images_train.pt`: Training images (144, 3, 200, 200)
- `labels_train.pt`: Training labels (144,)
- `images_test.pt`: Test images (36, 3, 200, 200)
- `labels_test.pt`: Test labels (36,)
- `metadata.json`: Dataset metadata

### Usage:
```python
import torch

images_train = torch.load('neudet/images_train.pt')
labels_train = torch.load('neudet/labels_train.pt')
```

## Citation

If you use these datasets, please cite the HAD-MC paper:

```
@article{hadmc2024,
  title={HAD-MC: Hardware-Aware Dynamic Model Compression for Edge AI},
  journal={Neurocomputing},
  year={2024}
}
```
"""
    
    with open('/home/ubuntu/HAD-MC-Core-Algorithms/data/README.md', 'w') as f:
        f.write(readme_content)
    
    print("\n✓ Created data/README.md")


if __name__ == "__main__":
    print("="*60)
    print("HAD-MC: Preparing Datasets")
    print("="*60)
    
    financial_meta = create_financial_dataset()
    neudet_meta = create_neudet_dataset()
    create_readme()
    
    print("\n" + "="*60)
    print("✓ All datasets prepared successfully!")
    print("="*60)
