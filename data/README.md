# HAD-MC Datasets

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
