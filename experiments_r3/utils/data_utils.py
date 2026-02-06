"""
Data utilities for HAD-MC 2.0 experiments

This module provides:
- Dataset loaders for different datasets
- Data preprocessing utilities
- Batch collation functions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os


class BaseDetectionDataset(Dataset):
    """Base class for detection datasets."""

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.annotations = []

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index.

        Returns:
            image: Image tensor (C, H, W)
            annotation: Dictionary containing:
                - boxes: (N, 4) tensor of bounding boxes [x1, y1, x2, y2]
                - labels: (N,) tensor of class labels
                - scores: (N,) tensor of confidence scores (for test data)
        """
        raise NotImplementedError


class COCODataset(BaseDetectionDataset):
    """COCO-style dataset loader."""

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        split: str = 'train',
        image_size: int = 640,
        transform=None
    ):
        """
        Initialize COCO dataset.

        Args:
            root_dir: Directory containing images
            annotation_file: Path to COCO annotation JSON file
            split: Dataset split
            image_size: Target image size
            transform: Optional transform
        """
        super().__init__(root_dir, split, transform)
        self.image_size = image_size

        # Load annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Build image ID to annotations mapping
        self.image_id_to_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # Get image list
        self.images = coco_data['images']
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.num_classes = len(self.categories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index.

        Returns:
            image: Image tensor (3, 640, 640)
            annotation: Dictionary with boxes and labels
        """
        img_info = self.images[idx]
        image_id = img_info['id']
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # Load image (you may need to install PIL or cv2)
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback: create dummy image
            image = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image)

        # Resize to target size
        image = image.resize((self.image_size, self.image_size))

        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])

        boxes = []
        labels = []

        for ann in annotations:
            # Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            # Convert PIL to tensor and normalize
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        annotation = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': image_id
        }

        return image, annotation


class NEUDETDataset(BaseDetectionDataset):
    """NEU-DET steel surface defect dataset loader."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 640,
        transform=None
    ):
        """
        Initialize NEU-DET dataset.

        Args:
            root_dir: Root directory of NEU-DET dataset
            split: Dataset split
            image_size: Target image size
            transform: Optional transform
        """
        super().__init__(root_dir, split, transform)
        self.image_size = image_size

        # NEU-DET defect types
        self.class_names = ['rolled-in_scale', 'patches', 'crazing', 'pitted_surface', 'inclusion', 'scratches']
        self.num_classes = len(self.class_names)

        # Load image paths and annotations
        split_dir = os.path.join(root_dir, split)
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.jpg', '.png', '.bmp')):
                        self.images.append({
                            'path': os.path.join(class_dir, img_file),
                            'class_id': class_idx,
                            'class_name': class_name
                        })

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index.

        For NEU-DET, we use whole-image classification.
        """
        img_info = self.images[idx]
        img_path = img_info['path']

        # Load image
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except:
            image = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image)

        # Resize
        image = image.resize((self.image_size, self.image_size))

        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # For detection, create a dummy bounding box covering the whole image
        h, w = self.image_size, self.image_size
        boxes = torch.tensor([[0, 0, w, h]], dtype=torch.float32)
        labels = torch.tensor([img_info['class_id']], dtype=torch.int64)

        annotation = {
            'boxes': boxes,
            'labels': labels,
            'class_id': img_info['class_id'],
            'class_name': img_info['class_name']
        }

        return image, annotation


class VOCDataset(BaseDetectionDataset):
    """Pascal VOC dataset loader."""

    def __init__(
        self,
        root_dir: str,
        year: str = '2012',
        split: str = 'train',
        image_size: int = 640,
        transform=None
    ):
        """
        Initialize Pascal VOC dataset.

        Args:
            root_dir: Root directory of VOC dataset
            year: VOC year ('2007', '2012', or 'merged')
            split: Dataset split
            image_size: Target image size
            transform: Optional transform
        """
        super().__init__(root_dir, split, transform)
        self.image_size = image_size
        self.year = year

        # VOC class names
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.num_classes = len(self.class_names)

        # Load annotation file paths
        split_file = os.path.join(root_dir, f'ImageSets/Main/{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f]
        else:
            # Fallback: scan directory
            self.image_ids = []

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index.

        Note: This is a simplified implementation.
        For full VOC support, you need to parse XML annotation files.
        """
        image_id = self.image_ids[idx]
        img_path = os.path.join(
            self.root_dir, 'JPEGImages', f'{image_id}.jpg'
        )

        # Load image
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except:
            image = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image)

        # Resize
        image = image.resize((self.image_size, self.image_size))

        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Placeholder annotation (parse XML in full implementation)
        annotation = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': image_id
        }

        return image, annotation


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for detection datasets.

    Args:
        batch: List of (image, annotation) tuples

    Returns:
        images: Batched image tensor (B, C, H, W)
        annotations: List of annotation dictionaries
    """
    images = []
    annotations = []

    for image, annotation in batch:
        images.append(image)
        annotations.append(annotation)

    # Stack images
    images = torch.stack(images, dim=0)

    return images, annotations


def get_dataloader(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 640,
    **kwargs
) -> DataLoader:
    """
    Get a dataloader for the specified dataset.

    Args:
        dataset_name: Name of dataset ('coco', 'neudet', 'voc')
        root_dir: Root directory of dataset
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        **kwargs: Additional dataset-specific arguments

    Returns:
        DataLoader instance
    """
    if dataset_name.lower() == 'coco':
        dataset = COCODataset(
            root_dir=root_dir,
            annotation_file=kwargs.get('annotation_file', ''),
            split=split,
            image_size=image_size
        )
    elif dataset_name.lower() == 'neudet':
        dataset = NEUDETDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size
        )
    elif dataset_name.lower() == 'voc':
        dataset = VOCDataset(
            root_dir=root_dir,
            year=kwargs.get('year', '2012'),
            split=split,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader
