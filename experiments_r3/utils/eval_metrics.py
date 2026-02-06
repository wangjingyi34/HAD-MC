"""
Comprehensive Evaluation Metrics for HAD-MC 2.0

This module implements all evaluation metrics used in the paper:
- Accuracy metrics (mAP, mAP@0.5, mAP@0.5:0.95)
- Efficiency metrics (latency, energy, throughput)
- FPR metric (critical for reviewer #2)
- Model size metrics (parameters, FLOPs, memory)

All metrics report mean and standard deviation for statistical significance.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) for a single class.

    Args:
        recalls: Recall values array
        precisions: Precision values array

    Returns:
        ap: Average Precision value

    Reference:
        The 11-point interpolation method from PASCAL VOC.
        More precise version: Calculate area under precision-recall curve.
    """
    # Ensure arrays have same length
    assert len(recalls) == len(precisions), "recalls and precisions must have same length"

    # Add boundary points
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate area under curve using trapezoidal rule
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    if len(indices) == 0:
        return 0.0

    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return float(ap)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        iou: IoU value in [0, 1]
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate mAP (Mean Average Precision).

    Args:
        predictions: List of predictions, each containing:
            - 'boxes': list of [x1, y1, x2, y2]
            - 'scores': list of confidence scores
            - 'labels': list of class labels
        ground_truths: List of ground truth annotations, each containing:
            - 'boxes': list of [x1, y1, x2, y2]
            - 'labels': list of class labels
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes (auto-detected if None)

    Returns:
        map_value: mAP value
        ap_per_class: Dictionary mapping class_id to AP value
    """
    # Collect all classes
    all_classes = set()
    for pred in predictions:
        all_classes.update(pred['labels'])
    for gt in ground_truths:
        all_classes.update(gt['labels'])

    if num_classes is None:
        num_classes = len(all_classes)

    ap_per_class = {}

    for class_id in all_classes:
        # Get predictions and ground truths for this class
        class_preds = []
        class_gts = []

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Get predictions for this class
            for j, label in enumerate(pred['labels']):
                if label == class_id:
                    class_preds.append({
                        'image_id': i,
                        'box': pred['boxes'][j],
                        'score': pred['scores'][j]
                    })

            # Get ground truths for this class
            for j, label in enumerate(gt['labels']):
                if label == class_id:
                    class_gts.append({
                        'image_id': i,
                        'box': gt['boxes'][j],
                        'matched': False
                    })

        if len(class_gts) == 0:
            ap_per_class[class_id] = 0.0
            continue

        # Sort predictions by score (descending)
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)

        # Calculate TP and FP
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(class_gts):
                if gt['matched'] or gt['image_id'] != pred['image_id']:
                    continue

                iou = calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                class_gts[best_gt_idx]['matched'] = True
            else:
                fp[i] = 1

        # Calculate cumulative TP and FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        # Calculate precision and recall
        precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
        recalls = cum_tp / (len(class_gts) + 1e-10)

        # Calculate AP
        ap_per_class[class_id] = calculate_ap(recalls, precisions)

    # Calculate mAP
    if len(ap_per_class) == 0:
        map_value = 0.0
    else:
        map_value = np.mean(list(ap_per_class.values()))

    return float(map_value), ap_per_class


def calculate_map_coco(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[np.ndarray] = None
) -> float:
    """
    Calculate COCO-style mAP@0.5:0.95.

    Args:
        predictions: List of predictions
        ground_truths: List of ground truth annotations
        iou_thresholds: Array of IoU thresholds (default: 0.5 to 0.95, step 0.05)

    Returns:
        map_coco: COCO mAP value
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    maps = []
    for iou_thresh in iou_thresholds:
        map_value, _ = calculate_map(predictions, ground_truths, iou_thresh)
        maps.append(map_value)

    return float(np.mean(maps))


def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 50,
    num_runs: int = 100,
    device: Optional[str] = None
) -> Tuple[float, float]:
    """
    Measure model inference latency with proper warmup and synchronization.

    Args:
        model: PyTorch model to measure
        input_tensor: Input tensor
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        device: Device to run on (auto-detected if None)

    Returns:
        latency_mean: Mean latency in milliseconds
        latency_std: Standard deviation of latency in milliseconds
    """
    model.eval()

    # Detect device if not specified
    if device is None:
        if hasattr(input_tensor, 'device'):
            device = str(input_tensor.device)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Synchronize CUDA operations
    if 'cuda' in device:
        torch.cuda.synchronize()

    # Measure runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)

            if 'cuda' in device:
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return float(np.mean(latencies)), float(np.std(latencies))


def measure_energy(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    device: Optional[str] = None
) -> Tuple[float, float]:
    """
    Measure model inference energy consumption.

    Args:
        model: PyTorch model to measure
        input_tensor: Input tensor
        num_runs: Number of measurement runs
        device: Device to run on (auto-detected if None)

    Returns:
        energy_mean: Mean energy in Joules
        energy_std: Standard deviation of energy in Joules

    Note:
        Requires NVIDIA GPU with NVML support.
        Returns (0, 0) if NVML is not available.
    """
    model.eval()

    # Try to import NVML
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_nvml = True
    except:
        has_nvml = False
        print("Warning: NVML not available, energy measurement skipped")
        return 0.0, 0.0

    # Detect device
    if device is None:
        if hasattr(input_tensor, 'device'):
            device = str(input_tensor.device)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'cuda' not in device or not has_nvml:
        return 0.0, 0.0

    energies = []
    with torch.no_grad():
        for _ in range(num_runs):
            # Get initial power (in watts)
            power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            start_time = time.perf_counter()
            _ = model(input_tensor)

            if 'cuda' in device:
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Get final power (in watts)
            power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            # Calculate energy (Joules = Watts × Seconds)
            avg_power = (power_start + power_end) / 2.0
            duration = end_time - start_time
            energy = avg_power * duration
            energies.append(energy)

    pynvml.nvmlShutdown()

    return float(np.mean(energies)), float(np.std(energies))


def measure_throughput(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    batch_sizes: List[int] = [1, 8, 16, 32],
    num_warmup: int = 50,
    num_runs: int = 100
) -> Dict[int, Tuple[float, float]]:
    """
    Measure model throughput (FPS) at different batch sizes.

    Args:
        model: PyTorch model to measure
        input_tensor: Input tensor
        batch_sizes: List of batch sizes to test
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs

    Returns:
        Dictionary mapping batch_size to (throughput_mean, throughput_std)
    """
    results = {}

    for batch_size in batch_sizes:
        # Create batch input
        batch_input = input_tensor.repeat(batch_size, *([1] * (len(input_tensor.shape) - 1)))

        # Measure latency
        latency_mean, _ = measure_latency(model, batch_input, num_warmup, num_runs)

        # Calculate throughput (FPS)
        throughput = batch_size / (latency_mean / 1000.0)  # FPS = batch_size / (latency in seconds)

        results[batch_size] = (throughput, throughput)

    return results


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size metrics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing:
            - num_params: Number of parameters
            - param_size_mb: Parameter size in MB
            - flops: Number of FLOPs (estimated)
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate parameter size in MB (assuming float32)
    param_size_mb = num_params * 4.0 / (1024.0 * 1024.0)

    # Estimate FLOPs (rough estimate for CNNs)
    flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Conv2d FLOPs: kernel_h * kernel_w * in_channels * out_channels * output_h * output_w
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            flops += kernel_size * module.in_channels * module.out_channels * 224 * 224  # Assume 224x224 input
        elif isinstance(module, torch.nn.Linear):
            # Linear FLOPs: in_features * out_features
            flops += module.in_features * module.out_features

    return {
        'num_params': int(num_params),
        'num_trainable_params': int(num_trainable_params),
        'param_size_mb': float(param_size_mb),
        'flops': int(flops)
    }


def calculate_frame_level_fpr(
    model: torch.nn.Module,
    predictions: List[Dict],
    ground_truths: List[Dict],
    confidence_threshold: float = 0.5
) -> Tuple[float, Dict[str, int]]:
    """
    Calculate frame-level False Positive Rate (FPR).

    CRITICAL DEFINITION FOR REVIEWER #2:
    =====================================
    Frame-level FPR is defined as:

        FPR = FP / (FP + TN) = FP / N_negative

    Where:
        - FP (False Positive): Number of frames WITHOUT ground truth that have
          at least one detection above the confidence threshold
        - TN (True Negative): Number of frames WITHOUT ground truth that have
          NO detections above the confidence threshold
        - N_negative: Total number of negative frames (frames without ground truth)

    Frame-level classification rule:
        - If a frame has at least one detection with confidence > threshold,
          it is classified as POSITIVE (fire/smoke detected)
        - If a frame has NO detections with confidence > threshold,
          it is classified as NEGATIVE (normal frame)

    Args:
        model: PyTorch model
        predictions: List of predictions, each containing:
            - 'boxes': list of bounding boxes
            - 'scores': list of confidence scores
        ground_truths: List of ground truth annotations, each containing:
            - 'has_positive': boolean indicating if frame contains targets
            - 'boxes': list of bounding boxes (if has_positive is True)
        confidence_threshold: Confidence threshold for positive classification

    Returns:
        fpr: Frame-level False Positive Rate
        details: Dictionary containing detailed statistics:
            - TP: True Positives
            - FP: False Positives
            - TN: True Negatives
            - FN: False Negatives
            - FPR: False Positive Rate
            - TPR: True Positive Rate (Recall)
            - Precision: Precision value
    """
    # Initialize counters
    tp = 0  # True Positive
    fp = 0  # False Positive
    tn = 0  # True Negative
    fn = 0  # False Negative

    for pred, gt in zip(predictions, ground_truths):
        # Determine if ground truth is positive (has targets)
        is_positive_frame = gt.get('has_positive', False)

        # Determine if model detects anything
        # Frame is classified as POSITIVE if at least one detection has confidence > threshold
        has_detection = any(score > confidence_threshold for score in pred['scores'])

        if is_positive_frame:
            # Positive frame
            if has_detection:
                tp += 1  # Correctly detected
            else:
                fn += 1  # Missed (false negative)
        else:
            # Negative frame (no targets in ground truth)
            if has_detection:
                fp += 1  # Incorrectly detected (false positive)
            else:
                tn += 1  # Correctly identified as negative (true negative)

    # Calculate FPR
    n_negative = fp + tn
    fpr = fp / n_negative if n_negative > 0 else 0.0

    # Calculate additional metrics
    n_positive = tp + fn
    tpr = tp / n_positive if n_positive > 0 else 0.0  # Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    details = {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'FPR': float(fpr),
        'TPR': float(tpr),  # Recall
        'Precision': float(precision),
        'N_positive': int(n_positive),
        'N_negative': int(n_negative)
    }

    return float(fpr), details


def calculate_fpr_at_fixed_recall(
    model: torch.nn.Module,
    predictions: List[Dict],
    ground_truths: List[Dict],
    target_recall: float = 0.95,
    threshold_range: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict]:
    """
    Calculate FPR at a fixed recall level (e.g., 95% recall).

    This is important for fair comparison across different methods.
    We find the confidence threshold that achieves the target recall,
    then report the FPR at that threshold.

    Args:
        model: PyTorch model
        predictions: List of predictions
        ground_truths: List of ground truth annotations
        target_recall: Target recall level (e.g., 0.95 for 95%)
        threshold_range: Range of thresholds to search (default: 0.01 to 1.0, step 0.01)

    Returns:
        fpr: FPR at target recall
        best_threshold: Confidence threshold that achieves target recall
        details: Full evaluation details at best threshold
    """
    if threshold_range is None:
        threshold_range = np.arange(0.01, 1.0, 0.01)

    best_fpr = 1.0
    best_threshold = 0.5
    best_details = None

    for threshold in threshold_range:
        # Apply threshold to predictions
        filtered_predictions = []
        for pred in predictions:
            # Keep only detections above threshold
            mask = np.array(pred['scores']) > threshold
            filtered_pred = {
                'boxes': [pred['boxes'][i] for i in range(len(mask)) if mask[i]],
                'scores': [pred['scores'][i] for i in range(len(mask)) if mask[i]]
            }
            filtered_predictions.append(filtered_pred)

        # Calculate frame-level metrics
        fpr, details = calculate_frame_level_fpr(
            model, filtered_predictions, ground_truths, threshold
        )

        # Check if we achieve target recall with lower FPR
        recall = details['TPR']
        if recall >= target_recall and fpr < best_fpr:
            best_fpr = fpr
            best_threshold = threshold
            best_details = details

    if best_details is None:
        # If we couldn't achieve target recall, return best attempt
        best_details = details

    return float(best_fpr), float(best_threshold), best_details


class MetricTracker:
    """
    Track metrics across multiple runs for statistical analysis.
    """

    def __init__(self):
        self.metrics = defaultdict(list)

    def add(self, run_id: str, metrics: Dict[str, float]):
        """
        Add metrics for a specific run.

        Args:
            run_id: Identifier for the run
            metrics: Dictionary of metric name to value
        """
        for key, value in metrics.items():
            self.metrics[key].append(value)

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary containing:
                - mean: Mean value
                - std: Standard deviation
                - min: Minimum value
                - max: Maximum value
                - median: Median value
                - n: Number of samples
        """
        values = self.metrics.get(metric_name, [])
        if len(values) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'n': 0
            }

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'n': len(values)
        }

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tracked metrics.

        Returns:
            Dictionary mapping metric name to statistics
        """
        result = {}
        for metric_name in self.metrics.keys():
            result[metric_name] = self.get_statistics(metric_name)
        return result

    def to_dict(self) -> Dict[str, List[float]]:
        """Return all tracked metrics as a dictionary."""
        return dict(self.metrics)
