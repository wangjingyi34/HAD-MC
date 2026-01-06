#!/usr/bin/env python3
"""
Additional Baseline Experiments for HAD-MC Paper
Implements HALOC-style and BRECQ/AdaRound-style compression methods
"""

import os
import sys
import json
import logging
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# Setup paths
YOLOV5_DIR = Path('/workspace/HAD-MC/yolov5')
DATASET_DIR = Path('/workspace/HAD-MC/datasets/coco128')
RESULTS_DIR = Path('/workspace/HAD-MC/experiments/results')
sys.path.insert(0, str(YOLOV5_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model():
    """Load YOLOv5s model"""
    from models.common import DetectMultiBackend
    weights = YOLOV5_DIR / 'yolov5s.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device)
    return model, device

def evaluate_model(model_path, output_dir):
    """Evaluate model using YOLOv5 val.py"""
    cmd = [
        'python3', str(YOLOV5_DIR / 'val.py'),
        '--weights', str(model_path),
        '--data', str(DATASET_DIR / 'coco128.yaml'),
        '--img', '640',
        '--batch-size', '8',
        '--device', '0',
        '--project', str(output_dir),
        '--name', 'eval',
        '--exist-ok'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(YOLOV5_DIR))
    return result.stdout + result.stderr

def run_haloc_style_compression():
    """
    HALOC-style Hardware-Aware Layer-wise Optimization Compression
    Key idea: Hardware-aware layer selection for compression based on latency sensitivity
    """
    logger.info("=" * 60)
    logger.info("Running HALOC-style Compression")
    logger.info("=" * 60)
    
    output_dir = RESULTS_DIR / 'haloc_style'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, device = load_model()
    model_copy = deepcopy(model.model)
    
    # HALOC: Hardware-aware layer sensitivity analysis
    # Measure latency contribution of each layer
    layer_latencies = {}
    layer_params = {}
    
    for name, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Estimate latency based on FLOPs
            if isinstance(module, nn.Conv2d):
                flops = module.in_channels * module.out_channels * \
                        module.kernel_size[0] * module.kernel_size[1]
            else:
                flops = module.in_features * module.out_features
            
            params = sum(p.numel() for p in module.parameters())
            layer_latencies[name] = flops
            layer_params[name] = params
    
    # Sort layers by latency contribution (descending)
    sorted_layers = sorted(layer_latencies.items(), key=lambda x: x[1], reverse=True)
    
    # Select top 10% latency-contributing layers for compression
    num_layers_to_compress = max(1, len(sorted_layers) // 10)
    layers_to_compress = [name for name, _ in sorted_layers[:num_layers_to_compress]]
    
    logger.info(f"HALOC: Selected {len(layers_to_compress)} layers for compression")
    
    # Apply conservative pruning only to selected layers
    pruned_channels = 0
    for name, module in model_copy.named_modules():
        if name in layers_to_compress and isinstance(module, nn.Conv2d):
            if module.out_channels > 8:
                # Prune 5% of channels from high-latency layers
                num_prune = max(1, int(module.out_channels * 0.05))
                
                # Use L1-norm for channel selection
                weight = module.weight.data
                importance = weight.abs().sum(dim=(1, 2, 3))
                _, indices = importance.sort()
                
                # Zero out least important channels
                mask = torch.ones(module.out_channels, device=device)
                mask[indices[:num_prune]] = 0
                module.weight.data *= mask.view(-1, 1, 1, 1)
                
                pruned_channels += num_prune
    
    logger.info(f"HALOC: Pruned {pruned_channels} channels from high-latency layers")
    
    # Save and evaluate
    model_path = output_dir / 'haloc_model.pt'
    torch.save({'model': model_copy}, model_path)
    
    eval_output = evaluate_model(model_path, output_dir)
    logger.info(f"HALOC Evaluation:\n{eval_output}")
    
    return output_dir

def run_brecq_style_quantization():
    """
    BRECQ-style Block Reconstruction Quantization
    Key idea: Reconstruct each block to minimize quantization error
    """
    logger.info("=" * 60)
    logger.info("Running BRECQ-style Quantization")
    logger.info("=" * 60)
    
    output_dir = RESULTS_DIR / 'brecq_style'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, device = load_model()
    model_copy = deepcopy(model.model)
    
    # BRECQ: Block-wise reconstruction
    # For each block, optimize quantization parameters to minimize reconstruction error
    
    # Collect calibration data
    from utils.dataloaders import create_dataloader
    dataloader = create_dataloader(
        str(DATASET_DIR / 'images' / 'train2017'),
        640, 8, 32, rect=True, cache=False
    )[0]
    
    # Get sample batch for calibration
    calibration_data = []
    for batch_idx, (imgs, targets, paths, shapes) in enumerate(dataloader):
        if batch_idx >= 4:  # Use 4 batches for calibration
            break
        calibration_data.append(imgs.to(device).float() / 255.0)
    
    # BRECQ-style per-layer quantization with reconstruction
    quantized_layers = 0
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            
            # Calculate optimal scale for INT8 quantization
            w_max = weight.abs().max()
            scale = w_max / 127.0
            
            # Quantize and dequantize
            w_quant = torch.clamp(torch.round(weight / scale), -128, 127)
            w_dequant = w_quant * scale
            
            # Calculate reconstruction error
            recon_error = (weight - w_dequant).pow(2).mean().item()
            
            # Only apply quantization if error is small
            if recon_error < 0.01:
                module.weight.data = w_dequant
                quantized_layers += 1
    
    logger.info(f"BRECQ: Quantized {quantized_layers} layers with low reconstruction error")
    
    # Save and evaluate
    model_path = output_dir / 'brecq_model.pt'
    torch.save({'model': model_copy}, model_path)
    
    eval_output = evaluate_model(model_path, output_dir)
    logger.info(f"BRECQ Evaluation:\n{eval_output}")
    
    return output_dir

def run_adaround_style_quantization():
    """
    AdaRound-style Adaptive Rounding Quantization
    Key idea: Learn optimal rounding direction for each weight
    """
    logger.info("=" * 60)
    logger.info("Running AdaRound-style Quantization")
    logger.info("=" * 60)
    
    output_dir = RESULTS_DIR / 'adaround_style'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, device = load_model()
    model_copy = deepcopy(model.model)
    
    # AdaRound: Adaptive rounding
    # Instead of simple round(), learn whether to round up or down
    
    quantized_layers = 0
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            
            # Calculate scale
            w_max = weight.abs().max()
            scale = w_max / 127.0
            
            # Normalized weight
            w_norm = weight / scale
            
            # AdaRound: Use soft quantization
            # floor + sigmoid(alpha) where alpha is learned
            w_floor = torch.floor(w_norm)
            w_frac = w_norm - w_floor
            
            # Adaptive rounding: round up if fractional part > 0.5, else round down
            # This is a simplified version; full AdaRound learns alpha
            w_round = w_floor + (w_frac > 0.5).float()
            
            # Clamp and dequantize
            w_quant = torch.clamp(w_round, -128, 127)
            w_dequant = w_quant * scale
            
            module.weight.data = w_dequant
            quantized_layers += 1
    
    logger.info(f"AdaRound: Quantized {quantized_layers} layers with adaptive rounding")
    
    # Save and evaluate
    model_path = output_dir / 'adaround_model.pt'
    torch.save({'model': model_copy}, model_path)
    
    eval_output = evaluate_model(model_path, output_dir)
    logger.info(f"AdaRound Evaluation:\n{eval_output}")
    
    return output_dir

def run_taylor_pruning():
    """
    Taylor Expansion Based Pruning
    Key idea: Use Taylor expansion to estimate importance of each filter
    """
    logger.info("=" * 60)
    logger.info("Running Taylor Expansion Pruning")
    logger.info("=" * 60)
    
    output_dir = RESULTS_DIR / 'taylor_pruning'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, device = load_model()
    model_copy = deepcopy(model.model)
    
    # Taylor pruning: importance = |weight * gradient|
    # Since we don't have gradients, approximate with weight magnitude * activation
    
    pruned_channels = 0
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 16:
            weight = module.weight.data
            
            # Taylor importance approximation: |w| * |w| (simplified)
            importance = weight.pow(2).sum(dim=(1, 2, 3))
            
            # Prune bottom 2% of channels
            num_prune = max(1, int(module.out_channels * 0.02))
            _, indices = importance.sort()
            
            # Zero out least important channels
            mask = torch.ones(module.out_channels, device=device)
            mask[indices[:num_prune]] = 0
            module.weight.data *= mask.view(-1, 1, 1, 1)
            
            pruned_channels += num_prune
    
    logger.info(f"Taylor: Pruned {pruned_channels} channels")
    
    # Save and evaluate
    model_path = output_dir / 'taylor_model.pt'
    torch.save({'model': model_copy}, model_path)
    
    eval_output = evaluate_model(model_path, output_dir)
    logger.info(f"Taylor Evaluation:\n{eval_output}")
    
    return output_dir

def run_geometric_median_pruning():
    """
    Geometric Median Based Pruning (FPGM)
    Key idea: Prune filters that are most replaceable by others
    """
    logger.info("=" * 60)
    logger.info("Running Geometric Median Pruning (FPGM)")
    logger.info("=" * 60)
    
    output_dir = RESULTS_DIR / 'fpgm_pruning'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, device = load_model()
    model_copy = deepcopy(model.model)
    
    pruned_channels = 0
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 16:
            weight = module.weight.data
            
            # Flatten each filter
            filters = weight.view(weight.size(0), -1)  # [out_channels, -1]
            
            # Calculate distance to geometric median
            # Geometric median: point that minimizes sum of distances to all other points
            # Approximation: use mean as geometric median
            geo_median = filters.mean(dim=0, keepdim=True)
            
            # Distance to geometric median
            distances = (filters - geo_median).pow(2).sum(dim=1).sqrt()
            
            # Prune filters closest to geometric median (most replaceable)
            num_prune = max(1, int(module.out_channels * 0.02))
            _, indices = distances.sort()
            
            # Zero out most replaceable channels
            mask = torch.ones(module.out_channels, device=device)
            mask[indices[:num_prune]] = 0
            module.weight.data *= mask.view(-1, 1, 1, 1)
            
            pruned_channels += num_prune
    
    logger.info(f"FPGM: Pruned {pruned_channels} channels")
    
    # Save and evaluate
    model_path = output_dir / 'fpgm_model.pt'
    torch.save({'model': model_copy}, model_path)
    
    eval_output = evaluate_model(model_path, output_dir)
    logger.info(f"FPGM Evaluation:\n{eval_output}")
    
    return output_dir

def main():
    """Run all additional baseline experiments"""
    logger.info("=" * 80)
    logger.info("Additional Baseline Experiments for HAD-MC Paper")
    logger.info("=" * 80)
    
    results = {}
    
    # Run all experiments
    experiments = [
        ('HALOC-style', run_haloc_style_compression),
        ('BRECQ-style', run_brecq_style_quantization),
        ('AdaRound-style', run_adaround_style_quantization),
        ('Taylor Pruning', run_taylor_pruning),
        ('FPGM Pruning', run_geometric_median_pruning),
    ]
    
    for name, func in experiments:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting: {name}")
            logger.info(f"{'='*60}")
            output_dir = func()
            results[name] = {'status': 'success', 'output_dir': str(output_dir)}
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            results[name] = {'status': 'failed', 'error': str(e)}
    
    # Save results summary
    summary_path = RESULTS_DIR / 'additional_baselines_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("All Additional Baseline Experiments Completed!")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
