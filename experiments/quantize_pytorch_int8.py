#!/usr/bin/env python3
"""PyTorch INT8动态量化"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# 添加YOLOv5路径
sys.path.append('/workspace/HAD-MC/yolov5')

def quantize_model(model_path, output_path):
    """量化模型"""
    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu')
    model = ckpt['model'].float()
    
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
    
    print(f"Saving quantized model to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': quantized_model}, output_path)
    
    # 检查模型大小
    original_size = Path(model_path).stat().st_size / 1024 / 1024
    quantized_size = output_path.stat().st_size / 1024 / 1024
    compression_ratio = original_size / quantized_size
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return str(output_path)

if __name__ == '__main__':
    model_path = '/workspace/HAD-MC/experiments/results/fp32_baseline/train/weights/best.pt'
    output_path = Path('/workspace/HAD-MC/experiments/results/phase1_comprehensive/pytorch_int8/model.pt')
    
    quantized_path = quantize_model(model_path, output_path)
    print(f"\n✓ Quantization completed: {quantized_path}")
