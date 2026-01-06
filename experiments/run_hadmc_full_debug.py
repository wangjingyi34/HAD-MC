#!/usr/bin/env python3
"""HAD-MC完整流程：剪枝 + 量化 + 知识蒸馏（带详细调试）"""

import torch
import sys
from pathlib import Path
import traceback

# 添加路径
sys.path.append('/workspace/HAD-MC/yolov5')
sys.path.append('/workspace/HAD-MC')

from hadmc.pruning import GradientSensitivityPruner
from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.distillation import FeatureAlignedDistiller
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_yaml

def run_hadmc_full():
    """运行完整HAD-MC流程"""
    print("=" * 80)
    print("HAD-MC Full Pipeline (Debug Mode)")
    print("=" * 80)
    
    device = torch.device('cuda:0')
    
    # 加载FP32模型
    fp32_model_path = '/workspace/HAD-MC/experiments/results/fp32_baseline/train/weights/best.pt'
    print(f"\n[1/4] Loading FP32 model from {fp32_model_path}...")
    ckpt = torch.load(fp32_model_path, map_location=device)
    teacher_model = ckpt['model'].float().to(device)
    teacher_model.eval()
    
    # 创建学生模型（复制教师模型）
    student_model = ckpt['model'].float().to(device)
    
    # 准备数据
    data_yaml = '/workspace/HAD-MC/yolov5/data/coco128.yaml'
    data_dict = check_dataset(check_yaml(data_yaml))
    train_path = data_dict['train']
    
    print(f"\n[2/4] Creating dataloader...")
    dataloader = create_dataloader(
        train_path,
        imgsz=640,
        batch_size=16,
        stride=32,
        hyp=None,
        augment=False,
        cache=False,
        rect=False,
        rank=-1,
        workers=8,
        image_weights=False,
        quad=False,
        prefix='train: '
    )[0]
    
    # 计算原始模型的FLOPs
    print(f"\n[3/4] Calculating original FLOPs...")
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        original_flops, _ = profile(student_model, inputs=(dummy_input,), verbose=False)
        print(f"Original FLOPs: {original_flops / 1e9:.2f} GFLOPs")
        target_flops = original_flops * 0.5
        print(f"Target FLOPs: {target_flops / 1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"WARNING: FLOPs calculation failed: {e}")
        target_flops = 8e9  # 默认值
    
    # 1. 梯度敏感剪枝
    print(f"\n[4/4] Gradient-Sensitivity Pruning...")
    print("-" * 80)
    
    try:
        print("Creating pruner...")
        pruner = GradientSensitivityPruner(
            model=student_model,
            train_loader=dataloader,
            flops_target=target_flops,
            device=device
        )
        print(f"Pruner created: {type(pruner)}")
        
        print("Calling pruner.run()...")
        result = pruner.run(prune_ratio=0.5)
        print(f"Pruner returned: {type(result)}")
        
        pruned_model = result
        print(f"✓ Pruning completed")
        
    except Exception as e:
        print(f"ERROR: Pruning failed: {e}")
        traceback.print_exc()
        print("Using original model without pruning")
        pruned_model = student_model
    
    # 2. 层级自适应量化
    print(f"\n[5/5] Layer-wise Adaptive Quantization...")
    print("-" * 80)
    
    try:
        print("Creating quantizer...")
        quantizer = LayerwisePrecisionAllocator(
            model=pruned_model,
            calibration_loader=dataloader,
            tau_h=1e-3,
            tau_l=1e-5,
            device=device
        )
        print(f"Quantizer created: {type(quantizer)}")
        
        print("Calling quantizer.run()...")
        result = quantizer.run(target_bits=8)
        print(f"Quantizer returned: {type(result)}")
        
        quantized_model = result
        print(f"✓ Quantization completed")
        
    except Exception as e:
        print(f"ERROR: Quantization failed: {e}")
        traceback.print_exc()
        print("Using pruned model without quantization")
        quantized_model = pruned_model
    
    # 3. 特征对齐知识蒸馏
    print(f"\n[6/6] Feature-Aligned Knowledge Distillation...")
    print("-" * 80)
    
    try:
        print("Creating distiller...")
        distiller = FeatureAlignedDistiller(
            teacher_model=teacher_model,
            student_model=quantized_model,
            device=device
        )
        print(f"Distiller created: {type(distiller)}")
        
        print("Calling distiller.run()...")
        result = distiller.run(
            train_loader=dataloader,
            epochs=2,  # 减少到2个epoch以加快速度
            lr=0.001
        )
        print(f"Distiller returned: {type(result)}")
        
        final_model = result
        print(f"✓ Distillation completed")
        
    except Exception as e:
        print(f"ERROR: Distillation failed: {e}")
        traceback.print_exc()
        print("Using quantized model without distillation")
        final_model = quantized_model
    
    # 保存模型
    output_path = Path('/workspace/HAD-MC/experiments/results/phase1_comprehensive/hadmc_full/model.pt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为YOLOv5兼容格式
    print(f"\nSaving model...")
    torch.save({
        'model': final_model,
        'epoch': -1,
        'optimizer': None,
        'date': None
    }, output_path)
    
    print(f"✓ HAD-MC model saved to {output_path}")
    
    # 计算最终模型大小
    model_size = output_path.stat().st_size / 1024 / 1024
    original_size = Path(fp32_model_path).stat().st_size / 1024 / 1024
    compression_ratio = original_size / model_size
    
    print(f"\nModel Size:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Compressed: {model_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return str(output_path)

if __name__ == '__main__':
    try:
        model_path = run_hadmc_full()
        print(f"\n{'=' * 80}")
        print("HAD-MC Full Pipeline Completed!")
        print(f"{'=' * 80}")
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
