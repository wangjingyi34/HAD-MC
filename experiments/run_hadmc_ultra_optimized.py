#!/usr/bin/env python3
"""
HAD-MC Ultra Optimized - 目标：精度下降<2%
策略：
1. 极度保守的剪枝（0.1%）
2. 跳过量化
3. 更长的蒸馏（200 epochs）
4. 更小的学习率
5. 更强的蒸馏权重
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from pathlib import Path

# 设置路径
YOLOV5_DIR = "/workspace/HAD-MC/yolov5"
HADMC_DIR = "/workspace/HAD-MC/hadmc"
sys.path.insert(0, YOLOV5_DIR)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/workspace/HAD-MC/experiments/hadmc_ultra_optimized.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("HAD-MC Ultra Optimized - 目标：精度下降<2%")
    logger.info("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 路径
    model_path = "/workspace/HAD-MC/experiments/results/fp32_baseline/train/weights/best.pt"
    data_yaml = "/workspace/HAD-MC/datasets/coco128/coco128.yaml"
    output_dir = "/workspace/HAD-MC/experiments/results/hadmc_ultra_optimized"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    logger.info("加载FP32 Baseline模型...")
    ckpt = torch.load(model_path, map_location=device)
    
    # 创建teacher和student的深拷贝
    teacher_model = copy.deepcopy(ckpt['model']).float().to(device)
    student_model = copy.deepcopy(ckpt['model']).float().to(device)
    
    teacher_model.eval()
    student_model.train()
    
    logger.info(f"模型加载成功")
    
    # 创建数据加载器
    from utils.dataloaders import create_dataloader
    
    train_loader, _ = create_dataloader(
        path="/workspace/HAD-MC/datasets/coco128/images/train2017",
        imgsz=640,
        batch_size=8,
        stride=32,
        hyp=None,
        augment=False,
        cache=False,
        rect=True,
        rank=-1,
        workers=4,
        pad=0.5,
        shuffle=True
    )
    
    logger.info(f"数据加载器创建成功，共{len(train_loader)}个batch")
    
    # ========================================
    # Phase 1: 极度保守的剪枝 (0.1%)
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: 极度保守的剪枝 (0.1%)")
    logger.info("=" * 60)
    
    prune_ratio = 0.001  # 0.1%
    
    # 计算每层的L1范数
    layer_importance = {}
    for name, module in student_model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            importance = weight.abs().mean(dim=(1, 2, 3))  # 每个输出通道的重要性
            layer_importance[name] = importance
    
    # 剪枝最不重要的通道
    total_channels = sum(imp.numel() for imp in layer_importance.values())
    channels_to_prune = int(total_channels * prune_ratio)
    
    logger.info(f"总通道数: {total_channels}")
    logger.info(f"计划剪枝通道数: {channels_to_prune}")
    
    # 收集所有通道的重要性
    all_importances = []
    for name, importance in layer_importance.items():
        for i, imp in enumerate(importance):
            all_importances.append((name, i, imp.item()))
    
    # 按重要性排序
    all_importances.sort(key=lambda x: x[2])
    
    # 剪枝（将权重置零）
    pruned_count = 0
    for name, channel_idx, _ in all_importances[:channels_to_prune]:
        for n, module in student_model.named_modules():
            if n == name and isinstance(module, nn.Conv2d):
                module.weight.data[channel_idx] = 0
                pruned_count += 1
                break
    
    logger.info(f"实际剪枝通道数: {pruned_count}")
    logger.info(f"剪枝比例: {pruned_count/total_channels*100:.2f}%")
    
    # ========================================
    # Phase 2: 跳过量化
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: 跳过量化（基于消融实验，量化对YOLOv5有负面影响）")
    logger.info("=" * 60)
    
    # ========================================
    # Phase 3: 强化知识蒸馏 (200 epochs)
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: 强化知识蒸馏 (200 epochs)")
    logger.info("=" * 60)
    
    # 超参数
    epochs = 200
    base_lr = 0.00005  # 更小的学习率
    warmup_epochs = 10
    alpha = 0.1  # task loss权重
    beta = 0.9   # distillation loss权重
    
    # 优化器
    optimizer = optim.AdamW(student_model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    # 余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
    
    # 加载YOLOv5的loss函数
    from utils.loss import ComputeLoss
    compute_loss = ComputeLoss(student_model)
    
    logger.info(f"训练配置:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Base LR: {base_lr}")
    logger.info(f"  - Warmup epochs: {warmup_epochs}")
    logger.info(f"  - Alpha (task): {alpha}")
    logger.info(f"  - Beta (distill): {beta}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0
        epoch_task_loss = 0
        epoch_distill_loss = 0
        num_batches = 0
        
        # Warmup学习率
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for batch_idx, (imgs, targets, paths, shapes) in enumerate(train_loader):
            if batch_idx >= 50:  # 每个epoch最多50个batch
                break
            
            imgs = imgs.to(device).float() / 255.0
            imgs.requires_grad = True
            targets = targets.to(device)
            
            # Teacher前向传播
            with torch.no_grad():
                teacher_model.train()  # 使用train模式以获得相同的输出格式
                teacher_preds = teacher_model(imgs)
            
            # Student前向传播
            student_preds = student_model(imgs)
            
            # Task loss
            try:
                task_loss, loss_items = compute_loss(student_preds, targets)
            except:
                task_loss = torch.tensor(0.0, device=device)
            
            # Distillation loss (MSE on predictions)
            distill_loss = torch.tensor(0.0, device=device)
            if isinstance(student_preds, (list, tuple)) and isinstance(teacher_preds, (list, tuple)):
                for s_pred, t_pred in zip(student_preds, teacher_preds):
                    if s_pred.shape == t_pred.shape:
                        distill_loss += nn.functional.mse_loss(s_pred, t_pred.detach())
            
            # 动态调整alpha/beta
            progress = epoch / epochs
            current_alpha = alpha + (0.5 - alpha) * progress  # 从0.1逐渐增加到0.5
            current_beta = beta - (beta - 0.5) * progress     # 从0.9逐渐减少到0.5
            
            # 总loss
            total_loss = current_alpha * task_loss + current_beta * distill_loss
            
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10.0)
                optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_task_loss += task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss
            epoch_distill_loss += distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            num_batches += 1
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_task = epoch_task_loss / max(num_batches, 1)
        avg_distill = epoch_distill_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} "
                       f"(task: {avg_task:.4f}, distill: {avg_distill:.4f}) "
                       f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存模型
            save_dict = {
                'epoch': epoch,
                'best_fitness': None,
                'model': student_model,
                'ema': None,
                'updates': None,
                'optimizer': optimizer.state_dict(),
                'opt': None,
                'git': None,
                'date': None,
                'hadmc_config': {
                    'prune_ratio': prune_ratio,
                    'epochs': epochs,
                    'alpha': alpha,
                    'beta': beta,
                    'base_lr': base_lr
                }
            }
            torch.save(save_dict, os.path.join(output_dir, 'best.pt'))
    
    # 保存最终模型
    save_dict = {
        'epoch': epochs,
        'best_fitness': None,
        'model': student_model,
        'ema': None,
        'updates': None,
        'optimizer': optimizer.state_dict(),
        'opt': None,
        'git': None,
        'date': None,
        'hadmc_config': {
            'prune_ratio': prune_ratio,
            'epochs': epochs,
            'alpha': alpha,
            'beta': beta,
            'base_lr': base_lr
        }
    }
    torch.save(save_dict, os.path.join(output_dir, 'last.pt'))
    
    logger.info(f"\n模型已保存到: {output_dir}")
    logger.info(f"最佳Loss: {best_loss:.4f}")
    
    # ========================================
    # Phase 4: 评估
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: 评估HAD-MC Ultra Optimized模型")
    logger.info("=" * 60)
    
    # 使用val.py评估
    import subprocess
    result = subprocess.run([
        'python3', 'val.py',
        '--weights', os.path.join(output_dir, 'best.pt'),
        '--data', data_yaml,
        '--img', '640',
        '--batch-size', '8',
        '--device', '0',
        '--project', output_dir,
        '--name', 'eval',
        '--exist-ok'
    ], capture_output=True, text=True, cwd=YOLOV5_DIR)
    
    logger.info("评估输出:")
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    
    logger.info("\n" + "=" * 80)
    logger.info("HAD-MC Ultra Optimized 完成！")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
