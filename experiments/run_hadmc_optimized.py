#!/usr/bin/env python3
"""
HAD-MC Optimized for YOLOv5 Object Detection
============================================

Optimizations based on ablation study findings:
1. Skip quantization (negative impact on YOLOv5)
2. Use very conservative pruning (0.5%)
3. Optimize distillation parameters
4. Increase distillation epochs (100)
5. Use cosine annealing learning rate

Author: Manus AI Agent
Date: 2026-01-06
"""

import torch
import sys
import copy
import logging
import os
import math

# Fix import path - must be done before any other imports
os.chdir('/workspace/HAD-MC/yolov5')
sys.path.insert(0, '/workspace/HAD-MC/yolov5')
sys.path.insert(0, '/workspace/HAD-MC/hadmc')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/workspace/HAD-MC/experiments/hadmc_optimized.log')
    ]
)
logger = logging.getLogger('hadmc_optimized')

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss

# Import HAD-MC components
from hadmc_yolov5 import YOLOv5GradientSensitivityPruner

class OptimizedYOLOv5Distiller:
    """
    Optimized Knowledge Distillation for YOLOv5
    
    Key optimizations:
    1. Use cosine annealing learning rate
    2. Warm-up phase for first 5 epochs
    3. Better loss balancing
    4. Gradient clipping
    """
    
    def __init__(self, teacher_model, student_model, device='cuda'):
        self.teacher = teacher_model
        self.student = student_model
        self.device = device
        
        # Move models to device
        self.teacher.to(device)
        self.student.to(device)
        
        # Set teacher to eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Initialize loss function
        self.compute_loss = ComputeLoss(self.student)
        
        logger.info("OptimizedYOLOv5Distiller initialized")
    
    def run(self, train_loader, epochs=100, base_lr=0.0001, warmup_epochs=5):
        """
        Run optimized knowledge distillation
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            base_lr: Base learning rate
            warmup_epochs: Number of warmup epochs
        
        Returns:
            Distilled student model
        """
        logger.info(f"Starting optimized distillation: {epochs} epochs, base_lr={base_lr}")
        
        # Set student to training mode
        self.student.train()
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=base_lr,
            weight_decay=0.0005
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=base_lr * 0.01
        )
        
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            task_loss_sum = 0.0
            distill_loss_sum = 0.0
            num_batches = 0
            
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            for batch_idx, (imgs, targets, paths, _) in enumerate(train_loader):
                if batch_idx >= 50:  # Limit batches per epoch
                    break
                
                # Move to device
                imgs = imgs.to(self.device).float() / 255.0
                imgs.requires_grad = True
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - student
                student_preds = self.student(imgs)
                
                # Compute task loss
                task_loss, _ = self.compute_loss(student_preds, targets)
                
                # Forward pass - teacher (no grad)
                with torch.no_grad():
                    self.teacher.train()  # Use training mode for consistent output
                    teacher_preds = self.teacher(imgs)
                    self.teacher.eval()
                
                # Compute distillation loss (MSE on predictions)
                distill_loss = self._compute_distill_loss(student_preds, teacher_preds)
                
                # Dynamic loss balancing based on epoch
                # Start with more task loss, gradually increase distillation
                progress = epoch / epochs
                alpha = 0.3 + 0.2 * progress  # 0.3 -> 0.5
                beta = 0.7 - 0.2 * progress   # 0.7 -> 0.5
                
                # Combined loss
                total_loss = alpha * task_loss + beta * distill_loss
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                task_loss_sum += task_loss.item()
                distill_loss_sum += distill_loss.item()
                num_batches += 1
            
            # Update scheduler after warmup
            if epoch >= warmup_epochs:
                scheduler.step()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_task = task_loss_sum / max(num_batches, 1)
            avg_distill = distill_loss_sum / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} "
                       f"(task: {avg_task:.4f}, distill: {avg_distill:.4f}), "
                       f"LR: {current_lr:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.student.state_dict())
                logger.info(f"  -> New best model saved (loss: {best_loss:.4f})")
        
        # Restore best model
        if best_model_state is not None:
            self.student.load_state_dict(best_model_state)
            logger.info(f"Restored best model with loss: {best_loss:.4f}")
        
        logger.info("Optimized distillation completed")
        return self.student
    
    def _compute_distill_loss(self, student_preds, teacher_preds):
        """Compute distillation loss between student and teacher predictions"""
        total_loss = 0.0
        count = 0
        
        for s_pred, t_pred in zip(student_preds, teacher_preds):
            if s_pred.shape == t_pred.shape:
                # MSE loss on raw predictions
                mse_loss = torch.nn.functional.mse_loss(s_pred, t_pred)
                total_loss += mse_loss
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss / count


def main():
    """Main function to run optimized HAD-MC"""
    
    logger.info("=" * 60)
    logger.info("HAD-MC Optimized for YOLOv5")
    logger.info("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_model_path = '/workspace/HAD-MC/experiments/results/fp32_baseline/train/weights/best.pt'
    output_dir = '/workspace/HAD-MC/experiments/results/phase1_comprehensive/hadmc_optimized'
    
    # Optimized parameters
    prune_ratio = 0.005  # 0.5% (very conservative)
    distill_epochs = 100
    base_lr = 0.0001
    warmup_epochs = 5
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Device: {device}")
    logger.info(f"Prune ratio: {prune_ratio}")
    logger.info(f"Distill epochs: {distill_epochs}")
    logger.info(f"Base LR: {base_lr}")
    
    # Load models
    logger.info("Loading models...")
    teacher_ckpt = torch.load(baseline_model_path, map_location=device)
    teacher_model = teacher_ckpt['model'].float().to(device)
    teacher_model.eval()
    
    student_ckpt = torch.load(baseline_model_path, map_location=device)
    student_model = copy.deepcopy(student_ckpt['model'].float().to(device))
    
    logger.info("Models loaded successfully")
    
    # Create dataloader
    logger.info("Creating dataloader...")
    train_loader, _ = create_dataloader(
        '/workspace/HAD-MC/datasets/coco128/images/train2017',
        640, 8, 32,
        hyp={'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0,
             'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0,
             'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0},
        augment=False,
        rect=True,
        rank=-1,
        workers=4,
        prefix='train: '
    )
    
    # Phase 1: Gradient Sensitivity Pruning (very conservative)
    logger.info("=" * 60)
    logger.info("Phase 1: Gradient Sensitivity Pruning (0.5%)")
    logger.info("=" * 60)
    
    pruner = YOLOv5GradientSensitivityPruner(
        student_model, train_loader, flops_target=0.5, device=device
    )
    student_model = pruner.run(prune_ratio=prune_ratio)
    
    # Phase 2: Skip Quantization (based on ablation findings)
    logger.info("=" * 60)
    logger.info("Phase 2: Skipping Quantization (based on ablation findings)")
    logger.info("=" * 60)
    logger.info("Quantization showed negative impact on YOLOv5, skipping...")
    
    # Phase 3: Optimized Knowledge Distillation
    logger.info("=" * 60)
    logger.info("Phase 3: Optimized Knowledge Distillation")
    logger.info("=" * 60)
    
    distiller = OptimizedYOLOv5Distiller(teacher_model, student_model, device=device)
    student_model = distiller.run(
        train_loader,
        epochs=distill_epochs,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs
    )
    
    # Save model
    logger.info("=" * 60)
    logger.info("Saving optimized HAD-MC model...")
    logger.info("=" * 60)
    
    save_path = os.path.join(output_dir, 'model.pt')
    torch.save({'model': student_model}, save_path)
    logger.info(f"Model saved to: {save_path}")
    
    # Calculate model size
    model_size = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"Model size: {model_size:.2f} MB")
    
    logger.info("=" * 60)
    logger.info("HAD-MC Optimized completed!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To evaluate, run:")
    logger.info(f"  python val.py --weights {save_path} --data data/coco128.yaml")


if __name__ == '__main__':
    main()
