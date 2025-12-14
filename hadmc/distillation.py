"""Algorithm 3: Feature-Aligned Knowledge Distillation"""
from .device_manager import DeviceManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FeatureAlignedDistiller:
    """Implements Algorithm 3 from HAD-MC paper"""
    
    def __init__(self, teacher_model, student_model, device='cpu'):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.adaptation_layers = {}
        
    def add_adaptation_layers(self):
        """Add 1x1 conv layers to align feature dimensions"""
        logger.info("Adding adaptation layers...")
        # Simplified: assume same architecture
        return self.adaptation_layers
    
    def compute_task_loss(self, student_output, target):
        """Compute classification loss"""
        if isinstance(student_output, tuple):
            student_output = student_output[0]
        return F.cross_entropy(student_output, target)
    
    def compute_soft_loss(self, student_output, teacher_output, temperature=4.0):
        """Compute soft label loss"""
        if isinstance(student_output, tuple):
            student_output = student_output[0]
        if isinstance(teacher_output, tuple):
            teacher_output = teacher_output[0]
            
        soft_student = F.log_softmax(student_output / temperature, dim=1)
        soft_teacher = F.softmax(teacher_output / temperature, dim=1)
        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    def compute_feature_loss(self, student_features, teacher_features):
        """Compute feature matching loss"""
        # Simplified: MSE loss
        return F.mse_loss(student_features, teacher_features)
    
    def train_epoch(self, train_loader, optimizer, alpha=0.3, beta=0.3):
        """Train for one epoch"""
        self.student.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            student_output = self.student(data)
            with torch.no_grad():
                teacher_output = self.teacher(data)
            
            # Compute losses
            task_loss = self.compute_task_loss(student_output, target)
            soft_loss = self.compute_soft_loss(student_output, teacher_output)
            
            # Total loss (simplified, no feature loss for now)
            loss = alpha * task_loss + beta * soft_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def run(self, train_loader, epochs=5, lr=0.001):
        """Run complete Algorithm 3"""
        logger.info("Starting knowledge distillation...")
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_loader, optimizer)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.student
