"""Comprehensive test suite for all HAD-MC algorithms"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.pruning import GradientSensitivityPruner
from hadmc.distillation import FeatureAlignedDistiller
from hadmc.fusion import OperatorFuser
from hadmc.incremental_update import IncrementalUpdater


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8)
    return loader


@pytest.fixture
def simple_model():
    """Create simple model for testing"""
    return SimpleModel()


class TestQuantization:
    """Test Algorithm 1: Layer-wise Precision Allocation"""
    
    def test_allocator_initialization(self, simple_model, sample_data):
        """Test allocator can be initialized"""
        allocator = LayerwisePrecisionAllocator(
            simple_model, sample_data,
            tau_h=1e-3, tau_l=1e-5
        )
        assert allocator is not None
        assert allocator.model is not None
    
    def test_gradient_sensitivity_calculation(self, simple_model, sample_data):
        """Test gradient sensitivity calculation"""
        allocator = LayerwisePrecisionAllocator(simple_model, sample_data)
        allocator.calculate_gradient_sensitivity()
        assert len(allocator.gradient_sensitivity) > 0
    
    def test_precision_allocation(self, simple_model, sample_data):
        """Test precision allocation"""
        allocator = LayerwisePrecisionAllocator(simple_model, sample_data)
        allocator.calculate_gradient_sensitivity()
        precision_map = allocator.allocate_precision()
        assert len(precision_map) > 0
        assert all(p in ['FP32', 'INT8', 'INT4'] for p in precision_map.values())
    
    def test_full_quantization_pipeline(self, simple_model, sample_data):
        """Test full quantization pipeline"""
        allocator = LayerwisePrecisionAllocator(simple_model, sample_data)
        quantized_model = allocator.run()
        assert quantized_model is not None
        
        # Test inference
        x = torch.randn(1, 3, 32, 32)
        output = quantized_model(x)
        assert output.shape == (1, 10)


class TestPruning:
    """Test Algorithm 2: Gradient Sensitivity-Guided Pruning"""
    
    def test_pruner_initialization(self, simple_model, sample_data):
        """Test pruner can be initialized"""
        pruner = GradientSensitivityPruner(
            simple_model, sample_data, flops_target=0.5
        )
        assert pruner is not None
    
    def test_channel_importance_calculation(self, simple_model, sample_data):
        """Test channel importance calculation"""
        pruner = GradientSensitivityPruner(
            simple_model, sample_data, flops_target=0.5
        )
        pruner.calculate_channel_importance()
        assert len(pruner.channel_importance) > 0
    
    def test_pruning_reduces_parameters(self, simple_model, sample_data):
        """Test that pruning reduces model parameters"""
        original_params = sum(p.numel() for p in simple_model.parameters())
        
        pruner = GradientSensitivityPruner(
            simple_model, sample_data, flops_target=0.5
        )
        pruned_model = pruner.run()
        
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        assert pruned_params <= original_params
    
    def test_pruned_model_inference(self, simple_model, sample_data):
        """Test pruned model can perform inference"""
        pruner = GradientSensitivityPruner(
            simple_model, sample_data, flops_target=0.5
        )
        pruned_model = pruner.run()
        
        x = torch.randn(1, 3, 32, 32)
        output = pruned_model(x)
        assert output.shape == (1, 10)


class TestDistillation:
    """Test Algorithm 3: Feature-Aligned Knowledge Distillation"""
    
    def test_distiller_initialization(self, simple_model):
        """Test distiller can be initialized"""
        teacher = simple_model
        student = SimpleModel()
        distiller = FeatureAlignedDistiller(teacher, student)
        assert distiller is not None
    
    def test_task_loss_computation(self, simple_model):
        """Test task loss computation"""
        teacher = simple_model
        student = SimpleModel()
        distiller = FeatureAlignedDistiller(teacher, student)
        
        output = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        loss = distiller.compute_task_loss(output, target)
        assert loss.item() > 0
    
    def test_soft_loss_computation(self, simple_model):
        """Test soft loss computation"""
        teacher = simple_model
        student = SimpleModel()
        distiller = FeatureAlignedDistiller(teacher, student)
        
        student_output = torch.randn(4, 10)
        teacher_output = torch.randn(4, 10)
        loss = distiller.compute_soft_loss(student_output, teacher_output)
        assert loss.item() >= 0
    
    def test_distillation_training(self, simple_model, sample_data):
        """Test distillation training"""
        teacher = simple_model
        student = SimpleModel()
        distiller = FeatureAlignedDistiller(teacher, student)
        
        distilled_model = distiller.run(sample_data, epochs=2)
        assert distilled_model is not None
        
        # Test inference
        x = torch.randn(1, 3, 32, 32)
        output = distilled_model(x)
        assert output.shape == (1, 10)


class TestFusion:
    """Test Algorithm 4: Operator Fusion"""
    
    def test_fuser_initialization(self, simple_model):
        """Test fuser can be initialized"""
        fuser = OperatorFuser(simple_model)
        assert fuser is not None
    
    def test_fusion_pattern_detection(self, simple_model):
        """Test fusion pattern detection"""
        fuser = OperatorFuser(simple_model)
        fused_model = fuser.run()
        assert fused_model is not None
    
    def test_fused_model_inference(self, simple_model):
        """Test fused model can perform inference"""
        fuser = OperatorFuser(simple_model)
        fused_model = fuser.run()
        
        x = torch.randn(1, 3, 32, 32)
        output = fused_model(x)
        assert output.shape == (1, 10)


class TestIncrementalUpdate:
    """Test Algorithm 5: Hash-based Incremental Update"""
    
    def test_updater_initialization(self):
        """Test updater can be initialized"""
        updater = IncrementalUpdater(block_size=4096)
        assert updater is not None
    
    def test_model_division(self, simple_model):
        """Test model division into blocks"""
        updater = IncrementalUpdater(block_size=1024)
        blocks = updater.divide_into_blocks(simple_model)
        assert len(blocks) > 0
    
    def test_hash_computation(self, simple_model):
        """Test hash computation"""
        updater = IncrementalUpdater()
        blocks = updater.divide_into_blocks(simple_model)
        hashes = updater.compute_hashes(blocks)
        assert len(hashes) == len(blocks)
    
    def test_delta_computation(self, simple_model):
        """Test delta computation between models"""
        old_model = simple_model
        new_model = SimpleModel()
        
        # Modify new model slightly
        with torch.no_grad():
            new_model.fc.weight += 0.1
        
        updater = IncrementalUpdater()
        changed_blocks = updater.compute_delta(old_model, new_model)
        assert len(changed_blocks) > 0
    
    def test_bandwidth_reduction(self, simple_model):
        """Test bandwidth reduction calculation"""
        old_model = simple_model
        new_model = SimpleModel()
        
        updater = IncrementalUpdater()
        updater.compute_delta(old_model, new_model)
        reduction = updater.get_bandwidth_reduction()
        assert 0 <= reduction <= 1


class TestIntegration:
    """Integration tests for combined algorithms"""
    
    def test_sequential_pipeline(self, simple_model, sample_data):
        """Test running all algorithms sequentially"""
        model = simple_model
        
        # 1. Quantization
        allocator = LayerwisePrecisionAllocator(model, sample_data)
        model = allocator.run()
        
        # 2. Pruning
        pruner = GradientSensitivityPruner(model, sample_data, flops_target=0.7)
        model = pruner.run()
        
        # 3. Distillation
        teacher = SimpleModel()
        distiller = FeatureAlignedDistiller(teacher, model)
        model = distiller.run(sample_data, epochs=1)
        
        # 4. Fusion
        fuser = OperatorFuser(model)
        model = fuser.run()
        
        # Final inference test
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_model_size_reduction(self, simple_model, sample_data):
        """Test that pipeline reduces model size"""
        original_size = sum(p.numel() * p.element_size() for p in simple_model.parameters())
        
        # Apply compression
        allocator = LayerwisePrecisionAllocator(simple_model, sample_data)
        compressed_model = allocator.run()
        
        pruner = GradientSensitivityPruner(compressed_model, sample_data, flops_target=0.5)
        compressed_model = pruner.run()
        
        compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
        
        # Size should be reduced (or at least not increased)
        assert compressed_size <= original_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
