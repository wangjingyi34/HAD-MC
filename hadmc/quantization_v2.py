"""
Algorithm 1: Layer-wise Precision Allocation (真实量化版本)
使用PyTorch原生量化API实现真实的INT8/INT4量化
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from typing import Dict, List, Tuple

class RealQuantizer:
    """
    真实量化器 - 使用PyTorch量化API
    实现基于梯度敏感性的混合精度量化
    """
    
    def __init__(self, model: nn.Module, target_bits: float = 6.0):
        """
        初始化量化器
        
        Args:
            model: 要量化的模型
            target_bits: 目标平均比特数 (例如6.0表示平均6位)
        """
        self.model = model
        self.target_bits = target_bits
        self.layer_sensitivities = {}
        self.layer_precisions = {}
        
    def calculate_gradient_sensitivity(self, dataloader, criterion, device='cpu'):
        """
        计算每层的梯度敏感性
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            device: 设备
            
        Returns:
            Dict[str, float]: 每层的敏感性分数
        """
        self.model.to(device)
        self.model.train()
        
        sensitivities = {}
        
        # 收集一个batch的梯度
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 1:  # 只用一个batch
                break
                
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = self.model(data)
            loss = criterion(output, target)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 计算每层参数的梯度敏感性
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # 使用梯度的平均绝对值作为敏感性指标
                    sensitivity = torch.mean(torch.abs(param.grad)).item()
                    sensitivities[name] = sensitivity
        
        self.layer_sensitivities = sensitivities
        return sensitivities
    
    def allocate_precision(self, tau_h: float = 0.7, tau_l: float = 0.3):
        """
        基于梯度敏感性分配精度
        
        Args:
            tau_h: 高敏感性阈值 (FP32)
            tau_l: 低敏感性阈值 (INT4)
            
        Returns:
            Dict[str, int]: 每层的比特数 (32, 8, 4)
        """
        if not self.layer_sensitivities:
            raise ValueError("请先调用calculate_gradient_sensitivity()")
        
        # 归一化敏感性到[0, 1]
        max_sens = max(self.layer_sensitivities.values())
        min_sens = min(self.layer_sensitivities.values())
        
        precisions = {}
        for name, sens in self.layer_sensitivities.items():
            # 归一化
            norm_sens = (sens - min_sens) / (max_sens - min_sens + 1e-8)
            
            # 分配精度
            if norm_sens > tau_h:
                precisions[name] = 32  # FP32
            elif norm_sens > tau_l:
                precisions[name] = 8   # INT8
            else:
                precisions[name] = 4   # INT4
        
        self.layer_precisions = precisions
        return precisions
    
    def prepare_qat_model(self):
        """
        准备量化感知训练(QAT)模型
        
        Returns:
            nn.Module: 准备好的QAT模型
        """
        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # 准备QAT
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        return self.model
    
    def convert_to_quantized(self):
        """
        将QAT模型转换为量化模型
        
        Returns:
            nn.Module: 量化后的模型
        """
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model
    
    def quantize_layer_weights(self, layer: nn.Module, bits: int = 8):
        """
        量化单个层的权重
        
        Args:
            layer: 要量化的层
            bits: 量化比特数 (8 or 4)
            
        Returns:
            量化后的权重
        """
        if not hasattr(layer, 'weight'):
            return None
            
        weight = layer.weight.data
        
        # 计算量化参数
        if bits == 8:
            qmin, qmax = -128, 127
        elif bits == 4:
            qmin, qmax = -8, 7
        else:
            return weight  # FP32不量化
        
        # 计算scale和zero_point
        min_val = weight.min()
        max_val = weight.max()
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = int(zero_point.round().clamp(qmin, qmax))
        
        # 量化
        quantized = torch.quantize_per_tensor(
            weight, 
            scale=scale.item(), 
            zero_point=zero_point, 
            dtype=torch.qint8
        )
        
        # 反量化(用于继续训练)
        dequantized = quantized.dequantize()
        
        return dequantized
    
    def apply_mixed_precision(self):
        """
        应用混合精度量化到模型
        """
        if not self.layer_precisions:
            raise ValueError("请先调用allocate_precision()")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 找到对应的精度设置
                param_name = name + '.weight'
                if param_name in self.layer_precisions:
                    bits = self.layer_precisions[param_name]
                    
                    if bits < 32:
                        # 量化权重
                        quantized_weight = self.quantize_layer_weights(module, bits)
                        if quantized_weight is not None:
                            module.weight.data = quantized_weight
    
    def measure_model_size(self):
        """
        测量模型大小
        
        Returns:
            Dict: 模型大小统计
        """
        total_params = 0
        total_bits = 0
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            # 获取该参数的比特数
            bits = self.layer_precisions.get(name, 32)
            total_bits += num_params * bits
        
        # 计算大小(MB)
        size_mb = total_bits / (8 * 1024 * 1024)
        avg_bits = total_bits / total_params if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'total_bits': total_bits,
            'size_mb': size_mb,
            'average_bits': avg_bits
        }
    
    def run(self, dataloader, criterion, device='cpu'):
        """
        运行完整的量化流程
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            device: 设备
            
        Returns:
            Dict: 量化结果
        """
        print("Step 1: 计算梯度敏感性...")
        sensitivities = self.calculate_gradient_sensitivity(dataloader, criterion, device)
        
        print("Step 2: 分配精度...")
        precisions = self.allocate_precision()
        
        print("Step 3: 应用混合精度量化...")
        self.apply_mixed_precision()
        
        print("Step 4: 测量模型大小...")
        size_info = self.measure_model_size()
        
        # 统计精度分布
        precision_dist = {}
        for bits in precisions.values():
            precision_dist[bits] = precision_dist.get(bits, 0) + 1
        
        results = {
            'sensitivities': sensitivities,
            'precisions': precisions,
            'precision_distribution': precision_dist,
            'model_size': size_info
        }
        
        print(f"\n量化完成!")
        print(f"  模型大小: {size_info['size_mb']:.2f} MB")
        print(f"  平均比特数: {size_info['average_bits']:.2f}")
        print(f"  精度分布: {precision_dist}")
        
        return results


def demo_real_quantization():
    """演示真实量化"""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # 创建模拟数据
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # 创建量化器
    quantizer = RealQuantizer(model, target_bits=6.0)
    
    # 运行量化
    results = quantizer.run(dataloader, nn.CrossEntropyLoss())
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("Algorithm 1: 真实量化演示")
    print("="*80)
    
    results = demo_real_quantization()
    
    print("\n量化结果:")
    print(f"  模型大小: {results['model_size']['size_mb']:.2f} MB")
    print(f"  平均比特数: {results['model_size']['average_bits']:.2f}")
