"""
Algorithm 4: Operator Fusion (真实参数融合版本)
实现真正的Conv-BN参数融合
数学公式: W_fused = gamma/sqrt(var+eps) * W_conv
         b_fused = gamma/sqrt(var+eps) * (b_conv - mean) + beta
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import copy

class RealFusion:
    """真实的算子融合器 - 实现Conv-BN参数级别的融合"""
    
    def __init__(self, model: nn.Module):
        """
        初始化融合器
        
        Args:
            model: 要融合的模型
        """
        self.model = model
        self.fusion_opportunities = []
        
    def detect_fusion_patterns(self) -> List[Tuple[str, str, str]]:
        """
        检测可融合的模式 (Conv + BN + ReLU)
        
        Returns:
            List of (conv_name, bn_name, relu_name) tuples
        """
        patterns = []
        modules = list(self.model.named_modules())
        
        for i in range(len(modules) - 1):
            name1, module1 = modules[i]
            name2, module2 = modules[i + 1]
            
            # 检测 Conv + BN
            if isinstance(module1, nn.Conv2d) and isinstance(module2, nn.BatchNorm2d):
                # 检查是否有ReLU
                relu_name = None
                if i + 2 < len(modules):
                    name3, module3 = modules[i + 2]
                    if isinstance(module3, nn.ReLU):
                        relu_name = name3
                
                patterns.append((name1, name2, relu_name))
        
        self.fusion_opportunities = patterns
        return patterns
    
    def fuse_conv_bn_params(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        融合Conv和BN的参数
        
        数学公式:
        W_fused = gamma / sqrt(var + eps) * W_conv
        b_fused = gamma / sqrt(var + eps) * (b_conv - mean) + beta
        
        Args:
            conv: 卷积层
            bn: 批归一化层
            
        Returns:
            (fused_weight, fused_bias): 融合后的权重和偏置
        """
        # 获取BN参数
        gamma = bn.weight.data  # shape: (out_channels,)
        beta = bn.bias.data     # shape: (out_channels,)
        mean = bn.running_mean  # shape: (out_channels,)
        var = bn.running_var    # shape: (out_channels,)
        eps = bn.eps
        
        # 获取Conv参数
        W_conv = conv.weight.data  # shape: (out_channels, in_channels, kH, kW)
        b_conv = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, device=W_conv.device)
        
        # 计算缩放因子 S = gamma / sqrt(var + eps)
        # shape: (out_channels,)
        S = gamma / torch.sqrt(var + eps)
        
        # 融合权重: W_fused = S * W_conv
        # 需要将S扩展为 (out_channels, 1, 1, 1) 以进行广播
        S_expanded = S.view(-1, 1, 1, 1)
        W_fused = W_conv * S_expanded
        
        # 融合偏置: b_fused = S * (b_conv - mean) + beta
        b_fused = S * (b_conv - mean) + beta
        
        return W_fused, b_fused
    
    def create_fused_conv(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, relu: nn.ReLU = None) -> nn.Module:
        """
        创建融合后的Conv层
        
        Args:
            conv: 原始卷积层
            bn: 批归一化层
            relu: ReLU层(可选)
            
        Returns:
            融合后的模块
        """
        # 融合参数
        fused_weight, fused_bias = self.fuse_conv_bn_params(conv, bn)
        
        # 创建新的Conv层
        fused_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True  # 融合后总是有bias
        )
        
        # 设置融合后的参数
        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias
        
        # 如果有ReLU,创建Sequential
        if relu is not None:
            return nn.Sequential(fused_conv, nn.ReLU())
        else:
            return fused_conv
    
    def apply_fusion(self) -> nn.Module:
        """
        应用融合到模型
        
        Returns:
            融合后的模型
        """
        if not self.fusion_opportunities:
            self.detect_fusion_patterns()
        
        # 创建模型的深拷贝
        fused_model = copy.deepcopy(self.model)
        
        # 对每个融合机会进行融合
        for conv_name, bn_name, relu_name in self.fusion_opportunities:
            # 获取模块
            conv = dict(self.model.named_modules())[conv_name]
            bn = dict(self.model.named_modules())[bn_name]
            relu = dict(self.model.named_modules())[relu_name] if relu_name else None
            
            # 创建融合模块
            fused_module = self.create_fused_conv(conv, bn, relu)
            
            # 替换模型中的模块
            # 这里需要根据实际模型结构进行替换
            # 简化处理:直接在Sequential中替换
            parent_name = '.'.join(conv_name.split('.')[:-1])
            if parent_name:
                parent = dict(fused_model.named_modules())[parent_name]
                if isinstance(parent, nn.Sequential):
                    # 找到conv的索引
                    conv_idx = int(conv_name.split('.')[-1])
                    parent[conv_idx] = fused_module
                    
                    # 移除BN和ReLU
                    # 注意:这里需要小心处理索引
        
        return fused_model
    
    def verify_fusion(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, fused_conv: nn.Conv2d, test_input: torch.Tensor) -> float:
        """
        验证融合的正确性
        
        Args:
            conv: 原始卷积层
            bn: 批归一化层
            fused_conv: 融合后的卷积层
            test_input: 测试输入
            
        Returns:
            最大误差
        """
        # 设置为评估模式
        conv.eval()
        bn.eval()
        fused_conv.eval()
        
        # 原始输出
        with torch.no_grad():
            original_output = bn(conv(test_input))
            fused_output = fused_conv(test_input)
        
        # 计算误差
        max_error = torch.max(torch.abs(original_output - fused_output)).item()
        
        return max_error
    
    def run(self) -> dict:
        """
        运行融合流程
        
        Returns:
            融合结果
        """
        print("Step 1: 检测融合模式...")
        patterns = self.detect_fusion_patterns()
        print(f"  发现 {len(patterns)} 个融合机会")
        
        for i, (conv_name, bn_name, relu_name) in enumerate(patterns, 1):
            relu_str = f" + {relu_name}" if relu_name else ""
            print(f"  {i}. {conv_name} + {bn_name}{relu_str}")
        
        print("\nStep 2: 执行参数融合...")
        fusion_count = 0
        max_errors = []
        
        for conv_name, bn_name, relu_name in patterns:
            conv = dict(self.model.named_modules())[conv_name]
            bn = dict(self.model.named_modules())[bn_name]
            
            # 创建融合层
            fused_conv = self.create_fused_conv(conv, bn, None)
            
            # 验证融合正确性
            test_input = torch.randn(1, conv.in_channels, 32, 32)
            max_error = self.verify_fusion(conv, bn, fused_conv, test_input)
            max_errors.append(max_error)
            
            fusion_count += 1
            print(f"  融合 {conv_name} + {bn_name}: 最大误差 = {max_error:.2e}")
        
        results = {
            'fusion_count': fusion_count,
            'patterns': patterns,
            'max_errors': max_errors,
            'avg_error': sum(max_errors) / len(max_errors) if max_errors else 0
        }
        
        print(f"\n融合完成!")
        print(f"  融合数量: {fusion_count}")
        print(f"  平均误差: {results['avg_error']:.2e}")
        
        return results


def demo_real_fusion():
    """演示真实融合"""
    
    # 创建包含Conv-BN-ReLU的模型
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1, bias=False),  # 无bias
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    
    # 初始化BN层的参数(模拟训练后的状态)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.data = torch.randn(module.num_features)
            module.running_var.data = torch.rand(module.num_features) + 0.5
            module.weight.data = torch.randn(module.num_features)
            module.bias.data = torch.randn(module.num_features)
    
    # 创建融合器
    fusion = RealFusion(model)
    
    # 运行融合
    results = fusion.run()
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("Algorithm 4: 真实Conv-BN融合演示")
    print("="*80)
    
    results = demo_real_fusion()
    
    print("\n融合结果:")
    print(f"  融合数量: {results['fusion_count']}")
    print(f"  数值误差: {results['avg_error']:.2e} (应该<1e-5)")
