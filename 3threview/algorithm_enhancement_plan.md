# HAD-MC 2.0 算法提升方案 (skilup.md)

**版本:** 1.0
**日期:** 2026年2月3日
**作者:** 12位教授级专家联合撰写
**目标:** 为Claude-Code提供详细的算法提升指导，实现HAD-MC到HAD-MC 2.0的升级

---

## 目录

1. [算法升级概述](#1-算法升级概述)
2. [多智能体强化学习框架设计](#2-多智能体强化学习框架设计)
3. [PPO控制器实现](#3-ppo控制器实现)
4. [硬件抽象层实现](#4-硬件抽象层实现)
5. [专用推理引擎实现](#5-专用推理引擎实现)
6. [五个智能体详细设计](#6-五个智能体详细设计)
7. [奖励函数设计](#7-奖励函数设计)
8. [训练流程](#8-训练流程)
9. [代码架构](#9-代码架构)
10. [测试与验证](#10-测试与验证)

---

## 1. 算法升级概述

### 1.1 升级目标

将HAD-MC从启发式流水线方法升级为基于多智能体强化学习（MARL）的协同优化框架。

### 1.2 核心改进

| 改进点 | HAD-MC 1.0 | HAD-MC 2.0 |
|--------|------------|------------|
| 优化范式 | 启发式顺序优化 | MARL协同优化 |
| 硬件适配 | 手动配置 | 自动适配（HAL） |
| 推理引擎 | 通用推理 | 专用推理（DIE） |
| 压缩策略 | 固定策略 | 学习策略 |

### 1.3 技术路线

```
HAD-MC 1.0
    │
    ▼
┌─────────────────────────────────────┐
│  Step 1: 设计MARL框架               │
│  - 定义状态空间、动作空间、奖励函数  │
│  - 设计智能体协同机制               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Step 2: 实现PPO控制器              │
│  - 策略网络设计                     │
│  - 价值网络设计                     │
│  - PPO训练算法                      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Step 3: 实现硬件抽象层（HAL）      │
│  - 统一硬件接口                     │
│  - 延迟查找表                       │
│  - 硬件配置文件                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Step 4: 实现专用推理引擎（DIE）    │
│  - 稀疏性感知执行                   │
│  - 混合精度推理                     │
│  - 算子融合执行                     │
└─────────────────────────────────────┘
    │
    ▼
HAD-MC 2.0
```

---

## 2. 多智能体强化学习框架设计

### 2.1 框架架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MARL Controller                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │ Pruning │ │Quantize │ │Distill  │ │ Fusion  │ │ Update  ││
│  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   ││
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘│
│       │           │           │           │           │     │
│       └───────────┴───────────┼───────────┴───────────┘     │
│                               │                             │
│                    ┌──────────▼──────────┐                  │
│                    │   Central Policy    │                  │
│                    │   (PPO Controller)  │                  │
│                    └──────────┬──────────┘                  │
└───────────────────────────────┼─────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Hardware Abstraction  │
                    │       Layer (HAL)     │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
    ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
    │   NVIDIA A100 │   │  Ascend 310   │   │   Hygon DCU   │
    └───────────────┘   └───────────────┘   └───────────────┘
```

### 2.2 状态空间设计

状态空间S包含三个部分：模型状态、硬件状态、压缩状态。

```python
class State:
    """MARL状态空间定义"""
    
    def __init__(self):
        # 模型状态 (Model State)
        self.model_state = {
            'num_layers': int,           # 层数
            'layer_types': List[str],    # 层类型 ['conv', 'bn', 'relu', ...]
            'channel_counts': List[int], # 每层通道数
            'param_counts': List[int],   # 每层参数量
            'flop_counts': List[int],    # 每层FLOPs
            'activation_sizes': List[int], # 每层激活大小
        }
        
        # 硬件状态 (Hardware State)
        self.hardware_state = {
            'compute_capability': float, # 计算能力 (TFLOPS)
            'memory_bandwidth': float,   # 内存带宽 (GB/s)
            'memory_capacity': float,    # 内存容量 (GB)
            'power_budget': float,       # 功耗预算 (W)
            'supported_precisions': List[str], # 支持的精度格式
            'has_tensor_core': bool,     # 是否有Tensor Core
            'has_sparsity_support': bool, # 是否支持稀疏性加速
        }
        
        # 压缩状态 (Compression State)
        self.compression_state = {
            'pruning_ratios': List[float],  # 每层剪枝率
            'bit_widths': List[int],        # 每层位宽
            'distillation_progress': float, # 蒸馏进度 [0, 1]
            'fused_patterns': List[str],    # 已融合的算子模式
            'current_accuracy': float,      # 当前精度
            'current_latency': float,       # 当前延迟
            'current_energy': float,        # 当前能耗
        }
    
    def to_tensor(self) -> torch.Tensor:
        """将状态转换为张量，用于神经网络输入"""
        # 模型状态编码
        model_features = self._encode_model_state()
        
        # 硬件状态编码
        hardware_features = self._encode_hardware_state()
        
        # 压缩状态编码
        compression_features = self._encode_compression_state()
        
        # 拼接所有特征
        return torch.cat([model_features, hardware_features, compression_features])
    
    def _encode_model_state(self) -> torch.Tensor:
        """编码模型状态"""
        features = []
        
        # 层数（归一化）
        features.append(self.model_state['num_layers'] / 100)
        
        # 层类型（one-hot编码）
        layer_type_mapping = {'conv': 0, 'bn': 1, 'relu': 2, 'pool': 3, 'fc': 4}
        for layer_type in self.model_state['layer_types']:
            one_hot = [0] * len(layer_type_mapping)
            if layer_type in layer_type_mapping:
                one_hot[layer_type_mapping[layer_type]] = 1
            features.extend(one_hot)
        
        # 通道数（归一化）
        max_channels = max(self.model_state['channel_counts'])
        for channels in self.model_state['channel_counts']:
            features.append(channels / max_channels)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_hardware_state(self) -> torch.Tensor:
        """编码硬件状态"""
        features = []
        
        # 计算能力（归一化到[0, 1]，假设最大100 TFLOPS）
        features.append(self.hardware_state['compute_capability'] / 100)
        
        # 内存带宽（归一化到[0, 1]，假设最大3000 GB/s）
        features.append(self.hardware_state['memory_bandwidth'] / 3000)
        
        # 内存容量（归一化到[0, 1]，假设最大100 GB）
        features.append(self.hardware_state['memory_capacity'] / 100)
        
        # 功耗预算（归一化到[0, 1]，假设最大500 W）
        features.append(self.hardware_state['power_budget'] / 500)
        
        # 支持的精度格式（multi-hot编码）
        precision_mapping = {'FP32': 0, 'FP16': 1, 'INT8': 2, 'INT4': 3}
        precision_features = [0] * len(precision_mapping)
        for precision in self.hardware_state['supported_precisions']:
            if precision in precision_mapping:
                precision_features[precision_mapping[precision]] = 1
        features.extend(precision_features)
        
        # 特殊功能
        features.append(1 if self.hardware_state['has_tensor_core'] else 0)
        features.append(1 if self.hardware_state['has_sparsity_support'] else 0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_compression_state(self) -> torch.Tensor:
        """编码压缩状态"""
        features = []
        
        # 剪枝率
        features.extend(self.compression_state['pruning_ratios'])
        
        # 位宽（归一化到[0, 1]）
        for bw in self.compression_state['bit_widths']:
            features.append(bw / 32)
        
        # 蒸馏进度
        features.append(self.compression_state['distillation_progress'])
        
        # 当前性能指标
        features.append(self.compression_state['current_accuracy'])
        features.append(self.compression_state['current_latency'] / 100)  # 假设最大100ms
        features.append(self.compression_state['current_energy'] / 100)   # 假设最大100J
        
        return torch.tensor(features, dtype=torch.float32)
```

### 2.3 动作空间设计

每个智能体有自己的动作空间，动作空间可以是离散的或连续的。

```python
class ActionSpace:
    """MARL动作空间定义"""
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        
        # 剪枝智能体动作空间（离散）
        self.pruning_actions = {
            'layer_idx': range(num_layers),  # 选择哪一层
            'pruning_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
        
        # 量化智能体动作空间（离散）
        self.quantization_actions = {
            'layer_idx': range(num_layers),
            'bit_width': [4, 8, 16, 32],
        }
        
        # 蒸馏智能体动作空间（连续）
        self.distillation_actions = {
            'temperature': (1.0, 20.0),  # 温度范围
            'alpha': (0.0, 1.0),         # 蒸馏损失权重
        }
        
        # 融合智能体动作空间（离散）
        self.fusion_actions = {
            'pattern': [
                'none',           # 不融合
                'conv_bn',        # Conv + BN
                'conv_relu',      # Conv + ReLU
                'conv_bn_relu',   # Conv + BN + ReLU
                'conv_bn_add',    # Conv + BN + Add (残差)
                'conv_bn_add_relu', # Conv + BN + Add + ReLU
            ],
            'start_layer': range(num_layers),
        }
        
        # 更新智能体动作空间（离散）
        self.update_actions = {
            'strategy': ['full', 'incremental', 'hash_based'],
            'update_ratio': [0.1, 0.2, 0.3, 0.4, 0.5],  # 增量更新比例
        }
    
    def sample_pruning_action(self) -> dict:
        """采样剪枝动作"""
        return {
            'layer_idx': random.choice(list(self.pruning_actions['layer_idx'])),
            'pruning_ratio': random.choice(self.pruning_actions['pruning_ratio']),
        }
    
    def sample_quantization_action(self) -> dict:
        """采样量化动作"""
        return {
            'layer_idx': random.choice(list(self.quantization_actions['layer_idx'])),
            'bit_width': random.choice(self.quantization_actions['bit_width']),
        }
    
    def sample_distillation_action(self) -> dict:
        """采样蒸馏动作"""
        temp_range = self.distillation_actions['temperature']
        alpha_range = self.distillation_actions['alpha']
        return {
            'temperature': random.uniform(temp_range[0], temp_range[1]),
            'alpha': random.uniform(alpha_range[0], alpha_range[1]),
        }
    
    def sample_fusion_action(self) -> dict:
        """采样融合动作"""
        return {
            'pattern': random.choice(self.fusion_actions['pattern']),
            'start_layer': random.choice(list(self.fusion_actions['start_layer'])),
        }
    
    def sample_update_action(self) -> dict:
        """采样更新动作"""
        return {
            'strategy': random.choice(self.update_actions['strategy']),
            'update_ratio': random.choice(self.update_actions['update_ratio']),
        }
```

---

## 3. PPO控制器实现

### 3.1 策略网络设计

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """PPO策略网络"""
    
    def __init__(self, state_dim: int, action_dims: dict, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 剪枝智能体头
        self.pruning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['pruning']),
        )
        
        # 量化智能体头
        self.quantization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['quantization']),
        )
        
        # 蒸馏智能体头（输出均值和标准差）
        self.distillation_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['distillation']),
        )
        self.distillation_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['distillation']),
            nn.Softplus(),  # 确保标准差为正
        )
        
        # 融合智能体头
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['fusion']),
        )
        
        # 更新智能体头
        self.update_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['update']),
        )
    
    def forward(self, state: torch.Tensor) -> dict:
        """前向传播，返回所有智能体的动作分布"""
        # 提取共享特征
        features = self.feature_extractor(state)
        
        # 计算各智能体的动作分布
        pruning_logits = self.pruning_head(features)
        quantization_logits = self.quantization_head(features)
        distillation_mean = self.distillation_mean_head(features)
        distillation_std = self.distillation_std_head(features)
        fusion_logits = self.fusion_head(features)
        update_logits = self.update_head(features)
        
        return {
            'pruning': F.softmax(pruning_logits, dim=-1),
            'quantization': F.softmax(quantization_logits, dim=-1),
            'distillation_mean': distillation_mean,
            'distillation_std': distillation_std,
            'fusion': F.softmax(fusion_logits, dim=-1),
            'update': F.softmax(update_logits, dim=-1),
        }
    
    def sample_actions(self, state: torch.Tensor) -> tuple:
        """采样动作并计算对数概率"""
        distributions = self.forward(state)
        
        actions = {}
        log_probs = {}
        
        # 剪枝动作（离散）
        pruning_dist = torch.distributions.Categorical(distributions['pruning'])
        actions['pruning'] = pruning_dist.sample()
        log_probs['pruning'] = pruning_dist.log_prob(actions['pruning'])
        
        # 量化动作（离散）
        quantization_dist = torch.distributions.Categorical(distributions['quantization'])
        actions['quantization'] = quantization_dist.sample()
        log_probs['quantization'] = quantization_dist.log_prob(actions['quantization'])
        
        # 蒸馏动作（连续）
        distillation_dist = torch.distributions.Normal(
            distributions['distillation_mean'],
            distributions['distillation_std']
        )
        actions['distillation'] = distillation_dist.sample()
        log_probs['distillation'] = distillation_dist.log_prob(actions['distillation']).sum(dim=-1)
        
        # 融合动作（离散）
        fusion_dist = torch.distributions.Categorical(distributions['fusion'])
        actions['fusion'] = fusion_dist.sample()
        log_probs['fusion'] = fusion_dist.log_prob(actions['fusion'])
        
        # 更新动作（离散）
        update_dist = torch.distributions.Categorical(distributions['update'])
        actions['update'] = update_dist.sample()
        log_probs['update'] = update_dist.log_prob(actions['update'])
        
        # 总对数概率
        total_log_prob = sum(log_probs.values())
        
        return actions, total_log_prob


class ValueNetwork(nn.Module):
    """PPO价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回状态价值估计"""
        return self.network(state)
```

### 3.2 PPO训练算法

```python
class PPOController:
    """PPO控制器，协调所有智能体的训练"""
    
    def __init__(
        self,
        state_dim: int,
        action_dims: dict,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda',
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络
        self.policy_network = PolicyNetwork(state_dim, action_dims).to(device)
        self.value_network = ValueNetwork(state_dim).to(device)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = ExperienceBuffer()
    
    def select_actions(self, state: torch.Tensor) -> tuple:
        """选择动作"""
        with torch.no_grad():
            actions, log_prob = self.policy_network.sample_actions(state)
            value = self.value_network(state)
        return actions, log_prob, value
    
    def compute_gae(self, rewards: list, values: list, dones: list) -> tuple:
        """计算广义优势估计（GAE）"""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    
    def update(self, batch_size: int = 64, num_epochs: int = 10):
        """PPO更新"""
        # 从缓冲区获取数据
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get_all()
        
        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.stack(states).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # 多轮更新
        for _ in range(num_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的对数概率
                _, new_log_probs = self.policy_network.sample_actions(batch_states)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算裁剪后的目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                values = self.value_network(batch_states).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # 计算熵损失（鼓励探索）
                distributions = self.policy_network.forward(batch_states)
                entropy = self._compute_entropy(distributions)
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 更新策略网络
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # 更新价值网络
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
    
    def _compute_entropy(self, distributions: dict) -> torch.Tensor:
        """计算策略熵"""
        entropy = 0
        
        # 离散分布的熵
        for key in ['pruning', 'quantization', 'fusion', 'update']:
            dist = torch.distributions.Categorical(distributions[key])
            entropy += dist.entropy()
        
        # 连续分布的熵
        dist = torch.distributions.Normal(
            distributions['distillation_mean'],
            distributions['distillation_std']
        )
        entropy += dist.entropy().sum(dim=-1)
        
        return entropy
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])


class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, state, action, log_prob, reward, done, value):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def get_all(self):
        """获取所有经验"""
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
            self.values,
        )
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
```

---

## 4. 硬件抽象层实现

### 4.1 HAL接口定义

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json

class HardwareAbstractionLayer(ABC):
    """硬件抽象层基类"""
    
    @abstractmethod
    def get_compute_capability(self) -> float:
        """获取计算能力（TFLOPS）"""
        pass
    
    @abstractmethod
    def get_memory_bandwidth(self) -> float:
        """获取内存带宽（GB/s）"""
        pass
    
    @abstractmethod
    def get_memory_capacity(self) -> float:
        """获取内存容量（GB）"""
        pass
    
    @abstractmethod
    def get_power_budget(self) -> float:
        """获取功耗预算（W）"""
        pass
    
    @abstractmethod
    def get_supported_precisions(self) -> List[str]:
        """获取支持的精度格式"""
        pass
    
    @abstractmethod
    def measure_latency(self, model, input_tensor) -> float:
        """测量推理延迟（ms）"""
        pass
    
    @abstractmethod
    def measure_energy(self, model, input_tensor) -> float:
        """测量能耗（J）"""
        pass
    
    @abstractmethod
    def estimate_latency(self, model_config: dict) -> float:
        """使用LUT估计延迟（ms）"""
        pass


class NVIDIAHardwareAbstractionLayer(HardwareAbstractionLayer):
    """NVIDIA GPU的HAL实现"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载延迟查找表
        self.latency_lut = self._load_latency_lut()
    
    def _load_latency_lut(self) -> dict:
        """加载延迟查找表"""
        lut_path = self.config.get('latency_lut_path', 'latency_lut.json')
        try:
            with open(lut_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def get_compute_capability(self) -> float:
        return self.config['compute_capability']
    
    def get_memory_bandwidth(self) -> float:
        return self.config['memory_bandwidth']
    
    def get_memory_capacity(self) -> float:
        return self.config['memory_capacity']
    
    def get_power_budget(self) -> float:
        return self.config['power_budget']
    
    def get_supported_precisions(self) -> List[str]:
        return self.config['supported_precisions']
    
    def measure_latency(self, model, input_tensor) -> float:
        """实际测量推理延迟"""
        import torch
        import time
        
        model.eval()
        model.cuda()
        input_tensor = input_tensor.cuda()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # 同步
        torch.cuda.synchronize()
        
        # 测量
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()
        
        return (end - start) / num_iterations * 1000  # 转换为毫秒
    
    def measure_energy(self, model, input_tensor) -> float:
        """测量能耗（需要NVIDIA SMI支持）"""
        import subprocess
        import torch
        
        # 获取初始功耗
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        initial_power = float(result.stdout.strip())
        
        # 运行推理
        latency = self.measure_latency(model, input_tensor)
        
        # 获取最终功耗
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        final_power = float(result.stdout.strip())
        
        # 估算能耗
        avg_power = (initial_power + final_power) / 2
        energy = avg_power * (latency / 1000)  # W * s = J
        
        return energy
    
    def estimate_latency(self, model_config: dict) -> float:
        """使用LUT估计延迟"""
        total_latency = 0
        
        for layer_config in model_config['layers']:
            layer_type = layer_config['type']
            layer_params = layer_config['params']
            
            # 构建LUT键
            key = self._build_lut_key(layer_type, layer_params)
            
            if key in self.latency_lut:
                total_latency += self.latency_lut[key]
            else:
                # 如果LUT中没有，使用插值估计
                total_latency += self._interpolate_latency(layer_type, layer_params)
        
        return total_latency
    
    def _build_lut_key(self, layer_type: str, params: dict) -> str:
        """构建LUT键"""
        if layer_type == 'conv':
            return f"conv_{params['in_channels']}_{params['out_channels']}_{params['kernel_size']}_{params['stride']}_{params['precision']}"
        elif layer_type == 'fc':
            return f"fc_{params['in_features']}_{params['out_features']}_{params['precision']}"
        else:
            return f"{layer_type}_{hash(str(params))}"
    
    def _interpolate_latency(self, layer_type: str, params: dict) -> float:
        """插值估计延迟"""
        # 简化的延迟估计模型
        if layer_type == 'conv':
            flops = params['in_channels'] * params['out_channels'] * params['kernel_size'] ** 2
            return flops / (self.get_compute_capability() * 1e12) * 1000  # ms
        elif layer_type == 'fc':
            flops = params['in_features'] * params['out_features']
            return flops / (self.get_compute_capability() * 1e12) * 1000  # ms
        else:
            return 0.1  # 默认值
```

### 4.2 硬件配置文件

```json
// nvidia_a100.json
{
    "platform": "NVIDIA_A100",
    "vendor": "NVIDIA",
    "compute_capability": 19.5,
    "memory_bandwidth": 2039,
    "memory_capacity": 80,
    "power_budget": 400,
    "supported_precisions": ["FP32", "FP16", "BF16", "INT8", "INT4"],
    "has_tensor_core": true,
    "has_sparsity_support": true,
    "latency_lut_path": "lut/nvidia_a100_lut.json"
}

// nvidia_rtx3090.json
{
    "platform": "NVIDIA_RTX3090",
    "vendor": "NVIDIA",
    "compute_capability": 35.6,
    "memory_bandwidth": 936,
    "memory_capacity": 24,
    "power_budget": 350,
    "supported_precisions": ["FP32", "FP16", "INT8"],
    "has_tensor_core": true,
    "has_sparsity_support": false,
    "latency_lut_path": "lut/nvidia_rtx3090_lut.json"
}

// ascend_310.json
{
    "platform": "Ascend_310",
    "vendor": "Huawei",
    "compute_capability": 22.0,
    "memory_bandwidth": 256,
    "memory_capacity": 16,
    "power_budget": 8,
    "supported_precisions": ["FP32", "FP16", "INT8"],
    "has_tensor_core": false,
    "has_sparsity_support": false,
    "latency_lut_path": "lut/ascend_310_lut.json"
}
```

---

## 5. 专用推理引擎实现

### 5.1 DIE核心类

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class DedicatedInferenceEngine:
    """专用推理引擎"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.original_model = model
        self.config = config
        self.optimized_model = None
        
        # 稀疏性配置
        self.sparsity_config = config.get('sparsity', {})
        
        # 混合精度配置
        self.precision_config = config.get('precision', {})
        
        # 融合配置
        self.fusion_config = config.get('fusion', {})
    
    def optimize(self):
        """优化模型"""
        self.optimized_model = self.original_model
        
        # 应用算子融合
        if self.fusion_config.get('enabled', False):
            self.optimized_model = self._apply_fusion(self.optimized_model)
        
        # 应用稀疏性优化
        if self.sparsity_config.get('enabled', False):
            self.optimized_model = self._apply_sparsity(self.optimized_model)
        
        # 应用混合精度
        if self.precision_config.get('enabled', False):
            self.optimized_model = self._apply_mixed_precision(self.optimized_model)
        
        return self.optimized_model
    
    def _apply_fusion(self, model: nn.Module) -> nn.Module:
        """应用算子融合"""
        fusion_patterns = self.fusion_config.get('patterns', [])
        
        for pattern in fusion_patterns:
            if pattern == 'conv_bn':
                model = self._fuse_conv_bn(model)
            elif pattern == 'conv_bn_relu':
                model = self._fuse_conv_bn_relu(model)
            elif pattern == 'conv_relu':
                model = self._fuse_conv_relu(model)
        
        return model
    
    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """融合Conv和BN"""
        import torch.nn.utils.fusion as fusion
        
        # 遍历模型，找到连续的Conv-BN对
        modules = list(model.named_modules())
        fused_modules = {}
        skip_next = False
        
        for i, (name, module) in enumerate(modules):
            if skip_next:
                skip_next = False
                continue
            
            if isinstance(module, nn.Conv2d) and i + 1 < len(modules):
                next_name, next_module = modules[i + 1]
                if isinstance(next_module, nn.BatchNorm2d):
                    # 融合Conv和BN
                    fused = fusion.fuse_conv_bn_eval(module, next_module)
                    fused_modules[name] = fused
                    skip_next = True
        
        # 替换模块
        for name, fused_module in fused_modules.items():
            self._set_module(model, name, fused_module)
        
        return model
    
    def _fuse_conv_bn_relu(self, model: nn.Module) -> nn.Module:
        """融合Conv、BN和ReLU"""
        # 类似于_fuse_conv_bn，但还需要处理ReLU
        # 这里使用自定义的融合模块
        
        class FusedConvBNReLU(nn.Module):
            def __init__(self, conv, bn):
                super().__init__()
                self.fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            
            def forward(self, x):
                return torch.relu(self.fused_conv(x))
        
        # 遍历模型，找到连续的Conv-BN-ReLU序列
        # ... 实现类似于_fuse_conv_bn
        
        return model
    
    def _fuse_conv_relu(self, model: nn.Module) -> nn.Module:
        """融合Conv和ReLU"""
        class FusedConvReLU(nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv
            
            def forward(self, x):
                return torch.relu(self.conv(x))
        
        # ... 实现
        return model
    
    def _apply_sparsity(self, model: nn.Module) -> nn.Module:
        """应用稀疏性优化"""
        sparsity_pattern = self.sparsity_config.get('pattern', '2:4')
        
        if sparsity_pattern == '2:4':
            model = self._apply_2_4_sparsity(model)
        elif sparsity_pattern == 'unstructured':
            model = self._apply_unstructured_sparsity(model)
        
        return model
    
    def _apply_2_4_sparsity(self, model: nn.Module) -> nn.Module:
        """应用2:4结构化稀疏"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # 对每4个元素，保留绝对值最大的2个
                weight_flat = weight.view(-1, 4)
                _, indices = torch.topk(torch.abs(weight_flat), k=2, dim=1)
                mask = torch.zeros_like(weight_flat)
                mask.scatter_(1, indices, 1)
                weight_flat *= mask
                
                module.weight.data = weight_flat.view(weight.shape)
        
        return model
    
    def _apply_unstructured_sparsity(self, model: nn.Module) -> nn.Module:
        """应用非结构化稀疏"""
        sparsity_ratio = self.sparsity_config.get('ratio', 0.5)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), sparsity_ratio)
                mask = torch.abs(weight) >= threshold
                module.weight.data *= mask.float()
        
        return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """应用混合精度"""
        layer_precisions = self.precision_config.get('layer_precisions', {})
        
        for name, module in model.named_modules():
            if name in layer_precisions:
                precision = layer_precisions[name]
                if precision == 'FP16':
                    module.half()
                elif precision == 'INT8':
                    module = torch.quantization.quantize_dynamic(
                        module, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
        
        return model
    
    def _set_module(self, model: nn.Module, name: str, module: nn.Module):
        """设置模型中的模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)
    
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行推理"""
        if self.optimized_model is None:
            self.optimize()
        
        self.optimized_model.eval()
        with torch.no_grad():
            output = self.optimized_model(input_tensor)
        
        return output
```

---

## 6. 五个智能体详细设计

### 6.1 剪枝智能体

```python
class PruningAgent:
    """剪枝智能体"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.pruning_ratios = {}
    
    def get_action_space(self) -> dict:
        """获取动作空间"""
        num_layers = len(list(self.model.modules()))
        return {
            'layer_idx': list(range(num_layers)),
            'pruning_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    
    def apply_action(self, action: dict):
        """应用剪枝动作"""
        layer_idx = action['layer_idx']
        pruning_ratio = action['pruning_ratio']
        
        # 获取目标层
        layers = list(self.model.modules())
        if layer_idx >= len(layers):
            return
        
        target_layer = layers[layer_idx]
        
        # 只对Conv和Linear层进行剪枝
        if not isinstance(target_layer, (nn.Conv2d, nn.Linear)):
            return
        
        # 应用剪枝
        self._prune_layer(target_layer, pruning_ratio)
        self.pruning_ratios[layer_idx] = pruning_ratio
    
    def _prune_layer(self, layer: nn.Module, ratio: float):
        """对单层进行剪枝"""
        if ratio == 0:
            return
        
        weight = layer.weight.data
        
        # 计算L1范数
        if isinstance(layer, nn.Conv2d):
            importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
        else:
            importance = torch.sum(torch.abs(weight), dim=1)
        
        # 确定要剪枝的通道数
        num_prune = int(len(importance) * ratio)
        
        # 获取要剪枝的通道索引
        _, prune_indices = torch.topk(importance, num_prune, largest=False)
        
        # 将这些通道的权重置零
        if isinstance(layer, nn.Conv2d):
            weight[prune_indices, :, :, :] = 0
        else:
            weight[prune_indices, :] = 0
        
        layer.weight.data = weight
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'pruning_ratios': self.pruning_ratios.copy(),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'nonzero_params': sum((p != 0).sum().item() for p in self.model.parameters()),
        }
```

### 6.2 量化智能体

```python
class QuantizationAgent:
    """量化智能体"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.bit_widths = {}
    
    def get_action_space(self) -> dict:
        """获取动作空间"""
        num_layers = len(list(self.model.modules()))
        return {
            'layer_idx': list(range(num_layers)),
            'bit_width': [4, 8, 16, 32],
        }
    
    def apply_action(self, action: dict):
        """应用量化动作"""
        layer_idx = action['layer_idx']
        bit_width = action['bit_width']
        
        # 获取目标层
        layers = list(self.model.modules())
        if layer_idx >= len(layers):
            return
        
        target_layer = layers[layer_idx]
        
        # 只对Conv和Linear层进行量化
        if not isinstance(target_layer, (nn.Conv2d, nn.Linear)):
            return
        
        # 应用量化
        self._quantize_layer(target_layer, bit_width)
        self.bit_widths[layer_idx] = bit_width
    
    def _quantize_layer(self, layer: nn.Module, bit_width: int):
        """对单层进行量化"""
        if bit_width == 32:
            return  # FP32不需要量化
        
        weight = layer.weight.data
        
        # 计算量化参数
        min_val = weight.min()
        max_val = weight.max()
        scale = (max_val - min_val) / (2 ** bit_width - 1)
        zero_point = -min_val / scale
        
        # 量化
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 2 ** bit_width - 1)
        
        # 反量化
        dequantized = (quantized - zero_point) * scale
        
        layer.weight.data = dequantized
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'bit_widths': self.bit_widths.copy(),
            'avg_bit_width': sum(self.bit_widths.values()) / len(self.bit_widths) if self.bit_widths else 32,
        }
```

### 6.3 蒸馏智能体

```python
class DistillationAgent:
    """蒸馏智能体"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, config: dict):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.temperature = 4.0
        self.alpha = 0.5
    
    def get_action_space(self) -> dict:
        """获取动作空间"""
        return {
            'temperature': (1.0, 20.0),
            'alpha': (0.0, 1.0),
        }
    
    def apply_action(self, action: dict):
        """应用蒸馏动作"""
        self.temperature = action['temperature']
        self.alpha = action['alpha']
    
    def compute_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        # 教师模型输出
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        
        # 学生模型输出
        student_outputs = self.student_model(inputs)
        
        # 软标签损失
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)
        
        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
        }
```

### 6.4 融合智能体

```python
class FusionAgent:
    """融合智能体"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.fused_patterns = []
    
    def get_action_space(self) -> dict:
        """获取动作空间"""
        num_layers = len(list(self.model.modules()))
        return {
            'pattern': ['none', 'conv_bn', 'conv_relu', 'conv_bn_relu', 'conv_bn_add', 'conv_bn_add_relu'],
            'start_layer': list(range(num_layers)),
        }
    
    def apply_action(self, action: dict):
        """应用融合动作"""
        pattern = action['pattern']
        start_layer = action['start_layer']
        
        if pattern == 'none':
            return
        
        # 应用融合
        success = self._apply_fusion(pattern, start_layer)
        if success:
            self.fused_patterns.append({'pattern': pattern, 'start_layer': start_layer})
    
    def _apply_fusion(self, pattern: str, start_layer: int) -> bool:
        """应用特定的融合模式"""
        modules = list(self.model.named_modules())
        
        if start_layer >= len(modules):
            return False
        
        if pattern == 'conv_bn':
            return self._fuse_conv_bn(modules, start_layer)
        elif pattern == 'conv_relu':
            return self._fuse_conv_relu(modules, start_layer)
        elif pattern == 'conv_bn_relu':
            return self._fuse_conv_bn_relu(modules, start_layer)
        
        return False
    
    def _fuse_conv_bn(self, modules: list, start_layer: int) -> bool:
        """融合Conv和BN"""
        if start_layer + 1 >= len(modules):
            return False
        
        name1, module1 = modules[start_layer]
        name2, module2 = modules[start_layer + 1]
        
        if not isinstance(module1, nn.Conv2d) or not isinstance(module2, nn.BatchNorm2d):
            return False
        
        # 执行融合
        fused = torch.nn.utils.fusion.fuse_conv_bn_eval(module1, module2)
        
        # 替换模块
        self._set_module(self.model, name1, fused)
        self._set_module(self.model, name2, nn.Identity())
        
        return True
    
    def _fuse_conv_relu(self, modules: list, start_layer: int) -> bool:
        """融合Conv和ReLU"""
        # 类似实现
        return False
    
    def _fuse_conv_bn_relu(self, modules: list, start_layer: int) -> bool:
        """融合Conv、BN和ReLU"""
        # 类似实现
        return False
    
    def _set_module(self, model: nn.Module, name: str, module: nn.Module):
        """设置模型中的模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'fused_patterns': self.fused_patterns.copy(),
            'num_fusions': len(self.fused_patterns),
        }
```

### 6.5 更新智能体

```python
class UpdateAgent:
    """更新智能体"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.update_strategy = 'full'
        self.update_ratio = 1.0
        self.hash_table = {}
    
    def get_action_space(self) -> dict:
        """获取动作空间"""
        return {
            'strategy': ['full', 'incremental', 'hash_based'],
            'update_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    
    def apply_action(self, action: dict):
        """应用更新动作"""
        self.update_strategy = action['strategy']
        self.update_ratio = action['update_ratio']
    
    def update_model(self, new_data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
        """更新模型"""
        if self.update_strategy == 'full':
            self._full_update(new_data, optimizer)
        elif self.update_strategy == 'incremental':
            self._incremental_update(new_data, optimizer)
        elif self.update_strategy == 'hash_based':
            self._hash_based_update(new_data, optimizer)
    
    def _full_update(self, new_data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
        """完整更新"""
        self.model.train()
        for inputs, labels in new_data:
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
    
    def _incremental_update(self, new_data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
        """增量更新"""
        # 只更新部分参数
        params_to_update = []
        for name, param in self.model.named_parameters():
            if torch.rand(1).item() < self.update_ratio:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        
        # 创建新的优化器
        incremental_optimizer = torch.optim.Adam(params_to_update, lr=optimizer.defaults['lr'])
        
        self.model.train()
        for inputs, labels in new_data:
            incremental_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            incremental_optimizer.step()
        
        # 恢复所有参数的requires_grad
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _hash_based_update(self, new_data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
        """基于哈希的更新"""
        # 计算新数据的哈希
        for inputs, labels in new_data:
            data_hash = self._compute_hash(inputs)
            
            if data_hash in self.hash_table:
                # 数据已见过，跳过
                continue
            
            # 新数据，进行更新
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 记录哈希
            self.hash_table[data_hash] = True
    
    def _compute_hash(self, tensor: torch.Tensor) -> str:
        """计算张量的哈希"""
        import hashlib
        return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            'update_strategy': self.update_strategy,
            'update_ratio': self.update_ratio,
            'hash_table_size': len(self.hash_table),
        }
```

---

## 7. 奖励函数设计

### 7.1 多目标奖励函数

```python
class RewardFunction:
    """多目标奖励函数"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 权重系数
        self.alpha = config.get('alpha', 0.5)  # 精度权重
        self.beta = config.get('beta', 0.3)    # 延迟权重
        self.gamma = config.get('gamma', 0.2)  # 能耗权重
        
        # 基线值
        self.baseline_accuracy = config.get('baseline_accuracy', 0.95)
        self.baseline_latency = config.get('baseline_latency', 10.0)  # ms
        self.baseline_energy = config.get('baseline_energy', 1.0)     # J
        
        # 约束
        self.min_accuracy = config.get('min_accuracy', 0.90)
        self.max_latency = config.get('max_latency', 5.0)  # ms
        self.max_energy = config.get('max_energy', 0.5)    # J
    
    def compute(self, accuracy: float, latency: float, energy: float) -> float:
        """计算奖励"""
        # 归一化
        norm_accuracy = accuracy / self.baseline_accuracy
        norm_latency = 1 - latency / self.baseline_latency
        norm_energy = 1 - energy / self.baseline_energy
        
        # 基础奖励
        base_reward = (
            self.alpha * norm_accuracy +
            self.beta * norm_latency +
            self.gamma * norm_energy
        )
        
        # 约束惩罚
        penalty = 0
        if accuracy < self.min_accuracy:
            penalty += 10 * (self.min_accuracy - accuracy)
        if latency > self.max_latency:
            penalty += 10 * (latency - self.max_latency) / self.max_latency
        if energy > self.max_energy:
            penalty += 10 * (energy - self.max_energy) / self.max_energy
        
        # 总奖励
        total_reward = base_reward - penalty
        
        return total_reward
    
    def compute_pareto_reward(self, accuracy: float, latency: float, energy: float, pareto_front: list) -> float:
        """计算基于Pareto前沿的奖励"""
        # 检查是否在Pareto前沿上
        is_pareto = self._is_pareto_optimal(accuracy, latency, energy, pareto_front)
        
        if is_pareto:
            # 在Pareto前沿上，给予额外奖励
            return self.compute(accuracy, latency, energy) + 1.0
        else:
            # 不在Pareto前沿上，计算到前沿的距离
            distance = self._distance_to_pareto(accuracy, latency, energy, pareto_front)
            return self.compute(accuracy, latency, energy) - 0.1 * distance
    
    def _is_pareto_optimal(self, accuracy: float, latency: float, energy: float, pareto_front: list) -> bool:
        """检查是否是Pareto最优"""
        for point in pareto_front:
            if (point['accuracy'] >= accuracy and 
                point['latency'] <= latency and 
                point['energy'] <= energy and
                (point['accuracy'] > accuracy or point['latency'] < latency or point['energy'] < energy)):
                return False
        return True
    
    def _distance_to_pareto(self, accuracy: float, latency: float, energy: float, pareto_front: list) -> float:
        """计算到Pareto前沿的距离"""
        min_distance = float('inf')
        for point in pareto_front:
            distance = (
                (accuracy - point['accuracy']) ** 2 +
                (latency - point['latency']) ** 2 +
                (energy - point['energy']) ** 2
            ) ** 0.5
            min_distance = min(min_distance, distance)
        return min_distance
```

---

## 8. 训练流程

### 8.1 完整训练流程

```python
class HADMC2Trainer:
    """HAD-MC 2.0训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 初始化组件
        self.ppo_controller = PPOController(
            state_dim=config['state_dim'],
            action_dims=config['action_dims'],
            lr=config.get('lr', 3e-4),
            device=config.get('device', 'cuda'),
        )
        
        self.hal = NVIDIAHardwareAbstractionLayer(config['hal_config_path'])
        
        self.reward_function = RewardFunction(config['reward_config'])
        
        # 训练参数
        self.num_episodes = config.get('num_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 100)
        self.update_interval = config.get('update_interval', 10)
    
    def train(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        """训练主循环"""
        best_reward = float('-inf')
        pareto_front = []
        
        for episode in range(self.num_episodes):
            # 重置环境
            current_model = copy.deepcopy(model)
            state = self._get_state(current_model)
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                # 选择动作
                state_tensor = state.to_tensor().unsqueeze(0).to(self.config['device'])
                actions, log_prob, value = self.ppo_controller.select_actions(state_tensor)
                
                # 执行动作
                current_model = self._apply_actions(current_model, actions)
                
                # 评估
                accuracy = self._evaluate_accuracy(current_model, val_loader)
                latency = self.hal.measure_latency(current_model, self._get_sample_input())
                energy = self.hal.measure_energy(current_model, self._get_sample_input())
                
                # 计算奖励
                reward = self.reward_function.compute_pareto_reward(accuracy, latency, energy, pareto_front)
                
                # 更新Pareto前沿
                pareto_front = self._update_pareto_front(pareto_front, accuracy, latency, energy)
                
                # 获取新状态
                next_state = self._get_state(current_model)
                done = step == self.max_steps_per_episode - 1
                
                # 存储经验
                self.ppo_controller.buffer.add(
                    state_tensor.squeeze(0),
                    actions,
                    log_prob,
                    reward,
                    done,
                    value.item(),
                )
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # 更新策略
            if (episode + 1) % self.update_interval == 0:
                self.ppo_controller.update()
            
            # 记录
            print(f"Episode {episode + 1}/{self.num_episodes}, Reward: {episode_reward:.4f}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_best_model(current_model)
        
        return self._load_best_model()
    
    def _get_state(self, model: nn.Module) -> State:
        """获取当前状态"""
        state = State()
        
        # 填充模型状态
        state.model_state['num_layers'] = len(list(model.modules()))
        # ... 其他状态
        
        # 填充硬件状态
        state.hardware_state['compute_capability'] = self.hal.get_compute_capability()
        # ... 其他状态
        
        return state
    
    def _apply_actions(self, model: nn.Module, actions: dict) -> nn.Module:
        """应用所有智能体的动作"""
        # 创建智能体
        pruning_agent = PruningAgent(model, {})
        quantization_agent = QuantizationAgent(model, {})
        # ... 其他智能体
        
        # 应用动作
        pruning_agent.apply_action(self._decode_pruning_action(actions['pruning']))
        quantization_agent.apply_action(self._decode_quantization_action(actions['quantization']))
        # ... 其他动作
        
        return model
    
    def _evaluate_accuracy(self, model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
        """评估模型精度"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
    
    def _update_pareto_front(self, pareto_front: list, accuracy: float, latency: float, energy: float) -> list:
        """更新Pareto前沿"""
        new_point = {'accuracy': accuracy, 'latency': latency, 'energy': energy}
        
        # 移除被新点支配的点
        pareto_front = [
            p for p in pareto_front
            if not (accuracy >= p['accuracy'] and latency <= p['latency'] and energy <= p['energy'] and
                   (accuracy > p['accuracy'] or latency < p['latency'] or energy < p['energy']))
        ]
        
        # 检查新点是否被支配
        is_dominated = any(
            p['accuracy'] >= accuracy and p['latency'] <= latency and p['energy'] <= energy and
            (p['accuracy'] > accuracy or p['latency'] < latency or p['energy'] < energy)
            for p in pareto_front
        )
        
        if not is_dominated:
            pareto_front.append(new_point)
        
        return pareto_front
    
    def _get_sample_input(self) -> torch.Tensor:
        """获取样本输入"""
        return torch.randn(1, 3, 640, 640).cuda()
    
    def _save_best_model(self, model: nn.Module):
        """保存最佳模型"""
        torch.save(model.state_dict(), 'best_model.pth')
    
    def _load_best_model(self) -> nn.Module:
        """加载最佳模型"""
        model = self._create_model()
        model.load_state_dict(torch.load('best_model.pth'))
        return model
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        # 根据配置创建模型
        pass
    
    def _decode_pruning_action(self, action: torch.Tensor) -> dict:
        """解码剪枝动作"""
        action_idx = action.item()
        # 解码逻辑
        return {'layer_idx': action_idx // 10, 'pruning_ratio': (action_idx % 10) / 10}
    
    def _decode_quantization_action(self, action: torch.Tensor) -> dict:
        """解码量化动作"""
        action_idx = action.item()
        bit_widths = [4, 8, 16, 32]
        return {'layer_idx': action_idx // 4, 'bit_width': bit_widths[action_idx % 4]}
```

---

## 9. 代码架构

### 9.1 目录结构

```
hadmc/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── pruning_agent.py
│   ├── quantization_agent.py
│   ├── distillation_agent.py
│   ├── fusion_agent.py
│   └── update_agent.py
├── marl/
│   ├── __init__.py
│   ├── state.py
│   ├── action.py
│   ├── reward.py
│   ├── policy_network.py
│   ├── value_network.py
│   ├── ppo_controller.py
│   └── experience_buffer.py
├── hal/
│   ├── __init__.py
│   ├── base.py
│   ├── nvidia.py
│   ├── ascend.py
│   ├── hygon.py
│   └── configs/
│       ├── nvidia_a100.json
│       ├── nvidia_rtx3090.json
│       ├── ascend_310.json
│       └── hygon_dcu.json
├── die/
│   ├── __init__.py
│   ├── engine.py
│   ├── sparsity.py
│   ├── mixed_precision.py
│   └── fusion.py
├── trainer/
│   ├── __init__.py
│   └── hadmc_trainer.py
├── utils/
│   ├── __init__.py
│   ├── model_utils.py
│   ├── data_utils.py
│   └── visualization.py
└── configs/
    ├── default.yaml
    └── experiments/
        ├── coco.yaml
        ├── voc.yaml
        └── neudet.yaml
```

---

## 10. 测试与验证

### 10.1 单元测试

```python
import unittest
import torch

class TestPruningAgent(unittest.TestCase):
    def setUp(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.agent = PruningAgent(self.model, {})
    
    def test_action_space(self):
        action_space = self.agent.get_action_space()
        self.assertIn('layer_idx', action_space)
        self.assertIn('pruning_ratio', action_space)
    
    def test_apply_action(self):
        action = {'layer_idx': 0, 'pruning_ratio': 0.5}
        self.agent.apply_action(action)
        state = self.agent.get_state()
        self.assertEqual(state['pruning_ratios'].get(0, 0), 0.5)


class TestQuantizationAgent(unittest.TestCase):
    def setUp(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.agent = QuantizationAgent(self.model, {})
    
    def test_action_space(self):
        action_space = self.agent.get_action_space()
        self.assertIn('layer_idx', action_space)
        self.assertIn('bit_width', action_space)
    
    def test_apply_action(self):
        action = {'layer_idx': 0, 'bit_width': 8}
        self.agent.apply_action(action)
        state = self.agent.get_state()
        self.assertEqual(state['bit_widths'].get(0, 32), 8)


class TestPPOController(unittest.TestCase):
    def setUp(self):
        self.controller = PPOController(
            state_dim=100,
            action_dims={'pruning': 10, 'quantization': 4, 'distillation': 2, 'fusion': 6, 'update': 3},
        )
    
    def test_select_actions(self):
        state = torch.randn(1, 100)
        actions, log_prob, value = self.controller.select_actions(state)
        self.assertIn('pruning', actions)
        self.assertIn('quantization', actions)


if __name__ == '__main__':
    unittest.main()
```

---

**文档结束**

*本文档由12位教授级专家经过12轮讨论后联合撰写，旨在为HAD-MC算法升级提供全面、详细的指导。*

        max_grad_norm: float = 0.5,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络
        self.policy_net = PolicyNetwork(state_dim, action_dims).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = []
    
    def select_action(self, state: torch.Tensor) -> tuple:
        """选择动作"""
        with torch.no_grad():
            actions, log_prob = self.policy_net.sample_actions(state)
            value = self.value_net(state)
        return actions, log_prob, value
    
    def store_transition(self, state, actions, log_prob, reward, next_state, done, value):
        """存储转换"""
        self.buffer.append({
            'state': state,
            'actions': actions,
            'log_prob': log_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value,
        })
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, num_epochs: int = 10, batch_size: int = 64):
        """PPO更新"""
        if len(self.buffer) < batch_size:
            return {}
        
        # 准备数据
        states = torch.stack([t['state'] for t in self.buffer]).to(self.device)
        old_log_probs = torch.stack([t['log_prob'] for t in self.buffer]).to(self.device)
        rewards = [t['reward'] for t in self.buffer]
        values = [t['value'].item() for t in self.buffer]
        dones = [t['done'] for t in self.buffer]
        
        # 计算GAE
        with torch.no_grad():
            next_value = self.value_net(self.buffer[-1]['next_state']).item()
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(num_epochs):
            # 随机打乱
            indices = torch.randperm(len(self.buffer))
            
            for start in range(0, len(self.buffer), batch_size):
                end = min(start + batch_size, len(self.buffer))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的动作概率
                _, new_log_probs = self.policy_net.sample_actions(batch_states)
                
                # 计算概率比
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算裁剪的目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                new_values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 计算熵奖励（鼓励探索）
                distributions = self.policy_net.forward(batch_states)
                entropy = 0
                for key in ['pruning', 'quantization', 'fusion', 'update']:
                    dist = torch.distributions.Categorical(distributions[key])
                    entropy += dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 更新策略网络
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # 更新价值网络
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # 清空缓冲区
        self.buffer = []
        
        num_updates = num_epochs * (len(self.buffer) // batch_size + 1)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }

---

## 4. 硬件抽象层实现

### 4.1 HAL架构

```python
class HardwareAbstractionLayer:
    """硬件抽象层（HAL）"""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.config = self._load_platform_config(platform)
        self.latency_lut = self._build_latency_lut()
    
    def _load_platform_config(self, platform: str) -> dict:
        """加载平台配置"""
        configs = {
            'nvidia_a100': {
                'compute_capability': 156.0,  # TFLOPS (FP16)
                'memory_bandwidth': 2039,     # GB/s
                'memory_capacity': 80,        # GB
                'power_budget': 400,          # W
                'supported_precisions': ['FP32', 'FP16', 'INT8', 'INT4'],
                'has_tensor_core': True,
                'has_sparsity_support': True,
                'inference_engine': 'tensorrt',
            },
            'jetson_orin': {
                'compute_capability': 275.0,  # TOPS (INT8)
                'memory_bandwidth': 204.8,    # GB/s
                'memory_capacity': 64,        # GB
                'power_budget': 60,           # W
                'supported_precisions': ['FP32', 'FP16', 'INT8'],
                'has_tensor_core': True,
                'has_sparsity_support': True,
                'inference_engine': 'tensorrt',
            },
            'ascend_310': {
                'compute_capability': 22.0,   # TOPS (INT8)
                'memory_bandwidth': 25.6,     # GB/s
                'memory_capacity': 8,         # GB
                'power_budget': 8,            # W
                'supported_precisions': ['FP32', 'FP16', 'INT8'],
                'has_tensor_core': False,
                'has_sparsity_support': False,
                'inference_engine': 'acl',
            },
            'hygon_z100': {
                'compute_capability': 32.0,   # TFLOPS (FP32)
                'memory_bandwidth': 1024,     # GB/s
                'memory_capacity': 32,        # GB
                'power_budget': 300,          # W
                'supported_precisions': ['FP32', 'FP16'],
                'has_tensor_core': False,
                'has_sparsity_support': False,
                'inference_engine': 'rocm',
            },
        }
        return configs.get(platform, configs['nvidia_a100'])
    
    def _build_latency_lut(self) -> dict:
        """构建延迟查找表"""
        # 这里应该通过实际测量构建
        # 简化版本使用预设值
        lut = {}
        
        # 卷积层延迟模型 (ms)
        # latency = alpha * FLOPs / compute_capability + beta * memory_access / bandwidth
        lut['conv'] = lambda flops, mem: (
            flops / (self.config['compute_capability'] * 1e12) * 1000 +
            mem / (self.config['memory_bandwidth'] * 1e9) * 1000
        )
        
        # 全连接层延迟模型
        lut['fc'] = lambda flops, mem: (
            flops / (self.config['compute_capability'] * 1e12) * 1000 +
            mem / (self.config['memory_bandwidth'] * 1e9) * 1000
        )
        
        return lut
    
    def estimate_latency(self, model_config: dict) -> float:
        """估计模型延迟"""
        total_latency = 0
        
        for layer in model_config['layers']:
            layer_type = layer['type']
            flops = layer['flops']
            memory_access = layer['memory_access']
            
            if layer_type in self.lut:
                total_latency += self.lut[layer_type](flops, memory_access)
        
        return total_latency
    
    def get_optimal_precision(self, layer_type: str, accuracy_requirement: float) -> str:
        """获取最优精度"""
        precisions = self.config['supported_precisions']
        
        # 优先使用低精度（如果支持）
        if accuracy_requirement > 0.95 and 'FP32' in precisions:
            return 'FP32'
        elif accuracy_requirement > 0.90 and 'FP16' in precisions:
            return 'FP16'
        elif 'INT8' in precisions:
            return 'INT8'
        else:
            return precisions[0]
    
    def get_hardware_state(self) -> dict:
        """获取硬件状态（用于MARL状态空间）"""
        return {
            'compute_capability': self.config['compute_capability'],
            'memory_bandwidth': self.config['memory_bandwidth'],
            'memory_capacity': self.config['memory_capacity'],
            'power_budget': self.config['power_budget'],
            'supported_precisions': self.config['supported_precisions'],
            'has_tensor_core': self.config['has_tensor_core'],
            'has_sparsity_support': self.config['has_sparsity_support'],
        }
```

### 4.2 延迟查找表详细实现

```python
class LatencyLookupTable:
    """延迟查找表（LUT）"""
    
    def __init__(self, platform: str, model_name: str):
        self.platform = platform
        self.model_name = model_name
        self.lut = {}
        self.profiler = HardwareProfiler(platform)
    
    def build(self, model: nn.Module, input_shape: tuple, num_samples: int = 100):
        """构建延迟查找表"""
        print(f"Building latency LUT for {self.model_name} on {self.platform}...")
        
        # 遍历所有层
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                # 获取层配置
                config = self._get_layer_config(name, module)
                
                # 测量不同量化位宽的延迟
                for bit_width in [4, 8, 16, 32]:
                    latency = self._measure_latency(module, input_shape, bit_width, num_samples)
                    key = (name, bit_width)
                    self.lut[key] = latency
                    print(f"  {name} (W{bit_width}): {latency:.4f} ms")
        
        print(f"LUT built with {len(self.lut)} entries")
    
    def _get_layer_config(self, name: str, module: nn.Module) -> dict:
        """获取层配置"""
        if isinstance(module, nn.Conv2d):
            return {
                'type': 'conv2d',
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'groups': module.groups,
            }
        elif isinstance(module, nn.Linear):
            return {
                'type': 'linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
        elif isinstance(module, nn.BatchNorm2d):
            return {
                'type': 'batchnorm2d',
                'num_features': module.num_features,
            }
        return {}
    
    def _measure_latency(self, module: nn.Module, input_shape: tuple, 
                         bit_width: int, num_samples: int) -> float:
        """测量层延迟"""
        return self.profiler.profile_layer(module, input_shape, bit_width, num_samples)
    
    def query(self, layer_name: str, bit_width: int) -> float:
        """查询延迟"""
        key = (layer_name, bit_width)
        if key in self.lut:
            return self.lut[key]
        else:
            # 使用插值估计
            return self._interpolate(layer_name, bit_width)
    
    def _interpolate(self, layer_name: str, bit_width: int) -> float:
        """插值估计延迟"""
        # 找到最近的已知位宽
        known_bit_widths = [bw for (ln, bw) in self.lut.keys() if ln == layer_name]
        if not known_bit_widths:
            return 0.0
        
        # 线性插值
        lower_bw = max([bw for bw in known_bit_widths if bw <= bit_width], default=min(known_bit_widths))
        upper_bw = min([bw for bw in known_bit_widths if bw >= bit_width], default=max(known_bit_widths))
        
        if lower_bw == upper_bw:
            return self.lut[(layer_name, lower_bw)]
        
        lower_latency = self.lut[(layer_name, lower_bw)]
        upper_latency = self.lut[(layer_name, upper_bw)]
        
        ratio = (bit_width - lower_bw) / (upper_bw - lower_bw)
        return lower_latency + ratio * (upper_latency - lower_latency)
    
    def save(self, path: str):
        """保存LUT到文件"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.lut, f)
    
    def load(self, path: str):
        """从文件加载LUT"""
        import pickle
        with open(path, 'rb') as f:
            self.lut = pickle.load(f)
```

---

## 5. 专用推理引擎实现

### 5.1 DIE架构

```python
class DedicatedInferenceEngine:
    """专用推理引擎（DIE）"""
    
    def __init__(self, platform: str, hal: HardwareAbstractionLayer):
        self.platform = platform
        self.hal = hal
        self.optimized_model = None
        
        # 根据平台选择后端
        self.backend = self._select_backend(platform)
    
    def _select_backend(self, platform: str):
        """选择推理后端"""
        backends = {
            'nvidia_a100': TensorRTBackend,
            'jetson_orin': TensorRTBackend,
            'ascend_310': ACLBackend,
            'hygon_z100': ROCmBackend,
        }
        return backends.get(platform, TensorRTBackend)()
    
    def optimize(self, model: nn.Module, compression_config: dict) -> nn.Module:
        """优化模型"""
        # 1. 应用剪枝
        pruned_model = self._apply_pruning(model, compression_config['pruning'])
        
        # 2. 应用量化
        quantized_model = self._apply_quantization(pruned_model, compression_config['quantization'])
        
        # 3. 应用算子融合
        fused_model = self._apply_fusion(quantized_model, compression_config['fusion'])
        
        # 4. 编译为推理引擎格式
        self.optimized_model = self.backend.compile(fused_model)
        
        return self.optimized_model
    
    def _apply_pruning(self, model: nn.Module, pruning_config: dict) -> nn.Module:
        """应用剪枝"""
        for name, module in model.named_modules():
            if name in pruning_config:
                ratio = pruning_config[name]
                if ratio > 0:
                    # 结构化剪枝
                    prune.ln_structured(module, name='weight', amount=ratio, n=1, dim=0)
                    prune.remove(module, 'weight')
        return model
    
    def _apply_quantization(self, model: nn.Module, quantization_config: dict) -> nn.Module:
        """应用量化"""
        # 准备量化
        model.eval()
        
        # 设置量化配置
        qconfig_mapping = {}
        for name, bit_width in quantization_config.items():
            if bit_width == 8:
                qconfig_mapping[name] = torch.quantization.get_default_qconfig('fbgemm')
            elif bit_width == 4:
                # 4-bit量化需要自定义
                qconfig_mapping[name] = self._get_4bit_qconfig()
        
        # 应用量化
        model_prepared = torch.quantization.prepare(model, qconfig_mapping)
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    def _apply_fusion(self, model: nn.Module, fusion_config: dict) -> nn.Module:
        """应用算子融合"""
        # 定义融合模式
        fusion_patterns = {
            'conv_bn': [nn.Conv2d, nn.BatchNorm2d],
            'conv_relu': [nn.Conv2d, nn.ReLU],
            'conv_bn_relu': [nn.Conv2d, nn.BatchNorm2d, nn.ReLU],
        }
        
        # 应用融合
        for pattern_name, should_fuse in fusion_config.items():
            if should_fuse and pattern_name in fusion_patterns:
                pattern = fusion_patterns[pattern_name]
                model = torch.quantization.fuse_modules(model, pattern)
        
        return model
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行推理"""
        if self.optimized_model is None:
            raise RuntimeError("Model not optimized. Call optimize() first.")
        
        return self.backend.infer(self.optimized_model, input_tensor)


class TensorRTBackend:
    """TensorRT推理后端"""
    
    def __init__(self):
        self.engine = None
        self.context = None
    
    def compile(self, model: nn.Module) -> 'TRTEngine':
        """编译模型为TensorRT引擎"""
        import tensorrt as trt
        
        # 导出ONNX
        dummy_input = torch.randn(1, 3, 640, 640).cuda()
        onnx_path = '/tmp/model.onnx'
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13)
        
        # 构建TensorRT引擎
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16
        
        self.engine = builder.build_engine(network, config)
        self.context = self.engine.create_execution_context()
        
        return self
    
    def infer(self, engine, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行TensorRT推理"""
        # 分配缓冲区
        inputs, outputs, bindings, stream = self._allocate_buffers()
        
        # 复制输入
        np.copyto(inputs[0].host, input_tensor.cpu().numpy().ravel())
        
        # 执行推理
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        
        # 返回输出
        return torch.tensor(outputs[0].host).reshape(1, -1)
```

---

## 6. 五个智能体详细设计

### 6.1 剪枝智能体

```python
class PruningAgent:
    """剪枝智能体"""
    
    def __init__(self, num_layers: int, hidden_dim: int = 128):
        self.num_layers = num_layers
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(num_layers * 4, hidden_dim),  # 输入：每层的统计信息
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers * 10),  # 输出：每层10个剪枝率选项
        )
        
        # 剪枝率选项
        self.pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    def get_action(self, state: torch.Tensor) -> dict:
        """获取剪枝动作"""
        logits = self.policy_net(state)
        logits = logits.view(-1, self.num_layers, len(self.pruning_ratios))
        
        # 采样动作
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_indices = dist.sample()
        
        # 转换为剪枝率
        pruning_config = {}
        for i in range(self.num_layers):
            pruning_config[f'layer_{i}'] = self.pruning_ratios[action_indices[0, i].item()]
        
        return pruning_config, dist.log_prob(action_indices).sum()
    
    def apply_action(self, model: nn.Module, pruning_config: dict) -> nn.Module:
        """应用剪枝动作"""
        for name, module in model.named_modules():
            if name in pruning_config and isinstance(module, (nn.Conv2d, nn.Linear)):
                ratio = pruning_config[name]
                if ratio > 0:
                    # 使用L1范数结构化剪枝
                    prune.ln_structured(module, name='weight', amount=ratio, n=1, dim=0)
        return model


class QuantizationAgent:
    """量化智能体"""
    
    def __init__(self, num_layers: int, hidden_dim: int = 128):
        self.num_layers = num_layers
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(num_layers * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers * 4),  # 输出：每层4个位宽选项
        )
        
        # 位宽选项
        self.bit_widths = [4, 8, 16, 32]
    
    def get_action(self, state: torch.Tensor) -> dict:
        """获取量化动作"""
        logits = self.policy_net(state)
        logits = logits.view(-1, self.num_layers, len(self.bit_widths))
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_indices = dist.sample()
        
        quantization_config = {}
        for i in range(self.num_layers):
            quantization_config[f'layer_{i}'] = self.bit_widths[action_indices[0, i].item()]
        
        return quantization_config, dist.log_prob(action_indices).sum()
    
    def apply_action(self, model: nn.Module, quantization_config: dict) -> nn.Module:
        """应用量化动作"""
        # 这里需要根据具体的量化框架实现
        # 示例使用PyTorch的量化API
        model.eval()
        
        # 设置量化配置
        for name, module in model.named_modules():
            if name in quantization_config:
                bit_width = quantization_config[name]
                # 根据位宽设置量化参数
                if bit_width == 8:
                    module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                elif bit_width == 4:
                    # 4-bit量化需要自定义实现
                    pass
        
        return model


class DistillationAgent:
    """蒸馏智能体"""
    
    def __init__(self, hidden_dim: int = 128):
        # 策略网络（输出连续动作）
        self.mean_net = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # temperature, alpha
        )
        
        self.std_net = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus(),
        )
    
    def get_action(self, state: torch.Tensor) -> dict:
        """获取蒸馏动作"""
        mean = self.mean_net(state)
        std = self.std_net(state)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        # 限制范围
        temperature = torch.clamp(action[0, 0], 1.0, 20.0).item()
        alpha = torch.clamp(action[0, 1], 0.0, 1.0).item()
        
        distillation_config = {
            'temperature': temperature,
            'alpha': alpha,
        }
        
        return distillation_config, dist.log_prob(action).sum()
    
    def apply_action(self, student: nn.Module, teacher: nn.Module, 
                     distillation_config: dict, dataloader) -> nn.Module:
        """应用蒸馏动作"""
        temperature = distillation_config['temperature']
        alpha = distillation_config['alpha']
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        for batch in dataloader:
            inputs, labels = batch
            
            # 教师输出
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            
            # 学生输出
            student_outputs = student(inputs)
            
            # 蒸馏损失
            soft_loss = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=-1),
                F.softmax(teacher_outputs / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # 硬标签损失
            hard_loss = F.cross_entropy(student_outputs, labels)
            
            # 总损失
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return student


class FusionAgent:
    """融合智能体"""
    
    def __init__(self, num_fusion_points: int, hidden_dim: int = 128):
        self.num_fusion_points = num_fusion_points
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(num_fusion_points * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_fusion_points * 4),  # 4种融合模式
        )
        
        # 融合模式
        self.fusion_patterns = ['none', 'conv_bn', 'conv_relu', 'conv_bn_relu']
    
    def get_action(self, state: torch.Tensor) -> dict:
        """获取融合动作"""
        logits = self.policy_net(state)
        logits = logits.view(-1, self.num_fusion_points, len(self.fusion_patterns))
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_indices = dist.sample()
        
        fusion_config = {}
        for i in range(self.num_fusion_points):
            fusion_config[f'fusion_point_{i}'] = self.fusion_patterns[action_indices[0, i].item()]
        
        return fusion_config, dist.log_prob(action_indices).sum()
    
    def apply_action(self, model: nn.Module, fusion_config: dict) -> nn.Module:
        """应用融合动作"""
        # 使用PyTorch的融合API
        for fusion_point, pattern in fusion_config.items():
            if pattern != 'none':
                # 找到对应的模块并融合
                modules_to_fuse = self._find_modules_to_fuse(model, fusion_point, pattern)
                if modules_to_fuse:
                    torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        
        return model
    
    def _find_modules_to_fuse(self, model: nn.Module, fusion_point: str, 
                               pattern: str) -> list:
        """找到需要融合的模块"""
        # 根据融合点和模式找到对应的模块名称
        # 这里需要根据具体模型结构实现
        return []


class UpdateAgent:
    """更新智能体"""
    
    def __init__(self, hidden_dim: int = 128):
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * 5),  # 3种策略 × 5种更新比例
        )
        
        # 更新策略
        self.strategies = ['full', 'incremental', 'hash_based']
        self.update_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def get_action(self, state: torch.Tensor) -> dict:
        """获取更新动作"""
        logits = self.policy_net(state)
        logits = logits.view(-1, len(self.strategies), len(self.update_ratios))
        
        # 选择策略
        strategy_logits = logits.mean(dim=-1)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        strategy_dist = torch.distributions.Categorical(strategy_probs)
        strategy_idx = strategy_dist.sample()
        
        # 选择更新比例
        ratio_logits = logits[:, strategy_idx, :]
        ratio_probs = F.softmax(ratio_logits, dim=-1)
        ratio_dist = torch.distributions.Categorical(ratio_probs)
        ratio_idx = ratio_dist.sample()
        
        update_config = {
            'strategy': self.strategies[strategy_idx.item()],
            'update_ratio': self.update_ratios[ratio_idx.item()],
        }
        
        log_prob = strategy_dist.log_prob(strategy_idx) + ratio_dist.log_prob(ratio_idx)
        
        return update_config, log_prob
    
    def apply_action(self, model: nn.Module, update_config: dict, 
                     new_data) -> nn.Module:
        """应用更新动作"""
        strategy = update_config['strategy']
        update_ratio = update_config['update_ratio']
        
        if strategy == 'full':
            # 全量更新
            return self._full_update(model, new_data)
        elif strategy == 'incremental':
            # 增量更新
            return self._incremental_update(model, new_data, update_ratio)
        elif strategy == 'hash_based':
            # 基于哈希的更新
            return self._hash_based_update(model, new_data, update_ratio)
        
        return model
    
    def _full_update(self, model: nn.Module, new_data) -> nn.Module:
        """全量更新"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for batch in new_data:
            inputs, labels = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def _incremental_update(self, model: nn.Module, new_data, 
                            update_ratio: float) -> nn.Module:
        """增量更新"""
        # 只更新部分参数
        params_to_update = []
        for name, param in model.named_parameters():
            if torch.rand(1).item() < update_ratio:
                params_to_update.append(param)
        
        optimizer = torch.optim.Adam(params_to_update, lr=1e-4)
        
        for batch in new_data:
            inputs, labels = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def _hash_based_update(self, model: nn.Module, new_data, 
                           update_ratio: float) -> nn.Module:
        """基于哈希的更新"""
        # 使用哈希函数确定需要更新的参数
        import hashlib
        
        params_to_update = []
        for name, param in model.named_parameters():
            hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
            if hash_val < update_ratio * 100:
                params_to_update.append(param)
        
        optimizer = torch.optim.Adam(params_to_update, lr=1e-4)
        
        for batch in new_data:
            inputs, labels = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
```

---

## 7. 奖励函数设计

### 7.1 多目标奖励函数

```python
class RewardFunction:
    """多目标奖励函数"""
    
    def __init__(
        self,
        accuracy_weight: float = 1.0,
        latency_weight: float = 1.0,
        energy_weight: float = 0.5,
        size_weight: float = 0.3,
        accuracy_threshold: float = 0.90,
        latency_threshold: float = 10.0,  # ms
        energy_threshold: float = 100.0,  # mJ
    ):
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.size_weight = size_weight
        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold = latency_threshold
        self.energy_threshold = energy_threshold
    
    def compute(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        model_size: float,
        baseline_accuracy: float,
        baseline_latency: float,
        baseline_energy: float,
        baseline_size: float,
    ) -> float:
        """计算奖励"""
        # 精度奖励（相对于基线的变化）
        accuracy_reward = (accuracy - baseline_accuracy) / baseline_accuracy
        
        # 延迟奖励（减少越多越好）
        latency_reward = (baseline_latency - latency) / baseline_latency
        
        # 能耗奖励（减少越多越好）
        energy_reward = (baseline_energy - energy) / baseline_energy
        
        # 模型大小奖励（减少越多越好）
        size_reward = (baseline_size - model_size) / baseline_size
        
        # 约束惩罚
        penalty = 0
        if accuracy < self.accuracy_threshold:
            penalty += 10 * (self.accuracy_threshold - accuracy)
        if latency > self.latency_threshold:
            penalty += 5 * (latency - self.latency_threshold) / self.latency_threshold
        if energy > self.energy_threshold:
            penalty += 2 * (energy - self.energy_threshold) / self.energy_threshold
        
        # 总奖励
        reward = (
            self.accuracy_weight * accuracy_reward +
            self.latency_weight * latency_reward +
            self.energy_weight * energy_reward +
            self.size_weight * size_reward -
            penalty
        )
        
        return reward
    
    def compute_shaped_reward(
        self,
        current_state: dict,
        next_state: dict,
        action: dict,
    ) -> float:
        """计算形状化奖励（用于加速学习）"""
        # 基础奖励
        base_reward = self.compute(
            next_state['accuracy'],
            next_state['latency'],
            next_state['energy'],
            next_state['model_size'],
            current_state['accuracy'],
            current_state['latency'],
            current_state['energy'],
            current_state['model_size'],
        )
        
        # 探索奖励（鼓励尝试新配置）
        exploration_bonus = 0.1 * self._compute_novelty(action)
        
        # 进度奖励（鼓励朝目标前进）
        progress_bonus = 0.2 * self._compute_progress(current_state, next_state)
        
        return base_reward + exploration_bonus + progress_bonus
    
    def _compute_novelty(self, action: dict) -> float:
        """计算动作新颖性"""
        # 这里可以使用历史动作的统计信息
        # 简化版本返回随机值
        return torch.rand(1).item() * 0.1
    
    def _compute_progress(self, current_state: dict, next_state: dict) -> float:
        """计算进度"""
        # 计算朝目标的进度
        current_distance = self._distance_to_target(current_state)
        next_distance = self._distance_to_target(next_state)
        return current_distance - next_distance
    
    def _distance_to_target(self, state: dict) -> float:
        """计算到目标的距离"""
        accuracy_gap = max(0, self.accuracy_threshold - state['accuracy'])
        latency_gap = max(0, state['latency'] - self.latency_threshold)
        return accuracy_gap + latency_gap * 0.1
```

---

## 8. 训练流程

### 8.1 完整训练流程

```python
class HADMCTrainer:
    """HAD-MC 2.0训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        train_loader,
        val_loader,
        hal: HardwareAbstractionLayer,
        config: dict,
    ):
        self.model = model
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hal = hal
        self.config = config
        
        # 初始化智能体
        num_layers = self._count_layers(model)
        self.pruning_agent = PruningAgent(num_layers)
        self.quantization_agent = QuantizationAgent(num_layers)
        self.distillation_agent = DistillationAgent()
        self.fusion_agent = FusionAgent(num_layers // 3)
        self.update_agent = UpdateAgent()
        
        # 初始化PPO控制器
        state_dim = self._compute_state_dim(model)
        action_dims = {
            'pruning': num_layers * 10,
            'quantization': num_layers * 4,
            'distillation': 2,
            'fusion': (num_layers // 3) * 4,
            'update': 15,
        }
        self.ppo_controller = PPOController(state_dim, action_dims)
        
        # 初始化奖励函数
        self.reward_fn = RewardFunction(
            accuracy_threshold=config.get('accuracy_threshold', 0.90),
            latency_threshold=config.get('latency_threshold', 10.0),
        )
        
        # 初始化DIE
        self.die = DedicatedInferenceEngine(hal.platform, hal)
        
        # 记录基线性能
        self.baseline = self._evaluate_baseline()
    
    def _count_layers(self, model: nn.Module) -> int:
        """计算模型层数"""
        count = 0
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                count += 1
        return count
    
    def _compute_state_dim(self, model: nn.Module) -> int:
        """计算状态维度"""
        # 模型状态 + 硬件状态 + 压缩状态
        num_layers = self._count_layers(model)
        return num_layers * 4 + 10 + num_layers * 3
    
    def _evaluate_baseline(self) -> dict:
        """评估基线性能"""
        accuracy = self._evaluate_accuracy(self.model)
        latency = self.hal.estimate_latency(self._get_model_config(self.model))
        
        return {
            'accuracy': accuracy,
            'latency': latency,
            'energy': latency * 0.1,  # 简化的能耗估计
            'model_size': sum(p.numel() for p in self.model.parameters()) * 4 / 1e6,  # MB
        }
    
    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """评估模型精度"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
    
    def _get_model_config(self, model: nn.Module) -> dict:
        """获取模型配置"""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                flops = module.in_channels * module.out_channels * \
                        module.kernel_size[0] * module.kernel_size[1]
                memory_access = module.weight.numel() * 4
                layers.append({
                    'type': 'conv',
                    'flops': flops,
                    'memory_access': memory_access,
                })
            elif isinstance(module, nn.Linear):
                flops = module.in_features * module.out_features
                memory_access = module.weight.numel() * 4
                layers.append({
                    'type': 'fc',
                    'flops': flops,
                    'memory_access': memory_access,
                })
        return {'layers': layers}
    
    def train(self, num_episodes: int = 100):
        """训练主循环"""
        best_reward = float('-inf')
        best_config = None
        
        for episode in range(num_episodes):
            # 获取当前状态
            state = self._get_state()
            
            # 选择动作
            actions, log_prob, value = self.ppo_controller.select_action(state)
            
            # 应用动作
            compressed_model = self._apply_actions(actions)
            
            # 评估压缩后的模型
            metrics = self._evaluate_model(compressed_model)
            
            # 计算奖励
            reward = self.reward_fn.compute(
                metrics['accuracy'],
                metrics['latency'],
                metrics['energy'],
                metrics['model_size'],
                self.baseline['accuracy'],
                self.baseline['latency'],
                self.baseline['energy'],
                self.baseline['model_size'],
            )
            
            # 获取下一状态
            next_state = self._get_state(compressed_model)
            done = episode == num_episodes - 1
            
            # 存储转换
            self.ppo_controller.store_transition(
                state, actions, log_prob, reward, next_state, done, value
            )
            
            # 更新PPO
            if (episode + 1) % 10 == 0:
                update_info = self.ppo_controller.update()
                print(f"Episode {episode + 1}: Reward = {reward:.4f}, "
                      f"Accuracy = {metrics['accuracy']:.4f}, "
                      f"Latency = {metrics['latency']:.2f}ms")
            
            # 保存最佳配置
            if reward > best_reward:
                best_reward = reward
                best_config = actions
                self._save_checkpoint(compressed_model, actions, episode)
        
        return best_config, best_reward
    
    def _get_state(self, model: nn.Module = None) -> torch.Tensor:
        """获取当前状态"""
        if model is None:
            model = self.model
        
        # 构建状态向量
        state_parts = []
        
        # 模型状态
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 权重统计
                weight = module.weight.data
                state_parts.extend([
                    weight.mean().item(),
                    weight.std().item(),
                    weight.abs().mean().item(),
                    (weight == 0).float().mean().item(),  # 稀疏度
                ])
        
        # 硬件状态
        hw_state = self.hal.get_hardware_state()
        state_parts.extend([
            hw_state['compute_capability'] / 100,
            hw_state['memory_bandwidth'] / 3000,
            hw_state['memory_capacity'] / 100,
            hw_state['power_budget'] / 500,
            1 if hw_state['has_tensor_core'] else 0,
            1 if hw_state['has_sparsity_support'] else 0,
        ])
        
        return torch.tensor(state_parts, dtype=torch.float32).unsqueeze(0)
    
    def _apply_actions(self, actions: dict) -> nn.Module:
        """应用所有智能体的动作"""
        model = copy.deepcopy(self.model)
        
        # 应用剪枝
        model = self.pruning_agent.apply_action(model, actions['pruning'])
        
        # 应用量化
        model = self.quantization_agent.apply_action(model, actions['quantization'])
        
        # 应用蒸馏
        model = self.distillation_agent.apply_action(
            model, self.teacher_model, actions['distillation'], self.train_loader
        )
        
        # 应用融合
        model = self.fusion_agent.apply_action(model, actions['fusion'])
        
        return model
    
    def _evaluate_model(self, model: nn.Module) -> dict:
        """评估模型"""
        accuracy = self._evaluate_accuracy(model)
        latency = self.hal.estimate_latency(self._get_model_config(model))
        model_size = sum(p.numel() for p in model.parameters()) * 4 / 1e6
        
        return {
            'accuracy': accuracy,
            'latency': latency,
            'energy': latency * 0.1,
            'model_size': model_size,
        }
    
    def _save_checkpoint(self, model: nn.Module, config: dict, episode: int):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'episode': episode,
        }
        torch.save(checkpoint, f'checkpoint_ep{episode}.pt')
```

---

## 9. 代码架构

### 9.1 目录结构

```
hadmc/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── pruning_agent.py
│   ├── quantization_agent.py
│   ├── distillation_agent.py
│   ├── fusion_agent.py
│   └── update_agent.py
├── controllers/
│   ├── __init__.py
│   ├── ppo_controller.py
│   └── marl_coordinator.py
├── hardware/
│   ├── __init__.py
│   ├── hal.py
│   ├── latency_lut.py
│   └── profiler.py
├── inference/
│   ├── __init__.py
│   ├── die.py
│   ├── tensorrt_backend.py
│   ├── acl_backend.py
│   └── rocm_backend.py
├── rewards/
│   ├── __init__.py
│   └── reward_function.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── evaluator.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── visualization.py
└── configs/
    ├── default.yaml
    ├── jetson_orin.yaml
    ├── ascend_310.yaml
    └── hygon_z100.yaml
```

### 9.2 配置文件示例

```yaml
# configs/default.yaml
model:
  name: yolov5s
  input_shape: [1, 3, 640, 640]
  num_classes: 80

hardware:
  platform: nvidia_a100
  precision: fp16

training:
  num_episodes: 500
  batch_size: 64
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01

constraints:
  accuracy_threshold: 0.90
  latency_threshold: 10.0  # ms
  energy_threshold: 100.0  # mJ

agents:
  pruning:
    enabled: true
    ratios: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  quantization:
    enabled: true
    bit_widths: [4, 8, 16, 32]
  distillation:
    enabled: true
    temperature_range: [1.0, 20.0]
    alpha_range: [0.0, 1.0]
  fusion:
    enabled: true
    patterns: [none, conv_bn, conv_relu, conv_bn_relu]
  update:
    enabled: true
    strategies: [full, incremental, hash_based]
```

---

## 10. 测试与验证

### 10.1 单元测试

```python
# tests/test_agents.py
import unittest
import torch
from hadmc.agents import PruningAgent, QuantizationAgent

class TestPruningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PruningAgent(num_layers=10)
    
    def test_get_action(self):
        state = torch.randn(1, 40)
        config, log_prob = self.agent.get_action(state)
        
        self.assertEqual(len(config), 10)
        self.assertTrue(all(0 <= v <= 0.9 for v in config.values()))
        self.assertIsInstance(log_prob, torch.Tensor)
    
    def test_apply_action(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.Conv2d(64, 128, 3),
        )
        config = {'layer_0': 0.5, 'layer_1': 0.3}
        
        pruned_model = self.agent.apply_action(model, config)
        self.assertIsNotNone(pruned_model)


class TestQuantizationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QuantizationAgent(num_layers=10)
    
    def test_get_action(self):
        state = torch.randn(1, 40)
        config, log_prob = self.agent.get_action(state)
        
        self.assertEqual(len(config), 10)
        self.assertTrue(all(v in [4, 8, 16, 32] for v in config.values()))


if __name__ == '__main__':
    unittest.main()
```

### 10.2 集成测试

```python
# tests/test_integration.py
import unittest
import torch
from hadmc import HADMCTrainer, HardwareAbstractionLayer

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # 创建简单模型
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10),
        )
        
        # 创建HAL
        self.hal = HardwareAbstractionLayer('nvidia_a100')
        
        # 创建数据加载器
        self.train_loader = self._create_dummy_loader()
        self.val_loader = self._create_dummy_loader()
    
    def _create_dummy_loader(self):
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        return torch.utils.data.DataLoader(dataset, batch_size=10)
    
    def test_training_loop(self):
        trainer = HADMCTrainer(
            model=self.model,
            teacher_model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            hal=self.hal,
            config={'accuracy_threshold': 0.5, 'latency_threshold': 100.0},
        )
        
        best_config, best_reward = trainer.train(num_episodes=5)
        
        self.assertIsNotNone(best_config)
        self.assertIsInstance(best_reward, float)


if __name__ == '__main__':
    unittest.main()
```

---

*本文档由12位教授级专家联合撰写，经过12轮讨论、反思、修改后形成，旨在为HAD-MC 2.0的算法升级提供全面、详细、可执行的指导方案。*

        max_grad_norm: float = 0.5,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络
        self.policy = PolicyNetwork(state_dim, action_dims).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value.parameters(), 'lr': lr},
        ])
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> tuple:
        """计算广义优势估计（GAE）"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, batch_size: int = 64, num_epochs: int = 10):
        """执行PPO更新"""
        # 从缓冲区获取数据
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get()
        
        # 计算GAE
        with torch.no_grad():
            next_value = self.value(states[-1:])
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = {k: v[batch_indices] for k, v in actions.items()}
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的动作概率
                distributions = self.policy(batch_states)
                new_log_probs = self._compute_log_probs(distributions, batch_actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                new_values = self.value(batch_states).squeeze()
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 熵损失
                entropy = self._compute_entropy(distributions)
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
    
    def _compute_log_probs(self, distributions: dict, actions: dict) -> torch.Tensor:
        """计算动作的对数概率"""
        log_probs = []
        
        # 剪枝
        pruning_dist = torch.distributions.Categorical(distributions['pruning'])
        log_probs.append(pruning_dist.log_prob(actions['pruning']))
        
        # 量化
        quantization_dist = torch.distributions.Categorical(distributions['quantization'])
        log_probs.append(quantization_dist.log_prob(actions['quantization']))
        
        # 蒸馏
        distillation_dist = torch.distributions.Normal(
            distributions['distillation_mean'],
            distributions['distillation_std']
        )
        log_probs.append(distillation_dist.log_prob(actions['distillation']).sum(dim=-1))
        
        # 融合
        fusion_dist = torch.distributions.Categorical(distributions['fusion'])
        log_probs.append(fusion_dist.log_prob(actions['fusion']))
        
        # 更新
        update_dist = torch.distributions.Categorical(distributions['update'])
        log_probs.append(update_dist.log_prob(actions['update']))
        
        return sum(log_probs)
    
    def _compute_entropy(self, distributions: dict) -> torch.Tensor:
        """计算策略熵"""
        entropies = []
        
        # 离散分布的熵
        for key in ['pruning', 'quantization', 'fusion', 'update']:
            dist = torch.distributions.Categorical(distributions[key])
            entropies.append(dist.entropy().mean())
        
        # 连续分布的熵
        distillation_dist = torch.distributions.Normal(
            distributions['distillation_mean'],
            distributions['distillation_std']
        )
        entropies.append(distillation_dist.entropy().sum(dim=-1).mean())
        
        return sum(entropies)


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = {'pruning': [], 'quantization': [], 'distillation': [], 'fusion': [], 'update': []}
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, state, actions, log_prob, reward, done, value):
        """添加一条经验"""
        self.states.append(state)
        for key in actions:
            self.actions[key].append(actions[key])
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def get(self):
        """获取所有经验"""
        states = torch.stack(self.states)
        actions = {k: torch.stack(v) for k, v in self.actions.items()}
        log_probs = torch.stack(self.log_probs)
        rewards = torch.tensor(self.rewards)
        dones = torch.tensor(self.dones)
        values = torch.stack(self.values).squeeze()
        
        return states, actions, log_probs, rewards, dones, values
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = {'pruning': [], 'quantization': [], 'distillation': [], 'fusion': [], 'update': []}
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
```

---

## 4. 硬件抽象层实现

### 4.1 HAL核心类

```python
class HardwareAbstractionLayer:
    """硬件抽象层：统一不同硬件平台的接口"""
    
    def __init__(self, config_path: str = None):
        self.hardware_configs = {}
        self.latency_lut = {}  # 延迟查找表
        self.current_hardware = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """加载硬件配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for hw_name, hw_config in config['hardware'].items():
            self.hardware_configs[hw_name] = HardwareConfig(
                name=hw_name,
                compute_capability=hw_config['compute_capability'],
                memory_bandwidth=hw_config['memory_bandwidth'],
                memory_capacity=hw_config['memory_capacity'],
                power_budget=hw_config['power_budget'],
                supported_precisions=hw_config['supported_precisions'],
                has_tensor_core=hw_config.get('has_tensor_core', False),
                has_sparsity_support=hw_config.get('has_sparsity_support', False),
            )
    
    def set_hardware(self, hardware_name: str):
        """设置当前硬件"""
        if hardware_name not in self.hardware_configs:
            raise ValueError(f"Unknown hardware: {hardware_name}")
        self.current_hardware = self.hardware_configs[hardware_name]
    
    def build_latency_lut(self, model: nn.Module, input_shape: tuple):
        """构建延迟查找表"""
        if self.current_hardware is None:
            raise RuntimeError("Hardware not set. Call set_hardware() first.")
        
        self.latency_lut = {}
        
        # 遍历模型的每一层
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layer_lut = {}
                
                # 测量不同配置下的延迟
                for bit_width in self.current_hardware.supported_precisions:
                    for sparsity in [0.0, 0.25, 0.5, 0.75, 0.9]:
                        latency = self._measure_layer_latency(
                            module, input_shape, bit_width, sparsity
                        )
                        layer_lut[(bit_width, sparsity)] = latency
                
                self.latency_lut[name] = layer_lut
    
    def _measure_layer_latency(
        self,
        module: nn.Module,
        input_shape: tuple,
        bit_width: str,
        sparsity: float
    ) -> float:
        """测量单层延迟"""
        # 创建测试输入
        x = torch.randn(input_shape).to(self.current_hardware.device)
        
        # 应用量化和稀疏性
        quantized_module = self._quantize_module(module, bit_width)
        sparse_module = self._apply_sparsity(quantized_module, sparsity)
        
        # 预热
        for _ in range(10):
            _ = sparse_module(x)
        
        # 测量
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = sparse_module(x)
        torch.cuda.synchronize()
        end = time.time()
        
        return (end - start) / 100 * 1000  # 毫秒
    
    def estimate_latency(self, compression_config: dict) -> float:
        """根据压缩配置估计总延迟"""
        total_latency = 0
        
        for layer_name, layer_config in compression_config.items():
            if layer_name in self.latency_lut:
                bit_width = layer_config.get('bit_width', 'FP32')
                sparsity = layer_config.get('sparsity', 0.0)
                
                # 从LUT查找延迟
                key = (bit_width, sparsity)
                if key in self.latency_lut[layer_name]:
                    total_latency += self.latency_lut[layer_name][key]
                else:
                    # 插值估计
                    total_latency += self._interpolate_latency(
                        self.latency_lut[layer_name], bit_width, sparsity
                    )
        
        return total_latency
    
    def get_hardware_features(self) -> torch.Tensor:
        """获取当前硬件的特征向量"""
        if self.current_hardware is None:
            raise RuntimeError("Hardware not set")
        
        return self.current_hardware.to_tensor()


@dataclass
class HardwareConfig:
    """硬件配置数据类"""
    name: str
    compute_capability: float  # TFLOPS
    memory_bandwidth: float    # GB/s
    memory_capacity: float     # GB
    power_budget: float        # W
    supported_precisions: List[str]
    has_tensor_core: bool
    has_sparsity_support: bool
    device: str = 'cuda:0'
    
    def to_tensor(self) -> torch.Tensor:
        """转换为特征张量"""
        features = [
            self.compute_capability / 100,
            self.memory_bandwidth / 3000,
            self.memory_capacity / 100,
            self.power_budget / 500,
            1.0 if self.has_tensor_core else 0.0,
            1.0 if self.has_sparsity_support else 0.0,
        ]
        
        # 精度支持（one-hot）
        precision_map = {'FP32': 0, 'FP16': 1, 'INT8': 2, 'INT4': 3}
        precision_features = [0.0] * len(precision_map)
        for p in self.supported_precisions:
            if p in precision_map:
                precision_features[precision_map[p]] = 1.0
        features.extend(precision_features)
        
        return torch.tensor(features, dtype=torch.float32)
```

### 4.2 硬件配置文件示例

```yaml
# hardware_configs.yaml
hardware:
  nvidia_a100:
    compute_capability: 156.0  # TFLOPS (FP16 Tensor Core)
    memory_bandwidth: 2039.0   # GB/s
    memory_capacity: 80.0      # GB
    power_budget: 400.0        # W
    supported_precisions: ['FP32', 'FP16', 'INT8', 'INT4']
    has_tensor_core: true
    has_sparsity_support: true
  
  ascend_310:
    compute_capability: 22.0   # TFLOPS (INT8)
    memory_bandwidth: 25.6     # GB/s
    memory_capacity: 8.0       # GB
    power_budget: 8.0          # W
    supported_precisions: ['FP32', 'FP16', 'INT8']
    has_tensor_core: false
    has_sparsity_support: false
  
  hygon_dcu:
    compute_capability: 32.0   # TFLOPS (FP16)
    memory_bandwidth: 1024.0   # GB/s
    memory_capacity: 32.0      # GB
    power_budget: 300.0        # W
    supported_precisions: ['FP32', 'FP16', 'INT8']
    has_tensor_core: true
    has_sparsity_support: false
```

---

## 5. 专用推理引擎实现

### 5.1 DIE核心类

```python
class DedicatedInferenceEngine:
    """专用推理引擎：优化压缩模型的推理"""
    
    def __init__(self, hal: HardwareAbstractionLayer):
        self.hal = hal
        self.optimized_model = None
        self.execution_plan = None
    
    def optimize(self, model: nn.Module, compression_config: dict) -> nn.Module:
        """优化模型以适应当前硬件"""
        # 1. 应用算子融合
        fused_model = self._apply_operator_fusion(model, compression_config)
        
        # 2. 应用稀疏性优化
        sparse_model = self._apply_sparsity_optimization(fused_model, compression_config)
        
        # 3. 应用混合精度
        mixed_precision_model = self._apply_mixed_precision(sparse_model, compression_config)
        
        # 4. 生成执行计划
        self.execution_plan = self._generate_execution_plan(mixed_precision_model)
        
        self.optimized_model = mixed_precision_model
        return self.optimized_model
    
    def _apply_operator_fusion(self, model: nn.Module, config: dict) -> nn.Module:
        """应用算子融合"""
        fused_model = copy.deepcopy(model)
        
        # 查找可融合的模式
        fusion_patterns = [
            ('Conv2d', 'BatchNorm2d'),
            ('Conv2d', 'BatchNorm2d', 'ReLU'),
            ('Conv2d', 'ReLU'),
            ('Linear', 'ReLU'),
        ]
        
        for pattern in fusion_patterns:
            fused_model = self._fuse_pattern(fused_model, pattern)
        
        return fused_model
    
    def _fuse_pattern(self, model: nn.Module, pattern: tuple) -> nn.Module:
        """融合特定模式的算子"""
        modules = list(model.named_modules())
        
        i = 0
        while i < len(modules) - len(pattern) + 1:
            # 检查是否匹配模式
            match = True
            for j, expected_type in enumerate(pattern):
                if modules[i + j][1].__class__.__name__ != expected_type:
                    match = False
                    break
            
            if match:
                # 执行融合
                if pattern == ('Conv2d', 'BatchNorm2d'):
                    fused = self._fuse_conv_bn(modules[i][1], modules[i + 1][1])
                elif pattern == ('Conv2d', 'BatchNorm2d', 'ReLU'):
                    fused = self._fuse_conv_bn_relu(
                        modules[i][1], modules[i + 1][1], modules[i + 2][1]
                    )
                elif pattern == ('Conv2d', 'ReLU'):
                    fused = self._fuse_conv_relu(modules[i][1], modules[i + 2][1])
                elif pattern == ('Linear', 'ReLU'):
                    fused = self._fuse_linear_relu(modules[i][1], modules[i + 1][1])
                
                # 替换模块
                self._replace_module(model, modules[i][0], fused)
                # 移除被融合的模块
                for j in range(1, len(pattern)):
                    self._remove_module(model, modules[i + j][0])
                
                i += len(pattern)
            else:
                i += 1
        
        return model
    
    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """融合Conv和BN"""
        # 获取BN参数
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        # 计算融合后的权重和偏置
        std = torch.sqrt(var + eps)
        fused_weight = conv.weight * (gamma / std).view(-1, 1, 1, 1)
        
        if conv.bias is not None:
            fused_bias = (conv.bias - mean) * gamma / std + beta
        else:
            fused_bias = -mean * gamma / std + beta
        
        # 创建融合后的Conv
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups, True
        )
        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias
        
        return fused_conv
    
    def _apply_sparsity_optimization(self, model: nn.Module, config: dict) -> nn.Module:
        """应用稀疏性优化"""
        if not self.hal.current_hardware.has_sparsity_support:
            return model
        
        # 转换为稀疏格式
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                sparsity = config.get(name, {}).get('sparsity', 0.0)
                if sparsity > 0:
                    # 应用2:4稀疏性（NVIDIA Ampere架构）
                    module.weight.data = self._apply_2_4_sparsity(module.weight.data)
        
        return model
    
    def _apply_2_4_sparsity(self, weight: torch.Tensor) -> torch.Tensor:
        """应用2:4结构化稀疏性"""
        # 将权重reshape为(N, 4)的形式
        original_shape = weight.shape
        weight_flat = weight.view(-1, 4)
        
        # 对每4个元素，保留绝对值最大的2个
        _, indices = torch.topk(weight_flat.abs(), k=2, dim=1)
        mask = torch.zeros_like(weight_flat)
        mask.scatter_(1, indices, 1)
        
        sparse_weight = weight_flat * mask
        return sparse_weight.view(original_shape)
    
    def _apply_mixed_precision(self, model: nn.Module, config: dict) -> nn.Module:
        """应用混合精度"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                bit_width = config.get(name, {}).get('bit_width', 'FP32')
                
                if bit_width == 'FP16':
                    module.half()
                elif bit_width == 'INT8':
                    module = torch.quantization.quantize_dynamic(
                        module, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
        
        return model
    
    def _generate_execution_plan(self, model: nn.Module) -> dict:
        """生成执行计划"""
        plan = {
            'layers': [],
            'memory_allocation': {},
            'stream_schedule': [],
        }
        
        # 分析层依赖关系
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'input_shape': None,  # 需要在运行时确定
                    'output_shape': None,
                    'estimated_latency': self.hal.latency_lut.get(name, {}).get(('FP32', 0.0), 0),
                }
                plan['layers'].append(layer_info)
        
        return plan
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """执行推理"""
        if self.optimized_model is None:
            raise RuntimeError("Model not optimized. Call optimize() first.")
        
        with torch.no_grad():
            return self.optimized_model(x)
```

---

## 6. 五个智能体详细设计

### 6.1 剪枝智能体

```python
class PruningAgent:
    """剪枝智能体：基于梯度敏感度的结构化剪枝"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.importance_scores = {}
    
    def compute_importance(self) -> dict:
        """计算每层每个通道的重要性分数"""
        self.model.train()
        
        # 注册钩子收集梯度
        gradients = {}
        handles = []
        
        def hook_fn(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_backward_hook(hook_fn(name))
                handles.append(handle)
        
        # 前向和反向传播
        for images, targets in self.train_loader:
            self.model.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            break  # 只需要一个batch
        
        # 移除钩子
        for handle in handles:
            handle.remove()
        
        # 计算重要性分数
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                grad = gradients.get(name, torch.zeros_like(weight))
                
                # Taylor展开近似
                importance = (weight * grad).abs().sum(dim=(1, 2, 3))
                self.importance_scores[name] = importance
        
        return self.importance_scores
    
    def prune(self, pruning_ratios: dict) -> nn.Module:
        """执行剪枝"""
        if not self.importance_scores:
            self.compute_importance()
        
        pruned_model = copy.deepcopy(self.model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in pruning_ratios:
                ratio = pruning_ratios[name]
                if ratio > 0:
                    self._prune_layer(module, name, ratio)
        
        return pruned_model
    
    def _prune_layer(self, module: nn.Conv2d, name: str, ratio: float):
        """剪枝单层"""
        importance = self.importance_scores[name]
        num_channels = len(importance)
        num_prune = int(num_channels * ratio)
        
        if num_prune == 0:
            return
        
        # 找到最不重要的通道
        _, indices = torch.topk(importance, num_prune, largest=False)
        
        # 创建掩码
        mask = torch.ones(num_channels, dtype=torch.bool)
        mask[indices] = False
        
        # 应用掩码
        module.weight.data = module.weight.data[mask]
        if module.bias is not None:
            module.bias.data = module.bias.data[mask]
        module.out_channels = mask.sum().item()
```

### 6.2 量化智能体

```python
class QuantizationAgent:
    """量化智能体：逐层精度分配"""
    
    def __init__(self, model: nn.Module, calibration_loader: DataLoader):
        self.model = model
        self.calibration_loader = calibration_loader
        self.quantization_params = {}
    
    def calibrate(self):
        """校准量化参数"""
        self.model.eval()
        
        # 收集激活统计信息
        activation_stats = {}
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = {'min': [], 'max': []}
                activation_stats[name]['min'].append(output.min().item())
                activation_stats[name]['max'].append(output.max().item())
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # 运行校准数据
        with torch.no_grad():
            for images, _ in self.calibration_loader:
                self.model(images)
        
        # 移除钩子
        for handle in handles:
            handle.remove()
        
        # 计算量化参数
        for name, stats in activation_stats.items():
            min_val = min(stats['min'])
            max_val = max(stats['max'])
            self.quantization_params[name] = {
                'min': min_val,
                'max': max_val,
                'scale': (max_val - min_val) / 255,
                'zero_point': int(-min_val / ((max_val - min_val) / 255)),
            }
    
    def quantize(self, bit_widths: dict) -> nn.Module:
        """执行量化"""
        if not self.quantization_params:
            self.calibrate()
        
        quantized_model = copy.deepcopy(self.model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in bit_widths:
                bit_width = bit_widths[name]
                if bit_width < 32:
                    self._quantize_layer(module, name, bit_width)
        
        return quantized_model
    
    def _quantize_layer(self, module: nn.Module, name: str, bit_width: int):
        """量化单层"""
        params = self.quantization_params.get(name, {})
        
        if bit_width == 8:
            # INT8量化
            scale = params.get('scale', 1.0)
            zero_point = params.get('zero_point', 0)
            
            weight = module.weight.data
            quantized_weight = torch.round(weight / scale + zero_point).clamp(0, 255)
            module.weight.data = (quantized_weight - zero_point) * scale
        
        elif bit_width == 4:
            # INT4量化
            scale = params.get('scale', 1.0) * 16
            zero_point = params.get('zero_point', 0) // 16
            
            weight = module.weight.data
            quantized_weight = torch.round(weight / scale + zero_point).clamp(0, 15)
            module.weight.data = (quantized_weight - zero_point) * scale
```

### 6.3 蒸馏智能体

```python
class DistillationAgent:
    """蒸馏智能体：特征对齐蒸馏"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        self.teacher = teacher_model
        self.student = student_model
        self.teacher.eval()
    
    def distill(
        self,
        train_loader: DataLoader,
        temperature: float = 4.0,
        alpha: float = 0.5,
        epochs: int = 10,
        lr: float = 1e-4
    ) -> nn.Module:
        """执行知识蒸馏"""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            
            for images, targets in train_loader:
                optimizer.zero_grad()
                
                # 教师输出
                with torch.no_grad():
                    teacher_outputs = self.teacher(images)
                    teacher_features = self._get_intermediate_features(self.teacher, images)
                
                # 学生输出
                student_outputs = self.student(images)
                student_features = self._get_intermediate_features(self.student, images)
                
                # 计算损失
                # 1. 软标签损失
                soft_loss = F.kl_div(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # 2. 硬标签损失
                hard_loss = F.cross_entropy(student_outputs, targets)
                
                # 3. 特征对齐损失
                feature_loss = self._compute_feature_loss(teacher_features, student_features)
                
                # 总损失
                loss = alpha * soft_loss + (1 - alpha) * hard_loss + 0.1 * feature_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        return self.student
    
    def _get_intermediate_features(self, model: nn.Module, x: torch.Tensor) -> list:
        """获取中间层特征"""
        features = []
        handles = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # 在关键层注册钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and 'layer' in name:
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        with torch.no_grad():
            model(x)
        
        for handle in handles:
            handle.remove()
        
        return features
    
    def _compute_feature_loss(self, teacher_features: list, student_features: list) -> torch.Tensor:
        """计算特征对齐损失"""
        loss = 0
        
        for t_feat, s_feat in zip(teacher_features, student_features):
            # 如果维度不匹配，使用适配层
            if t_feat.shape != s_feat.shape:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
            
            loss += F.mse_loss(s_feat, t_feat)
        
        return loss / len(teacher_features) if teacher_features else 0
```

### 6.4 融合智能体

```python
class FusionAgent:
    """融合智能体：算子融合优化"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fusion_patterns = [
            ('Conv2d', 'BatchNorm2d'),
            ('Conv2d', 'BatchNorm2d', 'ReLU'),
            ('Conv2d', 'ReLU'),
            ('Linear', 'ReLU'),
            ('Conv2d', 'BatchNorm2d', 'Add', 'ReLU'),  # 残差块
        ]
    
    def analyze(self) -> list:
        """分析可融合的模式"""
        fusable_patterns = []
        modules = list(self.model.named_modules())
        
        for i, (name, module) in enumerate(modules):
            for pattern in self.fusion_patterns:
                if self._match_pattern(modules, i, pattern):
                    fusable_patterns.append({
                        'start_idx': i,
                        'start_name': name,
                        'pattern': pattern,
                    })
        
        return fusable_patterns
    
    def _match_pattern(self, modules: list, start_idx: int, pattern: tuple) -> bool:
        """检查是否匹配模式"""
        if start_idx + len(pattern) > len(modules):
            return False
        
        for j, expected_type in enumerate(pattern):
            actual_type = modules[start_idx + j][1].__class__.__name__
            if actual_type != expected_type:
                return False
        
        return True
    
    def fuse(self, fusion_config: dict) -> nn.Module:
        """执行融合"""
        fused_model = copy.deepcopy(self.model)
        
        for pattern_info in fusion_config.get('patterns', []):
            pattern = pattern_info['pattern']
            start_name = pattern_info['start_name']
            
            if pattern == ('Conv2d', 'BatchNorm2d'):
                fused_model = self._fuse_conv_bn(fused_model, start_name)
            elif pattern == ('Conv2d', 'BatchNorm2d', 'ReLU'):
                fused_model = self._fuse_conv_bn_relu(fused_model, start_name)
            # ... 其他融合模式
        
        return fused_model
    
    def _fuse_conv_bn(self, model: nn.Module, conv_name: str) -> nn.Module:
        """融合Conv和BN"""
        # 获取模块
        conv = dict(model.named_modules())[conv_name]
        bn_name = self._get_next_module_name(model, conv_name)
        bn = dict(model.named_modules())[bn_name]
        
        # 计算融合参数
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        std = torch.sqrt(var + eps)
        fused_weight = conv.weight * (gamma / std).view(-1, 1, 1, 1)
        
        if conv.bias is not None:
            fused_bias = (conv.bias - mean) * gamma / std + beta
        else:
            fused_bias = -mean * gamma / std + beta
        
        # 创建融合层
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups, True
        )
        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias
        
        # 替换模块
        self._replace_module(model, conv_name, fused_conv)
        self._remove_module(model, bn_name)
        
        return model
    
    def _get_next_module_name(self, model: nn.Module, current_name: str) -> str:
        """获取下一个模块的名称"""
        modules = list(model.named_modules())
        for i, (name, _) in enumerate(modules):
            if name == current_name and i + 1 < len(modules):
                return modules[i + 1][0]
        return None
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """替换模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def _remove_module(self, model: nn.Module, name: str):
        """移除模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
```

### 6.5 更新智能体

```python
class UpdateAgent:
    """更新智能体：增量更新和哈希更新"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hash_tables = {}
        self.weight_clusters = {}
    
    def build_hash_tables(self, num_clusters: int = 256):
        """构建权重哈希表"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data.flatten()
                
                # K-means聚类
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                labels = kmeans.fit_predict(weight.cpu().numpy().reshape(-1, 1))
                
                self.hash_tables[name] = {
                    'centroids': torch.tensor(kmeans.cluster_centers_.flatten()),
                    'labels': torch.tensor(labels),
                }
                self.weight_clusters[name] = kmeans
    
    def hash_update(self, new_weights: dict) -> nn.Module:
        """哈希更新：只更新聚类中心"""
        if not self.hash_tables:
            self.build_hash_tables()
        
        updated_model = copy.deepcopy(self.model)
        
        for name, module in updated_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in new_weights:
                # 获取新权重
                new_weight = new_weights[name]
                
                # 更新聚类中心
                kmeans = self.weight_clusters[name]
                new_centroids = kmeans.fit(new_weight.flatten().cpu().numpy().reshape(-1, 1))
                
                # 重建权重
                labels = self.hash_tables[name]['labels']
                centroids = torch.tensor(new_centroids.cluster_centers_.flatten())
                reconstructed = centroids[labels].view(module.weight.shape)
                
                module.weight.data = reconstructed.to(module.weight.device)
        
        return updated_model
    
    def incremental_update(
        self,
        new_data_loader: DataLoader,
        update_ratio: float = 0.1,
        epochs: int = 5,
        lr: float = 1e-5
    ) -> nn.Module:
        """增量更新：只更新部分层"""
        updated_model = copy.deepcopy(self.model)
        
        # 确定要更新的层
        all_layers = [name for name, m in updated_model.named_modules() 
                      if isinstance(m, (nn.Conv2d, nn.Linear))]
        num_update = int(len(all_layers) * update_ratio)
        layers_to_update = all_layers[-num_update:]  # 更新后面的层
        
        # 冻结不更新的层
        for name, param in updated_model.named_parameters():
            layer_name = '.'.join(name.split('.')[:-1])
            if layer_name not in layers_to_update:
                param.requires_grad = False
        
        # 训练
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, updated_model.parameters()),
            lr=lr
        )
        
        for epoch in range(epochs):
            updated_model.train()
            for images, targets in new_data_loader:
                optimizer.zero_grad()
                outputs = updated_model(images)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
        
        return updated_model
```

---

## 7. 奖励函数设计

### 7.1 多目标奖励函数

```python
class RewardFunction:
    """多目标奖励函数"""
    
    def __init__(
        self,
        accuracy_weight: float = 1.0,
        latency_weight: float = 0.5,
        energy_weight: float = 0.3,
        size_weight: float = 0.2,
        target_accuracy: float = 0.95,
        target_latency: float = 5.0,  # ms
        target_energy: float = 10.0,  # mJ
        target_size: float = 5.0,     # MB
    ):
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.size_weight = size_weight
        self.target_accuracy = target_accuracy
        self.target_latency = target_latency
        self.target_energy = target_energy
        self.target_size = target_size
    
    def compute(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        size: float,
        baseline_accuracy: float = 1.0,
        baseline_latency: float = 10.0,
        baseline_energy: float = 20.0,
        baseline_size: float = 10.0,
    ) -> float:
        """计算奖励"""
        # 精度奖励（保持精度）
        accuracy_reward = accuracy / baseline_accuracy
        if accuracy < self.target_accuracy:
            accuracy_reward -= (self.target_accuracy - accuracy) * 10  # 惩罚
        
        # 延迟奖励（降低延迟）
        latency_reward = baseline_latency / max(latency, 0.1)
        if latency > self.target_latency:
            latency_reward -= (latency - self.target_latency) / self.target_latency
        
        # 能耗奖励（降低能耗）
        energy_reward = baseline_energy / max(energy, 0.1)
        if energy > self.target_energy:
            energy_reward -= (energy - self.target_energy) / self.target_energy
        
        # 大小奖励（减小模型）
        size_reward = baseline_size / max(size, 0.1)
        if size > self.target_size:
            size_reward -= (size - self.target_size) / self.target_size
        
        # 总奖励
        total_reward = (
            self.accuracy_weight * accuracy_reward +
            self.latency_weight * latency_reward +
            self.energy_weight * energy_reward +
            self.size_weight * size_reward
        )
        
        return total_reward
    
    def compute_pareto_reward(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        size: float,
        pareto_front: list,
    ) -> float:
        """基于Pareto前沿的奖励"""
        # 计算到Pareto前沿的距离
        point = np.array([accuracy, -latency, -energy, -size])
        
        min_distance = float('inf')
        for pareto_point in pareto_front:
            distance = np.linalg.norm(point - pareto_point)
            min_distance = min(min_distance, distance)
        
        # 如果在Pareto前沿上，给予额外奖励
        if self._is_pareto_optimal(point, pareto_front):
            return 10.0
        
        # 否则，奖励与距离成反比
        return 1.0 / (1.0 + min_distance)
    
    def _is_pareto_optimal(self, point: np.ndarray, pareto_front: list) -> bool:
        """检查点是否在Pareto前沿上"""
        for pareto_point in pareto_front:
            if np.all(pareto_point >= point) and np.any(pareto_point > point):
                return False
        return True
```

---

## 8. 训练流程

### 8.1 完整训练流程

```python
class HADMCTrainer:
    """HAD-MC 2.0 完整训练流程"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hal: HardwareAbstractionLayer,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hal = hal
        self.config = config
        
        # 初始化组件
        self.state_dim = config.get('state_dim', 256)
        self.action_dims = config.get('action_dims', {
            'pruning': 100,  # 10 layers × 10 ratios
            'quantization': 40,  # 10 layers × 4 bit_widths
            'distillation': 2,  # temperature, alpha
            'fusion': 60,  # 10 layers × 6 patterns
            'update': 15,  # 3 strategies × 5 ratios
        })
        
        # PPO控制器
        self.ppo = PPOController(
            state_dim=self.state_dim,
            action_dims=self.action_dims,
            lr=config.get('lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_epsilon=config.get('clip_epsilon', 0.2),
        )
        
        # 智能体
        self.pruning_agent = PruningAgent(model, train_loader)
        self.quantization_agent = QuantizationAgent(model, val_loader)
        self.distillation_agent = DistillationAgent(model, copy.deepcopy(model))
        self.fusion_agent = FusionAgent(model)
        self.update_agent = UpdateAgent(model)
        
        # 奖励函数
        self.reward_fn = RewardFunction(
            target_accuracy=config.get('target_accuracy', 0.95),
            target_latency=config.get('target_latency', 5.0),
        )
        
        # 推理引擎
        self.die = DedicatedInferenceEngine(hal)
        
        # 训练历史
        self.history = {
            'rewards': [],
            'accuracies': [],
            'latencies': [],
            'sizes': [],
        }
    
    def train(self, num_episodes: int = 1000, steps_per_episode: int = 50):
        """训练主循环"""
        best_reward = float('-inf')
        best_config = None
        
        for episode in range(num_episodes):
            # 重置环境
            state = self._get_initial_state()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # 选择动作
                actions, log_prob = self.ppo.policy.sample_actions(state)
                value = self.ppo.value(state)
                
                # 执行动作
                next_state, reward, done, info = self._step(actions)
                
                # 存储经验
                self.ppo.buffer.add(state, actions, log_prob, reward, done, value)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # PPO更新
            if len(self.ppo.buffer.states) >= self.config.get('batch_size', 64):
                update_info = self.ppo.update()
                print(f"Episode {episode + 1}: Reward={episode_reward:.4f}, "
                      f"Policy Loss={update_info['policy_loss']:.4f}")
            
            # 记录历史
            self.history['rewards'].append(episode_reward)
            
            # 保存最佳配置
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_config = self._get_current_config()
                self._save_checkpoint(f'best_episode_{episode + 1}.pt')
        
        return best_config
    
    def _get_initial_state(self) -> torch.Tensor:
        """获取初始状态"""
        state = State()
        
        # 模型状态
        state.model_state = self._extract_model_state()
        
        # 硬件状态
        state.hardware_state = self.hal.get_hardware_features()
        
        # 压缩状态（初始为未压缩）
        state.compression_state = {
            'pruning_ratios': [0.0] * 10,
            'bit_widths': [32] * 10,
            'distillation_progress': 0.0,
            'fused_patterns': ['none'] * 10,
            'current_accuracy': 1.0,
            'current_latency': 10.0,
            'current_energy': 20.0,
        }
        
        return state.to_tensor().to(self.ppo.device)
    
    def _step(self, actions: dict) -> tuple:
        """执行一步"""
        # 解析动作
        pruning_config = self._decode_pruning_action(actions['pruning'])
        quant_config = self._decode_quantization_action(actions['quantization'])
        distill_config = self._decode_distillation_action(actions['distillation'])
        fusion_config = self._decode_fusion_action(actions['fusion'])
        update_config = self._decode_update_action(actions['update'])
        
        # 应用压缩
        compressed_model = copy.deepcopy(self.model)
        
        # 1. 剪枝
        compressed_model = self.pruning_agent.prune(pruning_config)
        
        # 2. 量化
        compressed_model = self.quantization_agent.quantize(quant_config)
        
        # 3. 蒸馏
        self.distillation_agent.student = compressed_model
        compressed_model = self.distillation_agent.distill(
            self.train_loader,
            temperature=distill_config['temperature'],
            alpha=distill_config['alpha'],
            epochs=1,
        )
        
        # 4. 融合
        compressed_model = self.fusion_agent.fuse(fusion_config)
        
        # 5. 优化推理
        optimized_model = self.die.optimize(compressed_model, {
            **pruning_config, **quant_config
        })
        
        # 评估
        accuracy = self._evaluate_accuracy(optimized_model)
        latency = self.hal.estimate_latency({**pruning_config, **quant_config})
        energy = self._estimate_energy(optimized_model)
        size = self._compute_model_size(optimized_model)
        
        # 计算奖励
        reward = self.reward_fn.compute(accuracy, latency, energy, size)
        
        # 更新状态
        next_state = self._get_state_from_model(optimized_model)
        
        # 检查是否完成
        done = accuracy < 0.5 or latency > 100  # 失败条件
        
        info = {
            'accuracy': accuracy,
            'latency': latency,
            'energy': energy,
            'size': size,
        }
        
        return next_state, reward, done, info
    
    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """评估模型精度"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total
    
    def _compute_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        return param_size / 1024 / 1024
    
    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        torch.save({
            'policy': self.ppo.policy.state_dict(),
            'value': self.ppo.value.state_dict(),
            'optimizer': self.ppo.optimizer.state_dict(),
            'history': self.history,
        }, filename)
```

---

## 9. 代码架构

### 9.1 目录结构

```
hadmc/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── pruning_agent.py
│   ├── quantization_agent.py
│   ├── distillation_agent.py
│   ├── fusion_agent.py
│   └── update_agent.py
├── controllers/
│   ├── __init__.py
│   ├── ppo_controller.py
│   └── marl_coordinator.py
├── hardware/
│   ├── __init__.py
│   ├── hal.py
│   ├── latency_lut.py
│   └── configs/
│       ├── nvidia_a100.yaml
│       ├── ascend_310.yaml
│       └── hygon_dcu.yaml
├── inference/
│   ├── __init__.py
│   ├── die.py
│   ├── sparse_executor.py
│   └── mixed_precision.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── reward.py
│   └── buffer.py
├── utils/
│   ├── __init__.py
│   ├── state.py
│   ├── action.py
│   └── metrics.py
└── experiments/
    ├── __init__.py
    ├── run_yolov5.py
    ├── run_fsds.py
    └── run_neudet.py
```

---

## 10. 测试与验证

### 10.1 单元测试

```python
import unittest

class TestPruningAgent(unittest.TestCase):
    def setUp(self):
        self.model = create_test_model()
        self.train_loader = create_test_loader()
        self.agent = PruningAgent(self.model, self.train_loader)
    
    def test_compute_importance(self):
        importance = self.agent.compute_importance()
        self.assertIsInstance(importance, dict)
        self.assertTrue(len(importance) > 0)
    
    def test_prune(self):
        pruning_ratios = {'layer1': 0.5, 'layer2': 0.3}
        pruned_model = self.agent.prune(pruning_ratios)
        
        # 验证参数减少
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        self.assertLess(pruned_params, original_params)


class TestPPOController(unittest.TestCase):
    def setUp(self):
        self.state_dim = 256
        self.action_dims = {
            'pruning': 100,
            'quantization': 40,
            'distillation': 2,
            'fusion': 60,
            'update': 15,
        }
        self.ppo = PPOController(self.state_dim, self.action_dims)
    
    def test_sample_actions(self):
        state = torch.randn(1, self.state_dim)
        actions, log_prob = self.ppo.policy.sample_actions(state)
        
        self.assertIsInstance(actions, dict)
        self.assertEqual(len(actions), 5)
        self.assertIsInstance(log_prob, torch.Tensor)
    
    def test_update(self):
        # 添加一些经验
        for _ in range(100):
            state = torch.randn(self.state_dim)
            actions = {
                'pruning': torch.randint(0, 100, (1,)),
                'quantization': torch.randint(0, 40, (1,)),
                'distillation': torch.randn(2),
                'fusion': torch.randint(0, 60, (1,)),
                'update': torch.randint(0, 15, (1,)),
            }
            log_prob = torch.randn(1)
            reward = torch.randn(1).item()
            done = False
            value = torch.randn(1)
            
            self.ppo.buffer.add(state, actions, log_prob, reward, done, value)
        
        # 执行更新
        update_info = self.ppo.update()
        
        self.assertIn('policy_loss', update_info)
        self.assertIn('value_loss', update_info)


if __name__ == '__main__':
    unittest.main()
```

---

*本文档提供了HAD-MC 2.0算法提升的完整实现指南，包括MARL框架、PPO控制器、硬件抽象层、专用推理引擎、五个智能体的详细设计、奖励函数、训练流程和代码架构。*
