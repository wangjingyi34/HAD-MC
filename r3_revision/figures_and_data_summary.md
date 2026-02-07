# 图表和实验数据摘要

## 生成的图表列表
1. fig_sota_comparison.png/pdf - SOTA方法对比（准确率、压缩率、加速比）
2. fig_ablation_study.png/pdf - 消融研究（各组件准确率和延迟）
3. fig_ppo_vs_dqn.png/pdf - PPO vs DQN训练曲线
4. fig_cross_dataset.png/pdf - 跨数据集验证
5. fig_cross_platform.png/pdf - 跨平台延迟 + 批量吞吐量
6. fig_training_convergence.png/pdf - 训练收敛曲线
7. fig_latency_lut.png/pdf - 延迟查找表
8. fig_radar_comparison.png/pdf - 多目标雷达图
9. fig_pareto_front.png/pdf - Pareto前沿

## 关键实验数据（用于论文表格）

### Table: SOTA Comparison on NEU-DET (A100 GPU)
| Method | Accuracy(%) | Params | Size(MB) | Latency(ms) | Speedup | Compression |
|--------|-------------|--------|----------|-------------|---------|-------------|
| Baseline | 100.00 | 11,171,910 | 42.62 | 2.04 | 1.00x | 0% |
| AMC | 100.00 | 2,796,582 | 10.67 | 2.02 | 1.01x | 75.0% |
| HAQ | 100.00 | 4,367,406 | 16.66(INT8:4.17) | 2.04 | 1.00x | 60.9% |
| DECORE | 99.72 | 2,796,582 | 10.67 | 1.99 | 1.02x | 75.0% |
| **HAD-MC 2.0** | **100.00** | **2,794,182** | **10.66(INT8:2.66)** | **1.49** | **1.37x** | **75.0%** |

### Table: Ablation Study
| Component | Accuracy(%) | Params | Latency(ms) |
|-----------|-------------|--------|-------------|
| Baseline | 100.00 | 11,171,910 | 2.04 |
| Pruning Only | 100.00 | 2,796,582 | 2.01 |
| Quantization Only | 100.00 | 11,171,910 | 2.03 |
| Distillation Only | 100.00 | 2,796,582 | 2.00 |
| Pruning+Quantization | 100.00 | 2,796,582 | 1.99 |
| Pruning+Distillation | 100.00 | 2,796,582 | 2.01 |
| Full HAD-MC 2.0 | 100.00 | 2,794,182 | 1.49 |

### Table: PPO vs DQN
| Controller | Best Reward | Final Reward | Best Accuracy(%) |
|------------|-------------|--------------|------------------|
| PPO (Ours) | 3.656 | 3.396 | 100.00 |
| DQN | 3.578 | 3.407 | 99.72 |

### Table: Cross-Dataset
| Dataset | Baseline Acc(%) | Compressed Acc(%) | Compression |
|---------|-----------------|-------------------|-------------|
| NEU-DET (6-class) | 100.00 | 100.00 | 75.0% |
| FS-DS (3-class) | 100.00 | 100.00 | 75.0% |
| Financial (2-class) | 100.00 | 100.00 | 70.2% |

### Table: Cross-Platform Latency
| Platform | Latency(ms) | Power(W) | Energy Efficiency |
|----------|-------------|----------|-------------------|
| NVIDIA A100 | 2.06 | 250 | 0.515 mJ/inf |
| Jetson Orin | 10.69 | 15 | 0.160 mJ/inf |
| Ascend 310 | 6.37 | 8 | 0.051 mJ/inf |
| Hygon DCU | 8.22 | 150 | 1.233 mJ/inf |

### Table: Latency LUT (A100)
| Layer | Latency(μs) | Params |
|-------|-------------|--------|
| Conv2d 3×3 (3→64) | 54.1 | 1,792 |
| Conv2d 3×3 (64→128) | 72.5 | 73,856 |
| Conv2d 3×3 (128→256) | 73.1 | 295,168 |
| Conv2d 3×3 (256→512) | 103.7 | 1,180,160 |
| Conv2d 1×1 (64→64) | 59.1 | 4,160 |
| Conv2d 1×1 (128→128) | 59.4 | 16,512 |
| Linear (512→256) | 51.0 | 131,328 |
| Linear (256→128) | 50.6 | 32,896 |

### Batch Throughput (A100)
| Batch Size | Latency(ms) | Throughput(FPS) |
|------------|-------------|-----------------|
| 1 | 2.06 | 487 |
| 4 | 2.09 | 1,911 |
| 8 | 2.12 | 3,770 |
| 16 | 2.12 | 7,540 |
| 32 | 3.66 | 8,748 |
| 64 | 6.61 | 9,678 |
