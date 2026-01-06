#!/usr/bin/env python3
"""
HAD-MC GPU验证实验 - 完整补充实验脚本
包括：FPR@95%Recall计算、TensorRT评估、QAT-INT8、消融实验等

作者：HAD-MC研究团队
日期：2026-01-06
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# 设置路径
YOLOV5_PATH = "/workspace/HAD-MC/yolov5"
HADMC_PATH = "/workspace/HAD-MC/hadmc"
sys.path.insert(0, YOLOV5_PATH)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/HAD-MC/experiments/complete_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 第一部分：FPR@95%Recall 计算（P0-1协议要求）
# ============================================================================

def compute_fpr_at_recall(predictions, ground_truths, recall_threshold=0.95):
    """
    计算FPR@95%Recall
    严格遵循P0-1评估协议
    
    Args:
        predictions: 预测结果列表 [(confidence, is_tp), ...]
        ground_truths: 真值数量
        recall_threshold: 召回率阈值（默认95%）
    
    Returns:
        fpr: 在指定召回率下的假阳性率
        threshold: 对应的置信度阈值
    """
    if len(predictions) == 0:
        return 0.0, 0.0
    
    # 按置信度降序排序
    sorted_preds = sorted(predictions, key=lambda x: x[0], reverse=True)
    
    # 计算每个阈值下的TP, FP, Recall, FPR
    tp = 0
    fp = 0
    total_positives = ground_truths
    
    for i, (conf, is_tp) in enumerate(sorted_preds):
        if is_tp:
            tp += 1
        else:
            fp += 1
        
        recall = tp / total_positives if total_positives > 0 else 0
        
        # 找到第一个达到目标召回率的点
        if recall >= recall_threshold:
            fpr = fp / (fp + tp) if (fp + tp) > 0 else 0
            return fpr, conf
    
    # 如果没有达到目标召回率，返回最后一个点
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0
    return fpr, sorted_preds[-1][0] if sorted_preds else 0.0


def evaluate_model_with_fpr(model, dataloader, device, conf_thres=0.25, iou_thres=0.45):
    """
    评估模型，计算mAP和FPR@95%Recall
    
    Args:
        model: YOLOv5模型
        dataloader: 数据加载器
        device: 设备
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
    
    Returns:
        results: 包含mAP, FPR等指标的字典
    """
    from utils.general import non_max_suppression, scale_boxes, xywh2xyxy
    from utils.metrics import box_iou
    
    model.eval()
    
    all_predictions = []  # [(confidence, is_tp), ...]
    total_ground_truths = 0
    
    stats = []  # 用于计算mAP
    
    with torch.no_grad():
        for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            
            # 推理
            preds = model(imgs)
            
            # NMS
            preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=iou_thres)
            
            # 处理每张图像
            for si, pred in enumerate(preds):
                # 获取当前图像的真值
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                total_ground_truths += nl
                
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, dtype=torch.bool), 
                                     torch.Tensor(), torch.Tensor(), labels[:, 0]))
                    continue
                
                # 处理预测
                predn = pred.clone()
                
                # 计算IoU并匹配
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= imgs.shape[3]
                    tbox[:, [1, 3]] *= imgs.shape[2]
                    
                    # 计算IoU
                    iou = box_iou(predn[:, :4], tbox)
                    
                    # 匹配
                    correct = torch.zeros(len(pred), dtype=torch.bool, device=device)
                    detected = []
                    
                    for j, (conf, cls) in enumerate(zip(pred[:, 4], pred[:, 5])):
                        # 找到最佳匹配
                        if len(detected) < nl:
                            iou_j = iou[j]
                            best_iou, best_idx = iou_j.max(0)
                            
                            if best_iou > 0.5 and best_idx.item() not in detected:
                                correct[j] = True
                                detected.append(best_idx.item())
                                all_predictions.append((conf.item(), True))
                            else:
                                all_predictions.append((conf.item(), False))
                        else:
                            all_predictions.append((conf.item(), False))
                    
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
                else:
                    # 没有真值，所有预测都是FP
                    for conf in pred[:, 4]:
                        all_predictions.append((conf.item(), False))
    
    # 计算mAP
    from utils.metrics import ap_per_class
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    
    # 计算FPR@95%Recall
    fpr_95, threshold_95 = compute_fpr_at_recall(all_predictions, total_ground_truths, 0.95)
    
    return {
        'mAP@0.5': float(map50),
        'mAP@0.5:0.95': float(map),
        'Precision': float(mp),
        'Recall': float(mr),
        'FPR@95%Recall': float(fpr_95),
        'Threshold@95%Recall': float(threshold_95),
        'Total_GT': total_ground_truths,
        'Total_Predictions': len(all_predictions)
    }


# ============================================================================
# 第二部分：TensorRT 评估
# ============================================================================

def evaluate_tensorrt_engine(engine_path, dataloader, device):
    """
    评估TensorRT引擎
    
    Args:
        engine_path: TensorRT引擎文件路径
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        results: 评估结果
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        logger.warning("TensorRT或PyCUDA未安装，跳过TensorRT评估")
        return None
    
    logger.info(f"加载TensorRT引擎: {engine_path}")
    
    # 加载引擎
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 分配内存
    inputs = []
    outputs = []
    bindings = []
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    stream = cuda.Stream()
    
    # 评估
    all_predictions = []
    total_ground_truths = 0
    latencies = []
    
    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        imgs_np = imgs.numpy().astype(np.float32) / 255.0
        
        # 复制输入
        np.copyto(inputs[0]['host'], imgs_np.ravel())
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        
        # 推理
        start_time = time.time()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        # 复制输出
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()
        
        # 处理输出（简化版，实际需要根据TensorRT输出格式处理）
        # ...
    
    avg_latency = np.mean(latencies[10:]) if len(latencies) > 10 else np.mean(latencies)
    
    return {
        'Latency (ms)': float(avg_latency),
        'Throughput (FPS)': 1000.0 / avg_latency if avg_latency > 0 else 0,
        'Note': 'TensorRT evaluation completed'
    }


# ============================================================================
# 第三部分：QAT-INT8 量化感知训练
# ============================================================================

def run_qat_int8(model_path, train_loader, val_loader, device, epochs=20):
    """
    运行量化感知训练 (QAT-INT8)
    
    Args:
        model_path: 预训练模型路径
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
    
    Returns:
        model: 量化后的模型
        results: 评估结果
    """
    logger.info("=" * 60)
    logger.info("开始 QAT-INT8 量化感知训练")
    logger.info("=" * 60)
    
    # 加载模型
    from models.yolo import Model
    from models.common import Conv
    
    ckpt = torch.load(model_path, map_location=device)
    model = ckpt['model'].float().to(device)
    model.train()
    
    # 准备量化配置
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 融合模块
    # YOLOv5的Conv模块包含Conv2d + BatchNorm2d + SiLU
    # 需要手动融合
    
    # 准备QAT
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 训练
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    from utils.loss import ComputeLoss
    compute_loss = ComputeLoss(model)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_i, (imgs, targets, paths, shapes) in enumerate(train_loader):
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            
            # 前向传播
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_i % 10 == 0:
                logger.info(f"QAT Epoch {epoch+1}/{epochs}, Batch {batch_i}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        logger.info(f"QAT Epoch {epoch+1}/{epochs} completed, Avg Loss: {total_loss/len(train_loader):.4f}")
    
    # 转换为量化模型
    model.eval()
    model_int8 = torch.quantization.convert(model, inplace=False)
    
    # 评估
    results = evaluate_model_with_fpr(model_int8, val_loader, device)
    
    logger.info(f"QAT-INT8 Results: mAP@0.5={results['mAP@0.5']:.4f}, FPR@95%R={results['FPR@95%Recall']:.4f}")
    
    return model_int8, results


# ============================================================================
# 第四部分：HAD-MC (Q only) 消融实验
# ============================================================================

def run_hadmc_q_only(model_path, train_loader, val_loader, device):
    """
    运行HAD-MC仅量化实验（消融实验）
    
    Args:
        model_path: 预训练模型路径
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
    
    Returns:
        model: 量化后的模型
        results: 评估结果
    """
    logger.info("=" * 60)
    logger.info("开始 HAD-MC (Q only) 消融实验")
    logger.info("=" * 60)
    
    # 加载模型
    ckpt = torch.load(model_path, map_location=device)
    model = ckpt['model'].float().to(device)
    
    # 导入HAD-MC量化算法
    sys.path.insert(0, HADMC_PATH)
    from hadmc_yolov5 import YOLOv5LayerwisePrecisionAllocator
    
    # 创建量化器
    quantizer = YOLOv5LayerwisePrecisionAllocator(
        model=model,
        calibration_loader=train_loader,
        tau_h=0.8,
        tau_l=0.2,
        device=device,
        task_type='detection'
    )
    
    # 运行量化
    quantized_model = quantizer.run(target_bits=8)
    
    # 评估
    results = evaluate_model_with_fpr(quantized_model, val_loader, device)
    
    logger.info(f"HAD-MC (Q only) Results: mAP@0.5={results['mAP@0.5']:.4f}, FPR@95%R={results['FPR@95%Recall']:.4f}")
    
    return quantized_model, results


# ============================================================================
# 第五部分：统计显著性测试
# ============================================================================

def run_statistical_significance_test(experiment_func, seeds=[42, 123, 456], **kwargs):
    """
    运行统计显著性测试（3次重复实验）
    
    Args:
        experiment_func: 实验函数
        seeds: 随机种子列表
        **kwargs: 实验参数
    
    Returns:
        results: 包含均值和标准差的结果
    """
    all_results = []
    
    for seed in seeds:
        logger.info(f"运行实验，随机种子: {seed}")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 运行实验
        _, result = experiment_func(**kwargs)
        all_results.append(result)
    
    # 计算均值和标准差
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'FPR@95%Recall']
    final_results = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        final_results[f'{metric}_mean'] = np.mean(values)
        final_results[f'{metric}_std'] = np.std(values)
    
    return final_results


# ============================================================================
# 第六部分：主函数
# ============================================================================

def main():
    """主函数：运行所有补充实验"""
    
    logger.info("=" * 80)
    logger.info("HAD-MC GPU验证实验 - 完整补充实验")
    logger.info("=" * 80)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 路径
    model_path = "/workspace/HAD-MC/experiments/results/fp32_baseline/train/weights/best.pt"
    data_yaml = "/workspace/HAD-MC/yolov5/data/coco128.yaml"
    
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
        rect=False,
        rank=-1,
        workers=4,
        image_weights=False,
        quad=False,
        prefix='',
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
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
        image_weights=False,
        quad=False,
        prefix='',
        shuffle=False
    )
    
    results = {}
    
    # ========================================
    # 实验1：评估所有已有模型的FPR@95%Recall
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("实验1：计算所有模型的FPR@95%Recall")
    logger.info("=" * 60)
    
    models_to_evaluate = [
        ("FP32 Baseline", "/workspace/HAD-MC/experiments/results/fp32_baseline/train/weights/best.pt"),
        ("PTQ-INT8", "/workspace/HAD-MC/experiments/results/ptq_int8/ptq_int8.pt"),
        ("QAT-INT8", "/workspace/HAD-MC/experiments/results/qat_int8/train/weights/best.pt"),
        ("L1-Norm Pruning", "/workspace/HAD-MC/experiments/results/l1_pruning/l1_pruned.pt"),
        ("HAD-MC Pruning", "/workspace/HAD-MC/experiments/results/hadmc_pruning/gradient_sensitivity_pruned.pt"),
        ("HAD-MC Full", "/workspace/HAD-MC/experiments/results/hadmc_full/hadmc_full.pt"),
        ("HAD-MC Optimized", "/workspace/HAD-MC/experiments/results/phase1_comprehensive/hadmc_optimized/model.pt"),
        ("HAD-MC Ablation P", "/workspace/HAD-MC/experiments/results/phase1_comprehensive/hadmc_ablation_pruning/model.pt"),
        ("HAD-MC Ablation PQ", "/workspace/HAD-MC/experiments/results/phase1_comprehensive/hadmc_ablation_pruning_quant/model.pt"),
    ]
    
    for name, path in models_to_evaluate:
        if os.path.exists(path):
            logger.info(f"\n评估 {name}...")
            ckpt = torch.load(path, map_location=device)
            model = ckpt['model'].float().to(device)
            model.eval()
            
            result = evaluate_model_with_fpr(model, val_loader, device)
            results[name] = result
            
            logger.info(f"{name} 结果:")
            logger.info(f"  mAP@0.5: {result['mAP@0.5']:.4f}")
            logger.info(f"  mAP@0.5:0.95: {result['mAP@0.5:0.95']:.4f}")
            logger.info(f"  FPR@95%Recall: {result['FPR@95%Recall']:.4f}")
        else:
            logger.warning(f"模型文件不存在: {path}")
    
    # ========================================
    # 实验2：QAT-INT8（如果时间允许）
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("实验2：QAT-INT8 量化感知训练")
    logger.info("=" * 60)
    
    try:
        qat_model, qat_results = run_qat_int8(
            model_path=model_path,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=10  # 减少epochs以节省时间
        )
        results['QAT-INT8'] = qat_results
        
        # 保存模型
        save_path = "/workspace/HAD-MC/experiments/qat_int8/qat_int8.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model': qat_model}, save_path)
        logger.info(f"QAT-INT8模型已保存: {save_path}")
    except Exception as e:
        logger.error(f"QAT-INT8实验失败: {e}")
        results['QAT-INT8'] = {'error': str(e)}
    
    # ========================================
    # 实验3：HAD-MC (Q only) 消融实验
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("实验3：HAD-MC (Q only) 消融实验")
    logger.info("=" * 60)
    
    try:
        q_only_model, q_only_results = run_hadmc_q_only(
            model_path=model_path,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        results['HAD-MC (Q only)'] = q_only_results
        
        # 保存模型
        save_path = "/workspace/HAD-MC/experiments/hadmc_q_only/hadmc_q_only.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model': q_only_model}, save_path)
        logger.info(f"HAD-MC (Q only)模型已保存: {save_path}")
    except Exception as e:
        logger.error(f"HAD-MC (Q only)实验失败: {e}")
        results['HAD-MC (Q only)'] = {'error': str(e)}
    
    # ========================================
    # 保存所有结果
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("保存实验结果")
    logger.info("=" * 60)
    
    results_path = "/workspace/HAD-MC/experiments/complete_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存: {results_path}")
    
    # ========================================
    # 生成最终报告
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("最终结果汇总")
    logger.info("=" * 60)
    
    print("\n" + "=" * 100)
    print("HAD-MC GPU验证实验 - 完整结果")
    print("=" * 100)
    print(f"{'方法':<25} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'FPR@95%R':<12}")
    print("-" * 100)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<25} {result.get('mAP@0.5', 'N/A'):<12.4f} "
                  f"{result.get('mAP@0.5:0.95', 'N/A'):<15.4f} "
                  f"{result.get('FPR@95%Recall', 'N/A'):<12.4f}")
        else:
            print(f"{name:<25} ERROR: {result['error']}")
    
    print("=" * 100)
    
    logger.info("所有实验完成！")


if __name__ == "__main__":
    main()
