# -*- coding: utf-8 -*-
import pandas as pd
from tabulate import tabulate

def generate_comparison_table():
    """
    生成HAD-MC与现有方法(HAQ, AutoML, TensorRT)的详细对比表。
    使用pandas DataFrame存储数据，并以Markdown格式输出。
    """
    # 定义对比数据
    data = {
        "Feature": [
            "主要焦点 (Primary Focus)",
            "硬件感知度 (Hardware Awareness)",
            "优化技术 (Optimization Technique)",
            "目标阶段 (Target Stage)",
            "通用性/灵活性 (Generality/Flexibility)",
            "核心优势 (Key Advantage)"
        ],
        "HAD-MC": [
            "硬件感知动态模型压缩与部署",
            "高 (动态、显式考虑目标硬件)",
            "动态剪枝、压缩、可能包含量化",
            "部署/推理 (Deployment/Inference)",
            "针对特定压缩框架/方法",
            "动态适应硬件和运行时条件，实现高效部署"
        ],
        "HAQ": [
            "硬件感知量化",
            "高 (量化方案针对目标硬件定制)",
            "混合精度量化 (Mixed-Precision Quantization)",
            "训练/后训练量化 (Training/Post-Training Quantization)",
            "针对特定量化方法",
            "为特定硬件提供最优量化方案，最大化效率"
        ],
        "AutoML": [
            "自动化机器学习流程设计",
            "可变/中 (可包含硬件感知NAS/HPO)",
            "神经架构搜索 (NAS)、超参数优化 (HPO)",
            "训练/模型设计 (Training/Model Design)",
            "广泛的方法论，适用于各种任务",
            "自动搜索最优模型架构和超参数，减少人工干预"
        ],
        "TensorRT": [
            "高性能深度学习推理优化与部署",
            "高 (专为NVIDIA GPU优化)",
            "图优化、层融合、内核自动调优、精度校准",
            "部署/推理 (Deployment/Inference)",
            "特定于NVIDIA硬件和深度学习模型",
            "在NVIDIA GPU上实现最大推理速度和吞吐量"
        ]
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 打印Markdown格式的表格
    # 使用tabulate库以'pipe'格式（即Markdown表格格式）输出
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    print("--- 对比表 (Markdown 格式) ---")
    print(markdown_table)
    print("-----------------------------")
    
    # 返回Markdown表格字符串，以便在测试结果中使用
    return markdown_table

if __name__ == "__main__":
    # 执行生成并打印表格
    generate_comparison_table()

# 详细注释:
# 1. 导入pandas和tabulate库，用于数据处理和表格格式化。
# 2. 定义generate_comparison_table函数，封装核心逻辑。
# 3. data字典存储对比数据，键为列名，值为对应列的数据列表。
# 4. Feature列定义了对比的维度，包括焦点、硬件感知度、技术、阶段、通用性和优势。
# 5. HAD-MC、HAQ、AutoML、TensorRT列分别填充了对应方法在各个维度上的描述。
# 6. pd.DataFrame(data)将字典转换为结构化的DataFrame。
# 7. tabulate(df, headers='keys', tablefmt='pipe', showindex=False)将DataFrame转换为Markdown表格字符串。
#    - headers='keys' 使用字典的键作为表头。
#    - tablefmt='pipe' 指定输出格式为Markdown表格（使用|分隔）。
#    - showindex=False 隐藏DataFrame的索引。
# 8. __name__ == "__main__": 块确保脚本可以直接运行，并打印表格。
# 9. 函数返回Markdown表格字符串，便于后续测试结果捕获。
