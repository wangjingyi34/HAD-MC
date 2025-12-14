# -*- coding: utf-8 -*-
"""
文件: p1_5_symbol_table_generator.py
描述: HAD-MC项目P1-5任务的实现。
      该脚本定义了论文中使用的数学符号及其含义，并将其生成为Markdown格式的符号表文件。
"""

import os
from typing import Dict, List, Tuple

# 定义HAD-MC项目（假设为高空无人机模型控制）相关的数学符号
# 结构: (符号, 含义/描述, 类型/单位)
SYMBOL_DEFINITIONS: List[Tuple[str, str, str]] = [
    ("$\mathbf{x}_k$", "系统在时刻 $k$ 的状态向量", "$\mathbb{R}^n$"),
    ("$\mathbf{u}_k$", "系统在时刻 $k$ 的控制输入向量", "$\mathbb{R}^m$"),
    ("$f(\cdot)$", "系统动力学模型（状态转移函数）", "函数"),
    ("$J(\mathbf{x}, \mathbf{u})$", "代价函数/目标函数", "标量"),
    ("$\mathbf{Q}$", "状态惩罚项的权重矩阵", "正定矩阵"),
    ("$\mathbf{R}$", "控制输入惩罚项的权重矩阵", "正定矩阵"),
    ("$\mathcal{X}$", "状态空间约束集合", "集合"),
    ("$\mathcal{U}$", "控制输入约束集合", "集合"),
    ("$T$", "控制时域长度", "整数"),
    ("$\hat{\mathbf{x}}$", "状态向量的估计值", "$\mathbb{R}^n$"),
    ("$\delta t$", "采样时间间隔", "秒 (s)"),
    ("$v$", "无人机速度", "米/秒 (m/s)"),
    ("$\psi$", "无人机航向角", "弧度 (rad)"),
]

def generate_symbol_table(symbols: List[Tuple[str, str, str]], output_path: str) -> None:
    """
    将符号定义列表转换为Markdown格式的表格并写入文件。

    Args:
        symbols: 包含 (符号, 含义, 类型/单位) 元组的列表。
        output_path: 符号表文件的输出路径。
    """
    print(f"开始生成符号表到: {output_path}")

    # 构造Markdown表格内容
    header = "| 符号 | 含义/描述 | 类型/单位 |\n"
    separator = "| :--- | :--- | :--- |\n"
    
    rows = []
    for symbol, meaning, unit in symbols:
        # 确保Markdown表格格式正确，符号和含义中可能包含LaTeX
        row = f"| {symbol} | {meaning} | {unit} |\n"
        rows.append(row)

    content = "# 数学符号表\n\n"
    content += "本文档列出了HAD-MC项目论文中使用的主要数学符号及其定义。\n\n"
    content += header
    content += separator
    content += "".join(rows)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"符号表生成成功。文件路径: {output_path}")
    except IOError as e:
        print(f"写入文件失败: {e}")
        raise

def test_symbol_table_generation(output_file: str) -> str:
    """
    测试符号表生成功能，并返回测试结果字符串。
    """
    print("--- 开始测试符号表生成 ---")
    
    # 1. 调用生成函数
    try:
        generate_symbol_table(SYMBOL_DEFINITIONS, output_file)
    except Exception as e:
        return f"测试失败: 生成过程中发生异常: {e}"

    # 2. 验证文件是否存在
    if not os.path.exists(output_file):
        return f"测试失败: 输出文件 {output_file} 不存在。"

    # 3. 验证文件内容（简单检查行数和内容）
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.readlines()
    except IOError:
        return f"测试失败: 无法读取输出文件 {output_file}。"

    # 预期行数: 标题(1) + 空行(1) + 描述(1) + 空行(1) + 表头(1) + 分隔线(1) + 数据行(len(SYMBOL_DEFINITIONS))
    expected_min_lines = 6 + len(SYMBOL_DEFINITIONS)
    if len(content) < expected_min_lines:
        return f"测试失败: 文件行数不足。预期最少 {expected_min_lines} 行，实际 {len(content)} 行。"

    # 简单检查表头
    if not "| 符号 | 含义/描述 | 类型/单位 |" in content[4]:
        return "测试失败: 表头内容不正确。"

    print("--- 测试完成 ---")
    return f"状态: 成功; 文件: {output_file}; 符号数量: {len(SYMBOL_DEFINITIONS)}"

if __name__ == "__main__":
    # 定义输出文件路径
    OUTPUT_FILE_PATH = "/home/ubuntu/symbol_table.md"
    
    # 执行测试
    test_result = test_symbol_table_generation(OUTPUT_FILE_PATH)
    
    # 打印最终测试结果
    print(f"\n最终测试结果:\n{test_result}")

    # 额外的验证步骤：读取生成的文件内容，以便在提交说明中引用
    if "成功" in test_result:
        try:
            with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                print("\n--- 生成的符号表内容预览 (前5行) ---")
                for i, line in enumerate(f):
                    if i < 5:
                        print(line.strip())
                    else:
                        break
                print("---------------------------------------")
        except Exception:
            pass # 忽略读取失败
