#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成混淆矩阵图片的脚本
读取error_samples.csv文件，生成混淆矩阵可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import os
from tqdm import *

def generate_confusion_matrix(csv_file, output_file=None):
    """
    从CSV文件生成混淆矩阵图片
    
    Args:
        csv_file (str): CSV文件路径
        output_file (str): 输出图片文件路径，默认为'confusion_matrix.png'
    """
    
    # 读取CSV文件
    print(f"正在读取文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 检查必要的列是否存在
    required_columns = ['true_label', 'predicted_label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    # 提取真实标签和预测标签
    true_labels = df['true_label'].astype(str)
    predicted_labels = df['predicted_label'].astype(str)
    
    # 获取所有唯一的标签
    all_labels = sorted(set(true_labels) | set(predicted_labels))
    
    # 生成混淆矩阵
    print("正在生成混淆矩阵...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
    
    # 设置输出文件名
    if output_file is None:
        output_file = 'confusion_matrix.png'
    
    # 创建可视化
    plt.figure(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=all_labels,
                yticklabels=all_labels,
                cbar_kws={'label': '样本数量'})
    
    # 设置标题和标签（使用英文避免中文字符问题）
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 设置颜色条标签为英文
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Number of Samples', rotation=270, labelpad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {output_file}")
    
    # 显示一些统计信息
    total_samples = len(df)
    correct_predictions = np.sum(np.diag(cm))
    accuracy = correct_predictions / total_samples
    
    print(f"\n统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"唯一标签数量: {len(all_labels)}")
    
    return cm, all_labels

def main():
    parser = argparse.ArgumentParser(description='从CSV文件生成混淆矩阵图片')
    parser.add_argument('--input', '-i', default='error_samples.csv', 
                       help='输入CSV文件路径 (默认: error_samples.csv)')
    parser.add_argument('--output', '-o', default='confusion_matrix.png',
                       help='输出图片文件路径 (默认: confusion_matrix.png)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        return
    
    try:
        # 生成混淆矩阵
        cm, labels = generate_confusion_matrix(args.input, args.output)
        
        print("\n混淆矩阵生成完成！")
        
    except Exception as e:
        print(f"生成混淆矩阵时出错: {e}")
        print("请确保CSV文件包含 'true_label' 和 'predicted_label' 列")

if __name__ == "__main__":
    main()
