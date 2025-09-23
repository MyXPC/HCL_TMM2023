#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据error_samples.csv文件组织预测错误的样本
功能：读取CSV文件，创建以真实类别名为名称的文件夹，并将预测错误的文件复制到对应文件夹
"""

import os
import csv
import shutil
import argparse
from pathlib import Path

def build_filename_to_info_map(dataset_path):
    """
    构建文件名到类别信息的映射字典
    返回：filename_to_info字典，包含类别编号和类别名称
    """
    # 获取所有类别文件夹
    category_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            category_dirs.append(item)
    
    # 按文件夹名排序
    category_dirs.sort()
    
    # 构建文件名到完整路径的映射
    filename_to_path = {}
    
    # 遍历所有类别目录
    for category_dir in category_dirs:
        category_path = os.path.join(dataset_path, category_dir)
        if not os.path.isdir(category_path):
            continue
            
        # 遍历该类别目录中的所有文件
        for root, dirs, files in os.walk(category_path):
            for filename in files:
                full_path = os.path.join(root, filename)
                filename_to_path[filename] = full_path
    
    return filename_to_path

def organize_error_samples(csv_file, dataset_path, output_dir):
    """
    组织预测错误的样本到对应类别的文件夹中
    
    Args:
        csv_file (str): error_samples.csv文件路径
        dataset_path (str): 数据集根目录路径
        output_dir (str): 输出目录，将在此创建类别文件夹
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件名到路径的映射
    print("正在构建文件名到路径的映射...")
    filename_to_path = build_filename_to_info_map(dataset_path)
    print(f"映射构建完成，共找到 {len(filename_to_path)} 个文件")
    
    # 读取CSV文件
    error_samples = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            error_samples.append(row)
    
    print(f"读取到 {len(error_samples)} 个错误样本")
    
    # 按真实类别分组
    category_groups = {}
    for sample in error_samples:
        true_class = sample['true_class_name']
        if true_class not in category_groups:
            category_groups[true_class] = []
        category_groups[true_class].append(sample)
    
    print(f"共发现 {len(category_groups)} 个不同的真实类别")
    
    # 为每个类别创建文件夹并复制文件
    total_copied = 0
    for true_class, samples in category_groups.items():
        # 创建类别文件夹（处理可能存在的非法文件名字符）
        safe_class_name = "".join(c for c in true_class if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        class_dir = os.path.join(output_dir, safe_class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 复制每个错误样本到对应类别文件夹，并使用预测类别名重命名
        for sample in samples:
            filename = sample['filename']
            predicted_class = sample['predicted_class_name']
            
            # 从映射中获取文件的完整路径
            if filename in filename_to_path:
                source_path = filename_to_path[filename]
                
                # 生成新的文件名：使用预测类别名 + 原始文件扩展名
                file_ext = os.path.splitext(filename)[1]
                predicted_safe_name = "".join(c for c in predicted_class if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                new_filename = f"{predicted_safe_name}{file_ext}"
                dest_path = os.path.join(class_dir, new_filename)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    total_copied += 1
                    if total_copied % 100 == 0:
                        print(f"已复制 {total_copied} 个文件...")
                except Exception as e:
                    print(f"复制文件 {filename} 时出错: {e}")
            else:
                print(f"警告: 无法找到文件 {filename}")
    
    print(f"完成! 共复制 {total_copied} 个文件到 {output_dir}")
    print(f"创建了 {len(category_groups)} 个类别文件夹")

def main():
    parser = argparse.ArgumentParser(description='组织预测错误的样本到对应类别的文件夹中')
    parser.add_argument('--csv', type=str, required=True, 
                       help='error_samples.csv文件路径')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集根目录路径（包含类别子文件夹）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录，将在此创建类别文件夹')
    
    args = parser.parse_args()
    
    # 检查CSV文件是否存在
    if not os.path.exists(args.csv):
        print(f"错误: CSV文件不存在: {args.csv}")
        return
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.dataset):
        print(f"错误: 数据集目录不存在: {args.dataset}")
        return
    
    organize_error_samples(args.csv, args.dataset, args.output)

if __name__ == "__main__":
    main()
