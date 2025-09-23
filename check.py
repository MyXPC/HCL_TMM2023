import os
import csv
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
    
    # 创建类别名称到索引的映射（从0开始）
    class_to_idx = {category_dirs[i]: i for i in range(len(category_dirs))}
    
    # 创建类别编号到类别名称的映射
    idx_to_class = {i: category_dirs[i] for i in range(len(category_dirs))}
    
    # 构建文件名到类别信息的映射
    filename_to_info = {}
    
    # 遍历所有类别目录
    for category_dir in category_dirs:
        category_path = os.path.join(dataset_path, category_dir)
        if not os.path.isdir(category_path):
            continue
            
        # 获取类别索引和标签
        category_index = class_to_idx[category_dir]
        category_label = str(category_index).zfill(4)
        
        # 遍历该类别目录中的所有文件
        for root, dirs, files in os.walk(category_path):
            for filename in files:
                filename_to_info[filename] = {
                    'label': category_label,
                    'class_name': category_dir
                }
    
    return filename_to_info, idx_to_class

def check_predictions(csv_file, dataset_path, output_file):
    """
    检查预测结果的准确性并输出错误样本
    """
    print("正在构建文件名到类别信息的映射...")
    filename_to_info, idx_to_class = build_filename_to_info_map(dataset_path)
    print(f"映射构建完成，共找到 {len(filename_to_info)} 个文件")
    
    correct_count = 0
    total_count = 0
    error_samples = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            filename = row['filename']
            predicted_label = row['class_label']
            
            # 从映射中获取真实标签信息
            if filename in filename_to_info:
                true_info = filename_to_info[filename]
                true_label = true_info['label']
                true_class_name = true_info['class_name']
                
                # 获取预测的类别名称
                predicted_class_name = ""
                try:
                    predicted_idx = int(predicted_label)
                    if predicted_idx in idx_to_class:
                        predicted_class_name = idx_to_class[predicted_idx]
                    else:
                        predicted_class_name = "未知类别"
                except ValueError:
                    predicted_class_name = "无效预测编号"
                
                total_count += 1
                
                if predicted_label == true_label:
                    correct_count += 1
                else:
                    error_samples.append({
                        'filename': filename,
                        'predicted_label': predicted_label,
                        'predicted_class_name': predicted_class_name,
                        'true_label': true_label,
                        'true_class_name': true_class_name
                    })
                    print(f"错误: {filename} - 预测: {predicted_label} ({predicted_class_name}), 真实: {true_label} ({true_class_name})")
            else:
                print(f"警告: 无法找到文件 {filename} 的真实标签")
    
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # 输出错误样本到文件（包含预测和真实的类别名称）
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'predicted_label', 'predicted_class_name', 'true_label', 'true_class_name']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in error_samples:
            writer.writerow(sample)
    
    # 输出统计信息
    print(f"\n=== 预测准确性统计 ===")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"错误预测数: {len(error_samples)}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"错误样本已保存到: {output_file}（包含预测和真实的类别名称）")
    
    return accuracy, error_samples

def main():
    parser = argparse.ArgumentParser(description='检查预测结果的准确性')
    parser.add_argument('--csv_file', default='my_predictions.csv', help='预测结果CSV文件路径')
    parser.add_argument('--dataset_path', default='../../dataset/val400', help='数据集路径')
    parser.add_argument('--output_file', default='error_samples.csv', help='错误样本输出文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"错误: CSV文件 {args.csv_file} 不存在")
        return
    
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在")
        return
    
    print(f"开始检查预测准确性...")
    print(f"CSV文件: {args.csv_file}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出文件: {args.output_file}")
    print("-" * 50)
    
    accuracy, error_samples = check_predictions(
        args.csv_file, args.dataset_path, args.output_file
    )

if __name__ == "__main__":
    main()
