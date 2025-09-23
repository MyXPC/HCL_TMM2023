#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
注意力可视化脚本 - 输入一张图片，加载HCL模型并输出注意力图像
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

from src.ms_layer import *
from src.utils import *

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_model(model_path, net_arch, num_classes=496):
    """加载单个模型
    
    Args:
        model_path: 模型文件路径
        net_arch: 网络架构名称
        num_classes: 类别数量
        
    Returns:
        nn.Module: 加载的模型
    """
    # 根据网络架构创建模型
    if net_arch == 'resnet50':
        model = load_ms_layer(model_name='resnet50_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
    elif net_arch == 'resnet101':
        model = load_ms_layer(model_name='resnet101_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
    elif net_arch == 'resnet152':
        model = load_ms_layer(model_name='resnet152_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
    else:
        raise ValueError(f'Unsupported network architecture: {net_arch}')
    
    # 加载模型权重
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 处理不同类型的checkpoint
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module.state_dict()
    elif hasattr(checkpoint, 'state_dict'):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif hasattr(checkpoint, 'module'):
        checkpoint = checkpoint.module.state_dict()
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def create_transform():
    """创建图像预处理变换
    
    Returns:
        transforms.Compose: 图像变换组合
    """
    return transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_and_preprocess_image(image_path, transform):
    """加载和预处理图像
    
    Args:
        image_path: 图像文件路径
        transform: 图像变换
        
    Returns:
        torch.Tensor: 预处理后的图像张量
        PIL.Image: 原始图像
    """
    # 加载图像
    original_image = Image.open(image_path).convert('RGB')
    
    # 应用变换
    image_tensor = transform(original_image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    
    return image_tensor, original_image

def compute_saliency_map(feature_map):
    """计算显著性图
    
    Args:
        feature_map: 特征图张量 [batch, channels, height, width]
        
    Returns:
        torch.Tensor: 显著性图 [batch, 1, height, width]
    """
    eps = 1e-8
    b, c, h, w = feature_map.shape
    
    # 计算显著性图：对通道维度求和并归一化
    saliency = torch.sum(feature_map, dim=1) * (1.0 / (c + eps))
    saliency = saliency.contiguous()
    saliency = saliency.view(b, 1, h, w)  # 重塑为单通道显著性图
    
    return saliency

def visualize_attention(image_tensor, saliency_map, output_path, alpha=0.6):
    """可视化注意力图
    
    Args:
        image_tensor: 原始图像张量 [1, 3, H, W]
        saliency_map: 显著性图 [1, 1, H, W]
        output_path: 输出图像路径
        alpha: 透明度混合参数
    """
    # 将图像张量转换为numpy数组
    image_np = image_tensor.squeeze(0).cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    
    # 处理显著性图
    saliency_np = saliency_map.squeeze().cpu().numpy()
    
    # 归一化显著性图到0-1范围
    saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
    
    # 调整显著性图大小以匹配原始图像
    saliency_resized = cv2.resize(saliency_np, (image_np.shape[1], image_np.shape[0]))
    
    # 应用颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 混合原始图像和热力图
    overlayed = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    
    # 保存结果
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('原始图像')
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('注意力热力图')
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(overlayed)
    plt.title('注意力叠加图')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"注意力图像已保存到: {output_path}")

def extract_attention_regions(model, image_tensor, device):
    """提取注意力区域
    
    Args:
        model: 加载的模型
        image_tensor: 输入图像张量
        device: 计算设备
        
    Returns:
        tuple: (显著性图, 坐标信息, 原始特征图)
    """
    # 将模型和输入数据移动到设备
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    # 设置为评估模式
    model.eval()
    
    with torch.no_grad():
        # 前向传播获取多尺度输出和显著性信息
        output_1, output_2, output_3, output_concat, coord, _, _, xf_ori = model(image_tensor, 4)
        
        # 计算显著性图
        saliency_map = compute_saliency_map(xf_ori)
        
        return saliency_map, coord, xf_ori

def copy_directory_structure(src_dir, dst_dir):
    """复制目录结构（不复制图片文件）
    
    Args:
        src_dir: 源目录路径
        dst_dir: 目标目录路径
    """
    for root, dirs, files in os.walk(src_dir):
        # 计算相对路径
        rel_path = os.path.relpath(root, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        
        # 创建目标目录
        os.makedirs(dst_path, exist_ok=True)
        
        # 不复制图片文件，只创建目录结构
        print(f"创建目录: {dst_path}")

def process_single_image(image_path, model, transform, device, output_dir, image_name=None):
    """处理单张图像并生成注意力可视化结果
    
    Args:
        image_path: 图像文件路径
        model: 加载的模型
        transform: 图像变换
        device: 计算设备
        output_dir: 输出目录
        image_name: 图像名称（用于文件夹模式）
        
    Returns:
        bool: 处理是否成功
    """
    try:
        # 加载和预处理图像
        image_tensor, original_image = load_and_preprocess_image(image_path, transform)
        
        # 提取注意力信息
        saliency_map, coord, xf_ori = extract_attention_regions(model, image_tensor, device)
        
        # 确定输出路径
        if image_name:
            # 文件夹模式：在输出目录下创建与图片名相同的文件夹
            image_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_output_dir, exist_ok=True)
            output_path = os.path.join(image_output_dir, 'attention_visualization.png')
        else:
            # 单图片模式：直接在输出目录下保存
            output_path = os.path.join(output_dir, 'attention_visualization.png')
        
        # 可视化注意力
        visualize_attention(image_tensor, saliency_map, output_path)
        
        # 保存原始显著性图
        saliency_np = saliency_map.squeeze().cpu().numpy()
        saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
        saliency_np = (saliency_np * 255).astype(np.uint8)
        saliency_image = Image.fromarray(saliency_np)
        
        if image_name:
            saliency_image.save(os.path.join(image_output_dir, 'saliency_map.png'))
        else:
            saliency_image.save(os.path.join(output_dir, 'saliency_map.png'))
        
        # 保存坐标信息
        coord_np = coord.squeeze().cpu().numpy()
        
        if image_name:
            np.savetxt(os.path.join(image_output_dir, 'attention_coordinates.txt'), coord_np, fmt='%.2f')
        else:
            np.savetxt(os.path.join(output_dir, 'attention_coordinates.txt'), coord_np, fmt='%.2f')
        
        # 复制原始图像到输出目录（仅文件夹模式）
        if image_name:
            original_image.save(os.path.join(image_output_dir, os.path.basename(image_path)))
        
        return True
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return False

def process_folder(input_folder, model, transform, device, output_dir):
    """处理文件夹中的所有图像
    
    Args:
        input_folder: 输入文件夹路径
        model: 加载的模型
        transform: 图像变换
        device: 计算设备
        output_dir: 输出目录
    """
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 复制目录结构到输出目录（不复制图片）
    print("复制目录结构...")
    copy_directory_structure(input_folder, output_dir)
    
    # 遍历所有图像文件
    processed_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                
                # 计算相对路径和图像名称
                rel_path = os.path.relpath(root, input_folder)
                if rel_path == '.':
                    image_name = os.path.splitext(file)[0]
                else:
                    image_name = os.path.join(rel_path, os.path.splitext(file)[0])
                
                print(f"处理图像: {image_path}")
                
                # 处理单张图像
                success = process_single_image(image_path, model, transform, device, output_dir, image_name)
                
                if success:
                    processed_count += 1
                    print(f"✓ 成功处理: {file}")
                else:
                    failed_count += 1
                    print(f"✗ 处理失败: {file}")
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 张图像")
    print(f"处理失败: {failed_count} 张图像")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HCL注意力可视化')
    parser.add_argument('--input_path', type=str, required=True, 
                       help='输入路径（单张图像路径或文件夹路径）')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录路径')
    parser.add_argument('--net', type=str, default='resnet50', help='网络架构: resnet50, resnet101, resnet152')
    parser.add_argument('--output_dir', type=str, default='attention_results', help='输出目录')
    parser.add_argument('--gpu', type=str, default='0', help='使用的GPU编号')
    parser.add_argument('--mode', type=str, choices=['single', 'folder', 'auto'], default='auto',
                       help='运行模式: single(单图片), folder(文件夹), auto(自动检测)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径 {args.input_path} 不存在")
        return
    
    # 自动检测模式
    if args.mode == 'auto':
        if os.path.isfile(args.input_path):
            args.mode = 'single'
        elif os.path.isdir(args.input_path):
            args.mode = 'folder'
        else:
            print(f"错误: 无法识别输入路径类型: {args.input_path}")
            return
    
    # 加载模型（使用网络1，因为它处理原始图像并提取显著性信息）
    model_path = os.path.join(args.model_dir, 'best_total_concat-net1.pth')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    print("加载模型中...")
    model = load_model(model_path, args.net)
    
    # 创建图像变换
    transform = create_transform()
    
    # 根据模式处理输入
    if args.mode == 'single':
        print(f"单图片模式: 处理图像 {args.input_path}")
        
        # 检查是否为图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if not any(args.input_path.lower().endswith(ext) for ext in image_extensions):
            print(f"错误: 输入文件不是支持的图像格式: {args.input_path}")
            return
        
        # 处理单张图像
        success = process_single_image(args.input_path, model, transform, device, args.output_dir)
        
        if success:
            print("注意力可视化完成！")
            print(f"结果保存在目录: {args.output_dir}")
            print(f"- 注意力可视化图: attention_visualization.png")
            print(f"- 原始显著性图: saliency_map.png")
            print(f"- 注意力坐标: attention_coordinates.txt")
        else:
            print("处理图像失败")
    
    elif args.mode == 'folder':
        print(f"文件夹模式: 处理文件夹 {args.input_path}")
        
        # 处理文件夹中的所有图像
        process_folder(args.input_path, model, transform, device, args.output_dir)
        
        print(f"\n所有结果保存在目录: {args.output_dir}")
        print("每个图像的结果保存在对应的子文件夹中，包含:")
        print("- 注意力可视化图: attention_visualization.png")
        print("- 原始显著性图: saliency_map.png")
        print("- 注意力坐标: attention_coordinates.txt")
        print("- 原始图像文件")

if __name__ == "__main__":
    main()
