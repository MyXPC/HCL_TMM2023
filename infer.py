#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
推理脚本 - 使用训练好的HCL模型对未知类别的图片进行推理
输出格式：图片文件名, 四位数字类别名（不满四位的前面补0）
"""

from __future__ import print_function
import os
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import time
import csv
import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from src.ms_layer import *
from src.utils import *
from src.Imagefolder_modified import Imagefolder_modified

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    """设置随机种子以确保实验的可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# 命令行参数解析器
parser = argparse.ArgumentParser(description='HCL Inference')
parser.add_argument('--test_dir', type=str, required=True, help='测试图片目录路径')
parser.add_argument('--model_dir', type=str, required=True, help='训练好的模型目录路径')
parser.add_argument('--net', type=str, default='resnet50', help='网络架构: resnet50, resnet101, resnet152')
parser.add_argument('--output_file', type=str, default='predictions.csv', help='输出CSV文件名')
parser.add_argument('--batch_size', type=int, default=16, help='推理批次大小')
parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
parser.add_argument('--gpu', type=str, default='0', help='使用的GPU编号')

def load_models(model_dir, net_arch, num_classes):
    """加载所有训练好的模型（网络1、网络2、网络3和显著性采样器）
    
    Args:
        model_dir: 模型目录路径
        net_arch: 网络架构名称
        num_classes: 类别数量
        
    Returns:
        tuple: (net1, net2, net3, saliency_sampler)
    """
    # 根据网络架构创建模型
    if net_arch == 'resnet50':
        net1 = load_ms_layer(model_name='resnet50_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
        net2 = load_ms_layer(model_name='resnet50_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
        net3 = load_ms_layer(model_name='resnet50_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
    elif net_arch == 'resnet101':
        net1 = load_ms_layer(model_name='resnet101_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
        net2 = load_ms_layer(model_name='resnet101_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
        net3 = load_ms_layer(model_name='resnet101_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
    elif net_arch == 'resnet152':
        net1 = load_ms_layer(model_name='resnet152_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
        net2 = load_ms_layer(model_name='resnet152_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
        net3 = load_ms_layer(model_name='resnet152_ms', classes_nums=num_classes, pretrain=False, require_grad=False)
    else:
        raise ValueError(f'Unsupported network architecture: {net_arch}')
    
    # 创建显著性采样器
    saliency_sampler = Saliency_Sampler()
    
    # 加载模型权重
    def load_model_weights(model, model_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # 处理不同类型的checkpoint
        if isinstance(checkpoint, torch.nn.DataParallel):
            # 如果是DataParallel包装的模型，提取状态字典
            checkpoint = checkpoint.module.state_dict()
        elif hasattr(checkpoint, 'state_dict'):
            # 如果checkpoint有state_dict属性，提取状态字典
            checkpoint = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 如果是包含state_dict的字典
            checkpoint = checkpoint['state_dict']
        elif hasattr(checkpoint, 'module'):
            # 如果checkpoint有module属性，提取module的状态字典
            checkpoint = checkpoint.module.state_dict()
        
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    # 加载所有模型
    net1_path = os.path.join(model_dir, 'best_total_concat-net1.pth')
    net2_path = os.path.join(model_dir, 'best_total_concat-net2.pth')
    net3_path = os.path.join(model_dir, 'best_total_concat-net3.pth')
    ss_path = os.path.join(model_dir, 'best_total_concat-ss.pth')
    
    if not all(os.path.exists(path) for path in [net1_path, net2_path, net3_path, ss_path]):
        print("Some model files are missing. Please check the model directory.")
        return None, None, None, None
    
    print("Loading all models...")
    net1 = load_model_weights(net1, net1_path)
    net2 = load_model_weights(net2, net2_path)
    net3 = load_model_weights(net3, net3_path)
    
    # 加载显著性采样器
    if torch.cuda.is_available():
        ss_checkpoint = torch.load(ss_path, weights_only=False)
    else:
        ss_checkpoint = torch.load(ss_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 处理显著性采样器的checkpoint - 可能是整个模型对象
    if isinstance(ss_checkpoint, torch.nn.DataParallel):
        # 如果是DataParallel包装的模型，提取状态字典
        ss_checkpoint = ss_checkpoint.module.state_dict()
    elif hasattr(ss_checkpoint, 'state_dict'):
        # 如果checkpoint有state_dict属性，提取状态字典
        ss_checkpoint = ss_checkpoint.state_dict()
    elif isinstance(ss_checkpoint, dict) and 'state_dict' in ss_checkpoint:
        # 如果是包含state_dict的字典
        ss_checkpoint = ss_checkpoint['state_dict']
    elif hasattr(ss_checkpoint, 'module'):
        # 如果checkpoint有module属性，提取module的状态字典
        ss_checkpoint = ss_checkpoint.module.state_dict()
    
    saliency_sampler.load_state_dict(ss_checkpoint)
    
    saliency_sampler.eval()
    
    return net1, net2, net3, saliency_sampler

def create_test_transform():
    """创建测试数据变换
    
    Returns:
        transforms.Compose: 测试数据变换组合
    """
    return transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_image_files(directory):
    """获取目录中的所有图像文件
    
    Args:
        directory: 图片目录路径
        
    Returns:
        list: 图像文件路径列表
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    
    return sorted(image_files)

class InferenceDataset(torch.utils.data.Dataset):
    """推理数据集类"""
    
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        filename = os.path.basename(image_path)
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, filename, idx
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个空白图像作为占位符
            blank_image = torch.zeros(3, 448, 448)
            return blank_image, filename, idx

def predict(net1, net2, net3, saliency_sampler, dataloader, device):
    """进行批量预测 - 模仿main.py中的验证阶段流程
    
    Args:
        net1: 网络1模型
        net2: 网络2模型
        net3: 网络3模型
        saliency_sampler: 显著性采样器
        dataloader: 数据加载器
        device: 计算设备
        
    Returns:
        tuple: (预测结果列表, 文件名列表)
    """
    predictions = []
    filenames = []
    
    # 将所有模型移动到设备
    net1.to(device)
    net2.to(device)
    net3.to(device)
    saliency_sampler.to(device)
    
    # 设置为评估模式
    net1.eval()
    net2.eval()
    net3.eval()
    saliency_sampler.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            images, batch_filenames, indices = batch
            images = images.to(device)
            
            # 网络1前向传播 - 获取多尺度输出和显著性信息
            output_1_1, output_2_1, output_3_1, output_concat_1, coord, _, _, xf_ori = net1(images, 4)
            
            # 基于显著性采样生成目标区域图像
            inputs_obj = saliency_sampler(images.clone(), xf_ori)
            
            # 处理坐标信息用于显著性区域提取
            coord = coord.detach().cpu()
            coord = coord.numpy()
            coord = np.uint8(coord)
            inputs_salient = images.clone()
            inputs_batch_size = images.size(0)
            
            # 对每个样本提取显著性区域
            for i in range(inputs_batch_size):
                a, b, c, d = coord[i]  # 获取边界框坐标(x,y,width,height)
                saliency_figure = images[i,:,:,:].clone()
                # 提取显著性区域（32倍下采样坐标映射）
                show = saliency_figure[:,32*b:32*int(b+d),32*a:32*int(a+c)]
                show = show.unsqueeze(0)
                # 插值回原始尺寸
                show = F.interpolate(show, size=[448,448], mode='bilinear', align_corners=True)
                show = show.squeeze(0)
                inputs_salient[i,:,:,:] = show
            
            # 网络2前向传播 - 处理目标区域图像
            output_1_2, output_2_2, output_3_2, output_concat_2, _, _, _, _ = net2(inputs_obj, 4)
            
            # 网络3前向传播 - 处理显著性区域图像（不生成拼图）
            output_1_3, output_2_3, output_3_3, output_concat_3, _, _, _, _ = net3(inputs_salient, 4)
            
            # 融合三个网络的输出进行最终预测
            outputs_concat = output_concat_1 + output_concat_2 + output_concat_3
            _, predicted = torch.max(outputs_concat.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            filenames.extend(batch_filenames)
    
    return predictions, filenames

def format_class_label(class_idx):
    """格式化类别标签为四位数字
    
    Args:
        class_idx: 类别索引
        
    Returns:
        str: 四位数字的类别标签
    """
    return f"{class_idx:04d}"

def save_predictions_to_csv(filenames, predictions, output_file):
    """保存预测结果到CSV文件
    
    Args:
        filenames: 文件名列表
        predictions: 预测结果列表
        output_file: 输出CSV文件路径
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'class_label'])
        
        for filename, pred in zip(filenames, predictions):
            class_label = format_class_label(pred)
            writer.writerow([filename, class_label])
    
    print(f"Predictions saved to {output_file}")

def main():
    """主函数"""
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 获取图像文件
    image_files = get_image_files(args.test_dir)
    if not image_files:
        print(f"No image files found in {args.test_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # 创建数据变换
    transform = create_test_transform()
    
    # 创建数据集和数据加载器
    dataset = InferenceDataset(image_files, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 加载模型
    # 假设类别数量为200（根据训练代码中的默认值）
    num_classes = 496
    
    # 加载所有模型
    net1, net2, net3, saliency_sampler = load_models(args.model_dir, args.net, num_classes)
    
    if net1 is None or net2 is None or net3 is None or saliency_sampler is None:
        print("Failed to load models. Exiting.")
        return
    
    # 进行预测
    predictions, filenames = predict(net1, net2, net3, saliency_sampler, dataloader, device)
    
    # 保存结果到CSV
    save_predictions_to_csv(filenames, predictions, args.output_file)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
