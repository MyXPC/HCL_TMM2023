# HCL项目启动命令大全

本文档整理了HCL项目的所有启动命令，方便快速复制使用。

## 目录
- [基础训练命令](#基础训练命令)
- [分布式训练命令](#分布式训练命令)
- [混合精度训练命令](#混合精度训练命令)
- [推理命令](#推理命令)
- [性能测试命令](#性能测试命令)
- [参数说明](#参数说明)

## 基础训练命令

### 单机多卡训练（推荐）
```bash
# 使用4张GPU训练，批次大小30，resnet50网络
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --bs 30 \
    --net 'resnet50' \
    --data "../../dataset/train400" \
    --val "../../dataset/val400" \
    --save_dir "./runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3 \
    --gpus 4 \
    --num_workers 4
```

### 单卡训练
```bash
# 使用单张GPU训练
CUDA_VISIBLE_DEVICES=0 python main.py \
    --bs 15 \
    --net 'resnet50' \
    --data "../../dataset/train400" \
    --val "../../dataset/val400" \
    --save_dir "./runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0 \
    --gpus 1 \
    --num_workers 4
```

## 分布式训练命令

### 使用torch.distributed.launch（推荐方式）
```bash
# 使用4个进程进行DDP训练，批次大小15（每个进程）
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12355 main.py \
    --bs 15 \
    --net 'resnet50' \
    --data "../dataset/train/400" \
    --val "../dataset/val/val400" \
    --save_dir "./runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3 \
    --use_ddp \
    --num_workers 4
```

### 使用torchrun（PyTorch 1.9+推荐）
```bash
# 使用torchrun启动DDP训练，批次大小8（每个进程）
torchrun --nproc_per_node=4 --master_port=12355 main.py \
    --bs 8 \
    --net 'resnet50' \
    --data "../../dataset/web496/train400" \
    --val "../../dataset/web496/val400" \
    --save_dir "./runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3 \
    --use_ddp \
    --num_workers 4
```

## 混合精度训练命令

### AMP混合精度训练
```bash
# 使用混合精度训练，减少显存占用，加快训练速度
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --bs 30 \
    --net 'resnet50' \
    --data "../dataset/train/400" \
    --val "../../dataset/val400" \
    --save_dir "../HCL_data/runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3 \
    --gpus 4 \
    --use_amp \
    --num_workers 4
```

## 推理命令

### 批量推理
```bash
# 对测试集进行推理，生成预测结果CSV文件
python infer.py \
    --test_dir ../../dataset/allval400 \
    --model_dir ../HCL_model/epoch100 \
    --net resnet50 \
    --batch_size 30 \
    --output_file my_predictions.csv
```

## 注意力可视化命令

### 单图片模式（处理单张图像）
```bash
# 对单张图片进行注意力可视化，生成热力图和叠加图
python attention_visualization.py \
    --input_path ../HCL_data/test_img/101.White_Pelican.jpg \
    --model_dir ../HCL_model/epoch100 \
    --net resnet50 \
    --output_dir ../attention_results \
    --gpu 0 \
    --mode single
```

### 文件夹模式（批量处理整个文件夹）
```bash
# 处理文件夹中的所有图像，保持目录结构
python attention_visualization.py \
    --input_path ../HCL_data/test_images \
    --model_dir ../HCL_model/epoch100 \
    --net resnet50 \
    --output_dir ../batch_attention_results \
    --gpu 0 \
    --mode folder
```

### 自动检测模式（推荐）
```bash
# 自动检测输入路径类型（文件或文件夹）
python attention_visualization.py \
    --input_path ../HCL_data/error_samples_organized \
    --model_dir ../HCL_model/epoch100 \
    --net resnet50 \
    --output_dir ../attention_results \
    --gpu 0 \
    --mode auto
```

### 批量注意力可视化（旧版脚本兼容）
```bash
# 创建批量处理脚本 batch_attention.sh（兼容旧版参数）
#!/bin/bash

MODEL_DIR="path/to/trained/models"
OUTPUT_BASE="batch_attention_results"
IMAGE_DIR="path/to/image/folder"

# 创建输出目录
mkdir -p $OUTPUT_BASE

# 处理目录中的所有图片
for img_file in $IMAGE_DIR/*.jpg $IMAGE_DIR/*.png; do
    if [ -f "$img_file" ]; then
        filename=$(basename "$img_file" | cut -d. -f1)
        output_dir="$OUTPUT_BASE/$filename"
        
        echo "Processing: $img_file"
        python attention_visualization.py \
            --input_path "$img_file" \
            --model_dir "$MODEL_DIR" \
            --net resnet50 \
            --output_dir "$output_dir" \
            --gpu 0 \
            --mode single
    fi
done
```

## 性能测试命令

### 训练速度测试
```bash
# 使用小数据集测试训练速度（1个epoch）
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --bs 30 \
    --net 'resnet50' \
    --data "../../dataset/time" \
    --val "../../dataset/time" \
    --save_dir "./runs" \
    --epochs 1 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3 \
    --gpus 4
```

## 预测结果分析命令

### 检查预测准确性
```bash
# 检查预测结果的准确性，生成错误样本报告
python check.py \
    --csv_file my_predictions.csv \
    --dataset_path ../../dataset/val400 \
    --output_file error_samples.csv
```

### 组织错误样本
```bash
# 将错误样本按真实类别组织到文件夹中
python organize_error_samples.py \
    --csv error_samples.csv \
    --dataset ../../dataset/val400 \
    --output ../HCL_data/error_samples_organized
```

## 参数说明

### 主要参数
- `--bs`: **每个GPU上的批次大小**（不是总batch size）
- `--net`: 网络架构（resnet50/resnet101/resnet152/resnet18）
- `--data`: 训练数据集路径
- `--val`: 验证数据集路径
- `--save_dir`: 模型保存目录
- `--epochs`: 训练总轮数
- `--drop_rate`: 丢弃率，控制噪声样本处理（0.25-0.35）
- `--gpu`: 使用的GPU编号（逗号分隔）
- `--gpus`: 使用的GPU数量

**DP和DDP模式下的batch_size含义不相同**：在DataParallel（DP）和DistributedDataParallel（DDP）模式下，`--bs`参数分别表示所有GPU上的批次大小和每一个GPU上的批次大小。

### 高级参数
- `--each_class`: 每个类别的样本数量（用于限制数据集大小）
- `--num_workers`: 数据加载工作进程数（默认：4）
- `--pretrained_model1/2/3`: 从断点加载预训练模型
- `--continue_epoch`: 从指定轮数继续训练
- `--use_amp`: 启用混合精度训练（减少显存占用）
- `--use_ddp`: 启用分布式数据并行训练

### 注意力可视化参数
- `--input_path`: 输入路径（单张图像路径或文件夹路径，必需）
- `--model_dir`: 训练好的模型目录路径（必需）
- `--net`: 网络架构（resnet50/resnet101/resnet152，默认：resnet50）
- `--output_dir`: 输出结果目录（默认：attention_results）
- `--gpu`: 使用的GPU编号（默认：0）
- `--mode`: 运行模式（single/文件夹/folder/auto，默认：auto）
  - `single`: 单图片模式，处理单张图像
  - `folder`: 文件夹模式，批量处理整个文件夹
  - `auto`: 自动检测模式，根据输入路径类型自动选择

## 环境要求

确保已安装以下依赖：
```bash
pip install torch torchvision tqdm numpy Pillow
```

对于DDP训练，需要PyTorch 1.6+版本。

## 使用建议

1. **初次训练**: 使用基础训练命令开始
2. **多卡训练**: 使用DDP命令获得更好的扩展性
3. **显存不足**: 使用混合精度训练或减小批次大小
4. **推理部署**: 使用推理命令生成预测结果
5. **注意力可视化**: 
   - **单张图像分析**: 使用`--mode single`分析单张图像的注意力分布
   - **批量处理**: 使用`--mode folder`批量处理整个文件夹，保持目录结构
   - **自动检测**: 使用`--mode auto`让脚本自动检测输入类型
6. **性能测试**: 使用性能测试命令评估训练速度

## 常见问题

1. **CUDA内存不足**: 减小`--bs`参数或使用混合精度训练
2. **DDP训练失败**: 检查端口是否被占用，尝试更换`--master_port`
3. **数据路径错误**: 确保数据集路径正确，相对路径基于当前目录
4. **模型加载失败**: 检查模型文件是否存在且格式正确

## 更新日志

- 2024-01-01: 创建启动命令文档
- 2024-01-02: 添加参数说明和使用建议
- 2024-01-03: 添加注意力可视化命令和参数说明
- 2025-09-23: 更新注意力可视化脚本，支持单图片和文件夹两种输出模式
  - 新增`--input_path`参数替代`--image_path`
  - 新增`--mode`参数支持single/folder/auto三种模式
  - 文件夹模式自动复制目录结构并批量处理

---

**注意**: 所有路径都是相对路径，请根据实际项目结构进行调整。训练前请确保数据集已正确准备。
