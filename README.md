# HCL (分层对比学习) - 细粒度视觉分类

本仓库实现了HCL（Hierarchical Contrastive Learning，分层对比学习）框架，用于处理网络来源的噪声数据并进行细粒度视觉分类（FGVC）。

## 项目概述

HCL框架专门设计用于处理网络来源的噪声数据集，通过以下方式实现：

- 使用三个并行的基于ResNet的网络，采用不同的输入变换
- 实现分层对比学习来区分干净样本和噪声样本
- 采用多尺度特征提取以获得更好的表示学习
- 使用基于显著性的采样和拼图变换进行数据增强

## 主要特性

- **多尺度特征提取**：在不同尺度（14x14、28x28、56x56）提取特征，实现全面的表示学习
- **分层对比学习**：通过在多个层次对比特征来学习鲁棒的表示
- **噪声鲁棒性**：在训练过程中自动识别和处理噪声样本
- **基于显著性的采样**：使用显著性图聚焦重要区域
- **拼图变换**：通过空间变换增强学习效果

## 架构设计

框架包含三个主要组件：

1. **主网络 (Net1)**：处理原始图像并提取显著性图
2. **目标网络 (Net2)**：处理基于显著性裁剪的区域
3. **部件网络 (Net3)**：处理拼图变换后的图像

## 安装要求

### 环境要求
- Python 3.6+
- PyTorch 1.7+
- torchvision
- OpenCV
- einops
- PIL

### 安装步骤
```bash
git clone https://github.com/MyXPC/HCL_TMM2023.git
cd HCL_TMM2023
pip install -r requirements.txt
```

## 数据集支持

代码支持任意图像分类数据集，数据集应按以下结构组织：
```
dataset/
├── train400/
│   ├── 类别1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── 类别2/
│   └── ...
└── val400/
    ├── 类别1/
    ├── 类别2/
    └── ...
```

**注意**: 代码使用 `--data` 和 `--val` 参数指定训练集和验证集路径，支持任意自定义数据集。

## 使用方法

### 训练模型

使用自定义参数手动训练：

```bash
python main.py \
    --bs <批次大小> \
    --net <resnet50|resnet101|resnet152|resnet18> \
    --data <训练数据集路径> \
    --val <验证数据集路径> \
    --save_dir <模型保存目录> \
    --epochs <训练轮数> \
    --drop_rate <丢弃率> \
    --gpu <GPU编号> \
    --gpus <GPU数量> \
    --num_workers <数据加载进程数>
```

### 示例训练命令

```bash
# 使用4张GPU训练，resnet50网络
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

### 参数说明

- `--bs`: **每个GPU上的批次大小**（不是总batch size，默认：30）
- `--net`: 骨干网络架构（resnet50、resnet101、resnet152、resnet18）
- `--data`: 训练数据集路径（必需）
- `--val`: 验证数据集路径（必需）
- `--save_dir`: 模型保存目录（必需）
- `--epochs`: 训练总轮数（默认：100）
- `--drop_rate`: 丢弃率，控制噪声样本处理（默认：0.35）
- `--gpu`: 使用的GPU编号（逗号分隔，如：0,1）
- `--gpus`: 使用的GPU数量
- `--each_class`: 每个类别的样本数量（可选，用于限制数据集大小）
- `--num_workers`: 数据加载工作进程数（默认：4）
- `--pretrained_model1/2/3`: 从断点加载预训练模型
- `--continue_epoch`: 从指定轮数继续训练
- `--use_amp`: 启用混合精度训练（减少显存占用）
- `--use_ddp`: 启用分布式数据并行训练

**注意**: `--bs` 参数指定的是每个GPU上的批次大小。总batch size = `--bs` × GPU数量。例如，使用4张GPU且 `--bs 30` 时，总batch size为120。

**DP和DDP模式下的batch_size含义相同**：在DataParallel（DP）和DistributedDataParallel（DDP）模式下，`--bs`参数都表示每个GPU上的批次大小。

## 项目结构

```
HCL_TMM2023/
├── main.py              # 主训练脚本
├── ms_layer.py          # 多尺度层实现
├── Resnet.py            # ResNet模型定义
├── utils.py             # 工具函数
├── autoaugment.py       # 自动数据增强
├── Imagefolder_modified.py # 修改的图像数据集加载器
├── infer.py             # 推理脚本
├── attention_visualization.py # 注意力可视化脚本
├── check.py             # 预测结果检查脚本
├── organize_error_samples.py # 错误样本组织脚本
├── generate_confusion_matrix.py # 混淆矩阵生成脚本
├── src/                 # 源代码目录
├── src_ddp/             # 分布式训练源代码目录
└── README.md           # 项目说明文档
```

## 核心算法

### 分层对比学习 (HCL Loss)
HCL损失函数通过以下方式工作：
1. 计算多个分类器的输出
2. 使用Jensen-Shannon散度评估样本一致性
3. 动态识别和更新干净样本
4. 在不同层次进行对比学习

### 多尺度特征提取
- 在ResNet的不同阶段提取特征
- 使用自适应池化处理不同尺度的特征
- 通过卷积块进一步处理特征

### 显著性采样
- 基于特征图的显著性检测重要区域
- 使用高斯滤波生成采样网格
- 对重要区域进行聚焦采样

## 实验结果

项目在多个细粒度视觉分类数据集上取得了优异的表现，特别是在处理网络来源的噪声数据方面表现出色。

## 引用

如果您使用了本代码，请引用相关论文：
```
@article{hcl2023,
  title={Hierarchical Contrastive Learning for Fine-Grained Visual Classification},
  author={Author et al.},
  journal={IEEE Transactions on Multimedia},
  year={2023}
}
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请通过GitHub Issues提交或联系项目维护者。

## 更新日志

- **2023-XX-XX**: 初始版本发布
- 支持ResNet50/101/152骨干网络
- 实现分层对比学习框架
- 提供多个数据集的训练脚本
