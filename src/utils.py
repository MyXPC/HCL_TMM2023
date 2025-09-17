import numpy as np  # 导入NumPy数值计算库
import random  # 导入随机数模块
import torch  # 导入PyTorch主库
import torchvision  # 导入torchvision计算机视觉库
from torch.autograd import Variable  # 导入自动求导变量
from torchvision import transforms, models  # 导入数据变换和预训练模型
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数模块

from .Resnet import *  # 导入ResNet模型    

from .ms_layer import MS_resnet_layer  # 导入多尺度ResNet层


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像文件



def accuracy(output, target, topk=(1,)):
    """计算指定k值的准确率@k
    
    Args:
        output: 模型输出logits
        target: 真实标签
        topk: 要计算的top-k准确率元组，如(1,5)表示top1和top5准确率
        
    Returns:
        list: 包含每个k值准确率的列表
    """
    maxk = max(topk)  # 获取最大的k值
    batch_size = target.size(0)  # 批次大小

    # 获取topk预测结果
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置预测结果
    # 计算正确预测的布尔矩阵
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 计算每个k值的正确数量
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))  # 转换为百分比
    return res

class AverageMeter(object):
    """平均值计量器 - 计算并存储平均值和当前值
    
    用于跟踪训练过程中的损失、准确率等指标的移动平均值
    """
    def __init__(self):
        """初始化平均值计量器"""
        self.reset()

    def reset(self):
        """重置所有统计值"""
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """更新统计值
        
        Args:
            val: 新的值
            n: 该值对应的样本数量，默认为1
        """
        self.val = val
        self.sum += val * n  # 累加值
        self.count += n  # 累加计数
        self.avg = self.sum / self.count  # 计算平均值

class SortMean(object):
    """排序均值计算器 - 维护固定长度的值列表并计算平均值
    
    用于计算最近N个值的移动平均值，保持固定窗口大小
    """
    def __init__(self, num_batch):
        """初始化排序均值计算器
        
        Args:
            num_batch: 窗口大小，保留的最近值数量
        """
        self.reset(num_batch)
    
    def reset(self,num_batch):
        """重置计算器
        
        Args:
            num_batch: 窗口大小
        """
        self.rate_list = []  # 值列表
        self.num = num_batch  # 窗口大小
        self.avg = 0  # 平均值

    def update(self,val):
        """更新值列表并重新计算平均值
        
        Args:
            val: 新的值
        """
        if len(self.rate_list)<self.num:
            self.rate_list.append(val)  # 列表未满，直接添加
        else:  
            self.rate_list.append(val)
            self.rate_list.pop(0)  # 列表已满，移除最旧的值
        self.avg = np.mean(self.rate_list)  # 计算平均值

class SortMove(object):
    """移动平均计算器 - 使用指数加权移动平均(EWMA)
    
    使用指数衰减权重计算移动平均值，对近期值给予更高权重
    """
    def __init__(self, weight=0.9):
        """初始化移动平均计算器
        
        Args:
            weight: 权重因子，控制历史值的衰减速度（0-1之间）
        """
        self.reset(weight)
        
    def reset(self,weight):
        """重置计算器
        
        Args:
            weight: 权重因子
        """
        self.avg = 0  # 移动平均值
        self.weight = weight  # 权重因子
        self.flag = False  # 初始化标志

    def update(self,val):
        """更新移动平均值
        
        Args:
            val: 新的值
        """
        if self.flag:
            # 使用指数加权移动平均公式更新
            self.avg = self.weight*self.avg + (1-self.weight)*val
        else: 
            self.avg = val  # 第一次更新，直接使用当前值
            self.flag = True  # 设置标志为True

def cosine_anneal_schedule(t, nb_epoch, lr):
    """余弦退火学习率调度器
    
    根据余弦函数调整学习率，实现学习率的周期性变化
    
    Args:
        t: 当前训练轮次
        nb_epoch: 总训练轮次
        lr: 初始学习率
        
    Returns:
        float: 调整后的学习率
    """
    cos_inner = np.pi * (t % (nb_epoch))  # 计算余弦内部值
    cos_inner /= (nb_epoch)  # 归一化
    cos_out = np.cos(cos_inner) + 1  # 计算余弦输出（0-2范围）

    return float(lr / 2 * cos_out)  # 返回调整后的学习率



def load_ms_layer(model_name,classes_nums, pretrain=True, require_grad=True):
    """加载多尺度ResNet层模型
    
    根据指定的ResNet架构创建并配置多尺度ResNet模型
    
    Args:
        model_name: 模型名称，支持resnet18_ms, resnet50_ms, resnet101_ms, resnet152_ms
        classes_nums: 分类类别数量
        pretrain: 是否使用预训练权重，默认为True
        require_grad: 是否要求梯度，控制参数是否可训练，默认为True
        
    Returns:
        nn.Module: 配置好的多尺度ResNet模型
    """
    print('==> Building model..')
    if model_name == 'resnet50_ms':
        net = resnet50(pretrained=pretrain)  # 加载ResNet50
        for param in net.parameters():
            param.requires_grad = require_grad  # 设置梯度要求
        net = MS_resnet_layer(net,50, 512, classes_nums)  # 包装为多尺度层
    elif model_name == 'resnet18_ms':
        net = resnet18(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,18, 512, classes_nums)
    elif model_name == 'resnet101_ms':
        net = resnet101(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,101, 512, classes_nums)
    elif model_name == 'resnet152_ms':
        net = resnet152(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = MS_resnet_layer(net,152, 512, classes_nums)

    return net
