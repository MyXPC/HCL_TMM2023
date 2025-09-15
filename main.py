from __future__ import print_function
import os
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像文件
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import time

import warnings
warnings.filterwarnings('ignore')

from ms_layer import *  # 导入多尺度层模块

from torchvision import transforms
tensor_to_image = transforms.ToPILImage()  # 张量转图像转换器






ini_seed = 42  # 设置随机种子

def set_seed(seed = ini_seed):
    """设置随机种子以确保实验的可重复性"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

set_seed()  # 初始化随机种子




from utils import *  # 导入工具函数
from Imagefolder_modified import Imagefolder_modified  # 导入修改的图像文件夹加载器
from autoaugment import AutoAugImageNetPolicy  # 导入自动数据增强策略



# 命令行参数解析器
parser = argparse.ArgumentParser(description='HCL')
parser.add_argument('--epochs',  type=int, default=100, help='训练总轮数')
parser.add_argument('--each_class',  default=None, type=int,help='每个类别的样本数量')
parser.add_argument('--bs',  type=int, default=30,help='批次大小')
parser.add_argument('--net',  type=str, default='resnet50', help='网络架构: resnet50, resnet101, resnet152')
parser.add_argument('--data',  type=str, default='bird',help='数据集类型: bird, aircraft, car')
parser.add_argument('--gpu', default='0,1', type=str, help='使用的GPU编号')
parser.add_argument('--gpus', default=None, type=int, help='使用的GPU卡数量')
parser.add_argument('--save_dir',  type=str, default='' ,help='保存目录')
parser.add_argument("--num_workers", default=4, type=int, help='数据加载工作进程数')
parser.add_argument('--pretrained_model1', default=None, type=str, help='从断点加载网络1')
parser.add_argument('--pretrained_model2', default=None, type=str, help='从断点加载网络2')
parser.add_argument('--pretrained_model3', default=None, type=str, help='从断点加载网络3')
parser.add_argument('--continue_epoch', default=None, type=int, help='从指定轮数继续训练')
parser.add_argument('--drop_rate', type=float, default=0.35, help ='丢弃率，控制噪声样本处理')



# 解析命令行参数
args = parser.parse_args()
gpu = args.gpu
args.gpus = len(gpu.split(','))
print(args.gpu, '使用的GPU卡数量:', args.gpus)
args.save_dir = args.data + '_net_{}'.format(args.net)+ args.save_dir  # 设置保存目录



def contra_loss(features1, features2, labels, labels_flag):
    """对比损失函数 - 计算双向对比损失"""
    loss = contra_loss_ori(features1, features2, labels, labels_flag) + contra_loss_ori(features2, features1, labels, labels_flag)
    loss = loss/2.0  # 取平均值
    
    return loss




def contra_loss_ori(features1, features2, labels, labels_flag):
    """原始对比损失计算函数"""
    fac = 0.1  # 温度参数
    eps = 1e-6  # 防止除零的小值
    
    B, _ = features1.shape  # 批次大小
    features1 = F.normalize(features1)  # 归一化特征
    features2 = F.normalize(features2)

    cos_matrix = features1.mm(features2.t())  # 计算余弦相似度矩阵
    cos_matrix = cos_matrix/fac  # 缩放相似度

    logprobs = torch.nn.functional.log_softmax(cos_matrix, dim=-1)  # 计算对数概率

    focus = torch.ones(labels_flag.shape[0]).cuda()  # 创建注意力掩码
    ind = labels_flag==-1  # 找到噪声样本
    focus[ind]=0  # 噪声样本不参与对比学习

    focus_ori = focus.clone()  # 保存原始注意力掩码

    focus = focus.unsqueeze(1)  # 扩展维度
    focus = focus.repeat(1, labels_flag.size(0))  # 复制到匹配矩阵大小

    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()  # 创建正样本矩阵

    focus = focus*pos_label_matrix  # 应用正样本掩码

    logprobs = logprobs * focus  # 应用注意力掩码

    logprobs = torch.sum(logprobs, dim=-1)/(torch.sum(focus, dim=-1) + eps)  # 计算平均对数概率
   
    loss = -logprobs * focus_ori  # 计算损失
    loss = torch.sum(loss)/ (torch.sum(focus_ori) + eps)  # 计算平均损失
   
    return loss



def jigsaw_generator(images, n):
    """拼图生成器 - 将图像分割成n×n的块并随机打乱"""
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])  # 创建所有可能的块位置
    block_size = 448 // n  # 计算每个块的大小
    rounds = n ** 2  # 总块数

    # 创建位置标签张量
    location_labels_x=torch.FloatTensor(images.size(0),rounds)
    location_labels_y=torch.FloatTensor(images.size(0),rounds)
    location_labels=torch.FloatTensor(images.size(0),rounds*2)

    location_labels_x=location_labels_x.cuda()
    location_labels_y=location_labels_y.cuda()
    location_labels=location_labels.cuda()

    random.shuffle(l)  # 随机打乱块顺序

    jigsaws = images.clone()  # 克隆原始图像

    # 执行拼图变换
    for i in range(rounds):
        x, y = l[i]  # 当前块的目标位置

        # 交换块位置
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

        # 记录位置标签
        if i==0:
            location_labels_x[...,n*x+y]=float(0)
            location_labels_y[...,n*x+y]=float(0)
        else:
            temp_x, temp_y= l[i-1]  # 上一个块的位置
            location_labels_x[...,n*x+y]=float(temp_x)
            location_labels_y[...,n*x+y]=float(temp_y)
    

    location_labels=torch.cat((location_labels_x, location_labels_y), -1)  # 合并位置标签

    return jigsaws, location_labels  # 返回拼图图像和位置标签



class LabelSmoothing(nn.Module):
    """
    标签平滑损失函数 - 通过平滑标签来防止过拟合
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        初始化标签平滑模块
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing  # 置信度（正确标签的权重）
        self.smoothing = smoothing  # 平滑因子（错误标签的权重）

    def forward(self, x, target):
        """前向传播计算标签平滑损失"""
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)  # 计算对数概率

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))  # 负对数似然损失
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)  # 平滑损失（所有类别的平均）
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss  # 组合损失

        return loss



class HCL_loss(nn.Module):
    """
    分层对比学习损失函数 (Hierarchical Contrastive Learning Loss)
    用于处理噪声标签和进行多尺度对比学习
    """
    def __init__(self, labels_all=0, Tepoch =10, drop_rate = 0.25, class_num=200):
        """
        初始化HCL损失函数
        :param labels_all: 标签总数
        :param Tepoch: 过渡轮数，控制对比学习权重
        :param drop_rate: 丢弃率，控制噪声样本处理
        :param class_num: 类别数量
        """
        super(HCL_loss, self).__init__()
        self.Tepoch = Tepoch  # 过渡轮数
        self.drop_rate = drop_rate  # 丢弃率
        self.class_num = class_num  # 类别数量
        
        # 多尺度池化层
        self.maxpooling_patch7 = nn.MaxPool2d(kernel_size=7, stride=7)  # 7x7最大池化
        self.maxpooling_patch14 = nn.MaxPool2d(kernel_size=14, stride=14)  # 14x14最大池化
        self.ada_maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 自适应1x1池化
        self.ada_maxpool2 = nn.AdaptiveMaxPool2d((2, 2))  # 自适应2x2池化
        self.ada_maxpool3 = nn.AdaptiveMaxPool2d((3, 3))  # 自适应3x3池化
        self.ada_maxpool4 = nn.AdaptiveMaxPool2d((4, 4))  # 自适应4x4池化
        self.pool =  nn.AdaptiveMaxPool2d((2, 2))  # 自适应2x2池化
        
        self.label_smooth_loss = LabelSmoothing(0.1)  # 标签平滑损失
        self.BCE_loss = nn.BCEWithLogitsLoss()  # 二元交叉熵损失

    def get_update(self, logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, labels, epoch, drop_rate_fine):
        """获取需要更新的样本索引 - 基于损失排序和遗忘率"""
        # 计算不同分类器组合的损失
        sloss_sum1, hloss1, js1, ce_loss1 = self.loss_sum_calculate(logits_1,logits_2,labels,epoch)
        sloss_sum2, hloss2, js2, ce_loss2 = self.loss_sum_calculate(logits_3,logits_4,labels,epoch)
        sloss_sum3, hloss3, js3, ce_loss3 = self.loss_sum_calculate(logits_5,logits_6,labels,epoch)
        sloss_sum4, hloss4, js4, ce_loss4 = self.loss_sum_calculate(logits_7,logits_8,labels,epoch)

        # 加权损失总和（第四个分类器组合权重为2）
        wsloss_sum1 = sloss_sum1 + sloss_sum2 + sloss_sum3 + 2 * sloss_sum4
        ce_loss= ce_loss1 + ce_loss2 + ce_loss3 + 2 * ce_loss4  # 交叉熵损失总和

        ind_sorted1 = torch.argsort(wsloss_sum1.data)  # 按损失排序样本索引
        forget_rate = min(epoch, self.Tepoch)/self.Tepoch * drop_rate_fine  # 计算当前遗忘率

        num_remember = math.ceil((1 - forget_rate) * logits_1.shape[0])  # 计算需要记住的样本数量
        ind_update1 = ind_sorted1[:num_remember]  # 获取需要更新的样本索引

        return ind_update1, wsloss_sum1, ce_loss



    def get_new_contrast_feature(self, xl_concat1, xl_concat2, labels, ind_update, ind_noise, xf_ori):
        
        xl1_contra_new=[]
        xl2_contra_new=[]
        xf_ori_new = []

        if len(ind_noise) == 0:
            contrast_labels_new = labels[ind_update]
            xl1_contra_new = xl_concat1[ind_update]
            xl2_contra_new = xl_concat2[ind_update]
            xf_ori_new = xf_ori[ind_update]
            labels_flag = labels[ind_update]

        else:      
            contrast_labels_new = labels.clone()
            contrast_labels_new = contrast_labels_new[:(len(ind_update)+len(ind_noise))]

            labels_flag = labels.clone()
            labels_flag = labels_flag[:(len(ind_update)+len(ind_noise))]

            clean_num = len(ind_update)

            for i in range(len(ind_update)):
                contrast_labels_new[i] = labels[ind_update[i]]
                labels_flag[i] = labels[ind_update[i]]

            for i in range(len(ind_noise)):
                labels_flag[i+clean_num] = -1
                contrast_labels_new[i+clean_num] = labels[ind_noise[i]]

            for i in ind_update:
                xl1_contra_new.append(xl_concat1[i])
                xl2_contra_new.append(xl_concat2[i])
                xf_ori_new.append(xf_ori[i])

            for i in ind_noise:
                xl1_contra_new.append(xl_concat1[i])
                xl2_contra_new.append(xl_concat2[i])
                xf_ori_new.append(xf_ori[i])

            xl1_contra_new = torch.stack(xl1_contra_new)
            xl2_contra_new = torch.stack(xl2_contra_new)
            xf_ori_new = torch.stack(xf_ori_new)

        return xl1_contra_new, xl2_contra_new, contrast_labels_new, xf_ori_new, labels_flag
    


    def get_contrast_loss(self, x1, x2, contrast_labels_new, xf_ori, flag=1, labels_flag=None):
        if flag == 1:
            contrast_loss = contra_loss(x1, x2, contrast_labels_new, labels_flag) 

        else:

            xl_obj_new = self.ada_maxpool1(x1)
            xl_obj_new = xl_obj_new.contiguous().view(xl_obj_new.size(0), -1)

            xl_part_new = self.pool(x2)
            xl_part_new = xl_part_new.contiguous().view(xl_part_new.size(0), xl_part_new.size(1), -1)

            xf = xf_ori.clone().detach()
            
            eps = 1e-8
            b=xf.size(0)
            c=xf.size(1)
            h=xf.size(2)
            w=xf.size(3)
        
            saliency = torch.sum(xf, dim=1)*(1.0/(c+eps))
            saliency = saliency.contiguous()
            xs = saliency.view(b, 1, h, w)
            xs = self.pool(xs)
            xs = xs.contiguous().view(b, -1)
            
            contrast_loss = 0.0
            for i in range(4):
                contrast_loss += (1.0/4)*contra_loss(xl_obj_new, xl_part_new[:,:,i], contrast_labels_new, labels_flag)

        return contrast_loss



    def forward(self, logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, labels, epoch, index, output_1_3, output_2_3, output_3_3, output_concat_3, xl_concat1, xl_concat2, xl_concat3, 
                xl3_ori, xl3_obj, xl3_part, xf_ori):
        
        drop_rate_fine = args.drop_rate
        ind_update1, wsloss_sum1, ce_loss1 = self.get_update(logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, labels, epoch, drop_rate_fine)
        ind_update2, wsloss_sum2, ce_loss2 = self.get_update(logits_2, output_1_3, logits_4, output_2_3, logits_6, output_3_3, logits_8, output_concat_3, labels, epoch, drop_rate_fine)

        ind_update1_new = []
        ind_update2_new = []      
        ind_update = []
        ind_noise = []

        for i in ind_update1:
            ind_update1_new.append(i.item())

        for i in ind_update2:
            ind_update2_new.append(i.item())

        for i in ind_update1_new:
            if i in ind_update2_new:
                ind_update.append(i)
     
        for i in range(logits_1.size(0)):
            if i not in ind_update:
                ind_noise.append(i)


        xl1_contra_new, xl2_contra_new, contrast_labels_new, xf_ori_new, labels_flag = self.get_new_contrast_feature(xl_concat1, xl_concat2, labels, ind_update, ind_noise, xf_ori)
        xl_ori_new, xl_part_new1, _ , _ , _ = self.get_new_contrast_feature(xl3_ori.clone(), xl3_part.clone(), labels, ind_update, ind_noise, xf_ori)     

        contrast_loss1 = self.get_contrast_loss(xl1_contra_new, xl2_contra_new, contrast_labels_new, xf_ori=xf_ori_new, flag=1, labels_flag = labels_flag)
        contrast_loss2 = self.get_contrast_loss(xl_ori_new, xl_part_new1, contrast_labels_new, xf_ori=xf_ori_new, flag=2, labels_flag = labels_flag)

        if epoch < self.Tepoch:
            contrast_factor = 0
        else:
            contrast_factor = 1

        contrast_loss =  contrast_loss1 + contrast_loss2 #+ contrast_loss3
        total_loss = wsloss_sum1[ind_update].mean() +  wsloss_sum2[ind_update].mean()  + contrast_factor*contrast_loss

        return total_loss



    def loss_sum_calculate(self,logits_1,logits_2,labels,epoch):
        softmax1 = F.softmax(logits_1, dim=1)
        softmax2 = F.softmax(logits_2, dim=1)
        M = (softmax1+softmax2)/2.

        loss_1 = self.label_smooth_loss(logits_1, labels)
        loss_2 = self.label_smooth_loss(logits_2, labels)

        H = torch.sum(-torch.log(softmax1 + 1e-7) * softmax1, dim=-1) + \
              torch.sum(-torch.log(softmax2 + 1e-7) * softmax2, dim=-1)
        js = F.kl_div(M.log(), softmax2, reduction='none').sum(1)  + F.kl_div(M.log(), softmax1, reduction='none').sum(1) 
        js = js/2.0
 
        js = js * 10
        sloss_sum = loss_1 + loss_2 + js

        ce_loss =  loss_1 + loss_2

        return sloss_sum, H, js, ce_loss



def train(nb_epoch, batch_size, store_name, start_epoch=0):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print('use cuda:',use_cuda)

   
    # Data
    print('==> Preparing data..')
    if args.data == 'bird':
        transform_train = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.RandomCrop(448, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.RandomCrop(448, padding=8),
            transforms.RandomHorizontalFlip(),
            AutoAugImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])


    trainset = Imagefolder_modified(root='./data/web-{}/train'.format(args.data), transform=transform_train, number = args.each_class)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
    print('train image number is ', len(trainset))


    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = Imagefolder_modified(root='./data/web-{}/val'.format(args.data), transform=transform_test, number = args.each_class)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
    print('val image number is ', len(testset))



    # Model
    if args.net == 'resnet50':
        net1 = load_ms_layer(model_name='resnet50_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net2 = load_ms_layer(model_name='resnet50_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net3 = load_ms_layer(model_name='resnet50_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        saliency_sampler = Saliency_Sampler()

    elif args.net == 'resnet101':
        net1 = load_ms_layer(model_name='resnet101_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net2 = load_ms_layer(model_name='resnet101_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net3 = load_ms_layer(model_name='resnet101_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        saliency_sampler = Saliency_Sampler()

    elif args.net == 'resnet152':
        net1 = load_ms_layer(model_name='resnet152_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net2 = load_ms_layer(model_name='resnet152_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net3 = load_ms_layer(model_name='resnet152_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        saliency_sampler = Saliency_Sampler()



    if args.gpus > 1:
        net1 = torch.nn.DataParallel(net1)
        net2 = torch.nn.DataParallel(net2)
        net3 = torch.nn.DataParallel(net3)



    net1.cuda()
    net2.cuda()
    net3.cuda()
    saliency_sampler.cuda()



    HclLoss = HCL_loss(labels_all=0, Tepoch =10, drop_rate = 0.25, class_num=len(trainset.classes))
    if args.gpus > 1:
        optimizer = optim.SGD([
            {'params': net1.module.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net1.module.conv_block1.parameters(), 'lr': 0.002},
            {'params': net1.module.classifier1.parameters(), 'lr': 0.002},
            {'params': net1.module.conv_block2.parameters(), 'lr': 0.002},
            {'params': net1.module.classifier2.parameters(), 'lr': 0.002},
            {'params': net1.module.conv_block3.parameters(), 'lr': 0.002},
            {'params': net1.module.classifier3.parameters(), 'lr': 0.002},
            {'params': net1.module.features.parameters(), 'lr': 0.0002},
            {'params': net1.module.conv_block_map.parameters(), 'lr': 0.002},

            {'params': net2.module.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net2.module.conv_block1.parameters(), 'lr': 0.002},
            {'params': net2.module.classifier1.parameters(), 'lr': 0.002},
            {'params': net2.module.conv_block2.parameters(), 'lr': 0.002},
            {'params': net2.module.classifier2.parameters(), 'lr': 0.002},
            {'params': net2.module.conv_block3.parameters(), 'lr': 0.002},
            {'params': net2.module.classifier3.parameters(), 'lr': 0.002},
            {'params': net2.module.features.parameters(), 'lr': 0.0002},
            {'params': net2.module.conv_block_map.parameters(), 'lr': 0.002},

            {'params': net3.module.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net3.module.conv_block1.parameters(), 'lr': 0.002},
            {'params': net3.module.classifier1.parameters(), 'lr': 0.002},
            {'params': net3.module.conv_block2.parameters(), 'lr': 0.002},
            {'params': net3.module.classifier2.parameters(), 'lr': 0.002},
            {'params': net3.module.conv_block3.parameters(), 'lr': 0.002},
            {'params': net3.module.classifier3.parameters(), 'lr': 0.002},
            {'params': net3.module.features.parameters(), 'lr': 0.0002},
            {'params': net3.module.conv_block_map.parameters(), 'lr': 0.002},
        ], momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.SGD([
            {'params': net1.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net1.conv_block1.parameters(), 'lr': 0.002},
            {'params': net1.classifier1.parameters(), 'lr': 0.002},
            {'params': net1.conv_block2.parameters(), 'lr': 0.002},
            {'params': net1.classifier2.parameters(), 'lr': 0.002},
            {'params': net1.conv_block3.parameters(), 'lr': 0.002},
            {'params': net1.classifier3.parameters(), 'lr': 0.002},
            {'params': net1.features.parameters(), 'lr': 0.0002},
            {'params': net1.conv_block_map.parameters(), 'lr': 0.002},

            {'params': net2.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net2.conv_block1.parameters(), 'lr': 0.002},
            {'params': net2.classifier1.parameters(), 'lr': 0.002},
            {'params': net2.conv_block2.parameters(), 'lr': 0.002},
            {'params': net2.classifier2.parameters(), 'lr': 0.002},
            {'params': net2.conv_block3.parameters(), 'lr': 0.002},
            {'params': net2.classifier3.parameters(), 'lr': 0.002},
            {'params': net2.features.parameters(), 'lr': 0.0002},
            {'params': net2.conv_block_map.parameters(), 'lr': 0.002},

            {'params': net3.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net3.conv_block1.parameters(), 'lr': 0.002},
            {'params': net3.classifier1.parameters(), 'lr': 0.002},
            {'params': net3.conv_block2.parameters(), 'lr': 0.002},
            {'params': net3.classifier2.parameters(), 'lr': 0.002},
            {'params': net3.conv_block3.parameters(), 'lr': 0.002},
            {'params': net3.classifier3.parameters(), 'lr': 0.002},
            {'params': net3.features.parameters(), 'lr': 0.0002},
            {'params': net3.conv_block_map.parameters(), 'lr': 0.002},
        ], momentum=0.9, weight_decay=1e-5)

    
    if os.path.exists(exp_dir + '/HCL_results_train.txt'):
        os.remove(exp_dir + '/HCL_results_train.txt')
    if os.path.exists(exp_dir + '/HCL_test.txt'):
        os.remove(exp_dir + '/HCL_test.txt')

    
    max_val_acc_concat = 0


    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002, 0.002]*3


    for epoch in range(start_epoch, nb_epoch):

        # print('Epoch: %d' % epoch)
        start = time.time()
        net1.train()
        net2.train()
        net3.train()
        saliency_sampler.train()


        train_loss = 0
        correct = 0
        total = 0
        idx = 0


        for batch_idx, (inputs, targets, index) in enumerate(trainloader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

       
            output_1_1, output_2_1, output_3_1, output_concat_1, coord, xl_concat1, xl3_ori, xf_ori = net1(inputs, 4)

            
            inputs_obj = saliency_sampler(inputs.clone(), xf_ori)
            coord = coord.detach().cpu()
            coord = coord.numpy()
            coord = np.uint8(coord)
            inputs_salient = inputs.clone()
            inputs_batch_size = inputs.size(0)
            for i in range(inputs_batch_size):
                a,b,c,d = coord[i]
                saliency_figure = inputs[i,:,:,:].clone()
                show = saliency_figure[:,32*b:32*(b+d),32*a:32*(a+c)]
                show = show.unsqueeze(0)
                show = F.interpolate(show, size=[448,448], mode='bilinear', align_corners=True)
                show=show.squeeze(0)

                inputs_salient[i,:,:,:] = show
            

            output_1_2, output_2_2, output_3_2, output_concat_2, _ , xl_concat2, xl3_obj, _ = net2(inputs_obj, 4)


            inputs_part, _  = jigsaw_generator(inputs_salient, 2)
            output_1_3, output_2_3, output_3_3, output_concat_3, _ , xl_concat3, xl3_part, _ = net3(inputs_part, 4)


            loss = HclLoss(output_1_1,output_1_2, output_2_1, output_2_2,output_3_1,output_3_2,output_concat_1,output_concat_2, targets, epoch, index, 
            output_1_3, output_2_3, output_3_3, output_concat_3, xl_concat1, xl_concat2, xl_concat3, xl3_ori, xl3_obj, xl3_part, xf_ori)                 


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            #  training log
            _, predicted1 = torch.max(output_concat_1.data, 1)
            _, predicted2 = torch.max(output_concat_2.data, 1)
            _, predicted3 = torch.max(output_concat_3.data, 1)
            total += targets.size(0)
            correct += (predicted1.eq(targets.data).cpu().sum() + predicted2.eq(targets.data).cpu().sum() + predicted3.eq(targets.data).cpu().sum())/3.
            train_loss += loss.item()

        train_acc = 100. * float(correct) / total

        with open(exp_dir + '/HCL_results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                epoch, train_acc, train_loss/ (idx + 1) ))


        if epoch < 10 or epoch%5==0 or epoch>nb_epoch-20:
            net1.eval()
            net2.eval()
            net3.eval()
            saliency_sampler.eval()


            topconcat_val = AverageMeter()


            total = 0
            idx = 0


            with torch.no_grad():
                for batch_idx, (inputs, targets, _) in enumerate(testloader):
                    idx = batch_idx
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output_1_1, output_2_1, output_3_1, output_concat_1, coord, _, _ , xf_ori  = net1(inputs, 4)

                    inputs_obj = saliency_sampler(inputs.clone(), xf_ori)

                    coord = coord.detach().cpu()
                    coord = coord.numpy()
                    coord = np.uint8(coord)
                    inputs_salient = inputs.clone()
                    inputs_batch_size = inputs.size(0)
                    for i in range(inputs_batch_size):
                        a,b,c,d = coord[i]
                        saliency_figure = inputs[i,:,:,:].clone()
                        show = saliency_figure[:,32*b:32*(b+d),32*a:32*(a+c)]
                        show = show.unsqueeze(0)
                        show = F.interpolate(show, size=[448,448], mode='bilinear', align_corners=True)
                        show=show.squeeze(0)

                        inputs_salient[i,:,:,:] = show

                    output_1_2, output_2_2, output_3_2, output_concat_2, _, _, _, _ = net2(inputs_obj, 4)

                    output_1_3, output_2_3, output_3_3, output_concat_3, _, _ , _, _ = net3(inputs_salient, 4)


                    outputs_concat = output_concat_1 + output_concat_2 + output_concat_3
                    prec1 = accuracy(outputs_concat.float().data, targets)[0]
                    topconcat_val.update(prec1.item(), inputs.size(0))


            val_acc_concat = topconcat_val.avg



            show_param = 'epoch: %d |sum Loss: %.3f | train Acc: %.3f%%  | test Acc: %.3f%% time%.1fmin(%.1fh)\n' % (
                    epoch, train_loss/ (idx + 1),
                    train_acc, val_acc_concat, (time.time()-start)/60, (time.time()-start)*(nb_epoch-epoch-1)/3600 )
                        


            if val_acc_concat > max_val_acc_concat:
                max_val_acc_concat = val_acc_concat

                print('*'+show_param)

                net1.cpu()
                torch.save(net1, exp_dir + '/best_total_concat-net1.pth')
                net1.cuda()

                net2.cpu()
                torch.save(net2, exp_dir + '/best_total_concat-net2.pth')
                net2.cuda()

                net3.cpu()
                torch.save(net3, exp_dir + '/best_total_concat-net3.pth')
                net3.cuda()

                saliency_sampler.cpu()
                torch.save(saliency_sampler, exp_dir + '/best_total_concat-ss.pth')
                saliency_sampler.cuda()

            else:
                print(show_param)


            with open(exp_dir + '/HCL_test.txt', 'a') as file:
                file.write('Iteration %d, test acc = %.5f\n' % (
                epoch, val_acc_concat))


    print('--------------------------------------------\n')


    print('best test acc: {} '.format(max_val_acc_concat))


    with open(exp_dir + '/HCL_test.txt', 'a') as file:
        file.write('best test acc: {}'.format(max_val_acc_concat))


if __name__=="__main__":
    start_time = time.time()
    train(nb_epoch=args.epochs,             # number of epoch
            batch_size=args.bs,         # batch size
            store_name=args.save_dir,     # folder for output
            start_epoch=0,         # the start epoch
    )       

    print('--------------------------------------------')
    print('total time: {:.1f}h'.format((time.time()-start_time)/3600))


    with open(args.save_dir + '/HCL_test.txt', 'a') as file:
        file.write('\ntotal time: {:.1f}h'.format((time.time()-start_time)/3600))
