import torch.nn as nn  # 导入PyTorch神经网络模块
import torch  # 导入PyTorch主库

from einops import rearrange  # 导入einops重排操作
import numpy as np  # 导入NumPy数值计算库
import cv2  # 导入OpenCV计算机视觉库

import torch.nn.functional as F  # 导入PyTorch函数模块



def makeGaussian(size, fwhm = 3, center=None):
    """创建方形高斯核

    size是正方形边的长度
    fwhm是全宽半最大值，可以看作是有效半径

    Args:
        size: 高斯核大小
        fwhm: 全宽半最大值，控制高斯分布宽度
        center: 高斯中心坐标，None表示中心位置

    Returns:
        np.ndarray: 方形高斯核矩阵
    """

    x = np.arange(0, size, 1, float)  # 创建x坐标数组
    y = x[:,np.newaxis]  # 创建y坐标数组

    if center is None:
        x0 = y0 = size // 2  # 默认中心位置
    else:
        x0 = center[0]  # 指定x中心
        y0 = center[1]  # 指定y中心

    # 计算高斯分布
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



class Saliency_Sampler(nn.Module):
    """显著性采样器 - 基于显著性图对图像进行采样
    
    通过分析特征图的显著性来生成采样网格，用于聚焦重要区域
    """
    def __init__(self, task_input_size=448):
        """初始化显著性采样器
        
        Args:
            task_input_size: 任务输入图像大小，默认448x448
        """
        super(Saliency_Sampler, self).__init__()
        
        self.grid_size = 31  # 网格大小
        self.padding_size = 30  # 填充大小
        self.global_size = self.grid_size+2*self.padding_size  # 全局大小
        self.input_size_net = task_input_size  # 网络输入尺寸
        
        # 创建高斯权重用于滤波
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size+1, fwhm = 13))

        # 创建高斯滤波器
        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size+1,2*self.padding_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights  # 设置高斯权重

        # 创建基础位置矩阵
        self.P_basis = torch.zeros(2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    # 计算位置基础值，用于生成采样网格
                    self.P_basis[k,i,j] = k*(i-self.padding_size)/(self.grid_size-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_size-1.0)
        
        self.l_relu = nn.LeakyReLU(0.2)  # LeakyReLU激活函数


    def create_grid(self, x):
        """创建采样网格 - 基于显著性图生成用于采样的网格
        
        Args:
            x: 输入显著性图
            
        Returns:
            torch.Tensor: 采样网格，用于grid_sample操作
        """
        # 创建基础位置矩阵并扩展到批次大小
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda(),requires_grad=False)
        P[0,:,:,:] = self.P_basis
        P = P.expand(x.size(0),2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)

        # 连接输入并进行滤波操作
        x_cat = torch.cat((x,x),1)  # 在通道维度连接
        p_filter = self.filter(x)  # 应用高斯滤波
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size,self.global_size)  # 位置加权
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size,self.grid_size)  # 再次滤波

        # 分离x和y方向的滤波结果
        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)

        # 归一化滤波结果
        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter

        # 转换为网格坐标（-1到1范围）
        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)  # 限制范围
        ygrids = torch.clamp(ygrids,min=-1,max=1)
        xgrids = xgrids.view(-1,1,self.grid_size,self.grid_size)
        ygrids = ygrids.view(-1,1,self.grid_size,self.grid_size)

        # 合并x和y网格
        grid = torch.cat((xgrids,ygrids),1)

        # 插值到目标尺寸
        grid = F.interpolate(grid, size=[self.input_size_net, self.input_size_net], mode='bilinear', align_corners=True)

        # 调整维度顺序以匹配grid_sample要求
        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid


    def forward(self, x, xf_ori, flag = 1):
        """前向传播 - 基于显著性图对输入图像进行采样
        
        Args:
            x: 输入图像
            xf_ori: 原始特征图，用于计算显著性
            flag: 采样标志，1表示使用显著性采样
            
        Returns:
            torch.Tensor: 采样后的图像
        """
        if flag == 1:
            scale_factor = 0.25  # 缩放因子
                       
            xf = xf_ori.clone().detach()  # 克隆并分离特征图

            eps = 1e-8  # 防止除零的小值
            b=xf.size(0)  # 批次大小
            c=xf.size(1)  # 通道数
            h=xf.size(2)  # 高度
            w=xf.size(3)  # 宽度
            
            # 计算显著性图（通道求和并归一化）
            saliency = torch.sum(xf, dim=1)*(1.0/(c+eps))
            saliency = saliency.contiguous()
            xs = saliency.view(b, 1, h, w)
            xs = xs/scale_factor  # 缩放显著性
            
            # 插值到网格大小并应用softmax
            xs = F.interpolate(xs, size=[self.grid_size, self.grid_size], mode='bilinear', align_corners=True)
            xs = xs.view(-1, self.grid_size*self.grid_size)
            xs = nn.Softmax()(xs)  # 应用softmax得到概率分布

            # 重塑并填充
            xs = xs.view(-1, 1, self.grid_size, self.grid_size)
            xs_hm = nn.ReflectionPad2d(self.padding_size)(xs)  # 反射填充

            # 创建采样网格并进行采样
            grid = self.create_grid(xs_hm)
            x_sampled = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return x_sampled



class MS_resnet_layer(nn.Module):
    def __init__(self, model,net_id, feature_size, classes_num):
        super(MS_resnet_layer, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        if net_id == 50 or net_id == 101 or net_id == 152:
            self.num_ftrs = 2048 * 1 * 1
        elif net_id == 18:
            self.num_ftrs = 512 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2 * 3),
            nn.Linear(self.num_ftrs//2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(feature_size, classes_num),
        )

        self.ada_maxpool14 = nn.AdaptiveMaxPool2d((14, 14))

        self.conv_block_map = nn.Sequential(
            
            BasicConv(1024*3, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, feature_size, kernel_size=1, stride=1, padding=0, relu=False)

        )

        self.ada_maxpool1 = nn.AdaptiveMaxPool2d((1, 1))


    #saliency map计算
    def saliency_extraction(self, xf_ori):

        xf = xf_ori.clone()
        
        eps = 1e-8
        b=xf.size(0)
        c=xf.size(1)
        h=xf.size(2)
        w=xf.size(3)


        coord = torch.zeros(b, 4)
        coord = coord.cuda()


        saliency = torch.sum(xf, dim=1)*(1.0/(c+eps))

        saliency = saliency.contiguous()
        saliency = saliency.view(b, -1)


        sa_min = torch.min(saliency, dim=1)[0]
        sa_max = torch.max(saliency, dim=1)[0]
        interval = sa_max - sa_min


        sa_min = sa_min.contiguous()
        sa_min = sa_min.view(b, 1)
        sa_min = sa_min.expand(h, w, b, 1)
        sa_min = sa_min.contiguous()
        sa_min = rearrange(sa_min, 'h w b 1 -> b 1 h w')


        interval = interval.contiguous()
        interval = interval.view(b, 1)
        interval = interval.expand(h, w, b, 1)
        interval = interval.contiguous()
        interval = rearrange(interval, 'h w b 1 -> b 1 h w')


        saliency = saliency.contiguous()
        saliency = saliency.view(b, 1, h, w)

        saliency = saliency - sa_min
        saliency = saliency/(interval+eps)

        saliency = torch.clamp(saliency, eps, 1)

 
        for i in range(b):
            img1 = saliency[i,:,:,:]
            img2 = img1.view(1, h, w)
            img2 = img2*255
            img2 = img2.detach().cpu()
            img2 = img2.numpy()
            mat1 = np.uint8(img2)
            mat1 = mat1.transpose(1,2,0)
            thres, mat2 = cv2.threshold(mat1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours(mat2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            area = []

            if len(contours)==0:
                coord[i, 0]=0
                coord[i, 1]=0
                coord[i, 2]=w
                coord[i, 3]=h
            else:

                for k in range(len(contours)):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))

                p, q, r, s = cv2.boundingRect(contours[max_idx]) 
                coord[i, 0]=p
                coord[i, 1]=q
                coord[i, 2]=r
                coord[i, 3]=s


        coord = coord.detach()

        return coord, coord



    def forward(self, x , layer):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xf_ori = xf5.clone()

        _, coord = self.saliency_extraction(xf5)


        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)


        xl1_contrast = self.ada_maxpool14(xl1.clone())
        xl2_contrast = self.ada_maxpool14(xl2.clone())
        xl3_contrast = xl3.clone()
        xl_part_contrast = torch.cat((xl1_contrast, xl2_contrast, xl3_contrast), dim=1)
        xl_part_contrast = self.conv_block_map(xl_part_contrast)
        xl_concat = self.ada_maxpool1(xl_part_contrast.clone())
        xl_concat = xl_concat.contiguous().view(xl_concat.size(0), -1)

     
        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)
        if layer == 1:
            return xc1
        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)
        if layer == 2:
            return xc2
        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)
        if layer == 3:
            return xc3
        
        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)

        if layer == 0:
            return x_concat
        return xc1, xc2, xc3, x_concat, coord, xl_concat, xl_part_contrast, xf_ori 
    
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
