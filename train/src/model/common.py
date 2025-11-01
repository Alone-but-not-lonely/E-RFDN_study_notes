'''
# =======================================
# model 模块的入口
# =======================================
- 文件作用: 定义卷积、残差块、上采样层、均值偏移层等通用组件（供 EDSR、RCAN、VDSR 等模型调用）
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# =========================================================================
# 默认卷积函数 default_conv
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 创建一个带自动 padding 的 2D 卷积
* 参数: 
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - kernel_size: 卷积核大小
    - bias: 是否包含偏置项
# =========================================================================
'''
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias) # padding = kernel_size // 2 使得卷积前后尺寸一致

'''
# =========================================================================
# 图像归一化层 MeanShift
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承: PyTorch 的 nn.Conv2d 类
* 作用: 本质上就是一个 1×1 卷积，用来做均值平移和标准化/反标准化
* 参数:
    - rgb_range: 图像像素值范围，通常为 1 或 255
    - rgb_mean: 图像的 RGB 通道均值
    - rgb_std: 图像的 RGB 通道标准差
    - sign: 归一化方向，-1 表示减去均值，1 表示加上均值（复原到原图域）
# =========================================================================
'''
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1) # 定义一个 3→3（输入输出通道数） 的 1×1 卷积
        std = torch.Tensor(rgb_std) # 转换为 Tensor 类型
        #---* 标准化卷积核 *---#
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # torch.eye(3) 生成一个 3×3 的单位矩阵
        # view(3, 3, 1, 1) 将其转换为 3×3×1×1 的四维张量，符合卷积核的形状
        # weight.data 是卷积层的权重参数，用于存储卷积核的数值
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1) # 每个元素都是 1/标准差
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std # 定义偏置项，复原到原图域
        for p in self.parameters():
            p.requires_grad = False # 不可训练，纯粹数据预处理/后处理

'''
# =========================================================================
# 基础块 BasicBlock 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承: PyTorch 的 nn.Sequential 类
* 结构: 卷积 + BN + 激活
* 参数:
    - stride: 卷积核的步幅，默认值为 1
    - bn: 是否使用批量归一化层，默认值为 True
    - act: 激活函数，默认值为 ReLU 函数
# =========================================================================
'''
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)] # 卷积层
        if bn:
            m.append(nn.BatchNorm2d(out_channels)) # BN 层, 使用批量归一化
        if act is not None:
            m.append(act) # 激活函数

        super(BasicBlock, self).__init__(*m) # 把层组合进 Sequential 类，返回一个可直接 forward 的模块

'''
# =========================================================================
# 残差块 ResBlock 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承: PyTorch 的 nn.Module 类
* 结构: 卷积 + BN + 激活 + 卷积 + BN
* 参数:
    - n_feats: 特征图通道数
    - bn: 是否使用批量归一化层，默认值为 True
    - act: 激活函数，默认值为 ReLU 函数
    - res_scale: 残差缩放因子，默认值为 1
# =========================================================================
'''
class ResBlock(nn.Module):
    ''' 初始化 '''
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__() # 调用父类的方法初始化残差块
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias)) # 卷积层
            if bn:
                m.append(nn.BatchNorm2d(n_feats)) # BN 层, 使用批量归一化
            if i == 0:
                m.append(act) # 激活函数

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    ''' 前向传播 '''
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)  # 经过残差块处理后, 再乘以缩放因子
        res += x  # 残差连接, 加上输入特征图

        return res

'''
# =========================================================================
# 上采样器 Upsampler 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承: PyTorch 的 nn.Sequential 类
* 结构: 卷积 + 上采样 + BN + 激活
* 参数:
    - conv: 卷积层
    - scale: 上采样因子
    - n_feats: 特征图通道数
    - bn: 是否使用批量归一化层，默认值为 False
    - act: 激活函数，默认值为 False
    - bias: 是否使用偏置项，默认值为 True
# =========================================================================
'''
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        #---* 上采样因子为 2^n 时 *---#
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))): # 循环 log2(scale) 次, 每次上采样因子翻倍
                # 卷积层, 输出通道数翻倍
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                # PixelShuffle 上采样层, 上采样因子翻倍
                m.append(nn.PixelShuffle(2))
                # 是否使用批量归一化 BN 层
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                # 是否使用激活函数
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        #---* 上采样因子为 3 时 *---#
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))  # 卷积层, 输出通道数三倍
            m.append(nn.PixelShuffle(3))  # PixelShuffle 上采样层, 上采样因子三倍
            # 是否使用批量归一化 BN 层
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            # 是否使用激活函数
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError # 其他上采样因子: 未实现

        super(Upsampler, self).__init__(*m)