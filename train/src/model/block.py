'''
# =======================================
# 基础模块定义 —— 与测试时 RFDN.py 同级的 block.py 一样
# =======================================
- 文件作用: 这是网络最基础的“零件库”。
    - 定义了构成 RFDN 的所有基础组件，如卷积层、激活函数、注意力模块（ESA）和核心的残差特征蒸馏块（RFDB）。
'''

import torch.nn as nn

import torch
import torch.nn.functional as F

'''
# =========================================================================
# 封装一个标准的 2D 卷积层
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数:
    - dilation: 膨胀卷积，用于增加感受野而不减少特征图尺寸，1 表示普通卷积
    - groups: 分组卷积，用于减少计算量，1 表示普通卷积
# =========================================================================
'''
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    # 自动计算 padding，使得卷积后特征图尺寸不变 (当 stride=1 时)
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

'''
# =========================================================================
# 封装归一化层
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数:
    - norm_type: 归一化类型，可选 'batch' 或 'instance'
        - batch: 批量归一化，对每个批次的所有样本进行归一化
        - instance: 实例归一化，对每个样本的所有通道进行归一化
    - nc: 输入通道数
# =========================================================================
'''
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

'''
# =========================================================================
# 封装填充层
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数:
    - pad_type: 填充类型，可选 'reflect' 或 'replicate'
        - reflect: 反射填充，以图像的边界作为“镜面”，将边界另一侧的像素值“反射”过来作为填充值。关键点在于，镜面本身（即边界像素）不参与反射过程。
            # 例如，[1, 2, 3, 4] 反射填充 2 次变为 [3, 2, 1, 2, 3, 4, 3, 2]
        - replicate: 复制填充，从边界向内部复制填充
            # 例如，[1, 2, 3, 4] 复制填充 2 次变为 [1, 1, 1, 2, 3, 4, 4, 4]
    - padding: 填充宽度
# =========================================================================
'''
def pad(pad_type, padding):
    pad_type = pad_type.lower() # 转换为小写，确保不区分大小写
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

'''
# =========================================================================
# 计算有效填充宽度
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数:
    - kernel_size: 卷积核大小
    - dilation: 膨胀系数
# =========================================================================
'''
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

'''
# =========================================================================
# 封装一个 "卷积 -> 归一化 -> 激活" 的标准卷积块
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数:
    - in_nc: 输入通道数
    - out_nc: 输出通道数
    - kernel_size: 卷积核大小
    - stride: 步长
    - bias: 是否使用偏置项
    - norm_type: 归一化类型，可选 'batch' 或 'instance'
    - act_type: 激活函数类型，可选 'relu'、'lrelu'、'prelu'
        - relu: 普通的 ReLU 激活函数
        - lrelu: LeakyReLU 激活函数，用于解决 ReLU 中的“死亡神经元”问题
        - prelu: 参数化 ReLU 激活函数，可学习斜率
# =========================================================================
'''
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    # 若填充类型不是 'zero'，则添加填充层 p
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    # 若填充类型是 'zero'，则 padding 为前面得到的 padding，并在卷积层中使用实现填充（也就是没有独立的填充层）
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a) # 将这些层按顺序组合起来 p是填充层 c是卷积层 n是归一化层 a是激活层

'''
# =========================================================================
# 封装激活函数层
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数:
    - act_type: 激活函数类型，可选 'relu'、'lrelu'、'prelu'
    - inplace: 是否在原地操作，即是否直接修改输入张量，而不是创建新的张量
    - neg_slope: LeakyReLU 中的负斜率，默认 0.05
    - n_prelu: PReLU 中的参数数量，默认 1
# =========================================================================
'''
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

'''
# =========================================================================
# 封装残差连接层（ShortcutBlock 类）
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：实现残差连接（Residual Connection），用于解决深度神经网络中的梯度消失问题
* 参数: submodule: 子模块，即残差连接的部分
# =========================================================================
'''
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        # super 是 Python 中用于调用父类（超类）的方法的函数
        # 这里用于调用 ShortcutBlock 的父类 nn.Module 的初始化方法
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

'''
# =========================================================================
# 计算特征图的通道均值
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数: F: 输入特征图，形状为 [batch_size, channels, height, width]
# =========================================================================
'''
def mean_channels(F):
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
    # assert 语句用于在运行时检查一个条件是否为 True
    # 如果条件为 False，则会抛出 AssertionError 异常
    # 这里用于检查输入张量 F 的维度是否为 4（即批量大小、通道数、高度、宽度）
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
    assert(F.dim() == 4)
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
    # 对张量的第3维（宽度维度）进行求和，同时保持维度数量不变
    # 再对第2维（高度维度）进行求和，同时保持维度数量不变
    # spatial_sum 形状为 [batch_size, channels, 1, 1] ，即每个通道的空间总和
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    # 分母 F.size(2) * F.size(3) 表示空间维度的总元素个数（高度×宽度）
    # 得到每个通道的空间平均值
    return spatial_sum / (F.size(2) * F.size(3))

'''
# =========================================================================
# 计算特征图的通道标准差
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 参数: F: 输入特征图，形状为 [batch_size, channels, height, width]
# =========================================================================
'''
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
    # .pow(2) 表示对每个元素进行平方操作
    # 对张量的第3维（宽度维度）进行求和，同时保持维度数量不变
    # 再对第2维（高度维度）进行求和，同时保持维度数量不变
    # F_variance 形状为 [batch_size, channels, 1, 1] ，即每个通道的空间方差
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    # .pow(0.5) 表示对每个元素进行开方操作，即得到标准差
    return F_variance.pow(0.5)

'''
# =========================================================================
# 封装一个有序的模块容器
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：将多个模块按顺序组合起来，形成一个新的模块
* 参数: *args: 多个模块，每个模块可以是 nn.Module 或 nn.Sequential 类型
# =========================================================================
'''
def sequential(*args):
    if len(args) == 1:
        # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
        # isinstance() 函数用于检查一个对象是否是指定的类型或类型元组
        # OrderedDict：有序字典类型，用于存储键值对，保持插入顺序
        # raise 语句用于在运行时抛出异常
        # 这里用于抛出 NotImplementedError 异常，因为 sequential 不支持 OrderedDict 输入
        # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    #---* 遍历 args 中的每个模块 *---#
    for module in args:
        if isinstance(module, nn.Sequential):
            # 如果模块是 nn.Sequential 类型，说明它包含多个子模块
            # 则遍历其所有子模块，并将其添加到 modules 列表中
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            # 如果模块是 nn.Module 类型，说明它是一个单独的模块
            # 则直接将其添加到 modules 列表中
            modules.append(module)
    #---* 使用 nn.Sequential 封装所有模块，按顺序组合起来 *---#
    return nn.Sequential(*modules)

'''
# =========================================================================
# ESA (Enhanced Spatial Attention) 模块
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：一个轻量级的注意力模块，让网络关注特征图中更重要的空间区域
* 参数: 
    - n_feats: 输入特征图的通道数，即特征通道数
    - conv: 卷积层类型，用于构建卷积层
# =========================================================================
'''
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        ''' 定义各个网咯层的参数 '''
        super(ESA, self).__init__()
        f = n_feats // 4 # 将特征通道数缩减为 1/4，以减少计算量
        self.conv1 = conv(n_feats, f, kernel_size=1) # 1x1 卷积降维
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0) # 带步长的卷积，用于缩小特征图尺寸
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1) # 1x1 卷积升维，恢复通道数
        self.sigmoid = nn.Sigmoid() # Sigmoid 函数生成 0-1 之间的注意力权重
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ''' 具体的前向传播过程 '''
        c1_ = (self.conv1(x)) # 等价于 c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) # 大尺寸池化以捕获全局信息
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        # 使用双线性插值将特征图上采样回原始尺寸；bilinear：双线性插值上采样；align_corners：是否对齐角点，默认 False
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf) # 融合全局和局部信息
        m = self.sigmoid(c4) # 生成注意力图 (attention map)
        
        return x * m # 将原始输入特征 x 与注意力图 m 逐元素相乘，实现特征重标定

'''
# =========================================================================
# RFDB (Residual Feature Distillation Block) （实际上是 E-RFDB）
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：一个残差特征蒸馏块，用于提取和压缩特征表示，是 RFDN 中最核心的模块
* 参数: 
    - in_channels: 输入特征图的通道数，即特征通道数
    - distillation_rate: 蒸馏率，默认 0.25
# =========================================================================
'''
class RFDB(nn.Module):
    ''' 定义各个网咯层的参数 '''
    def __init__(self, in_channels, distillation_rate=0.25): # distillation_rate 在这个实现中没有用到
        super(RFDB, self).__init__()
        # 在这个版本的代码中，作者将蒸馏通道数 dc 固定为输入通道数的一半
        self.dc = self.distilled_channels = in_channels//2 # ** 相当于默认蒸馏率为 0.5 !
        self.rc = self.remaining_channels = in_channels # 剩余通道数等于输入通道数

        #---* 第一次蒸馏层定义 *---#
        self.c1_d = conv_layer(in_channels, self.dc, 1) # 1x1 卷积，用于“蒸馏”特征
        self.c1_r = conv_layer(in_channels, self.rc, 3) # 3x3 卷积，用于处理“剩余”特征
        #---* 第二次蒸馏层定义 *---#
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        #---* 第三次蒸馏层定义 *---#
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)

        #---* 第四次蒸馏层定义 *---#
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05) # LeakyReLU 激活函数；neg_slope：负斜率，默认 0.05

        #---* 特征融合层定义 *---#
        self.c5 = conv_layer(self.dc*4, in_channels, 1) # 1x1 卷积，用于融合所有蒸馏出的特征

        #---* ESA 注意力模块定义 *---#
        self.esa = ESA(in_channels, nn.Conv2d)

    ''' 具体的前向传播过程 '''
    def forward(self, input):
        #---* 第一次蒸馏 *---#
        distilled_c1 = self.act(self.c1_d(input)) # 1x1 卷积提取出的特征
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input) # 剩余特征与输入相加 (局部残差连接)

        #---* 第二次蒸馏 *---#
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1) # 局部残差连接

        #---* 第三次蒸馏 *---#
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2) # 局部残差连接

        #---* 第四次蒸馏 + 激活 *---#
        r_c4 = self.act(self.c4(r_c3)) # 比原论文框架图中多出了一部分激活函数

        #---* 通道拼接 *---#
        # 将所有蒸馏出的特征 (distilled_c*) 和 r_c4 在通道维度上拼接
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        #---* 1x1 卷积融合特征 + ESA *---#
        out_fused = self.esa(self.c5(out))

        return out_fused

'''
# =========================================================================
# 上采样模块（PixelShuffle 上采样）
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：将特征图的空间维度放大，同时减少通道数，用于恢复高分辨率特征图
* 参数: 
    - in_channels: 输入特征图的通道数
    - out_channels: 输出特征图的通道数
    - upscale_factor: 上采样因子，默认 2，表示将特征图的空间维度放大 2 倍
    - kernel_size: 卷积核大小，默认 3
    - stride: 卷积步长，默认 1
# =========================================================================
'''
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    # 先用一个卷积层将通道数扩大 upscale_factor^2 倍
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    # PixelShuffle 层会将 (C * r^2, H, W) 的特征图重排为 (C, H * r, W * r)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle) # 将卷积和 PixelShuffle 组合起来
