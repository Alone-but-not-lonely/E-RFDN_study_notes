'''
# =======================================
# 判别器
# =======================================
- 文件作用: 定义判别器网络，用于判断输入图像是否为真实图像
'''
from model import common

import torch.nn as nn

'''
# =========================================================================
# 判别器 Discriminator 类, 继承自 nn.Module
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 自定义的判别器类, 负责判断输入图像是否为真实图像
# =========================================================================
'''
class Discriminator(nn.Module):
    ''' 初始化判别器网络 '''
    def __init__(self, args):
        #---* 调用父类初始化方法 *---#
        super(Discriminator, self).__init__()

        in_channels = args.n_colors # 输入通道数
        out_channels = 64 # 输出通道数
        depth = 7 # 网络深度

        ''' 基础卷积块构建函数 '''
        #------------------------------------------------------------------
        # 作用: 构建基础卷积块, 并返回按顺序堆叠的神经网络层：2D 卷积层 -> 2D 批量归一化层 -> 激活函数
        # 参数: _in_channels 输入通道数, _out_channels 输出通道数, stride 卷积层步长
        # Sequential 容器: 按顺序堆叠神经网络层，并自动处理前向传播
        #------------------------------------------------------------------
        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                #---* 2D 卷积层 *---#
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3, # 卷积核大小
                    padding=1, # 填充大小
                    stride=stride, # 步长
                    bias=False # 不使用偏置项
                ),
                #---* 2D 批量归一化层 *---#
                nn.BatchNorm2d(_out_channels),
                #---* 激活函数 *---#
                nn.LeakyReLU(negative_slope=0.2, inplace=True) # 负斜率为 0.2 的 LeakyReLU 激活函数
            )

        #---* 构建特征提取层 *---#
        # 构建第一个基础卷积块，并加入特征提取层列表
        m_features = [_block(in_channels, out_channels)]
        # 构建剩余基础卷积块
        for i in range(depth): # range(7): 0-6
            in_channels = out_channels
            if i % 2 == 1: # 奇数层
                stride = 1  # 保持分辨率（步长为1）
                out_channels *= 2 # 增加通道数（丰富特征）
            else: # 偶数层
                stride = 2 # 降低分辨率，扩大感受野（抽象特征）
            # 构建新的卷积块，并加入特征提取层列表
            m_features.append(_block(in_channels, out_channels, stride=stride)) 

        #---* 计算最终特征图大小（用于构建分类器层的输入维度） *---#
        # 解释: 由于特征提取层中使用了步长为 2 的卷积层（偶数层），所以特征图的分辨率会减半
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # args.patch_size: 原始输入图块的尺寸
        # (depth + 1)//2: 计算层级深度（每两层下降一次）
        # 2 ** ((depth + 1)//2): 即 2 的 (depth + 1)//2 次幂，是当前层对应的缩放倍数
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        patch_size = args.patch_size // (2**((depth + 1) // 2))
        #---* 构建分类层 *---#
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024), # 全连接层：将特征图拉平成一维向量，映射到 1024 维空间
            nn.LeakyReLU(negative_slope=0.2, inplace=True), # 负斜率为 0.2 的 LeakyReLU 激活函数
            nn.Linear(1024, 1) # 输出单个判别分数（二分类任务）, 将 1024 维向量映射到一个标量
        ]

        self.features = nn.Sequential(*m_features) # 特征提取层：按顺序堆叠基础卷积块
        self.classifier = nn.Sequential(*m_classifier) # 分类器层：全连接层 + LeakyReLU + 输出层

    ''' 前向传播 '''
    # ---------------------------------------------------------------------
    # 作用: 定义判别器的前向传播过程，用于判断输入图像是否为真实图像
    # 参数: x 输入图像
    # 返回值: output 判别分数张量, 形状为 (batch_size, 1)
    # ---------------------------------------------------------------------
    def forward(self, x):
        #---* 提取特征 *---#
        features = self.features(x) # 通过特征提取层提取特征，得到特征图张量：(batch_size, channels, height, width)
        #---* 分类 *---#
        # features.size(0): 取第 0 个维度的大小，也就是 batch_size
        # -1: 表示“让 PyTorch 自动计算剩下的维度大小”。
        # .view(new_shape): 张量形状重塑（reshape）操作，不改变数据，只改变张量的“形状（shape）”
        #   - 这里是把每张特征图“拉平成一维向量”: [batch_size, channels, height, width] -> [batch_size, channels * height * width]
        output = self.classifier(features.view(features.size(0), -1)) # 通过分类器层进行分类，得到判别分数张量：(batch_size, 1)

        return output # 输出判别分数, 范围 [-inf, inf]

