'''
# =======================================
# 基于 VGG19 网络的感知损失(Perceptual Loss)模块
# =======================================
- 文件作用: 通过 VGG19 提取“感知特征”，计算生成图（SR）与真实图（HR）之间的高层语义差距。
    - 感知损失比单纯的像素 MSE 更能提升视觉质量。
'''
from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

'''
# =========================================================================
# VGG 类, 继承自 nn.Module
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 计算感知损失。
    - 提取 VGG19 网络的特征层；
    - 标准化输入（减均值除方差）；
    - 冻结参数，不让它参与训练。
* 参数:
    - conv_index: VGG层索引, 可选值为 '22'（前8层）或 '54'（前35层）
    - rgb_range: 图像像素值范围, 默认为 1（[0, 1]）
# =========================================================================
'''
class VGG(nn.Module):
    ''' 初始化 VGG 类 '''
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        #---* 加载预训练的 VGG19, 并提取特征层 *---#
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features] # 将 VGG 特征层转换为列表

        #---* 选择不同的 VGG 层作为特征提取器 *---#
        if conv_index.find('22') >= 0: # 使用前8层（到第二个卷积），代表提取低层特征（纹理、边缘）
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0: # 使用前35层（到第四个卷积），代表提取高层特征（语义、形状）
            self.vgg = nn.Sequential(*modules[:35])

        #---* VGG 的标准化参数 *---#
        vgg_mean = (0.485, 0.456, 0.406) # VGG 预训练时使用的均值
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range) # VGG 预训练时使用的标准差，根据 rgb_range 缩放
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std) # 标准化层，将输入图像减均值除方差

        #---* 冻结 VGG 参数 *---#
        # 因为我们不想训练 VGG，只想用它提取特征。
        # 所以关闭梯度计算，节省显存和计算量。
        for p in self.parameters():
            p.requires_grad = False # requires_grad=False 表示冻结参数，不参与训练

    ''' 前向传播函数 '''
    # ---------------------------------------------------------------------
    # 作用: 计算感知损失。
    # 输入:
    #   - sr: 超分辨率图像（生成图）
    #   - hr: 真实图像（目标图）
    # ---------------------------------------------------------------------
    def forward(self, sr, hr):
        # 内部函数
        def _forward(x):
            x = self.sub_mean(x) # 标准化
            x = self.vgg(x) # VGG特征提取
            return x
        
        #--* 超分辨率图像的特征 *---#
        vgg_sr = _forward(sr)
        #--* 真实图像的特征 *---#
        with torch.no_grad(): # 因为只需要提取特征，不需要计算梯度，所以用 no_grad 上下文管理器
            vgg_hr = _forward(hr.detach()) # 真实图像的特征（不计算梯度）

        #--* 计算 MSE 损失 *---#
        # 用 VGG 特征图的均方误差（MSE），表示“生成图”和“真实图”在感知空间的差距。
        loss = F.mse_loss(vgg_sr, vgg_hr)

        #--* 返回感知损失 *---#
        return loss
