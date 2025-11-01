'''
# =======================================
# RFDN 网络结构定义
# =======================================
- 文件作用: 定义了 RFDN 模型的整体架构，将 block.py 中定义的各种“砖块”（如 RFDB 模块、卷积层）组装成一个完整的网络。
'''

import torch
import torch.nn as nn
import block as B # 导入 block.py，并简写为 B

'''
# =========================================================================
# 兼容旧的训练框架
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：这个函数主要用于兼容旧的训练框架，在测试中可以直接忽略
# =========================================================================
'''
def make_model(args, parent=False):
    model = RFDN()
    return model

'''
# =========================================================================
# RFDN 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承：继承自 PyTorch 的基础模块 nn.Module
* 作用：定义 RFDN 网络的结构。
* 参数：
    * in_nc: 输入通道数 (彩色图为3)
    * nf: 网络中间层的特征通道数 (feature maps)
    * num_modules: RFDB 模块的数量
    * out_nc: 输出通道数 (彩色图为3)
    * upscale: 放大倍数 (例如 4 倍超分)
# =========================================================================
'''
class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        #---* 1. 浅层特征提取 *---#
        # 使用一个 3x3 卷积从输入图像中提取初始特征
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        #---* 2. 核心的深层特征提取模块 (由多个 RFDB 组成) *---#
        # 连续堆叠 4 个 RFDB 模块
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)

        #---* 3. 特征融合 *---#
        # 使用一个 1x1 卷积来融合 4 个 RFDB 模块的输出
        # 输入通道数是 nf * num_modules (50 * 4 = 200)
        # 输出通道数是 nf (50)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        #---* 4. 全局残差连接前的卷积层 *---#
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        #---* 5. 图像重建 *---#
        # 使用 pixelshuffle_block 来将特征图放大，并恢复通道数为 out_nc
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)

        #---* 6. 缩放比例索引 *---#
        # 这个变量在当前代码中未使用
        self.scale_idx = 0

    ''' 定义网络的前向传播逻辑 '''
    def forward(self, input):
        #---* 1. 输入图像通过浅层特征提取模块 *---#
        out_fea = self.fea_conv(input)

        #---* 2. 特征依次通过 4 个 RFDB 模块 *---#
        # 注意这里的连接方式是串行的，一个模块的输出是下一个模块的输入
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        #---* 3. 特征融合 *---#
        # 将 4 个 RFDB 的输出在通道维度上拼接 (Concatenation)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # torch.cat([...], dim=1) 会将特征图沿通道维度 (dimension 1) 堆叠起来
        # 假设输入 B1, B2, B3, B4 的形状都是 (N, 50, H, W)
        # 拼接后的形状是 (N, 200, H, W)
        # 然后通过 1x1 卷积 self.c 将通道数降维回 50
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1)) # 通道拼接的目的为了便于卷积一次性处理所有特征

        #---* 4. 全局残差连接 *---#
        # 将融合后的深层特征与最开始的浅层特征相加
        # 这有助于稳定训练，并让网络学习残差信息
        out_lr = self.LR_conv(out_B) + out_fea

        #---* 5. 上采样 *---#
        # 将处理后的低分辨率特征图 out_lr 通过上采样模块得到最终的高分辨率图像
        # 论文架构图中的 sub_pixel 前的 Conv3 层被包含在了 upsample_block 中
        output = self.upsampler(out_lr)

        return output

    ''' 设置缩放比例 '''
    # 这个函数在当前代码中未使用
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx