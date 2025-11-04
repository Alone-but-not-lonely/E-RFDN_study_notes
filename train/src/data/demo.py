'''
# =======================================
# 演示数据集加载器 → 测试集
# =======================================
- 文件作用: 定义 Demo 类，用于加载演示数据集。
'''
import os

from data import common # 引入 common 模块

import numpy as np
import imageio

import torch
import torch.utils.data as data

'''
# =========================================================================
# 演示数据集类 Demo, 继承自 data.Dataset
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：加载演示数据集，用于测试模型在真实场景下的表现。
# =========================================================================
'''
class Demo(data.Dataset):
    ''' 初始化演示数据集 '''
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f)) # 整合演示数据集图片路径
        self.filelist.sort() # 对文件列表进行排序，确保按文件名顺序加载

    ''' 获取演示数据集的一个样本 '''
    # ---------------------------------------------------------------------
    # 作用：获取索引为 idx 的演示样本
    # 返回值：
    #     * lr_t: 低分辨率图像张量
    #     * -1: 占位符，无标签
    #     * filename: 样本文件名
    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        lr = imageio.imread(self.filelist[idx]) # 读取低分辨率图像
        lr, = common.set_channel(lr, n_channels=self.args.n_colors) # 设置通道数
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range) # 转换为张量

        return lr_t, -1, filename

    ''' 获取演示数据集的样本数量 '''
    def __len__(self):
        return len(self.filelist)

    ''' 设置演示数据集的缩放比例 '''
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

