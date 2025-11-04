'''
# =======================================
# 基准数据集加载器 → 测试集
# =======================================
- 文件作用: 定义 Benchmark 类，专用于测试集/基准集的加载。
- 与其他文件关系: 继承自通用基类 SRData，主要改文件路径结构。
'''
import os

from data import common # 引入 common 模块，包含数据处理函数
from data import srdata # 引入 SRData 类

import numpy as np

import torch
import torch.utils.data as data

'''
# =========================================================================
# 基准数据集类 Benchmark, 继承自 SRData
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：专门用来加载基准测试集（如 Set5、Set14、B100、Urban100 等）的，这些数据集是用于评估（不打乱、不增强）的。
* 参数：
    * self: 类实例。
    * args: 配置对象，包含数据集路径、缩放因子、裁剪大小等参数。
    * name: 数据集名称(默认空字符串)。
    * train: 是否训练集(默认True)。
    * benchmark: 是否基准测试集(默认False)。
# =========================================================================
'''
class Benchmark(srdata.SRData):
    ''' 初始化基准数据集 '''
    def __init__(self, args, name='', train=True, benchmark=True):
        # 调用父类 SRData 的初始化方法
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    ''' 设置文件系统路径 '''
    # ---------------------------------------------------------------------
    # 修改文件夹层级：
    # - 基准数据集路径：dir_data/benchmark/name
    # - 高分辨率图像路径：dir_data/benchmark/name/HR
    # - 低分辨率图像路径：dir_data/benchmark/name/LR_bicubic（或 LR_bicubicL）
    # ---------------------------------------------------------------------
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png') # 暂不清楚为什么把 HR 的后缀设为空，但这样对 srdata 中的 _scan 函数扫描 HR 图像没有影响
