'''
# =======================================
# 视频数据集加载器 → 测试集
# =======================================
- 文件作用: 定义视频数据集类 Video，继承自 torch.utils.data.Dataset 类。
'''
import os

from data import common # 引入 common 模块

import cv2
import numpy as np
import imageio

import torch
import torch.utils.data as data

'''
# =======================================
# 视频数据集类 Video
# =======================================
- 作用: 定义数据集类 Video，负责从视频文件中加载 LR 图像、做裁剪增强、转 tensor 等。
'''
class Video(data.Dataset):
    ''' 初始化视频数据集 '''
    def __init__(self, args, name='Video', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.do_eval = False
        self.benchmark = benchmark

        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo)) # dir_demo 是视频文件路径
        # vidcap 是 VideoCapture 对象，包含视频文件的信息和状态
        self.vidcap = cv2.VideoCapture(args.dir_demo) # VideoCapture 是用于从视频文件、图像序列或摄像头等视频源捕获帧的核心类。
        self.n_frames = 0 # 当前帧索引
        # CAP_PROP_FRAME_COUNT 是视频总帧数
        # self.vidcap.get() 方法返回视频属性值，如总帧数、宽度、高度等
        self.total_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    ''' 获取视频帧 '''
    # ---------------------------------------------------------------------
    # 作用: 从视频中读取一帧 LR 图像，做裁剪增强、转 tensor 等处理。
    # 参数: idx (int) - 帧索引。
    # 返回: lr_t (torch.Tensor) - 处理后的 LR 图像 tensor。
    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        success, lr = self.vidcap.read() # read() 方法返回布尔值(是否成功读取帧)和帧图像
        if success:
            self.n_frames += 1
            lr, = common.set_channel(lr, n_channels=self.args.n_colors) # 确保 LR 图像通道数与模型一致
            lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range) # 转 tensor 并归一化到 [0, 1]

            return lr_t, -1, '{}_{:0>5}'.format(self.filename, self.n_frames) # {:0>5} 是格式化字符串，确保帧索引为 5 位，不足补 0
        else:
            self.vidcap.release() # 释放视频捕获对象
            return None

    ''' 获取数据集长度(视频总帧数) '''
    def __len__(self):
        return self.total_frames

    ''' 设置当前缩放因子索引 '''
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale