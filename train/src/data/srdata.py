'''
# =======================================
# 超分辨率数据集通用基类
# =======================================
- 文件作用: 定义数据集类 SRData，负责从文件夹中加载 HR/LR 图像、做裁剪增强、转 tensor 等。
- 与其他文件关系: 被其它数据集类继承，如 SR291、Benchmark 等。
'''
import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio # 用于读取图像: image I/O
import torch
import torch.utils.data as data

'''
# =========================================================================
# 数据集类 SRData
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：定义了核心类 SRData，继承自 torch.utils.data.Dataset，封装了数据加载、裁剪、增强等逻辑。
* 参数：
    * self: 类实例。
    * args: 配置对象，包含数据集路径、缩放因子、裁剪大小等参数。
    * name: 数据集名称(默认空字符串)。
    * train: 是否训练集(默认True)。
    * benchmark: 是否基准测试集(默认False)。
# =========================================================================
'''
class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        #---* 初始化参数 *---#
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True # 是否在训练时评估模型
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR') # 根据模型名决定是否输入大图
        self.scale = args.scale # 缩放因子列表
        self.idx_scale = 0 # 当前缩放因子索引
        
        #---* 设定文件路径结构 *---#
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # args.ext: 数据集扩展名，表示数据集的存储格式
        #   - img(image): 直接使用图像文件
        #   - sep(separator): 将图像数据转换为二进制格式存储
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._set_filesystem(args.dir_data) # 设置文件系统路径(数据集根目录): self.apath = args.dir_data/数据集名称
        if args.ext.find('img') < 0: # 如果扩展名不是 img，则进行二进制转换
            path_bin = os.path.join(self.apath, 'bin') # 设定二进制文件路径: self.apath/bin
            os.makedirs(path_bin, exist_ok=True) # 创建二进制文件目录，exist_ok=True 表示如果目录已存在则不报错

        list_hr, list_lr = self._scan() # 扫描数据集目录，获取 HR/LR 图像路径列表
        if args.ext.find('img') >= 0 or benchmark: # 如果扩展名是 img 或基准测试集，则直接使用
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0: # 如果扩展名是 sep，则进行二进制转换
            # 创建二进制文件目录: path_bin/hr数据集名称
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            # 创建不同缩放因子的 LR 目录: path_bin/lr数据集名称/X{缩放因子}
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
            
            # 转换 HR/LR 图像为二进制格式并存储
            self.images_hr, self.images_lr = [], [[] for _ in self.scale] # 初始化 HR/LR 图像路径列表
            for h in list_hr:
                b = h.replace(self.apath, path_bin) # 替换 HR 图像路径为二进制路径
                b = b.replace(self.ext[0], '.pt') # 替换 HR 扩展名为 .pt
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) # 检查并加载 HR 图像为二进制格式, b 为二进制文件路径
            for i, ll in enumerate(list_lr): # 遍历不同缩放因子的 LR 图像列表，i 为缩放因子索引，ll 为该缩放因子下的 LR 图像列表
                for l in ll: # 遍历该缩放因子下的 LR 图像列表，l 为 LR 图像路径
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) # 检查并加载 LR 图像为二进制格式, b 为二进制文件路径
        if train:
            n_patches = args.batch_size * args.test_every # 总补丁数: 
            n_images = len(args.data_train) * len(self.images_hr) # 总图像数: 数据集数量 * HR 图像数量
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1) # 重复次数: 总补丁数 // 总图像数，取整后取最大值，确保至少重复一次

    
    ''' 扫描文件 '''
    # ---------------------------------------------------------------------
    # args.ext: 数据集扩展名，表示数据集的存储格式
    #   - img(image): 直接使用图像文件
    #   - sep(separator): 将图像数据转换为二进制格式存储
    # ---------------------------------------------------------------------
    def _scan(self):
        #---* 找到 HR 图像路径 *---#
        # sorted(): 对列表进行排序，返回新列表
        names_hr = sorted(
            # glob.glob(): 返回所有匹配的文件路径列表
            # self.ext[0]: HR 图像扩展名
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])) # 扫描 HR 目录下所有符合扩展名的图像文件; '*' 是通配符, 表示匹配所有文件名
        )
        #---* 构建 LR 图像路径（按缩放因子命名） *---#
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            # basename(): 返回路径中的文件名部分，不包含目录路径
            # splitext(): 分离文件名和扩展名，返回元组 (文件名, 扩展名)
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale): # scale: 缩放因子列表, si: 缩放因子索引, s: 缩放因子
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1] # self.ext[1]: LR 图像扩展名
                    )
                ))

        return names_hr, names_lr

    ''' 设定文件路径结构 '''
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name) # 数据集根目录: dir_data/name
        self.dir_hr = os.path.join(self.apath, 'HR') # 高分辨率图像目录: apath/HR
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic') # 低分辨率图像目录: apath/LR_bicubic
        if self.input_large: self.dir_lr += 'L' # 如果输入大图，低分辨率目录后缀加 L: apath/LR_bicubicL
        self.ext = ('.png', '.png') # self.ext: ('HR的扩展名', 'LR的扩展名')

    ''' 检查并加载图像 '''
    # ---------------------------------------------------------------------
    # ext: 数据集扩展名，表示数据集的存储格式
    # img: 图像路径
    # f: 二进制文件路径
    # verbose: 是否打印日志信息
    # ---------------------------------------------------------------------
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0: # 如果文件不存在或扩展名包含 reset，则重新创建二进制文件
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f: # 以二进制写入模式打开文件 f 作为 _f
                # pickle.dump(obj, file, protocol=None): 将 Python 对象序列化为字节流并写入文件，实现数据的持久化存储‌
                pickle.dump(imageio.imread(img), _f) # 读取图像 img 并将其序列化后写入文件 _f

    ''' 数据取样流程 '''
    # ---------------------------------------------------------------------
    # 读图 → 裁剪 → 通道统一 → Tensor 转换 → 返回
    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename # 返回 LR 图像、HR 图像、文件名

    ''' 数据长度 '''
    # ---------------------------------------------------------------------
    # 训练模式：返回重复次数 * 图像数量
    # 测试模式：返回图像数量
    # ---------------------------------------------------------------------
    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    ''' 获取索引 '''
    # ---------------------------------------------------------------------
    # 训练模式：返回索引模图像数量，确保索引在有效范围内
    # 测试模式：直接返回索引
    # ---------------------------------------------------------------------
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    ''' 加载文件 '''
    # ---------------------------------------------------------------------
    # 返回：索引对应的 HR 图像路径、LR 图像路径、原本的文件名
    # ---------------------------------------------------------------------
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f) # pickle.load(file): 从文件中反序列化对象，恢复原始的 Python 对象
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    ''' 随机裁剪 '''
    # ---------------------------------------------------------------------
    # 主要调用 common.get_patch() 函数进行随机裁剪
    # 返回：裁剪后的 LR 图像、HR 图像
    # ---------------------------------------------------------------------
    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train: # 训练模式：随机裁剪
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size, # 自定义 HR patch 大小
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr) # 数据增强
        else: # 测试模式：中心裁剪，HR 图像是 LR 图像的 scale 倍
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    ''' 设置缩放比例 '''
    # ---------------------------------------------------------------------
    # 输入大图时：固定选择第一个缩放比例
    # 否则：随机选择缩放比例
    # ---------------------------------------------------------------------
    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)