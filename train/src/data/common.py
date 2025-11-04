'''
# =======================================
# 通用图像预处理
# =======================================
- 文件作用: 定义常用的图像处理函数（如随机裁剪、通道转换、数据增强等）。
- 与其他文件关系: 被所有数据集类调用。
'''
import random

import numpy as np
import skimage.color as sc # 用于 RGB/Y 通道转换

import torch

'''
# =========================================================================
# 随机裁剪函数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：在低分辨率（LR）图像上随机选一个小块区域，然后在高分辨率（HR）图像上截取对应位置的区域。
* 参数：
    * args: 第一个是低分辨率图像(LR)，后面是高分辨率图像(HR)。
    * patch_size: 输出 HR patch 大小，默认 96，即 96x96。
    * scale: 放大倍数。
    * multi: 是否多尺度训练。
    * input_large: 若为 True，表示输入图像本身已经是大尺寸的（如 VDSR），意味着 LR 图像和 HR 图像的尺寸是相同的。
# =========================================================================
'''
def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    #---* 得到输入图像（低分辨率）尺寸 *---#
    # .shape: (height, width, channels); [:2] 取前两个元素（高度和宽度）
    ih, iw = args[0].shape[:2]

    #---* 计算 LR/HR patch 的宽高（HR 比 LR 大 scale 倍） *---#
    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size # HR patch size
        ip = tp // scale    # LR patch size
    else:
        tp = patch_size
        ip = patch_size

    #---* 随机选择 LR 裁剪起点 *---#
    # 确保足够的裁剪区域，相当于在除图像右上角的patch size之外的区域随机选一个点
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    #---* 计算 HR 对应的起点坐标 *---#
    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    #---* 返回 LR patch 和 HR patch 对 *---#
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :], # args[0] → lr：LR patch
        # args[1:] → 是从第2张图开始的所有图，通常只有 hr：HR patch
        # * 解包操作符：不要把这个列表整体作为一个元素放进去，而是把它“拆开”放进外层列表。
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret # ret = [lr_patch, hr_patch]

'''
# =========================================================================
# 通道转换函数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：确保图像的通道数一致（灰度图扩展为3通道或RGB转Y通道）。
* 参数：
    * args: 输入图像（低分辨率和高分辨率）。
    * n_channels: 输出通道数（1 或 3）。
# =========================================================================
'''
def set_channel(*args, n_channels=3):
    def _set_channel(img):
        #---* 若图像为单通道二维图，扩展为三维 *---#
        if img.ndim == 2:
            # img: (height, width) -> (height, width, 1)
            # axis=2 表示在通道维度上扩展
            img = np.expand_dims(img, axis=2)

        c = img.shape[2] # 取图像的通道维度
        
        if n_channels == 1 and c == 3:
            #---* 若要求单通道而输入为 RGB 图，转换为 Y 通道 *---#
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            #---* 若要求多通道而输入是灰度图，复制三次 *---#
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

'''
# =========================================================================
# Numpy 转 Tensor
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：将 numpy 格式图像转成 PyTorch tensor。
* 参数：
    * rgb_range: 输入图像的 RGB 范围（默认 255）
# =========================================================================
'''
def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) # transpose: 转置（HWC → CHW）
        tensor = torch.from_numpy(np_transpose).float() # 转成浮点 tensor
        tensor.mul_(rgb_range / 255) # 按 rgb_range 归一化

        return tensor

    return [_np2Tensor(a) for a in args]

'''
# =========================================================================
# 数据增强函数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：随机执行水平翻转、垂直翻转和 90° 旋转。
* 参数：
    * hflip(Horizontal Flip): 是否随机水平翻转（默认 True）
    * vflip(Vertical Flip): 是否随机垂直翻转
    * rot(Rotation): 是否随机旋转（默认 True）
    * rot90(Rotation 90°): 是否随机旋转 90°
# =========================================================================
'''
def augment(*args, hflip=True, vflip=True, rot=True, rot90=True):
    #---* 每种操作独立以 50% 概率执行 *---#
    # random.random() 生成 [0, 1) 之间的随机数
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot90 and random.random() < 0.5

    def _augment(img):
        #---* 依次执行变换 *---#
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # img: (height, width, channels)
        # ::-1 表示逆序（::-1 表示从后往前取元素）
        # transpose: 转置（交换 height 和 width 维度，通道维度不变）
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :] 
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

