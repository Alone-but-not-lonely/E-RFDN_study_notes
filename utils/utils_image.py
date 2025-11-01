'''
# =======================================
# 图像处理工具脚本
# =======================================
- 文件作用: 封装所有与图像读写、格式转换、计算评估指标（PSNR）相关的函数。
    - 将这些功能集成为一个工具文件，可以大大简化主程序（如 test.py）的代码，使其更专注于核心逻辑。
'''

import os
import math
import random
import numpy as np
import torch
import cv2 # 导入 OpenCV 库，用于图像的读写和颜色空间转换
from torchvision.utils import make_grid
from datetime import datetime
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imageio

# 文件头部注释，同样说明了修改者和参考来源:
'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
https://github.com/twhui/SRGAN-pyTorch
https://github.com/xinntao/BasicSR
'''

# 定义一个包含常见图像文件扩展名的列表，用于后续判断文件是否为图像
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

# 判断一个文件名是否是图像文件
def is_image_file(filename):
    # any(...) 会检查 迭代对象 中是否有任何一个元素为 True。
    # filename.endswith(extension) 检查文件名是否以给定的扩展名结尾。
    # 这行代码会遍历 IMG_EXTENSIONS 列表，判断文件名是否以其中任何一个扩展名结尾。
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 获取当前时间的字符串表示，常用于创建带有时间戳的文件或文件夹名，以避免重名
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

# 使用 matplotlib 显示图像，主要用于调试
# cbar：是否显示颜色条，默认为 False。
# figsize：指定图的大小（宽度, 高度），默认为 None，即使用默认大小。
def imshow(x, title=None, cbar=False, figsize=None):
    # 这行代码创建一个新的 matplotlib 图（figure），并设置其大小（figsize）
    plt.figure(figsize=figsize)
    # np.squeeze(x) 用于移除数组中尺寸为1的维度，方便显示
        # 这在处理单通道图像（如灰度图）时非常有用
        # 因为 OpenCV 读取的图像通常是 HxWx1 格式，而 matplotlib 期望的是 HxW 格式
    # interpolation：指定图像显示时的插值方法，'nearest' 表示最近邻插值，保持像素的块状外观
    # cmap：指定颜色映射，'gray' 表示灰度颜色映射，适用于显示单通道图像
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


'''
# =======================================
# 获取文件夹下所有图像的路径
# =======================================
'''
# 获取指定文件夹（dataroot）下所有图像文件的完整路径列表
def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        # 调用内部函数 _get_paths_from_images 来实际执行搜索，并对结果进行排序
        paths = sorted(_get_paths_from_images(dataroot))
    return paths

# 这是一个内部辅助函数，实际执行遍历和搜索图像文件的操作
def _get_paths_from_images(path):
    # assert 用于断言，如果条件不为真，则程序会报错。这里确保传入的 path 是一个存在的目录。
        # 格式：assert 条件, 错误信息
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    # os.walk(path) 会递归地遍历指定目录下的所有子目录和文件
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            # 判断文件名是否是图像文件
            if is_image_file(fname):
                # os.path.join 用于智能地拼接路径，避免操作系统差异问题
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    # 确保在指定路径下找到了至少一张图像
    assert images, '{:s} has no valid image file'.format(path)
    return images


'''
# =======================================
# 文件夹操作
# =======================================
'''

# 创建单个文件夹
def mkdir(path):
    # 如果路径不存在
    if not os.path.exists(path):
        # 就创建它。os.makedirs 可以创建多层嵌套的目录。
        os.makedirs(path)

# 创建多个文件夹
def mkdirs(paths):
    # isinstance 判断一个对象是否是已知类型的实例
    # 这里检查 paths 是否是字符串类型
    if isinstance(paths, str):
        # 如果输入是单个字符串，就直接创建
        mkdir(paths)
    else:
        # 如果输入是一个列表或元组，就遍历并逐个创建
        for path in paths:
            mkdir(path)

# 创建文件夹，如果已存在则重命名旧的
def mkdir_and_rename(path):
    if os.path.exists(path):
        # 如果路径存在，就给旧文件夹加上时间戳重命名，起到备份作用
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    # 创建新的空文件夹
    os.makedirs(path)


'''
# =======================================
# 图像读写
# 注意: OpenCV 读入的是 BGR 格式的 numpy 数组
# =======================================
'''
# ----------------------------------------
# 读取图像为 uint8 格式的 numpy 数组 (HxWxC, RGB)
# ----------------------------------------
def imread_uint(path, n_channels=3):
    # path: 图像路径
    # n_channels: 期望的通道数 (1 for grayscale, 3 for RGB)
    if n_channels == 1:
        # cv2.imread(path, 0) 以灰度模式读取图像，得到 HxW 的二维数组
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        # np.expand_dims 增加一个维度，使其变为 HxWx1，统一格式
        img = np.expand_dims(img, axis=2)
    elif n_channels == 3:
        # cv2.IMREAD_UNCHANGED 会保留图像的原始通道，包括 alpha 通道（如果存在）
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        # 如果图像是灰度图 (只有两个维度)
        if img.ndim == 2:
            # 将灰度图转换为三通道的 RGB 图像（三个通道的值会相同）
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            # 【关键】将 OpenCV 默认的 BGR 颜色顺序转换为标准的 RGB 顺序
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

    return img

# 保存 numpy 数组为图像文件
def imsave(img, img_path):
    # 去掉可能存在的 batch 维度
    img = np.squeeze(img)
    # 如果是三通道图像
    if img.ndim == 3:
        # 【关键】将 RGB 顺序转换回 OpenCV 保存时需要的 BGR 顺序
        img = img[:, :, [2, 1, 0]]
    # 使用 OpenCV 将数组写入文件
    cv2.imwrite(img_path, img)


# ----------------------------------------
# 图像格式转换 (Numpy <-> Tensor)
# ----------------------------------------
# 将 uint8 的 Numpy 数组 (HxWxC) 转换为 4D 的 PyTorch Tensor (1xCxHxW)
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # np.ascontiguousarray 确保数组在内存中是连续的，提高处理效率
    # torch.from_numpy 将 numpy 数组转换为 tensor
    # .permute(2, 0, 1) 调整维度顺序，从 HxWxC 变为 CxHxW
    # .float() 将数据类型从 uint8 转换为 float32
    # .div(1.0) 这里除以1.0似乎没有作用，通常会是 .div(255.0) 来进行归一化，可能是代码遗留
    # .unsqueeze(0) 增加一个 batch 维度，从 CxHxW 变为 1xCxHxW
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(1.0).unsqueeze(0)

# 将 PyTorch Tensor 转换回 uint8 的 Numpy 数组 (HxWxC)
def tensor2uint(img):
    # .data 获取 tensor 的数据部分，避免梯度计算
    # .squeeze() 移除所有尺寸为1的维度（如 batch 维度）
    # .float() 确保是浮点数
    # .clamp_(0, 255) 将像素值限制在 0-255 范围内，防止溢出
    # .cpu() 将 tensor 从 GPU 移到 CPU
    # .numpy() 将 tensor 转换为 numpy 数组
    img = img.data.squeeze().float().clamp_(0, 255).cpu().numpy()
    # 如果是三维的 (CxHxW)
    if img.ndim == 3:
        # 调整维度顺序，从 CxHxW 变为 HxWxC
        img = np.transpose(img, (1, 2, 0))
    # .round() 四舍五入到最近的整数
    # np.uint8(...) 将数据类型转换为 uint8
    return np.uint8(img.round())

# ----------
# 评估指标
# ----------
# 计算两张图像之间的峰值信噪比 (PSNR)
def calculate_psnr(img1, img2, border=0):
    # img1 和 img2 都是 HxWxC 格式，像素范围 [0, 255] 的 numpy 数组
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    # 如果 border > 0，则在计算 PSNR 前，会切掉图像周围指定宽度的边界。
    # 这是因为超分图像的边界区域通常质量较差，切掉可以得到更公平的评估结果。
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    # 将数据类型转换为64位浮点数，以保证计算精度
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # 计算均方误差 (Mean Squared Error, MSE)
    mse = np.mean((img1 - img2)**2)
    # 如果 MSE 为0，表示两张图像完全相同，PSNR 为无穷大
    if mse == 0:
        return float('inf')
    # 根据 PSNR 的标准公式进行计算
    return 20 * math.log10(255.0 / math.sqrt(mse))

if __name__ == '__main__':
    # 当这个脚本被直接运行时，会执行这里的代码，可以用于简单的单元测试
    img = imread_uint('test.bmp',3)
