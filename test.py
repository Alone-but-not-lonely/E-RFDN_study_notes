'''
# =======================================
# 测试脚本 - 程序入口
# =======================================
- 文件作用: 用来测试训练好的 RFDN 模型的。
    - 负责加载模型、读取低分辨率（LR）测试图像、进行超分辨率（SR）处理、保存结果图像，并计算最终结果的 PSNR（峰值信噪比）。
- 调用关系：会调用 RFDN.py 中定义的网络结构，而 RFDN.py 又会使用 block.py 中定义的基础模块。utils_*.py 文件则提供了辅助功能。
'''

import os.path
import logging
import time
from collections import OrderedDict
import torch

#---* 导入项目内的工具脚本 *---#
from utils import utils_logger  # 日志记录工具
from utils import utils_image as util    # 图像处理工具
from RFDN import RFDN    # 导入核心的 RFDN 模型定义


'''
# =========================================================================
# 测试主函数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用：这个函数主要用于测试训练好的 RFDN 模型，加载模型、读取低分辨率（LR）测试图像、进行超分辨率（SR）处理、保存结果图像，并计算最终结果的 PSNR（峰值信噪比）。
# =========================================================================
'''
def main():
    #---* 1. 设置日志记录器 *---#
    # 初始化一个名为“AIM-track”的日志记录器，日志会输出到控制台和‘AIM-track.log’文件
    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')  # 获取该记录器实例

    #---* 2. 基本设置 *---#
    testsets = 'DIV2K'  # 测试数据集的根目录名
    testset_L = 'DIV2K_valid_LR_bicubic'  # 低分辨率（lR）图像所在的文件夹名
    # testset_L = 'DIV2K_test_LR_bicubic'  # 也可以切换到其他测试集

    torch.cuda.current_device() # 获取当前 CUDA 设备
    torch.cuda.empty_cache() # 清空 CUDA 缓存，释放显存
    # torch.backends.cudnn.benchmark = True # cuDNN 的自动优化，对于输入尺寸固定的网络可以加速，测试时可关闭
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 自动选择 GPU 或 CPU

    #---* 3. 加载模型 *---#
    model_path = os.path.join('trained_model', 'RFDN_AIM.pth')  # 指定预训练模型的路径
    model = RFDN()  #实例化 RFDN 网络，此时网络结构已定义，但权重是随机初始化的

    #---* 加载预训练的权重到模型中 *---#
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # torch.load(model_path) 会读取 .pth 文件
    # model.load_state_dict(...) 会将读取的权重加载到 model 实例中
    # strict=True 表示模型结构必须与权重文件完全匹配
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()    # 将模型设置为评估（evaluation）模式。这会关闭 Dropout 和 BatchNorm 等训练时特有的层

    #---* 冻结模型参数 *---#
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # model.named_parameters() 返回模型中所有可学习参数的名称和对应张量
        # k: 模型的参数名
        # v: 对应参数的张量（权重或偏置）
    # 遍历模型的所有参数，并设置 requires_grad = False
    # 这样做可以告诉 PyTorch 在前向传播时不需要计算梯度，从而节省显存和计算资源
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)    # 将模型移动到之前选择的设备（GPU 或 CPU）上

    #---* 4. 计算并打印模型的参数量 *---#
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # lambda ...：定义一个匿名函数，接收一个张量 x 作为输入，调用张量的 numel() 方法计算其包含的元素总数（即单个参数张量的大小）
    # model.parameters() 返回模型中所有可学习参数
    # map(..., ...)：对模型的每个参数张量应用上述匿名函数，生成一个包含所有参数张量元素数量的映射对象
    # sum(...) 将所有参数的数量加起来，得到总参数量
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters)) # .format 用于格式化字符串，将 number_parameters 插入到 {} 中

    #---* 5. 读取图像并进行超分 *---#
    L_folder = os.path.join(testsets, testset_L, 'X4')  # 构造低分辨率图像的完整路径
    E_folder = os.path.join(testsets, testset_L+'_results')  # 构造用于保存结果图像的路径
    util.mkdir(E_folder)    # 如果结果文件夹不存在，则创建它

    #---* 记录每个图像的 PSNR 和运行时间 *---#
    test_results = OrderedDict()    # 使用有序字典来存储结果
    test_results['runtime'] = []    # 创建一个列表来存储每张图片的推理时间

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0 # 图像计数器

    # 创建 CUDA 事件，用于精准测量 GPU 执行时间
    # enable_timing=True 开启时间测量功能
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img_SR = [] # 用于存储所有超分后的图像（numpy array）

    #---* 遍历 L_folder 文件夹下所有图像 *---#
    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) 读取并预处理低分辨率图像(img_L)
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img)) # 从路径中分离出文件名和扩展名
        # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
        # 格式化字符串 '{:->4d}--> {:>10s}'
        # d：表示十进制整数；s：表示字符串
        # 4、10：表示占4、10个字符宽度
        # -> ：对齐方式和填充字符， > 表示右对齐， - 是填充字符
            # 当 idx 不足4位时，左侧会用 - 字符填充
            # 当 img_name+ext 不足10位时，左侧会用空格填充
        # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext)) # 打印当前处理的图像信息

        img_L = util.imread_uint(img, n_channels=3) # 读取图像为 uint8 格式的 numpy 数组 (H, W, C)
        img_L = util.uint2tensor4(img_L) # 将 numpy 数组转换为 4D 的 PyTorch Tensor (1, C, H, W)
        img_L = img_L.to(device) # 将输入 Tensor 移动到 GPU

        # 记录 GPU 推理时间
        start.record() # 1. CPU向GPU指令流发送 "开始" 标记
        img_E = model(img_L) # 2. CPU向GPU发送 "执行模型" 的指令 *** 核心步骤: 将图像输入模型，进行前向传播，得到输出结果 ***
        end.record() # 3. CPU向GPU指令流发送 "结束" 标记
        torch.cuda.synchronize() # 4. CPU在这里暂停，等待GPU完成以上所有指令
        test_results['runtime'].append(start.elapsed_time(end)) # 计算并存储耗时（毫秒）

        # --------------------------------
        # (2) 后处理超分图像(img_E)
        # --------------------------------
        img_E = util.tensor2uint(img_E) # 将输出的 Tensor 转换回 uint8 格式的 numpy 数组
        img_SR.append(img_E) # 将处理完的图像存入列表，用于后续计算 PSNR

        # --------------------------------
        # (3) 保存结果图像 save results
        # --------------------------------
        util.imsave(img_E, os.path.join(E_folder, img_name+ext))

    #---* 计算并打印平均运行时间 *---#
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0 # 毫秒转秒
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

    #---* 6. 计算 PSNR *---#
    psnr = [] # 存储每张图的 PSNR 值
    idx = 0
    H_folder = 'DIV2K/DIV2K_valid_HR' # 高分辨率（Ground-Truth）图像的文件夹路径
    #---* 遍历真实的高分辨率图像 *---#
    for img in util.get_image_paths(H_folder):
        img_H = util.imread_uint(img, n_channels=3) # 读取 HR 图像
        # 计算我们生成的 SR 图像与真实的 HR 图像之间的 PSNR
        psnr.append(util.calculate_psnr(img_SR[idx], img_H))
        idx += 1
    #---* 计算并打印所有图像的平均 PSNR *---#
    logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))

if __name__ == '__main__':

    main() # 程序主入口