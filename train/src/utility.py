'''
# =======================================
# 通用工具函数与类
# =======================================
- 文件作用: 提供一些通用的工具函数与类，用于训练、测试、可视化等。
'''
import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg') # 指定后端为 Agg，在无显示环境（服务器）上仍能保存图片（不弹窗）
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

'''
# =========================================================================
# 计时器类 timer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 便捷的计时类，支持分段计时、累加、重置。常用于测数据加载时间与模型前向时间分开统计。
# =========================================================================
'''
class timer():
    def __init__(self):
        self.acc = 0 # 累计时间 "accumulate"
        self.tic()

    ''' 开始计时 '''
    def tic(self):
        self.t0 = time.time() # 记录当前时间

    ''' 结束计时 '''
    def toc(self, restart=False):
        diff = time.time() - self.t0 # 计算当前时间与记录时间的差值
        if restart: self.t0 = time.time() # 如果需要重启计时，更新记录时间为当前时间
        return diff

    ''' 暂停计时并累加 '''
    def hold(self):
        self.acc += self.toc() # 累加当前时间与记录时间的差值

    ''' 释放累计时间 '''
    def release(self):
        ret = self.acc
        self.acc = 0

        return ret # 返回累计时间

    ''' 重置累计时间 '''
    def reset(self):
        self.acc = 0

'''
# =========================================================================
# 后台写图片的子进程目标函数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 用于在后台线程中保存图片，避免训练时阻塞主线程。
# =========================================================================
'''
def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get() # 从队列中获取文件名与张量
            if filename is None: break
            imageio.imwrite(filename, tensor.numpy()) # 保存张量为图片文件; .numpy() 转换为 numpy 数组

'''
# =========================================================================
# 检查点类 checkpoint (核心：日志与结果管理)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 用于管理训练过程中的模型、损失函数、优化器、学习率调度器等。
* 功能: 保存、加载模型、损失函数、学习率等。
* 注意: 训练时需要实例化该类，测试时需要加载已训练模型。
# =========================================================================
'''
class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        #---* 如果 --load 指定已有实验，会尝试加载 psnr_log.pt（历史 PSNR 记录），并继续训练 *---#
        if not args.load: # 如果没有指定加载目录
            if not args.save: # 如果没有指定保存目录
                args.save = now # 则使用当前时间作为保存目录名
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt')) # 加载已有的 PSNR 日志文件
                print('Continue from epoch {}...'.format(len(self.log))) # 打印继续训练的起始 epoch 数
            else:
                args.load = '' # 如果加载目录不存在，则清空加载参数，从头开始训练

        #---* --reset 会删除实验文件夹（谨慎操作） *---#
        if args.reset:
            # os.system('rm -rf ' + self.dir) # rm 是 Linux 和 macOS 系统中用来删除文件（remove）的命令。
            import shutil
            if os.path.exists(self.dir):
                shutil.rmtree(self.dir) # 跨平台兼容性
            args.load = ''

        #---* 创建目录 *---#
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
        #---* 创建日志文件 *---#
        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w' # 如果日志文件已存在，追加模式；否则写模式
        self.log_file = open(self.get_path('log.txt'), open_type) # 打开日志文件
        #---* 写入当前时间与所有参数到 config.txt *---#
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8 # 用于保存图片的后台进程数量

    ''' 获取文件路径 '''
    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    ''' 保存模型、损失函数、学习率等 '''
    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best) # 保存模型
        trainer.loss.save(self.dir) # 保存损失函数
        trainer.loss.plot_loss(self.dir, epoch) # 绘制损失函数曲线

        self.plot_psnr(epoch) # 绘制 PSNR 曲线
        trainer.optimizer.save(self.dir) # 保存优化器状态
        torch.save(self.log, self.get_path('psnr_log.pt')) # 保存 PSNR 日志文件

    ''' 添加日志 '''
    def add_log(self, log):
        self.log = torch.cat([self.log, log]) # 追加新的 log 到 self.log 中

    ''' 写入日志 '''
    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        # 刷新日志文件（关闭再重启），确保写入立即生效
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    ''' 结束训练 '''
    def done(self):
        self.log_file.close() # 关闭日志文件，确保所有写入立即生效

    ''' 绘制 PSNR 曲线 '''
    def plot_psnr(self, epoch):
        #---* 绘制 x 轴（epoch 数） *---#
        axis = np.linspace(1, epoch, epoch) # linspace 用于生成等间距的数组
        #---* 为每个测试数据集绘制 PSNR 曲线 *---#
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d) # 为每个测试数据集创建标签
            fig = plt.figure() # 创建新的 figure
            plt.title(label) # 设置 figure 标题
            #---* 为每个 scale 绘制 y 轴（PSNR 值） *---#
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(), # 绘制当前数据集、当前缩放因子的 y 轴数据（PSNR 值）
                    label='Scale {}'.format(scale)
                )
            plt.legend() # 显示图例（不同缩放因子的曲线）
            plt.xlabel('Epochs') # 设置 x 轴标签
            plt.ylabel('PSNR') # 设置 y 轴标签
            plt.grid(True) # 显示网格线
            plt.savefig(self.get_path('test_{}.pdf'.format(d))) # 保存当前 figure 为 PDF 文件
            plt.close(fig)  # 关闭当前 figure，释放内存

    ''' 后台进程-开始 '''
    def begin_background(self):
        #---* 创建一个队列 *---#
        self.queue = Queue() # 用于在后台进程之间传递数据
        #---* 创建并启动多个后台进程 *---#
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        #---* 启动所有后台进程 *---#
        for p in self.process: p.start()

    ''' 后台进程-结束 '''
    def end_background(self):
        #---* 向队列中放入 None 信号，通知所有后台进程结束 *---#
        for _ in range(self.n_processes): self.queue.put((None, None))
        #---* 等待队列清空 *---#
        while not self.queue.empty(): time.sleep(1) # sleep 1s
        #---* 等待所有后台进程结束 *---#
        for p in self.process: p.join()

    ''' 保存结果 '''
    # save_list: [SR 图像, LR 图像, HR 图像]
    def save_results(self, dataset, filename, save_list, scale):
        #---* 检查是否需要保存结果 *---#
        if self.args.save_results:
            #---* 为每个结果文件创建路径 *---#
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            #---* 为每个结果文件添加后缀 *---#
            postfix = ('SR', 'LR', 'HR') # 结果文件后缀
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range) # 归一化到 [0, 255] 范围; rgb_range 为 1 或 255, v[0] 为 SR 图像
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu() # 转换为 CPU 上的字节张量; permute 用于改变张量的维度顺序, byte 用于将张量转换为字节类型
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu)) # 放入队列, 等待后台进程保存

''' 量化图像 '''
# 作用：把浮点 tensor 归一化并量化回像素值
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range # 像素范围
    # clamp 用于将张量限制在 [0, 255] 范围内, round 用于四舍五入, div 用于除以像素范围, 恢复到原始范围
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

''' 计算 PSNR '''
# dataset: 数据集对象, 用于判断是否为基准数据集
def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    #---* 安全检查 *---#
    # 如果 hr 没有内容（1 个元素），直接返回 0
    if hr.nelement() == 1: return 0 # nelement() 用于返回张量的元素总数
    #---* 计算像素误差 *---#
    diff = (sr - hr) / rgb_range # 归一化到 [-1, 1] 范围; rgb_range 为 1 或 255
    #---* 根据情况定裁剪边界与选择通道 *---#
    if dataset and dataset.dataset.benchmark:
        #---* 对基准测试集通常只在 Y 通道上计算 PSNR *---#
        shave = scale # 裁剪的像素边界
        if diff.size(1) > 1: # 如果 diff 有多个通道（如 RGB）
            gray_coeffs = [65.738, 129.057, 25.064] # YUV 转换系数, 用于将 RGB 转换为 Y 通道
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256 # 转换为 4D 张量, 用于广播乘法; view(1, 3, 1, 1) 用于将系数扩展为 4D 张量, 与 diff 进行乘法运算
            diff = diff.mul(convert).sum(dim=1) # 对每个像素的 RGB 通道进行加权求和, 得到 Y 通道的误差
    else:
        #---* 对非基准测试集, 通常在所有通道上计算 PSNR *---#
        shave = scale + 6 # 常见约定，多加 6 是因为某些算法的边缘处理导致更多边界不可靠
    #---* 移除边界像素, 获取有效区域 *---#
    valid = diff[..., shave:-shave, shave:-shave]
    #---* 计算均方误差 (MSE) *---#
    mse = valid.pow(2).mean()
    #---* 返回 PSNR 值, 单位为 dB *---#
    return -10 * math.log10(mse)

'''
# =========================================================================
# 优化器构造函数
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 构造优化器和学习率衰减器, 返回一个 CustomOptimizer 实例
# =========================================================================
'''
def make_optimizer(args, target):
    #---* optimizer（优化器） *---#
    trainable = filter(lambda x: x.requires_grad, target.parameters()) # trainable 是模型中所有 requires_grad=True 的参数过滤器
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay} # 优化器参数
    #---* 根据优化器类型选择对应的类 *---#
    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum # SGD 优化器参数, momentum 动量因子
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas # ADAM 优化器参数, betas 是一个元组, 用于计算动量和均方梯度
        kwargs_optimizer['eps'] = args.epsilon # ADAM 优化器参数, epsilon 是一个小常数, 用于数值稳定性
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon # RMSprop 优化器参数, epsilon 是一个小常数, 用于数值稳定性

    #---* scheduler（学习率衰减器） *---#
    milestones = list(map(lambda x: int(x), args.decay.split('-'))) # milestones: 学习率衰减的 epoch 点组成的列表
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma} # 学习率衰减参数; gamma: 学习率衰减因子
    scheduler_class = lrs.MultiStepLR # 学习率衰减类, 这里使用 MultiStepLR, 即根据 milestones 列表在指定 epoch 衰减学习率

    '''
    # =========================================================================
    # 自定义优化器类
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * 作用: 封装 optimizer 和 lr scheduler, 提供统一的接口
    # =========================================================================
    '''
    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        ''' 注册学习率衰减器 '''
        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs) # 把指定 scheduler 绑定到 self.scheduler
        
        ''' 保存优化器状态 '''
        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir)) # 保存优化器状态字典到 optimizer.pt

        ''' 加载优化器状态 '''
        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            # 若 epoch > 1 则把 scheduler 弃步推进（调用 step() 多次），以恢复 learning-rate schedule
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        ''' 获取优化器状态文件路径 '''
        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt') # 返回 optimizer 文件路径

        ''' 手动更新一次学习率 '''
        # 用于 epoch 结束时推进
        def schedule(self):
            self.scheduler.step()

        ''' 获取当前学习率 '''
        def get_lr(self):
            return self.scheduler.get_lr()[0] # 取 scheduler 的第一个 lr 值

        ''' 获取当前 epoch '''
        def get_last_epoch(self):
            return self.scheduler.last_epoch
    #---* 实例化自定义优化器 *---#
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    #---* 注册学习率衰减器 *---#
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    #---* 返回自定义优化器（包含 scheduler） *---#
    return optimizer

