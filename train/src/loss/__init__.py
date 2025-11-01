'''
# =======================================
# loss 模块的入口
# =======================================
- 文件作用: 主要的损失函数管理类，负责组合多种损失函数，计算总损失。
'''
import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg') # 设置 matplotlib 使用非交互式后端
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# =========================================================================
# Loss 类, 继承自 nn.modules.loss._Loss
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 自定义的损失函数管理类, 负责组合多种损失函数, 计算总损失
# =========================================================================
'''
class Loss(nn.modules.loss._Loss):
    ''' 初始化损失函数管理类 '''
    # ---------------------------------------------------------------------
    # 参数: 
    #   * args: 命令行参数
    #   * ckp: 检查点对象，用于保存训练过程中的模型参数、优化器状态等。
    # ---------------------------------------------------------------------
    def __init__(self, args, ckp):
        # 调用父类构造函数
        super(Loss, self).__init__()
        print('Preparing loss function:')

        #---* 初始化参数 *---#
        self.n_GPUs = args.n_GPUs # 用于多 GPU 训练, 表示使用的 GPU 数量
        self.loss = [] # 存储损失配置
        self.loss_module = nn.ModuleList() # 存储损失函数模块

        #---* 解析损失函数配置 *---#
        # 格式如: "1*MSE+0.5*VGG"
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')  # 分离权重和损失类型
            
            # 根据损失类型创建对应的损失函数
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg') # 动态导入VGG损失模块
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:], # 提取VGG版本，如"22"或"54"
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial') # 动态导入对抗性损失模块
                loss_function = getattr(module, 'Adversarial')( # Adversarial: 对抗性损失函数
                    args,
                    loss_type
                )

            # 存储损失配置
            self.loss.append({
                'type': loss_type, # 损失类型，如"MSE"、"VGG22"等
                'weight': float(weight), # 损失权重
                'function': loss_function} # 损失函数模块
            )
            # 如果是GAN损失，额外添加判别器损失记录项
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}) # DIS: 判别器损失

        # 如果有多重损失，添加总损失项
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None}) # Total: 总损失

        #---* 打印损失函数配置 *---#
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor() # 用于记录训练过程中的损失值

        #---* 设备配置 *---#
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()  # 半精度训练: 减少内存占用, 但可能影响精度
        #---* 多 GPU 训练 *---#
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel( # 使用 nn.DataParallel 并行计算损失
                self.loss_module, range(args.n_GPUs)
            )
        #---* 加载预训练模型 *---#
        if args.load != '': self.load(ckp.dir, cpu=args.cpu) # ckp.dir: ckp 目录, 用于加载预训练模型

    ''' 前向传播计算总损失 '''
    def forward(self, sr, hr):
        losses = []
        #---* 计算每个损失项 *---#
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss # 有效损失 = 权重 * 损失
                losses.append(effective_loss)
                # log[-1, i]: 记录当前 batch 的第 i 个损失值
                # .item(): 提取标量值, 用于记录损失值
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                # 记录判别器损失（来自对抗性损失模块）
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        #---* 计算总损失 *---#
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item() # 记录总损失; log[-1, -1]: 记录当前 batch 的总损失值

        return loss_sum

    ''' 更新学习率 '''
    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                # .scheduler: 学习率调度器, 用于动态调整学习率
                # .step(): 每个 epoch 调用一次, 更新学习率
                l.scheduler.step()

    ''' 开始记录损失日志 '''
    def start_log(self):
        # torch.cat(): 拼接张量, 新增一行记录当前 batch 的损失值
        # torch.zeros(1, len(self.loss)): 创建一个新的张量, 形状为 (1, len(self.loss)), 元素全为 0
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss)))) # 初始化损失日志, 新增一行记录当前 batch 的损失值

    ''' 结束记录损失日志 '''
    def end_log(self, n_batches):
        self.log[-1].div_(n_batches) # 归一化损失值, 除以 batch 数量, 得到每个样本的平均损失值

    ''' 显示当前 batch 的损失值 '''
    def display_loss(self, batch):
        n_samples = batch + 1 # 当前 batch 中的样本数量
        log = []
        for l, c in zip(self.loss, self.log[-1]): # zip: 并行遍历损失配置和损失值
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples)) # c / n_samples: 平均损失值

        return ''.join(log) # 转化为字符串，格式为 "[type1: loss1] [type2: loss2] ..."

    ''' 绘制损失曲线 '''
    # ---------------------------------------------------------------------
    # 作用: 绘制训练过程中的损失曲线, 并保存为 PDF 文件。
    # 参数: 
    #   * apath: 目录路径, 用于保存损失曲线图表。
    #   * epoch: 训练总 epoch 数, 用于生成 x 轴数据。
    # ---------------------------------------------------------------------
    def plot_loss(self, apath, epoch):
        #---* 生成 x 轴数据 *---#
        # axis: 绘制 x 轴的数组
        # np.linspace(): 生成等间隔的数组, 用于绘制 x 轴, 范围为 [1, epoch], 共 epoch 个点
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss): # enumerate: 枚举损失配置, 同时获取索引 i 和损失配置 l
            label = '{} Loss'.format(l['type']) # 损失类型(标签)
            fig = plt.figure() # 创建图表
            plt.title(label) # 图表标题
            #---* 绘制损失曲线 *---#
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # plt.plot(x, y, format_string, kwargs)
            #   - x和y: 表示曲线上点的横坐标和纵坐标的数组或列表
            #   - format_string: 格式化字符串, 用于指定线的样式和颜色, 这里为 'b-' 表示蓝色实线
            #   - kwargs: 其他参数, 这里为 label=label 表示图例标签为损失类型
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            plt.plot(axis, self.log[:, i].numpy(), 'b-', label=label)
            plt.legend() # 显示图例
            plt.xlabel('Epochs') # x 轴标签
            plt.ylabel('Loss') # y 轴标签
            plt.grid(True) # 显示网格
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type']))) # 保存图表为 PDF 文件
            plt.close(fig) # 关闭图表, 释放内存

    ''' 获取损失模块 '''
    def get_loss_module(self):
        if self.n_GPUs == 1:  # 如果只有一个 GPU, 则直接返回损失模块
            return self.loss_module
        else:
            return self.loss_module.module  # 如果有多个 GPU, 则返回损失模块的模块部分

    ''' 保存损失模块 '''
    # ---------------------------------------------------------------------
    # 作用: 保存损失模块的状态字典和损失日志, 用于后续恢复训练。
    # 参数: 
    #   * apath: 目录路径, 用于保存损失模块的状态字典和损失日志。
    # ---------------------------------------------------------------------
    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    ''' 加载损失模块 '''
    # ---------------------------------------------------------------------
    # 作用: 加载损失模块的状态字典和损失日志, 用于恢复训练。
    # 参数: 
    #   * apath: 目录路径, 用于加载损失模块的状态字典和损失日志。
    #   * cpu: 是否在 CPU 上加载, 默认在 GPU 上加载。
    # ---------------------------------------------------------------------
    def load(self, apath, cpu=False):
        # kwargs: 关键字参数, 用于指定加载张量的映射位置, 格式为 {'map_location': map_location}
        # lambda 参数1, 参数2, ... : 表达式, 用于定义匿名函数,
        #   - 等价于定义了一个普通函数：def 函数名(参数1, 参数2, ...): return 表达式

        #---* 判断是否在 CPU 上加载 *---#
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage} # 相当于不改变张量的映射位置, 还在原 storage 上
        else:
            kwargs = {} # 从下文看，就是不给函数传递任该部分参数

        #---* 加载损失模块的状态字典 *---#
        # torch.load(path, **kwargs): 加载张量时, 映射到指定位置, 默认为 GPU 上
        #   - eg. torch.load(path, map_location='cpu')
        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs # ** 解包操作：把一个字典拆开，把字典里的键和值分别当作关键字参数名和值传进函数，如传入 map_location='cpu'
        ))
        #---* 加载损失日志 *---#
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        #---* 调整每个损失模块的学习率调度器 *---#
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step() # 调整学习率

