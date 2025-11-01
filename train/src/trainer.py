'''
# =======================================
# 训练/测试逻辑封装
# =======================================
- 文件作用: 封装训练/验证流程、数据准备、学习率调度、日志记录、模型保存等。
'''
import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm # 进度条库

'''
# =========================================================================
# 训练器 Trainer 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 保存参数、scale 列表、checkpoint、训练/测试 loader 引用，模型与损失对象。
* 参数:
    - loader: 数据集加载器对象，包含数据集、批量采样器、批量合并函数、随机种子等信息
# =========================================================================
'''
class Trainer():
    ''' 初始化训练器 '''
    # ---------------------------------------------------------------------
    # 参数: 
    #   * args: 命令行参数
    #   * loader: 数据集加载器对象，包含数据集、批量采样器、批量合并函数、随机种子等信息
    #   * my_model: 模型对象，用于前向传播、参数更新等。
    #   * my_loss: 损失函数对象，用于计算模型输出与目标之间的差异。
    #   * ckp: 检查点对象，用于保存训练过程中的模型参数、优化器状态等。
    # ---------------------------------------------------------------------
    def __init__(self, args, loader, my_model, my_loss, ckp):
        #---* 初始化参数 *---#
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss

        #---* 创建优化器（并可能封装学习率调度器等） *---#
        # 注意：这个 optimizer 是管理模型参数的（生成器）。
        self.optimizer = utility.make_optimizer(args, self.model)

        # 如果配置中要加载已有模型（args.load 非空），就从 checkpoint 加载优化器状态（用于恢复训练进度）。
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        #---* 记录上一次的 error（用于早停或模型选择） *---#
        self.error_last = 1e8


    def train(self):
        #---* 更新损失函数的内部状态（如学习率衰减） *---#
        # 每个 epoch 调用 self.loss.step()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1 # 获取当前 epoch 数
        lr = self.optimizer.get_lr() # 获取当前学习率

        #---* 打印日志 *---#
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        # 开始 loss 的记录
        self.loss.start_log()
        # 把模型切换到训练模式
        self.model.train()

        # 准备两个计时器，用于计量数据准备和模型前向的耗时
        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        # 通常多尺度训练会在 DataLoader 内随机设置 scale，但这里强制设为第一个 scale
        # 作者临时这样做以便调试或固定尺度训练
        self.loader_train.dataset.set_scale(0)
        # 遍历训练 loader
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr) # 把 lr, hr 张量转到 device（GPU/CPU），并处理精度（float/half）
            timer_data.hold() # 记录数据准备耗时
            timer_model.tic() # 开始模型前向传播计时

            self.optimizer.zero_grad() # 清空梯度
            sr = self.model(lr, 0) # 前向传播，得到模型输出 sr; 第二个参数常用于 scale index / 标识
            loss = self.loss(sr, hr) # 计算损失
            loss.backward() # 反向传播，计算梯度
            if self.args.gclip > 0:
                utils.clip_grad_value_( # 梯度裁剪，防止梯度爆炸
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step() # 更新模型参数

            timer_model.hold() # 停止模型计时器

            # 每 print_every 个 batch 打印训练信息
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch), # 当前 batch 的损失
                    timer_model.release(), # 模型耗时
                    timer_data.release())) # 数据准备耗时

            # 启动数据计时器（用于下个循环）
            timer_data.tic()

        #---* 结束 loss 日志记录 *---#
        self.loss.end_log(len(self.loader_train))
        # 更新 error_last（用于比较）
        self.error_last = self.loss.log[-1, -1]
        #---* 让优化器做学习率调度 *---#
        self.optimizer.schedule()

    '''
    # =========================================================================
    # 测试/验证函数 test
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * 流程: 评估模型在测试集上的性能（PSNR）
    * 参数:
        - loader: 数据集加载器对象，包含数据集、批量采样器、批量合并函数、随机种子等信息
    # =========================================================================
    '''
    def test(self):
        #---* 关闭梯度计算（提升推理速度） *---#
        torch.set_grad_enabled(False)

        #---* 获取当前 epoch 数 *---#
        epoch = self.optimizer.get_last_epoch()
        #---* 初始化日志矩阵（1 行，测试集样本数，scale 数） *---#
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale)) # .zeros() 创建全 0 张量
        )
        #---* 切换模型为 eval 模式 *---#
        self.model.eval()

        timer_test = utility.timer() # 初始化计时器
        if self.args.save_results: self.ckp.begin_background()
        #---* 遍历测试数据集 *---#
        for idx_data, d in enumerate(self.loader_test):
            # 遍历每个 scale
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale) # 设置当前 scale
                # 遍历当前 scale 下的测试样本并进行评估
                for lr, hr, filename in tqdm(d, ncols=80): # d 为迭代对象，ncols=80 表示进度条宽度
                    lr, hr = self.prepare(lr, hr)
                    # 前向传播，得到模型输出 sr
                    sr = self.model(lr, idx_scale)
                    # 把浮点 sr 转换回整数像素范围（0~rgb_range），以便计算 PSNR、保存图像等
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    # 计算 PSNR 并累加到当前日志位置
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    # 若需要保存 ground-truth（save_gt）, 则把 lr、hr 加入待保存列表
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    # 若配置要求保存测试结果，则把 save_list（包含 sr、可选 lr/hr）写到磁盘（checkpoint 的方法负责路径与文件名）
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                #---* 计算并累加当前 scale 下的平均 PSNR *---#
                # log[num_epochs, num_datasets, num_scales] = 每个 epoch 每个数据集每个 scale 的 PSNR 累加值
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                #---* 沿 0 维（epoch 维）求最大值 *---#
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # values, indices = tensor.max(dim)
                #   - values: 每个位置在该维度上取到的最大值; 形状: 原张量除去 dim 后的形状
                #   - indices: 最大值对应的索引位置（即在哪个 index 上取到）; 形状: 原张量除去 dim 后的形状
                # 所以, best 其实是一个包含了两个张量的元组
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                best = self.ckp.log.max(0)
                # 打印当前 scale 下的 PSNR 与最佳 PSNR
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale], # 当前 epoch PSNR
                        best[0][idx_data, idx_scale], # 历史最高 PSNR
                        best[1][idx_data, idx_scale] + 1 # 对应 epoch 编号（从1开始）
                    )
                )
        
        #---* 打印前向（测试）总耗时 *---#
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        #---* 如果需要保存结果，结束后台写文件流程 *---#
        if self.args.save_results:
            self.ckp.end_background()
        #---* 如果不是纯测试模式，调用 checkpoint 保存模型/trainer 状态，如果当前为最佳则标注 is_best=True *---#
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        #---* 打印总耗时 *---#
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    '''
    # =========================================================================
    # 准备函数 prepare
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * 作用: 把传入的所有张量按同样规则处理并返回
    # =========================================================================
    '''
    def prepare(self, *args):
        #---* 选择 device *---#
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        ''' 内部函数 '''
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half() # 转为半精度浮点数 float16
            return tensor.to(device) # 移到指定设备

        #---* 把传入的所有张量按同样规则处理并返回 *---#
        return [_prepare(a) for a in args]

    '''
    # =========================================================================
    # 终止函数 terminate
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * 作用: 判断是否需要终止训练
    # =========================================================================
    '''
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs # 当前 epoch >= 最大 epoch 时返回 True

