'''
# =======================================
# model 模块的入口
# =======================================
- 文件作用: 定义一个 Model 类，用于封装不同模型的加载、保存、推理、TTA（x8增强）、chop分块推理
'''
import os
from importlib import import_module # 动态导入（根据 args.model 名字加载对应模型文件）

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

'''
# =========================================================================
#  Model 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承: PyTorch 的 nn.Module 类
* 作用: 整个模型的“外壳”，负责加载模型结构、权重、以及推理控制逻辑
# =========================================================================
'''
class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        #---* 保存控制参数 *---#
        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble # 是否用 x8 数据增强推理
        self.chop = args.chop # 是否采用分块推理（防止显存溢出）
        self.precision = args.precision # 计算精度
        self.cpu = args.cpu
        # 选择设备
        if self.cpu:
            self.device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps') # GPU for Mac
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models # 是否保存模型

        #---* 动态导入模型文件 *---#
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device) # 加载模型结构并迁移到指定设备
        #---* 模型精度转换 *---#
        if args.precision == 'half':
            self.model.half()
        #---* 加载模型权重 *---#
        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,  # 预训练模型路径
            resume=args.resume,  # 恢复训练模型路径
            cpu=args.cpu
        )
        #---* 把模型结构打印进日志文件 *---#
        print(self.model, file=ckp.log_file)

    ''' 前向传播 '''
    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        #---* 如果模型有 set_scale 方法，就调用它 *---#
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)
        if self.training:
            #---* 训练模式 *---#
            if self.n_GPUs > 1: # 多 GPU 训练时
                return P.data_parallel(self.model, x, range(self.n_GPUs)) # 并行计算
            else: # 单 GPU 训练时
                return self.model(x)  # 直接调用模型前向传播
        else:
            #---* 推理模式 *---#
            if self.chop:
                forward_function = self.forward_chop # 分块推理
            else:
                forward_function = self.model.forward # 直接调用模型前向传播

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function) # x8 数据增强推理
            else:
                return forward_function(x) # 直接调用模型前向传播

    ''' 保存模型权重 '''
    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]  # 最新模型路径

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))  # 最佳模型路径
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))  # 第 epoch 个模型路径
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)  # 保存模型状态字典

    ''' 加载模型权重（多模式） '''
    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        #---* 选择加载设备 *---#
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {'map_location': self.device}
        
        #---* 选择加载模型路径 *---#
        if resume == -1:  # 恢复训练模型路径
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:  # 预训练模型路径
            if pre_train == 'download': # 从 URL 下载预训练模型
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train: # 从指定路径加载预训练模型
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:  # 加载指定 epoch 的模型
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )
        #---* 加载模型权重 *---#
        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    ''' 分块推理 '''
    # ---------------------------------------------------------------------
    # 核心思想: 将大图分为 4 块（重叠边界 shave），分别送进模型，最后拼回来。
    # 作用: 防显存爆炸，适合大分辨率输入
    # 参数:
    #   * shave: 每块间重叠像素宽度，默认 10
    #   * min_size: 最小分块大小，默认 160000
    # 算法流程:
    #   * 将输入图像四分；
    #   * 如果每块太大，则递归调用自己继续分块；
    #   * 对每块分别前向；
    #   * 将输出拼回原图。
    # ---------------------------------------------------------------------
    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)  # 最多 4 个 GPU
        # height, width
        h, w = args[0].size()[-2:]
        #---* 定义分块的边界 *---#
        top = slice(0, h//2 + shave)  # 上半块
        bottom = slice(h - h//2 - shave, h)  # 下半块
        left = slice(0, w//2 + shave)  # 左半块
        right = slice(w - w//2 - shave, w)  # 右半块
        #---* 对输入图像进行分块 *---#
        x_chops = [torch.cat([ # torch.cat 沿通道维度拼接
            a[..., top, left],  # 上左块
            a[..., top, right],  # 上右块
            a[..., bottom, left],  # 下左块
            a[..., bottom, right]  # 下右块
        ]) for a in args]

        y_chops = []
        #---* 如果图像太小，直接前向传播 *---#
        if h * w < 4 * min_size:
            #---* 对每块分别前向传播 *---#
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]  # 取出第 i 块到第 i + n_GPUs 块（一次性处理的量）
                y = P.data_parallel(self.model, *x, range(n_GPUs))  # 前向传播: 每个 GPU 处理 x_chop[i:(i + n_GPUs)]
                if not isinstance(y, list): y = [y]  # 如果输出不是列表，转换为列表
                if not y_chops:  # 如果是第一次前向传播，初始化 y_chops
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0)) # 每个 GPU 处理的输出拼接到 y_chop 中
        #---* 如果图像太大，递归分块推理 *---#
        else:
            #---* 对每块分别递归分块推理 *---#
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)  # 递归分块推理
                if not isinstance(y, list): y = [y] # 如果输出不是列表，转换为列表
                if not y_chops:  # 如果是第一次前向传播，初始化 y_chops
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y) # 将分块推理的输出拼接到 y_chops 中

        #---* 定义拼接边界 *---#
        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None) # 下半块的右半部分
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None) # 右半块的上半部分

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2] # 取出 batch size 和通道数
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops] # 初始化输出张量 y，大小为 (b, c, h, w)
        #---* 将分块推理的输出拼接到 y 中 *---#
        for y_chop, _y in zip(y_chops, y): # _y 是输出张量 y 的第 i 块的引用，而不是副本
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0] # 如果只有一个输出，直接返回

        return y

    ''' 8 倍推理 '''
    # ---------------------------------------------------------------------
    # 核心思想: 将输入图像水平、垂直、转置 3 次，分别送进模型，最后取平均。
    # 作用: 提高模型对旋转、翻转等变换的鲁棒性
    # 参数:
    #   * forward_function: 前向传播函数，默认 self.forward_chop
    # 算法流程:
    #   * 对输入生成 8 种变换（原图 + 水平翻转 + 垂直翻转 + 转置 + 组合）；
    #   * 分别送入模型得到 8 个输出；
    #   * 把输出反变换回原方向；
    #   * 平均 8 个结果，提升鲁棒性和 PSNR。
    # ---------------------------------------------------------------------
    def forward_x8(self, *args, forward_function=None):
        ''' 变换 '''
        # -----------------------------------------------------------------
        # 参数:
        #   * v: 输入张量，形状为 (b, c, h, w)
        #   * op: 变换操作，可选 'v'（垂直翻转）、'h'（水平翻转）、't'（转置）
        # -----------------------------------------------------------------
        def _transform(v, op):
            #---* 精度转换 *---#
            if self.precision != 'single': v = v.float()
            #---* 转换为 numpy 数组 *---#
            v2np = v.data.cpu().numpy()
            #---* 变换操作 *---#
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            #---* 转换为张量并移动到当前设备 *---#
            ret = torch.Tensor(tfnp).to(self.device)
            #---* 如果精度为 half，转换为 half 精度 *---#
            if self.precision == 'half': ret = ret.half()

            return ret

        #---* 对输入生成 8 种变换 *---#
        list_x = []
        for a in args:
            x = [a]
            # 双层循环，x 元素不断增多，生成 8 种变换：1+1+2+4
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        #---* 分别送入模型得到 8 个输出 *---#
        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y] # 如果输出不是列表，转换为列表
            if not list_y: # 如果是第一次前向传播，初始化 list_y
                list_y = [[_y] for _y in y]
            else:
                # 将分块推理的输出拼接到 list_y 中
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        #---* 变换复原 *---#
        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3: # i = 4, 5, 6, 7 : 转置
                    _list_y[i] = _transform(_list_y[i], 't') 
                if i % 4 > 1: # i = 2, 3 : 水平翻转
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1: # i = 1, 3 : 垂直翻转
                    _list_y[i] = _transform(_list_y[i], 'v')

        #---* 平均 8 个结果并返回 *---#
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # torch.cat(_y, dim=0)：将每个变换的 8 个结果沿 batch 维度（第 0 维）拼接起来
        # .mean(dim=0, keepdim=True)：对拼接后的张量在第 0 维（batch 维度）进行平均
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0] # 如果只有一个输出，直接返回

        return y