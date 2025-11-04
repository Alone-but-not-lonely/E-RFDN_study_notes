'''
# =======================================
# data 模块的入口
# =======================================
- 文件作用: 
    - 加载训练集和测试集；
    - 创建 DataLoader；
    - 支持多数据集拼接（ConcatDataset）；
    - 自动识别 benchmark（标准测试集，如 Set5、Set14）或其他数据集。
- 与其他文件关系: 动态调用其他类、组装数据流
'''
from importlib import import_module # 动态导入模块（根据数据集名字自动加载对应文件）
# from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset # PyTorch 提供的工具，用于把多个数据集合并成一个

'''
# =========================================================================
# MyConcatDataset 类, 继承自 ConcatDataset
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 自定义的数据集拼接类, 用于合并多个数据集
* 与其它类关系: 被 Data 类调用
# =========================================================================
'''
class MyConcatDataset(ConcatDataset):
    ''' 初始化自定义数据集拼接类 '''
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets) # 调用父类 ConcatDataset 的初始化方法
        self.train = datasets[0].train # 记录是否是训练集（假设所有数据集的 train 属性都相同）

    ''' 一次性设置所有子数据集的放大倍数（scale） '''
    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale) # hasattr() 函数用于检查对象是否有指定的属性, "has attribute"

'''
# =========================================================================
# Data 类 —— 核心加载器
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 核心数据加载器类, 负责加载训练集和测试集, 创建 DataLoader
# =========================================================================
'''
class Data:
    ''' 初始化数据加载器 '''
    def __init__(self, args):
        #---* 为训练集创建 DataLoader *---#
        self.loader_train = None
        if not args.test_only: # 需要训练的时候
            datasets = []
            for d in args.data_train: # 遍历训练数据集名称
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower()) # 动态导入模块（根据数据集名字自动加载对应文件）
                # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
                # getattr(对象, 属性名) 函数用于返回对象的属性值, "get attribute"
                #   - 从文件 m 中获取 module_name, 并实例化对象
                #   - 调用类 module_name 的初始化方法, 并传入参数 args, name=d
                #   - 初始化后的 module_name 对象添加到列表 datasets 中
                # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets), # 合并所有训练数据集
                batch_size=args.batch_size,
                shuffle=True, # 训练时打乱数据顺序
                pin_memory=not args.cpu, # 启用 GPU 的 pinned memory 加速
                num_workers=args.n_threads, # 多线程加载数据, 提高数据加载效率
            )

        #---* 为每个测试数据集创建 DataLoader *---#
        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1, # 批大小固定为1
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
