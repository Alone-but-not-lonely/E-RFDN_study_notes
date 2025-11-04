'''
# =======================================
# DIV2K 数据集定义 → 训练集
# =======================================
- 文件作用: 定义 DIV2K 类，专用于 DIV2K 数据集的加载。
- 与其他文件关系: 继承自通用基类 SRData。
'''
import os
from data import srdata # 引入 SRData 类

'''
# =========================================================================
# DIV2K 数据集类, 继承自 SRData
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 包含训练集/验证集的范围解析, 以及重写 _scan() 和 _set_filesystem() 方法
* 参数:
    * self: 类实例。
    * args: 配置对象，包含数据集路径、缩放因子、裁剪大小等参数。
    * name: 数据集名称(默认'DIV2K')。
    * train: 是否训练集(默认True)。
    * benchmark: 是否基准测试集(默认False)。
# =========================================================================
'''
class DIV2K(srdata.SRData):
    ''' 初始化 DIV2K 数据集 '''
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        #---* 解析数据范围 *---#
        # 如 '1-800/801-900' → [[1, 800], [801, 900]]
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train: # 训练时
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1: # “只”测试时
                data_range = data_range[0]
            else: # 测试时
                data_range = data_range[1]

        # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
        # data_range: 训练集/验证集的范围, 是一个列表, 每个元素都是字符串, 如 [1, 800]
        # lambda x: y() 表示对 x 应用函数 y()
        # map(function, iterable, ...) 映射函数: 对可迭代对象的每个元素应用函数, 返回新的可迭代对象
        # list() 函数: 将可迭代对象转换为列表
        # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        #---* 调用父类 SRData 的初始化方法 *---#
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    ''' 扫描数据集文件路径 '''
    # ---------------------------------------------------------------------
    # 作用: 重写 _scan() 方法, 扫描数据集文件路径, 获取所有 HR/LR 图像文件名。
    # 返回: 
    #   * names_hr (list): 所有 HR 图像文件名。
    #   * names_lr (list of list): 所有 LR 图像文件名, 每个元素是一个列表, 对应一个尺寸的 LR 图像文件名。
    # ---------------------------------------------------------------------
    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan() # 调用父类的 _scan() 方法, 获取所有 HR/LR 图像文件名
        names_hr = names_hr[self.begin - 1:self.end] # 截取 HR 图像文件名, 范围是 [begin, end)
        names_lr = [n[self.begin - 1:self.end] for n in names_lr] # 截取 LR 图像文件名, 范围是 [begin, end)

        return names_hr, names_lr

    ''' 设置文件系统路径 '''
    # ---------------------------------------------------------------------
    # 作用: 重写 _set_filesystem() 方法, 设置 DIV2K 数据集的文件系统路径。
    # 参数: 
    #   * dir_data: 数据集根目录路径。
    # ---------------------------------------------------------------------
    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data) # 调用父类的 _set_filesystem() 方法, 设置基础路径
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR') # 高分辨率图像路径
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic') # 低分辨率图像路径
        if self.input_large: self.dir_lr += 'L' # 如果输入是大尺寸, 则在 LR 路径后添加 'L'

