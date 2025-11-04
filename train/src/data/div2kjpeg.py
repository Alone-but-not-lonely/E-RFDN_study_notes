'''
# =======================================
# DIV2K-JPEG 数据集类 → 训练集
# =======================================
- 文件作用: 定义 DIV2KJPEG 类，专用于 DIV2K-JPEG 数据集的加载。
- 与其他文件关系: 继承自 DIV2K
'''
import os
from data import srdata
from data import div2k # 导入 DIV2K 类

'''
# =========================================================================
# DIV2KJPEG 数据集类, 继承自 DIV2K
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 初始化 DIV2KJPEG 数据集，并重写 _set_filesystem() 方法
* 参数:
    * self: 类实例。
    * args: 配置对象，包含数据集路径、缩放因子、裁剪大小等参数。
    * name: 数据集名称(默认'')。
    * train: 是否训练集(默认True)。
    * benchmark: 是否基准测试集(默认False)。
# =========================================================================
'''
class DIV2KJPEG(div2k.DIV2K):
    ''' 初始化 DIV2KJPEG 数据集 '''
    def __init__(self, args, name='', train=True, benchmark=False):
        # 数据集名称一般为 DIV2K-Q{q_factor}
        # 从 name 中提取 q_factor 并转换为整数
        self.q_factor = int(name.replace('DIV2K-Q', '')) 
        # 调用父类初始化方法
        super(DIV2KJPEG, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    # 重写 _set_filesystem() 方法
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(
            self.apath, 'DIV2K_Q{}'.format(self.q_factor)
        )
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.jpg') # LR 图像后缀为 .jpg

