'''
# =======================================
# 通用工具函数与类
# =======================================
- 文件作用: 负责命令行参数解析（argparse），并做少量后处理，最后调用 template.set_template(args) 应用预设模板。
'''
import argparse # 命令行参数解析模块
import template # 模板模块

''' 创建一个 ArgumentParser 实例 '''
# 用来解析命令行参数
parser = argparse.ArgumentParser(description='EDSR and MDSR')

''' 定义可在命令行传入的参数 '''
#--* add_argument 参数说明 *---#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# type: 命令行参数的类型，默认是 str（字符串）
# default: 参数的默认值
# help: 参数的描述信息，用于在命令行中显示帮助信息
# action='store_true' 的参数在命令行出现即为 True，否则为 False
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#---* 通用参数 *---#
parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode') # 调试模式
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py') # 模版参数

#---* Hardware specifications 硬件相关参数 *---#
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading') # 线程数
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only') # 是否仅使用 CPU
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs') # GPU 数量
parser.add_argument('--seed', type=int, default=1,
                    help='random seed') # 随机种子

#---* Data specifications 数据相关参数 *---#
parser.add_argument('--dir_data', type=str, default='../../../dataset',
                    help='dataset directory') # 数据集根目录
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory') # 演示图像目录
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name') # 训练数据集名称
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name') # 测试数据集名称
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range') # 数据范围
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension') # 数据集文件扩展名
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale') # 超分辨率缩放比例
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size') # 输出补丁大小
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB') # RGB 最大值
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use') # 颜色通道数
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward') # 是否采用分块推理，防止显存溢出
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation') # 不使用数据增强

#---* Model specifications 模型相关参数 *---#
parser.add_argument('--model', default='EDSR',
                    help='model name') # 模型名称

parser.add_argument('--act', type=str, default='relu',
                    help='activation function') # 激活函数
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory') # 预训练模型目录
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory') # 预训练模型目录扩展
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks') # 残差块数量
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps') # 特征图数量
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling') # 残差缩放因子
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input') # 从输入中减去像素均值
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution') # 使用膨胀卷积
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)') # 测试时的 FP(浮点数) 精度

#---* Option for Residual dense network (RDN) 残差密集网络相关参数 *---#
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)') # 默认滤波器数量
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)') # 默认卷积核大小
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)') # RDN 参数配置

#---* Option for Residual channel attention network (RCAN) 残差通道注意力网络相关参数 *---#
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups') # 残差组数量
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction') # 特征图减少数量

#---* Training specifications 训练相关参数 *---#
parser.add_argument('--reset', action='store_true',
                    help='reset the training') # 是否重置训练
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches') # 每多少个批次进行测试
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train') # 训练轮数
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training') # 训练批次大小
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks') # 将批次拆分成较小的块
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test') # 是否使用自集成方法进行测试
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model') # 是否仅测试模型
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss') # 对抗损失的 k 值

#---* Optimization specifications 优化相关参数 *---#
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate') # 学习率
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type') # 学习率衰减类型
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay') # 步衰减学习率衰减因子
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)') # 优化器
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum') # SGD 动量
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta') # ADAM 测试参数 beta
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability') # ADAM 数值稳定性 epsilon
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay') # 权重衰减
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)') # 梯度裁剪阈值

#---* Loss specifications 损失函数相关参数 *---#
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration') # 损失函数配置
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error') # 跳过误差过大的批次

#---* Log specifications 日志相关参数 *---#
parser.add_argument('--save', type=str, default='test',
                    help='file name to save') # 保存文件名
parser.add_argument('--load', type=str, default='',
                    help='file name to load') # 要加载文件的名称
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint') # 从特定检查点恢复
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models') # 保存所有中间模型
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status') # 训练状态日志记录的批次间隔
parser.add_argument('--save_results', action='store_true',
                    help='save output results') # 保存输出结果
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together') # 同时保存低分辨率和高分辨率图像

''' 解析命令行参数 '''
args = parser.parse_args() # 解析命令行生成 args（一个 Namespace）
''' 应用命令行参数的预设配置 '''
template.set_template(args) # 把 args 传给 template.set_template 以应用某些预设配置

''' 参数处理 '''
args.scale = list(map(lambda x: int(x), args.scale.split('+'))) # 把 args.scale 从字符串变为整型列表
args.data_train = args.data_train.split('+') # 把 args.data_train 从字符串变为字符串列表
args.data_test = args.data_test.split('+') # 把 args.data_test 从字符串变为字符串列表

# 如果 epoch 设为 0，把它当作“无限训练”（使用非常大的数表示）
if args.epochs == 0:
    args.epochs = 1e8

#---* 兼容性处理 *---#
# 把命令行中某些值误传成字符串 'True'/'False' 时修正为布尔型
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False