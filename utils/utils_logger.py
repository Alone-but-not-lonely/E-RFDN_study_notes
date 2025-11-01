'''
# =======================================
# 日志工具脚本
# =======================================
- 文件作用: 提供了一个标准化的日志记录功能。
    - 在进行深度学习实验时，记录实验的配置、过程中的关键指标（如损失、PSNR）、错误信息以及最终结果是至关重要的。
    - 这个工具可以方便地将这些信息同时输出到控制台（方便实时查看）和日志文件（方便后续分析和存档）。
'''

import os
import sys
import datetime
import logging

# 文件头部注释，说明代码的修改者和参考来源:
'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
https://github.com/xinntao/BasicSR
'''

# 这是一个简单的打印函数，在打印信息前加上了当前时间戳
# 在这个项目中没有被直接使用，但可以作为一个轻量级的日志替代方案
# args: 要打印的内容，支持多个参数
# kwargs: 其他可选参数，如 end='\n' 等
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# ===============================
# logger
# logger_name = None = 'base' ???
# ===============================
'''

# 这是本文件的核心函数
'''
    设置并配置一个 logger（日志记录器）。
    logger_name: 日志记录器的名字，可以自定义，如 'AIM-track'。
    log_path:    日志文件的保存路径和名称。
    '''
def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    # logging.getLogger(logger_name) 获取一个指定名称的 logger 实例。
    # 如果同名的 logger 已存在，则返回现有实例；否则创建一个新的。
    log = logging.getLogger(logger_name)

    # log.hasHandlers() 检查这个 logger 是否已经被配置过处理器 (Handler)。
    # 处理器是用来决定日志信息被发送到哪里的（如文件、控制台等）。
    # 这个判断是为了防止重复配置，导致一条日志被打印多次。
    if log.hasHandlers():
        # 如果已经配置过，就打印一条提示信息，然后直接返回
        print('LogHandlers exist!')
    else:
        # 如果是第一次配置，就打印一条设置信息
        print('LogHandlers setup!')

        # 设置该 logger 处理的日志消息的最低严重级别。
        # logging.INFO 意味着它会处理 INFO, WARNING, ERROR, CRITICAL 级别的日志，
        # 但会忽略 DEBUG 级别的日志。
        level = logging.INFO

        # 创建一个格式化器 (Formatter)，用于定义日志消息的最终输出格式。
        # '%(asctime)s.%(msecs)03d : %(message)s' 表示格式为:
        # 年-月-日 时:分:秒.毫秒 : 实际的日志消息
        # datefmt='%y-%m-%d %H:%M:%S' 定义了 %(asctime)s 部分的日期时间格式。
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        
        # -------------------------------------------------------------
        # 1. 配置日志到文件的处理器 (FileHandler)
        # -------------------------------------------------------------
        # 创建一个文件处理器，它会将日志消息写入到指定的 log_path 文件中。
        # mode='a' 表示以追加 (append) 模式打开文件，新的日志会添加到文件末尾，而不是覆盖旧的。
        fh = logging.FileHandler(log_path, mode='a')
        # 将上面定义好的格式化器应用到这个文件处理器上。
        fh.setFormatter(formatter)
        log.setLevel(level)
        # 将这个文件处理器添加到 logger 中。
        log.addHandler(fh)
        # print(len(log.handlers))

        # -------------------------------------------------------------
        # 2. 配置日志到控制台的处理器 (StreamHandler)
        # -------------------------------------------------------------
        # 创建一个流处理器，它会将日志消息发送到标准输出流 (sys.stdout)，也就是你的终端/控制台。
        sh = logging.StreamHandler()
        # 同样，将格式化器应用到这个控制台处理器上。
        sh.setFormatter(formatter)
        # 将这个控制台处理器也添加到 logger 中。
        log.addHandler(sh)


'''
# ===============================
# (已弃用/未使用) 打印到文件和标准输出的类
# ===============================
'''
# 这是一个用类实现的同时打印到文件和控制台的方法。
# 在这个项目中，上面的 logger_info 函数已经实现了这个功能，所以这个类没有被使用。
# 但它展示了一种基础的实现思路：重定向 sys.stdout。
class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout # 保存原始的控制台输出流
        self.log = open(log_path, 'a') # 打开一个日志文件

    def write(self, message):
        self.terminal.write(message) # 往原始控制台写
        self.log.write(message) # 往日志文件写

    def flush(self):
        # flush 方法是文件对象的一个标准方法，这里保持接口一致性
        pass
