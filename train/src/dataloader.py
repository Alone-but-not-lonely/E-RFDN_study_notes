'''
# =======================================
# 多尺度 DataLoader 扩展
# =======================================
- 文件作用: 自定义的多尺度 (multi-scale) DataLoader 实现，扩展/定制 PyTorch 的 DataLoader，
        - 支持多线程 worker、按 scale 随机选择尺度、pin memory 等（用于训练时按不同 scale 采样 patch）。
'''
import threading
import random

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import BatchSampler
from torch.utils.data import _utils # 调用内部私有模块，用于自定义拓展
from torch.utils.data.dataloader import _DataLoaderIter # 调用内部私有模块，用于自定义拓展

from torch.utils.data._utils import collate # PyTorch 内部的批处理工具（负责将样本列表打包成 batch）
from torch.utils.data._utils import signal_handling # PyTorch 内部模块，用来设置 worker 的信号（如 SIGTERM、SIGCHLD）处理逻辑
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils import IS_WINDOWS
from torch.utils.data._utils.worker import ManagerWatchdog

from torch._six import queue # PyTorch 内部封装的跨版本安全 queue 模块（用于线程/进程间通信）

'''
# =========================================================================
# 多尺度循环函数 _ms_loop
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 是worker 进程将要运行的循环函数（每个 worker 启动一个进程跑这个函数）
* 参数:
    - dataset: 数据集对象，用于从索引中读取样本
    - index_queue: 索引队列，用于主进程向 worker 进程传递样本索引
    - data_queue: 数据队列，用于 worker 进程向主进程传递样本数据
    - done_event: 事件对象，用于主进程通知 worker 进程退出
    - collate_fn: 样本合并函数，用于将样本列表转换为批次张量
    - scale: 尺度列表，用于多尺度训练（训练时随机选择一个尺度）
    - seed: 随机种子，用于确保 worker 内随机过程可复现；种子(seed)是一个起始值,可以确保每次运行程序时生成的随机数序列都是相同的
    - init_fn: 初始化函数，用于每个 worker 进程启动时调用（例如设置 RNG 或其他 worker 专属初始化）
    - worker_id: worker 进程 ID，用于区分不同 worker
# =========================================================================
'''
def _ms_loop(dataset, index_queue, data_queue, done_event, collate_fn, scale, seed, init_fn, worker_id):
    #---* 初始化 worker 进程 *---#
    # collate、signal_handling、queue 等都是通过 import 引入到全局作用域
    try:
        collate._use_shared_memory = True # 启用共享内存以加速数据转移
        signal_handling._set_worker_signal_handlers() # 设置 worker 特殊信号处理（确保子进程能被正确管理）

        torch.set_num_threads(1) # 限制每个 worker 内部线程数，避免线程爆炸
        random.seed(seed) # 用给定 seed 初始化随机数，保证 worker 内随机过程可控
        torch.manual_seed(seed) # 用给定 seed 初始化 torch 随机数生成器，确保可复现

        data_queue.cancel_join_thread() # 避免在主进程 join 时阻塞

        # 若主进程提供 worker init 函数，就调用（例如设置 RNG 或其他 worker 专属初始化）
        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog() # 用于监控 manager 状态，避免孤儿进程

        # 主循环：持续从 index_queue 中取出任务，直到 done_event 被设置（主进程通知退出）
        while watchdog.is_alive():
            # 从 index_queue 中取出任务，包含 idx（用于排序）和 batch_indices（要读取的样本索引列表）
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL) # MP_STATUS_CHECK_INTERVAL 是超时时间，期间会继续轮询（用于健壮性）
            except queue.Empty: # 如果队列为空，说明主进程已关闭，继续轮询
                continue

            # 如果收到 None，表示主进程关闭，退出
            if r is None:
                assert done_event.is_set() # is_set() 方法用于检查事件是否已被设置（即主进程是否通知退出）
                return
            elif done_event.is_set():
                continue

            idx, batch_indices = r # 分别取出 idx（用于排序）和 batch_indices（要读取的样本索引列表）
            try:
                idx_scale = 0
                # 如果训练时启用了多尺度，随机选择一个尺度（每个 batch 随机选尺度）
                if len(scale) > 1 and dataset.train:
                    idx_scale = random.randrange(0, len(scale))
                    dataset.set_scale(idx_scale)

                # 从 dataset 中读取样本，根据 batch_indices 索引列表批量读取
                samples = collate_fn([dataset[i] for i in batch_indices])
                samples.append(idx_scale) # MSDataLoader 特有返回：多出一个 scale 指示
            except Exception:
                # 如果读取样本时出错，将异常信息封装后放入数据队列
                data_queue.put((idx, ExceptionWrapper(sys.exc_info()))) # ExceptionWrapper 封装异常信息；exc_info() 用于获取当前异常的信息（类型、值、栈轨迹）
            else:
                data_queue.put((idx, samples))
                del samples # 手动删除样本，释放内存（Python 会自动垃圾回收，但显式删除可以更早释放）

    #---* 处理 KeyboardInterrupt 异常（捕获 Ctrl-C 等退出事件） *---#
    except KeyboardInterrupt:
        pass  # 忽略 KeyboardInterrupt 异常，继续运行，也就是跳出循环，退出 worker 进程

'''
# =========================================================================
# 多尺度数据加载器迭代器 _MSDataLoaderIter 类
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 继承: PyTorch 的 _DataLoaderIter（内部迭代器）
* 作用: 重写部分行为以支持 _ms_loop
* 参数:
    - loader: 数据集加载器对象，包含数据集、批量采样器、批量合并函数、随机种子等信息
# =========================================================================
'''
class _MSDataLoaderIter(_DataLoaderIter): # Iterator(迭代器) for MSDataLoader

    ''' 初始化 '''
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        # pin_memory = True: 会在数据被加载到GPU之前, 先将数据从常规的 CPU 内存中复制到 CUDA 固定 (pinned) 内存中
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        #---* 初始化批量采样器迭代器 *---#
        self.sample_iter = iter(self.batch_sampler)

        #---* 初始化基础随机种子 *---#
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # .LongTensor(1): 创建一个 1 元素的 PyTorch 张量（tensor），数据类型是 long, 形状是 (1,)。
        # .random_(): 这是一个 in-place（原地） 方法，会用随机整数填充该张量。调用后，张量的唯一元素会被替换为一个随机的 long 整数。
        # .item(): 把单元素张量转换成一个 Python 标量
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        base_seed = torch.LongTensor(1).random_().item()

        #---* 若使用多个 worker，准备队列、状态变量、事件对象等 *---#
        if self.num_workers > 0:
            #---* 初始化 worker *---#
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue() # 用于存储 worker 进程返回的结果（样本索引和样本数据）
            self.batches_outstanding = 0 # 记录当前正在处理的 batch 数量（出队但未处理完成的数量）
            self.worker_pids_set = False # 用于标记 worker 进程的 PID 是否已设置
            self.shutdown = False # 用于标记是否已发送关闭信号给 worker 进程
            self.send_idx = 0  # 用于记录下一个要发送给 worker 的样本索引
            self.rcvd_idx = 0  # 用于记录下一个要接收的样本索引
            self.reorder_dict = {}  # 用于存储已接收但未排序的样本（键为样本索引，值为样本数据）
            self.done_event = multiprocessing.Event() # 用于标记 worker 进程是否已完成所有任务

            #---* 初始化基础随机种子 *---#
            base_seed = torch.LongTensor(1).random_()[0] # 相当于取张量第一个元素

            self.index_queues = []
            self.workers = []
            #---* 为每个 worker 创建一个独立的 index_queue，并启动 Process *---#
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()  # 取消队列的 join_thread，防止主进程等待 worker 完成
                # 创建 worker 进程
                w = multiprocessing.Process(
                    target=_ms_loop,  # 每个 worker 进程要执行的函数
                    args=( # 传递给 _ms_loop 函数的参数
                        self.dataset,
                        index_queue,
                        self.worker_result_queue,
                        self.done_event,
                        self.collate_fn,
                        self.scale,
                        base_seed + i, # 保证每个 worker seed 不同
                        self.worker_init_fn,
                        i
                    )
                )
                w.daemon = True # 设置为守护进程，主进程退出时会自动终止 worker 进程
                # 启动 worker 进程
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            #---* 如果开启 pin_memory，则启动一个单独线程把 worker 产出的数据移到 pinned memory *---#
            # 以便更快拷贝到 GPU
            if self.pin_memory:
                # 创建一个数据队列，用于存储从 worker 进程接收的数据
                self.data_queue = queue.Queue()
                # 创建一个线程，用于从 worker_result_queue 中接收数据并移动到 pinned memory
                pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop, # 线程要执行的函数
                    args=(
                        self.worker_result_queue,
                        self.data_queue,
                        torch.cuda.current_device(),
                        self.done_event
                    )
                )
                pin_memory_thread.daemon = True  # 设置为守护线程，主进程退出时会自动终止线程
                # 启动线程
                pin_memory_thread.start()
                # 存储线程对象，以便后续管理
                self.pin_memory_thread = pin_memory_thread
            else: # 若不开启 pin_memory，则直接使用 worker_result_queue 作为数据队列
                self.data_queue = self.worker_result_queue

           # 设置信号和子进程 PID 列表 
            _utils.signal_handling._set_worker_pids(
                id(self), # id(self): 获取 self 对象的 id, 作为 worker 进程的 group ID
                tuple(w.pid for w in self.workers) # tuple: 元组, 用于存储 worker 进程的 PID
            )
            #---* 设置 SIGCHLD 信号处理函数 *---#
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True # 标记 worker 进程的 PID 已设置

            #---* 预填充数据索引队列 *---#
            # 作用: 通过预填充足够数量的任务到队列中，可以确保所有worker进程都有工作可做，
            #   - 避免worker进程因等待任务而空闲，从而提高数据加载的吞吐量和训练效率。
            for _ in range(2 * self.num_workers):
                #---* _put_indices() *---#
                # 来源: 是从父类 _DataLoaderIter （PyTorch内部类）继承而来的
                # 作用: 从 sample_iter 获取下一批数据索引，然后将其发送到对应的worker进程的索引队列中，使worker进程能够开始加载数据
                self._put_indices() 

'''
# =========================================================================
# MSDataLoader 类, 继承自 DataLoader
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 自定义的 DataLoader 类, 用于多线程加载数据
* 角色: 外部接口
# =========================================================================
'''
class MSDataLoader(DataLoader):
    ''' 初始化 '''
    def __init__(self, cfg, *args, **kwargs):
        # 调用父类 DataLoader 的初始化方法
        super(MSDataLoader, self).__init__(
            *args, **kwargs, num_workers=cfg.n_threads
        )
        # 存储缩放因子
        self.scale = cfg.scale

    ''' 返回迭代器 '''
    # ---------------------------------------------------------------------
    # 作用: 返回自定义的 _MSDataLoaderIter，从而启用上面实现的 multi-scale worker loop
    # ---------------------------------------------------------------------
    def __iter__(self):
        return _MSDataLoaderIter(self)

