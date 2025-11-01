'''
# =======================================
# 程序入口
# =======================================
- 文件作用: 程序入口（if __name__ == '__main__'）；
        - 负责整体流程控制：初始化随机种子、检查点（checkpoint）、创建数据/模型/损失/训练器，并循环调用训练和测试。
'''
import torch

import utility
import data
import model
import loss
from option import args # 从 option 模块导入参数 args, 常是全局命令行参数解析结果
from trainer import Trainer # 从 trainer 模块导入 Trainer 类, 用于训练模型

'''
# =========================================================================
# 程序入口
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 作为程序入口, 负责初始化、训练、测试等流程
* 【关键修改】: 把所有执行逻辑都放进这个 if 语句块里
    - 告诉程序：“只有当这个脚本是作为主程序直接运行时，才执行训练逻辑；如果是被子进程导入，就不要执行。”
    - 避免在子进程中重复初始化模型导致多线程冲突。
# =========================================================================
'''
if __name__ == '__main__':
    #---* 设置全局随机种子，确保可复现 *---#
    torch.manual_seed(args.seed)
    #---* 初始化检查点/日志工具 *---#
    # 负责模型保存、日志、结果输出等
    checkpoint = utility.checkpoint(args)

    ''' 主函数 '''
    def main():
        global model # 全局模型变量, 用于在不同函数间共享模型实例
        if args.data_test == ['video']:
            #---* 视频测试模式 *---#
            from videotester import VideoTester
            model = model.Model(args, checkpoint) # 初始化视频模型
            t = VideoTester(args, model, checkpoint) # 初始化视频测试器
            t.test() # 执行视频测试
        else:
            #---* 图像测试模式 *---#
            if checkpoint.ok:  # 检查检查点是否正常
                loader = data.Data(args) # 构建数据加载器
                _model = model.Model(args, checkpoint) # 构建网络模型（生成器或完整模型）
                _loss = loss.Loss(args, checkpoint) if not args.test_only else None # 构建损失计算模块
                t = Trainer(args, loader, _model, _loss, checkpoint) # 创建训练器
                # 循环训练：持续训练直到满足终止条件
                while not t.terminate():
                    t.train()
                    t.test()

                # 训练结束后的收尾（保存日志、状态等）
                checkpoint.done()

    #---* 调用内嵌的主函数 *---#
    main()
