'''
# =======================================
# 对抗性损失
# =======================================
- 文件作用: 定义对抗性损失，用于训练生成器和判别器
'''
import utility
from types import SimpleNamespace # 用于创建简单的命名空间对象，用于存储参数

from model import common
from loss import discriminator # 导入判别器类

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
# =========================================================================
# 对抗性损失 Adversarial 类, 继承自 nn.Module
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* 作用: 
    - 初始化并管理判别器 + 优化器；
    - 在每次训练中：
        - 先训练判别器（让它能区分真假图）；
        - 再训练生成器（让它能骗过判别器）；
    - 返回生成器的损失（因为判别器只是辅助模块）。
* 参数:
    - args: 包含模型超参数的命名空间对象
    - gan_type: GAN类型, 可选值为 GAN、WGAN、WGAN_GP、RGAN
# =========================================================================
'''
class Adversarial(nn.Module):
    ''' 初始化对抗性损失类 '''
    def __init__(self, args, gan_type):
        #---* 初始化 *---#
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k # 每次更新判别器的次数

        #---* 创建判别器 *---#
        self.dis = discriminator.Discriminator(args)

        #---* 设置优化器参数 *---#
        # WGAN-GP 使用特定的优化器参数
        if gan_type == 'WGAN_GP':
            # see https://arxiv.org/pdf/1704.00028.pdf pp.4
            # optim_dict(optimizer dict): 优化器参数字典
            optim_dict = {
                'optimizer': 'ADAM', # 优化器：ADAM
                'betas': (0, 0.9), # β1=0, β2=0.9
                'epsilon': 1e-8, # 小的常量 ε，防止除零错误
                'lr': 1e-5, # 较小的学习率
                'weight_decay': args.weight_decay, # 权重衰减
                'decay': args.decay, # 学习率衰减
                'gamma': args.gamma # 学习率衰减因子
            }
            # 将优化器参数字典转换为命名空间对象
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args

        #---* 创建优化器 *---#
        self.optimizer = utility.make_optimizer(optim_args, self.dis) # 基于判别器与优化器参数创建优化器

    ''' 前向传播方法 '''
    # ---------------------------------------------------------------------
    # 作用: 训练判别器和生成器
    # 参数: fake 生成器生成的假图像, real 生成器生成真实图像，是生成器还是图像？？？
    # 返回值: 生成器的损失
    # ---------------------------------------------------------------------
    def forward(self, fake, real):
        #---* 训练判别器 *---#
        # 目标: 真图判真（输出接近1），假图判假（输出接近0）。
        self.loss = 0
        fake_detach = fake.detach() # 生成器生成的假图像，但断开梯度
        # 多次更新判别器
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach) # 判别器对 fake 的判别结果
            d_real = self.dis(real) # 判别器对 real 的判别结果
            retain_graph = False # 是否保留计算图，用于计算梯度惩罚项

            #---* 根据不同GAN类型计算判别器损失 *---#
            if self.gan_type == 'GAN':
                # 普通 GAN
                loss_d = self.bce(d_real, d_fake) # 二元交叉熵损失
            elif self.gan_type.find('WGAN') >= 0:
                # Wasserstein GAN 不用 BCE，而是直接比较两个分布的差距
                #   - Wasserstein 距离: 又称 Earth-Mover 距离,衡量的是两个分布间的最小转化成本。
                loss_d = (d_fake - d_real).mean() # 假图分数减去真图分数的均值
                if self.gan_type.find('GP') >= 0:
                    # 如果是 WGAN-GP（带梯度惩罚），还会多加一个 penalty
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1) # rand_like: 与 fake 形状相同的随机数张量; epsilon: ε
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon) # 线性插值，生成混合图像 hat; mul(): 逐元素相乘
                    hat.requires_grad = True # 要求 hat 计算梯度，用于后续计算
                    d_hat = self.dis(hat) # 判别器对 hat 的判别结果
                    # 计算 hat 的梯度
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True # only_inputs: 是否只计算输入的梯度
                    )[0]
                    # 重塑梯度张量，将其转换为二维矩阵
                    gradients = gradients.view(gradients.size(0), -1) # gradient.size(): (batch_size, 1, H, W)
                    # 计算每个样本的梯度 L2 范数(长度)
                    gradient_norm = gradients.norm(2, dim=1) # dim(dimension): 维度
                    # 计算梯度惩罚项 = 10 * (||grad|| - 1)^2 的均值
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    # 累加梯度惩罚项
                    loss_d += gradient_penalty
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif self.gan_type == 'RGAN':
                # 相对GAN损失（来自ESRGAN）
                # 它不是“单独判断真假”，而是比较真假之间的差异
                better_real = d_real - d_fake.mean(dim=0, keepdim=True) # 真图分数减去假图分数的均值
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True) # 假图分数减去真图分数的均值
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True

            #---* 更新判别器权重 *---#
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph) # 反向传播计算梯度
            self.optimizer.step() # 优化

            # WGAN权重裁剪
            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1) # 裁剪权重到 [-1, 1] 范围内，防止判别器过强

        self.loss /= self.gan_k # 平均损失

        #---* 训练生成器 (Generator) *---#
        d_fake_bp = self.dis(fake) # 这次不分离，让梯度传到生成器; d_fake_bp: 判别器对 fake 的判别结果
        if self.gan_type == 'GAN':
            # 让假图被判为真
            label_real = torch.ones_like(d_fake_bp) # 对假图判别结果中接近 1 的标签
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real) # 计算生成器的损失, 即判别结果与被判为真的标签的差异
        elif self.gan_type.find('WGAN') >= 0:
            # 让判别器对假图输出高分
            loss_g = -d_fake_bp.mean() # 越不逼真的假图，实际分数越低，偏负，这里取负之后越大
        elif self.gan_type == 'RGAN':
            # 让假图比真图“更像真图”
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True) # 真图分数减去假图分数的均值
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True) # 假图分数减去真图分数的均值
            loss_g = self.bce(better_fake, better_real) # 计算生成器的损失, 即 better_fake 与 better_real 的差异

        return loss_g # 返回生成器的损失
    
    ''' 状态字典方法 '''
    # ---------------------------------------------------------------------
    # 作用: 返回判别器与优化器的状态字典
    # 参数: *args, **kwargs 任意数量的位置参数和关键字参数
    # 返回值: state_dict 包含判别器与优化器状态的字典
    # ---------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        #---* 返回判别器状态字典 *---#
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        #---* 返回优化器状态字典 *---#
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer) # 判别器状态字典 + 优化器状态字典

    ''' 二元交叉熵损失方法 '''
    # ---------------------------------------------------------------------
    # 参数: real 真实标签张量, fake 假标签张量
    # 返回值: bce_loss 二元交叉熵损失值
    # ---------------------------------------------------------------------
    def bce(self, real, fake):
        label_real = torch.ones_like(real) # 真实图片的输出应接近1
        label_fake = torch.zeros_like(fake) # 假图片的输出应接近0
        bce_real = F.binary_cross_entropy_with_logits(real, label_real) # 计算真实标签的损失
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake) # 计算假标签的损失
        bce_loss = bce_real + bce_fake # 总损失为真实标签损失与假标签损失的和
        return bce_loss

# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
