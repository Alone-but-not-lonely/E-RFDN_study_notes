# E-RFDN_study_notes
仅仅是官方RFDN代码仓库中代码的注释版，是个人的学习笔记而已。

train/
├── experiment/                   ——> 存放实验结果
├── src/                          ——> **核心（在该目录下进行训练）
│   ├── data/                     ——> 数据处理模块
│   │   ├── __init__.py
│   │   ├── benchmark.py              ——> 基准数据集加载器
│   │   ├── common.py                 ——> 通用图像预处理
│   │   ├── srdata.py                 ——> 超分辨率数据集通用基类
│   │   ├── demo.py                   ——> 演示数据集加载器
│   │   ├── div2k.py                  ——> DIV2K 数据集加载器
│   │   ├── div2kjpeg.py              ——> DIV2K-JPEG 数据集加载器
│   │   ├── flickr2k.py               ——> Flickr2K 数据集加载器
│   │   └── ……                        ——> 其它数据集加载器类
│   ├── loss/                     ——> 损失值计算模块
│   │   ├── __init__.py
│   │   ├── adversarial.py            ——> 对抗性损失
│   │   ├── discriminator.py          ——> 判别器
│   │   └── vgg.py                    ——> 基于 VGG19 网络的感知损失(Perceptual Loss)模块
│   ├── model/                    ——> **模型结构定义模块
│   │   ├── __init__.py
│   │   ├── common.py                 ——> 通用组件模块
│   │   ├── rfdn.py                   ——> RFDN 网络结构定义
│   │   ├── block.py                  ——> 基础模块定义（rfdn 使用的）
│   │   └── ……                        ——> 其它模型网络结构
│   ├── __init__.py
│   ├── dataloader.py             ——> 多尺度 DataLoader 扩展
│   ├── demo.sh                   ——> 模型训练脚本
│   ├── option.py                 ——> 参数解析
│   ├── template.py               ——> 模版配置
│   ├── utility.py                ——> 通用工具函数与类
│   ├── trainer.py
│   └── main.py
├── LICENSE
└── README.md
