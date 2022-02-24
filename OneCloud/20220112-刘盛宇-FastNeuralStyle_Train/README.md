# Fast Neural Style 在 OneFlow 中的实现

本文将介绍艺术风格迁移算法的 OneFlow 实现。该算法可用于将图像的内容与另一幅图像的风格混合。

## 效果展示

项目提供了 sketch、mosaic、candy、udnie 和 rain princess 五种风格的模型文件和模型训练方法。以下是 sketch 风格迁移的示例效果。

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/sketch.jpeg" height="200px"> <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/amber.jpg" height="200px"> <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/amber-sketch-oneflow.jpg" height="200px">

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/sketch.jpeg" height="200px"> <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/cat.jpg" height="200px"> <img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/fast_neural_style/cat_sketch.jpg" height="200px">

## Fast Neural Style 介绍

### Neural Style

首先介绍 Neural Style， 在 2016 年[《A Neural Algorithm of Artistic Style》](https://arxiv.org/pdf/1508.06576.pdf) 提出的网络结构是较经典的风格迁移算法(Style Transfer)，可以使用这种方法把一张图片的风格“迁移”到另一张图片，即 Content 图像 + Style 图像 = Output 图像。

论文作者给出的网络结构如下图，输入为 Content 图像(C 图)和 Style 图像(S 图)，另外加一张随机噪声的背景图(G 图)。其中 Neural Style 的工作流程主要分为两部分：Content 图像的重构和 Style 图像的重构。

重构过程的 loss 的设计能让模型更好的学习，在原论文中，设计了两种 loss(Content Loss 和 Style Loss)，再加权求和后用于评估训练的效果。通俗讲，Content Loss 的作用主要是评估输出结果在 high-level 上是否“像”原图；Style Loss 的作用主要是评估输出结果在像素风格级别上是否“像”原图。

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220112-liushengyu-FastNeuralStyle_train/NeuralStyle.png" height="600px">

从上图可见在 C 图输入后，在 a、b、c 层仍然保留了原图像的细节，而 d、e 层开始渐渐丢失细节像素，保留高阶内容(high-level content)，那么我们可以让 G 图学习 C 图 d 层的输出，重构内容图像；同时让 G 图学习 S 图的输出，重构风格图像，这样融合得到最后输出。

这里以 VGG19 为例，具体分析训练过程，模型结构如下图。顺便一提，[flowvision](https://pypi.org/project/flowvision/) 提供了相关接口用于获取 VGG 各层的数据和相关信息，这方便我们更快地复现模型，在提供的代码中可以体现。

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220112-liushengyu-FastNeuralStyle_train/VGG16.PNG" height="400px">

具体地，它们学习过程中的损失函数计算如下：
* 内容损失函数(Content Loss)：G 图和 C 图输入分别经过 conv4_2 层输出的 MSE 即为 Content Loss。
* 风格损失函数(Style Loss)：G 图 和 S 图输入分别经过 conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 五层输出的 MSE 之和为 Style Loss。

最后如下图，计算 Content Loss 和 Style Loss 的加权和得到最后的 Total Loss。

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220112-liushengyu-FastNeuralStyle_train/loss.png" height="50px">

### Fast Neural Style

上述介绍的 Neural Style 在实际项目中很难运用，它的训练过程其实就是两张图片的融合过程，训练结束就得到最后输出图片，而这时保存的模型只包含这两张图片的信息。所以它每次都需要输入 C 图、S 图和 G 图，再最小化它们的损失，消耗资源和时间较大，又不能保存某种 style 的模型，每次融合都要重新迭代训练。

所以 Neural Style 中改变内容图像输入时，无法独立地迁移风格。而 Fast Neural Style 可以保存训练好的 style 模型，然后直接对 C 图实现相应的风格化。[《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)给出网络结构如下图：

<img src="https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220112-liushengyu-FastNeuralStyle_train/FastNeuralStyle.png" height="300px">

Fast Neural Style 模型中新增一个 Transform Net 结构，也是整个网络真正学习的部分，它是一个深度残差网络，用来将输入图像直接风格化。而后面的 Loss Network 和 Neural Style 的结构一样，但是它的参数不更新，它只用来提取特征再计算 content loss 和 style loss。

再结合整个网络的作用，可以理解 Transform Net 是专门用于存储“风格信息”的网络结构。

具体地，C 图像经过 Transform Net 直接得到 S 图，而这个 S 图再经过 Loss Network 计算 Total Loss，不断更新使损失最小，即可得到最优的 Style Transform Net 模型。

## 项目文件说明

``` text
|-- style_models/                   # oneflow 提供已训练好的模型(含五种风格)
|-- images/                         # 示例内容、样式和输出图像
|-- neural_style/                   # 训练代码
|-- checkpoints/                    # vgg16 和 vgg19 预训练模型及训练中间模型目录
|-- saved_models/                   # 训练完成模型保存目录
|-- requirements.txt                # 相关依赖库
```

## 使用方法

fork 本项目后(添加公开数据集中的 FastNeuralStyle 数据集)，点击 “运行” 启动项目。

启动后，根据 “ssh 信息” 连接云服务。

并切换至如下目录：

```
cd /workspace/
```

### 训练

在`config.py`中可以设置预训练模型(vgg16、vgg19)、数据集目录、超参数、模型保存目录等，如下：

```
# config for train.py
DATASET = "/dataset/65162261/v1/train2014/"  # 数据集路径
STYLE_IMAGE = "images/style-images/sketch.jpeg"  # 风格图像路径
SAVE_DIR = "saved_models/"  # 最终模型保存路径
STYLE_DIR = "style_log/"  # 训练过程中风格迁移结果日志路径
LOG_INTERVAL = 100  # 日志保存间隔
CHECKPOINT_INTERVAL = 200  # 训练过程中间模型保存间隔
CHECKPOINT_DIR = "checkpoints/"  # 训练过程中间模型保存路径
LOAD_CHECKPOINT_DIR = None

MODEL = "vgg16"  # 预训练模型，可选 vgg16 或 vgg19
SEED = 42  # 随机数种子
EPOCHS = 2  # epochs
BATCH = 1  # batch
LR = 0.001  # 学习率
CONTENT_WEIGHT = 30000  # CONTENT_WEIGHT 和 STYLE_WEIGHT 分别为 FastNeuralStyle 最后计算损失时的 Content Loss 和 Style Loss 的权重
STYLE_WEIGHT = 1e10
```

训练运行：

```
python ./neural_style/train.py
```

### 推理

在`config.py`中可以设置推理使用的风格模型、输入图像和输出图像路径，如下：

```
# config for infer.py
STYLE = "candy"
IMAGE_NAME = "cat"
MODEL_PATH = "style_models/{0}_oneflow/".format(STYLE)  # 风格模型路径
CONTENT_IMAGE = "images/content-images/{0}.jpg".format(IMAGE_NAME)  # 输入内容图像路径
OUTPUT_IMAGE = "images/output-images/{0}_{1}.jpg".format(IMAGE_NAME, STYLE)  # 输出图像路径
```

推理运行：

```
python ./neural_style/infer.py
```

**论文地址**

[Neural Style](https://arxiv.org/pdf/1508.06576.pdf)

[Fast Neural Style](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

**代码参考**：

[Pytorch-examples-fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style)