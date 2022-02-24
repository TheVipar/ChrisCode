# Image Colorization 图片上色

本项目使用 CNN 模型来实现黑白照片的上色。用户可以上传一张黑白图片（如风景、人物等现实生活场景），我们将返回出一张彩色的图片。

## 1. 原理介绍
本项目所用模型来自于论文  [_Colorful Image Colorization_](https://arxiv.org/pdf/1603.08511.pdf) , 由 Richard Zhang 等人于 2016 年在计算机图形学顶会 ECCV 上发表。文章提出了一种灰度图的自动着色方法，基于输入的黑白图像，预测出一种彩色方案。

### 1.1. 图片上色任务简介
图片上色任务旨在为黑白图像预测出一种`视觉上合理`的颜色方案，本质上是一个多模的预测问题。例如，一颗苹果可以是青色、红色或黄色，只要它不是紫色、黑色等奇怪的颜色，就符合人的认知。
因此，图片上色任务的目标不是恢复灰度图的真实颜色，而是`利用灰度图中物体的纹理、语义等信息作为线索，预测出所有可能的颜色，并从中选出最合理的上色方案。`

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/%E6%95%88%E6%9E%9C.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">图片出自论文</div>
</center>

项目采用深度学习图像领域使用最多的 ` ImageNet 数据集` 以训练、评估模型，ImageNet 包含超过 130 万张图片，均来源于现实场景，一些常见生活场景如天空、草地、沙漠等经常作为背景出现在图片中。

数据集很大程度决定了模型的性能，因此本项目的预测图片应尽可能` 来源于现实 `。

对于虚拟图片、平面图片的上色（如黑白漫画、3D图片的转彩任务），应该另当别论。详见 [_黑白漫画转彩色漫画_]()

### 1.2. 模型与算法
#### 1.2.1. Lab颜色空间
Lab 的全称是CIELAB，有时候也写成 CIE L * a * b *，其中 CIE 的全称是 International Commission on Illumination（国际照明委员会），是一个关于光照、颜色等的国际权威组织。

在 Lab 空间中，一种颜色用 L、a、b 三个数字表示。各分量的含义如下：
- L代表亮度
- a代表从绿色到红色的分量
- b代表从蓝色到黄色的分量

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/Lab.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">Lab颜色空间</div>
</center>
图 (a) 展示了 LAB 的色域（gamut），当 L=50 时，共有 313 种 ab 的取值。

```
Lab不仅包含了RGB，CMYK的所有色域，还能表现它们不能表现的色彩。人的肉眼能感知的色彩，都能通过Lab模型表现出来。
RGB模型在蓝色到绿色之间的过渡色彩过多，而在绿色到红色之间又缺少黄色和其他色彩。Lab色彩模型则弥补了RGB色彩模型色彩分布不均的不足。
```

```angular2html
如果想在数字图形的处理中保留尽量宽阔的色域和丰富的色彩，最好选择 Lab 颜色空间。
```

#### 1.2.2. 卷积神经网络CNN
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/model.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">模型结构</div>
</center>

上图展示了项目采用的 CNN 模型，由 8 个 conv 模块组成。每个 conv 模块包含 2 或 3 个重复的`conv`层 和 `ReLU` 层 ，以及 1 个 `BatchNorm` 层。
模型中没有池化层，而是通过空间下采样或上采样来实现分辨率的变化。

输入图片的 L 通道，使用一个 CNN 预测对应的 ab 通道取值，然后将 L、ab 通道拼接成最后一张 Lab 图像。

#### 1.2.3. 损失函数
对于图片上色模型，一种传统的计算损失函数的方法是采用`Euclid距离`，然而这种损失函数对于着色问题的固有歧义和多模特性不是很鲁棒，得到的最优解从直观上来看，有些灰色、不饱和。

因此，作者针对图片上色任务提出了一种`多分类交叉熵`损失函数用于训练模型，如下图所示。 其中 Z 代表真实颜色，Z hat 是预测结果。v(·)用来平衡那些出现频率较少的类的权重，即对于一些现实中并不常见的颜色，赋予较小的权重。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/loss.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">多分类交叉熵</div>
</center>
用作者的话描述：修改过后使得上色效果更有活力、更贴近现实。

### 1.3. 模型效果
在评估模型上色效果时，作者参考了图灵测试的思想：将合成的彩色照片和原照片同时展示给真人，要求他们从中分辨出人工合成的照片。结果显示，有32%的参与者把原图片当成了合成图片。
## 2. 项目文件说明

``` text
├───model                   # 预训练模型
├───templates               # 前端页面
├───colorizers
│   │   base_color.py       # 颜色类
│   │   colorizer.py        # 上色类
│   │   siggraph17.py       # 模型源码
│   └───util.py             # 工具类
├───requirements.txt        # Python 依赖包
└───server.py               # 后端启动程序
```

## 3. 系统架构

项目使用 Flask 来搭建 WEB 应用，连通用户前端，后端服务，和算法推理。

`server.py` 采用 `Flask` 在服务器端启动过一个 Web 服务。这个 FLASK 服务器前接用户来自浏览器的请求，后接用于推理图片结果的 Colorizer 对象 。整体架构如下所示：

```text
┌───────┐           ┌───────┐        ┌─────────┐
│       │    AJAX   │       │        │         │
│       ├───────────►       ├────────►         │
│ USER  │           │ FLASK │        │Colorizer│
│       ◄───────────┤       ◄────────┤         │
│       │    JSON   │       │        │         │
└───────┘           └───────┘        └─────────┘
```

### 3.1. 前端

前端实现代码在```templates/home.html```中，它提供了按钮```<上传图片>```用于用户拍照/上传图片：

```
<div class="upload">
    <input type="file" id="file" name="photo" multiple @change="upload">
</div>
```

当用户拍照/上传图片后，通过```提交form表单```将图片通过 POST 请求发送给后端。
```
<form method="POST" action="{{ base_dir+"/colorization"}}" enctype="multipart/form-data">
    <div class="colorize">
        <input type="submit" value="提交图片" class="button-new" style="margin-top:15px;"/>
    </div>
</form>
```
后端接收用户上传图片，传入调用模型，产生推理结果(即上色后的图像)，将其更新到前端页面上。
```
<img src="{{ url_for('static', filename=filePath_colorized) }}" />
```

### 3.2. 后端

在 server.py 中，注册了路由```/v1/index```：

```
@app.route(f"{base_dir}/v1/index", methods=["GET"])
#...
```

当接受 GET 请求时，返回主页面（home.html）。当接受 POST 请求时，则做两件事情：

- 保存图片（方便之后被前端引用显示）；
- 调用模型对图片进行预测并返回预测结果；

保存图片相关代码：

```
filename = generate_filenames(image_file.filename)
filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
image_file.save(filePath)
```

对图片进行预测相关代码：

```
    def colorize(self, img_path):
        self.model.eval()
        img = np.asarray(Image.open(img_path).convert('RGB'))  # ndarray
        ...
        L_orig, L_256 = rgb2L_tensor(img_orig=img, HW=(256, 256), resample=3)
        ...
        img_colorized = Lab2rgb_numpy(L_orig=L_orig, out_ab=self.model(L_256).cpu())
        assert img_colorized.shape == (h, w, 3)
        ...
        return img_colorized
```

### 3.3. 推理

后端推理使用PyTorch，通过读取已经训练好的模型文件```model/siggraph17.pth```初始化模型。

```
def __init__(self, model_path, use_gpu=False):
    self.model = SIGGRAPHGenerator().cuda() if use_gpu else SIGGRAPHGenerator()
    self.model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    self.use_gpu=use_gpu
```

输入待识别的图片，模型进行推理并输出识别结果：

```
@app.route(f'{base_dir}/colorization', methods=["POST"])
def up_photo():
    ...
    img_colorized = model.colorize(filePath_raw)
    filePath_colorized = os.path.join(app.config["COLORIZED_FOLDER"], "colorized_" + filename)
    io.imsave(filePath_colorized, img_colorized)
    return render_template('show.html', filePath_colorized="colorized_" + filename, base_dir=base_dir)
```

## 4. 项目部署和使用

### 4.1. 项目部署
步骤1~2：Fork项目时，编辑项目名称和项目描述，此处可以自定义。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/1.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">步骤1~2</div>
</center>
步骤3~4：部署项目。选中文件列表中的所有文件，见下图。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/2.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">步骤3~4</div>
</center>

步骤5：编辑部署的详细信息。此处自定义项目名称`Image Colorization`、分类`图像分类`、版本号`v0.0.1`和描述信息。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/3.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">步骤5</div>
</center>

步骤6：选择公开环境`oneflow-master+torch-1.9.1-cu11.1-cudnn8`，输入启动命令行`cd /worksoace && bash ./run.sh` ，端口号输入`5000`。点击确定
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/4.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">步骤6</div>
</center>
步骤7：点击测试，开始运行。


### 4.2. 用户界面
前端页面包括 home.html 和 show.html ，分别作为主页和效果展示页。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/upload.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">主页</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/OneCloud_img/ColNet/show.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;">效果展示页</div>
</center>
