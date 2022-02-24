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


# config for infer.py
STYLE = "sketch"
IMAGE_NAME = "cat"
MODEL_PATH = "style_models/{0}_oneflow/".format(STYLE)  # 风格模型路径
CONTENT_IMAGE = "images/content-images/{0}.jpg".format(IMAGE_NAME)  # 输入内容图像路径
OUTPUT_IMAGE = "images/output-images/{0}_{1}.jpg".format(IMAGE_NAME, STYLE)  # 输出图像路径
