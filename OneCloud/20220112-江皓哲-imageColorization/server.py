import os
import numpy as np
import torch
from flask import Flask, request, render_template
from skimage import io
import base64
from PIL import Image
from io import BytesIO
from colorizers.colorizer import Colorizer

app = Flask(__name__)

cur_path = os.path.abspath(os.getcwd())
model_path = cur_path + r"/model/siggraph17-df00044c.pth"

base_dir = os.environ.get('BASE_DIR', '')
print("base_dir : " + base_dir)
# load colorizer
model = Colorizer(model_path, use_gpu=True if torch.cuda.is_available() else False)


@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def hello_world():
    response = render_template('home.html', base_dir=base_dir)
    return response


@app.route(f'{base_dir}/colorization', methods=["POST"])
def up_photo():
    # 如果没有上传图片，则错误
    if request.files["photo"].filename == '':
        return render_template("error.html"), 500

    # 获取用户上传的图片,转成numpy数组
    img_file = request.files["photo"] # FileStorage类型
    img_file = Image.open(img_file).convert('RGB')# FileStorage转Image
    img_nparr = np.array(img_file)

    # 渲染图片,得到numpy数组,shape=(h,w,3)
    img_colorized_nparr = model.colorize(img_nparr=img_nparr)
    # 将渲染所得图片转成base64f字符串
    img_colorized_Image = Image.fromarray(np.uint8(img_colorized_nparr*255))
    output_buffer = BytesIO()
    img_colorized_Image.save(output_buffer,format="png")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode()

    # 将base64字符串传给前端
    return render_template('show.html', base64_str=base64_str, base_dir=base_dir)


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run("0.0.0.0",debug=True)
