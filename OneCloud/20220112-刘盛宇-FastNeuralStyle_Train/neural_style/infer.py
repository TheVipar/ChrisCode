import cv2
import oneflow as flow

from utils import recover_image, load_image_eval
from transformer_net import TransformerNet
from config import *


def stylize(content_path, model_path, output_path):
    content_image = load_image_eval(content_path)
    with flow.no_grad():
        style_model = TransformerNet()
        state_dict = flow.load(model_path)
        style_model.load_state_dict(state_dict)
        style_model.to("cuda")
        output = style_model(flow.Tensor(content_image).clamp(0, 255).to("cuda"))
    print(output_path)
    cv2.imwrite(output_path, recover_image(output.numpy()))


if __name__ == "__main__":
    stylize(CONTENT_IMAGE, MODEL_PATH, OUTPUT_IMAGE)
