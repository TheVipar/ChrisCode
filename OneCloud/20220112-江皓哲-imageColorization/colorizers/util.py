import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color


def rgb2L_tensor(img_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    rgb_256 = np.asarray(Image.fromarray(img_orig).resize((HW[1], HW[0]), resample=resample))

    L_orig = color.rgb2lab(img_orig)[:, :, 0]
    L_256 = color.rgb2lab(rgb_256)[:, :, 0]

    L_orig = torch.Tensor(L_orig)[None, None, :, :]  # (w,h)->(1,1,h,w)
    L_256 = torch.Tensor(L_256)[None, None, :, :]  # (256,256)->(1,1,256,256)

    return L_orig, L_256


def Lab2rgb_numpy(L_orig, out_ab, mode='bilinear'):
    # L_orig     	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = L_orig.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab
    # out_Lab_orig.size()= (1,3,H_orig,W_orig)
    out_Lab_orig = torch.cat((L_orig, out_ab_orig), dim=1)
    # 返回值维度是(H_orig,W_orig,3)
    return color.lab2rgb(out_Lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))
