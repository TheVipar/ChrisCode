from collections import namedtuple

import oneflow as flow
import oneflow.nn as nn


__all__ = [
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

# 31 is the raw depth of vgg16, while 37 is the raw depth of vgg19
slice_pos = {31: [4, 9, 16, 23], 37: [4, 9, 18, 27]}


class VGG_WITH_FEATURES(flow.nn.Module):
    def __init__(self, vgg_pretrained_features, requires_grad):
        super(VGG_WITH_FEATURES, self).__init__()
        self.slice1 = flow.nn.Sequential()
        self.slice2 = flow.nn.Sequential()
        self.slice3 = flow.nn.Sequential()
        self.slice4 = flow.nn.Sequential()
        pos = slice_pos[len(vgg_pretrained_features)]
        for x in range(pos[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(pos[0], pos[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(pos[1], pos[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(pos[2], pos[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
