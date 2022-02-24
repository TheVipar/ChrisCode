import oneflow as flow


class TransformerNet(flow.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.layer = flow.nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            flow.nn.InstanceNorm2d(32, affine=True),
            flow.nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            flow.nn.InstanceNorm2d(64, affine=True),
            flow.nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            flow.nn.InstanceNorm2d(128, affine=True),
            flow.nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            flow.nn.InstanceNorm2d(64, affine=True),
            flow.nn.ReLU(),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            flow.nn.InstanceNorm2d(32, affine=True),
            flow.nn.ReLU(),
            ConvLayer(32, 3, kernel_size=9, stride=1),
        )

    def forward(self, X):
        y = self.layer(X)
        y = flow.clamp(y, 0, 255)
        return y


class ConvLayer(flow.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = flow.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = flow.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(flow.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = flow.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = flow.nn.InstanceNorm2d(channels, affine=True)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(flow.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        if self.upsample:
            self.interpolate = flow.nn.UpsamplingNearest2d(scale_factor=upsample)
        self.reflection_pad = flow.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = flow.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.interpolate(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
