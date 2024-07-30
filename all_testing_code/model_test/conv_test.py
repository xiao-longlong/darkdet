import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False, bn=True, activate=True):
        super(ConvBlock, self).__init__()
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activate:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class MyModel(nn.Module):
    def __init__(self, channels):
        super(MyModel, self).__init__()
        self.channels = channels
        self.conv_layers = nn.Sequential(
            ConvBlock(3, self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
        )

    def forward(self, x):
        return self.conv_layers(x)

# 测试模型
channels = 16  # 假设 channels 是 64
model = MyModel(channels)

# 创建一个随机张量，形状为 [batch_size, channels, height, width]
input_tensor = torch.randn(6, 3, 256, 256)

# 前向传播
output_tensor = model(input_tensor)

# 打印输出张量的形状
print(output_tensor.shape)
