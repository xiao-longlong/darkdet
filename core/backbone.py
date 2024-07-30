import torch.nn as nn
from core.common import ConvBlock, ResidualBlock

class Darknet53(nn.Module):
    def __init__(self, trainable=True):
        super(Darknet53, self).__init__()
        self.trainable = trainable

        self.conv0 = ConvBlock(3, 32, kernel_size=3)
        self.conv1 = ConvBlock(32, 64, kernel_size=3, downsample=True)
        self.residual0 = ResidualBlock(64, 32, 64)

        self.conv4 = ConvBlock(64, 128, kernel_size=3, downsample=True)
        self.residual1 = ResidualBlock(128, 64, 128)
        self.residual2 = ResidualBlock(128, 64, 128)

        self.conv9 = ConvBlock(128, 256, kernel_size=3, downsample=True)
        self.residual3 = nn.ModuleList([ResidualBlock(256, 128, 256) for _ in range(8)])

        self.conv26 = ConvBlock(256, 512, kernel_size=3, downsample=True)
        self.residual11 = nn.ModuleList([ResidualBlock(512, 256, 512) for _ in range(8)])

        self.conv43 = ConvBlock(512, 1024, kernel_size=3, downsample=True)
        self.residual19 = nn.ModuleList([ResidualBlock(1024, 512, 1024) for _ in range(4)])

    def forward(self, input_data):
        input_data = self.conv0(input_data)
        input_data = self.conv1(input_data)
        input_data = self.residual0(input_data)

        input_data = self.conv4(input_data)
        input_data = self.residual1(input_data)
        input_data = self.residual2(input_data)

        input_data = self.conv9(input_data)
        for layer in self.residual3:
            input_data = layer(input_data)

        route_1 = input_data
        input_data = self.conv26(input_data)
        for layer in self.residual11:
            input_data = layer(input_data)

        route_2 = input_data
        input_data = self.conv43(input_data)
        for layer in self.residual19:
            input_data = layer(input_data)

        return route_1, route_2, input_data