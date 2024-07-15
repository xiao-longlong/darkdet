import torch
import torch.nn as nn
import torch.nn.functional as F
from util_filters import *

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

class ExtractParameters(nn.Module):
    def __init__(self, cfg):
        super(ExtractParameters, self).__init__()
        self.cfg = cfg
        self.output_dim = cfg.num_filter_parameters
        self.channels = cfg.base_channels
        self.conv_layers = nn.Sequential(
            ConvBlock(3, self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
        )
        self.fc1 = nn.Linear(4096, cfg.fc1_size)
        self.fc2 = nn.Linear(cfg.fc1_size, self.output_dim)

    def forward(self, net):
        print('extract_parameters CNN:')
        print('    ', net.shape)
        net = self.conv_layers(net)
        net = net.view(-1, 4096)
        features = F.leaky_relu(self.fc1(net), 0.1)
        filter_features = self.fc2(features)
        return filter_features

class ExtractParameters2(nn.Module):
    def __init__(self, cfg):
        super(ExtractParameters2, self).__init__()
        self.cfg = cfg
        self.output_dim = cfg.num_filter_parameters
        self.channels = 16
        self.conv_layers = nn.Sequential(
            ConvBlock(3, self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
            ConvBlock(2 * self.channels, 2 * self.channels, kernel_size=3, downsample=True, bn=False),
        )
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, self.output_dim)

    def forward(self, net):
        print('extract_parameters_2 CNN:')
        print('    ', net.shape)
        net = self.conv_layers(net)
        net = net.view(-1, 2048)
        features = F.leaky_relu(self.fc1(net), 0.1)
        filter_features = self.fc2(features)
        return filter_features

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filter_num1, filter_num2, bn=True, activate=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, filter_num1, kernel_size=1, downsample=False, bn=bn, activate=activate)
        self.conv2 = ConvBlock(filter_num1, filter_num2, kernel_size=3, downsample=False, bn=bn, activate=activate)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += shortcut
        return x

def route(previous_output, current_output):
    return torch.cat((current_output, previous_output), dim=1)

def upsample(input_data, method="deconv"):
    assert method in ["resize", "deconv"]
    if method == "resize":
        return F.interpolate(input_data, scale_factor=2, mode='nearest')
    elif method == "deconv":
        return nn.ConvTranspose2d(input_data.shape[1], input_data.shape[1], kernel_size=2, stride=2, padding=0)(input_data)
