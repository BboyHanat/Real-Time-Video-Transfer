"""
Name : transform_net.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-11-20 14:30
Desc:
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable


def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
    upsample=None, instance_norm=True, relu=True):
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return layers

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x


class TransformNet(nn.Module):
    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9),
            *ConvLayer(base, base * 2, kernel_size=3, stride=2),
            *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
            *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False),
        )

    def forward(self, X):
        y = self.downsampling(X)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y


if __name__ == '__main__':
    style_net = TransformNet()

    one = Variable(torch.ones(1, 3, 436, 436))
    res = style_net(one)
    print(res.shape)
    pass