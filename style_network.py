import torch
import torchvision 
import torch.nn as nn 
import numpy as np
from collections import OrderedDict

from torch.autograd import Variable

def conv_block(name, in_C, out_C, activation='ReLU', kernel_size=3, stride=1, padding=1, InstanceNorm=True):
    block = nn.Sequential()
    if activation == 'ReLU':
        activate = nn.ReLU
    elif activation == 'Tanh':
        activate = nn.Tanh
    conv = ' Conv'
    block.add_module(name + conv, nn.Conv2d(in_C, out_C, kernel_size, stride, padding))
    if InstanceNorm:
        block.add_module(name + 'Inst_norm', nn.InstanceNorm2d(out_C))
    if activation == 'ReLU':
        block.add_module(name + ' ' + activation, activate(inplace=True))
    elif activation == 'Tanh':
        block.add_module(name + ' ' + activation, activate())
    
    return block


def res_block(name):
        res_block = nn.Sequential()
        res_block.add_module(name + '_1', conv_block(name + 'res1', 48, 48))
        res_block.add_module(name + '_2', conv_block(name + 'res2', 48, 48, activation=''))
        return res_block


class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        name = "StyleNet"

        self.layer1 = conv_block(name + ' 1', 3, 16)
        self.layer2 = conv_block(name + ' 2', 16, 32, stride=2)
        self.layer3 = conv_block(name + ' 3', 32, 48, stride=2)
        self.res1 = res_block(name + ' ResBlock1')
        self.res2 = res_block(name + ' ResBlock2')
        self.res3 = res_block(name + ' ResBlock3')
        self.res4 = res_block(name + ' ResBlock4')
        self.res5 = res_block(name + ' ResBlock5')
        self.layer4 = conv_block(name + ' 4', 48, 32, stride=1)
        self.layer5 = conv_block(name + ' 5', 32, 16, stride=1)
        self.layer6 = conv_block(name + ' 6', 16, 3, activation='Tanh', InstanceNorm=True)  # first test
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        res1 = self.res1(out3) + out3               # second test
        res2 = self.res2(res1) + res1
        res3 = self.res3(res2) + res2
        res4 = self.res4(res3) + res3
        res5 = self.res5(res4) + res4 + out3
        out4 = self.layer4(res5)
        out4 = self.up_sample(out4) + out2
        out5 = self.layer5(out4)
        out5 = self.up_sample(out5) + out1
        out6 = self.layer6(out5)
        return out6


if __name__ == '__main__':
    style_net = StyleNet()

    one = Variable(torch.ones(1, 3, 436, 436))
    res = style_net(one)
    print(res.shape)
    pass