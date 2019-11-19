import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import vgg


def vgg19(vgg_path, pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = vgg.VGG(vgg.make_layers(vgg.cfg['E']), **kwargs)
    if pretrained:
        state_dict = torch.load(vgg_path)
        state_dict = {k: v for k, v in state_dict.items() if 'class' not in k}
        model.load_state_dict(state_dict)
    return model


class LossNet(nn.Module):
    def __init__(self, vgg_path):
        super(LossNet, self).__init__()
        model_list = list(vgg19(vgg_path).features.children())
        self.conv1_1 = model_list[0]
        self.conv1_2 = model_list[2]
        self.conv2_1 = model_list[5]
        self.conv2_2 = model_list[7]
        self.conv3_1 = model_list[10]
        self.conv3_2 = model_list[12]
        self.conv3_3 = model_list[14]
        self.conv3_4 = model_list[16]
        self.conv4_1 = model_list[19]
        self.conv4_2 = model_list[21]
        self.conv4_3 = model_list[23]
        self.conv4_4 = model_list[25]
        self.conv5_1 = model_list[28]
        self.conv5_2 = model_list[30]
        self.conv5_3 = model_list[32]
        self.conv5_4 = model_list[34]

    def forward(self, x, out_key):
        out = {}
        out['conv1_1'] = F.relu(self.conv1_1(x))
        out['conv1_2'] = F.relu(self.conv1_2(out['conv1_1']))
        out['pool1'] = F.max_pool2d(out['conv1_2'], kernel_size=2)

        out['conv2_1'] = F.relu(self.conv2_1(out['pool1']))
        out['conv2_2'] = F.relu(self.conv2_2(out['conv2_1']))
        out['pool2'] = F.max_pool2d(out['conv2_2'], kernel_size=2)

        out['conv3_1'] = F.relu(self.conv3_1(out['pool2']))
        out['conv3_2'] = F.relu(self.conv3_2(out['conv3_1']))
        out['conv3_3'] = F.relu(self.conv3_3(out['conv3_2']))
        out['conv3_4'] = F.relu(self.conv3_4(out['conv3_3']))
        out['pool3'] = F.max_pool2d(out['conv3_4'], kernel_size=2)

        out['conv4_1'] = F.relu(self.conv4_1(out['pool3']))
        out['conv4_2'] = F.relu(self.conv4_2(out['conv4_1']))
        # out['conv4_3'] = F.relu(self.conv4_3(out['conv4_2']))
        # out['conv4_4'] = F.relu(self.conv4_4(out['conv4_3']))
        # out['pool4']   = F.max_pool2d(out['conv4_4'], kernel_size=2)
        #
        # out['conv5_1'] = F.relu(self.conv5_1(out['pool4']))
        # out['conv5_2'] = F.relu(self.conv5_2(out['conv5_1']))
        # out['conv5_3'] = F.relu(self.conv5_3(out['conv5_2']))
        # out['conv5_4'] = F.relu(self.conv5_4(out['conv5_3']))
        # out['pool5']   = F.max_pool2d(out['conv5_4'], kernel_size=2)

        return [out[key] for key in out_key]


class ContentLoss(nn.Module):
    def __init__(self, gpu):
        super(ContentLoss, self).__init__()
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, target):
        assert x.shape == target.shape, "input & target shape ain't same."
        b, c, h, w = x.shape

        return (1 / (c * h * w)) * torch.sqrt(torch.sum((x - target) ** 2))


class StyleLoss(nn.Module):
    def __init__(self, gpu):
        super(StyleLoss, self).__init__()
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, target):
        channel = x.shape[3]
        loss = (1 / (channel ** 2)) * self.loss(GramMatrix()(x), GramMatrix()(target))
        return loss


class TemporalLoss(nn.Module):
    """
    x: frame t 
    f_x1: optical flow(frame t-1)
    cm: confidence mask of optical flow 
    """

    def __init__(self, gpu):
        super(TemporalLoss, self).__init__()
        if gpu:
            loss = nn.MSELoss().cuda()
        else:
            loss = nn.MSELoss()
        self.loss = loss

    def forward(self, x, f_x1, cm):
        assert x.shape == f_x1.shape, "inputs are ain't same"
        _, c, h, w = x.shape
        power_sub = (x - f_x1) ** 2
        loss = torch.sum(cm * power_sub[:, 0, :, :] + cm * power_sub[:, 1, :, :] + cm * power_sub[:, 2, :, :]) / (w * h)
        return loss


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        for i_w in range(w - 1):
            sum_cols = (x[0, :, :, i_w + 1] - x[0, :, :, i_w]) ** 2
        sum = torch.sum(sum_cols)
        for i_h in range(h - 1):
            sum_rows = (x[0, :, [i_h + 1], :] - x[0, :, i_h, :]) ** 2
        sum += torch.sum(sum_rows)

        return sum ** 0.5


class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, x):
        a, b, c, d = x.shape
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
