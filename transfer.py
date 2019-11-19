import torch
import torch.optim as optim

from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image

from style_network import *
from loss_network import *
from dataset import get_loader
from opticalflow import opticalflow


class Transfer:
    def __init__(self, epoch, data_path, style_path, vgg_path, lr, spatial_a, spatial_b, spatial_r, temporal_lambda, gpu=False, img_shape=(640, 360)):
        self.epoch = epoch
        self.data_path = data_path
        self.style_path = style_path
        self.lr = lr
        self.gpu = gpu

        self.s_a = spatial_a
        self.s_b = spatial_b
        self.s_r = spatial_r 
        self.t_l = temporal_lambda

        self.style_net = StyleNet()
        self.loss_net = LossNet(vgg_path)
        self.style_layer = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2']

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
        self.img_shape = img_shape                                                                  

    def load_style(self):
        transform = transforms.Compose([transforms.ToTensor()])

        img = Image.open(self.style_path)
        img = img.resize(self.img_shape)
        img = transform(img).float()
        img = img.unsqueeze(0)
        img = Variable(img, requires_grad=True)
        return img

    def train(self):        
        style_img = self.load_style()

        if self.gpu:
            self.style_net = self.style_net.cuda()
            self.loss_net = self.loss_net.cuda()
            style_img = style_img.cuda()

        adam = optim.Adam(self.style_net.parameters(), lr=self.lr)
        
        loader = get_loader(1, self.data_path, self.img_shape, self.transform)
        print('Data Load Success!!')

        print('Training Start!!')
        for count in range(self.epoch):
            for step, frames in enumerate(loader):
                print("step {}".format(step))
                for i in range(1, len(frames)):

                    x_t = frames[i]
                    x_t1 = frames[i-1]
                    if self.gpu:
                        x_t = x_t.cuda()
                        x_t1 = x_t1.cuda()

                    h_xt = self.style_net(x_t)
                    h_xt1 = self.style_net(x_t1)

                    s_xt = self.loss_net(x_t, self.style_layer)
                    s_xt1 = self.loss_net(x_t1, self.style_layer)
                    s_hxt = self.loss_net(h_xt, self.style_layer)
                    s_hxt1 = self.loss_net(h_xt1, self.style_layer)
                    s = self.loss_net(style_img, self.style_layer)

                    # ContentLoss, conv4_2
                    content_t = ContentLoss(self.gpu)(s_xt[3], s_hxt[3])
                    content_t1 = ContentLoss(self.gpu)(s_xt1[3], s_hxt1[3])
                    content_loss = content_t + content_t1

                    # StyleLoss
                    style_t = StyleLoss(self.gpu)(s[0], s_hxt[0])
                    style_t1 = StyleLoss(self.gpu)(s[0], s_hxt1[0])
                    tv_loss = TVLoss()(s_hxt[0])
                    for layer in range(1, len(self.style_layer)):
                        style_t += StyleLoss(self.gpu)(s[layer], s_hxt[layer])
                        style_t1 += StyleLoss(self.gpu)(s[layer], s_hxt1[layer])

                        # TVLoss
                        tv_loss += TVLoss()(s_hxt[layer])
                    style_loss = style_t + style_t1

                    # Optical flow
                    flow, mask = opticalflow(h_xt.data.numpy(), h_xt1.data.numpy())

                    # Temporal Loss
                    temporal_loss = TemporalLoss(self.gpu)(h_xt, flow, mask)
                    # Spatial Loss
                    spatial_loss = self.s_a * content_loss + self.s_b * style_loss + self.s_r * tv_loss

                    Loss = spatial_loss + self.t_l * temporal_loss
                    Loss.backward(retain_graph=True)
                    adam.step()
                    print("Loss is: {}, epoch: {}".format(Loss, i))
            torch.save(self.style_net.state_dict(), 'model/densenet_ocr_model_e{}.pth'.format(count))




# transfer = Transfer(10, '/data/User/杨远东/登峰造极/视频素材', 'data/1.jpg', 'model/vgg19-dcbb9e9d.pth', 0.1, 0.3, 0.3, 0.1, 0.2, gpu=True, img_shape=(640, 480))
transfer = Transfer(10, '/data/User/杨远东/登峰造极/视频素材', 'data/1.jpg', 'model/vgg19-dcbb9e9d.pth', 0.1, 0.3, 0.3, 0.1, 0.2, gpu=True, img_shape=(640, 480))
transfer.train()