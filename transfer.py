import logging
from logging.handlers import TimedRotatingFileHandler
import os
import torch.optim as optim

from torchvision import transforms
from PIL import Image

from transform_net import TransformNet
from style_network import *
from loss_network import *
from dataset import get_loader, get_image_loader
from opticalflow import opticalflow
import cv2


osp = os.path

trHandler = TimedRotatingFileHandler("train_log.log", when="w1", interval=4, backupCount=12)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d:%(filename)-12s[%(lineno)4d] %(levelname)-6s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
level = logging.DEBUG
trHandler.setFormatter(formatter)
trHandler.setLevel(level)
logger = logging.getLogger()
logger.addHandler(trHandler)

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
        self.style_layer = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4']

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
        self.img_shape = img_shape                                                                  

    def load_style(self):

        img = Image.open(self.style_path)
        img = img.resize(self.img_shape)
        img = np.asarray(img, np.float32)/255.0
        img = self.transform(img)
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
        sgd = optim.SGD(self.style_net.parameters(), lr=self.lr, momentum=0.9)
        adadelta = optim.Adadelta(self.style_net.parameters(), lr=self.lr)
        
        loader = get_loader(1, self.data_path, self.img_shape, self.transform)
        logger.info('Data Load Success!!')
        print('Data Load Success!!')

        logger.info('Training Start!!')
        print('Training Start!!')
        for count in range(self.epoch):
            for step, frames in enumerate(loader):
                logger.info('step {}'.format(str(step)))
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

                    for layer in range(1, len(self.style_layer)):
                        style_t += StyleLoss(self.gpu)(s[layer], s_hxt[layer])
                        style_t1 += StyleLoss(self.gpu)(s[layer], s_hxt1[layer])


                        # TVLoss
                    tv_loss = TVLoss()(s_hxt[3])
                    style_loss = style_t + style_t1
                    if self.gpu:
                        flow, mask = opticalflow(h_xt1.data.cpu().numpy(), h_xt.data.cpu().numpy())
                    # Optical flow
                    else:
                        flow, mask = opticalflow(h_xt1.data.numpy(), h_xt.data.numpy())
                    if self.gpu:
                        flow = flow.cuda()
                        mask = mask.cuda()
                    # Temporal Loss
                    temporal_loss = TemporalLoss(self.gpu)(h_xt, flow, mask)
                    # Spatial Loss
                    spatial_loss = self.s_a * content_loss + self.s_b * style_loss + self.s_r * tv_loss
                    print('content_loss is {}, style_loss is {}, tv_loss is {}'.format(self.s_a * content_loss, self.s_b * style_loss, self.s_r * tv_loss))
                    Loss = content_loss # spatial_loss + self.t_l * temporal_loss
                    Loss.backward(retain_graph=True)
                    adadelta.step()

                    logger.info('Loss is: {}, spatial_loss is: {}, temporal_loss is: {}, step: {} frame {}'.format(str(Loss), str(spatial_loss), str(temporal_loss), str(step), str(i)))
                    print('Loss is: {}, spatial_loss is: {}, temporal_loss is: {}, step: {} frame {}'.format(str(Loss), str(spatial_loss), str(temporal_loss), str(step), str(i)))
                    if i % 300 == 0 and i >= 300:
                        s_np_image = x_t.data.cpu().numpy()
                        s_np_image = np.squeeze(np.transpose(s_np_image, (0, 2, 3, 1)))
                        transform_np_s = (s_np_image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255
                        transform_np_s = transform_np_s.clip(0, 255)
                        s_np_image = np.asarray(transform_np_s, np.uint8)
                        s_np_image = cv2.cvtColor(s_np_image, cv2.COLOR_RGB2BGR)

                        np_image = h_xt.data.cpu().numpy()
                        np_image = np.squeeze(np.transpose(np_image, (0, 2, 3, 1)))
                        # transform_np = (np_image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255
                        transform_np = (np_image + 1) * 127.5
                        transform_np = transform_np.clip(0, 255)
                        np_image = np.asarray(transform_np, np.uint8)
                        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite('output/style_e{}_s{}_i{}.jpg'.format(count, step, i), np_image)
                        cv2.imwrite('output/source_e{}_s{}_i{}.jpg'.format(count, step, i), s_np_image)
            logger.info('model saving')
            print('model saving')
            torch.save(self.style_net.state_dict(), 'model/style_model_epoch_{}.pth'.format(count))
            logger.info('model save finish')
            print('model save finish')


class ImageTransfer:
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
        self.style_layer = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4']

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
        self.img_shape = img_shape

    def load_style(self):

        img = Image.open(self.style_path)
        img = img.resize(self.img_shape)
        img = np.asarray(img, np.float32) / 255.0
        img = self.transform(img)
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
        sgd = optim.SGD(self.style_net.parameters(), lr=self.lr, momentum=0.9)
        adadelta = optim.Adadelta(self.style_net.parameters(), lr=self.lr)

        loader = get_image_loader(8, self.data_path, self.img_shape, self.transform)
        logger.info('Data Load Success!!')
        print('Data Load Success!!')

        logger.info('Training Start!!')
        print('Training Start!!')
        for count in range(self.epoch):
            for step, frames in enumerate(loader):

                x_t = frames[0]
                if self.gpu:
                    x_t = x_t.cuda()

                h_xt = self.style_net(x_t)

                s_xt = self.loss_net(x_t, self.style_layer)
                s_hxt = self.loss_net(h_xt, self.style_layer)
                s = self.loss_net(style_img, self.style_layer)

                # ContentLoss, conv4_2
                content_loss = ContentLoss(self.gpu)(s_xt[3], s_hxt[3])
                #content_loss = ContentLoss(self.gpu)(x_t, h_xt)

                # StyleLoss
                style_loss = StyleLoss(self.gpu)(s[0], s_hxt[0])

                for layer in range(1, len(self.style_layer)):
                    style_loss += StyleLoss(self.gpu)(s[layer], s_hxt[layer])

                    # TVLoss
                tv_loss = TVLoss()(h_xt)

                # Spatial Loss
                spatial_loss = self.s_a * content_loss + self.s_r * tv_loss + self.s_b * style_loss
                print('content_loss is {}, style_loss is {}, tv_loss is {}'.format(self.s_a * content_loss, self.s_b * style_loss, self.s_r * tv_loss))
                Loss = torch.mean(spatial_loss)  # spatial_loss + self.t_l * temporal_loss
                Loss.backward(retain_graph=True)
                sgd.step()

                logger.info('Loss is: {}, spatial_loss is: {} step: {} '.format(str(Loss), str(spatial_loss), str(step)))
                print('Loss is: {}, spatial_loss is: {}, step: {}'.format(str(Loss), str(spatial_loss), str(step)))
                if step % 70 == 0 and step >= 70:
                    s_np_image = x_t.data.cpu().numpy()
                    s_np_image = np.squeeze(np.transpose(s_np_image, (0, 2, 3, 1))[0, :, :, :])
                    transform_np_s = (s_np_image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255
                    transform_np_s = transform_np_s.clip(0, 255)
                    s_np_image = np.asarray(transform_np_s, np.uint8)
                    s_np_image = cv2.cvtColor(s_np_image, cv2.COLOR_RGB2BGR)

                    np_image = h_xt.data.cpu().numpy()
                    np_image = np.squeeze(np.transpose(np_image, (0, 2, 3, 1))[0,:,:,:])
                    # transform_np = (np_image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255
                    transform_np = (np_image + 1) * 127.5
                    transform_np = transform_np.clip(0, 255)
                    np_image = np.asarray(transform_np, np.uint8)
                    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('output/style_e{}_s{}.jpg'.format(count, step), np_image)
                    cv2.imwrite('output/source_e{}_s{}.jpg'.format(count, step), s_np_image)
            logger.info('model saving')
            print('model saving')
            torch.save(self.style_net.state_dict(), 'model/style_model_epoch_{}.pth'.format(count))
            logger.info('model save finish')
            print('model save finish')




if __name__ == '__main1__':
    # transfer = Transfer(10, 'data', '1.jpeg', 'model/vgg19-dcbb9e9d.pth', 0.1, 0.3, 0.3, 0.1, 0.2, gpu=False, img_shape=(480, 320))
    transfer = Transfer(10,
                        '/data/User/杨远东/登峰造极/视频素材',
                        'data/1.jpg',
                        'model/vgg19-dcbb9e9d.pth',
                        lr=0.001,
                        spatial_a=1,
                        spatial_b=0.00001,
                        spatial_r=0.000001,
                        temporal_lambda=10000,
                        gpu=True,
                        img_shape=(640, 360))
    transfer.train()

if __name__ == '__main__':
    # transfer = ImageTransfer(10, 'data/PNG', '1.jpeg', 'model/vgg19-dcbb9e9d.pth',
    #                          lr=0.001,spatial_a=1,spatial_b=0.00001,spatial_r=0.000001,temporal_lambda=10000,
    #                          gpu=False,
    #                          img_shape=(640, 360))
    transfer = ImageTransfer(100,
                        '/data/User/杨远东/登峰造极/图片素材/buildings',
                        'data/1.jpg',
                        'model/vgg19-dcbb9e9d.pth',
                        lr=0.01,
                        spatial_a=1,
                        spatial_b=0.00001,
                        spatial_r=0.00001,
                        temporal_lambda=10000,
                        gpu=True,
                        img_shape=(640, 360))

    transfer.train()