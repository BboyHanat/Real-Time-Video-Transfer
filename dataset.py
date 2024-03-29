import os
import cv2
import torch
import numpy as np
import torch.utils.data as data

from torchvision import datasets, transforms


class Dataset(data.Dataset):
    def __init__(self, data_path, img_shape, transform):
        self.data_path = data_path
        self.img_shape = img_shape
        self.transform = transform
        self.video_list = [f for f in os.listdir(data_path) if '.DS_' not in f]

    def __getitem__(self, i):
        video = cv2.VideoCapture(os.path.join(self.data_path, self.video_list[i]))
        frames = list()

        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.resize(rgb_frame, self.img_shape)

            if self.transform is not None:
                rgb_frame = np.asarray(rgb_frame, np.float32) / 255.0
                rgb_frame = self.transform(rgb_frame)

            frames.append(rgb_frame)
        return frames

    def __len__(self):
        return len(self.video_list)


class ImageDataset(data.Dataset):
    def __init__(self, data_path, img_shape, transform):
        all_format = ['jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'PNG']
        self.data_path = data_path
        self.img_shape = img_shape
        self.transform = transform
        self.images = [os.path.join(data_path, f) for f in os.listdir(data_path)
                       if '.DS_' not in f and f.split('.')[-1] in all_format]

    def __getitem__(self, item):
        image_p = self.images[item]
        image = cv2.imread(image_p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_shape)
        if self.transform is not None:
            image = np.asarray(image, np.float32) / 255.0
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


def get_loader(batch_size, data_path, img_shape, transform, shuffle=True):
    dataset = Dataset(data_path, img_shape, transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return loader

def get_image_loader(batch_size, data_path, img_shape, transform, shuffle=True):
    dataset = Dataset(data_path, img_shape, transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return loader
