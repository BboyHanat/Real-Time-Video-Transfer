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
                rgb_frame = np.asarray(rgb_frame, np.float32)/255.0
                rgb_frame = self.transform(rgb_frame)

            frames.append(rgb_frame)
        return frames

    def __len__(self):
        return len(self.video_list)


def get_loader(batch_size, data_path, img_shape, transform, shuffle=True):
    dataset = Dataset(data_path, img_shape, transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader
