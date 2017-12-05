import os
import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import time


class ImageLoader(Dataset):
    def __init__(self, data_path, split, file_path='data.npz', flatten=True):
        file_path = os.path.join(data_path, file_path)
        self.dataset = np.load(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [transforms.ToTensor(), self.normalize]
        self.preprocess = transforms.Compose(transform)
        self.flatten = flatten

        self.split = split
        self.labels = np.array(self.dataset['arr_1'])
        self.images = np.array(self.dataset['arr_0'])

        # If Data is Frome CIFAR then only take the first 50000 indicies
        if file_path == 'data.npz':
            self.images = self.images[:50000, :]
            self.labels = self.labels[:50000, :]

        self.process = True if len(self.images.shape) > 2 else False

    def __getitem__(self, index):
        image = self.images[index]

        if self.process:
            img_tensor = self.preprocess(Image.fromarray(image.astype(np.int32), 'RGB'))
        else:
            img_tensor = torch.from_numpy(image)
            img_tensor = img_tensor.float()

        if self.flatten:
            if len(img_tensor.size()) > 2:
                batch_size = img_tensor.size(0)
                img_tensor = img_tensor.view(32 * 32 * 3)
        else:
            if len(img_tensor.size()) == 2:
                img_tensor = img_tensor.view(32, 32, 3)

        label = self.labels[index]
        label_tensor = torch.LongTensor(np.array([label]).astype(int))

        return img_tensor, label_tensor

    def __len__(self):
        return self.images.shape[0]
