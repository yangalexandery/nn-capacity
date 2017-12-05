import os
import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import time


class ImageLoader(Dataset):
    def __init__(self, data_path, split, file_path='data.npz'):
        file_path = os.path.join(data_path, file_path)
        self.dataset = np.load(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [transforms.ToTensor(), self.normalize]
        self.preprocess = transforms.Compose(transform)

        self.split = split
        self.labels = np.array(self.dataset['arr_1'])
        self.images = np.array(self.dataset['arr_0'])

        self.process = True if len(self.images.shape) > 2 else False

    def __getitem__(self, index):
        image = self.images[index]

        if self.process:
            img_tensor = self.preprocess(Image.fromarray(image))
        else:
            img_tensor = torch.from_numpy(image)
            img_tensor = img_tensor.float()

        label = self.labels[index]
        label_tensor = torch.LongTensor(np.array([label]).astype(int))

        return img_tensor, label_tensor

    def __len__(self):
        return self.images.shape[0]
