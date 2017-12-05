import os
import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
# from torchvision import transforms
import transforms
import affine_transforms
import torch
import time


class MiniPlace(Dataset):
    def __init__(self, data_path, split, num_corrupted):
        self.count = 0
        filepath_data = os.path.join(data_path, 'data.npz')
        filepath_corr = os.path.join(data.path, 'corr_perm.npz')

        self.dataset = np.load(filepath_data)
        self.images = self.dataset['arr_0'][:50000]
        self.labels = self.dataset['arr_1'][:50000]

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [transforms.ToTensor()]
        transform += [
            self.normalize]

        self.preprocess = transforms.Compose(transform)

        self.split = split

        if split == 'train':
            corr = np.load(filepath_corr)
            for i in range(num_corrupted):
                ind = corr['arr_0'][i]
                val = corr['arr_1'][i]
                self.labels[ind] = val

    def __getitem__(self, index):
        self.count += 1

        # if self.load_everything:
        #     image = self.images[index]
        # else:
        image = self.images[index]
        img_tensor = self.preprocess(Image.fromarray(image))

        # if self.split == 'test':
        #     return img_tensor, index

        label = self.labels[index]
        label_tensor = torch.LongTensor(np.array([label]).astype(int))

        return img_tensor, label_tensor

    def __len__(self):
        return self.images.shape[0]