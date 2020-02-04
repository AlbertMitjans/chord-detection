from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.stats as st
import re

from PIL import Image, ImageFilter
from transforms.pad_to_square import pad_to_square


class CornersDataset(Dataset):
    def __init__(self, root_dir, end_file, target_shape=(8, 12, 12), transform=None):
        self.img_names = []
        self.corners = []
        self.colors = []
        self.root_dir = root_dir
        self.transform = transform
        self.end_file = end_file
        self.target_shape = target_shape
        self.validation = False
        self.read_csv()

    def read_csv(self):
        for root, dirs, files in os.walk(self.root_dir):
            files.sort(key=natural_keys)
            for file in files:
                if file.endswith(self.end_file):
                    self.img_names.append(file)
                if file.endswith(".csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.corners.append(f)

    def evaluate(self):
        self.validation = True

    def __len__(self):
        return len(self.corners)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])

        img_number = os.path.basename(img_name)[:-4]

        corners = np.array([self.corners[idx]])
        corners = corners.astype(int).reshape(-1, 2)

        image = Image.open(img_name)
        image = transforms.ToTensor()(image).type(torch.float32)[:3]

        grid = transforms.ToTensor()(gaussian(image, corners, kernel=int(image.shape[1]/5), target_size=image[0].size())[0]).type(torch.float32)
        grid = grid/grid.max()

        sample = {'image': image, 'grid': grid, 'img_name': img_number, 'corners': corners}

        if self.transform:
            sample = self.transform(sample)

        sample['image'] = pad_to_square(sample['image'])
        sample['grid'] = pad_to_square(sample['grid'])

        '''fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sample['grid'][0], cmap='gray')
        ax[1].imshow(transforms.ToPILImage()(sample['image']))
        plt.show()'''

        return sample


def gaussian(image, corners, kernel=20, nsig=5, target_size=(304, 495)):
    target = np.zeros((4, target_size[0], target_size[1]))
    n = float(image.shape[1]) / float(target.shape[1])
    m = float(image.shape[2]) / float(target.shape[2])
    for i, (x, y) in enumerate(corners):
        if x != -1 and y != -1:
            a = int(x / n)
            b = int(y / m)
            x = np.linspace(-nsig, nsig, kernel)
            kern1d = np.diff(st.norm.cdf(x))
            kern2d = np.outer(kern1d, kern1d)
            ax = a - kern2d.shape[0] // 2
            ay = b - kern2d.shape[1] // 2
            paste(target[i], kern2d / kern2d.max(), (ay, ax))

    target = np.resize(target.sum(0), (1, target_size[0], target_size[1]))

    return target


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]