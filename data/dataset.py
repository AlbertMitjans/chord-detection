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
        self.tip = []
        self.knuckles1 = []
        self.knuckles2 = []
        self.notes = []
        self.frets = []
        self.strings = []
        self.features = []
        self.colors = []
        self.root_dir = root_dir
        self.transform = transform
        self.end_file = end_file
        self.target_shape = target_shape
        self.validation = False
        self.read_csv()

    def read_csv(self):
        print('Downloading dataset')
        for root, dirs, files in os.walk(self.root_dir):
            files.sort(key=natural_keys)
            for file in files:
                if file.endswith(self.end_file):
                    self.img_names.append(file)
                if file.endswith("_frets.csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.frets.append(f)
                if file.endswith("_strings.csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.strings.append(f)
                if file.endswith("_fingers.csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.tip.append(f)
                if file.endswith("_knuckle1.csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.knuckles1.append(f)
                if file.endswith("_knuckle2.csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.knuckles2.append(f)
                if file.endswith("_notes.csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.notes.append(f)

    def evaluate(self):
        self.validation = True

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])

        img_number = os.path.basename(img_name)[:-4]

        image = Image.open(img_name)
        image = transforms.ToTensor()(image).type(torch.float32)[:3]

        tip = np.array([self.tip[idx]])
        tip = tip.astype(int).reshape(-1, 2)

        '''knuckles1 = np.array([self.knuckles1[idx]])
        knuckles1 = knuckles1.astype(int).reshape(-1, 2)

        knuckles2 = np.array([self.knuckles2[idx]])
        knuckles2 = knuckles2.astype(int).reshape(-1, 2)

        notes = np.array([self.notes[idx]])
        notes = notes.astype(int).reshape(-1, 2)'''

        frets = np.array([self.frets[idx]])
        frets = frets.astype(int).reshape(-1, 2)

        strings = np.array([self.strings[idx]])
        strings = strings.astype(int).reshape(-1, 2)

        tip_grid = transforms.ToTensor()(gaussian(image, tip, kernel=int(image.shape[1]/5), target_size=image[0].size())).type(torch.float32)
        tip_grid = tip_grid/tip_grid.max()

        '''knuckles1_grid = transforms.ToTensor()(gaussian(image, knuckles1, kernel=int(image.shape[1] / 5), target_size=image[0].size())).type(torch.float32)
        knuckles1_grid = knuckles1_grid / knuckles1_grid.max()

        knuckles2_grid = transforms.ToTensor()(gaussian(image, knuckles2, kernel=int(image.shape[1] / 5), target_size=image[0].size())).type(torch.float32)
        knuckles2_grid = knuckles2_grid / knuckles2_grid.max()

        notes_grid = transforms.ToTensor()(gaussian(image, notes, kernel=int(image.shape[1] / 5), target_size=image[0].size())).type(torch.float32)
        notes_grid = notes_grid / notes_grid.max()'''

        target_grid = tip_grid
        target = tip

        frets_grid = transforms.ToTensor()(gaussian(image, frets, kernel=int(image.shape[1] / 10), target_size=image[0].size())).type(torch.float32)
        frets_grid = frets_grid / frets_grid.max()

        strings_grid = transforms.ToTensor()(gaussian(image, strings, kernel=int(image.shape[1] / 13), target_size=image[0].size())).type(torch.float32)
        strings_grid = strings_grid / strings_grid.max()

        sample = {'image': image, 'target': target_grid, 'frets': frets_grid, 'strings': strings_grid,
                  'img_name': img_number, 'target_coord': target, 'fret_coord': frets, 'string_coord': strings}

        if self.transform:
            sample = self.transform(sample)

        sample['image'] = pad_to_square(sample['image'])
        sample['target'] = pad_to_square(sample['target'])
        sample['frets'] = pad_to_square(sample['frets'])
        sample['strings'] = pad_to_square(sample['strings'])

        '''fig, ax = plt.subplots(2, 3)
        ax[0][0].axis('off')
        ax[0][1].axis('off')
        ax[0][2].axis('off')
        ax[1][0].axis('off')
        ax[1][1].axis('off')
        ax[1][2].axis('off')
        ax[0][0].imshow(transforms.ToPILImage()(sample['target'][0]), cmap='gray')
        ax[0][1].imshow(transforms.ToPILImage()(sample['target'][1]), cmap='gray')
        ax[1][0].imshow(transforms.ToPILImage()(sample['frets']), cmap='gray')
        ax[1][1].imshow(transforms.ToPILImage()(sample['strings']), cmap='gray')
        ax[1][2].imshow(transforms.ToPILImage()(sample['image']))
        plt.show()'''

        return sample


def gaussian(image, corners, kernel=5, nsig=5, target_size=(304, 495)):
    target = np.zeros((corners.shape[0], target_size[0], target_size[1]))
    n = float(image.shape[1]) / float(target.shape[1])
    m = float(image.shape[2]) / float(target.shape[2])
    for i, (x, y) in enumerate(corners):
        if x >= 0 and y >= 0:
            a = int(x / n)
            b = int(y / m)
            x = np.linspace(-nsig, nsig, kernel)
            kern1d = np.diff(st.norm.cdf(x))
            kern2d = np.outer(kern1d, kern1d)
            ax = a - kern2d.shape[0] // 2
            ay = b - kern2d.shape[1] // 2
            paste(target[i], kern2d / kern2d.max(), (ay, ax))

    target = target.sum(axis=0)

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