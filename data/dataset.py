from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.stats as st

from PIL import Image, ImageFilter
from transforms.pad_to_square import pad_to_square


class CornersDataset(Dataset):
    def __init__(self, root_dir, end_file, target_shape=(8, 12, 12), transform=None):
        self.img_path = []
        self.fingers = []
        self.hand = []
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
        files = open(self.root_dir).read().splitlines()
        print('Downloading dataset')
        for file in files:
            if file.endswith(self.end_file):
                self.img_path.append(file)
            elif file.endswith("_frets.csv"):
                f = pd.read_csv(file, header=None).values
                self.frets.append(f)
            elif file.endswith("_strings.csv"):
                f = pd.read_csv(file, header=None).values
                self.strings.append(f)
            elif file.endswith("_fingers.csv"):
                f = pd.read_csv(file, header=None).values
                self.fingers.append(f)
            elif file.endswith("_hand.csv"):
                f = pd.read_csv(file, header=None).values
                self.hand.append(f)
            if os.path.basename(os.path.dirname(file)) == '0' and file.endswith(self.end_file):
                self.hand.append([[0, 0], [0, 0]])

    def evaluate(self):
        self.validation = True

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_path[idx]

        img_folder = os.path.basename(os.path.dirname(img_path))

        img_name = os.path.basename(img_path)[:-4]

        img_number = os.path.basename(img_name)[5:]

        image = Image.open(img_path)
        image = transforms.ToTensor()(image).type(torch.float32)[:3]

        if img_folder == '2':
            hand_coord = np.array(self.hand[idx])
            image = image[:, int(hand_coord[0][1]):int(hand_coord[1][1]), int(hand_coord[0][0]):int(hand_coord[1][0])]

        fingers = np.array([self.fingers[idx]])
        fingers = fingers.astype(int).reshape(-1, 2)

        frets = np.array([self.frets[idx]])
        frets = frets.astype(int).reshape(-1, 2)

        strings = np.array([self.strings[idx]])
        strings = strings.astype(int).reshape(-1, 2)

        sample = {'image': image, 'finger_coord': fingers, 'fret_coord': frets, 'string_coord': strings}

        if self.transform:
            sample = self.transform(sample)

        fingers_grid = transforms.ToTensor()(gaussian(sample['image'], sample['finger_coord'], kernel=int(sample['image'].shape[1]/3), target_size=sample['image'][0].size())).type(torch.float32)
        fingers_grid = fingers_grid/fingers_grid.max()

        frets_grid = transforms.ToTensor()(gaussian(sample['image'], sample['fret_coord'], kernel=int(sample['image'].shape[1]/5), target_size=sample['image'][0].size())).type(torch.float32)
        frets_grid = frets_grid / frets_grid.max()

        strings_grid = transforms.ToTensor()(gaussian(sample['image'], sample['string_coord'], kernel=int(sample['image'].shape[1]/8), target_size=sample['image'][0].size())).type(torch.float32)
        strings_grid = strings_grid / strings_grid.max()

        sample = {'image': sample['image'], 'fingers': fingers_grid, 'frets': frets_grid, 'strings': strings_grid,
                  'img_number': img_number, 'finger_coord': sample['finger_coord'], 'fret_coord': sample['fret_coord'], 'string_coord': sample['string_coord']}

        sample['image'] = pad_to_square(sample['image'])
        sample['fingers'] = pad_to_square(sample['fingers'])
        sample['frets'] = pad_to_square(sample['frets'])
        sample['strings'] = pad_to_square(sample['strings'])

        '''im1 = transforms.ToPILImage()(sample['fingers'][0])
        im2 = transforms.ToPILImage()(sample['frets'][0])
        im3 = transforms.ToPILImage()(sample['strings'][0])

        im = Image.blend(transforms.ToPILImage()(sample['image']), Image.merge('RGB', (im1, im2, im3)), alpha=0.5)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(im1, cmap='gray')
        ax[0, 1].imshow(im2, cmap='gray')
        ax[1, 0].imshow(im3, cmap='gray')
        ax[1, 1].imshow(im)
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