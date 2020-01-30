from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import re
from PIL import Image
from transforms.pad_to_square import pad_to_square
import scipy.stats as st


class CornersDataset(Dataset):
    def __init__(self, root_dir, end_file, transform=None):
        self.img_names = []
        self.chords = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'targets.csv'), header=None).values.tolist()
        self.tabs = {}
        self.fingers = []
        self.end_file = end_file
        self.root_dir = root_dir
        self.transform = transform
        self.load_data()

    def load_data(self):
        for root, dirs, files in os.walk(self.root_dir):
            files.sort(key=natural_keys)
            for file in files:
                if file.endswith(self.end_file):
                    self.img_names.append(file)
                if file.endswith(".csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None).values
                    self.fingers.append(f)

        chord_dict = pd.read_excel(os.path.join(os.getcwd(), 'dataset', 'guitar_chords.xlsx'))

        for i in range(chord_dict.shape[0]):
            try:
                num_strings = len(self.tabs[chord_dict['Chord'][i]])
            except KeyError:
                num_strings = 0

            if num_strings < 6:
                self.tabs.setdefault(chord_dict['Chord'][i], []).append(chord_dict['Fret'][i])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])
        img_number = int(os.path.basename(img_name)[5:-4])

        chord = self.chords[img_number-1][0]

        fingers = self.fingers[idx]

        tab = torch.Tensor(self.tabs[chord])
        tab = one_hot_label(tab)

        image = Image.open(img_name)

        image = transforms.ToTensor()(image).type(torch.float32)[:3]

        sample = {'image': image, 'chord': chord, 'tab': tab, 'img_num': img_number}

        if self.transform:
            sample = self.transform(sample)

        sample['image'] = pad_to_square(sample['image'])

        '''import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(transforms.ToPILImage()(sample['image']))
        ax[1].imshow(sample['target'][0])
        plt.show()
        plt.waitforbuttonpress()
        plt.close('all')'''

        return sample


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


def one_hot_label(tabs):
    output = torch.zeros((6, 25))
    for idx, value in enumerate(tabs):
        if value != -1:
            output[idx, int(value)] = 1

    return output