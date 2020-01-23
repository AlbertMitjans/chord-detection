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


class CornersDataset(Dataset):
    def __init__(self, root_dir, end_file, transform=None):
        self.img_names = []
        self.chords = pd.read_csv('targets.txt')
        self.tabs = {}
        self.end_file = end_file
        self.root_dir = root_dir
        self.transform = transform
        self.validation = False
        self.load_data()

    def load_data(self):
        for root, dirs, files in os.walk(os.path.join(self.root_dir, 'images')):
            files.sort(key=natural_keys)
            for file in files:
                if file.endswith(self.end_file):
                    self.img_names.append(file)

        chord_dict = pd.read_excel('guitar_chords.xlsx')

        for i in range(chord_dict.shape[0]):
            try:
                num_strings = len(self.tabs[chord_dict['Chord'][i]])
            except KeyError:
                num_strings = 0

            if num_strings < 6:
                self.tabs.setdefault(chord_dict['Chord'][i][0], []).append(chord_dict['Fret'][i])

    def evaluate(self):
        self.validation = True

    def __len__(self):
        return len(self.chords)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])
        img_number = os.path.basename(img_name)[:-4]

        chord = self.chords[idx].astype(int)

        tab = np.array([self.tabs[chord]])

        image = Image.open(img_name)
        image = transforms.ToTensor()(image).type(torch.float32)

        sample = {'image': image, 'chord': chord, 'tab': tab, 'img_num': img_number}

        if not self.validation:
            if self.transform:
                sample = self.transform(sample)

        sample['image'] = pad_to_square(sample['image'])

        '''fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sample['grid'][0], cmap='gray')
        ax[1].imshow(sample['image'][0], cmap='gray')
        plt.show()
        plt.waitforbuttonpress()
        plt.close('all')'''

        return sample


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_tabs(directory, chord_dict):
    chords_tab = {}

    for i in range(chord_dict.shape[0]):
        try:
            num_strings = len(chords_tab[chord_dict['Chord'][i]])
        except KeyError:
            num_strings = 0

        if num_strings < 6:
            chords_tab.setdefault(chord_dict['Chord'][i], []).append(chord_dict['Fret'][i])

    target_chords = pd.read_csv('targets.txt')

    target_tabs = []

    for i in target_chords.values:
        target_tabs.append(chords_tab[i[0]])
