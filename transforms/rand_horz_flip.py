import torch
import numpy as np


class HorizontalFlip(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['image'] = torch.flip(sample['image'], [-1])
            sample['fingers'] = torch.flip(sample['fingers'], [-1])
            sample['frets'] = torch.flip(sample['frets'], [-1])
            sample['strings'] = torch.flip(sample['strings'], [-1])

        return sample