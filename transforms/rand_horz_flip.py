import torch
import numpy as np


class HorizontalFlip(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            image = sample['image']
            sample['image'] = torch.flip(image, [-1])

        return sample
