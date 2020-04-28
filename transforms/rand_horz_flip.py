import torch
import numpy as np


class HorizontalFlip(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['image'] = torch.flip(sample['image'], [-1])
            fingers = []
            frets = []
            strings = []
            for finger in sample['finger_coord']:
                fingers.append([sample['image'].shape[2] - finger[0], finger[1]])
            for fret in sample['fret_coord']:
                frets.append([sample['image'].shape[2] - fret[0], fret[1]])
            for string in sample['string_coord']:
                strings.append([sample['image'].shape[2] - string[0], string[1]])
            sample['finger_coord'] = np.array(fingers)
            sample['fret_coord'] = np.array(frets)
            sample['string_coord'] = np.array(strings)

        return sample