from __future__ import print_function, division
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if np.random.random() < 0.5:
            image = sample['image']
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(image.shape[1] * self.size), int(image.shape[2] * self.size)))
            image = transforms.ToTensor()(TF.crop(transforms.ToPILImage()(image), i, j, h, w))

            sample['image'] = image
            fingers = []
            frets = []
            strings = []
            for finger in sample['finger_coord']:
                fingers.append([finger[0] - j, finger[1] - i])
            for fret in sample['fret_coord']:
                frets.append([fret[0] - j, fret[1] - i])
            for string in sample['string_coord']:
                strings.append([string[0] - j, string[1] - i])
            sample['finger_coord'] = np.array(fingers)
            sample['fret_coord'] = np.array(frets)
            sample['string_coord'] = np.array(strings)

        return sample
