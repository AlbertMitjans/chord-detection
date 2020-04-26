from __future__ import print_function, division
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if np.random.random() < 0.5:
            image, fingers, frets, strings = sample['image'], sample['fingers'], sample['frets'], sample['strings']
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(image.shape[1] * self.size), int(image.shape[2] * self.size)))
            image = transforms.ToTensor()(TF.crop(transforms.ToPILImage()(image), i, j, h, w))
            fingers = transforms.ToTensor()(TF.crop(transforms.ToPILImage()(fingers), i, j, h, w))
            frets = transforms.ToTensor()(TF.crop(transforms.ToPILImage()(frets), i, j, h, w))
            strings = transforms.ToTensor()(TF.crop(transforms.ToPILImage()(strings), i, j, h, w))

            sample['image'] = image
            sample['fingers'] = fingers
            sample['frets'] = frets
            sample['strings'] = strings

        return sample
