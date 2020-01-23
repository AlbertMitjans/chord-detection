from __future__ import print_function, division
import numpy as np
import torchvision.transforms as transforms


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if np.random.random() < 0.5:
            image = sample['image']
            rc = transforms.RandomCrop((int(image.shape[1] * self.size), int(image.shape[2] * self.size)))
            crop_image = rc(transforms.ToPILImage()(image))
            res_image = transforms.ToTensor()(crop_image.resize((image.shape[2], image.shape[1])))

            sample['image'] = res_image

        return sample
