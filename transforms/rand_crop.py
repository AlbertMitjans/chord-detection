from __future__ import print_function, division
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image, ImageFilter


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if np.random.random() < 0.5:
            image, grid = sample['image'], sample['grid']
            rc = transforms.RandomCrop((int(image.shape[1] * self.size), int(image.shape[2] * self.size)))
            im_grid = torch.cat((image, grid))
            crop_im_grid = rc(transforms.ToPILImage()(im_grid))
            im1, im2, im3, gr1, gr2, gr3, gr4 = Image.Image.split(crop_im_grid)
            crop_image = Image.merge('RGB', (im1, im2, im3))
            crop_grid = Image.merge('RGBA', (gr1, gr2, gr3, gr4))
            res_image = transforms.ToTensor()(crop_image.resize((image.shape[2], image.shape[1])))
            grid = transforms.ToTensor()(crop_grid.resize((image.shape[2], image.shape[1])))

            sample['image'] = res_image
            sample['grid'] = grid

        return sample
