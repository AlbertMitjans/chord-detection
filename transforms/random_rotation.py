from __future__ import print_function, division
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class RandomRotation(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            image, fingers, frets, strings = sample['image'], sample['fingers'], sample['frets'], sample['strings']
            a, b, c, d = transforms.RandomAffine.get_params(degrees=(-20, 20), translate=None, scale_ranges=None, shears=None, img_size=(image.shape[1], image.shape[2]))
            image = transforms.ToTensor()(TF.affine(transforms.ToPILImage()(image), a, b, c, d))
            fingers = transforms.ToTensor()(TF.affine(transforms.ToPILImage()(fingers), a, b, c, d))
            frets = transforms.ToTensor()(TF.affine(transforms.ToPILImage()(frets), a, b, c, d))
            strings = transforms.ToTensor()(TF.affine(transforms.ToPILImage()(strings), a, b, c, d))

            sample['image'] = image
            sample['fingers'] = fingers
            sample['frets'] = frets
            sample['strings'] = strings

        return sample
