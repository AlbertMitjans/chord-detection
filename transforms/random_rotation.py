from __future__ import print_function, division
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import math


class RandomRotation(object):
    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        angle = angle*math.pi/180

        ox, oy = float(origin[0]), float(origin[1])
        px, py = float(point[0]), float(point[1])

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def __call__(self, sample):
        if np.random.random() < 1:
            image = sample['image']
            a, b, c, d = transforms.RandomAffine.get_params(degrees=(-20, 20), translate=None, scale_ranges=None, shears=None, img_size=(image.shape[1], image.shape[2]))
            fingers = []
            frets = []
            strings = []
            for finger in sample['finger_coord']:
                fingers.append(self.rotate((image.shape[2]/2 + 0.5, image.shape[1]/2 + 0.5), finger, a))
            for fret in sample['fret_coord']:
                frets.append(self.rotate((image.shape[2]/2 + 0.5, image.shape[1]/2 + 0.5), fret, a))
            for string in sample['string_coord']:
                strings.append(self.rotate((image.shape[2]/2 + 0.5, image.shape[1]/2 + 0.5), string, a))
            sample['finger_coord'] = np.array(fingers)
            sample['fret_coord'] = np.array(frets)
            sample['string_coord'] = np.array(strings)
            image = transforms.ToTensor()(TF.affine(transforms.ToPILImage()(image), a, b, c, d))
            sample['image'] = image

        return sample
