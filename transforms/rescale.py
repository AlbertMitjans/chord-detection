import torchvision.transforms as transform
import torch
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[-2:]
        if isinstance(self.output_size, int):
            if h < w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        resize = transform.Resize((new_h, new_w))

        img = transform.ToTensor()(resize(transform.ToPILImage()(image)))

        rx = img.shape[1]/image.shape[1]
        ry = img.shape[2]/image.shape[2]

        sample['image'] = img
        fingers = []
        frets = []
        strings = []
        for finger in sample['finger_coord']:
            fingers.append([rx*finger[0], ry*finger[1]])
        for fret in sample['fret_coord']:
            frets.append([rx*fret[0], ry*fret[1]])
        for string in sample['string_coord']:
            strings.append([rx*string[0], ry*string[1]])
        sample['finger_coord'] = np.array(fingers)
        sample['fret_coord'] = np.array(frets)
        sample['string_coord'] = np.array(strings)

        return sample
