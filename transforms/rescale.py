import torchvision.transforms as transform
import torch


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
        target = sample['target']
        frets = sample['frets']
        strings = sample['strings']

        h, w = image.shape[-2:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        resize = transform.Resize((new_h, new_w))

        img = transform.ToTensor()(resize(transform.ToPILImage()(image)))
        target1 = transform.ToTensor()(resize(transform.ToPILImage()(target[0])))
        target2 = transform.ToTensor()(resize(transform.ToPILImage()(target[1])))
        target = torch.cat((target1, target2))
        frets = transform.ToTensor()(resize(transform.ToPILImage()(frets)))
        strings = transform.ToTensor()(resize(transform.ToPILImage()(strings)))

        sample['image'] = img
        sample['target'] = target
        sample['frets'] = frets
        sample['strings'] = strings

        return sample
