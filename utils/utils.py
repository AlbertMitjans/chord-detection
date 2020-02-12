from __future__ import print_function, division

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max

from data.dataset import CornersDataset
from loss.loss import JointsMSELoss
from models.Stacked_Hourglass import HourglassNet, Bottleneck
from transforms.rand_crop import RandomCrop
from transforms.rand_horz_flip import HorizontalFlip
from transforms.rescale import Rescale


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_model_and_dataset(directory, device, lr=5e-6, weight_decay=0, momentum=0):
    # define the model
    model = HourglassNet(Bottleneck)
    model = nn.DataParallel(model)
    model.to(device)
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)

    checkpoint = torch.load("checkpoints/hg_s2_b1/model_best.pth.tar", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model = nn.Sequential(model, nn.Conv2d(16, 4, kernel_size=1).to(device))
    model = nn.DataParallel(model)
    model.to(device)

    end_file = '.jpg'

    cudnn.benchmark = True

    random_crop = RandomCrop(size=0.9)
    horizontal_flip = HorizontalFlip()
    rescale = Rescale((300, 300))

    train_dataset = CornersDataset(root_dir=directory + 'train_dataset', end_file=end_file,
                                   transform=transforms.Compose([horizontal_flip, rescale]))
    val_dataset = CornersDataset(root_dir=directory + 'val_dataset', end_file=end_file,
                                 transform=transforms.Compose([horizontal_flip, rescale]))

    return model, train_dataset, val_dataset, criterion, optimizer


def accuracy(output, target, global_precision):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    # we send the data to CPU
    output = output.cpu().detach().numpy().clip(0)
    target = target.cpu().detach().numpy()

    for batch_unit in range(batch_size):  # for each batch element
        precision = multiple_gaussians(output[batch_unit], target[batch_unit])

        for i in range(4):
            global_precision[i].update(precision[i])


def multiple_gaussians(output, target):
    precision = np.array([0, 0, 0, 0]).astype(np.float)
    # we calculate the positions of the max value in output and target
    for idx, _ in enumerate(output):
        max_target = np.array(np.unravel_index(target[idx].argmax(), target[idx].shape))
        max_output = np.array(np.unravel_index(output[idx].argmax(), output[idx].shape))

        l = np.abs(max_target - max_output)

        if l[0] <= 10 and l[1] <= 10:
            precision[idx] = 1.

    '''import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(target[0], cmap='gray')
    ax[1].imshow(output[0], cmap='gray')
    plt.show()

    print(precision)'''

    return precision
