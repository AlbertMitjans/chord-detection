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
from models.MTL_stacked_hourglass import HourglassNet, Bottleneck
from models.MTL_my_model import MyModel
from transforms.rand_crop import RandomCrop
from transforms.rand_horz_flip import HorizontalFlip
from transforms.rescale import Rescale
from transforms.random_rotation import RandomRotation


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


def init_model_and_dataset(directory, device, lr=5e-6, weight_decay=0):
    # define the model
    model = HourglassNet(Bottleneck)
    model2 = MyModel()
    model = nn.Sequential(model, model2)
    model = nn.DataParallel(model)
    model.to(device)
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)

    checkpoint = torch.load("checkpoints/best_ckpt/MTL_pretrained.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    end_file = '.jpg'

    cudnn.benchmark = True

    random_crop = RandomCrop(size=0.95)
    horizontal_flip = HorizontalFlip()
    random_rotation = RandomRotation()
    rescale = Rescale(300)

    train_dataset = CornersDataset(root_dir=directory + 'train.txt', end_file=end_file,
                                   transform=transforms.Compose([random_crop, horizontal_flip, random_rotation, rescale]))
    val_dataset = CornersDataset(root_dir=directory + 'val.txt', end_file=end_file,
                                 transform=transforms.Compose([rescale]))

    return model, train_dataset, val_dataset, criterion, optimizer


def accuracy(fingers, output, target, global_recall, global_precision, min_dist):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    # we send the data to CPU
    output = output.cpu().detach().numpy().clip(0)
    target = target.cpu().detach().numpy()

    for batch_unit in range(batch_size):  # for each batch element
        for joint in range(target.shape[1]):
            recall, precision, max_out = multiple_gaussians(output[batch_unit][joint], target[batch_unit][joint], min_dist)

            global_recall.update(recall)
            if max_out.shape[0] != 0:
                global_precision.update(precision)


def multiple_gaussians(output, target, min_dist):
    # we calculate the positions of the max value in output and target
    max_target = peak_local_max(target, min_distance=min_dist, exclude_border=False, indices=False)  # num_peaks=4)
    labels_target = label(max_target)[0]
    max_target = np.array(center_of_mass(max_target, labels_target, range(1, np.max(labels_target) + 1))).astype(np.int)

    truep = 0
    allp = 0

    max_out = peak_local_max(output, min_distance=min_dist, threshold_rel=0.5, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)

    max_values = []

    for index in max_out:
        max_values.append(output[index[0]][index[1]])

    max_out_2 = np.array([x for _, x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    for (a, b) in max_target:
        allp += 1
        for (c, d) in max_out_2:
            l = np.absolute((a - c, b - d))
            if l[0] <= min_dist and l[1] <= min_dist:
                delete = np.where(np.all(max_out_2 == np.array([c, d]), axis=1))[0]
                max_out_2 = np.delete(max_out_2, delete, 0)
                truep += 1
                break

    if max_out.shape[0] == 0 or max_target.shape[0] == 0:
        total_recall = 0
        total_precision = 0
    else:
        total_recall = truep / allp
        total_precision = truep / max_out.shape[0]

    '''import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(target[0], cmap='gray')
    ax[1].imshow(output[0], cmap='gray')
    plt.show()

    print(precision)'''

    return total_recall, total_precision, max_out