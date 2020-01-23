from __future__ import print_function, division

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms

from dataset.dataset import CornersDataset
from loss.loss import JointsMSELoss
from models.Stacked_Hourglass import HourglassNet, Bottleneck
from models.my_model import MyModel
from transforms.rand_crop import RandomCrop
from transforms.rand_horz_flip import HorizontalFlip


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


def init_model_and_dataset(depth, directory, lr=5e-6, weight_decay=0, momentum=0):
    # define the model
    model = HourglassNet(Bottleneck)
    model = nn.DataParallel(model).cuda()
    model2 = MyModel()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)

    checkpoint = torch.load("weights/hg_s2_b1/model_best.pth.tar")

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model = nn.Sequential(model, model2)

    end_file = '.png'

    cudnn.benchmark = True

    random_crop = RandomCrop(size=0.8)
    horizontal_flip = HorizontalFlip()

    train_dataset = CornersDataset(root_dir=directory + 'train_dataset', end_file=end_file, depth=depth,
                                   transform=transforms.Compose([random_crop, horizontal_flip]))
    val_dataset = CornersDataset(root_dir=directory + 'val_dataset', end_file=end_file, depth=depth,
                                 transform=transforms.Compose([random_crop, horizontal_flip]))

    return model, train_dataset, val_dataset, criterion, optimizer


def accuracy(corners, output, target, global_recall, global_precision):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    # we send the data to CPU
    output = output.cpu().detach().numpy().clip(0)
    target = target.cpu().detach().numpy()

    for batch_unit in range(batch_size):  # for each batch element
        recall, precision, max_out = multiple_gaussians(output[batch_unit], target[batch_unit])

        global_recall.update(recall)
        for i, (a, b) in enumerate(sorted(corners[batch_unit], key=lambda x: x[0], reverse=True)):
            if a != 0 and b != 0:
                global_precision[i].update(precision[i])

    return max_out