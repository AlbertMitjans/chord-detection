from __future__ import print_function, division

import torch.nn.parallel
import torch.optim
import time
import numpy as np

from test import test
from utils.utils import init_model_and_dataset, adjust_learning_rate, AverageMeter, accuracy
from utils.tb_visualizer import Logger


def train(ckpt, num_epochs, batch_size):
    num_workers = 0
    lr = 5e-4
    momentum = 0
    weight_decay = 0

    directory = 'dataset/'
    start_epoch = 0
    start_loss = 0
    print_freq = 100
    checkpoint_interval = 1
    evaluation_interval = 1

    logger = Logger('./logs')

    model, train_dataset, val_dataset, criterion_grid, optimizer = init_model_and_dataset(directory, lr,
                                                                                          weight_decay, momentum)
    val_dataset.evaluate()

    # load the pretrained network
    if ckpt is not None:
        checkpoint = torch.load(ckpt)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_loss = checkpoint['loss']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers, pin_memory=True)

    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_loss = AverageMeter()

        train_accuracy = AverageMeter()

        train_loss.update(start_loss)

        # switch to train mode
        model.train()

        end = time.time()
        for data_idx, data in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            input = data['image'].float().cuda()
            tab = data['tab'].float().cuda()

            # compute output
            output = model(input).split(input.shape[0], dim=0)
            loss = sum(i*criterion_grid(o, tab) for i, o in enumerate(output))

            # measure accuracy and record loss
            accuracy(output=output[-1].data, target=tab, accuracy=train_accuracy)
            train_loss.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if data_idx % print_freq == 0 and data_idx != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss.avg: {loss.avg:.4f}\t'
                      'Accuracy(%): {top1:.3f}\t'.format(
                    epoch, data_idx, len(train_loader), loss=train_loss,
                    top1=train_accuracy.avg * 100))

        if epoch % evaluation_interval == 0:
            # evaluate on validation set
            print('Train set:  ')

            t_recall, t_precision = test(train_loader, model)
            print('Validation set:  ')
            e_recall, e_precision = test(val_loader, model)

            # 1. Log scalar values (scalar summary)
            info = {'Train Loss': train_loss.avg, 'Train Recall': t_recall, 'Train Precision 1': t_precision[0],
                    'Train Precision 2': t_precision[1], 'Train Precision 3': t_precision[2],
                    'Train Precision 4': t_precision[3], 'Validation Recall': e_recall,
                    'Validation Precision 1': e_precision[0], 'Validation Precision 2': e_precision[1],
                    'Validation Precision 3': e_precision[2], 'Validation Precision 4': e_precision[3]}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

            # 3. Log training images (image summary)
            info = {'images': input.view(-1, 495, 495).cpu().numpy()}

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch)

        # remember best acc and save checkpoint
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss.avg
            }, "checkpoints/hg_ckpt_{0}.pth".format(epoch))
