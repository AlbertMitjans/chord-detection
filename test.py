from __future__ import print_function, division

import time

from utils.utils import AverageMeter, accuracy


def test(val_loader, model, device):
    batch_time = AverageMeter()
    eval_accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float()
        input.to(device)
        target = data['tab'].float().cuda()
        target.to(device)

        # compute output
        output = model(input)

        # measure accuracy
        accuracy(output=output.data, target=target, accuracy=eval_accuracy)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Accuracy(%): {top1:.3f}\t'.format(top1=eval_accuracy.avg * 100,))

    return eval_accuracy.avg
