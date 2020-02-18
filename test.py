from __future__ import print_function, division

import time
from PIL import Image
from torchvision.transforms import transforms
from transforms.pad_to_square import pad_to_square
import numpy as np

from utils.utils import AverageMeter, accuracy
from utils.img_utils import compute_gradient, save_img


def test(val_loader, model, device, save_imgs=False, show=False):
    batch_time = AverageMeter()
    eval_recall = AverageMeter()
    eval_precision = np.array([AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()])

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float().cuda()
        target = data['grid'].float().cuda()
        fingers = data['fingers']

        # compute output
        output = model(input).split(input.shape[0], dim=0)

        if show:
            import matplotlib.pyplot as plt
            import torchvision.transforms as transforms
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(target[0][0].cpu(), cmap='gray')
            ax[1].imshow(output[-1][0][0].cpu().detach(), cmap='gray')
            ax[2].imshow(transforms.ToPILImage()(input.cpu()[0]))
            plt.show()

        # measure accuracy
        accuracy(fingers=fingers, output=output[-1].data, target=target, global_recall=eval_recall,
                 global_precision=eval_precision)

        if save_imgs:
            save_img(input.cpu().detach()[0], output[-1].cpu().detach().numpy()[0], data['img_name'][0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Recall(%): {top1:.3f}\t' ' * Precision(%):  ({top2:.3f}, {top3:.3f}, {top4:.3f}, {top5:.3f})\t'
          .format(top1=eval_recall.avg * 100, top2=eval_precision[0].avg * 100, top3=eval_precision[1].avg * 100,
                  top4=eval_precision[2].avg * 100, top5=eval_precision[3].avg * 100))

    global_precision = np.array(
        [eval_precision[0].avg, eval_precision[1].avg, eval_precision[2].avg, eval_precision[3].avg])

    return eval_recall.avg, global_precision
