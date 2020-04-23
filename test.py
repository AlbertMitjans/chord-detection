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

    eval_fingers_recall = AverageMeter()
    eval_fingers_precision = AverageMeter()

    eval_frets_recall = AverageMeter()
    eval_frets_precision = AverageMeter()

    eval_strings_recall = AverageMeter()
    eval_strings_precision = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float().to(device)
        target = data['target'].float().to(device)
        frets = data['frets'].float().to(device)
        strings = data['strings'].float().to(device)
        target_coord = data['target_coord']
        frets_coord = data['fret_coord']
        strings_coord = data['string_coord']

        # compute output
        output = model(input)
        output1 = output[0].split(input.shape[0], dim=0)
        output2 = output[1].split(input.shape[0], dim=0)
        output3 = output[2].split(input.shape[0], dim=0)

        if show:
            import matplotlib.pyplot as plt
            import torchvision.transforms as transforms
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(target[0][0].cpu(), cmap='gray')
            ax[1].imshow(output1[-1][0][0].cpu().detach(), cmap='gray')
            ax[2].imshow(transforms.ToPILImage()(input.cpu()[0]))
            plt.show()

        # measure accuracy
        accuracy(output=output1[-1].data, target=target,
                 global_precision=eval_fingers_precision, global_recall=eval_fingers_recall, fingers=target_coord,
                 min_dist= 10)

        accuracy(output=output2[-1].data, target=frets,
                 global_precision=eval_frets_precision, global_recall=eval_frets_recall,
                 fingers=frets_coord.unsqueeze(0), min_dist=5)

        accuracy(output=output3[-1].data, target=strings,
                 global_precision=eval_strings_precision, global_recall=eval_strings_recall,
                 fingers=strings_coord.unsqueeze(0), min_dist=5)

        if save_imgs:
            save_img(input.cpu().detach()[0], output1[-1][0][0].cpu().detach().numpy(), 10, data['img_name'][0] + '_fingers')
            save_img(input.cpu().detach()[0], output2[-1][0][0].cpu().detach().numpy(), 5, data['img_name'][0] + '_frets')
            save_img(input.cpu().detach()[0], output3[-1][0][0].cpu().detach().numpy(), 5, data['img_name'][0] + '_strings')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('FINGERS: \t'
          'Recall(%): {top1:.3f}\t'
          'Precision(%): {top2:.3f}\n'
          'FRETS:   \t'
          'Recall(%): {top6:.3f}\t'
          'Precision(%): {top7:.3f}\n'
          'STRINGS: \t'
          'Recall(%): {top11:.3f}\t'
          'Precision(%): {top12:.3f}\n'
        .format(top1=eval_fingers_recall.avg * 100, top2=eval_fingers_precision.avg * 100,
        top6=eval_frets_recall.avg * 100, top7=eval_frets_precision.avg * 100,
        top11=eval_strings_recall.avg * 100, top12=eval_strings_precision.avg * 100))

    return eval_fingers_recall.avg, eval_frets_recall.avg, eval_strings_recall.avg, eval_fingers_precision.avg, \
           eval_frets_precision.avg, eval_strings_precision.avg
