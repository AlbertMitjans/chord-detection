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
    eval_fingers_precision = np.array([AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()])

    eval_frets_recall = AverageMeter()
    eval_frets_precision = np.array([AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()])

    eval_strings_recall = AverageMeter()
    eval_strings_precision = np.array([AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()])

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float().to(device)
        fingers = data['fingers'].float().to(device)
        frets = data['frets'].float().to(device)
        strings = data['strings'].float().to(device)
        fingers_coord = data['finger_coord']
        frets_coord = data['fret_coord']
        strings_coord = data['string_coord']

        # compute output
        output1 = model(input)[0].split(input.shape[0], dim=0)
        output2 = model(input)[1].split(input.shape[0], dim=0)
        output3 = model(input)[2].split(input.shape[0], dim=0)

        if show:
            import matplotlib.pyplot as plt
            import torchvision.transforms as transforms
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(fingers[0][0].cpu(), cmap='gray')
            ax[1].imshow(output1[-1][0][0].cpu().detach(), cmap='gray')
            ax[2].imshow(transforms.ToPILImage()(input.cpu()[0]))
            plt.show()

        # measure accuracy
        accuracy(output=output1[-1].data, target=fingers,
                 global_precision=eval_fingers_precision, global_recall=eval_fingers_recall, fingers=fingers_coord)

        accuracy(output=output2[-1].data, target=frets,
                 global_precision=eval_frets_precision, global_recall=eval_frets_recall,
                 fingers=frets_coord)

        accuracy(output=output3[-1].data, target=strings,
                 global_precision=eval_strings_precision, global_recall=eval_strings_recall,
                 fingers=strings_coord)

        if save_imgs:
            save_img(input.cpu().detach()[0], output1[-1].cpu().detach().numpy()[0], data['img_name'][0] + '_fingers')
            save_img(input.cpu().detach()[0], output2[-1].cpu().detach().numpy()[0], data['img_name'][0] + '_frets')
            save_img(input.cpu().detach()[0], output2[-1].cpu().detach().numpy()[0], data['img_name'][0] + '_strings')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('FINGERS: \t'
          'Recall(%): {top1:.3f}\t'
          'Precision num. corners (%): ({top2:.3f}, {top3:.3f}, {top4:.3f}, {top5:.3f})\n'
          'FRETS: \t'
          'Recall(%): {top6:.3f}\t'
          'Precision num. corners (%): ({top7:.3f}, {top8:.3f}, {top9:.3f}, {top10:.3f})\n'
          'STRINGS: \t'
          'Recall(%): {top11:.3f}\t'
          'Precision num. corners (%): ({top12:.3f}, {top13:.3f}, {top14:.3f}, {top15:.3f})\n'
        .format(top1=eval_fingers_recall.avg * 100, top2=eval_fingers_precision[0].avg * 100,
        top3=eval_fingers_precision[1].avg * 100,
        top4=eval_fingers_precision[2].avg * 100, top5=eval_fingers_precision[3].avg * 100,
        top6=eval_frets_recall.avg * 100, top7=eval_frets_precision[0].avg * 100,
        top8=eval_frets_precision[1].avg * 100,
        top9=eval_frets_precision[2].avg * 100, top10=eval_frets_precision[3].avg * 100,
        top11=eval_strings_recall.avg * 100, top12=eval_strings_precision[0].avg * 100,
        top13=eval_strings_precision[1].avg * 100,
        top14=eval_strings_precision[2].avg * 100, top15=eval_strings_precision[3].avg * 100))

    global_strings_precision = np.array(
        [eval_strings_precision[0].avg, eval_strings_precision[1].avg, eval_strings_precision[2].avg, eval_strings_precision[3].avg])

    global_fingers_precision = np.array(
        [eval_fingers_precision[0].avg, eval_fingers_precision[1].avg, eval_fingers_precision[2].avg,
         eval_fingers_precision[3].avg])

    global_frets_precision = np.array(
        [eval_frets_precision[0].avg, eval_frets_precision[1].avg, eval_frets_precision[2].avg,
         eval_frets_precision[3].avg])

    return eval_fingers_recall.avg, eval_frets_recall.avg, eval_strings_recall.avg, global_fingers_precision, global_frets_precision, global_strings_precision
