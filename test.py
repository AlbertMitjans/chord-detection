from __future__ import print_function, division

import time
from PIL import Image
from torchvision.transforms import transforms
from transforms.pad_to_square import pad_to_square
import numpy as np

from utils.utils import AverageMeter, accuracy
from utils.img_utils import compute_gradient, save_img


def test(val_loader, model, device, save_imgs=False):
    batch_time = AverageMeter()

    eval_precision = np.array([AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()])

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float().cuda()
        target = data['grid'].float().cuda()

        # compute output
        output = model(input).split(input.shape[0], dim=0)

        # measure accuracy
        accuracy(output=output[-1].data, target=target,
                 global_precision=eval_precision)

        if save_imgs:
            save_img(input.cpu().detach()[0], output[-1].cpu().detach().numpy()[0], data['img_name'][0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('* Precision(%):  ({top2:.3f}, {top3:.3f}, {top4:.3f}, {top5:.3f})\t'
          .format(top2=eval_precision[0].avg * 100, top3=eval_precision[1].avg * 100,
                  top4=eval_precision[2].avg * 100, top5=eval_precision[3].avg * 100))

    global_precision = np.array(
        [eval_precision[0].avg, eval_precision[1].avg, eval_precision[2].avg, eval_precision[3].avg])

    return global_precision
