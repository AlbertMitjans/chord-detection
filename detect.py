import torch
from models.Stacked_Hourglass import Bottleneck, HourglassNet
from models.my_model import MyModel
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.img_utils import local_max
import numpy as np


def rescale(image, size):
    h, w = image.shape[-2:]
    if isinstance(size, int):
        if h > w:
            new_h, new_w = size * h / w, size
        else:
            new_h, new_w = size, size * w / h
    else:
        new_h, new_w = size

    new_h, new_w = int(new_h), int(new_w)
    resize = transforms.Resize((new_h, new_w))
    img = transforms.ToTensor()(resize(transforms.ToPILImage()(image)))

    return img


def fill_values(fingers, frets, strings):
    strings_x_order = strings[(-strings)[:, 1].argsort()]

    # We compute the vectors of the strings

    v1 = []
    d1 = []
    for i in range(strings.shape[0] - 1):
        vector = strings[i+1] - strings[i]
        distance = (vector[1]**2 + vector[0]**2)**(1/2)
        v1.append(vector)
        d1.append(distance)

    v1min = v1[np.argmin(d1)]

    # We fill the empty values between two detected strings

    for idx, dist in enumerate(d1):
        if dist > np.min(d1)*1.5:
            vector = strings[idx + 1] - strings[idx]
            strings = np.append(strings, [strings[idx] + vector/2], axis=0)

    # We compute the vectors of the frets

    v2 = []
    d2 = []
    for i in range(frets.shape[0] - 1):
        vector = frets[i+1] - frets[i]
        distance = (vector[1]**2 + vector[0]**2)**(1/2)
        v2.append(vector)
        d2.append(distance)

    v2min = v2[np.argmin(d2)]

    # We fill the empty values between two detected frets

    for idx, dist in enumerate(d2):
        if dist > np.min(d2)*1.5:
            vector = frets[idx + 1] - frets[idx]
            frets = np.append(frets, [frets[idx] + vector/2], axis=0)

    # We fill the empty values of the frets w.r.t. the strings

    first_fret = frets[0]
    last_string = strings[-1]
    while np.abs(first_fret[1] - last_string[1]) > 10:
        first_fret = first_fret - v2min
        frets = np.append(frets, [first_fret], axis=0)

    # We fill the empty values of the strings w.r.t. the frets

    while first_fret[0] - last_string[0] > v1min[1]*1.5:
        last_string = last_string - v1min
        strings = np.append(strings, [last_string], axis=0)

    return frets, strings


directory = 'data/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = HourglassNet(Bottleneck)
model2 = MyModel()
model = nn.Sequential(model, model2)
model = nn.DataParallel(model)
model.to(device)

checkpoint = torch.load('checkpoints/hg_ckpt_98.pth')

model.load_state_dict(checkpoint['model_state_dict'])

for root, dirs, files in os.walk(os.path.join(directory, 'images')):
    for i, file in enumerate(files):
        if file.endswith('.jpg'):
            image = Image.open(os.path.join(root, file))
            image = transforms.ToTensor()(image).type(torch.float32)[:3]
            image = rescale(image, 300).unsqueeze(0).to(device)
            output = model(image)
            output1 = output[0].split(image.shape[0], dim=0)
            output2 = output[1].split(image.shape[0], dim=0)
            output3 = output[2].split(image.shape[0], dim=0)

            max1 = local_max(output1[-1][0][0].cpu().detach().clamp(0, 1).numpy(), 10)
            max1 = max1[max1[:, 0].argsort()]
            max2 = local_max(output2[-1][0][0].cpu().detach().clamp(0, 1).numpy(), 10)
            max2 = max2[(-max2)[:, 1].argsort()]
            max3 = local_max(output3[-1][0][0].cpu().detach().clamp(0, 1).numpy(), 5)
            max3 = max3[(-max3)[:, 0].argsort()]

            fig, ax = plt.subplots(2, 2)
            ax[0][0].imshow(transforms.ToPILImage()(output1[-1][0][0].cpu().detach().clamp(0, 1)), cmap='gray')
            ax[0][1].imshow(transforms.ToPILImage()(output2[-1][0][0].cpu().detach().clamp(0, 1)), cmap='gray')
            ax[1][0].imshow(transforms.ToPILImage()(output3[-1][0][0].cpu().detach().clamp(0, 1)), cmap='gray')
            ax[1][1].imshow(transforms.ToPILImage()(image[0].cpu().detach()))
            ax[1][1].scatter(max1[:, 1], max1[:, 0], s=3)
            ax[1][1].scatter(max2[:, 1], max2[:, 0], s=3)
            ax[1][1].scatter(max3[:, 1], max3[:, 0], s=3)

            frets, strings = fill_values(max1, max2, max3)

            ax[1][1].scatter(frets[:, 1], frets[:, 0], s=3)
            ax[1][1].scatter(strings[:, 1], strings[:, 0], s=3)
            plt.show()
            print(i)