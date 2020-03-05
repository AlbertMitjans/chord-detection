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
import pandas as pd
from utils.utils import AverageMeter


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


def fill_values(frets, strings):

    # We delete the bad values
    if strings.shape[0] > 1 and frets.shape[0] > 1:

        # We compute the vectors of the strings

        v1_all = []
        d1_all = []
        for i in range(strings.shape[0] - 1):
            vector = strings[i+1] - strings[i]
            distance = (vector[1]**2 + vector[0]**2)**(1/2)
            v1_all.append(vector)
            d1_all.append(distance)

        v1_allmin = v1_all[np.argmin(d1_all)]

        # We eliminate more string bad values

        v1 = []
        d1 = []
        strings_2 = []

        for i, val in enumerate(v1_all):
            if np.abs(np.dot(val, v1_allmin))/np.linalg.norm(val)/np.linalg.norm(v1_allmin) >= 0.8:
                v1.append(val)
                d1.append(d1_all[i])
                if strings_2.__len__() == 0:
                    strings_2.append(strings[i])
                strings_2.append(strings[i + 1])

        strings = np.array(strings_2)
        v1min = v1[np.argmin(d1)]

        # We fill the empty values between two detected strings

        for idx, dist in enumerate(d1):
            if dist > np.min(d1)*1.8 and strings.shape[0] < 6:
                vector = strings[idx + 1] - strings[idx]
                strings = np.append(strings, [strings[idx] + vector/2], axis=0)

        strings = strings[(-strings)[:, 0].argsort()]

        # We compute the vectors of the frets

        v2_all = []
        d2_all = []
        for i in range(frets.shape[0] - 1):
            vector = frets[i+1] - frets[i]
            distance = (vector[1]**2 + vector[0]**2)**(1/2)
            v2_all.append(vector)
            d2_all.append(distance)

        v2_allmin = v2_all[np.argmin(d2_all)]

        # We eliminate bad values of frets

        frets_all = frets
        v2 = []
        d2 = []
        frets = []

        for i, val in enumerate(v2_all):
            if np.abs(np.dot(val, v2_allmin))/np.linalg.norm(val)/np.linalg.norm(v2_allmin) >= 0.9:
                v2.append(val)
                d2.append(d2_all[i])
                if frets.__len__() == 0:
                    frets.append(frets_all[i])
                frets.append(frets_all[i + 1])

        frets = np.array(frets)

        v2min = v2[np.argmin(d2)]

        # We fill the empty values between two detected frets

        for idx, dist in enumerate(d2):
            if dist > np.min(d2)*1.5:
                vector = frets[idx + 1] - frets[idx]
                frets = np.append(frets, [frets[idx] + vector/2], axis=0)

        frets = frets[(-frets)[:, 1].argsort()]

        # We fill the empty values of the frets w.r.t. the strings

        first_fret = frets[0]
        last_string = strings[-1]
        while np.abs(np.dot(last_string - first_fret, v2min)) > (np.linalg.norm(v2min)**2)*0.8 and frets.shape[0] < 10:
            first_fret = first_fret - v2min
            frets = np.append(frets, [first_fret], axis=0)

        # We fill the empty values of the strings w.r.t. the frets

        while np.abs(np.dot(first_fret - last_string, v1min)) > (np.linalg.norm(v1min)**2)*1.5 and strings.shape[0] < 6:
            last_string = last_string + v1min
            strings = np.append(strings, [last_string], axis=0)

        # We fill the remaining values of the strings

        first_string = strings[0]
        while strings.shape[0] < 6 and (first_string[0] < 300 and first_string[1] < 300):
            first_string = first_string - v1min
            if first_string[0] < 300 and first_string[1] < 300:
                strings = np.append(strings, [first_string], axis=0)

    else:
        v1min = [-10, 0]
        v2min = [0, -20]

    return frets, strings, v1min, v2min


def make_tab(fingers, frets, strings, v_frets, v_strings):
    tab = np.zeros((15, 6))

    frets = frets[(-frets)[:, 1].argsort()]
    strings = strings[(-strings)[:, 0].argsort()]

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return
    def seg_intersect(a1, a2, b1, b2):
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        return (num / denom.astype(float)) * db + b1

    for finger in fingers:
        point1 = (seg_intersect(frets[0], frets[-1], finger, finger + v_strings))
        point2 = (seg_intersect(strings[0], strings[-1], finger, finger + v_frets))

        fret = frets[:, 1] - point1[1]

        for i, x in enumerate(fret):
            if x < 0:
                if i == fret.shape[0] - 1:
                    f = i - 1
                else:
                    if np.abs(fret[i - 1]) < np.abs(fret[i + 1]):
                        f = i - 1
                    elif np.abs(fret[i - 1]) >= np.abs(fret[i + 1]):
                        f = i
                break
            elif x == 0:
                f = i - 1
                break
            else:
                f = fret.shape[0] - 1

        string = strings[:, 0] - point2[0]
        for i, x in enumerate(string):
            if x < 0:
                if i == 0:
                    s = 1
                elif np.abs(x) <= string[i-1]/3:
                    s = i + 1
                elif np.abs(x) > string[i-1]/3:
                    s = i
                break
            elif x == 0:
                s = i + 1
                break
            else:
                s = 6

        tab[f][-s] = 1

    tab = tab[:np.max(np.where(tab == 1)[0]) + 2]

    return tab


def load_tabs():
    tabs = {}

    chord_dict = pd.read_excel(os.path.join(os.getcwd(), 'data', 'guitar_chords.xlsx'))

    for i in range(chord_dict.shape[0]):
        try:
            num_strings = len(tabs[chord_dict['Chord'][i]])
        except KeyError:
            num_strings = 0

        if num_strings < 6:
            tabs.setdefault(chord_dict['Chord'][i], []).append(chord_dict['Fret'][i])

    return tabs


directory = 'data/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = HourglassNet(Bottleneck)
model2 = MyModel()
model = nn.Sequential(model, model2)
model = nn.DataParallel(model)
model.to(device)

checkpoint = torch.load('checkpoints/hg_ckpt_33.pth')

model.load_state_dict(checkpoint['model_state_dict'])

target_chords = pd.read_csv(os.path.join(os.getcwd(), 'data', 'targets.csv'), header=None).values.tolist()

precision = AverageMeter()

for root, dirs, files in os.walk(os.path.join(directory, 'images')):
    for i, file in enumerate(files):
        if file.endswith('.jpg'):
            image = Image.open(os.path.join(root, file))
            image = transforms.ToTensor()(image).type(torch.float32)[:3]
            image = rescale(image, (300, 300)).unsqueeze(0).to(device)
            output = model(image)
            output1 = output[0].split(image.shape[0], dim=0)
            output2 = output[1].split(image.shape[0], dim=0)
            output3 = output[2].split(image.shape[0], dim=0)

            max1 = local_max(output1[-1][0][0].cpu().detach().clamp(0, 1).numpy(), min_dist=10, t_rel=0.4)
            max1 = max1[max1[:, 0].argsort()]
            max2 = local_max(output2[-1][0][0].cpu().detach().clamp(0, 1).numpy(), min_dist=10, t_rel=0.4)
            max2 = max2[(-max2)[:, 1].argsort()]
            max3 = local_max(output3[-1][0][0].cpu().detach().clamp(0, 1).numpy(), min_dist=5, t_rel=0.4)
            max3 = max3[(-max3)[:, 0].argsort()]

            frets, strings, v_strings, v_frets = fill_values(max2, max3)

            '''fig, ax = plt.subplots(1, 1)
            ax.imshow(transforms.ToPILImage()(image[0].cpu().detach()))
            ax.scatter(max1[:, 1], max1[:, 0], s=3)
            ax.scatter(frets[:, 1], frets[:, 0], s=3)
            ax.scatter(strings[:, 1], strings[:, 0], s=3)'''

            tab = make_tab(max1, frets, strings, v_frets, v_strings)

            target_tab = load_tabs()

            chord_conf = {}

            for chord in target_tab:
                chord_tab = target_tab[chord]
                tabs = np.zeros((np.max(chord_tab) + 1, 6))
                for i, fret in enumerate(chord_tab):
                    if fret != 0:
                        tabs[fret-1][i] = 1

                loc = np.transpose(np.where(tab == 1))

                points = 0.

                for (a, b) in loc:
                    tabs = np.pad(tabs, ((0, max(tab.shape[0] - tabs.shape[0], 0)), (0, 0)))
                    if tabs[a, b] == 1:
                        points += 1/loc.shape[0]
                    elif tabs[a, min(b + 1, 5)] == 1 or tabs[a, max(b - 1, 0)] == 1:
                        points += 0.3/loc.shape[0]
                    elif tabs[min(a + 1, tabs.shape[0]-1), b] == 1 or tabs[max(a - 1, 0), b] == 1:
                        points += 0.1/loc.shape[0]

                chord_conf.setdefault(chord, []).append(points)

            #plt.show()

            img_number = int(os.path.basename(file)[5:-4])

            target_chord = target_chords[img_number - 1][0]

            final_chord = max(chord_conf, key=chord_conf.get)

            score = final_chord == target_chord

            precision.update(score)

            print('{file}:   Target: {chord}  ,  Prediction: {chord2}'.format(file=file, chord=target_chord, chord2=final_chord))

print(precision.avg)

