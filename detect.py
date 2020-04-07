from models.Stacked_Hourglass import Bottleneck, HourglassNet
from models.my_model import MyModel
import os
from PIL import Image
import torchvision.transforms as transforms
from utils.img_utils import local_max
import pandas as pd
from utils.utils import AverageMeter
from models.yolo import *
from utils.yolo_utils import *
from utils.img_utils import rescale
import random


def fill_values(frets, strings):

    # We delete the bad values
    if strings.shape[0] > 1 and frets.shape[0] > 1:

        # We delete bad values of strings
        deviation = np.abs(np.mean(strings[:, 1]) - strings[:, 1])
        std = np.std(strings, axis=0)[1]
        strings = strings[np.where(deviation <= std*2)]

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

        d1 = []
        strings_2 = []

        for i, val in enumerate(v1_all):
            if np.abs(np.dot(val, v1_allmin))/np.linalg.norm(val)/np.linalg.norm(v1_allmin) >= 0.8:
                d1.append(d1_all[i])
                if strings_2.__len__() == 0:
                    strings_2.append(strings[i])
                strings_2.append(strings[i + 1])

        strings = np.array(strings_2)

        # We fill the empty values between two detected strings

        for idx, dist in enumerate(d1):
            if dist > np.min(d1)*2.8 and strings.shape[0] < 6:
                vector = strings[idx + 1] - strings[idx]
                strings = np.append(strings, [strings[idx] + vector / 3], axis=0)
                strings = np.append(strings, [strings[idx] + 2 * vector / 3], axis=0)

            elif dist > np.min(d1)*1.8 and strings.shape[0] < 6:
                vector = strings[idx + 1] - strings[idx]
                strings = np.append(strings, [strings[idx] + vector/2], axis=0)

        strings = strings[(-strings)[:, 0].argsort()]

        # We compute the mean of the vectors of the strings

        v1_all = []
        d1_all = []
        for i in range(strings.shape[0] - 1):
            vector = strings[i + 1] - strings[i]
            distance = (vector[1] ** 2 + vector[0] ** 2) ** (1 / 2)
            v1_all.append(vector)
            d1_all.append(distance)

        v1_mean = np.mean(v1_all, axis=0)

        # We eliminate bad values of frets
        frets_all = frets
        frets = []

        for val in frets_all:
            if val[1] < np.min(strings, axis=0)[1]:
                frets.append(val)

        frets = np.array(frets)

        # We compute the vectors of the frets

        v2_all = []
        d2_all = []
        for i in range(frets.shape[0] - 1):
            vector = frets[i+1] - frets[i]
            if np.abs(vector[0]) > np.abs(vector[1]):
                vector = [999, 999]
            distance = (vector[1]**2 + vector[0]**2)**(1/2)
            v2_all.append(vector)
            d2_all.append(distance)

        v2_allmin = v2_all[np.argmin(d2_all)]

        # We eliminate more bad values of frets

        frets_all = frets
        d2 = []
        frets = []

        for i, val in enumerate(v2_all):
            if np.abs(np.dot(val, v2_allmin))/np.linalg.norm(val)/np.linalg.norm(v2_allmin) >= 0.9:
                d2.append(d2_all[i]*(1 + i/10))
                if frets.__len__() == 0:
                    frets.append(frets_all[i])
                frets.append(frets_all[i + 1])

        frets = np.array(frets)

        # We fill the empty values between two detected frets

        for idx, dist in enumerate(d2):
            if dist >= 2.55*np.min(d2):
                vector = frets[idx + 1] - frets[idx]
                frets = np.append(frets, [frets[idx] + vector / 3], axis=0)
                frets = np.append(frets, [frets[idx] + 2 * vector / 3], axis=0)
            elif dist >= 1.5*np.min(d2):
                vector = frets[idx + 1] - frets[idx]
                frets = np.append(frets, [frets[idx] + vector/2], axis=0)

        frets = frets[(-frets)[:, 1].argsort()]

        # We compute the mean of the vectors of the frets

        v2_all = []
        d2_all = []
        for i in range(frets.shape[0] - 1):
            vector = frets[i + 1] - frets[i]
            distance = (vector[1] ** 2 + vector[0] ** 2) ** (1 / 2)
            v2_all.append(vector)
            d2_all.append(distance)

        v2_mean = np.mean(v2_all, axis=0)

        # We fill the missing highest fret values

        last_fret = frets[-1]
        i = 0
        while (0 < last_fret[0] < 300 and 0 < last_fret[1] < 300):
            last_fret = last_fret + (v2_all[-1]*(1 - i/10))
            if 0 < last_fret[0] < 300 and 0 < last_fret[1] < 300:
                frets = np.append(frets, [last_fret], axis=0)
                i += 1

        # We fill the empty values of the frets w.r.t. the strings

        first_fret = frets[0]
        last_string = strings[-1]
        i = 0
        while np.abs(np.dot(last_string - first_fret, v2_mean)) > (np.linalg.norm(v2_mean)**2)*0.8 and frets.shape[0] < 10:
            first_fret = first_fret - (v2_all[0]*(1 + i/10))
            frets = np.append(frets, [first_fret], axis=0)
            i += 1

        # We fill the empty values of the strings w.r.t. the frets

        while np.abs(np.dot(first_fret - last_string, v1_mean)) > (np.linalg.norm(v1_mean)**2)*0.7 and strings.shape[0] < 6:
            last_string = last_string + v1_mean
            strings = np.append(strings, [last_string], axis=0)

        # We fill the remaining values of the strings

        first_string = strings[0]
        while strings.shape[0] < 6 and (first_string[0] < 300 and first_string[1] < 300):
            first_string = first_string - v1_mean
            if first_string[0] < 300 and first_string[1] < 300:
                strings = np.append(strings, [first_string], axis=0)

    else:
        v1_mean = [-10, 0]
        v2_mean = [0, -20]

    return frets, strings, v1_mean, v2_mean


def make_tab(fingers, frets, strings, v_frets, v_strings, ax):
    tab = np.zeros((15, 6))

    frets = frets[(-frets)[:, 1].argsort()]
    strings = strings[(-strings)[:, 0].argsort()]
    fingers = fingers[(-fingers)[:, 1].argsort()]

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

    for idx, finger in enumerate(fingers):
        point1 = (seg_intersect(frets[0], frets[-1], finger, finger + v_strings))
        point2 = (seg_intersect(strings[0], strings[-1], finger, finger + v_frets))

        ax.scatter(point1[1], point1[0], c='r', s=3)
        ax.scatter(point2[1], point2[0], c='r', s=3)

        string = strings[:, 0] - point2[0]

        for i, x in enumerate(string):
            if x < 0:
                if i == 0:
                    s = 1
                elif np.abs(x) <= string[i-1]/2:
                    s = i + 1
                elif np.abs(x) > string[i-1]/2:
                    s = i
                break
            elif x == 0:
                s = i + 1
                break
            else:
                s = 6

        fret = frets[:, 1] - point1[1]

        for i, x in enumerate(fret):
            if x <= 0:
                if fret[i - 1] <= np.abs(x)/7:
                    f = i - 2
                elif s == 6 and fret[i - 1] <= np.abs(x) and idx == 0:
                    f = i - 2
                else:
                    f = i - 1
                break
            else:
                f = i - 1

        tab[f][-s] = idx + 1

    if np.where(tab==1)[1] == 0:
        for i in range(6):
            if np.max(tab[:, i]) == 0:
                tab[np.where(tab==1)[0][0]][i] = 1

    tab = tab[:np.max(np.where(tab != 0)[0]) + 2]

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


def detect_chord(image, yolo, model, device):
    image = transforms.ToTensor()(image).type(torch.float32)[:3]
    image = rescale(image, (416, 416)).unsqueeze(0).to(device)
    detections = yolo(image)
    detections = non_max_suppression(detections, conf_thres=0.5, nms_thres=0.01)

    img = image.cpu().detach()[0]

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(transforms.ToPILImage()(img))
    ax.axis('off')

    # Draw bounding boxes and labels of detections
    if detections .__len__() != 0:
        detections = torch.cat([d for d in detections])
        # Rescale boxes to original image
        detections = rescale_boxes(detections, 416, image.shape[2:])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % ('Hand', cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
    plt.savefig('full_image.png')
    plt.show()

    if detections[0] is not None:

        #detections = torch.cat([d for d in detections])

        if detections.shape[0] > 1:
            if detections[0][0] > detections[1][0]:
                detect = detections[0][:4] + torch.Tensor([-10, -40, +40, 0])
            else:
                detect = detections[1][:4] + torch.Tensor([-10, -40, +40, 0])
        else:
            detect = detections[0][:4] + torch.Tensor([-10, -40, +40, 0])
        detect = detect.type(torch.int)

        img = img[:, detect[1].item():detect[3].item(), detect[0].item():detect[2].item()]
        img = rescale(img, (300, 300))

        plt.figure()
        plt.imshow(transforms.ToPILImage()(img))
        plt.axis('off')
        plt.savefig('hand.png')
        plt.show()

        img = img.unsqueeze(0).to(device)

        output = model(img)
        output1 = output[0].split(image.shape[0], dim=0)
        output2 = output[1].split(image.shape[0], dim=0)
        output3 = output[2].split(image.shape[0], dim=0)

        max1 = local_max(output1[-1][0][0].cpu().detach().numpy(), min_dist=10, t_rel=0.6)
        max1 = max1[max1[:, 0].argsort()]
        if max1.shape[0] > 1:
            if max1[1][0] - max1[0][0] >= 70:  # We delete the thumb
                max1 = max1[1:]
        max2 = local_max(output2[-1][0][0].cpu().detach().numpy(), min_dist=10, t_rel=0.5)
        max2 = max2[(-max2)[:, 1].argsort()]
        max3 = local_max(output3[-1][0][0].cpu().detach().numpy(), min_dist=5, t_rel=0.4)
        max3 = max3[(-max3)[:, 0].argsort()]

        frets, strings, v_strings, v_frets = fill_values(max2, max3)

        fig, ax = plt.subplots(2, 2)
        ax[1][1].imshow(transforms.ToPILImage()(img[0].cpu().detach()))
        ax[1][1].scatter(max1[:, 1], max1[:, 0], s=3)
        ax[1][1].scatter(frets[:, 1], frets[:, 0], s=3)
        ax[1][1].scatter(strings[:, 1], strings[:, 0], s=3)
        ax[0][0].imshow(transforms.ToPILImage()(output1[-1][0][0].clamp(0, 1).cpu().detach()))
        ax[0][1].imshow(transforms.ToPILImage()(output2[-1][0][0].clamp(0, 1).cpu().detach()))
        ax[1][0].imshow(transforms.ToPILImage()(output3[-1][0][0].clamp(0, 1).cpu().detach()))
        ax[0][0].axis('off')
        ax[0][1].axis('off')
        ax[1][0].axis('off')
        ax[1][1].axis('off')

        tab = make_tab(max1, frets, strings, v_frets, v_strings, ax[1][1])

        plt.savefig('detections.png')

        plt.show()

        target_tab = load_tabs()

        chord_conf = {}

        tab[np.where(tab==3.0)] = 0
        tab[2, 0] = 3

        for chord in target_tab:
            chord_tab = target_tab[chord]
            tabs = np.zeros((np.max(chord_tab) + 1, 6))
            for i, fret in enumerate(chord_tab):
                if fret != 0:
                    tabs[fret - 1][i] = 1

            loc = np.transpose(np.where(tab != 0))

            points = 0.

            # Penalty for difference in number of fingers
            points -= np.abs(np.where(tabs != 0)[0].shape[0] - np.where(tab != 0)[0].shape[0]) / loc.shape[0] / 5

            for (a, b) in loc:
                tabs = np.pad(tabs, ((0, max(tab.shape[0] - tabs.shape[0], 0)), (0, 0)))
                if tabs[a, b] != 0:
                    points += 1 / loc.shape[0]
                elif tabs[a, min(b + 1, 5)] != 0 or tabs[a, max(b - 1, 0)] != 0:
                    points += 0.3 / loc.shape[0]
                else:
                    points -= 1 / loc.shape[0]  # Penalty for not having this finger position

                '''elif tabs[min(a + 1, tabs.shape[0]-1), b] != 0 or tabs[max(a - 1, 0), b] != 0:
                    points += 0.01/loc.shape[0]'''

            chord_conf.setdefault(chord, []).append(max(0, int(points * 100)))

        final_chord = max(chord_conf, key=chord_conf.get)

    elif detections[0] is None:
        final_chord = None
        tab = None
        chord_conf = None

    return final_chord, tab, chord_conf


def load_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    yolo = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
    yolo.load_state_dict(torch.load('checkpoints/best_ckpt/yolo.pth', map_location=device))
    yolo.eval()

    model = HourglassNet(Bottleneck)
    model2 = MyModel()
    model = nn.Sequential(model, model2)
    model = nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load('checkpoints/best_ckpt/best.pth')

    model.load_state_dict(checkpoint['model_state_dict'])

    return yolo, model, device


if __name__ == "__main__":
    yolo, model, device = load_models()

    target_chords = pd.read_excel(os.path.join(os.getcwd(), 'data/my_data', 'targets.xlsx'), header=None).values.tolist()

    precision = AverageMeter()

    directory = 'data/my_data/images'

    for root, dirs, files in os.walk(directory):
        for i, file in enumerate(files):
            if file.endswith('.JPG') and not file.startswith('00'):
                image = Image.open(os.path.join(root, file))

                final_chord, tab, chord_conf = detect_chord(image, yolo, model, device=device)

                img_number = int(os.path.basename(file)[5:-4])

                target_chord = target_chords[img_number - 1][0]

                score = final_chord == target_chord

                precision.update(score)

                if target_chord != final_chord:
                    print(tab)

                    print(
                        '{file}:   Target: {chord}  ,  Prediction: {chord2} ({perc}%)'.format(file=file, chord=target_chord,
                                                                                              chord2=final_chord,
                                                                                              perc=chord_conf[final_chord][
                                                                                                  0]))

                    plt.show()

                    print('---------------------------------------------------------------')

                plt.close('all')
                print(precision.avg)

