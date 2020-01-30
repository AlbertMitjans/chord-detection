import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset', 'images'))

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=directory, help="path of the directory where the images are")
parser.add_argument("--display_time", type=int, default=3, help="time of the display of the RGB image with the "
                                                                "colored circles")
opt = parser.parse_args()

i = 1

try:
    path = os.path.abspath(opt.path)
except AttributeError:
    print('Write a valid directory name')
    exit()


while True:
    try:
        img = plt.imread(os.path.join(opt.path, 'image{top1}.jpg'.format(top1=i)))
    except FileNotFoundError:
        break
    fig, ax = plt.subplots()
    fig.suptitle('Image {top1}'.format(top1=i))
    plt.imshow(img)

    fingers = []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        circle = plt.Circle((ix, iy), img.shape[0]/50, color='r')
        ax.add_artist(circle)
        if ix is None:
            fingers.append([-1, -1])
        else:
            fingers.append([ix, iy])

    for j in range(4):
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.waitforbuttonpress()

    plt.close('all')

    np.savetxt(os.path.join(directory, 'image{top1}.csv'.format(top1=i)), np.asarray(fingers), delimiter=',', fmt='%.3f')

    i += 1
