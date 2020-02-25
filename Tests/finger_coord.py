import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'images'))

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=directory, help="path of the directory where the images are")
parser.add_argument("--display_time", type=int, default=3, help="time of the display of the RGB image with the "
                                                                "colored circles")
opt = parser.parse_args()

i = 211

try:
    path = os.path.abspath(opt.path)
except AttributeError:
    print('Write a valid directory name')
    exit()


while True:
    try:
        img = plt.imread(os.path.join(opt.path, 'image{top1}.jpg'.format(top1=i)))
    except FileNotFoundError:
        print('png image')
        try:
            img = plt.imread(os.path.join(opt.path, 'image{top1}.png'.format(top1=i)))
        except FileNotFoundError:
            print('Image {num} not found'.format(num=i))
            i += 1
            continue

    fig, ax = plt.subplots(figsize=(80, 80))
    fig.suptitle('Image {top1}'.format(top1=i))
    plt.imshow(img)

    tip = []
    knuckle1 = []
    knuckle2 = []
    notes = []

    def onclick_1(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            tip.append([-1, -1])
        else:
            tip.append([ix, iy])
        ax.scatter(ix, iy, c='red')

    def onclick_2(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            knuckle1.append([-1, -1])
        else:
            knuckle1.append([ix, iy])
        ax.scatter(ix, iy, c='blue')

    def onclick_3(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            knuckle2.append([-1, -1])
        else:
            knuckle2.append([ix, iy])
        ax.scatter(ix, iy, c='green')

    def onclick_5(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            notes.append([-1, -1])
        else:
            notes.append([ix, iy])
        ax.scatter(ix, iy, c='purple')

    '''def onclick_2(event):
        ix, iy = event.xdata, event.ydata
        circle = plt.Circle((ix, iy), img.shape[0]/600, color='r')
        ax.add_artist(circle)
        if ix is None:
            stop['stop'] = True
        else:
            nails.append([ix, iy])'''

    for a in range(4):
        o1 = fig.canvas.mpl_connect('button_press_event', onclick_1)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o1)

    for a in range(4):
        o2 = fig.canvas.mpl_connect('button_press_event', onclick_2)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o2)

    for a in range(4):
        o3 = fig.canvas.mpl_connect('button_press_event', onclick_3)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o3)

    for a in range(4):
        o5 = fig.canvas.mpl_connect('button_press_event', onclick_5)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o5)

    '''stop = {'stop': False}

    while True:
        o2 = fig.canvas.mpl_connect('button_press_event', onclick_2)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o2)
        if stop['stop']:
            break'''

    plt.close('all')

    np.savetxt(os.path.join(directory, 'image{top1}_tip.csv'.format(top1=i)), np.asarray(tip), delimiter=',', fmt='%.3f')
    np.savetxt(os.path.join(directory, 'image{top1}_knuckle1.csv'.format(top1=i)), np.asarray(knuckle1), delimiter=',',
                fmt='%.3f')
    np.savetxt(os.path.join(directory, 'image{top1}_knuckle2.csv'.format(top1=i)), np.asarray(knuckle2), delimiter=',',
                fmt='%.3f')
    np.savetxt(os.path.join(directory, 'image{top1}_notes.csv'.format(top1=i)), np.asarray(notes), delimiter=',',
               fmt='%.3f')

    i += 1
