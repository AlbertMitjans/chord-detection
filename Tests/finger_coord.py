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

i = 0

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

    lines = []
    nails = []
    knuckles = []

    def onclick_1(event):
        ix, iy = event.xdata, event.ydata
        lines.append([ix, iy])
        ax.axhline(y=iy)
        ax.axvline(x=ix)

    def onclick_2(event):
        ix, iy = event.xdata, event.ydata
        circle = plt.Circle((ix, iy), img.shape[0]/600, color='r')
        ax.add_artist(circle)
        if ix is None:
            stop['stop'] = True
        else:
            nails.append([ix, iy])

    def onclick_3(event):
        ix, iy = event.xdata, event.ydata
        circle = plt.Circle((ix, iy), img.shape[0]/600, color='b')
        ax.add_artist(circle)
        if ix is None:
            knuckles.append([-1, -1])
        else:
            knuckles.append([ix, iy])

    '''for a in range(2):
        o1 = fig.canvas.mpl_connect('button_press_event', onclick_1)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o1)

    plt.xlim(lines[0][0], lines[1][0])
    plt.ylim(lines[1][1], lines[0][1])'''

    stop = {'stop': False}

    while True:
        o2 = fig.canvas.mpl_connect('button_press_event', onclick_2)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o2)
        if stop['stop']:
            break

    '''for j in range(4):
        o3 = fig.canvas.mpl_connect('button_press_event', onclick_3)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o3)'''

    plt.close('all')

    #np.savetxt(os.path.join(directory, 'image{top1}_fingers.csv'.format(top1=i)), np.asarray(nails), delimiter=',', fmt='%.3f')
    np.savetxt(os.path.join(directory, 'image{top1}_frets.csv'.format(top1=i)), np.asarray(nails), delimiter=',',
               fmt='%.3f')
    '''np.savetxt(os.path.join(directory, 'image{top1}_knuckles.csv'.format(top1=i)), np.asarray(knuckles), delimiter=',',
               fmt='%.3f')'''

    i += 1
