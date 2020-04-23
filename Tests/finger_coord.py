import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data/1'))

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=directory, help="path of the directory where the images are")
parser.add_argument("--display_time", type=int, default=3, help="time of the display of the RGB image with the "
                                                                "colored circles")
opt = parser.parse_args()

i = 4

try:
    path = os.path.abspath(opt.path)
except AttributeError:
    print('Write a valid directory name')
    exit()


while True:
    try:
        img = plt.imread(os.path.join(opt.path, 'image{top1}.JPG'.format(top1=i)))
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

    fingers = []
    strings = []
    frets = []
    hands = []

    def onclick_1(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            fingers.append([-1, -1])
        else:
            fingers.append([ix, iy])
        ax.scatter(ix, iy, c='red')
        plt.draw()

    def onclick_2(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            strings.append([-1, -1])
        else:
            strings.append([ix, iy])
        ax.scatter(ix, iy, c='blue')
        plt.draw()

    def onclick_3(event):
        ix, iy = event.xdata, event.ydata
        if ix is None and iy is None:
            frets.append([-1, -1])
        else:
            frets.append([ix, iy])
        ax.scatter(ix, iy, c='green')
        plt.draw()

    def onclick_5(event):
        ix, iy = event.xdata, event.ydata
        hands.append([ix, iy])
        ax.scatter(ix, iy, c='purple')
        plt.draw()

    '''def onclick_2(event):
        ix, iy = event.xdata, event.ydata
        circle = plt.Circle((ix, iy), img.shape[0]/600, color='r')
        ax.add_artist(circle)
        if ix is None:
            stop['stop'] = True
        else:
            nails.append([ix, iy])'''


    for a in range(2):
        o5 = fig.canvas.mpl_connect('button_press_event', onclick_5)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o5)
        plt.draw()

    im = img[int(hands[0][1]):int(hands[1][1]), int(hands[0][0]):int(hands[1][0])]

    fig, ax = plt.subplots(figsize=(80, 80))
    fig.suptitle('Image {top1}'.format(top1=i))
    plt.imshow(im)

    '''for a in range(4):
        o1 = fig.canvas.mpl_connect('button_press_event', onclick_1)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o1)

    for a in range(6):
        o2 = fig.canvas.mpl_connect('button_press_event', onclick_2)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o2)

    for a in range(10):
        o3 = fig.canvas.mpl_connect('button_press_event', onclick_3)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o3)'''

    '''stop = {'stop': False}

    while True:
        o2 = fig.canvas.mpl_connect('button_press_event', onclick_2)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(o2)
        if stop['stop']:
            break'''
    plt.waitforbuttonpress()
    plt.close('all')

    '''np.savetxt(os.path.join(directory, 'image{top1}_fingers.csv'.format(top1=i)), np.asarray(fingers), delimiter=',', fmt='%.3f')
    np.savetxt(os.path.join(directory, 'image{top1}_strings.csv'.format(top1=i)), np.asarray(strings), delimiter=',',
                fmt='%.3f')
    np.savetxt(os.path.join(directory, 'image{top1}_frets.csv'.format(top1=i)), np.asarray(frets), delimiter=',',
                fmt='%.3f')'''
    np.savetxt(os.path.join(directory, 'image{top1}_yolo2.csv'.format(top1=i)), np.asarray(hands), delimiter=',',
               fmt='%.3f')

    i += 1
