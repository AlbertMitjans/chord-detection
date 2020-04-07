import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data/my_data', 'images'))
i = 1

while True:
    png = False
    try:
        csv = pd.read_csv(os.path.join(directory, 'image{num}_hand.csv'.format(num=i)), header=None).values
        img = plt.imread(os.path.join(directory, 'image{top1}.JPG'.format(top1=i)))
    except FileNotFoundError:
        print('PNG file')
        png = True
        try:
            csv = pd.read_csv(os.path.join(directory, 'image{num}_hand.csv'.format(num=i)), header=None).values
            img = plt.imread(os.path.join(directory, 'image{top1}.png'.format(top1=i)))
        except FileNotFoundError:
            i += 1
            continue

    height = csv[1, 1] - csv[0, 1]
    width = csv[1, 0] - csv[0, 0]

    x_center = csv[0, 0] + width/2.
    y_center = csv[0, 1] + height/2.

    text = np.array((0, x_center/img.shape[1], y_center/img.shape[0], width/img.shape[1], height/img.shape[0]))

    np.savetxt('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\labels\\image{num}.txt'.format(num=i), text, newline=' ', fmt='%.10f')

    '''if png:
        plt.imsave('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\images\\image{num}.png'.format(num=i), img)
    if not png:
        plt.imsave(
            'C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\images\\image{num}.jpg'.format(num=i), img)'''

    i += 1
