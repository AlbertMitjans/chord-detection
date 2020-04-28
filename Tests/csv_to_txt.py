import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', '2'))
i = 304
j = 1

while True:
    csv = pd.read_csv(os.path.join(directory, 'image{num}_yolo.csv'.format(num=j)), header=None).values
    img = plt.imread(os.path.join(directory, 'image{top1}.jpg'.format(top1=j)))
    j += 1

    height = csv[1, 1] - csv[0, 1]
    width = csv[1, 0] - csv[0, 0]

    x_center = csv[0, 0] + width/2.
    y_center = csv[0, 1] + height/2.

    text = np.array((0, x_center/img.shape[1], y_center/img.shape[0], width/img.shape[1], height/img.shape[0]))

    np.savetxt('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\labels\\image{num}.txt'.format(num=i), text, newline=' ', fmt='%.10f')

    #plt.imsave('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\images\\image{num}.jpg'.format(num=i), img)

    print(i)

    i += 1
