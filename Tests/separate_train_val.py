import numpy as np
import os
import re
import random

train = []
val = []


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


for root, dirs, files in os.walk('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\images'):
    files.sort(key=natural_keys)
    for i, file in enumerate(files):
        if file.endswith('jpg'):
            x = random.random()
        if x < 0.8:
            train.append('data/custom/images/{file}'.format(file=file))
        elif x > 0.8:
            val.append('data/custom/images/{file}'.format(file=file))


'''for root, dirs, files in os.walk('C:\\Users\\Albert\\Desktop\\chord-detection\\data\\1'):
    files.sort(key=natural_keys)
    for i, file in enumerate(files):
        if file.endswith('jpg'):
            x = random.random()
        if x < 0.8:
            train.append('data/1/{file}'.format(file=file))
        elif x > 0.8:
            val.append('data/1/{file}'.format(file=file))'''

np.savetxt('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\train.txt', np.array(train), fmt='%s')
np.savetxt('C:\\Users\\Albert\\Desktop\\yolo\\data\\custom\\val.txt', np.array(val), fmt='%s')
