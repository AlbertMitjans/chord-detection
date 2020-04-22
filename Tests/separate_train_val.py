import numpy as np
import os
import re

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


for root, dirs, files in os.walk('C:\\Users\\Albert\\Desktop\\chord-detection\\data\\train_dataset'):
    files.sort(key=natural_keys)
    for i, file in enumerate(files):
        folder = file[0]
        train.append('data/{folder}/{file}'.format(file=file[2:], folder=folder))

np.savetxt('C:\\Users\\Albert\\Desktop\\chord-detection\\data\\train.txt', np.array(train), fmt='%s')
