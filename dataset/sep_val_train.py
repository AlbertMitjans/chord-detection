import pandas as pd
import os
import numpy as np
import shutil
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


directory = os.getcwd()

chords = pd.read_csv('targets.txt', header=None)

train_labels = []
val_labels = []

for root, dirs, files in os.walk(os.path.join(directory, 'images')):
    files.sort(key=natural_keys)
    for i, file in enumerate(files):
        x = np.random.random()
        if x < 0.8:
            shutil.move(os.path.join(directory, 'images', file), os.path.join(directory, 'train_dataset', file))
            train_labels.append(chords[0][i])
        else:
            shutil.move(os.path.join(directory, 'images', file), os.path.join(directory, 'val_dataset', file))
            val_labels.append(chords[0][i])

np.savetxt(os.path.join(directory, 'train_dataset', 'labels.txt'), train_labels, delimiter=',', fmt='%s')
np.savetxt(os.path.join(directory, 'val_dataset', 'labels.txt'), val_labels, delimiter=',', fmt='%s')