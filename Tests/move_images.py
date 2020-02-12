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


directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))

for root, dirs, files in os.walk(os.path.join(directory, 'images')):
    files.sort(key=natural_keys)
    for i, file in enumerate(files):
        if file.endswith('.jpg'):
            x = np.random.random()
            if x < 0.8:
                shutil.copy(os.path.join(directory, 'images', file), os.path.join(directory, 'train_dataset', file))
                shutil.copy(os.path.join(directory, 'images', os.path.splitext(file)[0] + '_fingers.csv'), os.path.join(directory, 'train_dataset', os.path.splitext(file)[0] + '_fingers.csv'))
                shutil.copy(os.path.join(directory, 'images', os.path.splitext(file)[0] + '_hand.csv'), os.path.join(directory, 'train_dataset', os.path.splitext(file)[0] + '_hand.csv'))
            else:
                shutil.copy(os.path.join(directory, 'images', file), os.path.join(directory, 'val_dataset', file))
                shutil.copy(os.path.join(directory, 'images', os.path.splitext(file)[0] + '_fingers.csv'),
                            os.path.join(directory, 'val_dataset', os.path.splitext(file)[0] + '_fingers.csv'))
                shutil.copy(os.path.join(directory, 'images', os.path.splitext(file)[0] + '_hand.csv'),
                            os.path.join(directory, 'val_dataset', os.path.splitext(file)[0] + '_hand.csv'))


