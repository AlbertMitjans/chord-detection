import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

directory = os.getcwd()

while True:
    i = np.random.randint(1, 224, dtype=int)
    try:
        im = Image.open(os.path.abspath(os.path.join(directory, '..', 'dataset/images/image' + str(i+1) + '.jpg')))
    except OSError:
        break
    edges = im.convert('RGB').filter(ImageFilter.FIND_EDGES)
    contours = im.convert('RGB').filter(ImageFilter.CONTOUR)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im)
    ax[1].imshow(edges)
    ax[2].imshow(contours)
    plt.show()
    plt.waitforbuttonpress()
    plt.close('all')



