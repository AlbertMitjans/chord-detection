from PIL import Image
import os

directory = os.getcwd()

for i in range(250):
    try:
        im = Image.open(os.path.abspath(os.path.join(directory, '..', 'dataset/images/image' + str(i+1) + '.jpg')))
    except OSError:
        print(i+1)