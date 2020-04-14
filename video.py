import cv2
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from detect import *

yolo, model, device = load_models()


def update(i):
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    final_chord, tab, chord_conf = detect_chord(frame, yolo, model, device=device)
    conf = chord_conf[final_chord][0]
    if conf == 0:
        final_chord = None
    im.set_data(frame)
    title.set_text(str(final_chord))
    print('{final_chord}: {conf}%'.format(final_chord=str(final_chord), conf=conf))

# Set up formatting for the movie files
writer = animation.FFMpegWriter(fps=5)

directory = os.getcwd()

vid = cv2.VideoCapture()
vid.open(os.path.join(directory, 'data/video.mp4'))

ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#Create plot
fig, ax = plt.subplots(1, 1)
ax.axis('off')
ax.set_title('Video')
im = ax.imshow(frame)
title = ax.text(0.5, 0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

ani = FuncAnimation(fig, update, interval=1000/1, save_count=1000)

def action(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

cid = plt.gcf().canvas.mpl_connect("key_press_event", action)

plt.show()

'''# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()'''

