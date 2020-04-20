import cv2
from matplotlib.animation import FuncAnimation
from detect import *
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

yolo, model, device = load_models()

white_image = Image.fromarray(np.full((230, 620), 255, dtype=np.uint8))


def update(i):
    global detection, average_detection, current_chord
    ret, frame = vid.read()

    if ret is False:
        video.release()
        ani.event_source.stop()
        plt.close('all')

    elif ret is True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        final_chord, tab, chord_conf, cropped_img, output_img = detect_chord(frame, yolo, model, device=device)

        for idx, chord in vid_chords:
            if i == idx:
                current_chord = chord

        if chord_conf is not None:
            if i % 5 == 0:
                if i == 0:
                    average_detection = chord_conf.copy()
                elif i != 0:
                    for item in detection:
                        average_detection[item] = [detection[item][0] / 6]

                detection = chord_conf.copy()

            if i != 0:
                for item in detection:
                    detection[item] = [detection[item][0] + chord_conf[item][0]]

            # we convert the images to uint8
            cropped_img = cropped_img.numpy()
            cropped_img = (cropped_img - cropped_img.min())/(cropped_img.max() - cropped_img.min())
            cropped_img = (cropped_img*255).astype(np.uint8).transpose((1, 2, 0))
            cropped_img = Image.fromarray(cropped_img)
            output_img = output_img.cpu().detach().numpy()
            output_img = (output_img - output_img.min())/(output_img.max() - output_img.min())
            output_img = (output_img*255).astype(np.uint8)
            output_img = Image.fromarray(output_img)

            c = Counter(average_detection)
            mc = c.most_common(1)

            frame = Image.fromarray(frame)
            frame.paste(cropped_img, (1100, 100))
            frame.paste(output_img, (1400, 100))
            frame.paste(white_image, (1200, 750))
            d = ImageDraw.Draw(frame)
            font = ImageFont.truetype("arial.ttf", 50)
            d.text((1380, 800), 'Chord: {chord}'.format(chord=current_chord),
                   fill=(255, 0, 0), font=font)
            d.text((1250, 850), 'Prediction: {chord} ({conf} %)'.format(chord=mc[0][0], conf=round(mc[0][1][0], 2)),
                   fill=(255, 0, 0), font=font)

            frame = np.array(frame)

            im.set_data(frame)

            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# Set up formatting for the movie files

directory = os.getcwd()

vid = cv2.VideoCapture()
vid.open(os.path.join(directory, 'data/guitarra.mp4'))

ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

global detection, average_detection, current_chord
detection = []
average_detection = []
vid_chords = [[0, 'C'], [60, 'G'], [135, 'Am'], [205, 'F'], [250, 'G'], [285, 'C'], [355, 'G'], [435, 'Am'], [510, 'F'],
              [540, 'G'], [590, 'C']]
current_chord = 'C'

# Create plot
fig, ax = plt.subplots(1, 1)
ax.axis('off')
ax.set_title('Video')
im = ax.imshow(frame)

video = cv2.VideoWriter('main_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080), True)

ani = FuncAnimation(fig, update, interval=1000/30, cache_frame_data=False)

plt.show()
