import cv2
from detect import *
from PIL import Image, ImageDraw, ImageFont

yolo, model, device = load_models()

white_image = Image.fromarray(np.full((230, 620), 255, dtype=np.uint8))


# Set up formatting for the movie files

directory = os.getcwd()

vid = cv2.VideoCapture()
vid.open(os.path.join(directory, 'data/videos/video21.mov'))
num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(num_frames):
    ret, frame = vid.read()

    if ret is True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        final_chord, final_chord_conf, _, chord_conf, cropped_img, output_img = detect_chord(frame, yolo, model,
                                                                                             device=device)
        print(i, final_chord, final_chord_conf, '%')