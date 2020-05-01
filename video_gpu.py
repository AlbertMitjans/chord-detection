import cv2
from matplotlib.animation import FuncAnimation
from detect import *
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from utils.utils import AverageMeter

yolo, model, device = load_models()

white_image = Image.fromarray(np.full((230, 620), 255, dtype=np.uint8))


# Set up formatting for the movie files

directory = os.getcwd()

vid = cv2.VideoCapture()
vid.open(os.path.join(directory, 'data/video7_albert.mov'))
num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

average_detection = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),
                     AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),
                     AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),
                     AverageMeter(), AverageMeter()]
detection = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vid_chords = np.array(pd.read_excel(os.path.join(os.getcwd(), 'data', 'video4_albert_chords.xlsx'), header=None).values.tolist())
current_chord = vid_chords[0][1]

video = cv2.VideoWriter('albert7.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080), True)

for i in range(num_frames):
    ret, frame = vid.read()

    if ret is False:
        video.release()

    elif ret is True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        final_chord, final_chord_conf, _, chord_conf, cropped_img, output_img = detect_chord(frame, yolo, model,
                                                                                             device=device)

        for idx, chord in vid_chords:
            if i == int(idx):
                current_chord = chord

        if chord_conf is not None:
            for idx, value in enumerate(chord_conf.values()):
                average_detection[idx].update(value[0])
                if i % 5 == 0:
                    detection[idx] = average_detection[idx].avg
                    if i != 0:
                        average_detection[idx].reset()

            # we convert the images to uint8
            cropped_img = cropped_img.numpy()
            cropped_img = (cropped_img - cropped_img.min()) / (cropped_img.max() - cropped_img.min())
            cropped_img = (cropped_img * 255).astype(np.uint8).transpose((1, 2, 0))
            cropped_img = Image.fromarray(cropped_img)
            output_img = output_img.cpu().detach().numpy()
            output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
            output_img = (output_img * 255).astype(np.uint8)
            output_img = Image.fromarray(output_img)

            c = np.max(detection)

            if c < 10:
                result = 'None'
            else:
                mc = list(chord_conf.keys())[np.where(detection == c)[0][0]]
                result = ''.join(i for i in mc if not i.isdigit())

            frame = Image.fromarray(frame)
            frame.paste(cropped_img, (1100, 100))
            frame.paste(output_img, (1400, 100))
            frame.paste(white_image, (1200, 750))
            d = ImageDraw.Draw(frame)
            d.text((1380, 800), 'Chord: {chord}'.format(chord=current_chord),
                   fill=(255, 0, 0))
            d.text((1250, 850), 'Prediction: {chord} ({conf} %)'.format(chord=result, conf=round(c, 2)),
                   fill=(255, 0, 0))

            frame = np.array(frame)

            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            print(i, final_chord, final_chord_conf, '%')