# Chord detection
This repo contains the code structure for the detection of guitar chords from input images.

## Installation

**Create conda environment**
```
$ conda create -n ENVIRONMENT_NAME python=3
$ conda activate ENVIRONMENT_NAME
```

**Clona and install requirements**
```
$ git clone https://github.com/AlbertMitjans/chord-detection.git
$ cd chord-detection/
$ conda install --file requirements.txt
```
**Download pretrained weights**
```
$ cd checkpoints/
$ bash get_weights.sh
```

## Hourglass network

### Run test

Evaluates the model on the dataset. The network outputs 3 heatmaps with the position of the detected frets, strings and fingers.

```
$ python3 main.py --train False --ckpt checkpoints/best_ckpt/model.pth
```

As default, for every image, the input and the output are saved in the *output/* folder.

<p align="center">
  <img width="450" height="175" src="assets/output1.png">
  <img width="450" height="175" src="assets/output2.png">
  <img width="450" height="175" src="assets/output3.png">
</p>

**Testing log**
```
* Recall(%): 54.830     * Precision(%):  (97.727, 79.804, 41.202, 14.815)    
```

The precision is computed for the (1, 2, 3, 4) corners detected with highest confidence (the gaussians with a highest value on its center).

### Run train

Trains the network from scratch or from a given checkpoint.

```
$ python3 main.py
```

**Training log**
```
Epoch: [5][300/312]	Loss.avg: 0.3615	Recall(%): 21.622	Precision num. corners (%): (22.591, 18.563, 15.833, 16.809)
```

**Tensorboard**

Track training progress in Tensorboard:
+ Initialize training
+ Run the command below inside the pytorch-corner-detection directory.
+ Go to [http://localhost:6006/](http://localhost:6006/)

```
$ tensorboard --logdir='logs' --port=6006
```

## Chord detection

### Image

Detects the chords in 92 of the images of the dataset.

```
$ python3 detect.py
```

### Video

Detects the chords of the video of the dataset.

```
$ python3 video.py
```
