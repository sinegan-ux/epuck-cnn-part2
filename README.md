# Object classification with a CIFAR-10 CNN for the e-puck in Webots

Applied Robotics and AI Project, Part 2 (Assignment 3).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sinegan-ux/epuck-cnn-part2/blob/main/notebooks/cnn_v3.ipynb)

The e-puck finds a red object with OpenCV, drives up to it, and classifies it with a
convolutional neural network (CNN) trained on CIFAR-10. The CNN runs only when an object
is centred and close, so normal driving and obstacle avoidance are not slowed down.

```
camera frame -> red mask (HSV) -> largest blob -> steer -> stop when close
             -> crop + resize to 32x32 -> CNN -> "class (confidence)" + saved image
```

## Contents

| Path | What it is |
|---|---|
| `notebooks/cnn_v3.ipynb` | Trains the CNN (SGD vs Adam), evaluates it, classifies our own images, converts it to TensorFlow Lite int8 and times it |
| `notebooks/my_images/` | Images of our own: car, horse and cat photos, plus horse and car rendered in Webots |
| `controllers/opencv/opencv.py` | Part 2 e-puck controller: red detection, approach, CNN classification, image saving |
| `controllers/opencv/final_model.keras` | Trained model loaded by the controller |
| `models/final_model_int8.tflite` | Int8 TensorFlow Lite version (8× smaller, same accuracy within 0.1 points) |
| `controllers/my_controller/` | Part 1 controller (colour blocks, "I see ...") |
| `controllers/scene_camera/` | Test helper: saves a 3D-view screenshot when the e-puck stops |
| `controllers/render_own_images/` | Renders the horse and car images in `notebooks/my_images/` |
| `worlds/part1_colour_blocks.wbt` | Part 1 world |
| `worlds/cnn_test_horse.wbt` | Part 2 test: scaled horse model in front of the e-puck |
| `worlds/cnn_test_block.wbt` | Part 2 test: Part 1 red block in front of the e-puck |
| `worlds/own_images.wbt` | Scene used to render the horse and car images |
| `results/` | Screenshots and camera images from the Webots tests |
| `presentation/` | Final slides |

## Running it

### 1. Train the model (Google Colab or local Jupyter)

Open `notebooks/cnn_v3.ipynb` and run all cells. In Colab, upload your photos into a
`my_images` folder (Files panel) before running the "own images" cell. The notebook writes
`final_model.keras` and `final_model_int8.tflite`.

### 2. Run the Webots test

1. Copy `final_model.keras` into `controllers/opencv/` (already there in this repository).
2. Webots must run the controller with a Python that has TensorFlow and OpenCV
   (`pip install tensorflow opencv-python`). Either select that Python in
   *Tools → Preferences → Python command*, or create `controllers/opencv/runtime.ini`:
   ```
   [python]
   COMMAND = C:/path/to/python.exe
   ```
3. Open `worlds/cnn_test_horse.wbt` and press play. The console prints for example
   `RED object detected. CNN prediction: horse (99.8%)`, and the camera image is saved to
   `controllers/opencv/images/`.

Without TensorFlow the controller still runs: it detects red objects, saves the images and
says that the CNN is disabled.

## Key settings (`controllers/opencv/opencv.py`)

| Setting | Value | Why |
|---|---|---|
| Red mask | H 0–10 or 170–179, S ≥ 140, V ≥ 40 | S ≥ 140 rejects the reddish wooden floor of the default arena (S ≤ ~125) |
| `STOP_AREA_FRAC` | 0.05 | Stop and classify when the object covers 5 % of the image |
| `CENTRE_TOLERANCE` | 0.15 | Object must be within 15 % of the image centre |
| `RESUME_AFTER_DETECTION` | True | Pause 2 s, turn away and keep searching (False = stop for good) |

## Results

| | |
|---|---|
| Test accuracy (10,000 CIFAR-10 test images) | **81.10 %** (SGD with momentum, dropout 0.2, 30 epochs on a Colab T4 GPU) |
| Easiest / hardest classes | automobile 94.8 %, truck 90.5 % / deer 67.9 %, cat 69.6 % |
| Most confused | cat → dog (170), dog → cat (98), horse → dog (88) |
| Own images | photos: car → automobile 100 %, horse → horse 100 %, cat → cat 60 % (dog 40 %); Webots renders: horse 99.9 %, car 99.7 % |
| Webots test | horse model found, approached and classified as **horse (99.8 %)**; red block → cat (65 %, not a CIFAR-10 class) |
| Inference per image (laptop CPU) | `model.predict` 68 ms, direct Keras call 16 ms, TFLite float32 0.7 ms, TFLite int8 0.7 ms |
| Model size | Keras 2.3 MB, TFLite int8 0.6 MB |

A first run with dropout 0.5 reached 77.25 % with SGD, but Adam stalled at 35.5 %. A short test
traced it to that dropout; with dropout 0.2 both optimisers train, and accuracy rose to 81.10 %.

Two controller fixes came out of the Webots tests:

* **Floor rejection:** the dark wooden floor squares passed the old red threshold
  (S ≥ 120), so the robot sometimes chased the floor. S ≥ 140 separates them.
* **Square crop:** stretching the 18×9-pixel horse box to 32×32 gave "cat 32 %"; a square
  crop that keeps the object's shape gives "horse 99.9 %".
