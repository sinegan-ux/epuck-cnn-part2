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
| `notebooks/demo_classify.ipynb` | Quick demo: upload images in Colab and the CNN shows its top-3 guesses ([open in Colab](https://colab.research.google.com/github/sinegan-ux/epuck-cnn-part2/blob/main/notebooks/demo_classify.ipynb)) |
| `notebooks/cnn_v3.ipynb` | Trains the CNN (SGD vs Adam), evaluates it, classifies our own images, converts it to TensorFlow Lite int8 and times it |
| `notebooks/my_images/` | Images of our own: car, horse and cat photos, plus horse and car rendered in Webots |
| `controllers/opencv/opencv.py` | Part 2 e-puck controller: red detection, approach, CNN classification, image saving |
| `controllers/opencv/final_model_int8.tflite` | Int8 TensorFlow Lite model used by the controller (default) |
| `controllers/opencv/final_model.keras` | Keras model (fallback) |
| `models/final_model_int8.tflite` | Int8 TensorFlow Lite version (4× smaller, same accuracy within 0.1 points) |
| `controllers/my_controller/` | Part 1 controller (colour blocks, "I see ...") |
| `controllers/scene_camera/` | Test helper: saves a 3D-view screenshot when the e-puck stops |
| `controllers/render_own_images/` | Renders the horse and car images in `notebooks/my_images/` |
| `worlds/part1_colour_blocks.wbt` | Part 1 world |
| `worlds/cnn_demo_horse.wbt` | **Live demo**: horse world that stays open (no auto-quit helper) |
| `worlds/cnn_test_horse.wbt` | Part 2 test: scaled horse model in front of the e-puck (saves a screenshot and quits) |
| `worlds/cnn_test_block.wbt` | Part 2 test: Part 1 red block in front of the e-puck |
| `worlds/own_images.wbt` | Scene used to render the horse and car images |
| `worlds/fps_benchmark.wbt` + `controllers/fps_benchmark/` | Frame-rate test of the control loop with and without the CNN |
| `results/` | Screenshots and camera images from the Webots tests |
| `presentation/` | Final slides (Assignment 3) |
| `report/` | Final project report (Assignment 4) |

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
| `USE_TFLITE` | True | Use the int8 TensorFlow Lite model (falls back to the Keras model if it cannot load) |

## Controller design

| Function / class | Role |
|---|---|
| `webots_image_to_bgr()` | Converts the camera's BGRA buffer to an OpenCV image |
| `find_red_object()` | HSV red mask, morphological opening, largest contour → bounding box and area; `None` when no red object is seen |
| `crop_for_cnn()` | Square crop around the box (keeps the object's shape), resize to 32×32, RGB, scale to [0, 1] |
| `CnnClassifier` | Loads TFLite int8 or Keras; returns class and confidence; disables itself cleanly if TensorFlow or the model is missing |
| `EpuckController` | State machine: **search** (wander, avoid obstacles) → **approach** (steer to the red object) → **classify** (stop, CNN, save image) → **turn away** (2 s pause, rotate, ignore red for 4 s) → search |

Error handling: no camera image or no red object → keep searching; model missing → colour-only mode; image save failure → message, no crash.

## Hardware notes (e-puck in Webots R2023b)

| Device | Details | Used for |
|---|---|---|
| Camera `camera` | 52×39 pixels, BGRA, about 0.84 rad field of view | Red detection and CNN input |
| Wheel motors `left wheel motor`, `right wheel motor` | Velocity control, maximum 6.28 rad/s (position set to infinity) | Cruise at 50 % speed; steering by speed difference |
| Proximity sensors `ps0`–`ps7` | Infrared, larger value = closer | Part 2 uses ps0/ps1 (front right) and ps6/ps7 (front left), threshold 70 |
| Control time step | 32 ms (`TIME_STEP`), world basic time step 16 ms | |

## Part 1 world setup (`worlds/part1_colour_blocks.wbt`, `controllers/my_controller`)

* The controller is a **Supervisor** (`supervisor TRUE` on the E-puck node) so it can create objects.
* At start it removes old blocks and spawns a **red, a blue and a green block, each 0.3 × 0.3 × 0.1 m**
  (Solid with a matching box `boundingObject`), at **random non-overlapping positions**: at least 0.2 m
  from the walls, 0.25 m from the robot and 0.4 m from each other. The arena is 2 × 2 m.
* A colour counts as seen when it covers **at least 2 %** of the camera image; the robot then prints
  `I see <colour>` and a summary of encountered and not-yet-encountered colours, and stops once all three
  are found.

## Frame-rate test (`worlds/fps_benchmark.wbt`, `controllers/fps_benchmark`)

The e-puck stands in front of the horse and processes 300 real camera frames in each mode; only the
controller's own work per frame is timed (laptop CPU, control step 16 ms). Results: `results/fps_benchmark.json`.

| Mode | ms per frame | Frame rate | Real time (< 16 ms)? |
|---|---|---|---|
| Colour detection only | 0.68 | ~1,460 FPS | yes |
| Keras CNN on every frame | 49.3 | ~20 FPS | **no** |
| TFLite int8 CNN on every frame | 4.1 | ~240 FPS | yes |
| As implemented (CNN only when an object is close and centred) | 0.75 | ~1,330 FPS | yes |

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
