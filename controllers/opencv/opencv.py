"""e-puck controller: red-object detection (OpenCV) + CIFAR-10 CNN classification.

Pipeline (one camera frame per time step):
    camera (BGRA) -> red mask in HSV -> largest red blob -> steer towards it
    -> when the blob is centred and large enough: stop, crop, resize to 32x32,
       classify with the CNN, print the label and save the annotated image
    -> pause, turn away and resume exploring (obstacle avoidance).

The CNN is optional. If TensorFlow or 'final_model.keras' is not available,
the controller still detects red objects and saves images, and says so once.
"""
import os

import cv2 as cv
import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TIME_STEP = 32                    # ms per control step
MAX_SPEED = 6.28                  # rad/s, e-puck wheel limit
CRUISE = 0.5 * MAX_SPEED

# Obstacle avoidance
DIST_LIMIT = 70.0                 # proximity reading that counts as "blocked"
AVOID_PERIOD = 0.5                # s between obstacle-avoidance decisions

# Red detection (HSV; OpenCV hue range is 0-179, red wraps around 0)
RED_LOWER_1 = np.array([0, 120, 70], dtype=np.uint8)
RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_LOWER_2 = np.array([170, 120, 70], dtype=np.uint8)
RED_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)
MIN_AREA_FRAC = 0.005             # blob must cover >= 0.5 % of the image to count
STOP_AREA_FRAC = 0.05             # stop and classify when blob covers >= 5 %
CENTRE_TOLERANCE = 0.15           # |blob centre - image centre| / width to be "centred"
STEER_GAIN = 1.5                  # how hard to turn towards an off-centre blob

# After a detection
HOLD_TIME = 2.0                   # s to stay stopped after classifying
TURN_AWAY_TIME = 1.5              # s to rotate away before exploring again
COOLDOWN_TIME = 4.0               # s during which red is ignored after turning away
RESUME_AFTER_DETECTION = True     # False = stop for good at the first red object

# CNN
MODEL_FILE = "final_model.keras"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CROP_PADDING = 0.10               # extra margin around the bounding box

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(CONTROLLER_DIR, "images")


# ----------------------------------------------------------------------
# Image processing (pure functions, no Webots dependency)
# ----------------------------------------------------------------------
def webots_image_to_bgr(raw, width, height):
    """Convert the BGRA byte buffer from Camera.getImage() to a BGR array."""
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return cv.cvtColor(bgra, cv.COLOR_BGRA2BGR)


def find_red_object(bgr):
    """Return (x, y, w, h, area_fraction) of the largest red blob, or None."""
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, RED_LOWER_1, RED_UPPER_1) | cv.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest)
    area_frac = float(np.count_nonzero(mask[y:y + h, x:x + w])) / mask.size
    if area_frac < MIN_AREA_FRAC:
        return None
    return x, y, w, h, area_frac


def crop_for_cnn(bgr, box):
    """Crop the box (with padding) and return a 1x32x32x3 RGB float32 batch in [0, 1]."""
    img_h, img_w = bgr.shape[:2]
    x, y, w, h = box
    pad_x, pad_y = int(w * CROP_PADDING), int(h * CROP_PADDING)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)

    crop = bgr[y0:y1, x0:x1]
    crop = cv.resize(crop, (32, 32), interpolation=cv.INTER_AREA)
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)          # CIFAR-10 is RGB
    return (crop.astype(np.float32) / 255.0)[np.newaxis, ...]


class CnnClassifier:
    """Wraps the trained Keras model; disabled (not crashing) if it cannot load."""

    def __init__(self, model_path):
        self.model = None
        try:
            import tensorflow as tf
            self._tf = tf
            self.model = tf.keras.models.load_model(model_path)
            # One warm-up call so the first real prediction is not slow.
            self.model(np.zeros((1, 32, 32, 3), np.float32), training=False)
            print(f"CNN loaded: {model_path}")
        except Exception as exc:  # missing TensorFlow, missing file, bad file
            print(f"CNN disabled ({type(exc).__name__}: {exc}). Colour detection only.")

    @property
    def enabled(self):
        return self.model is not None

    def predict(self, batch):
        """Return (class_name, confidence) for a 1x32x32x3 batch."""
        out = self.model(batch, training=False).numpy()[0]
        # The notebook model ends in softmax; still handle a logits model safely.
        if not np.isclose(out.sum(), 1.0, atol=1e-3) or out.min() < 0:
            out = np.exp(out - out.max())
            out /= out.sum()
        idx = int(np.argmax(out))
        return CLASS_NAMES[idx], float(out[idx])


def annotate(bgr, box):
    """Draw the bounding box on a copy of the image.

    The e-puck camera is tiny (52 x 39 by default), so the label is not drawn
    on the image; it goes in the file name and the console instead.
    """
    out = bgr.copy()
    x, y, w, h = box
    cv.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
    return out


# ----------------------------------------------------------------------
# Robot controller
# ----------------------------------------------------------------------
class EpuckController:
    SEARCH, HOLD, TURN_AWAY = "search", "hold", "turn_away"

    def __init__(self):
        from controller import Robot  # imported here so the helpers above are testable
        self.robot = Robot()

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for motor in (self.left_motor, self.right_motor):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        # ps0/ps1 face front-right, ps7/ps6 face front-left
        self.right_sensors = [self.robot.getDevice(n) for n in ("ps0", "ps1")]
        self.left_sensors = [self.robot.getDevice(n) for n in ("ps7", "ps6")]
        for sensor in self.right_sensors + self.left_sensors:
            sensor.enable(TIME_STEP)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(TIME_STEP)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()

        os.makedirs(SAVE_FOLDER, exist_ok=True)
        self.classifier = CnnClassifier(os.path.join(CONTROLLER_DIR, MODEL_FILE))

        self.state = self.SEARCH
        self.state_until = 0.0
        self.ignore_red_until = 0.0
        self.next_avoid_time = 0.0
        self.image_count = 0

    # --- motors ------------------------------------------------------
    def set_speed(self, left, right):
        self.left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
        self.right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

    def stop(self):
        self.set_speed(0.0, 0.0)

    # --- sensors -----------------------------------------------------
    @staticmethod
    def _blocked(sensors):
        return sum(s.getValue() for s in sensors) / len(sensors) > DIST_LIMIT

    def avoid_obstacles(self, now):
        """Wander forward, turning away from obstacles (decided every AVOID_PERIOD)."""
        if now < self.next_avoid_time:
            return
        self.next_avoid_time = now + AVOID_PERIOD

        blocked_left = self._blocked(self.left_sensors)
        blocked_right = self._blocked(self.right_sensors)
        if blocked_left and blocked_right:
            self.set_speed(-CRUISE, -CRUISE)            # back off
        elif blocked_left:
            self.set_speed(CRUISE, 0.2 * CRUISE)        # turn right
        elif blocked_right:
            self.set_speed(0.2 * CRUISE, CRUISE)        # turn left
        else:
            self.set_speed(CRUISE, CRUISE)

    # --- detection ---------------------------------------------------
    def handle_red(self, bgr, detection, now):
        """Steer towards the red blob; stop and classify once it is close and centred."""
        x, y, w, h, area_frac = detection
        offset = ((x + w / 2.0) - self.width / 2.0) / self.width   # -0.5 .. 0.5

        if area_frac < STOP_AREA_FRAC or abs(offset) > CENTRE_TOLERANCE:
            turn = STEER_GAIN * offset * CRUISE
            self.set_speed(CRUISE + turn, CRUISE - turn)
            return

        self.stop()
        if self.classifier.enabled:
            name, conf = self.classifier.predict(crop_for_cnn(bgr, (x, y, w, h)))
            label = f"{name}_{conf * 100:.0f}pct"
            print(f"[t={now:6.2f}s] RED object detected. CNN prediction: {name} ({conf:.1%})")
        else:
            label = "unclassified"
            print(f"[t={now:6.2f}s] RED object detected.")
        self.save_image(bgr, (x, y, w, h), label)

        self.state = self.HOLD
        self.state_until = now + HOLD_TIME

    def save_image(self, bgr, box, label):
        self.image_count += 1
        base = os.path.join(SAVE_FOLDER, f"red_object_{self.image_count:03d}_{label}")
        ok_raw = cv.imwrite(base + ".png", bgr)
        ok_ann = cv.imwrite(base + "_annotated.png", annotate(bgr, box))
        if ok_raw and ok_ann:
            print(f"  Saved {base}.png (+ _annotated.png)")
        else:
            print(f"  Failed to save images to {SAVE_FOLDER}")

    # --- main loop ---------------------------------------------------
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            now = self.robot.getTime()

            if self.state == self.HOLD:
                if now >= self.state_until:
                    if not RESUME_AFTER_DETECTION:
                        print("Stopped at the red object.")
                        return
                    self.state = self.TURN_AWAY
                    self.state_until = now + TURN_AWAY_TIME
                    self.set_speed(-0.5 * CRUISE, 0.5 * CRUISE)   # spin on the spot
                continue

            if self.state == self.TURN_AWAY:
                if now >= self.state_until:
                    self.state = self.SEARCH
                    self.ignore_red_until = now + COOLDOWN_TIME
                    self.next_avoid_time = now
                continue

            raw = self.camera.getImage()
            if raw is not None and now >= self.ignore_red_until:
                bgr = webots_image_to_bgr(raw, self.width, self.height)
                detection = find_red_object(bgr)
                if detection is not None:
                    self.handle_red(bgr, detection, now)
                    continue

            self.avoid_obstacles(now)


if __name__ == "__main__":
    EpuckController().run()
