"""Measure the e-puck control-loop rate (FPS) with and without the CNN.

The robot stands in front of the horse model and processes real camera frames.
For each mode it times only the controller's own work per step (camera read,
red detection, and the CNN if used) and reports the achievable frame rate.
Results are printed and written to results/fps_benchmark.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "opencv"))
import opencv as ctl                                    # reuse the real controller functions
from controller import Supervisor

STEPS = 300                                            # frames measured per mode
robot = Supervisor()
step = int(robot.getBasicTimeStep())
cam = robot.getDevice("camera")
cam.enable(step)
W, H = cam.getWidth(), cam.getHeight()
model_dir = os.path.join(HERE, "..", "opencv")
keras_clf = ctl.CnnClassifier("", os.path.join(model_dir, ctl.MODEL_FILE), use_tflite=False)
tflite_clf = ctl.CnnClassifier(os.path.join(model_dir, ctl.TFLITE_FILE), "", use_tflite=True)
for _ in range(10):
    robot.step(step)


def run(mode):
    times, cnn_calls = [], 0
    for _ in range(STEPS):
        robot.step(step)
        t0 = time.perf_counter()
        bgr = ctl.webots_image_to_bgr(cam.getImage(), W, H)
        det = ctl.find_red_object(bgr)
        if det is not None:
            box = det[:4]
            if mode == "keras_every_frame":
                keras_clf.predict(ctl.crop_for_cnn(bgr, box)); cnn_calls += 1
            elif mode == "tflite_every_frame":
                tflite_clf.predict(ctl.crop_for_cnn(bgr, box)); cnn_calls += 1
            elif mode == "as_implemented":
                # The controller classifies once per object, then pauses 2 s and turns away
                # (HOLD_TIME + TURN_AWAY_TIME + COOLDOWN_TIME = 7.5 s of simulated time).
                period = int((ctl.HOLD_TIME + ctl.TURN_AWAY_TIME + ctl.COOLDOWN_TIME) * 1000 / step)
                if run.counter % period == 0:
                    tflite_clf.predict(ctl.crop_for_cnn(bgr, box)); cnn_calls += 1
                run.counter += 1
        times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000
    return dict(mode=mode, mean_ms=round(float(ms.mean()), 3), p95_ms=round(float(np.percentile(ms, 95)), 3),
                fps=round(float(1000 / ms.mean()), 1), cnn_calls=cnn_calls, frames=STEPS,
                real_time_ok=bool(np.percentile(ms, 95) < step))
run.counter = 0

results = [run(m) for m in ("colour_only", "keras_every_frame", "tflite_every_frame", "as_implemented")]
out = dict(time_step_ms=step, camera=f"{W}x{H}", results=results)
os.makedirs(os.path.join(HERE, "..", "..", "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "..", "..", "results", "fps_benchmark.json"), "w"), indent=1)
for r in results:
    print(f"{r['mode']:20s} {r['mean_ms']:8.3f} ms/frame  (p95 {r['p95_ms']:.3f})  -> {r['fps']:8.1f} FPS  CNN calls {r['cnn_calls']}")
sys.stdout.flush()
robot.simulationQuit(0)
