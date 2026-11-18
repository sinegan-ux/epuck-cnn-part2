"""Supervisor: photograph each object in the world from a chosen viewpoint and save the
images to notebooks/my_images/ so the notebook can classify them."""
import os

import cv2 as cv
from controller import Supervisor

SHOTS = [  # (file name, viewpoint position, viewpoint orientation)
    ("webots_horse.png", [0.9, -7.0, 1.5], [-0.0459, 0.0403, 0.9981, 1.7005]),
    ("webots_car.png", [18.54, -12.89, 2.78], [-0.0712, 0.0409, 0.9966, 2.1016]),
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "notebooks", "my_images")

sup = Supervisor()
step = int(sup.getBasicTimeStep())
view = sup.getFromDef("VIEW")
os.makedirs(OUT, exist_ok=True)
for name, pos, ori in SHOTS:
    view.getField("position").setSFVec3f(pos)
    view.getField("orientation").setSFRotation(ori)
    for _ in range(10):
        sup.step(step)
    path = os.path.abspath(os.path.join(OUT, name))
    sup.exportImage(path, 95)
    for _ in range(5):
        sup.step(step)
    # Square crop centred on the object: the CNN resizes to 32x32, so a wide image would be squashed.
    img = cv.imread(path)
    h, w = img.shape[:2]
    x0 = max(0, min(w - h, int(w * 0.55) - h // 2))
    cv.imwrite(path, img[:, x0:x0 + h])
    print("saved", path)
sup.simulationQuit(0)
