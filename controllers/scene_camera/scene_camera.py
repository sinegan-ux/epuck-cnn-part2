"""Test helper: saves a screenshot of the 3D view once the e-puck has stopped, then quits."""
import os, sys
from controller import Supervisor
sup = Supervisor()
step = int(sup.getBasicTimeStep())
epuck = sup.getFromDef("EPUCK")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", sys.argv[1] if len(sys.argv) > 1 else "scene.png")
still = 0
while sup.step(step) != -1:
    t = sup.getTime()
    v = epuck.getVelocity()
    speed = (v[0] ** 2 + v[1] ** 2) ** 0.5
    still = still + 1 if (t > 1.0 and speed < 1e-3) else 0
    if still == 10 or t > 30:
        sup.exportImage(os.path.abspath(out), 90)
        print("scene saved", os.path.abspath(out), "t=%.2f" % t)
        for _ in range(int(500 / step)):
            sup.step(step)
        sup.simulationQuit(0)
        break
