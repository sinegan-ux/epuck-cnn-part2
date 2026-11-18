"""
E-puck Autonomous Block Detection and Reporting (Webots R2023b)

The controller:
  1. Spawns a RED, BLUE and GREEN block (0.3 x 0.3 x 0.1 m) at random places.
  2. Drives the e-puck around the arena while avoiding obstacles.
  3. Prints "I see red" / "I see blue" / "I see green" on the FIRST encounter
     of each colour, followed by a summary of what has been encountered.

REQUIREMENT: the E-puck node in the world must have  supervisor TRUE
and its controller field must be  my_controller.
"""

import random
from controller import Supervisor, Camera

# ---------------- Configuration ----------------
SPEED = 4.0                       # wheel speed (rad/s), e-puck max is 6.28
BLOCK_SIZE = (0.3, 0.3, 0.1)      # length, width, height (m) as per brief
OBSTACLE_THRESHOLD = 150          # proximity value meaning "something close"
MIN_PIXEL_FRACTION = 0.02         # colour must cover >= 2% of the camera image
WALL_MARGIN = 0.2                 # keep block centres this far from walls
ROBOT_CLEARANCE = 0.25            # keep blocks this far from the robot start
BLOCK_SEPARATION = 0.4            # min distance between block centres

COLOURS = {
    "red":   (1, 0, 0),
    "blue":  (0, 0, 1),
    "green": (0, 1, 0),
}

found = []   # colours encountered so far, in order of discovery


# ---------------- Environment setup ----------------
def get_arena_half_size(root):
    """Read floorSize from the RectangleArena so blocks stay inside it."""
    children = root.getField("children")
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node is not None and node.getTypeName() == "RectangleArena":
            size = node.getField("floorSize").getSFVec2f()
            return size[0] / 2.0, size[1] / 2.0
    return 0.5, 0.5   # fallback: default 1 m x 1 m arena


def create_blocks(robot, root):
    """Create the red, blue and green blocks at random, non-overlapping spots."""
    children = root.getField("children")
    half_x, half_y = get_arena_half_size(root)
    limit_x = max(half_x - WALL_MARGIN, 0.05)
    limit_y = max(half_y - WALL_MARGIN, 0.05)

    rx, ry, _ = robot.getSelf().getPosition()
    placed = []

    for colour, (r, g, b) in COLOURS.items():
        # Remove a block left over from a previously saved world
        def_name = colour.upper() + "_BLOCK"
        old = robot.getFromDef(def_name)
        if old is not None:
            old.remove()

        # Pick a random position that does not overlap the robot or other blocks
        x, y = 0.0, 0.0
        for _ in range(1000):
            x = random.uniform(-limit_x, limit_x)
            y = random.uniform(-limit_y, limit_y)
            clear_of_robot = max(abs(x - rx), abs(y - ry)) > ROBOT_CLEARANCE
            clear_of_blocks = all(
                max(abs(x - px), abs(y - py)) > BLOCK_SEPARATION
                for px, py in placed
            )
            if clear_of_robot and clear_of_blocks:
                break
        else:
            print("Warning: arena is small, %s block may overlap." % colour)

        placed.append((x, y))
        sx, sy, sz = BLOCK_SIZE

        block = f"""
        DEF {def_name} Solid {{
            translation {x} {y} {sz / 2}
            name "{colour}_block"
            children [
                Shape {{
                    appearance PBRAppearance {{
                        baseColor {r} {g} {b}
                        roughness 1
                        metalness 0
                    }}
                    geometry Box {{
                        size {sx} {sy} {sz}
                    }}
                }}
            ]
            boundingObject Box {{
                size {sx} {sy} {sz}
            }}
        }}
        """
        children.importMFNodeFromString(-1, block)
        print("Created %s block at x=%.2f, y=%.2f" % (colour, x, y))


# ---------------- Colour detection ----------------
def detect_colour(r, g, b):
    """Classify a pixel as red/green/blue if one channel clearly dominates."""
    if r > 90 and g < r * 0.35 and b < r * 0.35:
        return "red"
    if g > 90 and r < g * 0.35 and b < g * 0.35:
        return "green"
    if b > 90 and r < b * 0.35 and g < b * 0.35:
        return "blue"
    return None


def scan_camera(camera):
    """Return the fraction of the image covered by each target colour."""
    image = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()
    counts = {}

    for x in range(w):
        for y in range(h):
            r = Camera.imageGetRed(image, w, x, y)
            g = Camera.imageGetGreen(image, w, x, y)
            b = Camera.imageGetBlue(image, w, x, y)
            colour = detect_colour(r, g, b)
            if colour:
                counts[colour] = counts.get(colour, 0) + 1

    total = float(w * h)
    return {c: n / total for c, n in counts.items()}


# ---------------- Reporting ----------------
def report(colour):
    """Print the response for a first encounter, then the summary."""
    found.append(colour)
    print("I see " + colour)
    print("Encountered so far:", ", ".join(found))
    missing = [c for c in COLOURS if c not in found]
    print("Not encountered yet:", ", ".join(missing) if missing else "None")
    print("--------------------------------")


# ---------------- Main program ----------------
def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    root = robot.getRoot()
    if root is None:
        print("ERROR: set 'supervisor TRUE' on the E-puck node in Webots, "
              "save the world and reload it.")
        return

    random.seed()
    create_blocks(robot, root)

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    sensors = []
    for i in range(8):
        s = robot.getDevice("ps" + str(i))
        s.enable(timestep)
        sensors.append(s)

    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    left.setPosition(float("inf"))
    right.setPosition(float("inf"))
    left.setVelocity(0.0)
    right.setVelocity(0.0)

    turn_steps = 0
    turn_dir = 1      # +1 = turn left, -1 = turn right

    while robot.step(timestep) != -1:

        # Stop once all three colours have been reported
        if len(found) == len(COLOURS):
            left.setVelocity(0.0)
            right.setVelocity(0.0)
            print("All three blocks found. Stopping.")
            break

        # --- Vision: report the first encounter of each colour ---
        fractions = scan_camera(camera)
        for colour in COLOURS:
            if colour not in found and fractions.get(colour, 0) >= MIN_PIXEL_FRACTION:
                report(colour)
                break

        # --- Navigation: obstacle avoidance with random wandering ---
        ps = [s.getValue() for s in sensors]
        right_side = ps[0] + ps[1] + ps[2]
        left_side = ps[5] + ps[6] + ps[7]
        obstacle = max(ps[0], ps[1], ps[2], ps[5], ps[6], ps[7]) > OBSTACLE_THRESHOLD

        if turn_steps > 0:
            turn_steps -= 1
        elif obstacle:
            turn_dir = 1 if right_side > left_side else -1
            turn_steps = random.randint(10, 30)
        elif random.random() < 0.01:          # occasional random wander
            turn_dir = random.choice([-1, 1])
            turn_steps = random.randint(5, 20)

        if turn_steps > 0:
            left.setVelocity(-turn_dir * SPEED)
            right.setVelocity(turn_dir * SPEED)
        else:
            left.setVelocity(SPEED)
            right.setVelocity(SPEED)


if __name__ == "__main__":
    main()