"""Throwaway: two robosuite xarm7 robots on a 45deg V-mount in Genesis (no gripper).

Mount, per side (mirrored about y=0):
  - 4040 extrusion approximated as a 2x2 in bar, 10 in long
  - 5x8 in, 1 cm thick baseplate centered on the bar
  - xarm7 base centered on top of the plate
The two bars join at a ridge so each mounting face is 45 deg from the floor and
the arms are 90 deg apart (floor-xarm-vertical-xarm-floor). Assembly origin =
xy center of the mount at its lowest point, placed at the world origin.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from tqdm import tqdm
import genesis as gs

ASSETS = Path("/home/mhyatt000/repo/xarm-sim/robosuite/robosuite/models/assets")
ROBOT_XML = ASSETS / "robots/xarm7/robot.xml"
OUT = Path(__file__).parent / "patched_mjcf"
OUT.mkdir(exist_ok=True)

IN = 0.0254
THETA = np.deg2rad(45.0)
BAR_SECTION = 2 * IN
BAR_LEN = 10 * IN
PLATE_W, PLATE_L, PLATE_T = 5 * IN, 8 * IN, 0.01


def patch(src: Path, dst: Path) -> None:
    """Pin relative asset paths to the source location; they break once the XML moves."""
    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(src.parent))
    compiler.set("texturedir", str(src.parent))
    tree.write(dst)


gs.init(backend=gs.gpu, logging_level="error")

robot_patched = OUT / "xarm7_robot.xml"
patch(ROBOT_XML, robot_patched)

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

sin, cos = np.sin(THETA), np.cos(THETA)
# Ridge (top-inner edge of both bars) height such that the lowest bar corner is at z=0.
ridge = np.array([0.0, 0.0, BAR_LEN * sin + BAR_SECTION * cos])

for s in (-1.0, 1.0):
    n = np.array([0.0, s * sin, cos])  # mounting-face normal
    d = np.array([0.0, s * cos, -sin])  # down-slope direction
    t = -s * THETA  # rotation about x taking +z to n
    quat = (np.cos(t / 2), np.sin(t / 2), 0.0, 0.0)

    mid = ridge + d * BAR_LEN / 2  # face-plane point above the bar center
    scene.add_entity(
        gs.morphs.Box(size=(BAR_SECTION, BAR_LEN, BAR_SECTION), pos=tuple(mid - n * BAR_SECTION / 2), quat=quat, fixed=True)
    )
    scene.add_entity(
        gs.morphs.Box(size=(PLATE_W, PLATE_L, PLATE_T), pos=tuple(mid + n * PLATE_T / 2), quat=quat, fixed=True)
    )
    scene.add_entity(gs.morphs.MJCF(file=str(robot_patched), pos=tuple(mid + n * PLATE_T), quat=quat))

scene.build()

for _ in tqdm(range(1000)):
    scene.step()

scene.destroy()
