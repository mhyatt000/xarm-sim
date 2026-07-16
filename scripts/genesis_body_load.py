"""Throwaway: two robosuite xarm7 robots on a 45deg V-mount in Genesis (no gripper).

Mount, mirrored about y=0 (all 4040 extrusion approximated as 2x2 in bars):
  - 2 floor rails, 2 ft long, running along y, 3 in between centerlines
  - per side: 2 sloped bars, one under each end of the plate (5 in plate,
    2 in bars -> centerlines 3 in apart), mitered 45 deg at both ends:
    10 in on the mounting face, 6 in on the short face, so the bottom cut sits
    flat on the rails and the ridge cuts of the two sides join flush at y=0
  - 5x8 in, 1 cm thick baseplate centered across its two bars
  - xarm7 base centered on top of the plate
The sloped bars join at the ridge so each mounting face is 45 deg from the
floor and the arms are 90 deg apart (floor-xarm-vertical-xarm-floor); the rails
raise the V by 2 in. Assembly origin = xy center of the mount at its lowest
point, placed at the world origin.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
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
RAIL_LEN = 24 * IN
BAR_SPACING = 3 * IN  # centerline to centerline
PLATE_W, PLATE_L, PLATE_T = 5 * IN, 8 * IN, 0.01
X_OFFS = (-BAR_SPACING / 2, BAR_SPACING / 2)


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


def miter_bar_mesh(dst: Path) -> None:
    """Sloped-bar prism, centered like the equivalent box: local y along the length,
    +z the mounting face. 45 deg miters shorten the -z face by BAR_SECTION per end."""
    half = BAR_SECTION / 2
    profile = [  # (y, z) trapezoid
        (-BAR_LEN / 2, half),
        (BAR_LEN / 2, half),
        (BAR_LEN / 2 - BAR_SECTION, -half),
        (-BAR_LEN / 2 + BAR_SECTION, -half),
    ]
    verts = [(x, y, z) for x in (-half, half) for y, z in profile]
    trimesh.Trimesh(vertices=verts).convex_hull.export(dst)


gs.init(backend=gs.gpu, logging_level="error")

robot_patched = OUT / "xarm7_robot.xml"
patch(ROBOT_XML, robot_patched)
bar_mesh = OUT / "miter_bar_10in.stl"
miter_bar_mesh(bar_mesh)

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

for x in X_OFFS:
    scene.add_entity(
        gs.morphs.Box(size=(BAR_SECTION, RAIL_LEN, BAR_SECTION), pos=(x, 0.0, BAR_SECTION / 2), fixed=True)
    )

sin, cos = np.sin(THETA), np.cos(THETA)
# Ridge (top-inner edge of the sloped bars) height such that their horizontal
# bottom-cut face lands on the rail tops (z = BAR_SECTION). That face sits
# BAR_LEN*sin below the ridge: it starts at the down-slope end of the 10 in
# mounting face (the miter cut off the old box corner that hung lower).
ridge = np.array([0.0, 0.0, BAR_SECTION + BAR_LEN * sin])

for s in (-1.0, 1.0):
    n = np.array([0.0, s * sin, cos])  # mounting-face normal
    d = np.array([0.0, s * cos, -sin])  # down-slope direction
    t = -s * THETA  # rotation about x taking +z to n
    quat = (np.cos(t / 2), np.sin(t / 2), 0.0, 0.0)

    mid = ridge + d * BAR_LEN / 2  # face-plane point above the bar centers
    for x in X_OFFS:
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(bar_mesh),
                pos=tuple(np.array([x, 0.0, 0.0]) + mid - n * BAR_SECTION / 2),
                quat=quat,
                fixed=True,
            )
        )
    scene.add_entity(
        gs.morphs.Box(size=(PLATE_W, PLATE_L, PLATE_T), pos=tuple(mid + n * PLATE_T / 2), quat=quat, fixed=True)
    )
    scene.add_entity(gs.morphs.MJCF(file=str(robot_patched), pos=tuple(mid + n * PLATE_T), quat=quat))

scene.build()

for _ in tqdm(range(1000)):
    scene.step()

scene.destroy()
