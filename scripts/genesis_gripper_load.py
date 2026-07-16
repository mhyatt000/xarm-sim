"""Throwaway: check which robosuite gripper MJCFs Genesis can load.

Patches known Genesis-incompatibilities into copies of the XMLs, then tries:
  (a) raw:    gripper alone in a scene
  (b) attach: gripper mounted on xarm7 right_hand via RigidEntity.attach
"""

import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm
import genesis as gs

ASSETS = Path("/home/mhyatt000/repo/xarm-sim/robosuite/robosuite/models/assets")
GRIPPERS = ASSETS / "grippers"
ROBOT_XML = ASSETS / "robots/xarm7/robot.xml"
OUT = Path(__file__).parent / "patched_mjcf"
OUT.mkdir(exist_ok=True)

XMLS = [
    "rethink_gripper.xml",
    "panda_gripper.xml",
    "robotiq_gripper_85.xml",
    "robotiq_gripper_140.xml",
    "robotiq_gripper_s.xml",
    "jaco_three_finger_gripper.xml",
    "bd_gripper.xml",
    "xarm7_gripper.xml",
    "inspire_left_hand.xml",
    "inspire_right_hand.xml",
    "fourier_left_hand.xml",
    "fourier_right_hand.xml",
]


def patch(src: Path, dst: Path) -> list[str]:
    """Patch Genesis-incompatible MJCF constructs. Returns list of applied fixes."""
    fixes = []
    tree = ET.parse(src)
    root = tree.getroot()

    # Relative asset paths break once the XML moves; pin dirs to the source location.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(src.parent))
    compiler.set("texturedir", str(src.parent))

    # `group` is not a valid <default> attribute; mujoco's strict schema rejects it.
    for d in root.iter("default"):
        if "group" in d.attrib:
            del d.attrib["group"]
            fixes.append("dropped group= on <default>")

    # Duplicate mesh names: drop later definitions (robotiq_85's dup even points at
    # a file that doesn't exist); geoms fall back to the first definition.
    seen = set()
    for asset in root.iter("asset"):
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            if name in seen:
                asset.remove(mesh)
                fixes.append(f"dropped duplicate mesh {name}")
            else:
                seen.add(name)

    # A non-identity transform on the base body makes Genesis keep a `world` link,
    # which breaks RigidEntity.attach. Zero it; the mount offset must instead be
    # applied on the parent side when attaching for real.
    for body in root.find("worldbody").findall("body"):
        pos, quat = body.get("pos", "0 0 0"), body.get("quat", "1 0 0 0")
        if [float(x) for x in pos.split()] != [0, 0, 0] or [float(x) for x in quat.split()] != [1, 0, 0, 0]:
            body.set("pos", "0 0 0")
            body.set("quat", "1 0 0 0")
            fixes.append(f"zeroed base transform (was pos={pos} quat={quat})")

    # Genesis supports connect/weld/joint equalities only; tendon couplings raise.
    for eq in root.iter("equality"):
        for t in eq.findall("tendon"):
            eq.remove(t)
            fixes.append(f"stripped equality/tendon {t.get('name')}")

    tree.write(dst)
    return fixes


def try_build(scene: gs.Scene, xml: Path, attach: bool, robot_xml: Path) -> str:
    try:
        if attach:
            robot = scene.add_entity(gs.morphs.MJCF(file=str(robot_xml)))
            gripper = scene.add_entity(gs.morphs.MJCF(file=str(xml), batch_fixed_verts=True))
            gripper.attach(robot, "right_hand")
        else:
            scene.add_entity(gs.morphs.MJCF(file=str(xml)))
        scene.build()
        for _ in tqdm(range(1000)):
            scene.step()
        return "OK"
    except Exception as e:
        traceback.print_exc()
        return f"FAIL: {type(e).__name__}: {e}"


gs.init(backend=gs.gpu, logging_level="error")

robot_patched = OUT / "xarm7_robot.xml"
patch(ROBOT_XML, robot_patched)

results = {}
for name in XMLS:
    patched = OUT / name
    fixes = patch(GRIPPERS / name, patched)
    for mode in ("raw", "attach"):
        scene = gs.Scene(show_viewer=True)
        try:
            results[(name, mode)] = try_build(scene, patched, mode == "attach", robot_patched)
        finally:
            # del + gc.collect() is not reliable here: the viewer thread and other
            # internals can keep the scene alive, so it stays in gs._scene_registry and
            # the next show_viewer=True scene refuses to start. destroy() unregisters
            # synchronously and stops the viewer.
            scene.destroy()
            del scene
    if fixes:
        print(f"[{name}] patches: {'; '.join(fixes)}")

print("\n===== RESULTS =====")
for (name, mode), res in results.items():
    print(f"{name:38s} {mode:7s} {res}")
