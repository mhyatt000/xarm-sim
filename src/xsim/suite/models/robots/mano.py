"""Rigid MANO hand robot (free-floating, robot-is-the-gripper).

The MANO hand as a suite robot, following the MHR/RUKA recast: the 6-dof
floating base is the "arm" (3 world-axis prismatics + intrinsic x-y-z
revolutes, prepended to the hand URDF as ordinary joints — uniformly
position-controllable, unlike a Genesis FREE joint) and the 20 finger dofs are
the "gripper" (``ManoGrasp``), so the scripted experts drive it like any
arm+gripper robot.

Asset: assets/mano/mano_hand_planar.urdf — 21 links / 20 dofs, concave
watertight visuals (links_planar) + convex hull collisions (links_hull).
Hand frame (measured by FK probe): fingers extend along -x, palm normal +y
(positive flex curls the tips toward +y), thumb on the +z side — a LEFT
hand: palm-down FK puts its thumb at +y, where a right hand's would point
-y. (On camera, fingers -x / palm toward +x reads as the opposite chirality
— judge handedness by FK, not by looks.) Genesis dof order equals URDF file
order: per finger [flex1, abd, flex2, flex3] for index, middle, pinky,
ring, thumb.

Right hand: the left-hand source asset mirrored across x=0. Axis rule
(x,-y,-z) keeps joint angles chirality-consistent, so the mirrored hand at
the same qpos is the exact reflection of the source hand. Generated into
assets/mano_right (assets/mano_left is the stale pre-relabel mirror, kept
on disk because scripts/teleop_hand.py reads it).

Run: ``scripts/suite.py --robots ManoR``. See [[mano-suite-robot]].
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import genesis as gs
import numpy as np

from xsim.suite.models.robots.robot_model import ROBOT_MODEL_REGISTRY, RobotModel

PROJECT_ROOT = Path(__file__).resolve().parents[5]
_MANO_DIR = PROJECT_ROOT / "assets" / "mano"
_MANO_URDF = _MANO_DIR / "mano_hand_planar.urdf"
_MANO_RIGHT_DIR = PROJECT_ROOT / "assets" / "mano_right"

_N_BASE = 6
_N_FINGER = 20

_IN = 0.0254  # inch -> m. Home base pose: x=4in, y=-+8in (R/L), z=1ft (rig layout)
_HOME_X, _HOME_Y, _HOME_Z = 4 * _IN, 8 * _IN, 12 * _IN

# prismatic travel covers the Lift/Stack workspace from the home base pose
# (cube spawn x 0.20-0.40, y +-0.288, transport height 0.15; the old NIMBLE
# attempt found +-0.2 too small). Revolutes +-2pi so the euler-unwrap IK can
# reach both 2pi branches of the top-down yaw family (it sits on the +-pi wrap).
_LIN = 0.5
_ANG = 2.0 * math.pi
_BASE_JOINTS = [
    ("prismatic", "1 0 0", _LIN), ("prismatic", "0 1 0", _LIN), ("prismatic", "0 0 1", _LIN),
    ("revolute", "1 0 0", _ANG), ("revolute", "0 1 0", _ANG), ("revolute", "0 0 1", _ANG),
]


def _find_root_link(root: ET.Element) -> str:
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [l.get("name") for l in root.findall("link") if l.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")
    return roots[0]


def build_floating_urdf(src: Path) -> Path:
    """Prepend a 6-dof (3 prismatic + 3 revolute) base chain to ``src``.
    Writes a sibling *_floating.urdf (visuals untouched — the rigid hand's
    planar meshes ARE the rendered surface)."""
    tree = ET.parse(src)
    robot = tree.getroot()
    hand_root = _find_root_link(robot)

    def dummy(name: str) -> ET.Element:
        link = ET.Element("link", name=name)
        inl = ET.SubElement(link, "inertial")
        ET.SubElement(inl, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inl, "mass", value="1e-3")  # nonzero: massless links blow up
        ET.SubElement(inl, "inertia", ixx="1e-6", iyy="1e-6", izz="1e-6", ixy="0", ixz="0", iyz="0")
        return link

    new_links = [ET.Element("link", name="world")] + [dummy(f"fb_{i}") for i in range(_N_BASE - 1)]
    chain = ["world"] + [f"fb_{i}" for i in range(_N_BASE - 1)] + [hand_root]
    new_joints = []
    for i, (jtype, axis, lim) in enumerate(_BASE_JOINTS):
        j = ET.Element("joint", name=f"base_{i}", type=jtype)
        ET.SubElement(j, "parent", link=chain[i])
        ET.SubElement(j, "child", link=chain[i + 1])
        ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(j, "axis", xyz=axis)
        ET.SubElement(j, "limit", lower=f"{-lim}", upper=f"{lim}", effort="50", velocity="10")
        new_joints.append(j)
    for el in reversed(new_links + new_joints):
        robot.insert(0, el)

    dst = src.with_name(src.stem + "_floating.urdf")
    ET.indent(robot)
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    return dst


@dataclass
class ManoL(RobotModel):
    """Left hand: the source asset's own chirality (see module docstring)."""

    name: str = "ManoL"
    morph_kind: Literal["urdf", "mjcf"] = "urdf"
    morph_file: str = str(_MANO_URDF)  # source; wrapped with a floating base in make_morph
    fixed: bool = True  # world root fixed; the 6 base joints provide the float
    self_collision: bool = False  # curled postures jam on intra-hand pairs
    arm_dofs: int = _N_BASE
    # base_3 = -pi/2 rolls the hand palm-down (palm normal +y_hand -> world -z)
    default_arm_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, -math.pi / 2, 0.0, 0.0)
    ee_link_name: str = "link_00"
    # gravity-compensated entity (see build_entity), so the gains only shape
    # tracking, not droop; prismatics carry the hand + payload, revolutes the
    # small wrist inertia
    arm_kp: tuple[float, ...] = (400.0, 400.0, 400.0, 30.0, 30.0, 30.0)
    arm_kv: tuple[float, ...] = (40.0, 40.0, 40.0, 2.5, 2.5, 2.5)
    arm_force_limit: float = 50.0
    gripper_name: str | None = "ManoGraspL"
    base_pos: tuple[float, float, float] = (_HOME_X, _HOME_Y, _HOME_Z)  # left at +y
    ik_backend: Literal["genesis", "softcost"] = "genesis"

    def make_morph(self):
        floating = build_floating_urdf(Path(self.morph_file))
        return gs.morphs.URDF(
            file=str(floating),
            pos=self.base_pos,
            quat=self.base_quat,
            fixed=self.fixed,
            merge_fixed_links=self.merge_fixed_links,
        )

    def build_entity(self, scene):
        # the floating hand is a virtual 6-dof mount, not a supported arm:
        # gravity compensation models the mount holding the hand's weight, so
        # the base gains can stay soft without a ~9 cm droop.
        # skin tone #ebb496, matte (the URDF's <material> tag is unreferenced
        # by its visuals, so the color rides on the entity surface)
        return scene.add_entity(
            material=gs.materials.Rigid(gravity_compensation=1.0),
            morph=self.make_morph(),
            surface=gs.surfaces.Rough(color=(0.922, 0.706, 0.588)),
        )


def _mirror_mesh(src: Path, dst: Path) -> None:
    """Reflect a mesh across x=0: negate x and flip triangle winding so the
    surface stays outward-facing."""
    import trimesh

    m = trimesh.load(src, process=False)
    v = np.asarray(m.vertices).copy()
    v[:, 0] *= -1.0
    trimesh.Trimesh(v, np.asarray(m.faces)[:, ::-1], process=False).export(dst)


def mirror_mano_assets(src_dir: Path, dst_dir: Path) -> None:
    """Produce a right-hand asset set from the left-hand MANO source asset by
    reflecting across the sagittal plane (x -> -x). Idempotent. Rules:
      meshes         : negate x, flip winding (visuals AND collision hulls)
      joint origin   : negate x (parent-relative vector)
      joint axis     : (x,y,z) -> (x,-y,-z)  [proper-rotation equivalent of the
                       reflected axis, so joint limits are unchanged and the
                       mirrored hand at qpos q is the exact reflection]
      inertial       : com x -> -x; products ixy, ixz -> negated (M I M)
    """
    if (dst_dir / "mano_hand_planar.urdf").exists():
        return
    for sub in ("links_planar", "links_hull"):
        (dst_dir / sub).mkdir(parents=True, exist_ok=True)
        for obj in sorted((src_dir / sub).glob("*.obj")):
            _mirror_mesh(obj, dst_dir / sub / obj.name)

    tree = ET.parse(src_dir / "mano_hand_planar.urdf")
    robot = tree.getroot()
    for j in robot.findall("joint"):
        o = j.find("origin")
        if o is not None and o.get("xyz"):
            x, y, z = (float(t) for t in o.get("xyz").split())
            o.set("xyz", f"{-x:.6f} {y:.6f} {z:.6f}")
        a = j.find("axis")
        if a is not None and a.get("xyz"):
            x, y, z = (float(t) for t in a.get("xyz").split())
            a.set("xyz", f"{x:.6f} {-y:.6f} {-z:.6f}")
    for inl in robot.iter("inertial"):
        o = inl.find("origin")
        if o is not None and o.get("xyz"):
            x, y, z = (float(t) for t in o.get("xyz").split())
            o.set("xyz", f"{-x:.6f} {y:.6f} {z:.6f}")
        i = inl.find("inertia")
        if i is not None and i.get("ixy") is not None:
            i.set("ixy", f"{-float(i.get('ixy')):.9e}")
            i.set("ixz", f"{-float(i.get('ixz')):.9e}")
    ET.indent(robot)
    tree.write(dst_dir / "mano_hand_planar.urdf", xml_declaration=True, encoding="utf-8")


@dataclass
class ManoR(ManoL):
    """Right hand: the left-hand MANO source asset mirrored across x=0."""

    name: str = "ManoR"
    morph_file: str = str(_MANO_RIGHT_DIR / "mano_hand_planar.urdf")
    gripper_name: str | None = "ManoGrasp"
    base_pos: tuple[float, float, float] = (_HOME_X, -_HOME_Y, _HOME_Z)  # right at -y
    # pre-yawed home (world-yaw pi = intrinsic (+pi/2, 0, pi), MHR precedent):
    # the mirrored asset's fingers point +x_hand, so without the yaw the pair
    # reads as x-mirrored, not centerline-mirrored — this puts both hands
    # fingers -x, thumbs inboard
    default_arm_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, math.pi / 2, 0.0, math.pi)

    def __post_init__(self) -> None:
        # pure file ops (no genesis) -> safe at construction time
        mirror_mano_assets(_MANO_DIR, _MANO_RIGHT_DIR)


# convenience aliases so `--robots mano-r` / `mano-l` / `mano` also resolve
for _alias in ("mano-r", "mano_r", "manor", "mano"):
    ROBOT_MODEL_REGISTRY.setdefault(_alias, ManoR)
for _alias in ("mano-l", "mano_l", "manol"):
    ROBOT_MODEL_REGISTRY.setdefault(_alias, ManoL)
