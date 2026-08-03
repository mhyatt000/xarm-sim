"""xArm7 variants carrying robosuite grippers as attached entities.

The arm is assets/xarm7_nogripper.urdf (chain ends at link_eef). link_eef is
empirically identical to robosuite's right_hand mount frame — attaching their
xarm7_gripper.xml at identity reproduces the native URDF fingers to 0.02 mm —
so the flange->mount transform is identity and each gripper's own mount_*
fields carry any base offset.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xsim.suite.models.cameras import CameraSpec
from xsim.suite.models.robots.xarm7 import PROJECT_ROOT, XArm7


def roll_eef_urdf(src: Path, deg: float = 180.0, link: str = "link_eef") -> Path:
    """Sibling URDF with everything mounted on ``link`` rolled about its z axis
    (the flange/tool axis). Only the mount joints' origins move — the arm and
    the end-effector's internal geometry are untouched — so a gripper's
    measured TCP transforms by the same Rz. Idempotent; writes *_roll<deg>.urdf
    next to ``src`` so relative mesh paths keep resolving."""
    dst = src.with_name(f"{src.stem}_roll{int(deg)}.urdf")
    if dst.exists():
        return dst
    a = math.radians(deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])

    tree = ET.parse(src)
    for joint in tree.getroot().findall("joint"):
        if joint.find("parent").get("link") != link:
            continue
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        r, p, y = (float(v) for v in origin.get("rpy", "0 0 0").split())
        # URDF rpy is fixed-axis: R = Rz(y) Ry(p) Rx(r); pre-multiplying by Rz
        # rotates the mount frame in the parent (flange) frame
        cr, sr, cp, sp, cy, sy = (
            math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        )
        R = Rz @ np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        origin.set("xyz", " ".join(f"{v:.9g}" for v in Rz @ xyz))
        origin.set(
            "rpy",
            " ".join(f"{v:.9g}" for v in (
                math.atan2(R[2, 1], R[2, 2]),
                math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])),
                math.atan2(R[1, 0], R[0, 0]),
            )),
        )
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    return dst


@dataclass
class XArm7RS(XArm7):
    """The bare arm; leaves pick a gripper via gripper_name."""

    name: str = "XArm7RS"
    morph_file: str = str(PROJECT_ROOT / "assets/xarm7_nogripper.urdf")
    ee_link_name: str = "link_eef"
    gripper_name: str | None = None
    gripper_mount_link: str = "link_eef"
    cameras: tuple[CameraSpec, ...] = ()  # no calibrated wrist mount
    randomize_wrist: bool = False


@dataclass
class XArm7Rethink(XArm7RS):
    name: str = "XArm7Rethink"
    gripper_name: str | None = "RethinkGripper"


@dataclass
class XArm7Panda(XArm7RS):
    name: str = "XArm7Panda"
    gripper_name: str | None = "PandaGripper"


@dataclass
class XArm7BD(XArm7RS):
    name: str = "XArm7BD"
    gripper_name: str | None = "BDGripper"


# The GR1 hands' recorded base pose (Rx+90) is the humanoid wrist convention;
# on the xArm flange it points the fingers back along the forearm. The
# robot-side 180-about-x below composes to Rx(-90): fingers along +z_f (away
# from the flange, like the native gripper). Hand-base-frame tcp_* fields are
# invariant under this flip.
_HAND_MOUNT_QUAT = (0.0, 1.0, 0.0, 0.0)


@dataclass
class XArm7FourierL(XArm7RS):
    name: str = "XArm7FourierL"
    gripper_name: str | None = "FourierLeftHand"
    gripper_mount_quat: tuple[float, float, float, float] = _HAND_MOUNT_QUAT


@dataclass
class XArm7FourierR(XArm7RS):
    name: str = "XArm7FourierR"
    gripper_name: str | None = "FourierRightHand"
    gripper_mount_quat: tuple[float, float, float, float] = _HAND_MOUNT_QUAT


@dataclass
class XArm7InspireL(XArm7RS):
    name: str = "XArm7InspireL"
    gripper_name: str | None = "InspireLeftHand"
    gripper_mount_quat: tuple[float, float, float, float] = _HAND_MOUNT_QUAT


@dataclass
class XArm7InspireR(XArm7RS):
    name: str = "XArm7InspireR"
    gripper_name: str | None = "InspireRightHand"
    gripper_mount_quat: tuple[float, float, float, float] = _HAND_MOUNT_QUAT


@dataclass
class XArm7Jaco(XArm7RS):
    name: str = "XArm7Jaco"
    gripper_name: str | None = "JacoThreeFingerGripper"


@dataclass
class XArm7Robotiq85(XArm7RS):
    name: str = "XArm7Robotiq85"
    gripper_name: str | None = "Robotiq85Gripper"


@dataclass
class XArm7Robotiq140(XArm7RS):
    name: str = "XArm7Robotiq140"
    gripper_name: str | None = "Robotiq140Gripper"


@dataclass
class XArm7RobotiqS(XArm7RS):
    name: str = "XArm7RobotiqS"
    gripper_name: str | None = "RobotiqSGripper"


@dataclass
class XArm7RukaR(XArm7RS):
    """RUKA v2 right hand, baked into the merged URDF as 21 trailing dofs.

    The CAD-derived ruka_mount graft (40 deg pitch, +/-90 deg yaw) is part of
    the URDF and visually verified on the real rig — do not retune. Self-
    collision off: teleop drops all 231 intra-hand pairs; curled postures jam
    otherwise.
    """

    name: str = "XArm7RukaR"
    morph_file: str = str(PROJECT_ROOT / "assets/ruka/xarm7_ruka_right.urdf")
    gripper_name: str | None = "RukaHand"
    gripper_mount_link: str = ""  # fingers baked into the robot morph
    self_collision: bool = False


@dataclass
class XArm7RukaL(XArm7RukaR):
    name: str = "XArm7RukaL"
    morph_file: str = str(PROJECT_ROOT / "assets/ruka/xarm7_ruka_left.urdf")
    gripper_name: str | None = "RukaHandL"
