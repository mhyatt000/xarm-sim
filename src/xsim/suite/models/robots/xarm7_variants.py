"""xArm7 variants carrying robosuite grippers as attached entities.

The arm is assets/xarm7_nogripper.urdf (chain ends at link_eef). link_eef is
empirically identical to robosuite's right_hand mount frame — attaching their
xarm7_gripper.xml at identity reproduces the native URDF fingers to 0.02 mm —
so the flange->mount transform is identity and each gripper's own mount_*
fields carry any base offset.
"""

from __future__ import annotations

from dataclasses import dataclass

from xsim.suite.models.cameras import CameraSpec
from xsim.suite.models.robots.xarm7 import PROJECT_ROOT, XArm7


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
