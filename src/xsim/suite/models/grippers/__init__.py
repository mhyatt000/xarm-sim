"""Gripper models and factory."""

from __future__ import annotations

from xsim.suite.models.grippers.gripper_model import GRIPPER_REGISTRY, GripperModel
from xsim.suite.models.grippers.robosuite_grippers import (
    BDGripper,
    FourierLeftHand,
    FourierRightHand,
    InspireLeftHand,
    InspireRightHand,
    JacoThreeFingerGripper,
    PandaGripper,
    RethinkGripper,
    Robotiq85Gripper,
    Robotiq140Gripper,
    RobotiqSGripper,
)
from xsim.suite.models.grippers.xarm7_gripper import XArm7Gripper

__all__ = [
    "BDGripper",
    "FourierLeftHand",
    "FourierRightHand",
    "GRIPPER_REGISTRY",
    "GripperModel",
    "InspireLeftHand",
    "InspireRightHand",
    "JacoThreeFingerGripper",
    "PandaGripper",
    "RethinkGripper",
    "Robotiq140Gripper",
    "Robotiq85Gripper",
    "RobotiqSGripper",
    "XArm7Gripper",
    "gripper_factory",
]


def gripper_factory(name: str) -> GripperModel:
    if name not in GRIPPER_REGISTRY:
        raise ValueError(f"unknown gripper {name!r}; registered: {sorted(GRIPPER_REGISTRY)}")
    return GRIPPER_REGISTRY[name]()
