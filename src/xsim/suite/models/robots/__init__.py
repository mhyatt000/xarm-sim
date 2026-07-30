"""Robot models and factory."""

from __future__ import annotations

from xsim.suite.models.robots.dxarm7 import DXArm7L, DXArm7R
from xsim.suite.models.robots.dxarm7_variants import (
    DXArm7LBD,
    DXArm7LFourier,
    DXArm7LInspire,
    DXArm7LJaco,
    DXArm7LPanda,
    DXArm7LRethink,
    DXArm7LRobotiq85,
    DXArm7LRobotiq140,
    DXArm7LRobotiqS,
    DXArm7RBD,
    DXArm7RFourier,
    DXArm7RInspire,
    DXArm7RJaco,
    DXArm7RPanda,
    DXArm7RRethink,
    DXArm7RRobotiq85,
    DXArm7RRobotiq140,
    DXArm7RRobotiqS,
)
from xsim.suite.models.robots.mano import ManoL, ManoR
from xsim.suite.models.robots.robot_model import ROBOT_MODEL_REGISTRY, RobotModel
from xsim.suite.models.robots.xarm7 import XArm7
from xsim.suite.models.robots.xarm7_variants import (
    XArm7BD,
    XArm7FourierL,
    XArm7FourierR,
    XArm7InspireL,
    XArm7InspireR,
    XArm7Jaco,
    XArm7Panda,
    XArm7Rethink,
    XArm7Robotiq85,
    XArm7Robotiq140,
    XArm7RobotiqS,
    XArm7RS,
)

__all__ = [
    "DXArm7L",
    "DXArm7LBD",
    "DXArm7LFourier",
    "DXArm7LInspire",
    "DXArm7LJaco",
    "DXArm7LPanda",
    "DXArm7LRethink",
    "DXArm7LRobotiq140",
    "DXArm7LRobotiq85",
    "DXArm7LRobotiqS",
    "DXArm7R",
    "DXArm7RBD",
    "DXArm7RFourier",
    "DXArm7RInspire",
    "DXArm7RJaco",
    "DXArm7RPanda",
    "DXArm7RRethink",
    "DXArm7RRobotiq140",
    "DXArm7RRobotiq85",
    "DXArm7RRobotiqS",
    "ManoL",
    "ManoR",
    "ROBOT_MODEL_REGISTRY",
    "RobotModel",
    "XArm7",
    "XArm7BD",
    "XArm7FourierL",
    "XArm7FourierR",
    "XArm7InspireL",
    "XArm7InspireR",
    "XArm7Jaco",
    "XArm7Panda",
    "XArm7RS",
    "XArm7Rethink",
    "XArm7Robotiq140",
    "XArm7RobotiqS",
    "XArm7Robotiq85",
    "create_robot_model",
]


def create_robot_model(name: str) -> RobotModel:
    if name not in ROBOT_MODEL_REGISTRY:
        raise ValueError(f"unknown robot {name!r}; registered: {sorted(ROBOT_MODEL_REGISTRY)}")
    return ROBOT_MODEL_REGISTRY[name]()
