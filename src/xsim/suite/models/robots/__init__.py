"""Robot models and factory."""

from __future__ import annotations

from xsim.suite.models.robots.robot_model import ROBOT_MODEL_REGISTRY, RobotModel
from xsim.suite.models.robots.xarm7 import XArm7

__all__ = ["ROBOT_MODEL_REGISTRY", "RobotModel", "XArm7", "create_robot_model"]


def create_robot_model(name: str) -> RobotModel:
    if name not in ROBOT_MODEL_REGISTRY:
        raise ValueError(f"unknown robot {name!r}; registered: {sorted(ROBOT_MODEL_REGISTRY)}")
    return ROBOT_MODEL_REGISTRY[name]()
