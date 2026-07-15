"""xArm7 robot description."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from xsim.suite.models.robots.robot_model import RobotModel

PROJECT_ROOT = Path(__file__).resolve().parents[5]

# Low ready pose matching the real demos' start, TCP ~ (0.34, 0, 0.10) top-down.
_DEFAULT_ARM_QPOS = tuple(math.radians(v) for v in (0.0, -26.2, 0.0, 13.5, 0.0, 25.0, 90.0))


@dataclass
class XArm7(RobotModel):
    name: str = "XArm7"
    morph_kind: Literal["urdf", "mjcf"] = "urdf"
    morph_file: str = str(PROJECT_ROOT / "xarm7_standalone.urdf")
    merge_fixed_links: bool = False
    arm_dofs: int = 7
    default_arm_qpos: tuple[float, ...] = _DEFAULT_ARM_QPOS
    ee_link_name: str = "link_tcp"
    # kp is the Genesis Franka default; kv tuned near critical damping via
    # scripts/kpv_tune.py (values carried over from XARM7_ROBOT_CFG in src/xsim/task_env.py).
    arm_kp: tuple[float, ...] = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
    arm_kv: tuple[float, ...] = (135.0, 135.0, 105.0, 105.0, 60.0, 60.0, 60.0)
    arm_force_limit: float = 50.0
    gripper_name: str | None = "XArm7Gripper"
