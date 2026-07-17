"""xArm7 robot description."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from xsim.suite.models.cameras import CameraSpec, look_offset_T
from xsim.suite.models.mounts import Mount, PlateMount
from xsim.suite.models.robots.robot_model import RobotModel

PROJECT_ROOT = Path(__file__).resolve().parents[5]

# Low ready pose matching the real demos' start, TCP ~ (0.34, 0, 0.10) top-down.
_DEFAULT_ARM_QPOS = tuple(math.radians(v) for v in (0.0, -26.2, 0.0, 13.5, 0.0, 25.0, 90.0))

# RealSense D435 colour at 640x480: fx ~ 617 -> vFOV ~ 42.6 (no calibration data).
REALSENSE_FOV_DEG = 42.5
# Side-mounted wrist RealSense, matched frame-by-frame against the real stream
# (candidate "P1" of the 2026-07-02 mount sweep, carried over from task_env.py).
_XARM7_CAMERAS = (
    CameraSpec(
        "wrist",
        fov_deg=REALSENSE_FOV_DEG,
        attach_link="link_tcp",
        attach_offset=look_offset_T(
            back=0.14, side=0.085, lift=-0.03, pitch_deg=-5.0, yaw_deg=25.0, roll_deg=-90.0
        ),
    ),
)


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
    cameras: tuple[CameraSpec, ...] = _XARM7_CAMERAS
    # the real arm bolts onto a 5x8 in baseplate; TableArena.top_z = -0.01
    # already prices in its 1 cm thickness, this adds the visible body
    mount: Mount | None = field(default_factory=PlateMount)
