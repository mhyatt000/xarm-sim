"""xArm7 robot description."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from xsim.suite.models.cam_space import MountSampler
from xsim.suite.models.cameras import CameraSpec, look_offset_T
from xsim.suite.models.mounts import Mount, PlateMount
from xsim.suite.models.robots.robot_model import RobotModel

PROJECT_ROOT = Path(__file__).resolve().parents[5]

# The real rig's home pose (verified against the live arm 2026-07-19),
# TCP ~ (0.324, 0, 0.288) top-down. The old (0, -26.2, 0, 13.5, 0, 25, 90)
# low-ready start drifted from reality: real rollouts began ~5 sigma outside
# the DAgger visit distribution and img-v5 froze at a constant action.
_DEFAULT_ARM_QPOS = tuple(math.radians(v) for v in (0.0, -45.0, 0.0, 35.0, 0.0, 65.0, 90.0))

# RealSense D435 colour at 640x480: fx ~ 617 -> vFOV ~ 42.6 (no calibration data).
REALSENSE_FOV_DEG = 42.5
# Side-mounted wrist RealSense, matched frame-by-frame against the real stream
# (candidate "P1" of the 2026-07-02 mount sweep, carried over from task_env.py).
_WRIST_OFFSET = look_offset_T(
    back=0.14, side=0.085, lift=-0.03, pitch_deg=-5.0, yaw_deg=25.0, roll_deg=-90.0
)
_XARM7_CAMERAS = (
    CameraSpec("wrist", fov_deg=REALSENSE_FOV_DEG, attach_link="link_tcp",
               attach_offset=_WRIST_OFFSET),
)
# Randomized mount, link_tcp frame (derivation + plot: scripts/cam_wrist.py):
# apex = gripper base (URDF joint_tcp origin: 0.172 m behind the TCP), axis =
# common perpendicular of the fingertip line and the base->TCP line (exactly
# +x at qpos 0); positions in the ellipsoid at 11 cm along the axis, lookats
# in the oblong patch at the TCP. Up copied from the committed mount so
# sampled frames roll like the real camera.
_WRIST_MOUNT = MountSampler(
    name="wrist",
    fov_deg=REALSENSE_FOV_DEG,
    attach_link="link_tcp",
    up=tuple(row[1] for row in _WRIST_OFFSET[:3]),
    apex=(0.0, 0.0, -0.172),
    axis=(1.0, 0.0, 0.0),
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
    # wrist mount sampled per env/episode; False pins the committed bracket
    randomize_wrist: bool = True

    def __post_init__(self) -> None:
        if self.randomize_wrist:
            self.cameras = tuple(
                _WRIST_MOUNT if c.name == "wrist" else c for c in self.cameras
            )
