"""xArm7 gripper description."""

from __future__ import annotations

from dataclasses import dataclass

from xsim.suite.models.grippers.gripper_model import GripperModel


@dataclass
class XArm7Gripper(GripperModel):
    # xArm gripper joint convention is 0.0 = open, 0.85 = hard fully closed;
    # 0.58 is the task grasp target that holds a 31.75 mm cube without driving
    # through it (values carried over from XARM7_ROBOT_CFG in src/xsim/task_env.py).
    name: str = "XArm7Gripper"
    n_dofs: int = 6
    open_dof: float = 0.0
    close_dof: float = 0.85
    grasp_dof: float = 0.58
    finger_link_names: tuple[str, str] = ("left_finger", "right_finger")
    kp: float = 350.0
    kv: float = 35.0
    force_limit: float = 50.0
    # keep the TCP near the upper half of a 31.75 mm cube: half-width 0.015875
    # + 0.002125 puts the grasp TCP 18 mm above the table plane; lower and the
    # gripper visibly clips through the block
    grasp_dz: float = 0.002125
    max_open_width: float = 0.086
    held_radius: float = 0.035
    close_min_s: float = 0.4  # 12 ticks at the 30 Hz control rate
    close_timeout_s: float = 1.0  # 30 ticks
    open_s: float = 1 / 3  # 10 ticks
    # seated on the 31.75 mm cube the norm plateaus near 31.75/86 ~ 0.37
    hold_norm_lo: float = 0.20
    hold_norm_hi: float = 0.85
