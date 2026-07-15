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
