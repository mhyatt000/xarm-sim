"""Runtime robot that composes part controllers over a Genesis entity."""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from xsim.suite.controllers import Controller, GripperController, JointPositionController
from xsim.suite.models.grippers import GripperModel, gripper_factory
from xsim.suite.models.robots import RobotModel


class Robot:
    """Runtime robot: binds the entity created from its RobotModel and composes
    part controllers (arm + gripper from the gripper factory). RobotEnv calls
    control(action); the robot fans the action out to its parts."""

    def __init__(self, robot_model: RobotModel):
        self.model = robot_model
        self.gripper: GripperModel | None = (
            gripper_factory(robot_model.gripper_name) if robot_model.gripper_name else None
        )
        self.entity = None
        self.arm_controller: JointPositionController | None = None
        self.gripper_controller: GripperController | None = None
        self._controllers: list[Controller] = []

    def setup(self) -> None:
        if self.model.entity is None:
            raise RuntimeError(
                f"RobotModel {self.model.name!r} has no bound entity; "
                "was the Task added to a scene?"
            )
        self.entity = self.model.entity
        self._arm_idx = torch.arange(self.model.arm_dofs, device=gs.device)
        self.arm_controller = JointPositionController(
            self.entity,
            self._arm_idx,
            self.model.arm_kp,
            self.model.arm_kv,
            self.model.arm_force_limit,
        )
        if self.gripper is not None:
            self._finger_idx = torch.arange(
                self.model.arm_dofs,
                self.model.arm_dofs + self.gripper.n_dofs,
                device=gs.device,
            )
            self.gripper_controller = GripperController(
                self.entity, self._finger_idx, self.gripper
            )
        self._controllers = [
            c for c in (self.arm_controller, self.gripper_controller) if c is not None
        ]
        for c in self._controllers:
            c.setup()
        self._ee_link = self.entity.get_link(self.model.ee_link_name)
        init = list(self.model.default_arm_qpos) + (
            list(self.gripper.default_dofs) if self.gripper else []
        )
        self._init_qpos = torch.tensor(init, device=gs.device, dtype=gs.tc_float)

    @property
    def action_dim(self) -> int:
        return sum(c.action_dim for c in self._controllers)

    @property
    def action_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.arm_controller.joint_limits
        if self.gripper_controller is not None:
            lo = np.concatenate([lo, np.array([0.0], dtype=np.float64)])
            hi = np.concatenate([hi, np.array([1.0], dtype=np.float64)])
        return lo, hi

    def control(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(
                f"action has {a.shape[0]} dims, expected {self.action_dim}"
            )
        offset = 0
        for c in self._controllers:
            c.run(a[offset : offset + c.action_dim])
            offset += c.action_dim

    def reset(self, envs_idx=None) -> None:
        self.entity.set_qpos(
            self._init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=False
        )
        for c in self._controllers:
            c.reset()

    @property
    def joint_positions(self) -> np.ndarray:
        q = np.asarray(self.entity.get_dofs_position().detach().cpu()).reshape(-1)
        return q[: self.model.arm_dofs]

    @property
    def joint_velocities(self) -> np.ndarray:
        v = np.asarray(self.entity.get_dofs_velocity().detach().cpu()).reshape(-1)
        return v[: self.model.arm_dofs]

    @property
    def ee_pos(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_pos().detach().cpu()).reshape(-1)

    @property
    def ee_quat(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_quat().detach().cpu()).reshape(-1)

    @property
    def gripper_norm(self) -> float:
        if self.gripper is None:
            return 1.0
        q = np.asarray(self.entity.get_dofs_position().detach().cpu()).reshape(-1)
        g = q[self.model.arm_dofs]
        return float(np.clip(1.0 - g / self.gripper.close_dof, 0.0, 1.0))
