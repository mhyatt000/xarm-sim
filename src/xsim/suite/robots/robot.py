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
        a = np.asarray(action, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.action_dim:
            raise ValueError(
                f"action has shape {a.shape}, expected (n_envs, {self.action_dim})"
            )
        offset = 0
        for c in self._controllers:
            c.run(a[:, offset : offset + c.action_dim])
            offset += c.action_dim

    def reset(self, envs_idx=None) -> None:
        self.entity.set_qpos(
            self._init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=False
        )
        for c in self._controllers:
            c.reset()

    def set_arm_qpos(self, q: np.ndarray, envs_idx=None) -> None:
        """Seat the arm at ``q`` (n_envs, arm_dofs) with the gripper at its
        defaults; ``envs_idx`` selects which of the n_envs rows are applied."""
        qpos = self._init_qpos.unsqueeze(0).repeat(q.shape[0], 1).clone()
        qpos[:, : self.model.arm_dofs] = torch.as_tensor(
            q, device=gs.device, dtype=gs.tc_float
        )
        if envs_idx is not None:
            qpos = qpos[torch.as_tensor(np.atleast_1d(envs_idx), device=gs.device)]
        self.entity.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=False)
        for c in self._controllers:
            c.reset()

    def ik(self, pose: torch.Tensor, from_current: bool = False) -> np.ndarray:
        """Arm joint targets (n_envs, 7) for EE poses [n_envs, 7] = [x,y,z,qw,qx,qy,qz].

        ``from_current`` seeds the solve at each env's live qpos, returning the
        branch nearest the arm's actual configuration — required when the
        result is used as a per-tick regression label (home seeding can jump
        branches, making labels discontinuous in state; see img-v8b).
        """
        pose = pose.to(gs.device)
        init_qpos = (
            None
            if from_current
            else self._init_qpos.unsqueeze(0).expand(pose.shape[0], -1)
            if self.model.ik_init_at_home
            else None
        )
        q = self.entity.inverse_kinematics(
            link=self._ee_link,
            pos=pose[:, :3],
            quat=pose[:, 3:7],
            init_qpos=init_qpos,
            max_samples=self.model.ik_max_samples,
            max_solver_iters=self.model.ik_max_solver_iters,
            damping=self.model.ik_damping,
            dofs_idx_local=self._arm_idx,
        )
        return np.asarray(q[:, self._arm_idx].detach().cpu(), dtype=np.float64)

    @property
    def joint_positions(self) -> np.ndarray:
        q = np.asarray(self.entity.get_dofs_position().detach().cpu())
        return q[:, : self.model.arm_dofs]

    @property
    def joint_velocities(self) -> np.ndarray:
        v = np.asarray(self.entity.get_dofs_velocity().detach().cpu())
        return v[:, : self.model.arm_dofs]

    @property
    def ee_pos(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_pos().detach().cpu())

    @property
    def ee_quat(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_quat().detach().cpu())

    @property
    def ee_vel(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_vel().detach().cpu())

    @property
    def gripper_norm(self) -> np.ndarray:
        q = np.asarray(self.entity.get_dofs_position().detach().cpu())
        if self.gripper is None:
            return np.ones(q.shape[0], dtype=np.float64)
        g = q[:, self.model.arm_dofs]
        return np.clip(1.0 - g / self.gripper.close_dof, 0.0, 1.0)
