"""Environment layer that owns robots and the action interface."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import gymnasium as gym
import numpy as np

from xsim.suite.environments.base import GenesisEnv
from xsim.suite.models.robots import create_robot_model
from xsim.suite.robots import Robot


class RobotEnv(GenesisEnv):
    """Env with robots: owns Robot instances, defines the action interface from
    their controllers, and fans actions out. Envs never touch actuators —
    all control flows through robot.control()."""

    def __init__(self, robots: str | list[str] = "XArm7", **kwargs):
        names = [robots] if isinstance(robots, str) else list(robots)
        self.robots = [Robot(create_robot_model(name)) for name in names]
        super().__init__(**kwargs)

    def _setup_references(self) -> None:
        super()._setup_references()
        for robot in self.robots:
            robot.setup()
        lows, highs = zip(*(robot.action_limits for robot in self.robots))
        self.action_space = gym.spaces.Box(
            np.concatenate(lows).astype(np.float32),
            np.concatenate(highs).astype(np.float32),
            dtype=np.float32,
        )

    @property
    def action_dim(self) -> int:
        return sum(r.action_dim for r in self.robots)

    def _setup_observables(self) -> OrderedDict[str, Callable[[], np.ndarray]]:
        observables = super()._setup_observables()
        for i, robot in enumerate(self.robots):
            pf = f"robot{i}_"
            observables[pf + "joint_pos"] = lambda robot=robot: robot.joint_positions
            observables[pf + "joint_vel"] = lambda robot=robot: robot.joint_velocities
            observables[pf + "eef_pos"] = lambda robot=robot: robot.ee_pos
            observables[pf + "eef_quat"] = lambda robot=robot: robot.ee_quat
            observables[pf + "gripper_norm"] = lambda robot=robot: [robot.gripper_norm]
        return observables

    def _pre_action(self, action) -> None:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(
                f"Action has dimension {a.shape[0]}, expected {self.action_dim}"
            )
        offset = 0
        for robot in self.robots:
            robot.control(a[offset : offset + robot.action_dim])
            offset += robot.action_dim

    def _reset_internal(self) -> None:
        super()._reset_internal()
        for robot in self.robots:
            robot.reset()
