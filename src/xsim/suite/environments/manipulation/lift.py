"""Lift task: pick a red cube off the table."""

from __future__ import annotations

import numpy as np

from xsim.suite.environments.manipulation.manipulation_env import ManipulationEnv
from xsim.suite.models import BoxObject, TableArena, Task
from xsim.suite.utils import UniformRandomSampler


class Lift(ManipulationEnv):
    """xArm7 red-cube lift: sparse reward when the cube rises above the table.
    Suite counterpart of robosuite's Lift, composed from TableArena + BoxObject."""

    def __init__(
        self,
        robots: str | list[str] = "XArm7",
        cube_size: float = 0.03175,
        cube_color: tuple[float, float, float] = (0.48, 0.05, 0.04),
        x_range: tuple[float, float] = (0.35, 0.58),
        y_range: tuple[float, float] = (-0.15, 0.15),
        lift_height: float = 0.05,
        reward_shaping: bool = False,
        placement_initializer: UniformRandomSampler | None = None,
        **kwargs,
    ):
        self.cube_size = cube_size
        self.cube_color = cube_color
        self.lift_height = lift_height
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer or UniformRandomSampler(
            x_range, y_range
        )
        super().__init__(robots=robots, **kwargs)

    def _load_model(self) -> None:
        self.arena = TableArena()
        s = self.cube_size
        self.cube = BoxObject(
            "cube", size=(s, s, s), color=self.cube_color, friction=2.0
        )
        self.model = Task(
            self.arena, [robot.model for robot in self.robots], [self.cube]
        )

    def _setup_observables(self):
        observables = super()._setup_observables()
        observables["cube_pos"] = self.cube.get_pos
        observables["cube_quat"] = self.cube.get_quat
        observables["robot0_gripper_to_cube_pos"] = (
            lambda: self.cube.get_pos() - self.robots[0].ee_pos
        )
        return observables

    def _reset_internal(self, envs_idx=None) -> None:
        super()._reset_internal(envs_idx)
        n = self.n_envs if envs_idx is None else len(np.atleast_1d(envs_idx))
        x, y, yaw = self.placement_initializer.sample(self.np_random, n)
        self.cube.set_pose(
            x, y, self.arena.top_z + self.cube.top_offset, yaw, envs_idx=envs_idx
        )

    def reward(self, action=None) -> np.ndarray:
        success = self._check_success()
        if self.reward_shaping:
            shaped = 0.5 * (
                1.0 - np.tanh(10.0 * self._gripper_to_target_dist(self.cube.get_pos()))
            )
            return np.where(success, 1.0, shaped).astype(np.float32)
        return success.astype(np.float32)

    def _check_success(self) -> np.ndarray:
        return self.cube.get_pos()[:, 2] > self.arena.top_z + self.lift_height

    def _check_terminated(self) -> np.ndarray:
        return self._check_success()
