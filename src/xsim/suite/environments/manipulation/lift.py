"""Lift task: pick a red cube off the table."""

from __future__ import annotations

import numpy as np

from xsim.suite.environments.manipulation.manipulation_env import ManipulationEnv
from xsim.suite.models import BoxObject, TableArena, TableEZ, Task
from xsim.suite.utils import UniformRandomSampler


class Lift(ManipulationEnv):
    """xArm7 red-cube lift: sparse reward for a stable, held lift.
    Suite counterpart of robosuite's Lift, composed from TableArena + BoxObject.

    Success requires, for ``success_hold_ticks`` CONSECUTIVE control steps
    (default 1): cube above ``lift_height``, cube speed RELATIVE TO THE
    END-EFFECTOR under ``success_max_speed``, cube within
    ``success_eef_xy_radius`` of the end-effector in the xy plane, and
    cube-robot contact. Relative speed distinguishes a fling (cube fast vs the
    hand) from a brisk transport (cube moving with the hand), so the hold-tick
    count no longer has to do that job.
    """

    arena_class: type[TableArena] = TableArena

    def __init__(
        self,
        robots: str | list[str] = "XArm7",
        cube_size: float = 0.03175,
        cube_color: tuple[float, float, float] = (0.48, 0.05, 0.04),
        # cube spawn: x 200-400mm, y +-1ft minus the cube half-extent (table is
        # exactly 2ft wide, y +-0.3048; +-0.288 keeps the 1.25in cube fully on
        # it). Matches the production training/eval distribution.
        x_range: tuple[float, float] = (0.20, 0.40),
        y_range: tuple[float, float] = (-0.288, 0.288),
        lift_height: float = 0.05,
        success_hold_ticks: int = 1,
        success_max_speed: float = 0.10,
        success_eef_xy_radius: float = 0.08,
        reward_shaping: bool = False,
        randomize_cameras: bool = True,
        placement_initializer: UniformRandomSampler | None = None,
        **kwargs,
    ):
        self.randomize_cameras = randomize_cameras
        self.cube_size = cube_size
        self.cube_color = cube_color
        self.lift_height = lift_height
        self.success_hold_ticks = success_hold_ticks
        self.success_max_speed = success_max_speed
        self.success_eef_xy_radius = success_eef_xy_radius
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer or UniformRandomSampler(
            x_range, y_range
        )
        super().__init__(robots=robots, **kwargs)
        self._success_hold = np.zeros(self.n_envs, dtype=np.int64)

    def _load_model(self) -> None:
        self.arena = self.arena_class(randomize_cameras=self.randomize_cameras)
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
        if envs_idx is None:
            self._success_hold[:] = 0
        else:
            self._success_hold[np.asarray(envs_idx)] = 0

    def reward(self, action=None) -> np.ndarray:
        success = self._check_success()
        if self.reward_shaping:
            shaped = 0.5 * (
                1.0 - np.tanh(10.0 * self._gripper_to_target_dist(self.cube.get_pos()))
            )
            return np.where(success, 1.0, shaped).astype(np.float32)
        return success.astype(np.float32)

    def _robot_contact(self) -> np.ndarray:
        """Per-env: is the cube in contact with any robot body?"""
        contacts = self.cube.entity.get_contacts(with_entity=self.robots[0].entity)
        mask = contacts.get("valid_mask") if isinstance(contacts, dict) else None
        if mask is None:
            return np.zeros(self.n_envs, dtype=bool)
        mask = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask)
        if mask.ndim == 1:  # non-parallelized scene: (n_contacts,)
            return np.full(self.n_envs, bool(mask.any()))
        return mask.any(axis=-1)

    def _raw_success(self) -> np.ndarray:
        """Instantaneous held-lift condition; success needs it for
        ``success_hold_ticks`` consecutive control steps."""
        pos = self.cube.get_pos()
        high = pos[:, 2] > self.arena.top_z + self.lift_height
        rel_vel = self.cube.get_vel() - self.robots[0].ee_vel
        slow = np.linalg.norm(rel_vel, axis=-1) < self.success_max_speed
        near = (
            np.linalg.norm(pos[:, :2] - self.robots[0].ee_pos[:, :2], axis=-1)
            < self.success_eef_xy_radius
        )
        return high & slow & near & self._robot_contact()

    def _post_action(self, action):
        # one hold-counter update per control step (base calls _check_success
        # multiple times per step; those reads must be idempotent)
        raw = self._raw_success()
        self._success_hold = np.where(raw, self._success_hold + 1, 0)
        return super()._post_action(action)

    def _check_success(self) -> np.ndarray:
        return self._success_hold >= self.success_hold_ticks

    def _check_terminated(self) -> np.ndarray:
        return self._check_success()


class LiftEZ(Lift):
    """Lift on the easier TableEZ arena: cameras sampled in 10 cm balls around
    the calibrated rig, cube spawn narrowed to |y| <= 3 in (x unchanged), and
    the arm starting at HOME by default (pass ``init_tcp_box`` explicitly to
    randomize the start, e.g. a hover box over the cube spawn region)."""

    arena_class = TableEZ

    def __init__(
        self,
        robots: str | list[str] = "XArm7",
        y_range: tuple[float, float] = (-0.0762, 0.0762),
        init_tcp_box: tuple | None = None,
        **kwargs,
    ):
        super().__init__(robots=robots, y_range=y_range,
                         init_tcp_box=init_tcp_box, **kwargs)
