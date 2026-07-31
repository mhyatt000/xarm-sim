"""Stack tasks: build a tower from three colored cubes."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from xsim.suite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
    pose_mats,
)
from xsim.suite.models import BoxObject, TableArena, Task
from xsim.suite.utils import UniformRandomSampler


class Stack(ManipulationEnv):
    """xArm7 three-cube stack: sparse reward for a settled, released tower.
    Suite counterpart of robosuite's Stack, extended to three cubes
    (red, green, yellow) composed from TableArena + BoxObject.

    Success requires, for ``success_hold_ticks`` CONSECUTIVE control steps:
    the cubes form a tower in one of ``stack_orders`` (any order here; fixed
    red-green-yellow in ``StackRGY``) with the base cube resting on the table,
    every cube slower than ``success_max_speed``, and no cube in contact with
    the robot — the tower only counts once the hand has released it.

    A pair counts as stacked when the upper cube's center is within
    ``stack_xy_tol`` of the lower cube's in the xy plane and within
    ``stack_z_tol`` of one cube edge above it.
    """

    arena_class: type[TableArena] = TableArena
    # (base, middle, top) triples indexing ``self.cubes`` that count as a tower
    stack_orders: tuple[tuple[int, int, int], ...] = tuple(permutations(range(3)))

    def __init__(
        self,
        robots: str | list[str] = "XArm7",
        cube_size: float = 0.03175,
        cube_colors: tuple[tuple[float, float, float], ...] = (
            (0.48, 0.05, 0.04),  # red, matches the Lift cube
            (0.04, 0.30, 0.07),  # green
            (0.71, 0.55, 0.06),  # yellow
        ),
        # same spawn rectangle as Lift; three draws with pairwise rejection
        x_range: tuple[float, float] = (0.20, 0.40),
        y_range: tuple[float, float] = (-0.288, 0.288),
        min_separation: float = 0.10,
        stack_xy_tol: float = 0.02,
        stack_z_tol: float = 0.01,
        success_hold_ticks: int = 1,
        success_max_speed: float = 0.10,
        reward_shaping: bool = False,
        randomize_cameras: bool = True,
        placement_initializer: UniformRandomSampler | None = None,
        **kwargs,
    ):
        self.randomize_cameras = randomize_cameras
        self.cube_size = cube_size
        self.cube_colors = cube_colors
        self.min_separation = min_separation
        self.stack_xy_tol = stack_xy_tol
        self.stack_z_tol = stack_z_tol
        self.success_hold_ticks = success_hold_ticks
        self.success_max_speed = success_max_speed
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer or UniformRandomSampler(
            x_range, y_range
        )
        super().__init__(robots=robots, **kwargs)
        self._success_hold = np.zeros(self.n_envs, dtype=np.int64)

    def _load_model(self) -> None:
        self.arena = self.arena_class(randomize_cameras=self.randomize_cameras)
        s = self.cube_size
        self.cubes = [
            BoxObject(name, size=(s, s, s), color=color, friction=2.0)
            for name, color in zip(
                ("cube_red", "cube_green", "cube_yellow"), self.cube_colors
            )
        ]
        self.model = Task(
            self.arena, [robot.model for robot in self.robots], self.cubes
        )

    def _setup_observables(self):
        observables = super()._setup_observables()
        for cube in self.cubes:
            observables[f"{cube.name}_pos"] = cube.get_pos
            observables[f"{cube.name}_quat"] = cube.get_quat
            observables[f"robot0_gripper_to_{cube.name}_pos"] = (
                lambda cube=cube: cube.get_pos() - self.robots[0].ee_pos
            )
        return observables

    def _sample_placements(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, yaw) arrays of shape (n, 3) with pairwise xy separation of at
        least ``min_separation`` between the three cubes in every env."""
        xs, ys, yaws = (np.empty((n, 3)) for _ in range(3))
        todo = np.arange(n)
        for _ in range(100):
            for j in range(3):
                x, y, yaw = self.placement_initializer.sample(
                    self.np_random, len(todo)
                )
                xs[todo, j], ys[todo, j], yaws[todo, j] = x, y, yaw
            xy = np.stack([xs[todo], ys[todo]], axis=-1)  # (k, 3, 2)
            iu, ju = np.triu_indices(3, 1)
            dists = np.linalg.norm(xy[:, iu] - xy[:, ju], axis=-1)  # (k, 3)
            todo = todo[(dists < self.min_separation).any(axis=-1)]
            if todo.size == 0:
                return xs, ys, yaws
        raise RuntimeError(
            f"could not place 3 cubes with min_separation={self.min_separation} "
            f"inside x_range={self.placement_initializer.x_range} "
            f"y_range={self.placement_initializer.y_range}"
        )

    def _reset_internal(self, envs_idx=None) -> None:
        super()._reset_internal(envs_idx)
        n = self.n_envs if envs_idx is None else len(np.atleast_1d(envs_idx))
        x, y, yaw = self._sample_placements(n)
        for j, cube in enumerate(self.cubes):
            cube.set_pose(
                x[:, j],
                y[:, j],
                self.arena.top_z + cube.top_offset,
                yaw[:, j],
                envs_idx=envs_idx,
            )
        if envs_idx is None:
            self._success_hold[:] = 0
        else:
            self._success_hold[np.asarray(envs_idx)] = 0

    def reward(self, action=None) -> np.ndarray:
        success = self._check_success()
        if self.reward_shaping:
            # 0.375 per stacked pair (0.75 for a full tower) + up to 0.25 for
            # reaching the nearest cube; capped at 1.0 by the success branch
            reach = 0.25 * (1.0 - np.tanh(10.0 * self._min_gripper_cube_dist()))
            shaped = reach + 0.375 * self._best_stacked_pairs()
            return np.where(success, 1.0, shaped).astype(np.float32)
        return success.astype(np.float32)

    def _min_gripper_cube_dist(self) -> np.ndarray:
        return np.min(
            [self._gripper_to_target_dist(c.get_pos()) for c in self.cubes], axis=0
        )

    def _pair_on(self, top_pos: np.ndarray, base_pos: np.ndarray) -> np.ndarray:
        """Per-env: is the cube at ``top_pos`` stacked on the one at ``base_pos``?"""
        xy = (
            np.linalg.norm(top_pos[:, :2] - base_pos[:, :2], axis=-1)
            < self.stack_xy_tol
        )
        z = np.abs(top_pos[:, 2] - base_pos[:, 2] - self.cube_size) < self.stack_z_tol
        return xy & z

    def _best_stacked_pairs(self) -> np.ndarray:
        """Per-env count (0, 1, or 2) of stacked pairs in the best allowed order."""
        pos = [c.get_pos() for c in self.cubes]
        best = np.zeros(self.n_envs)
        for base, mid, top in self.stack_orders:
            pairs = self._pair_on(pos[mid], pos[base]).astype(
                np.float64
            ) + self._pair_on(pos[top], pos[mid])
            best = np.maximum(best, pairs)
        return best

    def _raw_success(self) -> np.ndarray:
        """Instantaneous released-tower condition; success needs it for
        ``success_hold_ticks`` consecutive control steps."""
        pos = [c.get_pos() for c in self.cubes]
        tower = np.zeros(self.n_envs, dtype=bool)
        for base, mid, top in self.stack_orders:
            on_table = (
                np.abs(pos[base][:, 2] - (self.arena.top_z + self.cubes[base].top_offset))
                < self.stack_z_tol
            )
            tower |= (
                on_table
                & self._pair_on(pos[mid], pos[base])
                & self._pair_on(pos[top], pos[mid])
            )
        settled = np.all(
            [np.linalg.norm(c.get_vel(), axis=-1) < self.success_max_speed
             for c in self.cubes],
            axis=0,
        )
        released = ~np.any([self._robot_contact(c) for c in self.cubes], axis=0)
        return tower & settled & released

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

    def _datagen_object_poses(self) -> dict[str, np.ndarray]:
        return {c.name: pose_mats(c.get_pos(), c.get_quat()) for c in self.cubes}

    def _datagen_term_signals(self) -> dict[str, np.ndarray]:
        # place signals follow the first allowed order (the canonical one for
        # StackRGY); the base cube never gets a place signal
        signals = {f"grasp_{c.name}": self._grasped(c) for c in self.cubes}
        base, mid, top = self.stack_orders[0]
        for upper, lower in ((mid, base), (top, mid)):
            c, b = self.cubes[upper], self.cubes[lower]
            signals[f"place_{c.name}"] = self._pair_on(
                c.get_pos(), b.get_pos()
            ) & ~self._robot_contact(c)
        return signals


class StackRGY(Stack):
    """Stack with a fixed order: red on the table, green on red, yellow on top."""

    stack_orders = ((0, 1, 2),)
