"""Place task: put an object inside the bin or on top of the plate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from xsim.suite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
    pose_mats,
)
from xsim.suite.models import BoxObject, MeshObject, TableArena, Task
from xsim.suite.utils import UniformRandomSampler

_ASSETS = Path(__file__).resolve().parents[5] / "assets"


def _rotate_inv(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` (n, 3) by the inverse of ``quat`` (n, 4) wxyz."""
    w, u = quat[:, :1], -quat[:, 1:]
    t = 2.0 * np.cross(u, vec)
    return vec + w * t + np.cross(u, t)


class PlaceObj(ManipulationEnv):
    """xArm7 place task: put the object inside the bin or on top of the plate.

    One fixture is in the scene, shared by every env (batched Genesis envs
    share one scene graph): ``self.target`` is the bin (Uline S-12415 hopper
    bin, assets/bin.stl, CoACD-decomposed so the interior is hollow) or the
    plate (330 x 8 mm disc, assets/plate.stl). ``target`` picks it — "bin",
    "plate", or "random", one unseeded draw at construction for all envs (use
    an explicit kind, or the PlaceObjBin/PlaceObjPlate variants, when the
    build must be reproducible). The fixture spawns free on the far side of
    the table each reset; the resolved kind is the constant ``target_is_bin``
    observable.

    Success requires, for ``success_hold_ticks`` CONSECUTIVE control steps:
    the object placed relative to the fixture (center inside the bin's
    interior box, or over the plate disc and above its midplane),
    object-fixture contact, object slower than ``success_max_speed``, and no
    object-robot contact — placement only counts once the hand has released
    it.

    The object is a red cube by default; ``objaverse=True`` swaps it for a
    mesh drawn from Objaverse by ``objaverse_seed`` (``pip install
    objaverse``; the same mesh is used across envs). Objaverse assets carry
    arbitrary artist units, so they are rescaled to ``obj_max_extent`` —
    pass ``obj_max_extent=None`` to keep the file's native size.
    """

    arena_class: type[TableArena] = TableArena

    # bin.stl interior in the bin frame (base at z=0): conservative inner
    # halfwidths, floor top, and rim height, measured from the mesh
    bin_inner_half: tuple[float, float] = (0.120, 0.056)
    bin_floor_z: float = 0.0064
    bin_rim_z: float = 0.127
    plate_radius: float = 0.165
    plate_half_height: float = 0.004

    def __init__(
        self,
        robots: str | list[str] = "XArm7",
        target: str = "random",
        cube_size: float = 0.03175,
        cube_color: tuple[float, float, float] = (0.48, 0.05, 0.04),
        objaverse: bool = False,
        objaverse_seed: int | None = None,
        obj_max_extent: float | None = 0.05,
        # object on the near half, fixture along the far edge; the fixture
        # zone keeps it on the table at any draw (bin yaw capped at +-10 deg)
        # and the object keeps xy_radius clearance from it via rejection
        placement_initializer: UniformRandomSampler | None = None,
        target_initializer: UniformRandomSampler | None = None,
        min_clearance: float = 0.01,
        success_hold_ticks: int = 1,
        success_max_speed: float = 0.10,
        reward_shaping: bool = False,
        randomize_cameras: bool = True,
        **kwargs,
    ):
        if target not in ("random", "bin", "plate"):
            raise ValueError(
                f"target must be 'random', 'bin', or 'plate', got {target!r}"
            )
        if target == "random":
            target = "bin" if np.random.default_rng().random() < 0.5 else "plate"
        self.target_kind = target
        self.target_is_bin = target == "bin"
        self._placed = self._inside_bin if self.target_is_bin else self._on_plate
        self.randomize_cameras = randomize_cameras
        self.cube_size = cube_size
        self.cube_color = cube_color
        self.objaverse = objaverse
        self.objaverse_seed = objaverse_seed
        self.obj_max_extent = obj_max_extent
        self.min_clearance = min_clearance
        self.success_hold_ticks = success_hold_ticks
        self.success_max_speed = success_max_speed
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer or UniformRandomSampler(
            (0.20, 0.34), (-0.20, 0.20)
        )
        self.target_initializer = target_initializer or (
            UniformRandomSampler(
                (0.40, 0.46), (-0.10, 0.10), yaw_range=(-math.pi / 18, math.pi / 18)
            )
            if self.target_is_bin
            else UniformRandomSampler((0.40, 0.46), (-0.12, 0.12))
        )
        super().__init__(robots=robots, **kwargs)
        self._success_hold = np.zeros(self.n_envs, dtype=np.int64)

    def _load_model(self) -> None:
        self.arena = self.arena_class(randomize_cameras=self.randomize_cameras)
        if self.objaverse:
            self.obj = MeshObject(
                "obj",
                file=self._objaverse_file(),
                max_extent=self.obj_max_extent,
                friction=2.0,
                decompose=True,
            )
        else:
            s = self.cube_size
            self.obj = BoxObject(
                "obj", size=(s, s, s), color=self.cube_color, friction=2.0
            )
        if self.target_is_bin:
            self.target = MeshObject(
                "bin",
                file=str(_ASSETS / "bin.stl"),
                color=(0.12, 0.29, 0.65),
                friction=1.0,
                decompose=True,
            )
        else:
            self.target = MeshObject(
                "plate",
                file=str(_ASSETS / "plate.stl"),
                color=(0.04, 0.04, 0.04),
                friction=1.0,
            )
        self.model = Task(
            self.arena,
            [robot.model for robot in self.robots],
            [self.obj, self.target],
        )

    def _objaverse_file(self) -> str:
        try:
            import objaverse
        except ImportError as e:
            raise ImportError(
                "objaverse=True needs the objaverse package (pip install objaverse)"
            ) from e
        uids = sorted(objaverse.load_uids())
        rng = np.random.default_rng(self.objaverse_seed)
        uid = uids[int(rng.integers(len(uids)))]
        return objaverse.load_objects([uid])[uid]

    def _setup_observables(self):
        observables = super()._setup_observables()
        observables["obj_pos"] = self.obj.get_pos
        observables["obj_quat"] = self.obj.get_quat
        observables["target_pos"] = self.target.get_pos
        observables["target_quat"] = self.target.get_quat
        observables["target_is_bin"] = lambda: np.full(
            (self.n_envs, 1), float(self.target_is_bin), dtype=np.float32
        )
        observables["obj_to_target_pos"] = (
            lambda: self.target.get_pos() - self.obj.get_pos()
        )
        observables["robot0_gripper_to_obj_pos"] = (
            lambda: self.obj.get_pos() - self.robots[0].ee_pos
        )
        return observables

    def _sample_obj_placement(self, n: int, tx: np.ndarray, ty: np.ndarray):
        """Object (x, y, yaw) draws keeping xy_radius + min_clearance away
        from the fixture center at ``(tx, ty)`` in every env."""
        keep_out = self.target.xy_radius + self.obj.xy_radius + self.min_clearance
        ox, oy, oyaw = (np.empty(n) for _ in range(3))
        todo = np.arange(n)
        for _ in range(100):
            x, y, yaw = self.placement_initializer.sample(self.np_random, len(todo))
            ox[todo], oy[todo], oyaw[todo] = x, y, yaw
            close = np.hypot(ox[todo] - tx[todo], oy[todo] - ty[todo]) < keep_out
            todo = todo[close]
            if todo.size == 0:
                return ox, oy, oyaw
        raise RuntimeError(
            f"could not place the object clear of the {self.target_kind} inside "
            f"x_range={self.placement_initializer.x_range} "
            f"y_range={self.placement_initializer.y_range}"
        )

    def _reset_internal(self, envs_idx=None) -> None:
        super()._reset_internal(envs_idx)
        idx = (
            np.arange(self.n_envs)
            if envs_idx is None
            else np.atleast_1d(np.asarray(envs_idx))
        )
        n = len(idx)
        tx, ty, tyaw = self.target_initializer.sample(self.np_random, n)
        self.target.set_pose(
            tx, ty, self.arena.top_z + self.target.bottom_offset, tyaw,
            envs_idx=envs_idx,
        )
        ox, oy, oyaw = self._sample_obj_placement(n, tx, ty)
        self.obj.set_pose(
            ox, oy, self.arena.top_z + self.obj.bottom_offset, oyaw, envs_idx=envs_idx
        )
        self._success_hold[idx] = 0

    def reward(self, action=None) -> np.ndarray:
        success = self._check_success()
        if self.reward_shaping:
            # up to 0.25 for reaching the object + up to 0.5 for moving it
            # toward the fixture; capped at 1.0 by the success branch
            reach = 0.25 * (
                1.0 - np.tanh(10.0 * self._gripper_to_target_dist(self.obj.get_pos()))
            )
            carry_dist = np.linalg.norm(
                self.obj.get_pos()[:, :2] - self.target.get_pos()[:, :2], axis=-1
            )
            shaped = reach + 0.5 * (1.0 - np.tanh(3.0 * carry_dist))
            return np.where(success, 1.0, shaped).astype(np.float32)
        return success.astype(np.float32)

    def _inside_bin(self, local: np.ndarray) -> np.ndarray:
        """Per-env: object center within the bin's interior box (bin frame)."""
        hx, hy = self.bin_inner_half
        return (
            (np.abs(local[:, 0]) < hx)
            & (np.abs(local[:, 1]) < hy)
            & (local[:, 2] > self.bin_floor_z)
            & (local[:, 2] < self.bin_rim_z)
        )

    def _on_plate(self, local: np.ndarray) -> np.ndarray:
        """Per-env: object center over the plate disc and above its midplane
        (plate frame); resting is enforced by the contact requirement."""
        return (np.linalg.norm(local[:, :2], axis=-1) < self.plate_radius) & (
            local[:, 2] > self.plate_half_height
        )

    def _raw_success(self) -> np.ndarray:
        """Instantaneous released-placement condition; success needs it for
        ``success_hold_ticks`` consecutive control steps."""
        local = _rotate_inv(
            self.target.get_quat(), self.obj.get_pos() - self.target.get_pos()
        )
        placed = self._placed(local)
        touching = self._contact_mask(self.obj, self.target.entity)
        settled = np.linalg.norm(self.obj.get_vel(), axis=-1) < self.success_max_speed
        released = ~self._robot_contact(self.obj)
        return placed & touching & settled & released

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
        return {
            name: pose_mats(o.get_pos(), o.get_quat())
            for name, o in (("obj", self.obj), (self.target.name, self.target))
        }

    def _datagen_term_signals(self) -> dict[str, np.ndarray]:
        return {"grasp": self._grasped(self.obj)}


class PlaceObjBin(PlaceObj):
    """PlaceObj with the bin as the fixture."""

    def __init__(self, target: str = "bin", **kwargs):
        super().__init__(target=target, **kwargs)


class PlaceObjPlate(PlaceObj):
    """PlaceObj with the plate as the fixture."""

    def __init__(self, target: str = "plate", **kwargs):
        super().__init__(target=target, **kwargs)


class PlaceObjaverse(PlaceObj):
    """PlaceObj with an Objaverse object instead of the cube (draw controlled
    by ``objaverse_seed``)."""

    def __init__(self, objaverse: bool = True, **kwargs):
        super().__init__(objaverse=objaverse, **kwargs)
