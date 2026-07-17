"""Env registry and the base Genesis episode loop."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import genesis as gs
import gymnasium as gym
import numpy as np

from xsim.suite.utils import ensure_genesis_init

REGISTERED_ENVS: dict[str, type] = {}
_UNREGISTERED_ENVS = {"GenesisEnv", "RobotEnv", "ManipulationEnv"}


def register_env(cls: type) -> type:
    """Register an environment class under its class name."""
    REGISTERED_ENVS[cls.__name__] = cls
    return cls


def make(env_name: str, *args, **kwargs):
    """Instantiate a registered environment by name."""
    if env_name not in REGISTERED_ENVS:
        raise ValueError(
            f"Unknown environment {env_name!r}. Registered environments: "
            f"{sorted(REGISTERED_ENVS)}"
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Registers every concrete env subclass by class name."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        if name not in _UNREGISTERED_ENVS:
            register_env(cls)
        return cls


class GenesisEnv(gym.Env, metaclass=EnvMeta):
    """Base episode loop over a Genesis scene.

    Robosuite-style template hooks (_load_model / _initialize_sim /
    _setup_references / _setup_observables / _reset_internal) on the gymnasium
    5-tuple API. Subclasses compose the world in _load_model (set self.model to
    a Task); this class owns the scene, the control decimation, and the
    step/reset flow. The scene is built once — reset() only restores state
    (Genesis scenes cannot be rebuilt).

    Always batched: obs values are (n_envs, ...), actions (n_envs, action_dim),
    and reward/terminated/truncated/info["success"] are (n_envs,) arrays — even
    at n_envs=1. Episodes end per env; there is no auto-reset. Callers reset the
    finished subset via reset(envs_idx=...) (or options={"envs_idx": ...} when
    going through gym wrappers, whose reset() only forwards seed/options).
    """

    def __init__(
        self,
        physics_dt: float = 1.0 / 120.0,
        control_freq: float = 30.0,
        horizon: int = 300,
        n_envs: int = 1,
        show_viewer: bool = False,
        noslip_iterations: int = 0,
    ):
        ensure_genesis_init()
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.physics_dt = physics_dt
        self.control_freq = control_freq
        self.horizon = horizon
        self.n_envs = n_envs
        self.show_viewer = show_viewer
        self.noslip_iterations = noslip_iterations
        self.control_every = max(1, round(1.0 / (control_freq * physics_dt)))
        self.control_dt = physics_dt * self.control_every
        self.timestep = np.zeros(n_envs, dtype=np.int64)
        self._load_model()
        self._initialize_sim()
        self._setup_references()
        self._observables = self._setup_observables()
        self.single_observation_space = self._infer_observation_space()
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.n_envs
        )

    def _load_model(self) -> None:
        raise NotImplementedError

    def _scene_renderer(self):
        """Scene-level renderer option (None = Genesis default rasterizer).
        Overridden by layers that select a camera backend (e.g. madrona batch)."""
        return None

    def _initialize_sim(self) -> None:
        renderer = self._scene_renderer()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt, substeps=4),
            rigid_options=gs.options.RigidOptions(
                dt=self.physics_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                noslip_iterations=self.noslip_iterations,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=self.show_viewer,
            **({"renderer": renderer} if renderer is not None else {}),
        )
        self.model.add_to(self.scene)
        self._setup_cameras()  # nyx sensors must exist before build
        self.scene.build(n_envs=self.n_envs)

    def _setup_cameras(self) -> None:
        """Add cameras/sensors to the scene, pre-build."""

    def _setup_references(self) -> None:
        pass

    def _setup_observables(self) -> OrderedDict[str, Callable[[], np.ndarray]]:
        return OrderedDict()

    def _reset_internal(self, envs_idx=None) -> None:
        pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        envs_idx=None,
    ):
        super().reset(seed=seed)
        if envs_idx is None and options is not None:
            envs_idx = options.get("envs_idx")
        if envs_idx is None:
            self.timestep[:] = 0
        else:
            self.timestep[np.asarray(envs_idx)] = 0
        self._reset_internal(envs_idx)
        return self._get_observations(), {}

    def step(self, action):
        self.timestep += 1
        self._pre_action(action)
        for _ in range(self.control_every):
            self.scene.step()
            self._post_sim_step()
        reward, terminated, truncated, info = self._post_action(action)
        return self._get_observations(), reward, terminated, truncated, info

    def _pre_action(self, action) -> None:
        raise NotImplementedError

    def _post_sim_step(self) -> None:
        """After each physics step inside the decimation loop (e.g. camera sync)."""

    def _post_action(self, action) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        return (
            np.asarray(self.reward(action), dtype=np.float32),
            np.asarray(self._check_terminated(), dtype=bool),
            self.timestep >= self.horizon,
            {"success": np.asarray(self._check_success(), dtype=bool)},
        )

    def reward(self, action=None) -> np.ndarray:
        """Per-env rewards, shape (n_envs,)."""
        raise NotImplementedError

    def _check_success(self) -> np.ndarray:
        return np.zeros(self.n_envs, dtype=bool)

    def _check_terminated(self) -> np.ndarray:
        return np.zeros(self.n_envs, dtype=bool)

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(fn(), dtype=np.float32)
            for name, fn in self._observables.items()
        }

    def _infer_observation_space(self) -> gym.spaces.Dict:
        """Per-env space; the batched space is batch_space(single, n_envs)."""
        sample = self._get_observations()
        return gym.spaces.Dict(
            {
                k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape[1:], dtype=np.float32)
                for k, v in sample.items()
            }
        )
