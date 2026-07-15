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
        self.physics_dt = physics_dt
        self.control_freq = control_freq
        self.horizon = horizon
        self.n_envs = n_envs
        self.show_viewer = show_viewer
        self.noslip_iterations = noslip_iterations
        self.control_every = max(1, round(1.0 / (control_freq * physics_dt)))
        self.control_dt = physics_dt * self.control_every
        self.timestep = 0
        self._load_model()
        self._initialize_sim()
        self._setup_references()
        self._observables = self._setup_observables()
        self.observation_space = self._infer_observation_space()

    def _load_model(self) -> None:
        raise NotImplementedError

    def _initialize_sim(self) -> None:
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

    def _reset_internal(self) -> None:
        pass

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.timestep = 0
        self._reset_internal()
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

    def _post_action(self, action) -> tuple[float, bool, bool, dict]:
        reward = self.reward(action)
        success = self._check_success()
        return (
            reward,
            self._check_terminated(),
            self.timestep >= self.horizon,
            {"success": success},
        )

    def reward(self, action=None) -> float:
        raise NotImplementedError

    def _check_success(self) -> bool:
        return False

    def _check_terminated(self) -> bool:
        return False

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(fn(), dtype=np.float32)
            for name, fn in self._observables.items()
        }

    def _infer_observation_space(self) -> gym.spaces.Dict:
        sample = self._get_observations()
        return gym.spaces.Dict(
            {
                k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32)
                for k, v in sample.items()
            }
        )
