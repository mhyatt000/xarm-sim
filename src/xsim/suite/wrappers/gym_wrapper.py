"""Flatten dict observations into one vector (robosuite's GymWrapper)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymWrapper(gym.Wrapper):
    """Concatenates the env's dict obs into a single float32 vector.

    Key order is ``sorted(obs_keys)``, fixed at construction, so the layout is
    independent of dict insertion order. ``obs_keys=None`` uses every key in the
    env's observation space.
    """

    def __init__(self, env: gym.Env, obs_keys: list[str] | None = None):
        super().__init__(env)
        self.obs_keys = sorted(
            obs_keys if obs_keys is not None else env.observation_space.spaces
        )
        dim = sum(
            int(np.prod(env.observation_space[k].shape)) for k in self.obs_keys
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(dim,), dtype=np.float32
        )

    def _flatten(self, obs: dict) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self.obs_keys]
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, terminated, truncated, info
