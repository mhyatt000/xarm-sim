"""Flatten dict observations into one vector (robosuite's GymWrapper)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class GymWrapper(gym.Wrapper):
    """Concatenates the env's dict obs into a single (n_envs, D) float32 array.

    Key order is ``sorted(obs_keys)``, fixed at construction, so the layout is
    independent of dict insertion order. ``obs_keys=None`` uses every key in the
    env's observation space. Partial resets thread through gym.Wrapper.reset via
    ``options={"envs_idx": ...}``.
    """

    def __init__(self, env: gym.Env, obs_keys: list[str] | None = None):
        super().__init__(env)
        self.obs_keys = sorted(
            obs_keys if obs_keys is not None else env.observation_space.spaces
        )
        self._n_envs = env.unwrapped.n_envs
        # batched Dict space -> per-key per-env dims (leading axis is n_envs)
        dim = sum(
            int(np.prod(env.observation_space[k].shape[1:])) for k in self.obs_keys
        )
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(dim,), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self._n_envs
        )

    def _flatten(self, obs: dict) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs[k], dtype=np.float32).reshape(self._n_envs, -1)
                for k in self.obs_keys
            ],
            axis=-1,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, terminated, truncated, info
