"""Append object velocities to the dict observation."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ObjectVelocityWrapper(gym.Wrapper):
    """Adds ``<name>_vel`` / ``<name>_ang`` (linear/angular velocity, (n_envs, 3))
    observables for named task objects.

    Restores Markovness for dynamic phases (a single frame cannot distinguish a
    rising cube from a falling one). ``objects`` are attribute names on the base
    env (e.g. ``"cube"`` on Lift).
    """

    def __init__(self, env: gym.Env, objects: tuple[str, ...] = ("cube",)):
        super().__init__(env)
        self._objects = [(name, getattr(env.unwrapped, name)) for name in objects]
        single = dict(env.unwrapped.single_observation_space.spaces)
        for name, _ in self._objects:
            for suffix in ("_vel", "_ang"):
                single[name + suffix] = gym.spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                )
        self.single_observation_space = gym.spaces.Dict(single)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, env.unwrapped.n_envs
        )

    def _augment(self, obs: dict) -> dict:
        for name, obj in self._objects:
            obs[name + "_vel"] = obj.get_vel().astype(np.float32)
            obs[name + "_ang"] = obj.get_ang().astype(np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info
