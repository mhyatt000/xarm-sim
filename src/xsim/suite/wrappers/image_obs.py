"""Batched camera frames as observations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ImageObsWrapper(gym.Wrapper):
    """Adds ``obs["rgb"]``: (n_envs, V, 3, H, W) uint8 — one frame per
    instantiated camera per control step, views ordered by sorted camera name
    (``self.views``). Requires a batched render backend (nyx or madrona batch
    renderering; raster cameras are built on env 0 only).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = env.unwrapped
        self.views = sorted(base.cams)
        if not self.views:
            raise ValueError("ImageObsWrapper needs at least one instantiated camera")
        w, h = base.camera_res
        v = len(self.views)
        rgb_space = gym.spaces.Box(0, 255, shape=(v, 3, h, w), dtype=np.uint8)
        self.single_observation_space = gym.spaces.Dict(
            dict(base.single_observation_space.spaces, rgb=rgb_space)
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, base.n_envs
        )

    def _attach(self, obs: dict) -> dict:
        views = self.env.unwrapped.render_views(all_envs=True)
        rgb = np.stack([views[k] for k in self.views], axis=1)  # (B, V, H, W, 3)
        obs["rgb"] = np.ascontiguousarray(rgb.transpose(0, 1, 4, 2, 3))
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._attach(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._attach(obs), reward, terminated, truncated, info
