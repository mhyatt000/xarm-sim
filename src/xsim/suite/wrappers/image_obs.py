"""Batched camera frames as observations."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np


class ImageObsWrapper(gym.Wrapper):
    """Adds ``obs["rgb"]``: (n_envs, V, 3, H, W) uint8 — one frame per
    instantiated camera per control step, views ordered by sorted camera name
    (``self.views``). Requires a batched render backend (nyx or the madrona
    batch renderer; raster cameras are built on env 0 only).

    ``plates`` composites a static background image behind each named view
    wherever the segmentation mask reports no geometry — the splat backdrop at
    batch throughput (see scripts/make_plates.py). Plates only make sense for
    static cameras and require the batch backend (segmentation per env).
    """

    def __init__(self, env: gym.Env, plates: dict[str, np.ndarray | Path] | None = None,
                 seg_background: int = 0):
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
        self._seg_background = seg_background
        self._plates: dict[str, np.ndarray] | None = None
        if plates:
            import cv2

            self._plates = {}
            for name, plate in plates.items():
                if name not in self.views:
                    continue
                if isinstance(plate, (str, Path)):
                    plate = cv2.imread(str(plate))[:, :, ::-1]  # BGR -> RGB
                self._plates[name] = np.ascontiguousarray(
                    cv2.resize(np.asarray(plate), (w, h), interpolation=cv2.INTER_AREA)
                ).astype(np.uint8)

    @staticmethod
    def _np(x) -> np.ndarray:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def _frames(self) -> dict[str, np.ndarray]:
        base = self.env.unwrapped
        if not self._plates:
            return base.render_views(all_envs=True)
        out = {}
        for name in self.views:
            cam = base.cams[name]
            plate = self._plates.get(name)
            if plate is None:
                rgb = self._np(cam.render(rgb=True)[0])[..., :3]
            else:
                rgb_t, _, seg_t, _ = cam.render(rgb=True, segmentation=True)
                rgb = self._np(rgb_t)[..., :3]
                bg = self._np(seg_t) == self._seg_background  # (B, H, W)
                rgb = np.where(bg[..., None], plate[None], rgb)
            out[name] = rgb.astype(np.uint8)
        return out

    def _attach(self, obs: dict) -> dict:
        views = self._frames()
        rgb = np.stack([views[k] for k in self.views], axis=1)  # (B, V, H, W, 3)
        obs["rgb"] = np.ascontiguousarray(rgb.transpose(0, 1, 4, 2, 3))
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._attach(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._attach(obs), reward, terminated, truncated, info
