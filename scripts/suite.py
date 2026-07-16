"""Debug runner for the xsim.suite layered envs.

Builds a registered env by name and steps either random actions or the
scripted waypoint lift policy — a quick check that model composition,
controllers, policies, and the episode loop hold together.

    uv run python scripts/suite.py [--env Lift] [--steps 5] [--seed 0] [--n-envs 16]
    uv run python scripts/suite.py --policy waypoint --steps 200 --seed 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

import xsim.suite as suite
from xsim.suite.policies import LiftPolicy
from xsim.suite.renderers import NyxConfig


@dataclass
class Config:
    env: str = "Lift"
    robots: list[str] | None = None  # override the env's default robot set
    steps: int = 5
    seed: int = 0
    horizon: int = 300
    n_envs: int = 1
    show_viewer: bool = False
    policy: Literal["random", "waypoint"] = "random"
    steps_per_segment: int = 20
    noslip_iterations: int = 10
    render_backend: Literal["raster", "nyx"] = "raster"
    spp: int = 8                    # nyx samples per pixel
    video: Path | None = None       # write render() frames to an mp4 (cv2, no GUI)


def main(cfg: Config) -> None:
    env = suite.make(
        cfg.env,
        **({"robots": cfg.robots} if cfg.robots is not None else {}),
        horizon=cfg.horizon,
        n_envs=cfg.n_envs,
        show_viewer=cfg.show_viewer,
        noslip_iterations=cfg.noslip_iterations,
        render_backend=cfg.render_backend,
        renderer_config=NyxConfig(spp=cfg.spp) if cfg.render_backend == "nyx" else None,
    )
    writer = None

    def record() -> None:
        nonlocal writer
        if cfg.video is None:
            return
        import cv2

        frame = env.render()
        if writer is None:
            cfg.video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(cfg.video), cv2.VideoWriter_fourcc(*"mp4v"),
                1.0 / env.control_dt, (frame.shape[1], frame.shape[0]),
            )
        writer.write(frame[:, :, ::-1])  # RGB -> BGR
    print("action_space:", env.action_space)
    obs, info = env.reset(seed=cfg.seed)
    for name in sorted(obs):
        print(f"  obs[{name}]: shape={obs[name].shape}")
    policy = None
    if cfg.policy == "waypoint":
        policy = LiftPolicy(env, steps_per_segment=cfg.steps_per_segment)
        policy.reset(obs)
    record()
    for i in range(cfg.steps):
        action = policy.act(obs) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        record()
        done = terminated | truncated
        print(
            f"step {i}: reward={np.round(reward, 3)} terminated={terminated.astype(int)} "
            f"truncated={truncated.astype(int)} success={info['success'].astype(int)} "
            f"cube0={obs['cube_pos'][0].round(3)}"
        )
        if done.all():
            print(f"episode end at step {i}: success={info['success']}")
            if policy is not None:
                break
            obs, info = env.reset()
    if writer is not None:
        writer.release()
        print(f"video -> {cfg.video}")


if __name__ == "__main__":
    main(tyro.cli(Config))
