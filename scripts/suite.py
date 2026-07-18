"""Debug runner for the xsim.suite layered envs.

Builds a registered env by name and steps either random actions or the
scripted waypoint lift policy — a quick check that model composition,
controllers, policies, and the episode loop hold together.

    uv run python scripts/suite.py [--env Lift] [--steps 5] [--seed 0] [--n-envs 16]
    uv run python scripts/suite.py --policy waypoint --steps 200 --seed 0
"""

from __future__ import annotations
from tqdm import tqdm

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

import xsim.suite as suite
from xsim.suite.policies import LiftPolicy
from xsim.suite.renderers import BatchConfig, NyxConfig


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
    render_backend: Literal["raster", "nyx", "batch"] = "batch"
    spp: int = 8                    # nyx samples per pixel
    batch_rasterizer: bool = False  # batch backend: rasterizer instead of the raytracer
    # composite splat background plates behind static cams (batch backend only;
    # generate with scripts/make_plates.py)
    plates_dir: Path | None = None
    # drop splat gaussians below this opacity (speed; <=0.15 looks intact)
    prune_opacity: float = 0.15
    camera_res: tuple[int, int] = (640, 480)  # batch: keep VRAM in mind at high n_envs
    video: Path | None = None       # write render() frames to an mp4 (cv2, no GUI)
    # tile every env into a per-camera grid (nyx/batch — raster cams are single-env);
    # otherwise the video shows env 0
    video_all_envs: bool = True
    video_max_width: int = 2048     # per-camera grid width cap, px


def tile_grid(frames: np.ndarray, max_width: int) -> np.ndarray:
    """(B, H, W, 3) -> near-square grid canvas, tiles resized to fit max_width."""
    import cv2

    b, h, w, _ = frames.shape
    cols = math.ceil(math.sqrt(b))
    rows = math.ceil(b / cols)
    tw = max(2, min(w, max_width // cols)) // 2 * 2
    th = max(2, round(h * tw / w)) // 2 * 2
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i in range(b):
        r, c = divmod(i, cols)
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = cv2.resize(
            frames[i], (tw, th), interpolation=cv2.INTER_AREA
        )
    return canvas


def main(cfg: Config) -> None:
    env = suite.make(
        cfg.env,
        **({"robots": cfg.robots} if cfg.robots is not None else {}),
        horizon=cfg.horizon,
        n_envs=cfg.n_envs,
        show_viewer=cfg.show_viewer,
        noslip_iterations=cfg.noslip_iterations,
        render_backend=cfg.render_backend,
        camera_res=cfg.camera_res,
        renderer_config=(
            NyxConfig(spp=cfg.spp) if cfg.render_backend == "nyx"
            else BatchConfig(
                use_rasterizer=cfg.batch_rasterizer,
                splat_prune_opacity=cfg.prune_opacity,
            )
            if cfg.render_backend == "batch" else None
        ),
    )
    writer = None
    grid = (cfg.video_all_envs and cfg.n_envs > 1
            and cfg.render_backend in ("nyx", "batch"))
    plates = None
    if cfg.plates_dir is not None and cfg.render_backend == "batch":
        from xsim.suite.wrappers.image_obs import load_plates

        plates = load_plates(
            {p.stem: p for p in sorted(cfg.plates_dir.glob("*.png"))}, cfg.camera_res)

    def record() -> None:
        nonlocal writer
        if cfg.video is None:
            return
        import cv2

        if grid:
            if plates is not None:
                from xsim.suite.wrappers.image_obs import render_plated_views

                views = render_plated_views(env, plates)
            else:
                views = env.render_views(all_envs=True)
            frame = np.concatenate(
                [tile_grid(views[k], cfg.video_max_width) for k in sorted(views)],
                axis=1,
            )
        else:
            frame = env.render()
        if writer is None:
            cfg.video.parent.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(cfg.video), fps=1.0 / env.control_dt,
                codec="libx264",  # avc1/H.264 — browser-playable; mp4v/mpeg4 isn't
                pixelformat="yuv420p", macro_block_size=1,
            )
        writer.append_data(frame)  # RGB, no BGR swap
    print("action_space:", env.action_space)
    obs, info = env.reset(seed=cfg.seed)
    for name in sorted(obs):
        print(f"  obs[{name}]: shape={obs[name].shape}")
    policy = None
    if cfg.policy == "waypoint":
        policy = LiftPolicy(env, steps_per_segment=cfg.steps_per_segment)
        policy.reset(obs)
    record()
    for i in tqdm(range(cfg.steps)):
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
        writer.close()
        print(f"video -> {cfg.video}")


if __name__ == "__main__":
    main(tyro.cli(Config))
