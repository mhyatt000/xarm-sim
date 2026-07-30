"""Debug runner for the xsim.suite layered envs.

Builds a registered env by name and steps random actions, the scripted
waypoint lift policy, or the reactive FSM lift expert — a quick check that
model composition, controllers, policies, and the episode loop hold together.

    uv run python scripts/suite.py [--env Lift] [--steps 5] [--seed 0] [--n-envs 16]
    uv run python scripts/suite.py --policy waypoint --steps 200 --seed 0
    uv run python scripts/suite.py --policy expert --steps 200 --n-envs 16 --video expert.mp4

Source-demo recording for the MimicGen-style generator (xsim.datagen): keep
running expert episodes until N successes are banked to hdf5:

    uv run python scripts/suite.py --env Lift --policy expert --n-envs 16 \\
        --record-datagen demos/lift.h5 --record-n 10
    uv run python scripts/suite.py --env StackRGY --policy expert \\
        --n-envs 16 --horizon 500 --record-datagen demos/stack.h5

--policy expert picks the env's FSM expert, and on a multi-robot rig (e.g.
--robots DXArm7L DXArm7R) drives the nearest arm per env while the other parks:

    uv run python scripts/suite.py --env LiftRelease --robots DXArm7L DXArm7R \\
        --policy expert --steps 250 --n-envs 16
"""

from __future__ import annotations
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

import xsim.suite as suite
from xsim.suite.policies import LiftPolicy, expert_for
from xsim.suite.renderers import BatchConfig, NyxConfig
from xsim.utils.video import tile_grid


@dataclass
class Config:
    env: str = "Lift"
    robots: list[str] | None = None  # override the env's default robot set
    steps: int = 5
    seed: int = 0
    horizon: int = 300
    n_envs: int = 1
    show_viewer: bool = False
    policy: Literal["random", "waypoint", "expert"] = "random"
    steps_per_segment: int = 20
    # record successful policy episodes (datagen_info + actions + obs) to this
    # hdf5 until --record-n demos are banked (source demos for scripts/datagen.py)
    record_datagen: Path | None = None
    record_n: int = 10
    # keep recording this many ticks past an env's success (or until the policy
    # opens the gripper): envs terminate AT success, which would otherwise cut
    # the final subtask segment to a couple of frames (e.g. Lift ends ~2 ticks
    # after the cube leaves the table)
    record_tail: int = 6
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


def record_source_demos(cfg: Config, env, policy) -> None:
    """Bank ``cfg.record_n`` successful episodes (datagen_info + actions + obs)
    into ``cfg.record_datagen``, re-running batched episodes as needed."""
    from xsim.datagen import DemoRecorder

    rec = DemoRecorder(env, cfg.record_datagen)
    ep = 0
    while rec.n_demos < cfg.record_n:
        obs, _ = env.reset(seed=cfg.seed + 1000 * ep)
        policy.reset(obs)
        rec.begin_episode()
        n = env.n_envs
        live = np.ones(n, dtype=bool)
        tailing = np.zeros(n, dtype=bool)
        success = np.zeros(n, dtype=bool)
        rec_len = np.zeros(n, dtype=np.int64)
        tail = np.zeros(n, dtype=np.int64)
        was_closed = np.zeros(n, dtype=bool)
        for _ in range(cfg.horizon + cfg.record_tail):
            action = np.asarray(policy.act(obs))
            grip_open = action[:, -1] > 0.5
            tailing &= ~(was_closed & grip_open)  # policy released: stop the tail
            was_closed = ~grip_open
            rec.record_step(action, obs)
            rec_len += live | tailing
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            tailing |= live & done & info["success"]
            success |= live & info["success"]
            live &= ~done
            tail += tailing
            tailing &= tail <= cfg.record_tail
            if not (live | tailing).any():
                break
        keep = np.zeros_like(success)
        keep[np.flatnonzero(success)[: cfg.record_n - rec.n_demos]] = True
        written = rec.end_episode(keep, rec_len)
        ep += 1
        print(
            f"episode batch {ep}: success {int(success.sum())}/{env.n_envs}, "
            f"banked {written} (total {rec.n_demos}/{cfg.record_n})"
        )
    rec.close()
    print(f"source demos -> {cfg.record_datagen}")


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
    # per-object position observables (env-agnostic: Lift has cube_pos,
    # Stack has cube_{red,green,yellow}_pos, ...)
    pos_keys = sorted(
        k for k in obs if k.endswith("_pos") and not k.startswith("robot")
    )
    policy = None
    if cfg.policy == "waypoint":
        policy = LiftPolicy(env, steps_per_segment=cfg.steps_per_segment)
        policy.reset(obs)
    elif cfg.policy == "expert":
        policy = expert_for(env)
        policy.reset(obs)
    if cfg.record_datagen is not None:
        if policy is None:
            raise ValueError("--record-datagen needs a scripted policy, not random")
        if len(env.robots) > 1:
            raise ValueError("--record-datagen is single-arm only (datagen "
                             "reads robots[0] and the last action channel)")
        record_source_demos(cfg, env, policy)
        return
    record()
    for i in tqdm(range(cfg.steps)):
        action = policy.act(obs) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        record()
        done = terminated | truncated
        objs = " ".join(f"{k[:-4]}={obs[k][0].round(3)}" for k in pos_keys)
        print(
            f"step {i}: reward={np.round(reward, 3)} terminated={terminated.astype(int)} "
            f"truncated={truncated.astype(int)} success={info['success'].astype(int)} "
            f"{objs}"
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
