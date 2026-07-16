"""Evaluate a trained TD-MPC2 checkpoint on the suite Lift task.

Loads a checkpoint saved by ``scripts/tdmpc_train.py`` (best.pt/latest.pt), runs seeded
episodes through the same wrapped suite env stack, and reports success rate +
per-episode stats; optionally saves videos every Nth episode.

    uv run python scripts/tdmpc_eval.py --checkpoint outputs/tdmpc/v4/best.pt --episodes 20
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from rich import print

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    checkpoint: Path = PROJECT_ROOT / "outputs" / "tdmpc" / "v4" / "best.pt"
    episodes: int = 20
    seed: int = 51_000            # eval seeds = seed + i (same convention as training eval)
    video_every: int = 5          # save an mp4 of every Nth episode (0 = off)
    out: Path | None = None       # default: <checkpoint dir>/eval
    mpc: bool = True              # plan with MPPI (false = raw policy prior)
    compile: bool = False
    # env (mirrors tdmpc_train.Config)
    backend: Literal["gpu", "cpu"] = "gpu"
    sim_hz: int = 120
    control_freq: float = 15.0
    max_steps: int = 150
    max_delta_rad: float = 0.10
    noslip_iterations: int = 10


def main(cfg: Config) -> None:
    # reuse the training script's env/config builders (same dir -> plain import works)
    from tdmpc_train import Config as TrainConfig, build_env, make_agent, make_tdmpc_cfg

    tcfg = TrainConfig(backend=cfg.backend, sim_hz=cfg.sim_hz, control_freq=cfg.control_freq,
                       max_steps=cfg.max_steps, max_delta_rad=cfg.max_delta_rad,
                       noslip_iterations=cfg.noslip_iterations,
                       mpc=cfg.mpc, compile=cfg.compile)
    env = build_env(tcfg)
    out = cfg.out or cfg.checkpoint.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tdcfg = make_tdmpc_cfg(tcfg, env.observation_space.shape[0], env.action_space.shape[0], out)

    agent = make_agent(tdcfg)
    agent.load(str(cfg.checkpoint))
    print(f"[eval] loaded {cfg.checkpoint}")

    results = []
    for i in range(cfg.episodes):
        record = cfg.video_every > 0 and i % cfg.video_every == 0
        obs = env.reset(seed=cfg.seed + i)
        frames, done, t, info = [], False, 0, {}
        while not done:
            torch.compiler.cudagraph_mark_step_begin()
            action = agent.act(obs, t0=t == 0, eval_mode=True)
            obs, reward, done, info = env.step(action)
            t += 1
            if record:
                frames.append(env.render_views()["low"])
        res = dict(episode=i, seed=cfg.seed + i, success=bool(info["success"]),
                   steps=t, max_rise=float(info.get("max_rise", 0.0)))
        results.append(res)
        print(f"[ep{i:03d}] success={res['success']} steps={t} "
              f"max_rise={res['max_rise']*100:.1f}cm")
        if frames:
            import imageio.v3 as iio

            iio.imwrite(out / f"ep{i:03d}.mp4", np.stack(frames), fps=round(cfg.control_freq))

    rate = float(np.mean([r["success"] for r in results]))
    print(f"\n[result] success {sum(r['success'] for r in results)}/{len(results)} ({rate:.0%})")
    with (out / "results.json").open("w") as f:
        json.dump({"checkpoint": str(cfg.checkpoint), "success_rate": rate,
                   "episodes": results}, f, indent=2)
    print(f"[result] wrote {out / 'results.json'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
