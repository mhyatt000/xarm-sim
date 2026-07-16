"""Evaluate a trained TD-MPC2 checkpoint on the suite Lift task (batched envs).

Loads a checkpoint saved by ``scripts/tdmpc_train.py`` (best.pt/latest.pt), runs
seeded eval rounds (``n_envs`` episodes per round) through the same wrapped suite env
stack, and reports success rate + per-episode stats; optionally saves videos (env 0)
of every Nth round.

    uv run python scripts/tdmpc_eval.py --checkpoint outputs/tdmpc/v5/best.pt --episodes 32
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
    checkpoint: Path = PROJECT_ROOT / "outputs" / "tdmpc" / "v5" / "best.pt"
    episodes: int = 32            # rounded up to a multiple of n_envs
    seed: int = 51_000            # round r uses seed + r (same convention as training eval)
    video_every: int = 2          # save an mp4 (env 0) of every Nth round (0 = off)
    out: Path | None = None       # default: <checkpoint dir>/eval
    mpc: bool = True              # plan with MPPI (false = raw policy prior)
    compile: bool = True
    # env (mirrors tdmpc_train.Config)
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    sim_hz: int = 120
    control_freq: float = 15.0
    max_steps: int = 150
    max_delta_rad: float = 0.10
    binary_gripper: bool = True
    cube_vel_obs: bool = True         # v6/v7 checkpoints: pass --no-cube-vel-obs (32-d obs)
    noslip_iterations: int = 10


def main(cfg: Config) -> None:
    # reuse the training script's env/config builders (same dir -> plain import works)
    from tdmpc_train import Config as TrainConfig, build_env, make_agent, make_tdmpc_cfg

    tcfg = TrainConfig(n_envs=cfg.n_envs, backend=cfg.backend, sim_hz=cfg.sim_hz,
                       control_freq=cfg.control_freq, max_steps=cfg.max_steps,
                       max_delta_rad=cfg.max_delta_rad, binary_gripper=cfg.binary_gripper,
                       cube_vel_obs=cfg.cube_vel_obs,
                       noslip_iterations=cfg.noslip_iterations,
                       mpc=cfg.mpc, compile=cfg.compile)
    env = build_env(tcfg)
    env.autoreset = False
    out = cfg.out or cfg.checkpoint.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tdcfg = make_tdmpc_cfg(tcfg, env.obs_dim, env.action_dim, out)

    agent = make_agent(tdcfg, cfg.n_envs)
    agent.load(str(cfg.checkpoint))
    print(f"[eval] loaded {cfg.checkpoint}")

    B = cfg.n_envs
    rounds = max(1, -(-cfg.episodes // B))
    results = []
    for rd in range(rounds):
        record = cfg.video_every > 0 and rd % cfg.video_every == 0
        obs = env.reset(seed=cfg.seed + rd)
        agent.reset_envs(np.arange(B))
        frames = []
        live = np.ones(B, dtype=bool)
        ep_len = np.zeros(B, dtype=int)
        while live.any():
            torch.compiler.cudagraph_mark_step_begin()
            actions = agent.act(obs, eval_mode=True)
            obs, reward, done, info = env.step(actions)
            ep_len[live] += 1
            if record and live[0]:
                frames.append(env.render_views()["low"])
            for i in np.flatnonzero(live & done):
                res = dict(round=rd, env=int(i), seed=cfg.seed + rd,
                           success=bool(info["success"][i]), steps=int(ep_len[i]),
                           max_rise=float(info["max_rise"][i]))
                results.append(res)
                print(f"[r{rd} e{i:02d}] success={res['success']} steps={res['steps']} "
                      f"max_rise={res['max_rise']*100:.1f}cm")
            live &= ~done
        if frames:
            import imageio.v3 as iio

            iio.imwrite(out / f"round{rd:02d}.mp4", np.stack(frames),
                        fps=round(cfg.control_freq))

    rate = float(np.mean([r["success"] for r in results]))
    print(f"\n[result] success {sum(r['success'] for r in results)}/{len(results)} ({rate:.0%})")
    with (out / "results.json").open("w") as f:
        json.dump({"checkpoint": str(cfg.checkpoint), "success_rate": rate,
                   "episodes": results}, f, indent=2)
    print(f"[result] wrote {out / 'results.json'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
