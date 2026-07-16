"""Collect scripted lift demos in the suite Lift action space for TD-MPC2 bootstrapping.

Runs the batched :class:`xsim.suite.policies.LiftPolicy` through the training wrapper
stack (``DeltaActionWrapper`` -> ``GymWrapper`` -> ``TensorShim``, built by
``tdmpc_train.build_env``), ``n_envs`` episodes per round, so the recorded transitions
obey the exact dynamics, observation layout, sparse reward, and termination the RL
agent will train on:

    waypoint commands (EE poses, gripper) --IK--> absolute joint targets (B, 8)
        --> actions = absolute_to_delta(targets)  (rowwise clip((t - qpos)/max_delta))
        --> env.step(actions)

Episodes are stored in tdmpc2's buffer layout (first frame has NaN action/reward,
matching ``OnlineTrainer.to_td``) and saved as a list of TensorDicts:

    uv run python scripts/tdmpc_demos.py --episodes 96 --out outputs/tdmpc/demos.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
import tyro
from rich import print

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    episodes: int = 96                # rounded up to a multiple of n_envs
    seed: int = 0                     # round r uses seed + r (placements drawn per env)
    steps_per_segment: int = 20       # scripted policy pacing (40 @ 30 Hz -> 20 @ 15 Hz)
    out: Path = PROJECT_ROOT / "outputs" / "tdmpc" / "demos.pt"
    keep_failures: bool = False       # also save unsuccessful episodes (0-reward dynamics data)
    # env (mirrors tdmpc_train.Config)
    n_envs: int = 16
    backend: Literal["gpu", "cpu"] = "gpu"
    sim_hz: int = 120
    control_freq: float = 15.0
    max_steps: int = 150
    max_delta_rad: float = 0.10
    binary_gripper: bool = True
    cube_vel_obs: bool = True
    dev_penalty_coef: float = 0.0     # keep demos' recorded rewards purely sparse
    noslip_iterations: int = 10


def collect_round(env, policy, seed: int) -> tuple[list, list[dict]]:
    """One scripted sync round: n_envs episodes; returns (episode tds, per-env stats)."""
    from tdmpc_train import to_td

    B = env.n_envs
    obs = env.reset(seed=seed)
    policy.reset()
    tds = [[to_td(obs[i])] for i in range(B)]
    clipped = np.zeros(B, dtype=int)
    stats: dict[int, dict] = {}
    live = np.ones(B, dtype=bool)
    while live.any():
        a = env.absolute_to_delta(policy.act())
        clipped += live & (np.abs(a[:, :-1]).max(axis=1) >= 1.0)
        actions = torch.from_numpy(a)
        obs, reward, done, info = env.step(actions)
        for i in np.flatnonzero(live):
            tds[i].append(to_td(obs[i], actions[i], reward[i], info["terminated"][i]))
        for i in np.flatnonzero(live & done):
            stats[i] = dict(success=bool(info["success"][i]), steps=len(tds[i]) - 1,
                            max_rise=float(info["max_rise"][i]), reward=float(reward[i]))
        live &= ~done
    for i, s in stats.items():
        s["clipped_ticks"] = int(clipped[i])
    return [torch.cat(t) for t in tds], [stats[i] for i in range(B)]


def main(cfg: Config) -> None:
    # reuse the training script's env builder (same dir -> plain import works)
    from tdmpc_train import Config as TrainConfig, build_env

    from xsim.suite.policies import LiftPolicy

    env = build_env(TrainConfig(
        n_envs=cfg.n_envs, backend=cfg.backend, sim_hz=cfg.sim_hz,
        control_freq=cfg.control_freq, max_steps=cfg.max_steps,
        max_delta_rad=cfg.max_delta_rad, binary_gripper=cfg.binary_gripper,
        cube_vel_obs=cfg.cube_vel_obs, dev_penalty_coef=cfg.dev_penalty_coef,
        noslip_iterations=cfg.noslip_iterations,
    ))
    env.autoreset = False
    policy = LiftPolicy(env.unwrapped, steps_per_segment=cfg.steps_per_segment)

    rounds = max(1, -(-cfg.episodes // cfg.n_envs))
    episodes, stats = [], []
    t0 = time.perf_counter()
    for r in range(rounds):
        tds, round_stats = collect_round(env, policy, cfg.seed + r)
        for i, (td, s) in enumerate(zip(tds, round_stats)):
            stats.append(s)
            if s["success"] or cfg.keep_failures:
                episodes.append(td)
            print(f"[r{r} e{i:02d}] success={s['success']} steps={s['steps']} "
                  f"max_rise={s['max_rise']*100:.1f}cm clipped={s['clipped_ticks']}")

    n_success = sum(s["success"] for s in stats)
    rate = n_success / len(stats)
    dt = time.perf_counter() - t0
    print(f"\n[demos] {n_success}/{len(stats)} successful ({rate:.0%}) in {dt/60:.1f} min; "
          f"saving {len(episodes)} episodes")

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episodes": episodes,
        "stats": stats,
        "meta": dict(seed=cfg.seed, steps_per_segment=cfg.steps_per_segment,
                     env_cfg=dict(n_envs=cfg.n_envs, sim_hz=cfg.sim_hz,
                                  control_freq=cfg.control_freq,
                                  max_steps=cfg.max_steps, max_delta_rad=cfg.max_delta_rad,
                                  noslip_iterations=cfg.noslip_iterations),
                     success_rate=rate),
    }, cfg.out)
    print(f"[demos] wrote {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
