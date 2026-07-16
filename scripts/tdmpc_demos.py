"""Collect scripted lift demos in the suite Lift action space for TD-MPC2 bootstrapping.

Runs :class:`xsim.suite.policies.LiftPolicy` but executes every control tick through
the training wrapper stack (``DeltaActionWrapper`` -> ``GymWrapper`` -> ``TensorShim``,
built by ``tdmpc_train.build_env``), so the recorded transitions obey the exact
dynamics, observation layout, sparse reward, and termination the RL agent will train
on:

    waypoint command (EE pose, gripper) --IK--> absolute joint target
        --> action = absolute_to_delta(target)   (clip((target - qpos)/max_delta_rad))
        --> env.step(action)

Episodes are stored in tdmpc2's buffer layout (first frame has NaN action/reward,
matching ``OnlineTrainer.to_td``) and saved as a list of TensorDicts:

    uv run python scripts/tdmpc_demos.py --episodes 100 --out outputs/tdmpc/demos.pt
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
from tensordict import TensorDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    episodes: int = 100
    seed: int = 0                     # episode i uses seed + i
    steps_per_segment: int = 20       # scripted policy pacing (40 @ 30 Hz -> 20 @ 15 Hz)
    out: Path = PROJECT_ROOT / "outputs" / "tdmpc" / "demos.pt"
    keep_failures: bool = False       # also save unsuccessful episodes (0-reward dynamics data)
    # env (mirrors tdmpc_train.Config)
    backend: Literal["gpu", "cpu"] = "gpu"
    sim_hz: int = 120
    control_freq: float = 15.0
    max_steps: int = 150
    max_delta_rad: float = 0.10
    noslip_iterations: int = 10


def collect_episode(env, policy, seed: int) -> tuple[TensorDict, dict]:
    """Roll one scripted episode through env.step; returns (episode td, stats)."""
    obs = env.reset(seed=seed)
    policy.reset()

    frames = [_frame(obs)]
    clipped = 0
    done, info, r = False, {}, 0.0
    while not done:
        a = env.absolute_to_delta(policy.act())
        clipped += int(np.abs(a[:-1]).max() >= 1.0)  # saturated joint move this tick
        a = torch.from_numpy(a)
        obs, r, done, info = env.step(a)
        frames.append(_frame(obs, a, r, float(info["terminated"])))

    td = torch.cat(frames)
    stats = dict(success=bool(info.get("success")), steps=len(frames) - 1,
                 clipped_ticks=clipped, max_rise=float(info.get("max_rise", 0.0)),
                 reward=float(r))
    return td, stats


def _frame(obs, action=None, reward=None, terminated=None) -> TensorDict:
    """One buffer frame in tdmpc2's episode layout (cf. OnlineTrainer.to_td)."""
    return TensorDict(
        obs=torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).cpu(),
        action=torch.full((1, 8), float("nan"), device="cpu") if action is None
        else torch.as_tensor(action, dtype=torch.float32).unsqueeze(0).cpu(),
        reward=torch.tensor([float("nan") if reward is None else reward], device="cpu"),
        terminated=torch.tensor(
            [float("nan") if terminated is None else terminated], device="cpu"),
        batch_size=(1,),
    )


def main(cfg: Config) -> None:
    # reuse the training script's env builder (same dir -> plain import works)
    from tdmpc_train import Config as TrainConfig, build_env

    from xsim.suite.policies import LiftPolicy

    env = build_env(TrainConfig(
        backend=cfg.backend, sim_hz=cfg.sim_hz, control_freq=cfg.control_freq,
        max_steps=cfg.max_steps, max_delta_rad=cfg.max_delta_rad,
        noslip_iterations=cfg.noslip_iterations,
    ))
    policy = LiftPolicy(env.unwrapped, steps_per_segment=cfg.steps_per_segment)

    episodes, stats = [], []
    t0 = time.perf_counter()
    for i in range(cfg.episodes):
        td, s = collect_episode(env, policy, cfg.seed + i)
        stats.append(s)
        if s["success"] or cfg.keep_failures:
            episodes.append(td)
        print(f"[ep{i:03d}] success={s['success']} steps={s['steps']} "
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
                     env_cfg=dict(sim_hz=cfg.sim_hz, control_freq=cfg.control_freq,
                                  max_steps=cfg.max_steps, max_delta_rad=cfg.max_delta_rad,
                                  noslip_iterations=cfg.noslip_iterations),
                     success_rate=rate),
    }, cfg.out)
    print(f"[demos] wrote {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
