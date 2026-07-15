"""Debug runner for the xsim.suite layered envs.

Builds a registered env by name and steps either random actions or the
scripted waypoint lift policy — a quick check that model composition,
controllers, policies, and the episode loop hold together.

    uv run python scripts/suite.py [--env Lift] [--steps 5] [--seed 0]
    uv run python scripts/suite.py --policy waypoint --steps 200 --seed 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tyro

import xsim.suite as suite
from xsim.suite.policies import LiftPolicy


@dataclass
class Config:
    env: str = "Lift"
    steps: int = 5
    seed: int = 0
    horizon: int = 300
    show_viewer: bool = False
    policy: Literal["random", "waypoint"] = "random"
    steps_per_segment: int = 20
    noslip_iterations: int = 10


def main(cfg: Config) -> None:
    env = suite.make(
        cfg.env,
        horizon=cfg.horizon,
        show_viewer=cfg.show_viewer,
        noslip_iterations=cfg.noslip_iterations,
    )
    print("action_space:", env.action_space)
    obs, info = env.reset(seed=cfg.seed)
    for name in sorted(obs):
        print(f"  obs[{name}]: shape={obs[name].shape}")
    policy = None
    if cfg.policy == "waypoint":
        policy = LiftPolicy(env, steps_per_segment=cfg.steps_per_segment)
        policy.reset(obs)
    for i in range(cfg.steps):
        action = policy.act(obs) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step {i}: reward={reward:.3f} terminated={terminated} "
            f"truncated={truncated} info={info} cube={obs['cube_pos'].round(3)}"
        )
        if terminated or truncated:
            print(f"episode end at step {i}: success={info['success']}")
            if policy is not None:
                break
            obs, info = env.reset()


if __name__ == "__main__":
    main(tyro.cli(Config))
