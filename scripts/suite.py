"""Debug runner for the xsim.suite layered envs.

Builds a registered env by name and steps random actions — a quick check that
model composition, controllers, and the episode loop hold together.

    uv run python scripts/suite.py [--env Lift] [--steps 5] [--seed 0]
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

import xsim.suite as suite


@dataclass
class Config:
    env: str = "Lift"
    steps: int = 5
    seed: int = 0
    horizon: int = 50
    show_viewer: bool = False


def main(cfg: Config) -> None:
    env = suite.make(cfg.env, horizon=cfg.horizon, show_viewer=cfg.show_viewer)
    print("action_space:", env.action_space)
    obs, info = env.reset(seed=cfg.seed)
    for name in sorted(obs):
        print(f"  obs[{name}]: shape={obs[name].shape}")
    for i in range(cfg.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step {i}: reward={reward:.3f} terminated={terminated} "
            f"truncated={truncated} info={info} cube={obs['cube_pos'].round(3)}"
        )
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    main(tyro.cli(Config))
