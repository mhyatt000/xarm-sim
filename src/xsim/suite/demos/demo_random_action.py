"""Run the Lift environment with random actions."""

from __future__ import annotations

import xsim.suite as suite


def main() -> None:
    env = suite.make("Lift", horizon=10)
    obs, info = env.reset(seed=0)
    for key in sorted(obs):
        print(f"{key}: {obs[key].shape}")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={i} reward={reward:.3f} terminated={terminated} "
            f"truncated={truncated} info={info}"
        )


if __name__ == "__main__":
    main()
