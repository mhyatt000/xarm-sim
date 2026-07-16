"""Sparse-ish reward: task reward minus a penalty for deviating from a reference policy."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np


class DeviationPenaltyWrapper(gym.Wrapper):
    """``r' = r − coef · mean((a − a_ref)²)`` against a scripted reference policy.

    Prior-regularized RL: the optimal policy defaults to the reference behavior
    and deviates only where the deviation buys more task reward than the
    accumulated penalty. Calibrate ``coef`` so a typical successful episode's
    total penalty stays well under the sparse success reward (with per-tick
    deviations of ~0.05 over ~150 ticks, coef=0.002 costs ~0.015/episode; a
    fully random policy pays ~0.2-0.45).

    The wrapper owns its reference policy (``policy_factory(base_env)``, e.g.
    ``lambda e: LiftPolicy(e)``) and replans it from the current sim state on
    EVERY reset — full or partial (``envs_idx``). After any env restarts, all
    envs' references are re-derived from wherever they currently are, making
    the reference a receding-horizon plan rather than a stale time-indexed
    script. Must wrap the env level whose action space the agent uses (e.g.
    ``DeltaActionWrapper``: the wrapped env's ``absolute_to_delta`` converts
    the reference policy's absolute actions into agent-space actions).

    ``info["deviation"]`` carries the per-env mean squared deviation.
    """

    def __init__(self, env: gym.Env, policy_factory: Callable, coef: float = 0.002):
        super().__init__(env)
        assert hasattr(env, "absolute_to_delta"), (
            "DeviationPenaltyWrapper must wrap an action wrapper exposing "
            "absolute_to_delta (e.g. DeltaActionWrapper)"
        )
        self.coef = float(coef)
        self.policy = policy_factory(env.unwrapped)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        out = self.env.reset(seed=seed, options=options)
        self.policy.reset()
        return out

    def step(self, action):
        ref = self.env.absolute_to_delta(self.policy.act())
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(ref.shape), -1.0, 1.0)
        deviation = ((a - ref.astype(np.float64)) ** 2).mean(axis=-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, deviation=deviation)
        reward = np.asarray(reward, dtype=np.float64) - self.coef * deviation
        return obs, reward, terminated, truncated, info
